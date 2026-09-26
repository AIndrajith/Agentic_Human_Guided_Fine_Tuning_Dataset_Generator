import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.core.celery_client import PROCESS_DOCUMENTS_TASK, get_celery
from web_api.data_models.enums import Datatype, DocumentStatus, FileType, JobStatus, ModelStage
from web_api.data_models.ProcessingModels import (
    ChunkCompletionData,
    JobDocumentResponse,
    JobResponse,
    WorkerEmbedderEndpoint,
    WorkerModelEndpoint,
    WorkerProcessingConfig,
)
from web_api.db.models import Chunk, Document, JobDocument, ProcessingJob, Project, ProjectStageModel, User
from web_api.errors import ConflictError, DocumentNotFound, JobNotFound, ProjectNotFound, ValidationError
from web_api.services import llm_gateway
from web_api.services.credential_service import CredentialService
from web_api.services.ModelCatalogService import ModelCatalogService
from web_api.services.providers import STAGE_CAPABILITY, ModelSource
from web_api.services.QdrantService import collection_name

logger = logging.getLogger(__name__)

_ACTIVE = (DocumentStatus.QUEUED, DocumentStatus.PROCESSING)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProcessingService:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ---------- worker config ----------

    @staticmethod
    def _required_stages(project: Project) -> list[ModelStage]:
        stages = [ModelStage.META_AGENT, ModelStage.EMBEDDER]
        if project.data_type == Datatype.ACADEMIC:
            stages.append(ModelStage.VISION)
        return stages

    async def _stage_rows(self, project_id: uuid.UUID) -> dict[ModelStage, ProjectStageModel]:
        return {
            sm.stage: sm for sm in await self.session.scalars(
                select(ProjectStageModel).where(ProjectStageModel.project_id == project_id)
            )
        }

    async def _ensure_configured(self, project: Project) -> dict[ModelStage, ProjectStageModel]:
        rows = await self._stage_rows(project.id)
        if missing := [s.value for s in self._required_stages(project) if s not in rows]:
            raise ValidationError(f"Configure these model steps first: {', '.join(missing)}")
        return rows

    async def _endpoint(self, stage_model: ProjectStageModel) -> dict:
        credential = await CredentialService(self.session).get(stage_model.credential_id)
        connection = CredentialService.connection_for(credential)
        capability = STAGE_CAPABILITY[stage_model.stage]

        info = llm_gateway.catalog_lookup(credential.provider, stage_model.model_name)
        context_window = info.context_window if info else None
        if connection.spec.model_source == ModelSource.REGISTERED:
            registered = await ModelCatalogService(self.session).find_registered(credential.id, stage_model.model_name)
            if registered:
                if registered.base_model:   # Azure deployment -> details of its real model
                    info = llm_gateway.catalog_lookup(credential.provider, registered.base_model) or info
                # Ollama: read from the server at sync; Azure: filled from the base model
                context_window = registered.context_window or (info.context_window if info else None)
        return {
            "model": connection.model_id(stage_model.model_name, capability),
            **connection.litellm_kwargs(),
            "context_window": context_window,
            "supports_json_schema": info.supports_json_schema if info else False,
        }

    async def build_worker_config(self, project_id: uuid.UUID) -> WorkerProcessingConfig:
        """Keys + models the worker needs, fetched just in time (never sent through Redis)."""
        project = await self.session.get(Project, project_id)
        if not project:
            raise ProjectNotFound()
        rows = await self._ensure_configured(project)
        embedder = rows[ModelStage.EMBEDDER]
        if not embedder.embedding_dim:
            raise ValidationError("The embedder has no measured vector size. Save the embedder step again.")
        vision = rows.get(ModelStage.VISION)
        return WorkerProcessingConfig(
            project_id=project.id,
            data_type=project.data_type,
            qdrant_collection=collection_name(project.id),
            llm=WorkerModelEndpoint(**await self._endpoint(rows[ModelStage.META_AGENT])),
            embedder=WorkerEmbedderEndpoint(**await self._endpoint(embedder), dimension=embedder.embedding_dim),
            vision=WorkerModelEndpoint(**await self._endpoint(vision)) if vision else None,
        )

    # ---------- start ----------

    async def start_job(self, project: Project, document_ids: list[uuid.UUID] | None, user: User) -> JobResponse:
        query = select(Document).where(Document.project_id == project.id)
        if document_ids:
            query = query.where(Document.id.in_(document_ids))
        documents = list(await self.session.scalars(query.order_by(Document.created_at)))

        if document_ids:
            missing = set(document_ids) - {d.id for d in documents}
            if missing:
                raise DocumentNotFound(f"Not in this project: {', '.join(sorted(map(str, missing)))}")
        else:
            # "process everything" = everything not yet processed or failed
            documents = [d for d in documents if d.status in (DocumentStatus.UPLOADED, DocumentStatus.FAILED)]
        if not documents:
            raise ValidationError("No documents to process")

        if busy := [d.original_name for d in documents if d.status in _ACTIVE]:
            raise ConflictError(f"Already queued or processing: {', '.join(busy)}")
        if images := [d.original_name for d in documents if d.file_type != FileType.PDF]:
            raise ValidationError(f"Only PDFs can be processed for now: {', '.join(images)}")

        await self._ensure_configured(project)

        job =ProcessingJob(id=uuid.uuid4(), project_id=project.id, requested_by=user.id)
        job.celery_task_id = str(job.id)
        self.session.add(job)
        await self.session.flush()
        for document in documents:
            document.status = DocumentStatus.QUEUED
            document.error = None
            self.session.add(JobDocument(job_id=job.id, document_id=document.id))
        await self.session.commit()   # rows exist before any worker webhook can arrive

        payload = {
            "task_id": str(job.id),
            "project_id": str(project.id),
            "documents": [{"id": str(d.id), "file_size": d.size_bytes} for d in documents],
            "data_type": project.data_type.value,
            # no keys here: the worker fetches them from /internal/projects/{id}/processing-config
        }
        try:
            # kombu publishing is blocking I/O
            await asyncio.to_thread(
                get_celery().send_task, PROCESS_DOCUMENTS_TASK, args=[payload], task_id=job.celery_task_id
            )
        except Exception as e:
            logger.exception("Failed to enqueue job %s", job.id)
            await self._fail_job(job, documents, f"Could not queue the job: {e}")
            raise ConflictError("Could not queue the job (is Redis running?)") from e

        return await self.get_job(job.id)

    async def _fail_job(self, job: ProcessingJob, documents: list[Document], error: str) -> None:
        job.status, job.error, job.finished_at = JobStatus.FAILED, error, _now()
        for document in documents:
            document.status, document.error = DocumentStatus.FAILED, error
        await self.session.execute(
            update(JobDocument).where(JobDocument.job_id == job.id).values(status=DocumentStatus.FAILED, error=error)
        )
        await self.session.commit()

    # ---------- read ----------

    async def get_job(self, job_id: uuid.UUID) -> JobResponse:
        job = await self.session.get(ProcessingJob, job_id)
        if not job:
            raise JobNotFound()
        rows = await self.session.execute(
            select(JobDocument, Document.original_name)
            .join(Document, Document.id == JobDocument.document_id)
            .where(JobDocument.job_id == job.id)
            .order_by(Document.created_at)
        )
        return JobResponse(
            id=job.id,
            project_id=job.project_id,
            status=job.status,
            error=job.error,
            requested_by=job.requested_by,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            documents=[
                JobDocumentResponse(document_id=jd.document_id, original_name=name, status=jd.status, error=jd.error)
                for jd, name in rows
            ],
        )

    async def get_job_project_id(self, job_id: uuid.UUID) -> uuid.UUID:
        project_id = await self.session.scalar(select(ProcessingJob.project_id).where(ProcessingJob.id == job_id))
        if not project_id:
            raise JobNotFound()
        return project_id

    async def list_jobs(self, project: Project) -> list[ProcessingJob]:
        return list(await self.session.scalars(
            select(ProcessingJob).where(ProcessingJob.project_id == project.id).order_by(ProcessingJob.created_at.desc())
        ))

    # ---------- worker callbacks ----------

    async def _job_document(self, job_id: uuid.UUID, document_id: uuid.UUID) -> tuple[ProcessingJob, JobDocument, Document]:
        job = await self.session.get(ProcessingJob, job_id)
        job_document = await self.session.get(JobDocument, (job_id, document_id))
        document = await self.session.get(Document, document_id)
        if not job or not job_document or not document:
            raise JobNotFound(f"Document {document_id} is not part of job {job_id}")
        return job, job_document, document

    async def mark_started(self, document_id: uuid.UUID) -> None:
        """Worker began fetching this document: queued -> processing (all its queued job rows)."""
        document = await self.session.get(Document, document_id)
        if not document or document.status != DocumentStatus.QUEUED:
            return
        document.status = DocumentStatus.PROCESSING
        job_documents = list(await self.session.scalars(
            select(JobDocument).where(JobDocument.document_id == document_id, JobDocument.status == DocumentStatus.QUEUED)
        ))
        for job_document in job_documents:
            job_document.status = DocumentStatus.PROCESSING
            job = await self.session.get(ProcessingJob, job_document.job_id)
            if job.status == JobStatus.QUEUED:
                job.status, job.started_at = JobStatus.RUNNING, _now()
        await self.session.commit()

    async def record_completion(
        self, job_id: uuid.UUID, document_id: uuid.UUID, succeeded: bool,
        chunks: list[ChunkCompletionData], error: str | None,
    ) -> int:
        job, job_document, document = await self._job_document(job_id, document_id)

        # idempotent: a retried document replaces its previous chunk rows
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        if succeeded:
            self.session.add_all(
                Chunk(document_id=document_id, chunk_index=c.chunk_index, qdrant_point_id=c.qdrant_point_id)
                for c in chunks
            )
            project = await self.session.get(Project, document.project_id)
            project.embedding_locked = True   # vectors now exist for this embedder

        status = DocumentStatus.COMPLETED if succeeded else DocumentStatus.FAILED
        job_document.status, job_document.error = status, error
        document.status, document.error, document.processed_at = status, error, _now()
        await self.session.flush()
        await self._refresh_job_status(job)
        await self.session.commit()
        return len(chunks) if succeeded else 0

    async def record_failure(self, job_id: uuid.UUID, document_id: uuid.UUID, error: str, stage: str) -> None:
        await self.record_completion(job_id, document_id, False, [], f"[{stage}] {error}")

    async def _refresh_job_status(self, job: ProcessingJob) -> None:
        statuses = list(await self.session.scalars(select(JobDocument.status).where(JobDocument.job_id == job.id)))
        if job.started_at is None:
            job.started_at = _now()
        if any(s in _ACTIVE for s in statuses):
            job.status = JobStatus.RUNNING
            return
        if all(s == DocumentStatus.COMPLETED for s in statuses):
            job.status = JobStatus.COMPLETED
        elif all(s == DocumentStatus.FAILED for s in statuses):
            job.status = JobStatus.FAILED
        else:
            job.status = JobStatus.PARTIAL
        job.finished_at = _now()
