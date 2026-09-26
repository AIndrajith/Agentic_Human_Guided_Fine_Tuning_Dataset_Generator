import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.core.celery_client import PROCESS_DOCUMENTS_TASK, get_celery
from web_api.data_models.enums import DocumentStatus, FileType, JobStatus, ModelStage
from web_api.data_models.ProcessingModels import (
    ChunkCompletionData,
    JobDocumentResponse,
    JobResponse,
)
from web_api.db.models import Chunk, Document, JobDocument, ProcessingJob, Project, ProjectStageModel, User
from web_api.errors import ConflictError, DocumentNotFound, JobNotFound, ValidationError
from web_api.services.credential_service import CredentialService

logger = logging.getLogger(__name__)

_ACTIVE = (DocumentStatus.QUEUED, DocumentStatus.PROCESSING)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProcessingService:
    def __init__(self, session: AsyncSession):
        self.session = session

    # ---------- start ----------

    async def _resolve_credentials(self, project_id: uuid.UUID) -> dict:
        """Map the project's stage config to the worker's TaskCredentials shape.
        Stages that aren't configured are omitted; workers then fall back to their env."""
        rows = {
            sm.stage: sm for sm in await self.session.scalars(
                select(ProjectStageModel).where(ProjectStageModel.project_id == project_id)
            )
        }
        credentials_service = CredentialService(self.session)
        creds: dict[str, str | None] = {}

        if meta := rows.get(ModelStage.META_AGENT):
            provider, fields = await credentials_service.get_decrypted_fields(meta.credential_id)
            creds |= {
                "llm_provider": provider.value, "llm_model": meta.model_name,
                "llm_api_key": fields.get("api_key"), "llm_base_url": meta.base_url or fields.get("base_url"),
            }
        if embed := rows.get(ModelStage.EMBEDDER):
            provider, fields = await credentials_service.get_decrypted_fields(embed.credential_id)
            creds |= {
                "embed_provider": provider.value, "embed_model": embed.model_name,
                "embed_api_key": fields.get("api_key"), "embed_base_url": embed.base_url or fields.get("base_url"),
            }
        if vision := rows.get(ModelStage.VISION):
            _, fields = await credentials_service.get_decrypted_fields(vision.credential_id)
            creds |= {"vision_api_key": fields.get("api_key"), "vision_model": vision.model_name}
        return creds

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

        credentials = await self._resolve_credentials(project.id)

        job = ProcessingJob(id=uuid.uuid4(), project_id=project.id, requested_by=user.id)
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
            "credentials": credentials,
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
