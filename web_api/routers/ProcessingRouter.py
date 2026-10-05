import uuid

from fastapi import APIRouter, status

from web_api.data_models.ProcessingModels import (
    ExtractionDetailResponse,
    ExtractionResponse,
    JobResponse,
    JobSummaryResponse,
    StartProcessingRequest,
)
from web_api.deps.auth import CurrentUser, SessionDep
from web_api.deps.projects import ProjectMemberAccess, ensure_project_access
from web_api.services.ExtractionService import ExtractionService
from web_api.services.FileHandlerService import FileHandlerService
from web_api.services.ProcessingService import ProcessingService

router = APIRouter(tags=["Processing"])


@router.post(
    "/projects/{project_id}/processing-jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED
)
async def start_processing(
    project: ProjectMemberAccess, user: CurrentUser, session: SessionDep, request: StartProcessingRequest | None = None
):
    """Queue documents for processing. No ids = every uploaded or failed document."""
    document_ids = request.document_ids if request else None
    return await ProcessingService(session).start_job(project, document_ids, user)


@router.get("/projects/{project_id}/processing-jobs", response_model=list[JobSummaryResponse])
async def list_jobs(project: ProjectMemberAccess, session: SessionDep):
    return await ProcessingService(session).list_jobs(project)


@router.get("/processing-jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: uuid.UUID, user: CurrentUser, session: SessionDep):
    service = ProcessingService(session)
    await ensure_project_access(session, user, await service.get_job_project_id(job_id))
    return await service.get_job(job_id)


# ---------- extracted content ----------

@router.get("/projects/{project_id}/extractions", response_model=list[ExtractionResponse])
async def list_extractions(project: ProjectMemberAccess, session: SessionDep):
    return await ExtractionService(session).list_for_project(project.id)


@router.get("/documents/{document_id}/extraction", response_model=ExtractionDetailResponse)
async def get_extraction(document_id: uuid.UUID, user: CurrentUser, session: SessionDep):
    """Extracted text (and for academic: enriched markdown + image descriptions)."""
    document = await FileHandlerService(session).get_document(document_id)
    await ensure_project_access(session, user, document.project_id)
    return await ExtractionService(session).get_detail(document_id)
