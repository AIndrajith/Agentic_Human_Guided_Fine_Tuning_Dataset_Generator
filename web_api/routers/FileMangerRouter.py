import uuid

from fastapi import APIRouter, File, UploadFile, status

from web_api.data_models.DataModels import DocumentResponse
from web_api.deps.auth import CurrentUser, SessionDep
from web_api.deps.projects import ProjectMemberAccess, ensure_project_access
from web_api.services.FileHandlerService import FileHandlerService

router = APIRouter(tags=["File Management"])


@router.post(
    "/projects/{project_id}/documents",
    response_model=list[DocumentResponse],
    status_code=status.HTTP_201_CREATED,
)
async def upload_documents(
    project: ProjectMemberAccess,
    user: CurrentUser,
    session: SessionDep,
    files: list[UploadFile] = File(...),
):
    """Upload one or more PDFs/images to a project (all-or-nothing)."""
    return await FileHandlerService(session).save_files(project, files, user)


@router.get("/projects/{project_id}/documents", response_model=list[DocumentResponse])
async def list_project_documents(project: ProjectMemberAccess, session: SessionDep):
    return await FileHandlerService(session).list_project_documents(project)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: uuid.UUID, user: CurrentUser, session: SessionDep):
    service = FileHandlerService(session)
    document = await service.get_document(document_id)
    await ensure_project_access(session, user, document.project_id)
    return document


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, user: CurrentUser, session: SessionDep):
    service = FileHandlerService(session)
    document = await service.get_document(document_id)
    await ensure_project_access(session, user, document.project_id)
    await service.delete_document(document)
