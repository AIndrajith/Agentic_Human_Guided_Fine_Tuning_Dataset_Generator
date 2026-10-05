"""Worker-only endpoints. Every route requires the X-Internal-Token header."""
import base64
import logging
import uuid

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse

from web_api.data_models.ProcessingModels import (
    ExtractionResponse,
    ImageUploadResponse,
    InternalFileBase64,
    InternalFileMetadata,
    StoreAcademicRequest,
    StoreFictionRequest,
    WorkerProcessingConfig,
)
from web_api.db.models import Document, Project
from web_api.deps.auth import SessionDep
from web_api.deps.internal import InternalOnly
from web_api.errors import BadRequestError, DocumentNotFound
from web_api.services.ExtractionService import ExtractionService
from web_api.services.FileHandlerService import FileHandlerService
from web_api.services.MinioService import minio_service
from web_api.services.ProcessingService import ProcessingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["Internal APIs"], dependencies=[InternalOnly])

FILE_SIZE_THRESHOLD = 5 * 1024 * 1024  # 5MB; must match workers Config.FILE_SIZE_THRESHOLD


async def _metadata(session, document_id: uuid.UUID) -> tuple[Document, InternalFileMetadata]:
    document = await FileHandlerService(session).get_document(document_id)
    project = await session.get(Project, document.project_id)
    try:
        size = (await minio_service.stat(document.storage_key)).size
    except Exception as e:
        raise DocumentNotFound(f"File missing from storage: {e}") from e
    return document, InternalFileMetadata(
        document_id=str(document.id),
        filename=document.original_name,
        stored_filename=document.storage_key.rsplit("/", 1)[-1],
        file_size=size,
        file_type=document.file_type.value,
        data_category=project.data_type.value,
        project_id=str(document.project_id),
        should_stream=size >= FILE_SIZE_THRESHOLD,
    )


@router.get("/files/{document_id}/metadata", response_model=InternalFileMetadata)
async def get_file_metadata(document_id: uuid.UUID, session: SessionDep):
    """First call a worker makes for a document, so it also marks the document as processing."""
    _, metadata = await _metadata(session, document_id)
    await ProcessingService(session).mark_started(document_id)
    return metadata


@router.get("/files/{document_id}/base64", response_model=InternalFileBase64)
async def get_file_base64(document_id: uuid.UUID, session: SessionDep):
    document, metadata = await _metadata(session, document_id)
    if metadata.should_stream:
        raise BadRequestError(f"File too large ({metadata.file_size} bytes). Use the stream endpoint.")
    data = await minio_service.download(document.storage_key)
    return InternalFileBase64(**metadata.model_dump(), file_data=base64.b64encode(data).decode("ascii"))


@router.get("/files/{document_id}/stream")
async def stream_file(document_id: uuid.UUID, session: SessionDep):
    document, metadata = await _metadata(session, document_id)
    return StreamingResponse(
        minio_service.stream(document.storage_key),
        media_type=document.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{metadata.stored_filename}"',
            "X-Document-ID": metadata.document_id,
            "X-File-Size": str(metadata.file_size),
        },
    )


# ---------- processing config ----------

@router.get("/projects/{project_id}/processing-config", response_model=WorkerProcessingConfig)
async def get_processing_config(project_id: uuid.UUID, session: SessionDep):
    """Models, keys, vector size and Qdrant collection for a project. The worker calls this once per job,
    so keys never travel through Redis. 422 if a required model step isn't configured."""
    return await ProcessingService(session).build_worker_config(project_id)


# ---------- extracted content (writes) ----------

@router.post("/extracted/academic/images/{document_id}", response_model=ImageUploadResponse)
async def upload_academic_images(document_id: uuid.UUID, session: SessionDep, images: list[UploadFile] = File(...)):
    saved = await ExtractionService(session).upload_images(document_id, images)
    return ImageUploadResponse(document_id=document_id, images_saved=len(saved), saved_keys=saved)


@router.post("/extracted/fiction", response_model=ExtractionResponse)
async def store_extracted_fiction(request: StoreFictionRequest, session: SessionDep):
    return await ExtractionService(session).store_fiction(
        request.document_id, request.project_id, request.extracted_text, request.extraction_metadata
    )


@router.post("/extracted/academic", response_model=ExtractionResponse)
async def store_extracted_academic(request: StoreAcademicRequest, session: SessionDep):
    return await ExtractionService(session).store_academic(
        request.document_id, request.project_id, request.markdown_text, request.enriched_markdown,
        request.images, request.extraction_metadata,
    )
