import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from web_api.data_models.enums import DocumentStatus, JobStatus


# ---------- user-facing ----------

class StartProcessingRequest(BaseModel):
    # omit/empty = every document that is uploaded or failed
    document_ids: list[uuid.UUID] | None = None


class JobDocumentResponse(BaseModel):
    document_id:   uuid.UUID
    original_name: str
    status:        DocumentStatus
    error:         str | None


class JobSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:           uuid.UUID
    project_id:   uuid.UUID
    status:       JobStatus
    error:        str | None
    requested_by: uuid.UUID | None
    created_at:   datetime
    started_at:   datetime | None
    finished_at:  datetime | None


class JobResponse(JobSummaryResponse):
    documents: list[JobDocumentResponse]


# ---------- worker webhooks (shape matches workers/utils/webhook_notifier.py) ----------

class ChunkCompletionData(BaseModel):
    chunk_index:     int
    qdrant_point_id: uuid.UUID
    metadata:        Optional[dict] = None   # kept in Qdrant, not stored here


class ProcessingCompletionWebhook(BaseModel):
    task_id:          uuid.UUID   # = processing job id
    document_id:      uuid.UUID
    project_id:       uuid.UUID
    status:           str = Field(..., description="completed, failed, partial")
    chunks_processed: int
    total_chunks:     int
    chunks_data:      list[ChunkCompletionData]
    error_message:    Optional[str] = None
    completed_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProcessingFailedWebhook(BaseModel):
    task_id:       uuid.UUID
    document_id:   uuid.UUID
    error_message: str
    stage:         str


class WebhookAck(BaseModel):
    message:        str
    document_id:    uuid.UUID
    status:         DocumentStatus
    chunks_created: int = 0


# ---------- worker file access (shape matches workers.models.FileMetadata) ----------

class InternalFileMetadata(BaseModel):
    document_id:     str
    filename:        str
    stored_filename: str
    file_size:       int
    file_type:       str
    data_category:   str
    project_id:      str
    should_stream:   bool


class InternalFileBase64(InternalFileMetadata):
    file_data: str


# ---------- extracted content ----------

class ImageMetadataRequest(BaseModel):
    filename:             str
    description:          str
    position_in_markdown: int
    alt_text:             Optional[str] = None   # accepted for compatibility; not stored


class StoreFictionRequest(BaseModel):
    document_id:         uuid.UUID
    project_id:          uuid.UUID
    extracted_text:      str
    extraction_metadata: dict[str, Any]


class StoreAcademicRequest(BaseModel):
    document_id:         uuid.UUID
    project_id:          uuid.UUID
    markdown_text:       str
    enriched_markdown:   str
    images:              list[ImageMetadataRequest]
    extraction_metadata: dict[str, Any]


class ExtractedImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    filename:    str
    storage_key: str
    description: str
    position:    int | None


class ExtractionResponse(BaseModel):
    document_id: uuid.UUID
    char_count:  int
    image_count: int
    metadata:    dict[str, Any]
    images:      list[ExtractedImageResponse]
    created_at:  datetime


class ExtractionDetailResponse(ExtractionResponse):
    text:          str               # fiction: extracted text; academic: original markdown
    enriched_text: str | None = None  # academic: markdown with images replaced by descriptions


class ImageUploadResponse(BaseModel):
    document_id:  uuid.UUID
    images_saved: int
    saved_keys:   list[dict[str, str]]
