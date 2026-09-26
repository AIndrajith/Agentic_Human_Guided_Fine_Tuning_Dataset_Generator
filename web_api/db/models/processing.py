"""Processing pipeline tables. Created now; wired up in the processing phase."""
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from web_api.data_models.enums import DocumentStatus, JobStatus
from web_api.db.base import Base, CreatedAtMixin, UUIDPkMixin, str_enum


class ProcessingJob(UUIDPkMixin, CreatedAtMixin, Base):
    __tablename__ = "processing_jobs"

    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    status: Mapped[JobStatus] = mapped_column(
        str_enum(JobStatus, "job_status"), default=JobStatus.QUEUED, server_default=JobStatus.QUEUED.value
    )
    error: Mapped[str | None] = mapped_column(Text)
    requested_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class JobDocument(Base):
    __tablename__ = "job_documents"

    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    status: Mapped[DocumentStatus] = mapped_column(
        str_enum(DocumentStatus, "job_document_status"),
        default=DocumentStatus.QUEUED,
        server_default=DocumentStatus.QUEUED.value,
    )
    error: Mapped[str | None] = mapped_column(Text)


class Extraction(CreatedAtMixin, Base):
    """Extracted text lives in MinIO; this row holds the keys and stats."""
    __tablename__ = "extractions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    text_key: Mapped[str] = mapped_column(String(1024))
    enriched_key: Mapped[str | None] = mapped_column(String(1024))   # academic: images -> descriptions
    char_count: Mapped[int] = mapped_column(Integer)
    image_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # "metadata" is reserved on declarative classes, so the attribute is `meta`
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", default=dict, server_default=text("'{}'::jsonb")
    )


class ExtractedImage(UUIDPkMixin, Base):
    __tablename__ = "extracted_images"
    __table_args__ = (UniqueConstraint("document_id", "filename"),)

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(512))
    storage_key: Mapped[str] = mapped_column(String(1024))
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    position: Mapped[int | None] = mapped_column(Integer)   # char offset in original markdown


class Chunk(UUIDPkMixin, CreatedAtMixin, Base):
    """Chunk text + vectors live in Qdrant; this row links a chunk to its point."""
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    qdrant_point_id: Mapped[uuid.UUID] = mapped_column(unique=True)
