import logging
import re
import uuid
from pathlib import PurePosixPath
from typing import Any

from fastapi import UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.ProcessingModels import (
    ExtractedImageResponse,
    ExtractionDetailResponse,
    ExtractionResponse,
    ImageMetadataRequest,
)
from web_api.db.models import Document, ExtractedImage, Extraction
from web_api.errors import DocumentNotFound, ExtractionNotFound, UnsupportedFileType, ValidationError
from web_api.services.FileHandlerService import project_prefix
from web_api.services.MinioService import minio_service

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
_IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
                ".svg": "image/svg+xml", ".webp": "image/webp"}


def extraction_prefix(document: Document) -> str:
    return f"{project_prefix(document.project_id)}extracted/{document.id}/"


def safe_image_name(filename: str | None) -> str:
    """Strip any path and reject odd characters, so a filename can't escape the document's prefix."""
    name = PurePosixPath((filename or "").replace("\\", "/")).name
    if not _SAFE_NAME.match(name) or name.startswith("."):
        raise ValidationError(f"Invalid image filename: {filename!r}")
    return name


class ExtractionService:
    """Extracted text/markdown lives in MinIO; `extractions` / `extracted_images` rows hold keys + stats."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def _document(self, document_id: uuid.UUID, project_id: uuid.UUID | None = None) -> Document:
        document = await self.session.get(Document, document_id)
        if not document or (project_id and document.project_id != project_id):
            raise DocumentNotFound()
        return document

    async def _upsert(self, document: Document, **values: Any) -> Extraction:
        extraction = await self.session.get(Extraction, document.id)
        if extraction is None:
            extraction = Extraction(document_id=document.id, **values)
            self.session.add(extraction)
        else:
            for key, value in values.items():
                setattr(extraction, key, value)
        return extraction

    async def store_fiction(
        self, document_id: uuid.UUID, project_id: uuid.UUID, text: str, metadata: dict[str, Any]
    ) -> ExtractionResponse:
        document = await self._document(document_id, project_id)
        text_key = f"{extraction_prefix(document)}text.txt"
        await minio_service.upload(text_key, text.encode("utf-8"), "text/plain; charset=utf-8")
        await self._upsert(
            document, text_key=text_key, enriched_key=None, char_count=len(text), image_count=0, meta=metadata
        )
        await self.session.commit()
        return await self.get_summary(document.id)

    async def upload_images(self, document_id: uuid.UUID, images: list[UploadFile]) -> list[dict[str, str]]:
        document = await self._document(document_id)
        saved = []
        for image in images:
            name = safe_image_name(image.filename)
            content_type = _IMAGE_TYPES.get(PurePosixPath(name).suffix.lower())
            if not content_type:
                raise UnsupportedFileType(f"Unsupported image type: {name}")
            key = f"{extraction_prefix(document)}images/{name}"
            await minio_service.upload(key, await image.read(), content_type)
            saved.append({"filename": name, "minio_key": key})
        return saved

    async def store_academic(
        self, document_id: uuid.UUID, project_id: uuid.UUID, markdown: str, enriched: str,
        images: list[ImageMetadataRequest], metadata: dict[str, Any],
    ) -> ExtractionResponse:
        document = await self._document(document_id, project_id)
        prefix = extraction_prefix(document)
        text_key, enriched_key = f"{prefix}text.md", f"{prefix}enriched.md"
        await minio_service.upload(text_key, markdown.encode("utf-8"), "text/markdown; charset=utf-8")
        await minio_service.upload(enriched_key, enriched.encode("utf-8"), "text/markdown; charset=utf-8")

        await self._upsert(
            document, text_key=text_key, enriched_key=enriched_key,
            char_count=len(enriched), image_count=len(images), meta=metadata,
        )
        # replace image rows wholesale (reprocessing may produce a different set)
        await self.session.execute(delete(ExtractedImage).where(ExtractedImage.document_id == document.id))
        await self.session.flush()
        for img in images:
            name = safe_image_name(img.filename)
            self.session.add(ExtractedImage(
                document_id=document.id,
                filename=name,
                storage_key=f"{prefix}images/{name}",
                description=img.description,
                position=img.position_in_markdown if img.position_in_markdown >= 0 else None,
            ))
        await self.session.commit()
        return await self.get_summary(document.id)

    # ---------- reads ----------

    async def _images(self, document_id: uuid.UUID) -> list[ExtractedImageResponse]:
        rows = await self.session.scalars(
            select(ExtractedImage).where(ExtractedImage.document_id == document_id).order_by(ExtractedImage.position)
        )
        return [ExtractedImageResponse.model_validate(r) for r in rows]

    async def get_summary(self, document_id: uuid.UUID) -> ExtractionResponse:
        extraction = await self.session.get(Extraction, document_id)
        if not extraction:
            raise ExtractionNotFound()
        return ExtractionResponse(
            document_id=extraction.document_id,
            char_count=extraction.char_count,
            image_count=extraction.image_count,
            metadata=extraction.meta,
            images=await self._images(document_id),
            created_at=extraction.created_at,
        )

    async def get_detail(self, document_id: uuid.UUID) -> ExtractionDetailResponse:
        summary = await self.get_summary(document_id)
        extraction = await self.session.get(Extraction, document_id)
        text = (await minio_service.download(extraction.text_key)).decode("utf-8")
        enriched = None
        if extraction.enriched_key:
            enriched = (await minio_service.download(extraction.enriched_key)).decode("utf-8")
        return ExtractionDetailResponse(**summary.model_dump(), text=text, enriched_text=enriched)

    async def list_for_project(self, project_id: uuid.UUID) -> list[ExtractionResponse]:
        document_ids = list(await self.session.scalars(
            select(Extraction.document_id)
            .join(Document, Document.id == Extraction.document_id)
            .where(Document.project_id == project_id)
            .order_by(Document.created_at)
        ))
        return [await self.get_summary(doc_id) for doc_id in document_ids]
