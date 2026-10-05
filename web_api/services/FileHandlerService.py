import logging
import uuid
from dataclasses import dataclass

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.core.config import get_settings
from web_api.data_models.enums import FileType
from web_api.db.models import Document, Project, User
from web_api.errors import DocumentNotFound, FileTooLarge, UnsupportedFileType, ValidationError
from web_api.services.MinioService import minio_service
from web_api.services.QdrantService import qdrant_service

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FileKind:
    file_type: FileType
    content_type: str
    magic: tuple[bytes, ...]   # accepted leading bytes


_KINDS = {
    "pdf":  _FileKind(FileType.PDF,    "application/pdf", (b"%PDF-",)),
    "png":  _FileKind(FileType.IMAGES, "image/png",       (b"\x89PNG\r\n\x1a\n",)),
    "jpg":  _FileKind(FileType.IMAGES, "image/jpeg",      (b"\xff\xd8\xff",)),
    "jpeg": _FileKind(FileType.IMAGES, "image/jpeg",      (b"\xff\xd8\xff",)),
    "gif":  _FileKind(FileType.IMAGES, "image/gif",       (b"GIF87a", b"GIF89a")),
    "bmp":  _FileKind(FileType.IMAGES, "image/bmp",       (b"BM",)),
}

_READ_CHUNK = 1024 * 1024


def project_prefix(project_id: uuid.UUID) -> str:
    return f"projects/{project_id}/"


class FileHandlerService:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.max_bytes = get_settings().MAX_UPLOAD_MB * 1024 * 1024

    @staticmethod
    def _kind_for(filename: str) -> tuple[str, _FileKind]:
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        kind = _KINDS.get(extension)
        if not kind:
            raise UnsupportedFileType(f"Unsupported file type: .{extension or '?'}")
        return extension, kind

    async def _read_limited(self, file: UploadFile) -> bytes:
        """Read the upload, stopping as soon as it exceeds the size limit."""
        chunks, total = [], 0
        while chunk := await file.read(_READ_CHUNK):
            total += len(chunk)
            if total > self.max_bytes:
                raise FileTooLarge(f"'{file.filename}' exceeds the {self.max_bytes // (1024 * 1024)} MB limit")
            chunks.append(chunk)
        return b"".join(chunks)

    async def save_files(self, project: Project, files: list[UploadFile], uploaded_by: User) -> list[Document]:
        """Validate + upload all files, then insert all rows in one transaction.
        On any failure nothing is kept: uploaded objects are removed again."""
        files = [f for f in files if f.filename]
        if not files:
            raise ValidationError("No files provided")

        uploaded_keys: list[str] = []
        documents: list[Document] = []
        try:
            for file in files:
                extension, kind = self._kind_for(file.filename)
                data = await self._read_limited(file)
                if not data:
                    raise ValidationError(f"'{file.filename}' is empty")
                if not data.startswith(kind.magic):
                    raise UnsupportedFileType(f"'{file.filename}' content does not match its .{extension} extension")

                document_id = uuid.uuid4()
                key = f"{project_prefix(project.id)}documents/{document_id}.{extension}"
                await minio_service.upload(key, data, kind.content_type)
                uploaded_keys.append(key)

                documents.append(Document(
                    id=document_id,
                    project_id=project.id,
                    original_name=file.filename,
                    storage_key=key,
                    file_type=kind.file_type,
                    content_type=kind.content_type,
                    size_bytes=len(data),
                    uploaded_by=uploaded_by.id,
                ))

            self.session.add_all(documents)
            await self.session.commit()
        except BaseException:
            await self.session.rollback()
            for key in uploaded_keys:
                try:
                    await minio_service.delete(key)
                except Exception:
                    logger.exception("Failed to clean up orphaned upload %s", key)
            raise
        return documents

    async def get_document(self, document_id: uuid.UUID) -> Document:
        document = await self.session.get(Document, document_id)
        if not document:
            raise DocumentNotFound()
        return document

    async def list_project_documents(self, project: Project) -> list[Document]:
        return list(await self.session.scalars(
            select(Document).where(Document.project_id == project.id).order_by(Document.created_at)
        ))

    async def delete_document(self, document: Document) -> None:
        # DB first: a leftover object is harmless, a row pointing at nothing is not
        storage_key, project_id, document_id = document.storage_key, document.project_id, document.id
        await self.session.delete(document)
        await self.session.commit()
        try:
            await minio_service.delete(storage_key)
        except Exception:
            logger.exception("Document row deleted but MinIO object remains: %s", storage_key)
        await qdrant_service.drop_document(project_id, document_id)

    @staticmethod
    async def delete_project_files(project_id: uuid.UUID) -> None:
        """After the project row is deleted: remove its MinIO objects and its Qdrant collection."""
        await qdrant_service.drop_project(project_id)
        try:
            removed = await minio_service.delete_prefix(project_prefix(project_id))
            logger.info("Removed %d MinIO objects for project %s", removed, project_id)
        except Exception:
            logger.exception("Project deleted but MinIO cleanup failed for %s", project_prefix(project_id))
