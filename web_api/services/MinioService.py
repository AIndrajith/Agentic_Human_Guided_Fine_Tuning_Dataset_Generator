import asyncio
import io
import logging

from minio import Minio
from minio.deleteobjects import DeleteObject

from web_api.core.config import get_settings

logger = logging.getLogger(__name__)


class MinioService:
    def __init__(self):
        settings = get_settings()
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY.get_secret_value(),
            secure=settings.MINIO_SECURE,
        )
        self.bucket = settings.MINIO_BUCKET

    async def ensure_bucket(self) -> None:
        """Called once at startup (no network I/O at import time)."""
        try:
            if not await asyncio.to_thread(self.client.bucket_exists, self.bucket):
                await asyncio.to_thread(self.client.make_bucket, self.bucket)
                logger.info("Created MinIO bucket: %s", self.bucket)
        except Exception as e:
            logger.warning("MinIO bucket init failed (uploads will fail until MinIO is reachable): %s", e)

    async def upload(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        await asyncio.to_thread(
            self.client.put_object,
            self.bucket, key,
            io.BytesIO(data), len(data),
            content_type=content_type
        )
        return key

    async def download(self, key: str) -> bytes:
        response = await asyncio.to_thread(self.client.get_object, self.bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def stat(self, key: str):
        return await asyncio.to_thread(self.client.stat_object, self.bucket, key)

    async def delete(self, key: str):
        await asyncio.to_thread(self.client.remove_object, self.bucket, key)

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every object under `prefix`. Returns how many were removed."""
        def _delete() -> int:
            objects = [DeleteObject(o.object_name) for o in
                       self.client.list_objects(self.bucket, prefix=prefix, recursive=True)]
            if not objects:
                return 0
            errors = list(self.client.remove_objects(self.bucket, objects))  # lazy: must be consumed
            for err in errors:
                logger.error("MinIO delete failed for %s: %s", err.name, err.message)
            return len(objects) - len(errors)

        return await asyncio.to_thread(_delete)

    async def stream(self, key: str):
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self.client.get_object, self.bucket, key)
        try:
            while True:
                chunk = await loop.run_in_executor(None, response.read, 8192)
                if not chunk:
                    break
                yield chunk
        finally:
            response.close()
            response.release_conn()


minio_service = MinioService()
