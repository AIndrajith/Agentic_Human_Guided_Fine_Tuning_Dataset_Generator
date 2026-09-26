import httpx
import base64
from pathlib import Path
from workers.config import Config
from workers.models import FileMetadata
from workers.utils.retry import retry_async
import logging

logger = logging.getLogger(__name__)


class FileFetcherService:
    
    def __init__(self):
        self.api_base_url = Config.WEB_API_BASE_URL
        self.file_size_threshold = Config.FILE_SIZE_THRESHOLD
    
    async def fetch_file(
        self,
        document_id: str,
        save_path: Path
    ) -> FileMetadata:

        metadata = await retry_async(self._get_file_metadata, document_id, what="fetch file metadata")

        if metadata.should_stream:
            logger.info(f"Streaming large file ({metadata.file_size} bytes): {document_id}")
            await retry_async(self._fetch_via_stream, document_id, save_path, what="stream file")
        else:
            logger.info(f"Fetching small file via base64 ({metadata.file_size} bytes): {document_id}")
            await retry_async(self._fetch_via_base64, document_id, save_path, what="fetch file")
        
        return metadata
    
    async def _get_file_metadata(self, document_id: str) -> FileMetadata:
        url = f"{self.api_base_url}/internal/files/{document_id}/metadata"
        
        async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            return FileMetadata(**data)
    
    async def _fetch_via_base64(self, document_id: str, save_path: Path):

        url = f"{self.api_base_url}/internal/files/{document_id}/base64"
        
        async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=60.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

            file_bytes = base64.b64decode(data["file_data"])
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(file_bytes)
            
            logger.info(f"Saved file to {save_path}")
    
    async def _fetch_via_stream(self, document_id: str, save_path: Path):
        url = f"{self.api_base_url}/internal/files/{document_id}/stream"
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=300.0) as client:  # 5 min timeout for large files
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                
                with open(save_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
        
        logger.info(f"Streamed file to {save_path}")
