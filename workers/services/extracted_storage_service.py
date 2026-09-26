"""
Service for storing extracted content via internal API.
"""

import httpx
import logging
from typing import Dict, Any, List
from pathlib import Path
from workers.config import Config
from workers.utils.retry import retry_async

logger = logging.getLogger(__name__)


class ExtractedContentStorageService:
    """
    Service for storing extracted fiction and academic content
    via web_api internal endpoints. Temporary errors are retried; the last error is raised as-is.
    """

    def __init__(self):
        self.base_url = Config.WEB_API_BASE_URL
        self.timeout = 30.0  # seconds

    async def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()

    async def store_fiction_text(
        self,
        document_id: str,
        project_id: str,
        extracted_text: str,
        extraction_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store extracted fiction text (web_api saves it to MinIO + Postgres)."""
        payload = {
            "document_id": document_id,
            "project_id": project_id,
            "extracted_text": extracted_text,
            "extraction_metadata": extraction_metadata
        }
        result = await retry_async(
            self._post_json, f"{self.base_url}/internal/extracted/fiction", payload, what="store fiction text"
        )
        logger.info(f"Stored fiction text for document {document_id}")
        return result

    async def _post_images(self, url: str, image_files: List[Path]) -> Dict[str, Any]:
        handles = [open(p, "rb") for p in image_files]
        try:
            files = [("images", (p.name, h, "image/*")) for p, h in zip(image_files, handles)]
            async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=60.0) as client:  # uploads take longer
                response = await client.post(url, files=files)
                response.raise_for_status()
                return response.json()
        finally:
            for h in handles:
                h.close()

    async def upload_academic_images(
        self,
        document_id: str,
        image_files: List[Path]
    ) -> Dict[str, Any]:
        """Upload extracted images for an academic document."""
        existing = [p for p in image_files if p.exists()]
        for missing in set(image_files) - set(existing):
            logger.warning(f"Image file not found: {missing}")
        if not existing:
            logger.warning(f"No valid image files to upload for document {document_id}")
            return {"document_id": document_id, "images_saved": 0, "saved_keys": []}

        result = await retry_async(
            self._post_images, f"{self.base_url}/internal/extracted/academic/images/{document_id}", existing,
            what="upload academic images",
        )
        logger.info(f"Uploaded {result['images_saved']} images for document {document_id}")
        return result

    async def store_academic_content(
        self,
        document_id: str,
        project_id: str,
        markdown_text: str,
        enriched_markdown: str,
        images: List[Dict[str, Any]],
        extraction_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store extracted academic content (web_api saves it to MinIO + Postgres)."""
        payload = {
            "document_id": document_id,
            "project_id": project_id,
            "markdown_text": markdown_text,
            "enriched_markdown": enriched_markdown,
            "images": images,
            "extraction_metadata": extraction_metadata
        }
        result = await retry_async(
            self._post_json, f"{self.base_url}/internal/extracted/academic", payload, what="store academic content"
        )
        logger.info(f"Stored academic content for document {document_id}")
        return result
