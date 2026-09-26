"""
Service for storing extracted content via internal API.
"""

import httpx
import logging
from typing import Dict, Any, List
from pathlib import Path
from workers.config import Config

logger = logging.getLogger(__name__)


class ExtractedContentStorageService:
    """
    Service for storing extracted fiction and academic content
    via web_api internal endpoints.
    """
    
    def __init__(self):
        self.base_url = Config.WEB_API_BASE_URL
        self.timeout = 30.0  # seconds
    
    async def store_fiction_text(
        self,
        document_id: str,
        project_id: str,
        extracted_text: str,
        extraction_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store extracted fiction text (web_api saves it to MinIO + Postgres).
        
        Args:
            document_id: Document ID
            project_id: Project ID
            extracted_text: Full extracted text
            extraction_metadata: Metadata about extraction (page count, method, etc.)
        
        Returns:
            Response from API
        """
        url = f"{self.base_url}/internal/extracted/fiction"
        
        payload = {
            "document_id": document_id,
            "project_id": project_id,
            "extracted_text": extracted_text,
            "extraction_metadata": extraction_metadata
        }
        
        try:
            async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Stored fiction text for document {document_id}")
                return result
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to store fiction text for {document_id}: {str(e)}")
            raise RuntimeError(f"Failed to store fiction text: {str(e)}")
    
    async def upload_academic_images(
        self,
        document_id: str,
        image_files: List[Path]
    ) -> Dict[str, Any]:
        """
        Upload extracted images for an academic document.
        
        Args:
            document_id: Document ID
            image_files: List of image file paths to upload
        
        Returns:
            Response from API with saved paths
        """
        url = f"{self.base_url}/internal/extracted/academic/images/{document_id}"
        
        try:
            files = []
            for img_path in image_files:
                if not img_path.exists():
                    logger.warning(f"Image file not found: {img_path}")
                    continue
                
                files.append(
                    ("images", (img_path.name, open(img_path, "rb"), "image/*"))
                )
            
            if not files:
                logger.warning(f"No valid image files to upload for document {document_id}")
                return {"document_id": document_id, "images_saved": 0, "saved_keys": []}
            
            async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=60.0) as client:  # Longer timeout for file uploads
                response = await client.post(url, files=files)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Uploaded {result['images_saved']} images for document {document_id}")
                
                # Close file handles
                for _, (_, file_obj, _) in files:
                    file_obj.close()
                
                return result
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to upload images for {document_id}: {str(e)}")
            raise RuntimeError(f"Failed to upload images: {str(e)}")
    
    async def store_academic_content(
        self,
        document_id: str,
        project_id: str,
        markdown_text: str,
        enriched_markdown: str,
        images: List[Dict[str, Any]],
        extraction_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store extracted academic content (web_api saves it to MinIO + Postgres).
        
        Args:
            document_id: Document ID
            project_id: Project ID
            markdown_text: Original markdown from Marker
            enriched_markdown: Markdown with images replaced by descriptions
            images: List of image metadata dicts
            extraction_metadata: Metadata about extraction
        
        Returns:
            Response from API
        """
        url = f"{self.base_url}/internal/extracted/academic"
        
        payload = {
            "document_id": document_id,
            "project_id": project_id,
            "markdown_text": markdown_text,
            "enriched_markdown": enriched_markdown,
            "images": images,
            "extraction_metadata": extraction_metadata
        }
        
        try:
            async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Stored academic content for document {document_id}")
                return result
                
        except httpx.HTTPError as e:
            logger.error(f"Failed to store academic content for {document_id}: {str(e)}")
            raise RuntimeError(f"Failed to store academic content: {str(e)}")
