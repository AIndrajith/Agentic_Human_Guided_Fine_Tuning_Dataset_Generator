"""
Webhook notification utilities.
"""

import httpx
from typing import List, Optional
from datetime import datetime
from workers.config import Config
import logging

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Service for sending webhook notifications to web_api"""
    
    def __init__(self):
        self.api_base_url = Config.WEB_API_BASE_URL
    
    async def notify_processing_complete(
        self,
        task_id: str,
        document_id: str,
        project_id: str,
        status: str,
        chunks_processed: int,
        total_chunks: int,
        chunks_data: List[dict],
        error_message: Optional[str] = None
    ):
        """
        Notify web_api that document processing is complete.
        
        Args:
            task_id: Celery task ID
            document_id: Document ID
            project_id: Project ID
            status: "completed", "failed", or "partial"
            chunks_processed: Number of chunks successfully processed
            total_chunks: Total number of chunks
            chunks_data: List of chunk data (chunk_index, qdrant_point_id, metadata)
            error_message: Optional error message
        """
        url = f"{self.api_base_url}/webhooks/processing-complete"
        
        payload = {
            "task_id": task_id,
            "document_id": document_id,
            "project_id": project_id,
            "status": status,
            "chunks_processed": chunks_processed,
            "total_chunks": total_chunks,
            "chunks_data": chunks_data,
            "error_message": error_message,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        try:
            async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                logger.info(f"Successfully notified web_api of completion for document {document_id}")
                return response.json()
                
        except Exception as e:
            logger.error(f"Failed to notify web_api: {str(e)}")
            # Don't raise - notification failure shouldn't fail the task
            return None
    
    async def notify_processing_failed(
        self,
        task_id: str,
        document_id: str,
        error_message: str,
        stage: str
    ):
        """
        Notify web_api that document processing failed.
        
        Args:
            task_id: Celery task ID
            document_id: Document ID
            error_message: Error description
            stage: Stage where processing failed
        """
        url = f"{self.api_base_url}/webhooks/processing-failed"
        
        payload = {
            "task_id": task_id,
            "document_id": document_id,
            "error_message": error_message,
            "stage": stage
        }
        
        try:
            async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                logger.info(f"Notified web_api of failure for document {document_id}")
                return response.json()
                
        except Exception as e:
            logger.error(f"Failed to notify web_api of failure: {str(e)}")
            return None
