
from workers.models import TaskDocument, TaskCredentials
from workers.services.file_fetcher import FileFetcherService
from workers.services.text_extractor import TextExtractorService
from workers.services.chunking_service_fiction import ChunkingService
from workers.services.contextualizer import Contextualizer
from workers.services.embedding_service import EmbeddingService
from workers.services.bm25_service import BM25Service
from workers.services.storage_service import StorageService
from workers.services.extracted_storage_service import ExtractedContentStorageService
from workers.utils.temp_file_manager import TempFileManager
from workers.utils.webhook_notifier import WebhookNotifier
from workers.enums import ProcessingStage
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FictionProcessor:

    def __init__(self):
        self.file_fetcher = FileFetcherService()
        self.text_extractor = TextExtractorService()
        self.chunking_service = ChunkingService()
        self.bm25_service = BM25Service()
        self.storage_service = StorageService()
        self.extracted_storage = ExtractedContentStorageService()
        self.temp_file_manager = TempFileManager()
        self.webhook_notifier = WebhookNotifier()

    async def process_document(
        self,
        task_id: str,
        document: TaskDocument,
        project_id: str,
        credentials: Optional[TaskCredentials] = None,
    ) -> dict:
        self.contextualizer = Contextualizer(
            api_key=credentials.llm_api_key if credentials else None,
            model=credentials.llm_model if credentials else None,
            base_url=credentials.llm_base_url if credentials else None,
        )
        self.embedding_service = EmbeddingService(
            api_key=credentials.embed_api_key if credentials else None,
            model=credentials.embed_model if credentials else None,
            base_url=credentials.embed_base_url if credentials else None,
        )

        document_id = document.id
        current_stage = ProcessingStage.FETCHING_FILES
        
        try:
            # ===== 1. Fetch File =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.FETCHING_FILES.value}")
            current_stage = ProcessingStage.FETCHING_FILES
            
            file_path = self.temp_file_manager.get_file_path(
                task_id,
                f"{document_id}.pdf"
            )
            
            file_metadata = await self.file_fetcher.fetch_file(
                document_id=document_id,
                save_path=file_path
            )
            
            # ===== 2. Extract Text =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.EXTRACTING_TEXT.value}")
            current_stage = ProcessingStage.EXTRACTING_TEXT
            
            extracted_text = self.text_extractor.extract_text_from_pdf(file_path)
            
            if not extracted_text.strip():
                raise ValueError("Extracted text is empty")
            
            # ===== 2.5. Store Extracted Text =====
            logger.info(f"[{document_id}] Stage: STORING_EXTRACTED_TEXT")
            current_stage = "storing_extracted_text"
            
            extraction_metadata = {
                "page_count": self.text_extractor.get_page_count(file_path),
                "extraction_method": "PyMuPDF",
                "character_count": len(extracted_text),
                "file_size": file_metadata.file_size
            }
            
            try:
                await self.extracted_storage.store_fiction_text(
                    document_id=document_id,
                    project_id=project_id,
                    extracted_text=extracted_text,
                    extraction_metadata=extraction_metadata
                )
                logger.info(f"[{document_id}] Successfully stored extracted text in MongoDB")
            except Exception as e:
                logger.error(f"[{document_id}] Failed to store extracted text: {str(e)}")
                # Continue processing even if storage fails
            
            # ===== 3. Hierarchical Chunking =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.CHUNKING.value}")
            current_stage = ProcessingStage.CHUNKING
            
            context_chunks, child_chunks = self.chunking_service.create_hierarchical_chunks(extracted_text)
            logger.info(f"[{document_id}] Created {len(context_chunks)} parent chunks, {len(child_chunks)} child chunks")
            
            # ===== 4. Contextualize Chunks =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.CONTEXTUALIZING.value}")
            current_stage = ProcessingStage.CONTEXTUALIZING
            
            book_metadata = {
                "title": file_metadata.filename,
                "document_id": document_id,
                "project_id": project_id
            }
            
            contextualized_chunks = await self.contextualizer.contextualize_hierarchical(
                context_chunks=context_chunks,
                child_chunks=child_chunks,
                book_metadata=book_metadata
            )
            
            # ===== 5. Generate Embeddings =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.GENERATING_EMBEDDINGS.value}")
            current_stage = ProcessingStage.GENERATING_EMBEDDINGS
            
            # Embed combined text (context description + original chunk)
            combined_texts = [chunk.combined_text for chunk in contextualized_chunks]
            dense_vectors = await self.embedding_service.generate_embeddings(combined_texts)
            
            # ===== 6. Generate BM25 Sparse Vectors =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.GENERATING_BM25.value}")
            current_stage = ProcessingStage.GENERATING_BM25
            
            # Use combined text for BM25 (includes context + original)
            sparse_vectors = self.bm25_service.generate_sparse_vectors_batch(combined_texts)
            
            # ===== 7. Store in Qdrant =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.STORING_VECTORS.value}")
            current_stage = ProcessingStage.STORING_VECTORS
            
            point_ids = await self.storage_service.store_chunks(
                chunks=contextualized_chunks,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
                document_id=document_id,
                project_id=project_id,
                book_metadata=book_metadata,
                data_category="fiction"
            )
            
            # ===== 9. Notify Web API =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.NOTIFYING_COMPLETION.value}")
            current_stage = ProcessingStage.NOTIFYING_COMPLETION
            
            # Prepare chunks data for webhook
            chunks_data = [
                {
                    "chunk_index": chunk.index,
                    "qdrant_point_id": point_id,
                    "metadata": chunk.metadata
                }
                for chunk, point_id in zip(contextualized_chunks, point_ids)
            ]
            
            await self.webhook_notifier.notify_processing_complete(
                task_id=task_id,
                document_id=document_id,
                project_id=project_id,
                status="completed",
                chunks_processed=len(contextualized_chunks),
                total_chunks=len(contextualized_chunks),
                chunks_data=chunks_data
            )
            
            # ===== 10. Cleanup =====
            logger.info(f"[{document_id}] Cleaning up temp files")
            self.temp_file_manager.cleanup_task_directory(task_id, force=True)
            
            logger.info(f"[{document_id}] Processing completed successfully")
            
            return {
                "status": "completed",
                "document_id": document_id,
                "chunks_processed": len(contextualized_chunks),
                "parent_chunks": len(context_chunks),
                "child_chunks": len(child_chunks)
            }
            
        except Exception as e:
            logger.error(f"[{document_id}] Processing failed at stage {current_stage.value}: {str(e)}")
            
            # Notify web_api of failure
            await self.webhook_notifier.notify_processing_failed(
                task_id=task_id,
                document_id=document_id,
                error_message=str(e),
                stage=current_stage.value
            )
            
            # Keep temp files for debugging
            logger.info(f"[{document_id}] Keeping temp files for debugging")
            
            raise
