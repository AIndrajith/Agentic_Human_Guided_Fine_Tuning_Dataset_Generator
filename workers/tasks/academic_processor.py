from workers.models import TaskDocument, TaskCredentials
from workers.services.file_fetcher import FileFetcherService
from workers.services.pdf_to_markdown import PDFToMarkdownService
from workers.services.vision_service import VisionService
from workers.services.chunking_service_academic import AcademicChunkingService
from workers.services.contextualizer import Contextualizer
from workers.services.embedding_service import EmbeddingService
from workers.services.bm25_service import BM25Service
from workers.services.storage_service import StorageService
from workers.services.extracted_storage_service import ExtractedContentStorageService
from workers.utils.temp_file_manager import TempFileManager
from workers.utils.webhook_notifier import WebhookNotifier
from workers.enums import ProcessingStage
from workers.config import Config
from typing import Optional
import logging
import re
from pathlib import Path
from chonkie import MarkdownChef

logger = logging.getLogger(__name__)


class AcademicProcessor:

    def __init__(self):
        self.file_fetcher = FileFetcherService()
        self.pdf_to_markdown = PDFToMarkdownService()
        self.chunking_service = AcademicChunkingService()
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
        self.vision_service = VisionService(
            api_key=credentials.vision_api_key if credentials else None,
            model_name=credentials.vision_model if credentials else None,
        )
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
            # ===== 1. Fetch PDF File =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.FETCHING_FILES.value}")
            current_stage = ProcessingStage.FETCHING_FILES
            
            pdf_path = self.temp_file_manager.get_file_path(
                task_id,
                f"{document_id}.pdf"
            )
            
            file_metadata = await self.file_fetcher.fetch_file(
                document_id=document_id,
                save_path=pdf_path
            )
            
            # ===== 2. Convert PDF → Markdown =====
            logger.info(f"[{document_id}] Stage: PDF_TO_MARKDOWN")
            current_stage = "pdf_to_markdown"
            
            marker_output_dir = self.temp_file_manager.get_file_path(
                task_id,
                f"{document_id}_marker_output"
            )
            
            marker_output = self.pdf_to_markdown.convert_pdf(
                pdf_path=pdf_path,
                output_dir=marker_output_dir
            )
            
            logger.info(
                f"[{document_id}] Marker conversion complete: "
                f"{len(marker_output.markdown_text)} chars, "
                f"{len(marker_output.images)} images"
            )
            
            # ===== 3. Caption Images with Context =====
            if marker_output.images:
                logger.info(f"[{document_id}] Stage: IMAGE_CAPTIONING")
                current_stage = "image_captioning"
                
                # Parse markdown to get image references with positions
                chef = MarkdownChef(tokenizer=Config.CHONKIE_TOKENIZER)
                doc_parsed = chef.process(marker_output.markdown_text)
                
                image_descriptions = await self.vision_service.caption_images_with_context(
                    markdown_content=marker_output.markdown_text,
                    images=marker_output.images,
                    image_refs=doc_parsed.images  # MarkdownImage objects with positions
                )
                
                logger.info(f"[{document_id}] Captioned {len(image_descriptions)} images")
            else:
                image_descriptions = {}
                logger.info(f"[{document_id}] No images to caption")
            
            # ===== 4. Replace Images with Descriptions =====
            logger.info(f"[{document_id}] Stage: IMAGE_REPLACEMENT")
            
            enriched_markdown = self._replace_images_with_descriptions(
                marker_output.markdown_text,
                image_descriptions
            )
            
            logger.info(f"[{document_id}] Image replacement complete")
            
            # ===== 4.5. Store Extracted Academic Content =====
            logger.info(f"[{document_id}] Stage: STORING_EXTRACTED_ACADEMIC")
            current_stage = "storing_extracted_academic"
            
            try:
                # Step 1: Upload images to internal API
                if marker_output.images:
                    logger.info(f"[{document_id}] Uploading {len(marker_output.images)} images...")
                    image_paths = [img.path for img in marker_output.images]
                    
                    upload_result = await self.extracted_storage.upload_academic_images(
                        document_id=document_id,
                        image_files=image_paths
                    )
                    logger.info(f"[{document_id}] Uploaded {upload_result['images_saved']} images")
                
                # Step 2: Prepare image metadata
                image_metadata_list = [
                    {
                        "filename": img.filename,
                        "description": image_descriptions.get(img.filename, ""),
                        "position_in_markdown": marker_output.markdown_text.find(img.filename),
                        "alt_text": None
                    }
                    for img in marker_output.images
                ]
                
                # Step 3: Store markdown and metadata in MongoDB
                extraction_metadata = {
                    "marker_config": marker_output.metadata["marker_config"],
                    "character_count": len(enriched_markdown),
                    "image_count": len(marker_output.images),
                    "source_pdf": str(pdf_path)
                }
                
                await self.extracted_storage.store_academic_content(
                    document_id=document_id,
                    project_id=project_id,
                    markdown_text=marker_output.markdown_text,
                    enriched_markdown=enriched_markdown,
                    images=image_metadata_list,
                    extraction_metadata=extraction_metadata
                )
                logger.info(f"[{document_id}] Successfully stored academic content in MongoDB")
                
            except Exception as e:
                logger.error(f"[{document_id}] Failed to store academic content: {str(e)}")
                # Continue processing even if storage fails
            
            # ===== 5. Smart Hierarchical Chunking =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.CHUNKING.value}")
            current_stage = ProcessingStage.CHUNKING
            
            context_chunks, child_chunks = self.chunking_service.create_hierarchical_chunks(
                enriched_markdown
            )
            
            logger.info(
                f"[{document_id}] Created {len(context_chunks)} parent chunks, "
                f"{len(child_chunks)} child chunks"
            )
            
            # ===== 6. Contextualize Chunks =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.CONTEXTUALIZING.value}")
            current_stage = ProcessingStage.CONTEXTUALIZING
            
            paper_metadata = {
                "title": file_metadata.filename,
                "document_id": document_id,
                "project_id": project_id,
                "data_type": "academic"
            }
            
            contextualized_chunks = await self.contextualizer.contextualize_hierarchical(
                context_chunks=context_chunks,
                child_chunks=child_chunks,
                book_metadata=paper_metadata
            )
            
            # ===== 7. Generate Embeddings =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.GENERATING_EMBEDDINGS.value}")
            current_stage = ProcessingStage.GENERATING_EMBEDDINGS
            
            combined_texts = [chunk.combined_text for chunk in contextualized_chunks]
            dense_vectors = await self.embedding_service.generate_embeddings(combined_texts)
            
            # ===== 8. Generate BM25 Sparse Vectors =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.GENERATING_BM25.value}")
            current_stage = ProcessingStage.GENERATING_BM25
            
            sparse_vectors = self.bm25_service.generate_sparse_vectors(combined_texts)
            
            # ===== 9. Store in Qdrant =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.STORING_VECTORS.value}")
            current_stage = ProcessingStage.STORING_VECTORS
            
            await self.storage_service.store_vectors(
                collection_name=Config.QDRANT_COLLECTION_ACADEMIC,
                chunks=contextualized_chunks,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
                document_id=document_id,
                project_id=project_id
            )
            
            # ===== 10. Notify Completion =====
            logger.info(f"[{document_id}] Stage: {ProcessingStage.NOTIFYING_COMPLETION.value}")
            current_stage = ProcessingStage.NOTIFYING_COMPLETION
            
            await self.webhook_notifier.notify_completion(
                task_id=task_id,
                document_id=document_id,
                status="success"
            )
            
            # ===== 11. Cleanup Temp Files =====
            self.temp_file_manager.cleanup_task_files(task_id)
            
            logger.info(f"[{document_id}] Processing complete!")
            
            return {
                "status": "success",
                "document_id": document_id,
                "parent_chunks": len(context_chunks),
                "child_chunks": len(child_chunks),
                "images_captioned": len(image_descriptions),
                "chunker_metadata": self.chunking_service.get_chunker_metadata()
            }
            
        except Exception as e:
            logger.error(
                f"[{document_id}] Failed at stage {current_stage}: {str(e)}",
                exc_info=True
            )

            await self.webhook_notifier.notify_completion(
                task_id=task_id,
                document_id=document_id,
                status="failed",
                error=str(e)
            )
            
            raise
    
    def _replace_images_with_descriptions(
        self,
        markdown: str,
        image_descriptions: dict
    ) -> str:
        
        pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        
        def replace_image(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            img_filename = Path(img_path).name
            
            description = image_descriptions.get(img_filename, f"[Image: {img_filename}]")
            
            replacement = f"\n\n**[IMAGE: {alt_text or img_filename}]**\n{description}\n\n"
            
            return replacement

        enriched = re.sub(pattern, replace_image, markdown)
        
        return enriched
