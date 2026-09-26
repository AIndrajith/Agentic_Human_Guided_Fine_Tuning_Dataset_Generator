"""
Worker-specific enums (separate from web_api).
"""

from enum import Enum


class ChunkerType(str, Enum):
    """Chunking strategy types"""
    LLAMA_INDEX_SENTENCE = "llama_index_sentence"
    LANGCHAIN_RECURSIVE = "langchain_recursive"
    CUSTOM_PARAGRAPH = "custom_paragraph"


class ProcessingStage(str, Enum):
    """Processing pipeline stages"""
    FETCHING_FILES = "fetching_files"
    EXTRACTING_TEXT = "extracting_text"
    PDF_TO_MARKDOWN = "pdf_to_markdown"
    IMAGE_CAPTIONING = "image_captioning"
    IMAGE_REPLACEMENT = "image_replacement"
    STORING_EXTRACTED_CONTENT = "storing_extracted_content"
    DETECTING_CHAPTERS = "detecting_chapters"
    CHUNKING = "chunking"
    CONTEXTUALIZING = "contextualizing"
    GENERATING_EMBEDDINGS = "generating_embeddings"
    GENERATING_BM25 = "generating_bm25"
    STORING_VECTORS = "storing_vectors"
    NOTIFYING_COMPLETION = "notifying_completion"
    COMPLETED = "completed"
    FAILED = "failed"


class DataCategory(str, Enum):
    """Data types (worker's own enum, separate from web_api)"""
    FICTION = "fiction"
    ACADEMIC = "academic"
