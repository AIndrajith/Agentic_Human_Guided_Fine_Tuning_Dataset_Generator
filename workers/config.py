"""
Worker configuration and hyperparameters.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from workers.enums import ChunkerType

# shared repo-root .env (same file web_api reads); real env vars take precedence
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class Config:
    """Global worker configuration"""

    # ========== API Configuration ==========
    WEB_API_BASE_URL = os.getenv("WEB_API_URL", "http://localhost:8000")
    FILE_SIZE_THRESHOLD = 5 * 1024 * 1024  # 5MB
    # sent as X-Internal-Token on every web_api /internal/* and /webhooks/* call
    INTERNAL_API_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")

    @classmethod
    def internal_headers(cls) -> dict:
        return {"X-Internal-Token": cls.INTERNAL_API_TOKEN}
    
    # ========== Celery Configuration ==========
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    
    # ========== Temp File Configuration ==========
    TEMP_FILE_DIR = os.getenv("TEMP_FILE_DIR", "/tmp/celery_files")
    TEMP_FILE_RETENTION_HOURS = 1  # Keep failed files for 1 hour
    
    # ========== Chunking Configuration ==========
    CHUNKER_TYPE = ChunkerType.LLAMA_INDEX_SENTENCE
    CHUNK_SIZE = 1024  # tokens (deprecated - use hierarchical chunking)
    CHUNK_OVERLAP = 128  # tokens (deprecated - use hierarchical chunking)
    
    # ========== Hierarchical Chunking (Chonkie) ==========
    PARENT_CHUNK_SIZE = 30000  # tokens - context window for LLM
    PARENT_CHUNK_OVERLAP = 5000  # tokens - overlap between parent chunks
    CHILD_CHUNK_SIZE = 800  # tokens - retrieval chunks
    CHILD_CHUNK_OVERLAP = 100  # tokens - overlap between child chunks
    CHONKIE_TOKENIZER = "o200k_harmony"  # tokenizer for Chonkie
    CHONKIE_MIN_SENTENCES = 1  # minimum sentences per chunk
    
    # ========== Contextual Embedding Configuration ==========
    LLM_MODEL = "gpt-4o-mini"
    LLM_TEMPERATURE = 0.0
    LLM_MAX_CONTEXT_TOKENS = 100_000  # GPT-4o-mini context window
    LLM_TARGET_SECTION_TOKENS = 50_000  # Target size for context sections (deprecated)
    LLM_MAX_CHUNKS_PER_BATCH = 30
    LLM_MAX_OUTPUT_TOKENS_PER_CHUNK = 100  # deprecated - use context description limits
    LLM_MAX_CONCURRENT_CALLS = 10  # Semaphore limit
    
    # Context description limits (prepended to original chunk)
    CONTEXT_DESCRIPTION_MIN_TOKENS = 50
    CONTEXT_DESCRIPTION_MAX_TOKENS = 200
    
    # ========== Embedding Configuration ==========
    EMBEDDING_MODEL = "text-embedding-3-large"
    EMBEDDING_DIMENSION = 1536
    EMBEDDING_BATCH_SIZE = 100  # Batch size for embedding API calls
    
    # ========== Qdrant Configuration ==========
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION_FICTION = "fiction_chunks"
    QDRANT_COLLECTION_ACADEMIC = "academic_chunks"
    
    # ========== OpenAI Configuration ==========
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # ========== Marker Configuration (Academic PDFs) ==========
    MARKER_OUTPUT_FORMAT = "markdown"
    MARKER_USE_LLM = True
    MARKER_FORCE_OCR = True
    MARKER_REDO_INLINE_MATH = True
    MARKER_LLM_SERVICE = "marker.services.gemini.GoogleGeminiService"
    MARKER_TIMEOUT = 120  # seconds
    
    # ========== Vision Model Configuration (Image Captioning) ==========
    VISION_MODEL_PROVIDER = os.getenv("VISION_MODEL_PROVIDER", "gemini")  # gemini, openai, ollama
    VISION_MODEL_NAME = "gemini-2.0-flash-exp"
    VISION_CONTEXT_WINDOW = 300  # characters before/after image for context
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    # ========== Academic Chunking Configuration ==========
    ACADEMIC_PARENT_CHUNK_SIZE = 30000  # tokens - same as fiction
    ACADEMIC_PARENT_CHUNK_OVERLAP = 5000  # tokens - MUST have overlap
    ACADEMIC_CHILD_CHUNK_SIZE = 800  # tokens - no overlap (pre-segmentation)
    ACADEMIC_TABLE_CHUNK_MAX_ROWS = 50  # max rows per table chunk
    ACADEMIC_CODE_CHUNK_SIZE = 800  # tokens for code chunks
    ACADEMIC_MATH_KEEP_WHOLE = True  # never split math blocks
    
    # ========== Retry Configuration ==========
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2  # Exponential backoff: 2^attempt seconds
    
    # ========== Chapter Detection Configuration (DEPRECATED) ==========
    # NOTE: Chapter detection is deprecated in favor of hierarchical chunking
    CHAPTER_PATTERNS = [
        r"^Chapter \d+",
        r"^CHAPTER \d+",
        r"^Part \d+",
        r"^PART \d+",
        r"^\d+\.$",  # "1.", "2.", etc.
    ]

    #=========== Acdamic tempory file dir ==========
    TEMP_FILE_DIR = os.getenv("TEMP_FILE_DIR", "/tmp/celery_files/acdemic")
    TEMP_FILE_RETENTION_HOURS = 1 

class FictionConfig:
    """Fiction-specific configuration"""
    chunker_type = ChunkerType.LLAMA_INDEX_SENTENCE
    chunk_size = 1024
    chunk_overlap = 128


class AcademicConfig:
    """Academic-specific configuration (for future)"""
    chunker_type = ChunkerType.CUSTOM_PARAGRAPH
    chunk_size = 2048
    chunk_overlap = 256
