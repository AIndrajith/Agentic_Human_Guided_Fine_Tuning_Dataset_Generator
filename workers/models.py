from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class TaskDocument(BaseModel):
    id: str
    file_size: int


class TaskData(BaseModel):
    """Celery payload. Holds no keys: those come from ProcessingConfig."""
    task_id:     str
    project_id:  str
    documents:   List[TaskDocument]
    data_type:   str


class ModelEndpoint(BaseModel):
    """One LiteLLM model + its connection, from web_api's processing config."""
    model:                str                  # LiteLLM model id, e.g. "anthropic/claude-sonnet-4-5"
    api_key:              Optional[str] = None
    api_base:             Optional[str] = None
    api_version:          Optional[str] = None
    context_window:       Optional[int] = None
    supports_json_schema: bool = False

    def call_kwargs(self) -> Dict[str, str]:
        """Arguments for litellm.acompletion / aembedding (besides model + input)."""
        return {k: v for k, v in {
            "api_key": self.api_key, "api_base": self.api_base, "api_version": self.api_version,
        }.items() if v}


class EmbedderEndpoint(ModelEndpoint):
    dimension: int


class ProcessingConfig(BaseModel):
    """GET /internal/projects/{id}/processing-config. Fetched once per task, kept in memory only."""
    project_id:        str
    data_type:         str
    qdrant_collection: str
    llm:               ModelEndpoint            # context notes
    embedder:          EmbedderEndpoint
    vision:            Optional[ModelEndpoint] = None   # academic image descriptions


class FileMetadata(BaseModel):
    """File metadata from web_api"""
    document_id: str
    filename: str
    stored_filename: str
    file_size: int
    file_type: str
    data_category: str
    project_id: str
    should_stream: bool


class Chunk(BaseModel):
    """Represents a text chunk"""
    index: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Optional[Dict[str, Any]] = None


class Chapter(BaseModel):
    """Represents a book chapter (DEPRECATED - use ContextChunk)"""
    chapter_number: int
    title: Optional[str] = None
    start_char: int
    end_char: int
    text: str
    token_count: int


class ContextChunk(BaseModel):
    """Parent chunk for contextualization (30k tokens)"""
    context_id: int
    text: str
    token_count: int
    start_index: int
    end_index: int
    child_indices: List[int] = Field(default_factory=list)


class ChildChunk(BaseModel):
    """Child chunk for retrieval (800 tokens)"""
    index: int
    parent_context_id: int
    original_text: str
    start_index: int
    end_index: int
    token_count: int


class ContextualizedChildChunk(BaseModel):
    """Child chunk after LLM contextualization"""
    index: int
    parent_context_id: int
    original_text: str
    context_description: str
    combined_text: str  # context_description + "\n\n" + original_text
    start_index: int
    end_index: int
    token_count: int
    metadata: Optional[Dict[str, Any]] = None


class ContextualizedChunk(BaseModel):
    """Chunk with contextualized text (DEPRECATED - use ContextualizedChildChunk)"""
    index: int
    original_text: str
    contextualized_text: str
    metadata: Optional[Dict[str, Any]] = None


class ContextualOutput(BaseModel):
    """LLM output for contextualization"""
    contextual_chunks: List[str] = Field(
        description="Contextualized versions of all given chunks"
    )


class SparseVector(BaseModel):
    """Sparse vector for BM25"""
    indices: List[int]
    values: List[float]


class QdrantPoint(BaseModel):
    """Point to be stored in Qdrant"""
    id: str
    dense_vector: List[float]
    sparse_vector: SparseVector
    payload: Dict[str, Any]


class ProcessingCheckpoint(BaseModel):
    """Checkpoint for resuming failed processing"""
    task_id: str
    document_id: str
    last_completed_stage: str
    last_completed_chapter: int = 0
    last_completed_chunk: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None
