from enum import Enum


class FileType(str, Enum):
    PDF    = "pdf"
    IMAGES = "images"


class Datatype(str, Enum):
    FICTION  = "fiction"
    ACADEMIC = "academic"


class AppRole(str, Enum):
    ADMIN = "admin"
    USER  = "user"


class ProjectRole(str, Enum):
    OWNER  = "owner"
    WORKER = "worker"


class DocumentStatus(str, Enum):
    UPLOADED   = "uploaded"
    QUEUED     = "queued"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class JobStatus(str, Enum):
    QUEUED    = "queued"
    RUNNING   = "running"
    COMPLETED = "completed"
    PARTIAL   = "partial"
    FAILED    = "failed"


class EmailKind(str, Enum):
    INVITE = "invite"


class ModelProvider(str, Enum):
    # ── Chat / LLM ──────────────────────────────────
    OPENAI       = "openai"
    ANTHROPIC    = "anthropic"
    GOOGLE       = "google"
    GROQ         = "groq"
    MISTRAL      = "mistral"
    COHERE       = "cohere"
    TOGETHER     = "together"
    OPENROUTER   = "openrouter"
    AZURE_OPENAI = "azure_openai"
    OLLAMA       = "ollama"
    # ── Embedding / Reranking only ───────────────────
    VOYAGEAI     = "voyageai"
    JINA         = "jina"


class ModelStage(str, Enum):
    QUESTION_GENERATOR = "question_generator"
    ANSWER_GENERATOR   = "answer_generator"
    VALIDATOR          = "validator"
    META_AGENT         = "meta_agent"
    EMBEDDER           = "embedder"
    RERANKER           = "reranker"
    VISION             = "vision"      # image captioning + Marker LLM
