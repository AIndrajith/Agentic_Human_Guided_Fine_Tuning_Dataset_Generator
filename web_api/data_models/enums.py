from enum import Enum


class FileType(Enum):
    PDF    = "pdf"
    IMAGES = "images"


class Datatype(Enum):
    FICTION  = "fiction"
    ACADAMIC = "acdamic"


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
