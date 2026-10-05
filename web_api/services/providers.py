"""Everything provider-specific lives here.

Adding a provider = add a ModelProvider enum value + one ProviderSpec entry below.
New *models* need no code: they come from LiteLLM's catalog, the provider's live model list,
or (Ollama / Azure) the models an admin registers on the connection.
"""
from dataclasses import dataclass, field
from enum import Enum

from web_api.data_models.enums import ModelCapability, ModelProvider, ModelStage


class ModelSource(str, Enum):
    CATALOG    = "catalog"      # dropdown from LiteLLM's catalog (optionally checked against the provider)
    REGISTERED = "registered"   # dropdown from models the admin registered on the connection


class LiveListing(str, Enum):
    """How to ask the provider which models a key can use."""
    LITELLM           = "litellm"             # litellm.get_valid_models (OpenAI, Anthropic, Gemini)
    OPENAI_COMPATIBLE = "openai_compatible"   # GET {api_base}/models with a Bearer key


@dataclass(frozen=True)
class ProviderSpec:
    label: str
    litellm_provider: str                          # model prefix: "<litellm_provider>/<model>"
    secret_fields: tuple[str, ...] = ("api_key",)  # encrypted, never returned
    optional_secret_fields: tuple[str, ...] = ()
    settings_fields: tuple[str, ...] = ()          # plain settings, shown on the config page
    default_settings: dict[str, str] = field(default_factory=dict)
    model_source: ModelSource = ModelSource.CATALOG
    live_listing: LiveListing | None = None
    api_base: str | None = None                    # fixed base URL (OpenAI-compatible listing)
    catalog_providers: tuple[str, ...] = ()        # catalog `litellm_provider` values; default = (litellm_provider,)
    chat_litellm_provider: str | None = None       # different prefix for chat calls (Ollama)
    config_page: bool = False                      # UI shows a full configuration page

    @property
    def catalog_keys(self) -> tuple[str, ...]:
        return self.catalog_providers or (self.litellm_provider,)

    @property
    def all_fields(self) -> tuple[str, ...]:
        return self.secret_fields + self.optional_secret_fields + self.settings_fields


PROVIDERS: dict[ModelProvider, ProviderSpec] = {
    ModelProvider.OPENAI: ProviderSpec(
        label="OpenAI", litellm_provider="openai", live_listing=LiveListing.LITELLM,
    ),
    ModelProvider.ANTHROPIC: ProviderSpec(
        label="Anthropic", litellm_provider="anthropic", live_listing=LiveListing.LITELLM,
    ),
    ModelProvider.GOOGLE: ProviderSpec(
        label="Google Gemini", litellm_provider="gemini", live_listing=LiveListing.LITELLM,
    ),
    ModelProvider.GROQ: ProviderSpec(
        label="Groq", litellm_provider="groq",
        live_listing=LiveListing.OPENAI_COMPATIBLE, api_base="https://api.groq.com/openai/v1",
    ),
    ModelProvider.MISTRAL: ProviderSpec(
        label="Mistral", litellm_provider="mistral",
        live_listing=LiveListing.OPENAI_COMPATIBLE, api_base="https://api.mistral.ai/v1",
    ),
    ModelProvider.TOGETHER: ProviderSpec(
        label="Together AI", litellm_provider="together_ai",
        live_listing=LiveListing.OPENAI_COMPATIBLE, api_base="https://api.together.xyz/v1",
    ),
    ModelProvider.OPENROUTER: ProviderSpec(
        label="OpenRouter", litellm_provider="openrouter",
        live_listing=LiveListing.OPENAI_COMPATIBLE, api_base="https://openrouter.ai/api/v1",
    ),
    ModelProvider.COHERE: ProviderSpec(
        label="Cohere", litellm_provider="cohere", catalog_providers=("cohere", "cohere_chat"),
    ),
    ModelProvider.VOYAGEAI: ProviderSpec(label="Voyage AI", litellm_provider="voyage"),
    ModelProvider.JINA: ProviderSpec(label="Jina AI", litellm_provider="jina_ai"),
    ModelProvider.OLLAMA: ProviderSpec(
        label="Ollama", litellm_provider="ollama", chat_litellm_provider="ollama_chat",
        secret_fields=(), optional_secret_fields=("api_key",),   # key only if behind an auth proxy
        settings_fields=("base_url",), default_settings={"base_url": "http://127.0.0.1:11434"},
        model_source=ModelSource.REGISTERED, config_page=True,
    ),
    ModelProvider.AZURE_OPENAI: ProviderSpec(
        label="Azure OpenAI", litellm_provider="azure",
        settings_fields=("endpoint", "api_version"), default_settings={"api_version": "2024-10-21"},
        model_source=ModelSource.REGISTERED, config_page=True,
    ),
}

# connection field -> LiteLLM call argument
FIELD_TO_LITELLM_ARG = {
    "api_key": "api_key",
    "base_url": "api_base",
    "endpoint": "api_base",
    "api_version": "api_version",
}

STAGE_CAPABILITY: dict[ModelStage, ModelCapability] = {
    ModelStage.QUESTION_GENERATOR: ModelCapability.CHAT,
    ModelStage.ANSWER_GENERATOR:   ModelCapability.CHAT,
    ModelStage.VALIDATOR:          ModelCapability.CHAT,
    ModelStage.META_AGENT:         ModelCapability.CHAT,
    ModelStage.EMBEDDER:           ModelCapability.EMBEDDING,
    ModelStage.RERANKER:           ModelCapability.RERANK,
    ModelStage.VISION:             ModelCapability.VISION,
}


def get_spec(provider: ModelProvider) -> ProviderSpec:
    return PROVIDERS[provider]
