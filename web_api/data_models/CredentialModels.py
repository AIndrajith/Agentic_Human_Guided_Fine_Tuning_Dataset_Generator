import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from web_api.data_models.enums import ModelProvider


PROVIDER_CREDENTIAL_SCHEMA: dict[ModelProvider, list[str]] = {
    ModelProvider.OPENAI:       ["api_key"],
    ModelProvider.ANTHROPIC:    ["api_key"],
    ModelProvider.GOOGLE:       ["api_key"],
    ModelProvider.GROQ:         ["api_key"],
    ModelProvider.MISTRAL:      ["api_key"],
    ModelProvider.COHERE:       ["api_key"],
    ModelProvider.TOGETHER:     ["api_key"],
    ModelProvider.OPENROUTER:   ["api_key"],
    ModelProvider.AZURE_OPENAI: ["api_key", "endpoint", "api_version"],
    ModelProvider.VOYAGEAI:     ["api_key"],
    ModelProvider.JINA:         ["api_key"],
    ModelProvider.OLLAMA:       ["base_url"],
}


class CreateCredentialRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    provider: ModelProvider
    fields: dict[str, str]


class UpdateCredentialRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    fields: dict[str, str] | None = None   # replaces all secret fields when given


class CredentialProjectRef(BaseModel):
    project_id: uuid.UUID
    title: str


class CredentialResponse(BaseModel):
    """Never includes secret values."""
    id: uuid.UUID
    name: str
    provider: ModelProvider
    fields_present: list[str]
    used_by_projects: list[CredentialProjectRef]
    created_by: uuid.UUID | None
    created_at: datetime
    updated_at: datetime


class CredentialOptionResponse(BaseModel):
    """What project owners see when picking a credential for a stage."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    provider: ModelProvider


class ProviderSchemaResponse(BaseModel):
    provider:        ModelProvider
    required_fields: list[str]
