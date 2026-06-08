from beanie import Document
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pydantic import Field
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


class ProviderCredentialModel(Document):
    provider:         ModelProvider
    encrypted_fields: dict[str, str]
    updated_at:       datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "provider_credentials"


class SetCredentialRequest(BaseModel):
    fields: dict[str, str]


class CredentialStatusResponse(BaseModel):
    provider:       ModelProvider
    configured:     bool
    fields_present: list[str]
    updated_at:     Optional[datetime]


class ProviderSchemaResponse(BaseModel):
    provider:        ModelProvider
    required_fields: list[str]
