import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from web_api.data_models.enums import ModelCapability, ModelProvider, ModelStage, RegisteredModelStatus
from web_api.services.providers import ModelSource


class CreateCredentialRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    provider: ModelProvider
    secrets: dict[str, str] = Field(default_factory=dict)    # encrypted, never returned
    settings: dict[str, str] = Field(default_factory=dict)   # plain: base_url, endpoint, api_version


class UpdateCredentialRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    secrets: dict[str, str] | None = None    # replaces all secrets when given
    settings: dict[str, str] | None = None   # replaces all settings when given


class CredentialProjectRef(BaseModel):
    project_id: uuid.UUID
    title: str


class CredentialResponse(BaseModel):
    """Never includes secret values."""
    id: uuid.UUID
    name: str
    provider: ModelProvider
    secrets_present: list[str]
    settings: dict[str, str]
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
    """Tells the UI which form to render for a provider."""
    provider:               ModelProvider
    label:                  str
    secret_fields:          list[str]
    optional_secret_fields: list[str]
    settings_fields:        list[str]
    default_settings:       dict[str, str]
    model_source:           ModelSource   # catalog = dropdown from known models; registered = admin registers models
    live_listing:           bool          # the model list is checked against the provider
    config_page:            bool          # show a full configuration page (Ollama, Azure)


class ConnectionTestResponse(BaseModel):
    ok:      bool | None   # None: this provider can't be tested without a model
    message: str


# ---------- model dropdown ----------

class ModelOption(BaseModel):
    name:           str
    capabilities:   list[ModelCapability]
    context_window: int | None = None
    embedding_dim:  int | None = None
    status:         RegisteredModelStatus | None = None   # registered models only


class ModelOptionsResponse(BaseModel):
    credential_id:   uuid.UUID
    provider:        ModelProvider
    stage:           ModelStage | None
    source:          ModelSource
    provider_checked: bool           # the list was confirmed against the provider's live model list
    warning:         str | None = None
    allow_custom:    bool            # the UI may offer "Other..." (typed name, tested on save)
    models:          list[ModelOption]


# ---------- registered models (Ollama / Azure) ----------

class RegisterModelRequest(BaseModel):
    name:         str = Field(min_length=1, max_length=200)        # Ollama tag / Azure deployment name
    base_model:   str | None = Field(default=None, max_length=200)  # Azure: the real model, e.g. gpt-4o
    capabilities: list[ModelCapability] = Field(default_factory=list)  # empty = look up / detect


class RegisteredModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id:              uuid.UUID
    credential_id:   uuid.UUID
    name:            str
    base_model:      str | None
    capabilities:    list[ModelCapability]
    embedding_dim:   int | None
    context_window:  int | None
    status:          RegisteredModelStatus
    last_error:      str | None
    last_checked_at: datetime | None
    created_at:      datetime
