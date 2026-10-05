import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from web_api.data_models.enums import ModelProvider, ModelStage, RegisteredModelStatus
from web_api.db.base import Base, TimestampMixin, UUIDPkMixin, str_enum


class ProviderCredential(UUIDPkMixin, TimestampMixin, Base):
    """Global, admin-managed provider connection. Projects attach it per stage."""
    __tablename__ = "provider_credentials"

    name: Mapped[str] = mapped_column(String(100), unique=True)   # e.g. "OpenAI - team key"
    provider: Mapped[ModelProvider] = mapped_column(str_enum(ModelProvider, "model_provider"))
    encrypted_fields: Mapped[dict[str, Any]]                      # secrets: {field: fernet-token}
    settings: Mapped[dict[str, Any]] = mapped_column(             # non-secret: base_url, endpoint, api_version
        default=dict, server_default=text("'{}'::jsonb")
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class RegisteredModel(UUIDPkMixin, TimestampMixin, Base):
    """A model an admin registered on a connection whose models we can't look up (Ollama, Azure)."""
    __tablename__ = "credential_models"
    __table_args__ = (
        UniqueConstraint("credential_id", "name"),
        CheckConstraint("embedding_dim IS NULL OR embedding_dim > 0", name="embedding_dim_positive"),
    )

    credential_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("provider_credentials.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(200))                 # Ollama tag / Azure deployment name
    base_model: Mapped[str | None] = mapped_column(String(200))    # Azure: the real model, e.g. gpt-4o
    capabilities: Mapped[list[str]] = mapped_column(ARRAY(String(16)), default=list, server_default="{}")
    embedding_dim: Mapped[int | None] = mapped_column(Integer)
    context_window: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[RegisteredModelStatus] = mapped_column(
        str_enum(RegisteredModelStatus, "registered_model_status"), default=RegisteredModelStatus.UNTESTED
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    last_checked_at: Mapped[datetime | None]


class ProjectStageModel(Base):
    """Which model + credential a project uses for one pipeline stage."""
    __tablename__ = "project_stage_models"
    __table_args__ = (
        CheckConstraint("embedding_dim IS NULL OR embedding_dim > 0", name="embedding_dim_positive"),
    )

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    stage: Mapped[ModelStage] = mapped_column(str_enum(ModelStage, "model_stage"), primary_key=True)
    # RESTRICT: a credential in use can't be deleted
    credential_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("provider_credentials.id", ondelete="RESTRICT"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(200))
    embedding_dim: Mapped[int | None] = mapped_column(Integer)    # embedder stage only, measured by a test call
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
