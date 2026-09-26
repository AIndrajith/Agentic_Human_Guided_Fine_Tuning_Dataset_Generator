import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from web_api.data_models.enums import ModelProvider, ModelStage
from web_api.db.base import Base, TimestampMixin, UUIDPkMixin, str_enum


class ProviderCredential(UUIDPkMixin, TimestampMixin, Base):
    """Global, admin-managed API credential. Projects attach it per stage."""
    __tablename__ = "provider_credentials"

    name: Mapped[str] = mapped_column(String(100), unique=True)   # e.g. "OpenAI - team key"
    provider: Mapped[ModelProvider] = mapped_column(str_enum(ModelProvider, "model_provider"))
    encrypted_fields: Mapped[dict[str, Any]]                      # {field: fernet-token}
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


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
    base_url: Mapped[str | None] = mapped_column(String(500))     # Ollama / self-hosted
    embedding_dim: Mapped[int | None] = mapped_column(Integer)    # embedder stage only
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
