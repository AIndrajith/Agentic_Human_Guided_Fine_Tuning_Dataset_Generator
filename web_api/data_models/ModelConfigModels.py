import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from web_api.data_models.enums import ModelProvider, ModelStage


class StageModelConfig(BaseModel):
    """One stage's assignment: which global credential + which model.
    The embedding size is measured with a test call when saved, never typed."""
    credential_id: uuid.UUID
    model_name:    str = Field(min_length=1, max_length=200)


class SetModelConfigRequest(BaseModel):
    stages: dict[ModelStage, StageModelConfig]


class StageModelResponse(BaseModel):
    stage:           ModelStage
    credential_id:   uuid.UUID
    credential_name: str
    provider:        ModelProvider
    model_name:      str
    embedding_dim:   Optional[int]
    updated_at:      datetime


class ModelConfigResponse(BaseModel):
    project_id:       uuid.UUID
    embedding_locked: bool
    stages:           list[StageModelResponse]


class StageValidationResult(BaseModel):
    stage:         ModelStage
    ok:            bool
    embedding_dim: Optional[int] = None
    error:         Optional[str] = None


class ValidateModelConfigResponse(BaseModel):
    all_ok:  bool
    results: list[StageValidationResult]
