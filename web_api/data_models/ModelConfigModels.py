from beanie import Document, PydanticObjectId
from pydantic import BaseModel, model_validator
from typing import Optional, Dict
from datetime import datetime
from pydantic import Field
from web_api.data_models.enums import ModelProvider, ModelStage


class StageModelConfig(BaseModel):
    provider:   ModelProvider
    model_name: str
    base_url:   Optional[str] = None   # required for Ollama

    @model_validator(mode="after")
    def require_base_url_for_ollama(self):
        if self.provider == ModelProvider.OLLAMA and not self.base_url:
            raise ValueError("base_url is required when provider is ollama")
        return self


class ProjectModelConfigModel(Document):
    project_id:       PydanticObjectId
    stages:           Dict[str, StageModelConfig]   # key = ModelStage value string
    embedding_locked: bool = False                   # True once first document is embedded
    created_at:       datetime = Field(default_factory=datetime.utcnow)
    updated_at:       datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "project_model_configs"


class SetModelConfigRequest(BaseModel):
    stages: Dict[ModelStage, StageModelConfig]


class StageValidationResult(BaseModel):
    stage: ModelStage
    ok:    bool
    error: Optional[str] = None


class ValidateModelConfigResponse(BaseModel):
    all_ok:  bool
    results: list[StageValidationResult]
