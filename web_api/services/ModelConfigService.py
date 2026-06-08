from fastapi import HTTPException
from beanie import PydanticObjectId
from datetime import datetime

from web_api.data_models.ModelConfigModels import (
    ProjectModelConfigModel,
    SetModelConfigRequest,
    StageValidationResult,
    ValidateModelConfigResponse,
)
from web_api.data_models.enums import ModelStage
from web_api.services.llm_factory import STAGE_CAPABILITIES, ping_stage


class ModelConfigService:

    @staticmethod
    async def set_config(project_id: str, request: SetModelConfigRequest) -> ProjectModelConfigModel:
        obj_id = PydanticObjectId(project_id)
        existing = await ProjectModelConfigModel.find_one(
            ProjectModelConfigModel.project_id == obj_id
        )

        if existing and existing.embedding_locked and ModelStage.EMBEDDER in request.stages:
            new_embed = request.stages[ModelStage.EMBEDDER]
            old_embed = existing.stages.get(ModelStage.EMBEDDER.value)
            if old_embed and (
                new_embed.provider != old_embed.provider or
                new_embed.model_name != old_embed.model_name
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Embedder config is locked after first document was processed. "
                        "Changing it would corrupt existing Qdrant vectors."
                    )
                )

        for stage, config in request.stages.items():
            allowed = STAGE_CAPABILITIES.get(stage, set())
            if config.provider not in allowed:
                raise HTTPException(
                    status_code=422,
                    detail=f"Provider '{config.provider}' cannot serve stage '{stage}'. "
                           f"Allowed: {[p.value for p in allowed]}"
                )

        stages_dict = {stage.value: config for stage, config in request.stages.items()}

        if existing:
            existing.stages.update(stages_dict)
            existing.updated_at = datetime.utcnow()
            await existing.save()
            return existing

        config_doc = ProjectModelConfigModel(project_id=obj_id, stages=stages_dict)
        await config_doc.insert()
        return config_doc

    @staticmethod
    async def get_config(project_id: str) -> ProjectModelConfigModel:
        obj_id = PydanticObjectId(project_id)
        config = await ProjectModelConfigModel.find_one(
            ProjectModelConfigModel.project_id == obj_id
        )
        if not config:
            raise HTTPException(status_code=404, detail="No model config found for this project")
        return config

    @staticmethod
    async def validate_config(project_id: str, request: SetModelConfigRequest) -> ValidateModelConfigResponse:
        results: list[StageValidationResult] = []

        for stage, config in request.stages.items():
            allowed = STAGE_CAPABILITIES.get(stage, set())
            if config.provider not in allowed:
                results.append(StageValidationResult(
                    stage=stage,
                    ok=False,
                    error=f"Provider '{config.provider}' cannot serve stage '{stage}'"
                ))
                continue

            try:
                await ping_stage(config.provider, config.model_name, stage, config)
                results.append(StageValidationResult(stage=stage, ok=True))
            except Exception as e:
                results.append(StageValidationResult(stage=stage, ok=False, error=str(e)))

        return ValidateModelConfigResponse(
            all_ok=all(r.ok for r in results),
            results=results,
        )
