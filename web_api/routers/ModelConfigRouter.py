from fastapi import APIRouter
from web_api.data_models.ModelConfigModels import (
    ProjectModelConfigModel,
    SetModelConfigRequest,
    ValidateModelConfigResponse,
)
from web_api.services.ModelConfigService import ModelConfigService

router = APIRouter(prefix="/model-config", tags=["Model Configuration"])
service = ModelConfigService()


@router.post("/{project_id}", response_model=ProjectModelConfigModel)
async def set_model_config(project_id: str, request: SetModelConfigRequest):
    return await service.set_config(project_id, request)


@router.get("/{project_id}", response_model=ProjectModelConfigModel)
async def get_model_config(project_id: str):
    return await service.get_config(project_id)


@router.post("/{project_id}/validate", response_model=ValidateModelConfigResponse)
async def validate_model_config(project_id: str, request: SetModelConfigRequest):
    return await service.validate_config(project_id, request)
