from fastapi import APIRouter, status

from web_api.data_models.enums import ModelStage
from web_api.data_models.ModelConfigModels import (
    ModelConfigResponse,
    SetModelConfigRequest,
    ValidateModelConfigResponse,
)
from web_api.deps.auth import SessionDep
from web_api.deps.projects import ProjectMemberAccess, ProjectOwnerAccess
from web_api.services.ModelConfigService import ModelConfigService

router = APIRouter(prefix="/projects/{project_id}/model-config", tags=["Model Configuration"])


@router.get("", response_model=ModelConfigResponse)
async def get_model_config(project: ProjectMemberAccess, session: SessionDep):
    return await ModelConfigService(session).get_config(project)


@router.put("", response_model=ModelConfigResponse)
async def set_model_config(request: SetModelConfigRequest, project: ProjectOwnerAccess, session: SessionDep):
    """Attach credentials + models to stages (upsert; other stages untouched).
    Each stage is test-called before saving; the embedder's vector size is measured, not typed."""
    return await ModelConfigService(session).set_config(project, request)


@router.delete("/{stage}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_stage(stage: ModelStage, project: ProjectOwnerAccess, session: SessionDep):
    await ModelConfigService(session).remove_stage(project, stage)


@router.post("/validate", response_model=ValidateModelConfigResponse)
async def validate_model_config(request: SetModelConfigRequest, project: ProjectOwnerAccess, session: SessionDep):
    """Test the given assignments with a tiny live call to each provider. Saves nothing."""
    return await ModelConfigService(session).validate_config(request)
