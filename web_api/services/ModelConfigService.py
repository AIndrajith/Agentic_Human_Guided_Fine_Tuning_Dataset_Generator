from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.enums import ModelStage, RegisteredModelStatus
from web_api.data_models.ModelConfigModels import (
    ModelConfigResponse,
    SetModelConfigRequest,
    StageModelConfig,
    StageModelResponse,
    StageValidationResult,
    ValidateModelConfigResponse,
)
from web_api.db.models import Project, ProjectStageModel, ProviderCredential
from web_api.errors import AppError, ConflictError, CredentialNotFound, ModelTestFailed, ValidationError
from web_api.services import llm_gateway
from web_api.services.credential_service import CredentialService
from web_api.services.llm_gateway import ProbeResult
from web_api.services.ModelCatalogService import ModelCatalogService
from web_api.services.providers import STAGE_CAPABILITY, ModelSource, get_spec


class ModelConfigService:
    """Per-project stage -> (global credential, model) assignments."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_config(self, project: Project) -> ModelConfigResponse:
        rows = await self.session.execute(
            select(ProjectStageModel, ProviderCredential)
            .join(ProviderCredential, ProviderCredential.id == ProjectStageModel.credential_id)
            .where(ProjectStageModel.project_id == project.id)
            .order_by(ProjectStageModel.stage)
        )
        return ModelConfigResponse(
            project_id=project.id,
            embedding_locked=project.embedding_locked,
            stages=[
                StageModelResponse(
                    stage=stage_model.stage,
                    credential_id=credential.id,
                    credential_name=credential.name,
                    provider=credential.provider,
                    model_name=stage_model.model_name,
                    embedding_dim=stage_model.embedding_dim,
                    updated_at=stage_model.updated_at,
                )
                for stage_model, credential in rows
            ],
        )

    async def _test_stage(self, stage: ModelStage, config: StageModelConfig) -> ProbeResult:
        """Check the model fits the stage, then make one tiny real call with it."""
        credential = await self.session.get(ProviderCredential, config.credential_id)
        if not credential:
            raise CredentialNotFound(f"Credential {config.credential_id} not found (stage '{stage.value}')")
        spec = get_spec(credential.provider)
        capability = STAGE_CAPABILITY[stage]
        name = config.model_name.strip()

        registered = None
        if spec.model_source == ModelSource.REGISTERED:
            registered = await ModelCatalogService(self.session).find_registered(credential.id, name)
            if registered is None:
                raise ValidationError(
                    f"'{name}' is not registered on connection '{credential.name}'. Add it on the connection page first."
                )
            known = registered.capabilities
        else:
            info = llm_gateway.catalog_lookup(credential.provider, name)
            known = [c.value for c in info.capabilities] if info else []
            if info and not known:
                raise ValidationError(f"'{name}' is not a chat, embedding or rerank model")
        if known and capability.value not in known:
            raise ValidationError(
                f"'{name}' can't be used for stage '{stage.value}' (needs {capability.value}; it supports {', '.join(known)})"
            )

        connection = CredentialService.connection_for(credential)
        try:
            result = await llm_gateway.probe(connection, name, capability)
        except ModelTestFailed as e:
            raise ModelTestFailed(f"Stage '{stage.value}': {e.message}") from e

        if registered is not None:
            registered.status, registered.last_error = RegisteredModelStatus.OK, None
            registered.last_checked_at = datetime.now(timezone.utc)
            registered.embedding_dim = result.embedding_dim or registered.embedding_dim
        return result

    async def set_config(self, project: Project, request: SetModelConfigRequest) -> ModelConfigResponse:
        """Upsert the given stages (each is test-called first); stages not in the request are left unchanged."""
        for stage, config in request.stages.items():
            existing = await self.session.get(ProjectStageModel, (project.id, stage))
            if stage == ModelStage.EMBEDDER and project.embedding_locked and existing and (
                existing.credential_id != config.credential_id or existing.model_name != config.model_name.strip()
            ):
                raise ConflictError(
                    "Embedder is locked after the first document was processed; "
                    "changing it would make existing vectors incompatible."
                )

            result = await self._test_stage(stage, config)
            embedding_dim = result.embedding_dim if stage == ModelStage.EMBEDDER else None

            if existing:
                existing.credential_id = config.credential_id
                existing.model_name = config.model_name.strip()
                existing.embedding_dim = embedding_dim
            else:
                self.session.add(ProjectStageModel(
                    project_id=project.id, stage=stage, credential_id=config.credential_id,
                    model_name=config.model_name.strip(), embedding_dim=embedding_dim,
                ))

        await self.session.commit()
        return await self.get_config(project)

    async def remove_stage(self, project: Project, stage: ModelStage) -> None:
        if stage == ModelStage.EMBEDDER and project.embedding_locked:
            raise ConflictError("Embedder is locked after the first document was processed")
        existing = await self.session.get(ProjectStageModel, (project.id, stage))
        if existing:
            await self.session.delete(existing)
            await self.session.commit()

    async def validate_config(self, request: SetModelConfigRequest) -> ValidateModelConfigResponse:
        """Dry run: check each stage and make a tiny live call. Saves nothing."""
        results: list[StageValidationResult] = []
        for stage, config in request.stages.items():
            try:
                result = await self._test_stage(stage, config)
                results.append(StageValidationResult(stage=stage, ok=True, embedding_dim=result.embedding_dim))
            except AppError as e:
                results.append(StageValidationResult(stage=stage, ok=False, error=e.message))
        await self.session.rollback()   # _test_stage may have touched registered-model status
        return ValidateModelConfigResponse(all_ok=all(r.ok for r in results), results=results)
