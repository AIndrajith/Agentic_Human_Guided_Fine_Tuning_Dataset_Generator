from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.enums import ModelProvider, ModelStage
from web_api.data_models.ModelConfigModels import (
    ModelConfigResponse,
    SetModelConfigRequest,
    StageModelConfig,
    StageModelResponse,
    StageValidationResult,
    ValidateModelConfigResponse,
)
from web_api.db.models import Project, ProjectStageModel, ProviderCredential
from web_api.errors import ConflictError, CredentialNotFound, ValidationError
from web_api.services.credential_service import CredentialService
from web_api.services.llm_factory import STAGE_CAPABILITIES, ping_stage


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
                    base_url=stage_model.base_url,
                    embedding_dim=stage_model.embedding_dim,
                    updated_at=stage_model.updated_at,
                )
                for stage_model, credential in rows
            ],
        )

    async def _check_stage(self, stage: ModelStage, config: StageModelConfig) -> ModelProvider:
        """Credential exists and its provider can serve the stage. Returns the provider."""
        credential = await self.session.get(ProviderCredential, config.credential_id)
        if not credential:
            raise CredentialNotFound(f"Credential {config.credential_id} not found (stage '{stage.value}')")
        allowed = STAGE_CAPABILITIES.get(stage, set())
        if credential.provider not in allowed:
            raise ValidationError(
                f"Provider '{credential.provider.value}' cannot serve stage '{stage.value}'. "
                f"Allowed: {sorted(p.value for p in allowed)}"
            )
        if stage == ModelStage.EMBEDDER and not config.embedding_dim:
            raise ValidationError("embedding_dim is required for the embedder stage")
        if stage != ModelStage.EMBEDDER and config.embedding_dim:
            raise ValidationError("embedding_dim is only allowed on the embedder stage")
        return credential.provider

    async def set_config(self, project: Project, request: SetModelConfigRequest) -> ModelConfigResponse:
        """Upsert the given stages; stages not in the request are left unchanged."""
        for stage, config in request.stages.items():
            await self._check_stage(stage, config)

            existing = await self.session.get(ProjectStageModel, (project.id, stage))
            if stage == ModelStage.EMBEDDER and project.embedding_locked and existing and (
                existing.credential_id != config.credential_id
                or existing.model_name != config.model_name
                or existing.embedding_dim != config.embedding_dim
            ):
                raise ConflictError(
                    "Embedder is locked after the first document was processed; "
                    "changing it would make existing vectors incompatible."
                )

            if existing:
                existing.credential_id = config.credential_id
                existing.model_name = config.model_name
                existing.base_url = config.base_url
                existing.embedding_dim = config.embedding_dim
            else:
                self.session.add(ProjectStageModel(project_id=project.id, stage=stage, **config.model_dump()))

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
        credentials = CredentialService(self.session)
        results: list[StageValidationResult] = []

        for stage, config in request.stages.items():
            try:
                await self._check_stage(stage, config)
                provider, creds = await credentials.get_decrypted_fields(config.credential_id)
                await ping_stage(provider, creds, config.model_name, stage, config.base_url)
                results.append(StageValidationResult(stage=stage, ok=True))
            except Exception as e:
                results.append(StageValidationResult(stage=stage, ok=False, error=str(e)))

        return ValidateModelConfigResponse(all_ok=all(r.ok for r in results), results=results)
