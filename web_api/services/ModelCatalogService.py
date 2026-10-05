import logging
import uuid
from datetime import datetime, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.CredentialModels import (
    ModelOption,
    ModelOptionsResponse,
    RegisterModelRequest,
)
from web_api.data_models.enums import ModelCapability, ModelProvider, ModelStage, RegisteredModelStatus
from web_api.db.models import Project, ProjectStageModel, ProviderCredential, RegisteredModel
from web_api.errors import ConflictError, ModelInUse, ModelTestFailed, RegisteredModelNotFound, ValidationError
from web_api.services import llm_gateway
from web_api.services.credential_service import CredentialService
from web_api.services.llm_gateway import Connection, ModelInfo
from web_api.services.providers import STAGE_CAPABILITY, ModelSource, get_spec

logger = logging.getLogger(__name__)

# what to try when a registered model's capabilities are unknown
_DETECT_ORDER = (ModelCapability.CHAT, ModelCapability.EMBEDDING)


def _fits(capabilities, wanted: ModelCapability | None) -> bool:
    """Unknown capabilities (empty) are offered everywhere; the test call on save decides."""
    return wanted is None or not capabilities or wanted in capabilities


def _sorted_caps(capabilities) -> list[ModelCapability]:
    return sorted({ModelCapability(c) for c in capabilities}, key=lambda c: c.value)


class ModelCatalogService:
    """Model dropdowns for every provider, and registered models for Ollama / Azure connections."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.credentials = CredentialService(session)

    # ---------- dropdown ----------

    async def model_options(self, credential_id: uuid.UUID, stage: ModelStage | None) -> ModelOptionsResponse:
        credential = await self.credentials.get(credential_id)
        spec = get_spec(credential.provider)
        wanted = STAGE_CAPABILITY[stage] if stage else None
        common = dict(credential_id=credential.id, provider=credential.provider, stage=stage, source=spec.model_source)

        if spec.model_source == ModelSource.REGISTERED:
            rows = await self._registered(credential.id)
            options = [
                ModelOption(
                    name=r.name, capabilities=_sorted_caps(r.capabilities), context_window=r.context_window,
                    embedding_dim=r.embedding_dim, status=r.status,
                )
                for r in rows if r.status != RegisteredModelStatus.FAILED and _fits(r.capabilities, wanted)
            ]
            warning = None if rows else "No models registered on this connection yet."
            return ModelOptionsResponse(**common, provider_checked=False, allow_custom=False, warning=warning,
                                        models=options)

        models: list[ModelInfo] = llm_gateway.catalog_models(credential.provider)
        provider_checked, warning = False, None
        if spec.live_listing:
            live = await llm_gateway.live_model_names(self.credentials.connection_for(credential))
            if live is None:
                warning = "Couldn't reach the provider, showing known models."
            else:
                provider_checked = True
                # keep catalog models this key can use + brand-new models the catalog doesn't know yet
                new = sorted(n for n in live if llm_gateway.catalog_lookup(credential.provider, n) is None)
                models = [m for m in models if m.name in live] + [ModelInfo(name=n) for n in new]

        options = [
            ModelOption(name=m.name, capabilities=_sorted_caps(m.capabilities), context_window=m.context_window,
                        embedding_dim=m.embedding_dim)
            for m in models if _fits(m.capabilities, wanted)
        ]
        return ModelOptionsResponse(**common, provider_checked=provider_checked, allow_custom=True, warning=warning,
                                    models=options)

    # ---------- registered models ----------

    async def _registered(self, credential_id: uuid.UUID) -> list[RegisteredModel]:
        return list(await self.session.scalars(
            select(RegisteredModel).where(RegisteredModel.credential_id == credential_id).order_by(RegisteredModel.name)
        ))

    async def _registered_credential(self, credential_id: uuid.UUID) -> ProviderCredential:
        credential = await self.credentials.get(credential_id)
        if get_spec(credential.provider).model_source != ModelSource.REGISTERED:
            raise ValidationError(
                f"{get_spec(credential.provider).label} models come from the model catalog; nothing to register"
            )
        return credential

    async def get_registered(self, credential_id: uuid.UUID, model_id: uuid.UUID) -> RegisteredModel:
        model = await self.session.get(RegisteredModel, model_id)
        if not model or model.credential_id != credential_id:
            raise RegisteredModelNotFound()
        return model

    async def list_registered(self, credential_id: uuid.UUID) -> list[RegisteredModel]:
        await self._registered_credential(credential_id)
        return await self._registered(credential_id)

    async def find_registered(self, credential_id: uuid.UUID, name: str) -> RegisteredModel | None:
        return await self.session.scalar(
            select(RegisteredModel).where(RegisteredModel.credential_id == credential_id, RegisteredModel.name == name)
        )

    async def register(self, credential_id: uuid.UUID, request: RegisterModelRequest) -> RegisteredModel:
        credential = await self._registered_credential(credential_id)
        name = request.name.strip()
        if await self.find_registered(credential.id, name):
            raise ConflictError(f"'{name}' is already registered on this connection")

        connection = self.credentials.connection_for(credential)
        model = RegisteredModel(credential_id=credential.id, name=name, base_model=request.base_model)
        capabilities = set(request.capabilities)

        if credential.provider == ModelProvider.AZURE_OPENAI and request.base_model:
            if info := llm_gateway.catalog_lookup(ModelProvider.AZURE_OPENAI, request.base_model.strip()):
                capabilities = capabilities or info.capabilities
                model.context_window, model.embedding_dim = info.context_window, info.embedding_dim
        elif credential.provider == ModelProvider.OLLAMA:
            info = next((m for m in await self._ollama_models(connection) if m.name == name), None)
            if info is None:
                raise ValidationError(f"'{name}' is not on the Ollama server. Run `ollama pull {name}` there first.")
            capabilities = capabilities or info.capabilities
            model.context_window = info.context_window

        if credential.provider == ModelProvider.AZURE_OPENAI and not capabilities:
            raise ValidationError("Tell us what this deployment does: choose its capabilities or give its base model")

        model.capabilities = [c.value for c in _sorted_caps(capabilities)]
        self.session.add(model)
        await self.session.flush()
        await self._test(model, connection)
        await self.session.commit()
        return model

    async def _ollama_models(self, connection: Connection) -> list[ModelInfo]:
        try:
            return await llm_gateway.ollama_models(connection)
        except httpx.HTTPError as e:
            raise ValidationError(f"Could not reach Ollama at {connection.settings.get('base_url')}: {e}") from e

    async def sync_ollama(self, credential_id: uuid.UUID, test: bool = False) -> list[RegisteredModel]:
        """Read the models downloaded on the Ollama server into this connection."""
        credential = await self._registered_credential(credential_id)
        if credential.provider != ModelProvider.OLLAMA:
            raise ValidationError("Sync is only available for Ollama connections")
        connection = self.credentials.connection_for(credential)
        found = {m.name: m for m in await self._ollama_models(connection)}
        existing = {m.name: m for m in await self._registered(credential.id)}

        for name, info in found.items():
            model = existing.get(name)
            if model is None:
                model = RegisteredModel(credential_id=credential.id, name=name)
                self.session.add(model)
                existing[name] = model
            if info.capabilities:
                model.capabilities = [c.value for c in _sorted_caps(info.capabilities)]
            model.context_window = info.context_window or model.context_window
            if model.status == RegisteredModelStatus.FAILED and model.last_error == "Not found on the Ollama server":
                model.status, model.last_error = RegisteredModelStatus.UNTESTED, None

        for name, model in existing.items():
            if name not in found:   # removed from the server; kept so projects using it show a clear error
                model.status, model.last_error = RegisteredModelStatus.FAILED, "Not found on the Ollama server"

        await self.session.flush()
        if test:
            for name in found:
                await self._test(existing[name], connection)
        await self.session.commit()
        return await self._registered(credential.id)

    async def test_registered(self, credential_id: uuid.UUID, model_id: uuid.UUID) -> RegisteredModel:
        credential = await self._registered_credential(credential_id)
        model = await self.get_registered(credential.id, model_id)
        await self._test(model, self.credentials.connection_for(credential))
        await self.session.commit()
        return model

    async def delete_registered(self, credential_id: uuid.UUID, model_id: uuid.UUID) -> None:
        model = await self.get_registered(credential_id, model_id)
        titles = list(await self.session.scalars(
            select(Project.title)
            .join(ProjectStageModel, ProjectStageModel.project_id == Project.id)
            .where(ProjectStageModel.credential_id == credential_id, ProjectStageModel.model_name == model.name)
            .distinct()
        ))
        if titles:
            raise ModelInUse(f"'{model.name}' is used by project(s): {', '.join(titles)}. Change their model first.")
        await self.session.delete(model)
        await self.session.commit()

    async def _test(self, model: RegisteredModel, connection: Connection) -> None:
        """Probe every capability the model claims (or detect them). Records the result; never raises."""
        capabilities = [ModelCapability(c) for c in model.capabilities]
        errors: list[str] = []
        if capabilities:
            for capability in capabilities:
                try:
                    result = await llm_gateway.probe(connection, model.name, capability)
                    model.embedding_dim = result.embedding_dim or model.embedding_dim
                except ModelTestFailed as e:
                    errors.append(e.message)
        else:
            for capability in _DETECT_ORDER:
                try:
                    result = await llm_gateway.probe(connection, model.name, capability)
                except ModelTestFailed as e:
                    errors.append(e.message)
                    continue
                model.capabilities = [capability.value]
                model.embedding_dim = result.embedding_dim
                errors = []
                break

        model.status = RegisteredModelStatus.FAILED if errors else RegisteredModelStatus.OK
        model.last_error = "; ".join(errors)[:1000] if errors else None
        model.last_checked_at = datetime.now(timezone.utc)
