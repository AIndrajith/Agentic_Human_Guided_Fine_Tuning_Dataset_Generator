import uuid

from fastapi import APIRouter, status

from web_api.data_models.CredentialModels import (
    ConnectionTestResponse,
    CreateCredentialRequest,
    CredentialOptionResponse,
    CredentialResponse,
    ModelOptionsResponse,
    ProviderSchemaResponse,
    RegisteredModelResponse,
    RegisterModelRequest,
    UpdateCredentialRequest,
)
from web_api.data_models.enums import ModelStage
from web_api.deps.auth import AdminUser, CurrentUser, SessionDep
from web_api.services import llm_gateway
from web_api.services.credential_service import CredentialService
from web_api.services.ModelCatalogService import ModelCatalogService
from web_api.services.providers import PROVIDERS

router = APIRouter(prefix="/credentials", tags=["Credentials"])


@router.get("/schema", response_model=list[ProviderSchemaResponse])
async def get_all_schemas(user: CurrentUser):
    """Which form the UI should render per provider (simple form or configuration page)."""
    return [
        ProviderSchemaResponse(
            provider=provider,
            label=spec.label,
            secret_fields=list(spec.secret_fields),
            optional_secret_fields=list(spec.optional_secret_fields),
            settings_fields=list(spec.settings_fields),
            default_settings=spec.default_settings,
            model_source=spec.model_source,
            live_listing=spec.live_listing is not None,
            config_page=spec.config_page,
        )
        for provider, spec in PROVIDERS.items()
    ]


@router.get("/options", response_model=list[CredentialOptionResponse])
async def list_credential_options(user: CurrentUser, session: SessionDep):
    """Names only, for any signed-in user choosing a credential for a project stage."""
    return await CredentialService(session).list_options()


@router.get("", response_model=list[CredentialResponse])
async def list_credentials(admin: AdminUser, session: SessionDep):
    """All credentials with the projects using them. Never exposes key values."""
    return await CredentialService(session).list_credentials()


@router.post("", response_model=CredentialResponse, status_code=status.HTTP_201_CREATED)
async def create_credential(request: CreateCredentialRequest, admin: AdminUser, session: SessionDep):
    return await CredentialService(session).create_credential(request, admin)


@router.get("/{credential_id}", response_model=CredentialResponse)
async def get_credential(credential_id: uuid.UUID, admin: AdminUser, session: SessionDep):
    return await CredentialService(session).get_credential(credential_id)


@router.patch("/{credential_id}", response_model=CredentialResponse)
async def update_credential(
    credential_id: uuid.UUID, request: UpdateCredentialRequest, admin: AdminUser, session: SessionDep
):
    """Rename, rotate keys and/or change settings. Projects using it pick up the changes automatically."""
    return await CredentialService(session).update_credential(credential_id, request)


@router.delete("/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_credential(credential_id: uuid.UUID, admin: AdminUser, session: SessionDep):
    await CredentialService(session).delete_credential(credential_id)


@router.post("/{credential_id}/test", response_model=ConnectionTestResponse)
async def test_connection(credential_id: uuid.UUID, admin: AdminUser, session: SessionDep):
    """Check the connection works (URL reachable, key accepted). Saves nothing."""
    connection = await CredentialService(session).get_connection(credential_id)
    ok, message = await llm_gateway.test_connection(connection)
    return ConnectionTestResponse(ok=ok, message=message)


@router.get("/{credential_id}/models", response_model=ModelOptionsResponse)
async def list_models(
    credential_id: uuid.UUID, user: CurrentUser, session: SessionDep, stage: ModelStage | None = None
):
    """Model dropdown for a project stage: catalog models checked against the provider,
    or the connection's registered models (Ollama / Azure)."""
    return await ModelCatalogService(session).model_options(credential_id, stage)


# ---------- registered models (Ollama / Azure configuration page) ----------

@router.get("/{credential_id}/registered-models", response_model=list[RegisteredModelResponse])
async def list_registered_models(credential_id: uuid.UUID, admin: AdminUser, session: SessionDep):
    return await ModelCatalogService(session).list_registered(credential_id)


@router.post(
    "/{credential_id}/registered-models", response_model=RegisteredModelResponse, status_code=status.HTTP_201_CREATED
)
async def register_model(
    credential_id: uuid.UUID, request: RegisterModelRequest, admin: AdminUser, session: SessionDep
):
    """Add an Azure deployment or an Ollama model. It is tested right away; a failed test is saved
    with status `failed` and the reason, so it can be fixed and re-tested."""
    return await ModelCatalogService(session).register(credential_id, request)


@router.post("/{credential_id}/registered-models/sync", response_model=list[RegisteredModelResponse])
async def sync_ollama_models(credential_id: uuid.UUID, admin: AdminUser, session: SessionDep, test: bool = False):
    """Ollama only: read the models downloaded on the server. `test=true` also test-calls each one (slow)."""
    return await ModelCatalogService(session).sync_ollama(credential_id, test)


@router.post("/{credential_id}/registered-models/{model_id}/test", response_model=RegisteredModelResponse)
async def test_registered_model(credential_id: uuid.UUID, model_id: uuid.UUID, admin: AdminUser, session: SessionDep):
    return await ModelCatalogService(session).test_registered(credential_id, model_id)


@router.delete("/{credential_id}/registered-models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_registered_model(
    credential_id: uuid.UUID, model_id: uuid.UUID, admin: AdminUser, session: SessionDep
):
    """Blocked while a project uses the model."""
    await ModelCatalogService(session).delete_registered(credential_id, model_id)
