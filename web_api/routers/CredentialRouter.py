import uuid

from fastapi import APIRouter, status

from web_api.data_models.CredentialModels import (
    PROVIDER_CREDENTIAL_SCHEMA,
    CreateCredentialRequest,
    CredentialOptionResponse,
    CredentialResponse,
    ProviderSchemaResponse,
    UpdateCredentialRequest,
)
from web_api.deps.auth import AdminUser, CurrentUser, SessionDep
from web_api.services.credential_service import CredentialService

router = APIRouter(prefix="/credentials", tags=["Credentials"])


@router.get("/schema", response_model=list[ProviderSchemaResponse])
async def get_all_schemas(user: CurrentUser):
    """What fields the UI should render per provider."""
    return [
        ProviderSchemaResponse(provider=p, required_fields=fields)
        for p, fields in PROVIDER_CREDENTIAL_SCHEMA.items()
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
    """Rename and/or rotate keys. Projects using it pick up the new keys automatically."""
    return await CredentialService(session).update_credential(credential_id, request)


@router.delete("/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_credential(credential_id: uuid.UUID, admin: AdminUser, session: SessionDep):
    await CredentialService(session).delete_credential(credential_id)
