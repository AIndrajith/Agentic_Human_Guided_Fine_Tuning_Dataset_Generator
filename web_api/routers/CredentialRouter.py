from fastapi import APIRouter
from web_api.data_models.enums import ModelProvider
from web_api.data_models.CredentialModels import (
    SetCredentialRequest,
    CredentialStatusResponse,
    ProviderSchemaResponse,
    PROVIDER_CREDENTIAL_SCHEMA,
)
from web_api.services.credential_service import CredentialService

router = APIRouter(prefix="/credentials", tags=["Credentials"])


@router.get("/schema", response_model=list[ProviderSchemaResponse])
async def get_all_schemas():
    """Returns what fields the UI should render per provider."""
    return [
        ProviderSchemaResponse(provider=p, required_fields=fields)
        for p, fields in PROVIDER_CREDENTIAL_SCHEMA.items()
    ]


@router.get("/", response_model=list[CredentialStatusResponse])
async def list_credentials():
    """Which providers are configured. Never exposes actual key values."""
    return await CredentialService.list_status()


@router.post("/{provider}", response_model=CredentialStatusResponse)
async def set_credential(provider: ModelProvider, request: SetCredentialRequest):
    return await CredentialService.set_credential(provider, request)


@router.delete("/{provider}")
async def delete_credential(provider: ModelProvider):
    return await CredentialService.delete_credential(provider)
