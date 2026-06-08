from fastapi import HTTPException
from datetime import datetime

from web_api.data_models.enums import ModelProvider
from web_api.data_models.CredentialModels import (
    ProviderCredentialModel,
    SetCredentialRequest,
    CredentialStatusResponse,
    PROVIDER_CREDENTIAL_SCHEMA,
)
from web_api.services.encryption_service import EncryptionService


class CredentialService:

    @staticmethod
    async def set_credential(provider: ModelProvider, request: SetCredentialRequest) -> CredentialStatusResponse:
        required = PROVIDER_CREDENTIAL_SCHEMA.get(provider, [])
        missing = [f for f in required if f not in request.fields]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required fields for {provider}: {missing}"
            )

        enc = EncryptionService.get()
        encrypted = {k: enc.encrypt(v) for k, v in request.fields.items()}

        existing = await ProviderCredentialModel.find_one(
            ProviderCredentialModel.provider == provider
        )

        if existing:
            existing.encrypted_fields = encrypted
            existing.updated_at = datetime.utcnow()
            await existing.save()
        else:
            doc = ProviderCredentialModel(provider=provider, encrypted_fields=encrypted)
            await doc.insert()

        return CredentialStatusResponse(
            provider=provider,
            configured=True,
            fields_present=list(encrypted.keys()),
            updated_at=datetime.utcnow(),
        )

    @staticmethod
    async def get_decrypted_fields(provider: ModelProvider) -> dict[str, str]:
        doc = await ProviderCredentialModel.find_one(
            ProviderCredentialModel.provider == provider
        )
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"No credentials configured for provider: {provider}. Add them in the Credentials settings."
            )
        enc = EncryptionService.get()
        return {k: enc.decrypt(v) for k, v in doc.encrypted_fields.items()}

    @staticmethod
    async def list_status() -> list[CredentialStatusResponse]:
        docs = await ProviderCredentialModel.find_all().to_list()
        configured = {doc.provider: doc for doc in docs}

        results = []
        for provider in ModelProvider:
            doc = configured.get(provider)
            results.append(CredentialStatusResponse(
                provider=provider,
                configured=doc is not None,
                fields_present=list(doc.encrypted_fields.keys()) if doc else [],
                updated_at=doc.updated_at if doc else None,
            ))
        return results

    @staticmethod
    async def delete_credential(provider: ModelProvider):
        doc = await ProviderCredentialModel.find_one(
            ProviderCredentialModel.provider == provider
        )
        if not doc:
            raise HTTPException(status_code=404, detail=f"No credentials found for {provider}")
        await doc.delete()
        return {"message": f"Credentials for {provider} deleted"}
