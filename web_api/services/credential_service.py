import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.CredentialModels import (
    PROVIDER_CREDENTIAL_SCHEMA,
    CreateCredentialRequest,
    CredentialProjectRef,
    CredentialResponse,
    UpdateCredentialRequest,
)
from web_api.data_models.enums import ModelProvider
from web_api.db.models import Project, ProjectStageModel, ProviderCredential, User
from web_api.errors import ConflictError, CredentialInUse, CredentialNotFound, ValidationError
from web_api.services.encryption_service import EncryptionService


class CredentialService:
    """Global provider credentials: every admin can see and manage all of them."""

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _validate_fields(provider: ModelProvider, fields: dict[str, str]) -> dict[str, str]:
        allowed = PROVIDER_CREDENTIAL_SCHEMA.get(provider, [])
        cleaned = {k: v.strip() for k, v in fields.items()}
        missing = [f for f in allowed if not cleaned.get(f)]
        unknown = [f for f in cleaned if f not in allowed]
        if missing or unknown:
            problems = []
            if missing:
                problems.append(f"missing {missing}")
            if unknown:
                problems.append(f"unknown {unknown}")
            raise ValidationError(f"Invalid fields for {provider.value}: {'; '.join(problems)}. Expected {allowed}")
        return cleaned

    @staticmethod
    def _encrypt(fields: dict[str, str]) -> dict[str, str]:
        enc = EncryptionService.get()
        return {k: enc.encrypt(v) for k, v in fields.items()}

    async def _get(self, credential_id: uuid.UUID) -> ProviderCredential:
        credential = await self.session.get(ProviderCredential, credential_id)
        if not credential:
            raise CredentialNotFound()
        return credential

    async def _ensure_name_free(self, name: str, exclude_id: uuid.UUID | None = None) -> None:
        existing = await self.session.scalar(select(ProviderCredential).where(ProviderCredential.name == name))
        if existing and existing.id != exclude_id:
            raise ConflictError(f"A credential named '{name}' already exists")

    async def _usage(self, credential_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[CredentialProjectRef]]:
        rows = await self.session.execute(
            select(ProjectStageModel.credential_id, Project.id, Project.title)
            .join(Project, Project.id == ProjectStageModel.project_id)
            .where(ProjectStageModel.credential_id.in_(credential_ids))
            .distinct()
        )
        usage: dict[uuid.UUID, list[CredentialProjectRef]] = {cid: [] for cid in credential_ids}
        for credential_id, project_id, title in rows:
            usage[credential_id].append(CredentialProjectRef(project_id=project_id, title=title))
        return usage

    async def _to_response(self, credential: ProviderCredential) -> CredentialResponse:
        usage = await self._usage([credential.id])
        return self._response(credential, usage[credential.id])

    @staticmethod
    def _response(credential: ProviderCredential, used_by: list[CredentialProjectRef]) -> CredentialResponse:
        return CredentialResponse(
            id=credential.id,
            name=credential.name,
            provider=credential.provider,
            fields_present=sorted(credential.encrypted_fields),
            used_by_projects=used_by,
            created_by=credential.created_by,
            created_at=credential.created_at,
            updated_at=credential.updated_at,
        )

    async def list_credentials(self) -> list[CredentialResponse]:
        credentials = list(await self.session.scalars(
            select(ProviderCredential).order_by(ProviderCredential.provider, ProviderCredential.name)
        ))
        usage = await self._usage([c.id for c in credentials])
        return [self._response(c, usage[c.id]) for c in credentials]

    async def list_options(self) -> list[ProviderCredential]:
        return list(await self.session.scalars(
            select(ProviderCredential).order_by(ProviderCredential.provider, ProviderCredential.name)
        ))

    async def get_credential(self, credential_id: uuid.UUID) -> CredentialResponse:
        return await self._to_response(await self._get(credential_id))

    async def create_credential(self, request: CreateCredentialRequest, admin: User) -> CredentialResponse:
        name = request.name.strip()
        await self._ensure_name_free(name)
        fields = self._validate_fields(request.provider, request.fields)
        credential = ProviderCredential(
            name=name,
            provider=request.provider,
            encrypted_fields=self._encrypt(fields),
            created_by=admin.id,
        )
        self.session.add(credential)
        await self.session.commit()
        return self._response(credential, [])

    async def update_credential(self, credential_id: uuid.UUID, request: UpdateCredentialRequest) -> CredentialResponse:
        credential = await self._get(credential_id)
        if request.name is not None:
            name = request.name.strip()
            await self._ensure_name_free(name, exclude_id=credential.id)
            credential.name = name
        if request.fields is not None:
            credential.encrypted_fields = self._encrypt(self._validate_fields(credential.provider, request.fields))
        await self.session.commit()
        return await self._to_response(credential)

    async def delete_credential(self, credential_id: uuid.UUID) -> None:
        credential = await self._get(credential_id)
        used_by = (await self._usage([credential.id]))[credential.id]
        if used_by:
            titles = ", ".join(p.title for p in used_by)
            raise CredentialInUse(f"Credential is attached to project(s): {titles}. Detach it first.")
        await self.session.delete(credential)
        await self.session.commit()

    async def get_decrypted_fields(self, credential_id: uuid.UUID) -> tuple[ModelProvider, dict[str, str]]:
        credential = await self._get(credential_id)
        enc = EncryptionService.get()
        return credential.provider, {k: enc.decrypt(v) for k, v in credential.encrypted_fields.items()}
