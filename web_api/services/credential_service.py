import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.CredentialModels import (
    CreateCredentialRequest,
    CredentialProjectRef,
    CredentialResponse,
    UpdateCredentialRequest,
)
from web_api.data_models.enums import ModelProvider
from web_api.db.models import Project, ProjectStageModel, ProviderCredential, User
from web_api.errors import ConflictError, CredentialInUse, CredentialNotFound, ValidationError
from web_api.services.encryption_service import EncryptionService
from web_api.services.llm_gateway import Connection
from web_api.services.providers import get_spec

_URL_SETTINGS = ("base_url", "endpoint")


class CredentialService:
    """Global provider connections: every admin can see and manage all of them."""

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _check_fields(
        provider: ModelProvider, kind: str, values: dict[str, str], required: tuple[str, ...], optional: tuple[str, ...]
    ) -> dict[str, str]:
        cleaned = {k: v.strip() for k, v in values.items() if v and v.strip()}
        missing = [f for f in required if not cleaned.get(f)]
        unknown = [f for f in cleaned if f not in required + optional]
        if missing or unknown:
            problems = []
            if missing:
                problems.append(f"missing {missing}")
            if unknown:
                problems.append(f"unknown {unknown}")
            raise ValidationError(
                f"Invalid {kind} for {provider.value}: {'; '.join(problems)}. Expected {list(required + optional)}"
            )
        return cleaned

    @classmethod
    def _validate_secrets(cls, provider: ModelProvider, secrets: dict[str, str]) -> dict[str, str]:
        spec = get_spec(provider)
        return cls._check_fields(provider, "secrets", secrets, spec.secret_fields, spec.optional_secret_fields)

    @classmethod
    def _validate_settings(cls, provider: ModelProvider, settings: dict[str, str]) -> dict[str, str]:
        spec = get_spec(provider)
        merged = {**spec.default_settings, **{k: v for k, v in settings.items() if v and v.strip()}}
        cleaned = cls._check_fields(provider, "settings", merged, spec.settings_fields, ())
        for name in _URL_SETTINGS:
            if name in cleaned:
                if not cleaned[name].startswith(("http://", "https://")):
                    raise ValidationError(f"'{name}' must start with http:// or https://")
                cleaned[name] = cleaned[name].rstrip("/")
        return cleaned

    @staticmethod
    def _encrypt(fields: dict[str, str]) -> dict[str, str]:
        enc = EncryptionService.get()
        return {k: enc.encrypt(v) for k, v in fields.items()}

    async def get(self, credential_id: uuid.UUID) -> ProviderCredential:
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
            secrets_present=sorted(credential.encrypted_fields),
            settings=credential.settings,
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
        return await self._to_response(await self.get(credential_id))

    async def create_credential(self, request: CreateCredentialRequest, admin: User) -> CredentialResponse:
        name = request.name.strip()
        await self._ensure_name_free(name)
        credential = ProviderCredential(
            name=name,
            provider=request.provider,
            encrypted_fields=self._encrypt(self._validate_secrets(request.provider, request.secrets)),
            settings=self._validate_settings(request.provider, request.settings),
            created_by=admin.id,
        )
        self.session.add(credential)
        await self.session.commit()
        return self._response(credential, [])

    async def update_credential(self, credential_id: uuid.UUID, request: UpdateCredentialRequest) -> CredentialResponse:
        credential = await self.get(credential_id)
        if request.name is not None:
            name = request.name.strip()
            await self._ensure_name_free(name, exclude_id=credential.id)
            credential.name = name
        if request.secrets is not None:
            credential.encrypted_fields = self._encrypt(self._validate_secrets(credential.provider, request.secrets))
        if request.settings is not None:
            credential.settings = self._validate_settings(credential.provider, request.settings)
        await self.session.commit()
        return await self._to_response(credential)

    async def delete_credential(self, credential_id: uuid.UUID) -> None:
        credential = await self.get(credential_id)
        used_by = (await self._usage([credential.id]))[credential.id]
        if used_by:
            titles = ", ".join(p.title for p in used_by)
            raise CredentialInUse(f"Credential is attached to project(s): {titles}. Detach it first.")
        await self.session.delete(credential)
        await self.session.commit()

    @staticmethod
    def connection_for(credential: ProviderCredential) -> Connection:
        """Decrypted secrets + settings, ready for LLM calls. Never return this to a client."""
        enc = EncryptionService.get()
        return Connection(
            provider=credential.provider,
            secrets={k: enc.decrypt(v) for k, v in credential.encrypted_fields.items()},
            settings=dict(credential.settings),
        )

    async def get_connection(self, credential_id: uuid.UUID) -> Connection:
        return self.connection_for(await self.get(credential_id))
