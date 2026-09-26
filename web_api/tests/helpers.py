import uuid

import httpx

from web_api.data_models.enums import AppRole, ModelProvider, ModelStage
from web_api.db.models import ProjectStageModel, ProviderCredential, User
from web_api.db.session import get_sessionmaker
from web_api.services.encryption_service import EncryptionService
from web_api.services.SecurityService import SecurityService

PASSWORD = "Str0ng!Pass"
API_KEY = "sk-abc"


async def configure_stages(project_id: str, academic: bool = False) -> None:
    """Give a project the model steps processing needs, straight in the DB (no test calls)."""
    async with get_sessionmaker()() as session:
        credential = ProviderCredential(
            name=f"OpenAI {uuid.uuid4().hex[:6]}",
            provider=ModelProvider.OPENAI,
            encrypted_fields={"api_key": EncryptionService.get().encrypt(API_KEY)},
            settings={},
        )
        session.add(credential)
        await session.flush()
        stages = {
            ModelStage.META_AGENT: ("gpt-4o-mini", None),
            ModelStage.EMBEDDER: ("text-embedding-3-large", 3072),
        }
        if academic:
            stages[ModelStage.VISION] = ("gpt-4o-mini", None)
        for stage, (model_name, dim) in stages.items():
            session.add(ProjectStageModel(
                project_id=uuid.UUID(project_id), stage=stage, credential_id=credential.id,
                model_name=model_name, embedding_dim=dim,
            ))
        await session.commit()


async def create_user(email: str, role: AppRole = AppRole.USER, username: str | None = None) -> User:
    async with get_sessionmaker()() as session:
        user = User(
            email=email,
            username=username or email.split("@")[0],
            password_hash=SecurityService().hash_password(PASSWORD),
            app_role=role,
            must_change_password=False,
        )
        session.add(user)
        await session.commit()
        return user


async def login(client: httpx.AsyncClient, email: str, password: str = PASSWORD) -> dict[str, str]:
    r = await client.post("/users/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
