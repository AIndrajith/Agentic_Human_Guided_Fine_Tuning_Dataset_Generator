import httpx

from web_api.data_models.enums import AppRole
from web_api.db.models import User
from web_api.db.session import get_sessionmaker
from web_api.services.SecurityService import SecurityService

PASSWORD = "Str0ng!Pass"


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
