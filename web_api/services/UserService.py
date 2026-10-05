import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.data_models.enums import AppRole
from web_api.db.models import User
from web_api.errors import UserNotFound


class UserService:
    """Data access for users. Callers own the transaction (commit)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().lower()

    async def get(self, user_id: uuid.UUID) -> User:
        user = await self.session.get(User, user_id)
        if not user:
            raise UserNotFound()
        return user

    async def find_by_email(self, email: str) -> User | None:
        return await self.session.scalar(select(User).where(User.email == self.normalize_email(email)))

    async def find_by_username(self, username: str) -> User | None:
        return await self.session.scalar(select(User).where(User.username == username))

    async def find_by_token_hash(self, token_hash: str) -> User | None:
        return await self.session.scalar(select(User).where(User.setup_token_hash == token_hash))

    async def list_all(self) -> list[User]:
        return list(await self.session.scalars(select(User).order_by(User.created_at)))

    async def add_invited_user(self, email: str, app_role: AppRole) -> User:
        user = User(email=self.normalize_email(email), app_role=app_role)
        self.session.add(user)
        await self.session.flush()
        return user
