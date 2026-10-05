import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from web_api.core.config import Settings
from web_api.data_models.enums import AppRole, EmailKind
from web_api.db.models import EmailEvent, User
from web_api.errors import (
    AccountNotActive,
    ConflictError,
    InvalidCredentials,
    InvalidOrExpiredToken,
    UserAlreadyExists,
    UserNotFound,
)
from web_api.services.EmailService import EmailService
from web_api.services.JWTService import JWTService
from web_api.services.SecurityService import SecurityService
from web_api.services.UserService import UserService


class AuthService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        security_service: SecurityService,
        jwt_service: JWTService,
        email_service: EmailService,
    ):
        self.session = session
        self.users = UserService(session)
        self.security_service = security_service
        self.jwt_service = jwt_service
        self.email_service = email_service
        self.setup_token_ttl = timedelta(hours=settings.SETUP_TOKEN_EXPIRE_HOURS)

    async def _issue_invite(self, user: User) -> None:
        """Give the user a fresh setup token and email it. Caller commits."""
        token = self.security_service.generate_secure_token()
        user.setup_token_hash = self.security_service.hash_token(token)
        user.setup_token_expires_at = datetime.now(timezone.utc) + self.setup_token_ttl
        await self.session.flush()

        resend_id = await self.email_service.send_invite(user.email, token)
        if resend_id:
            self.session.add(EmailEvent(user_id=user.id, resend_id=resend_id, kind=EmailKind.INVITE))

    async def add_user(self, email: str, app_role: AppRole) -> User:
        if await self.users.find_by_email(email):
            raise UserAlreadyExists()
        user = await self.users.add_invited_user(email, app_role)
        # if the email fails, the exception propagates and the session rolls back the insert
        await self._issue_invite(user)
        await self.session.commit()
        return user

    async def resend_invite(self, email: str) -> User:
        user = await self.users.find_by_email(email)
        if not user:
            raise UserNotFound()
        if not user.must_change_password:
            raise ConflictError("This account is already set up")
        await self._issue_invite(user)
        await self.session.commit()
        return user

    async def setup_account(self, token: str, username: str, password: str) -> User:
        user = await self.users.find_by_token_hash(self.security_service.hash_token(token))
        if not user or not user.setup_token_expires_at:
            raise InvalidOrExpiredToken("Invalid token")
        if user.setup_token_expires_at < datetime.now(timezone.utc):
            raise InvalidOrExpiredToken("Token expired")
        if not user.is_active:
            raise AccountNotActive()
        if not user.must_change_password:
            raise ConflictError("This account is already set up")

        self.security_service.validate_password(password)
        existing = await self.users.find_by_username(username)
        if existing and existing.id != user.id:
            raise ConflictError("Username is already taken")

        user.username = username
        user.password_hash = self.security_service.hash_password(password)
        user.must_change_password = False
        # one-time token: clear it so the link can't be reused
        user.setup_token_hash = None
        user.setup_token_expires_at = None
        await self.session.commit()
        return user

    async def authenticate_user(self, email: str, password: str) -> str:
        user = await self.users.find_by_email(email)
        # same error for unknown email and wrong password: don't reveal which emails exist
        if not user or not self.security_service.verify_password(password, user.password_hash):
            raise InvalidCredentials()
        if not user.is_active:
            raise AccountNotActive()
        return self.create_access_token(user)

    def create_access_token(self, user: User) -> str:
        return self.jwt_service.create_token(
            subject=str(user.id),
            claims={"email": user.email, "username": user.username, "role": user.app_role.value},
        )

    async def set_active(self, user_id: uuid.UUID, is_active: bool, acting_admin: User) -> User:
        user = await self.users.get(user_id)
        if user.id == acting_admin.id and not is_active:
            raise ConflictError("You cannot deactivate your own account")
        user.is_active = is_active
        await self.session.commit()
        return user
