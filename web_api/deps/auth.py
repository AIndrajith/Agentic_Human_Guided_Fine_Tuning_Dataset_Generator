import uuid
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from web_api.core.config import get_settings
from web_api.data_models.enums import AppRole
from web_api.db.models import User
from web_api.db.session import get_session
from web_api.errors import AccountNotActive, AuthorizationError, InvalidOrExpiredToken
from web_api.services.AuthService import AuthService
from web_api.services.JWTService import JWTService

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _get_jwt_service(request: Request) -> JWTService:
    return request.app.state.jwt_service


def get_auth_service(request: Request, session: SessionDep) -> AuthService:
    """Per-request AuthService over the request's DB session and the app's stateless services."""
    state = request.app.state
    return AuthService(session, get_settings(), state.security_service, state.jwt_service, state.email_service)


async def get_current_user(
    session: SessionDep,
    token: str = Depends(_oauth2_scheme),
    jwt_service: JWTService = Depends(_get_jwt_service),
) -> User:
    """Verify the token, then load the user so deactivation takes effect immediately."""
    payload = jwt_service.verify_token(token)
    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        raise InvalidOrExpiredToken()
    user = await session.get(User, user_id)
    if not user:
        raise InvalidOrExpiredToken()
    if not user.is_active:
        raise AccountNotActive()
    return user


def _require_role(*allowed_roles: AppRole):
    async def dependency(user: User = Depends(get_current_user)) -> User:
        if user.app_role not in allowed_roles:
            raise AuthorizationError("Access denied. Insufficient permissions.")
        return user

    return dependency


CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(_require_role(AppRole.ADMIN))]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
