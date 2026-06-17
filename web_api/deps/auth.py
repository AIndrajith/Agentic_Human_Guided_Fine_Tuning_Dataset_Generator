from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from web_api.errors import InvalidCredentials,AuthorizationError
from web_api.services.JWTService import JWTService
from web_api.services.AuthService import AuthPayload, AuthService
from web_api.data_models.enums import AppRole


_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")


def _get_jwt_service(request: Request) -> JWTService:
    return request.app.state.jwt_service


def get_auth_service(request: Request) -> AuthService:
    """Accessor: hand routers the single AuthService built in lifespan."""
    return request.app.state.auth_service


def _get_current_user(
    token: str = Depends(_oauth2_scheme),
    jwt_service: JWTService = Depends(_get_jwt_service),
) -> AuthPayload:
    """Authentication only: verify the token, return the caller's identity."""
    try:
        payload_dict = jwt_service.verify_token(token)
        return AuthPayload(**payload_dict)
    except Exception:
        raise InvalidCredentials("Invalid authentication token") 


def _require_role(*allowed_roles: AppRole):
    """Factory: build a dependency that allows only the given app role(s)."""

    def dependency(
        current_user: AuthPayload = Depends(_get_current_user),
    ) -> AuthPayload:
        if current_user.role not in allowed_roles:
            raise AuthorizationError("Access denied. Insufficient permissions.")
        return current_user

    return dependency



AdminUser = Depends(_require_role(AppRole.ADMIN))
RegularUser = Depends(_require_role(AppRole.USER))
AnyUser = Depends(_get_current_user) 
