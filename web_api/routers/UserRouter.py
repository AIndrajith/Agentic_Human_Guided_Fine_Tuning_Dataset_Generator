from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from web_api.data_models.DataModels import (
    AddUserRequest,
    SetupAccountRequest,
    LoginResponse,
    UserResponse,
)
from web_api.services.AuthService import AuthService, AuthPayload
from web_api.deps.auth import AdminUser, AnyUser, get_auth_service

router = APIRouter(prefix="/users", tags=["User Management"])


@router.post("/add-user", response_model=UserResponse)
async def add_user(
    body: AddUserRequest,
    admin: AuthPayload = AdminUser,                     # only admins; the caller's identity
    auth: AuthService = Depends(get_auth_service),
):
    """Admin invites a new user. The user sets their own password via the emailed link."""
    user = await auth.add_user(body.email, body.app_role)
    return UserResponse.from_user(user)


@router.post("/setup")
async def setup_account(
    body: SetupAccountRequest,
    auth: AuthService = Depends(get_auth_service),
):
    """Public: an invited user sets their username + password using the setup token."""
    user = await auth.authenticate_initial_token(body.token)
    user_id = await auth.set_credentials_first_login(user.id, body.username, body.password)
    return {"user_id": str(user_id)}


@router.post("/login", response_model=LoginResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    auth: AuthService = Depends(get_auth_service),
):
    """Public: exchange email + password for a JWT. (OAuth2 form: the 'username' field = email.)"""
    token = await auth.authenticate_user(form.username, form.password)
    return LoginResponse(access_token=token)


@router.get("/me", response_model=AuthPayload)
async def me(current_user: AuthPayload = AnyUser):
    """Return the currently authenticated user's identity (from the JWT)."""
    return current_user
