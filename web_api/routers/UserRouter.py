import uuid

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from web_api.data_models.DataModels import (
    AddUserRequest,
    LoginResponse,
    ResendInviteRequest,
    SetupAccountRequest,
    SetUserActiveRequest,
    UserResponse,
)
from web_api.deps.auth import AdminUser, AuthServiceDep, CurrentUser, SessionDep
from web_api.services.UserService import UserService

router = APIRouter(prefix="/users", tags=["User Management"])


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def add_user(body: AddUserRequest, admin: AdminUser, auth: AuthServiceDep):
    """Admin invites a new user. The user sets their own password via the emailed link."""
    return await auth.add_user(body.email, body.app_role)


@router.get("", response_model=list[UserResponse])
async def list_users(admin: AdminUser, session: SessionDep):
    return await UserService(session).list_all()


@router.post("/resend-invite", response_model=UserResponse)
async def resend_invite(body: ResendInviteRequest, admin: AdminUser, auth: AuthServiceDep):
    """Issue a fresh setup link (the old one stops working)."""
    return await auth.resend_invite(body.email)


@router.patch("/{user_id}/active", response_model=UserResponse)
async def set_user_active(user_id: uuid.UUID, body: SetUserActiveRequest, admin: AdminUser, auth: AuthServiceDep):
    return await auth.set_active(user_id, body.is_active, acting_admin=admin)


@router.post("/setup", response_model=UserResponse)
async def setup_account(body: SetupAccountRequest, auth: AuthServiceDep):
    """Public: an invited user sets their username + password using the setup token."""
    return await auth.setup_account(body.token, body.username, body.password)


@router.post("/login", response_model=LoginResponse)
async def login(auth: AuthServiceDep, form: OAuth2PasswordRequestForm = Depends()):
    """Public: exchange email + password for a JWT. (OAuth2 form: the 'username' field = email.)"""
    token = await auth.authenticate_user(form.username, form.password)
    return LoginResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUser):
    return current_user
