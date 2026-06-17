from pydantic import BaseModel, EmailStr
from .enums import Datatype, AppRole
from typing import List


class AddUserRequest(BaseModel):
    email: EmailStr
    app_role: AppRole = AppRole.USER


class SetupAccountRequest(BaseModel):
    token: str
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    username: str | None = None
    app_role: AppRole
    is_active: bool

    @classmethod
    def from_user(cls, user) -> "UserResponse":
        return cls(
            id=str(user.id),
            email=user.email,
            username=user.username,
            app_role=user.app_role,
            is_active=user.is_active,
        )

class CreateProjectRequest(BaseModel):
    project_title: str
    project_description: str
    main_data_type: Datatype

class UpdateProjectRequest(BaseModel):
    project_title: str | None = None
    project_description: str | None = None
    main_data_type: Datatype | None = None


class ProcessDocumentsRequest(BaseModel):
    project_id: str
    document_ids: List[str]  