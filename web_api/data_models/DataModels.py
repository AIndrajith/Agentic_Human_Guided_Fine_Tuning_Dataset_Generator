"""API request/response schemas (Pydantic). ORM tables live in web_api/db/models."""
import uuid
from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from .enums import AppRole, Datatype, DocumentStatus, FileType, ProjectRole


class _ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------- Users / auth ----------

class AddUserRequest(BaseModel):
    email: EmailStr
    app_role: AppRole = AppRole.USER


class ResendInviteRequest(BaseModel):
    email: EmailStr


class SetupAccountRequest(BaseModel):
    token: str
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(max_length=128)


class SetUserActiveRequest(BaseModel):
    is_active: bool


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(_ORMModel):
    id: uuid.UUID
    email: EmailStr
    username: str | None = None
    app_role: AppRole
    is_active: bool
    must_change_password: bool
    created_at: datetime


# ---------- Projects ----------

class CreateProjectRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = ""
    data_type: Datatype

    @field_validator("title")
    @classmethod
    def _strip_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be blank")
        return v


class UpdateProjectRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    data_type: Datatype | None = None


class ProjectResponse(_ORMModel):
    id: uuid.UUID
    title: str
    description: str
    data_type: Datatype
    created_by: uuid.UUID
    embedding_locked: bool
    created_at: datetime
    updated_at: datetime


# ---------- Project members ----------

class AddMemberRequest(BaseModel):
    email: EmailStr
    role: ProjectRole = ProjectRole.WORKER


class UpdateMemberRequest(BaseModel):
    role: ProjectRole


class MemberResponse(BaseModel):
    user_id: uuid.UUID
    email: EmailStr
    username: str | None
    role: ProjectRole
    added_by: uuid.UUID | None
    added_at: datetime


# ---------- Documents ----------

class DocumentResponse(_ORMModel):
    id: uuid.UUID
    project_id: uuid.UUID
    original_name: str
    file_type: FileType
    content_type: str
    size_bytes: int
    status: DocumentStatus
    error: str | None
    uploaded_by: uuid.UUID | None
    created_at: datetime
    processed_at: datetime | None


# ---------- Processing (router ported next phase) ----------

class ProcessDocumentsRequest(BaseModel):
    project_id: str
    document_ids: List[str]
