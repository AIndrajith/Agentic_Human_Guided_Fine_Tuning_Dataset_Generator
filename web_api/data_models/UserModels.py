from beanie import Document, PydanticObjectId
from pydantic import Field, EmailStr
from datetime import datetime, timezone
from pymongo import IndexModel, ASCENDING
from web_api.data_models.enums import AppRole, ProjectRole


class UserModel(Document):
    email: EmailStr
    username: str
    hashed_password: str

    app_role: AppRole = AppRole.USER
    is_active: bool = True
    must_change_password: bool = False

    setup_token: str = None
    setup_token_expiry: datetime = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "users"
        indexes = [
            IndexModel([("email", ASCENDING)], unique=True),
            IndexModel([("username", ASCENDING)], unique=True)
        ]


class ProjectMemberModel(Document):
    project_id: PydanticObjectId
    user_id: PydanticObjectId
    project_role: ProjectRole
    added_by: PydanticObjectId
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "project_members"
        indexes = [
            # A user can hold only one role per project; also serves member lookups
            IndexModel([("project_id", ASCENDING), ("user_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING)]),
        ]


class EmailVerificationModel(Document):
    user_id: PydanticObjectId
    resend_id: str
    sended_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    class Settings:
        name = "email_verifications"
        indexes = [
            IndexModel([("resend_id", ASCENDING)], unique=True),
            IndexModel([("user_id", ASCENDING)]),
        ]
