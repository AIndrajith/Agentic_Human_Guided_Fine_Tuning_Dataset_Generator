import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Text, false, func
from sqlalchemy.orm import Mapped, mapped_column

from web_api.data_models.enums import Datatype, ProjectRole
from web_api.db.base import Base, TimestampMixin, UUIDPkMixin, str_enum


class Project(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "projects"

    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="", server_default="")
    data_type: Mapped[Datatype] = mapped_column(str_enum(Datatype, "data_type"))
    created_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"), index=True)
    # True once the first document is embedded; embedder can't change after that
    embedding_locked: Mapped[bool] = mapped_column(default=False, server_default=false())


class ProjectMember(Base):
    __tablename__ = "project_members"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    role: Mapped[ProjectRole] = mapped_column(str_enum(ProjectRole, "project_role"))
    added_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    added_at: Mapped[datetime] = mapped_column(server_default=func.now())
