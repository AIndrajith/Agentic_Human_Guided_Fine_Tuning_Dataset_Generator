import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, String, func, true
from sqlalchemy.orm import Mapped, mapped_column

from web_api.data_models.enums import AppRole, EmailKind
from web_api.db.base import Base, TimestampMixin, UUIDPkMixin, str_enum


class User(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("email = lower(email)", name="email_lowercase"),
    )

    email: Mapped[str] = mapped_column(String(320), unique=True)
    username: Mapped[str | None] = mapped_column(String(64), unique=True)   # set during /setup
    password_hash: Mapped[str | None] = mapped_column(String(255))           # set during /setup

    app_role: Mapped[AppRole] = mapped_column(
        str_enum(AppRole, "app_role"), default=AppRole.USER, server_default=AppRole.USER.value
    )
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    must_change_password: Mapped[bool] = mapped_column(default=True, server_default=true())

    # sha256 of the emailed token; the raw token is never stored
    setup_token_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    setup_token_expires_at: Mapped[datetime | None]


class EmailEvent(UUIDPkMixin, Base):
    __tablename__ = "email_events"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    resend_id: Mapped[str] = mapped_column(String(128), unique=True)
    kind: Mapped[EmailKind] = mapped_column(str_enum(EmailKind, "email_kind"))
    sent_at: Mapped[datetime] = mapped_column(server_default=func.now())
