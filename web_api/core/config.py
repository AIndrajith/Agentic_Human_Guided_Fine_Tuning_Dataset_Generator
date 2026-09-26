from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# repo-root .env, shared with workers
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # ---------- Database ----------
    DATABASE_URL: str                          # synth_app (runtime, data only)
    MIGRATION_DATABASE_URL: str | None = None  # synth_owner (Alembic, LangGraph setup)
    TEST_DATABASE_URL: str | None = None
    TEST_MIGRATION_DATABASE_URL: str | None = None
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # ---------- Auth ----------
    JWT_SECRET_KEY: SecretStr
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 20
    SETUP_TOKEN_EXPIRE_HOURS: int = 24
    ENCRYPTION_KEY: SecretStr

    # ---------- Email (Resend) ----------
    RESEND_API_KEY: SecretStr | None = None
    FROM_EMAIL: str | None = None
    EMAIL_SUBJECT: str = "Welcome to Our App"
    EMAIL_HTML_TEMPLATE: str = ""
    HOST_URL: str = "http://localhost:5173"

    # ---------- MinIO ----------
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: SecretStr = SecretStr("minioadmin")
    MINIO_BUCKET: str = "synthetic-data"
    MINIO_SECURE: bool = False
    MAX_UPLOAD_MB: int = 200

    # ---------- Celery / workers ----------
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    # shared secret workers send as X-Internal-Token on /internal/* and /webhooks/*
    INTERNAL_API_TOKEN: SecretStr

    # ---------- HTTP ----------
    CORS_ORIGINS: str = "http://localhost:5173"  # comma-separated

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
