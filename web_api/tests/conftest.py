"""Tests run against the `synthetic_data_test` DB with fake email + MinIO.

From the repo root:  uv run --project web_api pytest web_api/tests
"""
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Point the app at the test DB *before* anything imports settings/engine.
from web_api.core.config import Settings, get_settings

_base = Settings()
if not _base.TEST_DATABASE_URL:
    raise RuntimeError("TEST_DATABASE_URL must be set in .env to run tests")
os.environ["DATABASE_URL"] = _base.TEST_DATABASE_URL
os.environ["MIGRATION_DATABASE_URL"] = _base.TEST_MIGRATION_DATABASE_URL or ""
get_settings.cache_clear()

import httpx  # noqa: E402
from sqlalchemy import text  # noqa: E402

from web_api.data_models.enums import AppRole  # noqa: E402
from web_api.db.session import get_sessionmaker  # noqa: E402
from web_api.main import app  # noqa: E402
from web_api.routers import InternalRouter as internal_module  # noqa: E402
from web_api.services import ExtractionService as extraction_module  # noqa: E402
from web_api.services import FileHandlerService as file_module  # noqa: E402
from web_api.services import ProcessingService as processing_module  # noqa: E402
from web_api.tests.helpers import create_user, login  # noqa: E402

WEB_API_DIR = Path(__file__).resolve().parents[1]


def pytest_asyncio_loop_factories(config, item):
    # psycopg async needs a selector loop (Windows defaults to Proactor)
    return {"selector": asyncio.SelectorEventLoop}


@pytest.fixture(scope="session", autouse=True)
def migrated_test_db():
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(WEB_API_DIR / "alembic.ini"), "-x", "db=test", "upgrade", "head"],
        check=True, cwd=WEB_API_DIR, capture_output=True,
    )


class FakeEmailService:
    enabled = True

    def __init__(self):
        self.sent: dict[str, str] = {}   # email -> latest token

    async def send_invite(self, to_email: str, token: str) -> str | None:
        self.sent[to_email] = token
        return f"fake-{len(self.sent)}-{token[:8]}"


class FakeMinio:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    async def upload(self, key, data, content_type="application/octet-stream"):
        self.objects[key] = data
        return key

    async def delete(self, key):
        self.objects.pop(key, None)

    async def delete_prefix(self, prefix):
        keys = [k for k in self.objects if k.startswith(prefix)]
        for k in keys:
            del self.objects[k]
        return len(keys)

    async def download(self, key):
        return self.objects[key]

    async def stat(self, key):
        class _Stat:
            size = len(self.objects[key])
        return _Stat()

    async def stream(self, key):
        yield self.objects[key]


class FakeCelery:
    def __init__(self):
        self.sent: list[dict] = []

    def send_task(self, name, args=None, task_id=None):
        self.sent.append({"name": name, "payload": args[0], "task_id": task_id})


@pytest.fixture(scope="session")
async def app_client():
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


@pytest.fixture
def email(app_client):
    fake = FakeEmailService()
    app.state.email_service = fake
    return fake


@pytest.fixture
def minio(monkeypatch):
    fake = FakeMinio()
    # every module that did `from ...MinioService import minio_service`
    for module in (file_module, extraction_module, internal_module):
        monkeypatch.setattr(module, "minio_service", fake)
    return fake


@pytest.fixture
def celery(monkeypatch):
    fake = FakeCelery()
    monkeypatch.setattr(processing_module, "get_celery", lambda: fake)
    return fake


@pytest.fixture
def internal_headers():
    return {"X-Internal-Token": get_settings().INTERNAL_API_TOKEN.get_secret_value()}


@pytest.fixture(autouse=True)
async def clean_db(app_client):
    yield
    # synth_app has DML only (no TRUNCATE); order respects RESTRICT foreign keys
    async with get_sessionmaker()() as session:
        await session.execute(text("DELETE FROM project_stage_models"))
        await session.execute(text("DELETE FROM projects"))
        await session.execute(text("DELETE FROM provider_credentials"))
        await session.execute(text("DELETE FROM users"))
        await session.commit()


@pytest.fixture
async def admin_headers(app_client):
    await create_user("admin@test.io", AppRole.ADMIN)
    return await login(app_client, "admin@test.io")


@pytest.fixture
async def user_headers(app_client):
    await create_user("alice@test.io")
    return await login(app_client, "alice@test.io")
