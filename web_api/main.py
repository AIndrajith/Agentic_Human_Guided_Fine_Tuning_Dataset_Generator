import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web_api.core.config import get_settings
from web_api.db.checkpointer import open_checkpointer
from web_api.db.session import dispose_engine
from web_api.errors import register_error_handlers
from web_api.routers.CredentialRouter import router as credential_router
from web_api.routers.FileMangerRouter import router as file_router
from web_api.routers.ModelConfigRouter import router as model_config_router
from web_api.routers.ProjectMangerRouter import router as project_router
from web_api.routers.UserRouter import router as user_router
from web_api.services.EmailService import EmailService
from web_api.services.JWTService import JWTService
from web_api.services.MinioService import minio_service
from web_api.services.SecurityService import SecurityService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

settings = get_settings()  # fails fast on missing required env vars


def _check_event_loop() -> None:
    if sys.platform == "win32" and isinstance(asyncio.get_running_loop(), asyncio.ProactorEventLoop):
        raise RuntimeError(
            "psycopg async can't run on Windows' ProactorEventLoop. "
            "Start with: uvicorn web_api.main:app --loop asyncio:SelectorEventLoop"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_event_loop()
    # stateless singletons; per-request services are built in web_api/deps
    app.state.security_service = SecurityService()
    app.state.jwt_service = JWTService(settings)
    app.state.email_service = EmailService(settings)
    await minio_service.ensure_bucket()
    # LangGraph checkpointer (tables created by web_api.scripts.setup_checkpointer)
    checkpoint_pool, app.state.checkpointer = await open_checkpointer(settings.DATABASE_URL)
    try:
        yield
    finally:
        await checkpoint_pool.close()
        await dispose_engine()


app = FastAPI(title="Synthetic Data Generation", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_error_handlers(app)

app.include_router(user_router)
app.include_router(project_router)
app.include_router(file_router)
app.include_router(credential_router)
app.include_router(model_config_router)
# Not registered yet (still on MongoDB, ported next phase): ProcessingRouter, InternalRouter, WebhookRouter


@app.get("/")
async def root():
    return {"message": "Synthetic Data Generation API"}
