from fastapi import FastAPI
from contextlib import asynccontextmanager
from web_api.database import init_db, close_db
from web_api.routers.FileMangerRouter import router as file_router
from web_api.routers.ProjectMangerRouter import router as project_router
from web_api.routers.CredentialRouter import router as credential_router
from web_api.routers.ModelConfigRouter import router as model_config_router
from web_api.routers.UserRouter import router as user_router
from web_api.services.JWTService import JWTService
from web_api.services.SecurityService import SecurityService
from web_api.services.UserService import UserService
from web_api.services.EmailService import EmailService
from web_api.services.AuthService import AuthService
from web_api.errors import register_error_handlers
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # build the stateless service singletons once, after env is loaded
    security_service = SecurityService()
    user_service = UserService(security_service)
    app.state.jwt_service = JWTService()
    app.state.auth_service = AuthService(
        user_service, security_service, app.state.jwt_service, EmailService()
    )
    yield
    await close_db()


app = FastAPI(title="Synthetic Data Generation", lifespan=lifespan)

register_error_handlers(app)

app.include_router(file_router)
app.include_router(project_router)
app.include_router(credential_router)
app.include_router(model_config_router)
app.include_router(user_router)


@app.get("/")
async def root():
    return {"message": "Synthetic Data Generation API"}
