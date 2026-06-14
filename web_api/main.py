import utils.http_session
import utils.transport
import utils.session
from fastapi import FastAPI
from contextlib import asynccontextmanager
from web_api.database import init_db, close_db
from web_api.routers.FileMangerRouter import router as file_router
from web_api.routers.ProjectMangerRouter import router as project_router
from web_api.routers.CredentialRouter import router as credential_router
from web_api.routers.ModelConfigRouter import router as model_config_router
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(title="Synthetic Data Generation", lifespan=lifespan)

app.include_router(file_router)
app.include_router(project_router)
app.include_router(credential_router)
app.include_router(model_config_router)


@app.get("/")
async def root():
    return {"message": "Synthetic Data Generation API"}
