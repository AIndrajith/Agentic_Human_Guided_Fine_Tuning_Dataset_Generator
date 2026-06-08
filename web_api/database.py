from pymongo import AsyncMongoClient
from beanie import init_beanie
from web_api.data_models.BasicBeanieModels import DocumentModel, ProjectModel, ChunkModel
from web_api.data_models.ExtractedModels import ExtractedFictionModel, ExtractedAcademicModel
from web_api.data_models.ModelConfigModels import ProjectModelConfigModel
from web_api.data_models.CredentialModels import ProviderCredentialModel
import os


class Database:
    client: AsyncMongoClient = None

db = Database()


async def init_db():
    mongodb_url   = os.getenv("MONGODB_URL",   "mongodb://localhost:27017")
    database_name = os.getenv("DATABASE_NAME", "synthetic_data_db")

    db.client = AsyncMongoClient(mongodb_url)

    await init_beanie(
        database=db.client[database_name],
        document_models=[
            DocumentModel,
            ProjectModel,
            ChunkModel,
            ExtractedFictionModel,
            ExtractedAcademicModel,
            ProjectModelConfigModel,
            ProviderCredentialModel,
        ]
    )


async def close_db():
    if db.client:
        await db.client.close()
