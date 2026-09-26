# Import every model here so Base.metadata (and Alembic autogenerate) sees them.
from web_api.db.models.users import User, EmailEvent
from web_api.db.models.projects import Project, ProjectMember
from web_api.db.models.documents import Document
from web_api.db.models.credentials import ProviderCredential, ProjectStageModel
from web_api.db.models.processing import (
    ProcessingJob,
    JobDocument,
    Extraction,
    ExtractedImage,
    Chunk,
)

__all__ = [
    "User", "EmailEvent",
    "Project", "ProjectMember",
    "Document",
    "ProviderCredential", "ProjectStageModel",
    "ProcessingJob", "JobDocument", "Extraction", "ExtractedImage", "Chunk",
]
