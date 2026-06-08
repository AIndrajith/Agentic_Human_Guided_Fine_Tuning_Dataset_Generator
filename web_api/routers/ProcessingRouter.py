from fastapi import APIRouter, HTTPException
from beanie import PydanticObjectId
from celery import Celery
import uuid
import os

from web_api.data_models.BasicBeanieModels import DocumentModel, ProjectModel
from web_api.data_models.ModelConfigModels import ProjectModelConfigModel
from web_api.data_models.enums import ModelStage, ModelProvider
from web_api.data_models.DataModels import ProcessDocumentsRequest
from web_api.services.credential_service import CredentialService

router = APIRouter(prefix="/processing", tags=["Processing"])

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_client = Celery("web_api", broker=CELERY_BROKER_URL)


async def _resolve_credentials(project_id: str) -> dict:
    """
    Fetch decrypted credentials for the stages that Phase 1 workers need.
    Falls back to empty dict if no model config is set (workers will use Config env vars).
    """
    obj_id = PydanticObjectId(project_id)
    config_doc = await ProjectModelConfigModel.find_one(
        ProjectModelConfigModel.project_id == obj_id
    )
    if not config_doc:
        return {}

    credentials = {}

    meta_config = config_doc.stages.get(ModelStage.META_AGENT.value)
    if meta_config:
        try:
            creds = await CredentialService.get_decrypted_fields(meta_config.provider)
            credentials["llm_provider"]  = meta_config.provider
            credentials["llm_model"]     = meta_config.model_name
            credentials["llm_api_key"]   = creds.get("api_key")
            credentials["llm_base_url"]  = meta_config.base_url or creds.get("base_url")
        except HTTPException:
            pass

    embed_config = config_doc.stages.get(ModelStage.EMBEDDER.value)
    if embed_config:
        try:
            creds = await CredentialService.get_decrypted_fields(embed_config.provider)
            credentials["embed_provider"] = embed_config.provider
            credentials["embed_model"]    = embed_config.model_name
            credentials["embed_api_key"]  = creds.get("api_key")
            credentials["embed_base_url"] = embed_config.base_url or creds.get("base_url")
        except HTTPException:
            pass

    try:
        google_creds = await CredentialService.get_decrypted_fields(ModelProvider.GOOGLE)
        credentials["vision_api_key"] = google_creds.get("api_key")
    except HTTPException:
        pass

    return credentials


@router.post("/start")
async def start_processing(request: ProcessDocumentsRequest):
    try:
        project_obj_id = PydanticObjectId(request.project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID format")

    project = await ProjectModel.get(project_obj_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if request.document_ids:
        documents = []
        for doc_id in request.document_ids:
            try:
                doc_obj_id = PydanticObjectId(doc_id)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid document ID: {doc_id}")

            doc = await DocumentModel.get(doc_obj_id)
            if not doc:
                raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

            if str(doc.project_id) != request.project_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {doc_id} does not belong to project {request.project_id}"
                )

            documents.append(doc)
    else:
        documents = await DocumentModel.find(
            DocumentModel.project_id == project_obj_id
        ).to_list()

    if not documents:
        raise HTTPException(status_code=400, detail="No documents to process")

    task_id = str(uuid.uuid4())
    credentials = await _resolve_credentials(request.project_id)

    documents_data = [{"id": str(doc.id), "file_size": 0} for doc in documents]

    task_payload = {
        "task_id":    task_id,
        "project_id": request.project_id,
        "documents":  documents_data,
        "data_type":  project.main_data_type.value,
        "credentials": credentials,
    }

    celery_task = celery_client.send_task(
        "workers.tasks.process_documents",
        args=[task_payload],
        task_id=task_id,
    )

    return {
        "message":         "Processing started",
        "task_id":         task_id,
        "celery_task_id":  celery_task.id,
        "project_id":      request.project_id,
        "documents_count": len(documents),
        "data_type":       project.main_data_type.value,
        "monitor_url":     f"http://localhost:5555/task/{task_id}",
    }


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    task_result = celery_client.AsyncResult(task_id)

    response = {
        "task_id":    task_id,
        "state":      task_result.state,
        "ready":      task_result.ready(),
        "successful": task_result.successful() if task_result.ready() else None,
        "failed":     task_result.failed() if task_result.ready() else None,
    }

    if task_result.ready():
        if task_result.successful():
            response["result"] = task_result.result
        elif task_result.failed():
            response["error"] = str(task_result.info)

    return response
