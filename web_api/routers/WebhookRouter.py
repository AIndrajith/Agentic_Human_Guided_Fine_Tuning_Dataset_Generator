"""Workers report per-document results here. Requires the X-Internal-Token header."""
from fastapi import APIRouter

from web_api.data_models.enums import DocumentStatus
from web_api.data_models.ProcessingModels import ProcessingCompletionWebhook, ProcessingFailedWebhook, WebhookAck
from web_api.deps.auth import SessionDep
from web_api.deps.internal import InternalOnly
from web_api.services.ProcessingService import ProcessingService

router = APIRouter(prefix="/webhooks", tags=["Webhooks"], dependencies=[InternalOnly])


@router.post("/processing-complete", response_model=WebhookAck)
async def processing_complete(payload: ProcessingCompletionWebhook, session: SessionDep):
    """Document finished: store its chunk -> Qdrant point links and update statuses."""
    succeeded = payload.status != "failed"
    created = await ProcessingService(session).record_completion(
        payload.task_id, payload.document_id, succeeded, payload.chunks_data, payload.error_message
    )
    return WebhookAck(
        message="Processing completion recorded",
        document_id=payload.document_id,
        status=DocumentStatus.COMPLETED if succeeded else DocumentStatus.FAILED,
        chunks_created=created,
    )


@router.post("/processing-failed", response_model=WebhookAck)
async def processing_failed(payload: ProcessingFailedWebhook, session: SessionDep):
    await ProcessingService(session).record_failure(
        payload.task_id, payload.document_id, payload.error_message, payload.stage
    )
    return WebhookAck(message="Processing failure recorded", document_id=payload.document_id,
                      status=DocumentStatus.FAILED)
