
import asyncio
from celery import Task
from workers.celery_app import celery_app
from workers.models import TaskData
from workers.services.processing_config import ProcessingConfigError, fetch_processing_config
from workers.tasks.fiction_processor import FictionProcessor
from workers.tasks.academic_processor import AcademicProcessor
from workers.utils.webhook_notifier import WebhookNotifier
from workers.enums import DataCategory, ProcessingStage
import logging

logger = logging.getLogger(__name__)


class ProcessDocumentsTask(Task):
    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 600  # 10 minutes max backoff
    retry_jitter = True


@celery_app.task(
    bind=True,
    base=ProcessDocumentsTask,
    name="workers.tasks.process_documents"
)
def process_documents(self, task_data_dict: dict):
    """
    Args:
        task_data_dict: Dictionary containing task data
            {
                "task_id": "uuid",
                "project_id": "uuid",
                "documents": [{"id": "doc1", "file_size": 1024}],
                "data_type": "fiction" or "academic"
            }


    """
    task_data = TaskData(**task_data_dict)

    logger.info(
        f"Processing task {task_data.task_id}: "
        f"{len(task_data.documents)} documents, "
        f"type: {task_data.data_type}"
    )


    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _process_documents_async(self, task_data)
    )

    return result


async def _process_documents_async(task: Task, task_data: TaskData):

    results = []

    # Choose processor based on data type
    if task_data.data_type == DataCategory.FICTION.value:
        processor = FictionProcessor()
    elif task_data.data_type == DataCategory.ACADEMIC.value:
        processor = AcademicProcessor()
    else:
        raise ValueError(f"Unknown data type: {task_data.data_type}")

    # Models + keys come from web_api, once per task (they never travel through Redis)
    try:
        config = await fetch_processing_config(task_data.project_id)
    except ProcessingConfigError as e:
        # e.g. a model step isn't configured: retrying won't help, so fail every document now
        logger.error(f"Task {task_data.task_id}: cannot load processing config: {e}")
        notifier = WebhookNotifier()
        for doc in task_data.documents:
            await notifier.notify_processing_failed(
                task_id=task_data.task_id, document_id=doc.id,
                error_message=str(e), stage=ProcessingStage.LOADING_CONFIG.value,
            )
        return {"task_id": task_data.task_id, "total_documents": len(task_data.documents),
                "successful": 0, "failed": len(task_data.documents), "error": str(e)}

    # Process each document
    for doc in task_data.documents:
        try:
            logger.info(f"Processing document: {doc.id}")

            result = await processor.process_document(
                task_id=task_data.task_id,
                document=doc,
                project_id=task_data.project_id,
                config=config,
            )

            results.append({
                "document_id": doc.id,
                "status": "success",
                "result": result
            })

        except Exception as e:
            logger.error(f"Failed to process document {doc.id}: {str(e)}")

            results.append({
                "document_id": doc.id,
                "status": "failed",
                "error": str(e)
            })

            # If this is the last retry, don't raise
            if task.request.retries >= task.max_retries:
                logger.error(f"Max retries reached for document {doc.id}")
            else:
                # Raise to trigger retry
                raise

    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful

    logger.info(
        f"Task {task_data.task_id} completed: "
        f"{successful} successful, {failed} failed"
    )

    return {
        "task_id": task_data.task_id,
        "total_documents": len(task_data.documents),
        "successful": successful,
        "failed": failed,
        "results": results
    }
