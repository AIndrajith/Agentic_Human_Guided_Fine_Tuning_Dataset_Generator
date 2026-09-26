
import asyncio
from celery import Task
from workers.celery_app import celery_app
from workers.models import TaskData
from workers.services.processing_config import ProcessingConfigError, fetch_processing_config
from workers.tasks.fiction_processor import FictionProcessor
from workers.tasks.academic_processor import AcademicProcessor
from workers.utils.retry import is_transient
from workers.utils.webhook_notifier import WebhookNotifier
from workers.enums import DataCategory, ProcessingStage
import logging

logger = logging.getLogger(__name__)

# One event loop for the whole worker process. asyncio.run() would close the loop after every task,
# and LiteLLM/httpx keep connection pools that belong to the loop they were created on.
_loop: asyncio.AbstractEventLoop | None = None


def _run(coro):
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop.run_until_complete(coro)


class _ConfigUnavailable(Exception):
    """web_api couldn't be reached for the processing config. Nothing was processed yet, so the
    whole task is safe to retry later."""


@celery_app.task(
    bind=True,
    name="workers.tasks.process_documents",
    max_retries=3,
)
def process_documents(self: Task, task_data_dict: dict):
    """
    Args:
        task_data_dict: {"task_id", "project_id", "documents": [{"id", "file_size"}], "data_type"}

    Each document succeeds or fails on its own (reported to web_api by webhook); a failed document
    never makes the others run again. Temporary errors are retried at the step that hit them
    (see workers.utils.retry). Only "web_api unreachable before anything started" retries the task.
    """
    task_data = TaskData(**task_data_dict)

    logger.info(
        f"Processing task {task_data.task_id}: "
        f"{len(task_data.documents)} documents, "
        f"type: {task_data.data_type}"
    )

    try:
        return _run(_process_documents_async(task_data))
    except _ConfigUnavailable as e:
        logger.warning(f"Task {task_data.task_id}: {e}; retrying the task later")
        raise self.retry(exc=e, countdown=60 * (self.request.retries + 1))


async def _fail_all(task_data: TaskData, error: str, stage: ProcessingStage) -> dict:
    notifier = WebhookNotifier()
    for doc in task_data.documents:
        await notifier.notify_processing_failed(
            task_id=task_data.task_id, document_id=doc.id, error_message=error, stage=stage.value,
        )
    return {"task_id": task_data.task_id, "total_documents": len(task_data.documents),
            "successful": 0, "failed": len(task_data.documents), "error": error}


async def _process_documents_async(task_data: TaskData):

    results = []

    # Choose processor based on data type
    if task_data.data_type == DataCategory.FICTION.value:
        processor = FictionProcessor()
    elif task_data.data_type == DataCategory.ACADEMIC.value:
        processor = AcademicProcessor()
    else:
        logger.error(f"Task {task_data.task_id}: unknown data type {task_data.data_type!r}")
        return await _fail_all(task_data, f"Unknown data type: {task_data.data_type}", ProcessingStage.LOADING_CONFIG)

    # Models + keys come from web_api, once per task (they never travel through Redis)
    try:
        config = await fetch_processing_config(task_data.project_id)
    except ProcessingConfigError as e:
        # e.g. a model step isn't configured: retrying won't help, so fail every document now
        logger.error(f"Task {task_data.task_id}: cannot load processing config: {e}")
        return await _fail_all(task_data, str(e), ProcessingStage.LOADING_CONFIG)
    except Exception as e:
        if is_transient(e):
            raise _ConfigUnavailable(f"web_api unreachable for the processing config ({e})") from e
        raise

    # Process each document; one failure doesn't stop (or repeat) the others
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
            # the processor already told web_api (processing-failed webhook) at which stage it broke
            logger.error(f"Failed to process document {doc.id}: {type(e).__name__}: {e}")

            results.append({
                "document_id": doc.id,
                "status": "failed",
                "error": str(e)
            })

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
