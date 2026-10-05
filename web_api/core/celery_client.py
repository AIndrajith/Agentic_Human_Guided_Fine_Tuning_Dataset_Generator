from functools import lru_cache

from celery import Celery

from web_api.core.config import get_settings

PROCESS_DOCUMENTS_TASK = "workers.tasks.process_documents"


@lru_cache
def get_celery() -> Celery:
    """Producer-only client: web_api sends tasks by name, never imports worker code."""
    settings = get_settings()
    return Celery("web_api", broker=settings.CELERY_BROKER_URL, backend=settings.CELERY_RESULT_BACKEND)
