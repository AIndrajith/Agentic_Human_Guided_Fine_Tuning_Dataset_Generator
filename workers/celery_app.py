"""
Celery application configuration.
"""

from celery import Celery
from celery.signals import worker_ready
from workers.config import Config
from workers.utils.temp_file_manager import TempFileManager

# Create Celery app
celery_app = Celery(
    "synthetic_data_workers",
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND,
    include=[
        "workers.tasks.orchestrator",
        "workers.tasks.fiction_processor",
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # Task returns to queue if worker crashes
    worker_prefetch_multiplier=1,  # Process one task at a time
    task_time_limit=3600 * 6,  # 6 hours max per task
    task_soft_time_limit=3600 * 5,  # Soft limit: 5 hours
)


@worker_ready.connect
def _sweep_temp_files(**_):
    """Delete temp folders left by earlier failed runs (older than TEMP_FILE_RETENTION_HOURS)."""
    TempFileManager().cleanup_old_directories()

if __name__ == "__main__":
    celery_app.start()
