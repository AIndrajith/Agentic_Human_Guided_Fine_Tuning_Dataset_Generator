"""Fetch a project's models, keys, vector size and Qdrant collection from web_api."""
import logging

import httpx

from workers.config import Config
from workers.models import ProcessingConfig

logger = logging.getLogger(__name__)


class ProcessingConfigError(Exception):
    """web_api refused the config (e.g. a model step isn't configured). Retrying won't help."""


async def fetch_processing_config(project_id: str) -> ProcessingConfig:
    url = f"{Config.WEB_API_BASE_URL}/internal/projects/{project_id}/processing-config"
    async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=30.0) as client:
        response = await client.get(url)
    if 400 <= response.status_code < 500:
        try:
            message = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = response.text[:300]
        raise ProcessingConfigError(message)
    response.raise_for_status()
    config = ProcessingConfig(**response.json())
    logger.info(
        f"Loaded processing config for project {project_id}: llm={config.llm.model}, "
        f"embedder={config.embedder.model} ({config.embedder.dimension}d), collection={config.qdrant_collection}"
    )
    return config
