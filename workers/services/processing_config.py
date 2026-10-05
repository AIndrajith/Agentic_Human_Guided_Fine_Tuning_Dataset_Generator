"""Fetch a project's models, keys, vector size and Qdrant collection from web_api."""
import logging

import httpx

from workers.config import Config
from workers.models import ProcessingConfig
from workers.utils.retry import retry_async

logger = logging.getLogger(__name__)


class ProcessingConfigError(Exception):
    """web_api refused the config (e.g. a model step isn't configured). Retrying won't help."""


async def _get(url: str) -> httpx.Response:
    async with httpx.AsyncClient(headers=Config.internal_headers(), timeout=30.0) as client:
        response = await client.get(url)
    if response.status_code >= 500 or response.status_code == 429:
        response.raise_for_status()   # HTTPStatusError -> retried; other 4xx are answers, handled below
    return response


async def fetch_processing_config(project_id: str) -> ProcessingConfig:
    url = f"{Config.WEB_API_BASE_URL}/internal/projects/{project_id}/processing-config"
    response = await retry_async(_get, url, what="fetch processing config")
    if 400 <= response.status_code < 500:
        try:
            message = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = response.text[:300]
        raise ProcessingConfigError(message)
    config = ProcessingConfig(**response.json())
    logger.info(
        f"Loaded processing config for project {project_id}: llm={config.llm.model}, "
        f"embedder={config.embedder.model} ({config.embedder.dimension}d), collection={config.qdrant_collection}"
    )
    return config
