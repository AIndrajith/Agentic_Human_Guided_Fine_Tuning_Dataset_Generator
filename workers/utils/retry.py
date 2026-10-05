"""Retry only errors that can succeed on a second try: network problems, rate limits, server errors.

A wrong API key, an unknown model or a broken PDF fail at once instead of wasting minutes on retries.
"""
import asyncio
import logging
import random
from typing import Awaitable, Callable, TypeVar

import httpx
import litellm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from workers.config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T")

_TRANSIENT_LITELLM = (
    litellm.RateLimitError,
    litellm.APIConnectionError,   # includes litellm.Timeout
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.BadGatewayError,
)
_PERMANENT_LITELLM = (
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    litellm.NotFoundError,
    litellm.BadRequestError,      # includes context-window and content-policy errors
    litellm.UnprocessableEntityError,
)


def _retryable_status(status: int | None) -> bool:
    return status is not None and (status == 429 or status >= 500)


def is_transient(error: BaseException) -> bool:
    """True for errors worth retrying."""
    if isinstance(error, httpx.HTTPStatusError):
        return _retryable_status(error.response.status_code)
    if isinstance(error, httpx.TransportError):          # connect/read timeouts, connection refused
        return True
    if isinstance(error, _TRANSIENT_LITELLM):
        return True
    if isinstance(error, _PERMANENT_LITELLM):
        return False
    if isinstance(error, litellm.APIError):              # other provider errors: decide by HTTP status
        return _retryable_status(getattr(error, "status_code", None))
    if isinstance(error, UnexpectedResponse):            # Qdrant answered with an error
        return _retryable_status(error.status_code)
    if isinstance(error, ResponseHandlingException):     # Qdrant unreachable
        return True
    return False


def is_permanent_api_error(error: BaseException) -> bool:
    """A provider said no in a way retrying can't fix (bad key, unknown model, bad request)."""
    return isinstance(error, _PERMANENT_LITELLM) or (
        isinstance(error, httpx.HTTPStatusError) and not _retryable_status(error.response.status_code)
    )


async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args,
    what: str,
    attempts: int = Config.MAX_RETRIES,
    retry_if: Callable[[BaseException], bool] = is_transient,
    **kwargs,
) -> T:
    """Call `await fn(*args, **kwargs)`; on a retryable error wait 2s, 4s, ... (+ jitter) and try again."""
    for attempt in range(1, attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            if attempt == attempts or not retry_if(e):
                raise
            delay = Config.RETRY_BACKOFF_BASE ** attempt + random.uniform(0, 1)
            logger.warning(f"{what} failed ({type(e).__name__}: {e}); retry {attempt}/{attempts - 1} in {delay:.1f}s")
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")
