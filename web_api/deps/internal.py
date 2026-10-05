import hmac

from fastapi import Depends, Header

from web_api.core.config import get_settings
from web_api.errors import InvalidInternalToken

async def require_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    """Worker -> web_api calls must carry the shared INTERNAL_API_TOKEN."""
    expected = get_settings().INTERNAL_API_TOKEN.get_secret_value()
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise InvalidInternalToken()


InternalOnly = Depends(require_internal_token)
