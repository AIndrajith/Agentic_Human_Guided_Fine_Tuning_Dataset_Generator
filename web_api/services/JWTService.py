from datetime import datetime, timedelta, timezone

from jose import JOSEError, jwt

from web_api.core.config import Settings
from web_api.errors import InvalidOrExpiredToken


class JWTService:
    def __init__(self, settings: Settings):
        self.secret_key = settings.JWT_SECRET_KEY.get_secret_value()
        if not self.secret_key:
            raise RuntimeError("JWT_SECRET_KEY is empty")
        self.algorithm = settings.JWT_ALGORITHM
        self.token_expiry_minutes = settings.JWT_EXPIRE_MINUTES

    def create_token(self, subject: str, claims: dict) -> str:
        now = datetime.now(timezone.utc)
        to_encode = {
            **claims,
            "sub": subject,
            "iat": now,
            "exp": now + timedelta(minutes=self.token_expiry_minutes),
        }
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except JOSEError as e:
            raise InvalidOrExpiredToken() from e
        if not payload.get("sub"):
            raise InvalidOrExpiredToken()
        return payload
