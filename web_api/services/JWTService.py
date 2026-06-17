from jose import JOSEError, jwt
from datetime import datetime, timedelta, timezone
import os

class JWTService:
    def __init__(self, algorithm: str = "HS256", token_expiry_minutes: int = 20):
        try :
            self.secret_key = os.environ.get("JWT_SECRET_KEY")
        except KeyError:
           raise RuntimeError(f"Environment variable 'JWT_SECRET_KEY' is not set")
        
        self.algorithm = algorithm
        self.token_expiry_minutes = token_expiry_minutes

    def create_token(self, data: dict) -> str:
        to_encode = data.copy()
        to_encode.update({"exp": datetime.now(timezone.utc) + timedelta(minutes=self.token_expiry_minutes)})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JOSEError as e:
            raise RuntimeError("Invalid token") from e
