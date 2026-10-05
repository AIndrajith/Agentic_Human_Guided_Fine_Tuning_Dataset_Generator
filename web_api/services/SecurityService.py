import hashlib
import re
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from web_api.errors import ValidationError


class SecurityService:
    def __init__(self):
        self._hasher = PasswordHasher()

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generates a secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_token(token: str) -> str:
        """Setup tokens are stored as sha256 so a DB leak doesn't expose usable links."""
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def validate_password(password: str) -> None:
        error_messages = []

        if len(password) < 8:
            error_messages.append("Password must be at least 8 characters long")
        if not re.search(r"[A-Z]", password):
            error_messages.append("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", password):
            error_messages.append("Password must contain at least one lowercase letter")
        if not re.search(r"\d", password):
            error_messages.append("Password must contain at least one number")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]", password):
            error_messages.append("Password must contain at least one special character")

        if error_messages:
            raise ValidationError(" ".join(error_messages))

    def hash_password(self, password: str) -> str:
        return self._hasher.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str | None) -> bool:
        if not hashed_password:
            return False
        try:
            return self._hasher.verify(hashed_password, plain_password)
        except (VerificationError, InvalidHashError):
            return False
