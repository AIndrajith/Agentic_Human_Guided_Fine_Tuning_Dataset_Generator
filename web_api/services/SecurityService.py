import secrets
import re
from passlib.context import CryptContext

class SecurityService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

    @staticmethod
    def generate_secure_token( length: int = 32) -> str:
        """Generates a secure random token."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def validate_password( password: str) -> None:
        error_messages = []

        if len(password) < 8:
            error_messages.append("Password must be at least 8 characters long")

        if not re.search(r"[A-Z]", password):
            error_messages.append(
                "Password must contain at least one uppercase letter"
            )

        if not re.search(r"[a-z]", password):
            error_messages.append(
                "Password must contain at least one lowercase letter"
            )

        if not re.search(r"\d", password):
            error_messages.append(
                "Password must contain at least one number"
            )

        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]", password):
            error_messages.append(
                "Password must contain at least one special character"
            )
        
        if error_messages:
            raise ValueError(" ".join(error_messages))
        
    def hash_password(self,password: str) -> str:
        try:
            return self.pwd_context.hash(password)
        except ValueError:
            raise RuntimeError("Password hashing failed")
        
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False