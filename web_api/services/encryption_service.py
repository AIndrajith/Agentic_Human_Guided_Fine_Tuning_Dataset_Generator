import os
from cryptography.fernet import Fernet


class EncryptionService:
    _instance = None

    def __init__(self):
        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            raise RuntimeError(
                "ENCRYPTION_KEY env var is not set. "
                "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        self._fernet = Fernet(key.encode() if isinstance(key, str) else key)

    @classmethod
    def get(cls) -> "EncryptionService":
        if cls._instance is None:
            cls._instance = EncryptionService()
        return cls._instance

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, encrypted: str) -> str:
        return self._fernet.decrypt(encrypted.encode()).decode()
