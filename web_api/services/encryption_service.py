from functools import lru_cache

from cryptography.fernet import Fernet

from web_api.core.config import get_settings


class EncryptionService:
    def __init__(self, key: str):
        if not key:
            raise RuntimeError(
                "ENCRYPTION_KEY is not set. "
                "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        self._fernet = Fernet(key.encode())

    @classmethod
    def get(cls) -> "EncryptionService":
        return _instance()

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, encrypted: str) -> str:
        return self._fernet.decrypt(encrypted.encode()).decode()


@lru_cache
def _instance() -> EncryptionService:
    return EncryptionService(get_settings().ENCRYPTION_KEY.get_secret_value())
