from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from core.logging import get_logger

log = get_logger(__name__)


class VaultError(Exception):
    pass


class SecretNotFoundError(VaultError):
    pass


class VaultCorruptedError(VaultError):
    pass


class SecretVault:
    def __init__(
        self,
        master_key: str,
        vault_path: str = ".vault",
        salt: bytes | None = None,
    ) -> None:
        if len(master_key) < 32:
            raise VaultError(
                f"master_key must be at least 32 characters, got {len(master_key)}. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )

        self._vault_path = Path(vault_path)
        self._salt_path = Path(vault_path + ".salt")

        salt_bytes = salt or self._get_or_create_salt()
        self._fernet = self._derive_fernet(master_key, salt_bytes)

        self._cache: dict[str, str] | None = None

        log.info(
            "vault_initialized",
            vault_path=str(self._vault_path),
            exists=self._vault_path.exists(),
        )

    def _get_or_create_salt(self) -> bytes:
        if self._salt_path.exists():
            return self._salt_path.read_bytes()
        salt = os.urandom(16)
        self._salt_path.write_bytes(salt)
        return salt

    def _derive_fernet(self, master_key: str, salt: bytes) -> Fernet:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return Fernet(key)

    def _load(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache

        if not self._vault_path.exists():
            self._cache = {}
            return self._cache

        try:
            encrypted_data = self._vault_path.read_bytes()
            decrypted_json = self._fernet.decrypt(encrypted_data)
            self._cache = json.loads(decrypted_json)
            return self._cache
        except InvalidToken as e:
            raise VaultCorruptedError(
                f"Could not decrypt vault at {self._vault_path}. "
                "Either the VAULT_MASTER_KEY is wrong, or the file has been tampered with."
            ) from e
        except json.JSONDecodeError as e:
            raise VaultCorruptedError(
                f"Vault at {self._vault_path} decrypted but contains invalid JSON. "
                "The vault file may be corrupted."
            ) from e

    def _save(self, data: dict[str, str]) -> None:
        json_bytes = json.dumps(data, sort_keys=True).encode()
        encrypted = self._fernet.encrypt(json_bytes)

        # Atomic write: temp file → rename
        tmp_path = self._vault_path.with_suffix(".tmp")
        tmp_path.write_bytes(encrypted)
        tmp_path.rename(self._vault_path)

        # Update cache
        self._cache = data

    def store(self, key: str, value: str) -> None:
        if not key or not key.strip():
            raise VaultError("Secret key cannot be empty")

        value = value.strip()
        data = self._load()
        data[key] = value
        self._save(data)

        log.info("vault_secret_stored", key=key, value_length=len(value))

    def retrieve(self, key: str) -> str:
        data = self._load()
        if key not in data:
            raise SecretNotFoundError(
                f"Secret {key!r} not found in vault. "
                f"Available keys: {list(data.keys())}"
            )

        log.debug("vault_secret_retrieved", key=key)
        return data[key]

    def delete(self, key: str) -> bool:
        data = self._load()
        if key not in data:
            log.warning("vault_delete_key_not_found", key=key)
            return False

        del data[key]
        self._save(data)
        log.info("vault_secret_deleted", key=key)
        return True

    def list_keys(self) -> list[str]:
        return list(self._load().keys())

    def exists(self, key: str) -> bool:
        return key in self._load()

    def rotate_master_key(self, new_master_key: str) -> None:

        if len(new_master_key) < 32:
            raise VaultError("New master key must be at least 32 characters")

        data = self._load()

        new_salt = os.urandom(16)
        self._fernet = self._derive_fernet(new_master_key, new_salt)

        self._salt_path.write_bytes(new_salt)
        self._cache = None  # Force reload through new fernet
        self._save(data)

        log.warning("vault_master_key_rotated", secret_count=len(data))