"""Secrets management module for Zexus standard library."""

import secrets
import hashlib
import os
import base64
import hmac
import time
import json


class SecretsModule:
    """Provides secrets management with encryption, versioning, and audit logging."""

    @staticmethod
    def create_store(backend="memory"):
        """Create a new secret store.

        Args:
            backend: Storage backend type. Currently only 'memory' is supported.

        Returns:
            A secret store dict with a runtime encryption key.
        """
        runtime_key = secrets.token_bytes(32)
        return {
            "backend": backend,
            "secrets": {},
            "versions": {},
            "audit": [],
            "_runtime_key": runtime_key,
        }

    @staticmethod
    def put(store, name, value, metadata=None):
        """Store a secret, encrypted in memory with the store's runtime key.

        Args:
            store: The secret store.
            name: Name/key for the secret.
            value: The secret value (string).
            metadata: Optional metadata dict to attach.
        """
        encrypted = SecretsModule._encrypt(value, store["_runtime_key"])

        entry = {
            "encrypted_value": encrypted,
            "metadata": metadata or {},
            "created_at": time.time(),
            "updated_at": time.time(),
            "version": 1,
        }

        if name in store["secrets"]:
            old = store["secrets"][name]
            entry["version"] = old["version"] + 1

        store["secrets"][name] = entry

        if name not in store["versions"]:
            store["versions"][name] = []
        store["versions"][name].append({
            "version": entry["version"],
            "encrypted_value": encrypted,
            "timestamp": entry["updated_at"],
        })

        store["audit"].append({
            "action": "put",
            "name": name,
            "version": entry["version"],
            "timestamp": time.time(),
        })

    @staticmethod
    def get(store, name):
        """Retrieve and decrypt a secret value.

        Args:
            store: The secret store.
            name: Name/key of the secret.

        Returns:
            The decrypted secret value string.

        Raises:
            KeyError: If the secret does not exist.
        """
        if name not in store["secrets"]:
            raise KeyError(f"Secret '{name}' not found")

        entry = store["secrets"][name]

        store["audit"].append({
            "action": "get",
            "name": name,
            "version": entry["version"],
            "timestamp": time.time(),
        })

        return SecretsModule._decrypt(entry["encrypted_value"], store["_runtime_key"])

    @staticmethod
    def delete(store, name):
        """Delete a secret from the store.

        Args:
            store: The secret store.
            name: Name/key of the secret to delete.

        Raises:
            KeyError: If the secret does not exist.
        """
        if name not in store["secrets"]:
            raise KeyError(f"Secret '{name}' not found")

        del store["secrets"][name]

        store["audit"].append({
            "action": "delete",
            "name": name,
            "timestamp": time.time(),
        })

    @staticmethod
    def list_secrets(store):
        """List all secret names in the store (values are not returned).

        Args:
            store: The secret store.

        Returns:
            A list of secret name strings.
        """
        store["audit"].append({
            "action": "list",
            "timestamp": time.time(),
        })
        return list(store["secrets"].keys())

    @staticmethod
    def rotate(store, name, new_value):
        """Rotate a secret to a new value, preserving version history.

        Args:
            store: The secret store.
            name: Name/key of the secret to rotate.
            new_value: The new secret value.

        Raises:
            KeyError: If the secret does not exist.
        """
        if name not in store["secrets"]:
            raise KeyError(f"Secret '{name}' not found")

        old_version = store["secrets"][name]["version"]

        store["audit"].append({
            "action": "rotate",
            "name": name,
            "old_version": old_version,
            "new_version": old_version + 1,
            "timestamp": time.time(),
        })

        SecretsModule.put(store, name, new_value, store["secrets"][name].get("metadata"))

    @staticmethod
    def get_version(store, name, version=None):
        """Get a specific version of a secret.

        Args:
            store: The secret store.
            name: Name/key of the secret.
            version: Version number to retrieve. If None, returns the latest.

        Returns:
            The decrypted secret value for the requested version.

        Raises:
            KeyError: If the secret or version does not exist.
        """
        if name not in store["versions"]:
            raise KeyError(f"Secret '{name}' has no version history")

        history = store["versions"][name]

        if version is None:
            entry = history[-1]
        else:
            entry = None
            for v in history:
                if v["version"] == version:
                    entry = v
                    break
            if entry is None:
                raise KeyError(f"Version {version} not found for secret '{name}'")

        store["audit"].append({
            "action": "get_version",
            "name": name,
            "version": entry["version"],
            "timestamp": time.time(),
        })

        return SecretsModule._decrypt(entry["encrypted_value"], store["_runtime_key"])

    @staticmethod
    def seal(data, key):
        """Envelope encryption: encrypt data with a random DEK, then encrypt DEK with the provided key.

        Args:
            data: The plaintext string to seal.
            key: The key-encryption key (string).

        Returns:
            A dict containing the encrypted data, encrypted DEK, and IV.
        """
        dek = secrets.token_bytes(32)

        encrypted_data = SecretsModule._encrypt(data, dek)

        kek = hashlib.pbkdf2_hmac("sha256", key.encode(), b"zexus-seal", 100000)
        encrypted_dek = SecretsModule._xor_crypt(dek, kek)

        return {
            "encrypted_data": encrypted_data,
            "encrypted_dek": base64.b64encode(encrypted_dek).decode(),
            "algorithm": "xor-pbkdf2",
        }

    @staticmethod
    def unseal(sealed_data, key):
        """Reverse envelope encryption to recover plaintext.

        Args:
            sealed_data: The sealed dict from seal().
            key: The key-encryption key (string) used during sealing.

        Returns:
            The decrypted plaintext string.
        """
        kek = hashlib.pbkdf2_hmac("sha256", key.encode(), b"zexus-seal", 100000)
        encrypted_dek = base64.b64decode(sealed_data["encrypted_dek"])
        dek = SecretsModule._xor_crypt(encrypted_dek, kek)

        return SecretsModule._decrypt(sealed_data["encrypted_data"], dek)

    @staticmethod
    def generate_token(length=32):
        """Generate a cryptographically secure random token.

        Args:
            length: Number of random bytes (token will be hex-encoded, so twice this length).

        Returns:
            A hex-encoded random token string.
        """
        return secrets.token_hex(length)

    @staticmethod
    def from_env(name, required=True):
        """Load a secret from an environment variable.

        Args:
            name: The environment variable name.
            required: If True, raises an error when the variable is not set.

        Returns:
            A dict with the value and a tainted flag.

        Raises:
            EnvironmentError: If required is True and the variable is not set.
        """
        value = os.environ.get(name)

        if value is None and required:
            raise EnvironmentError(
                f"Required environment variable '{name}' is not set"
            )

        return {
            "value": value,
            "source": "env",
            "name": name,
            "tainted": True,
            "loaded_at": time.time(),
        }

    @staticmethod
    def audit_log(store):
        """Get the full audit log of all secret access operations.

        Args:
            store: The secret store.

        Returns:
            A list of audit log entry dicts.
        """
        return list(store["audit"])

    # --- Internal helpers ---

    @staticmethod
    def _derive_key(key_material, salt=None, length=32):
        """Derive a fixed-length key from arbitrary key material using PBKDF2."""
        if salt is None:
            salt = b"zexus-secrets-default-salt"
        if isinstance(key_material, str):
            key_material = key_material.encode()
        return hashlib.pbkdf2_hmac("sha256", key_material, salt, 100000, dklen=length)

    @staticmethod
    def _xor_crypt(data, key):
        """XOR-based encryption/decryption (symmetric). Not production-grade."""
        if isinstance(data, str):
            data = data.encode()
        if isinstance(key, str):
            key = key.encode()
        extended_key = (key * (len(data) // len(key) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, extended_key))

    @staticmethod
    def _encrypt(plaintext, key_material):
        """Encrypt a plaintext string using XOR with a PBKDF2-derived key.

        Returns a dict with salt, ciphertext, and HMAC for integrity.
        """
        salt = secrets.token_bytes(16)
        derived = SecretsModule._derive_key(key_material, salt=salt)
        plaintext_bytes = plaintext.encode() if isinstance(plaintext, str) else plaintext
        ciphertext = SecretsModule._xor_crypt(plaintext_bytes, derived)
        mac = hmac.new(derived, ciphertext, hashlib.sha256).hexdigest()

        return {
            "salt": base64.b64encode(salt).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "mac": mac,
        }

    @staticmethod
    def _decrypt(encrypted, key_material):
        """Decrypt ciphertext produced by _encrypt, verifying HMAC integrity."""
        salt = base64.b64decode(encrypted["salt"])
        ciphertext = base64.b64decode(encrypted["ciphertext"])
        derived = SecretsModule._derive_key(key_material, salt=salt)

        expected_mac = hmac.new(derived, ciphertext, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_mac, encrypted["mac"]):
            raise ValueError("Integrity check failed: secret may have been tampered with")

        plaintext_bytes = SecretsModule._xor_crypt(ciphertext, derived)
        return plaintext_bytes.decode()
