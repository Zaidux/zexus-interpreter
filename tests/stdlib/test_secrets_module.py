"""Tests for stdlib SecretsModule."""

import os
import pytest
from src.zexus.stdlib.secrets_module import SecretsModule


class TestSecretStore:
    def test_create_store(self):
        store = SecretsModule.create_store()
        assert store is not None

    def test_put_and_get(self):
        store = SecretsModule.create_store()
        SecretsModule.put(store, "db_password", "super_secret_123")
        assert SecretsModule.get(store, "db_password") == "super_secret_123"

    def test_get_missing(self):
        store = SecretsModule.create_store()
        with pytest.raises(KeyError):
            SecretsModule.get(store, "nonexistent")

    def test_delete(self):
        store = SecretsModule.create_store()
        SecretsModule.put(store, "key", "value")
        SecretsModule.delete(store, "key")
        with pytest.raises(KeyError):
            SecretsModule.get(store, "key")

    def test_list_secrets(self):
        store = SecretsModule.create_store()
        SecretsModule.put(store, "a", "1")
        SecretsModule.put(store, "b", "2")
        names = SecretsModule.list_secrets(store)
        assert "a" in names
        assert "b" in names

    def test_rotate(self):
        store = SecretsModule.create_store()
        SecretsModule.put(store, "key", "old_value")
        SecretsModule.rotate(store, "key", "new_value")
        assert SecretsModule.get(store, "key") == "new_value"

    def test_version_history(self):
        store = SecretsModule.create_store()
        SecretsModule.put(store, "key", "v1")
        SecretsModule.rotate(store, "key", "v2")
        SecretsModule.rotate(store, "key", "v3")
        # Latest should be v3
        assert SecretsModule.get(store, "key") == "v3"


class TestEnvelopeEncryption:
    def test_seal_unseal(self):
        data = "sensitive data here"
        key = "master_key_123"
        sealed = SecretsModule.seal(data, key)
        assert sealed is not None
        unsealed = SecretsModule.unseal(sealed, key)
        assert unsealed == data

    def test_seal_wrong_key(self):
        data = "sensitive"
        sealed = SecretsModule.seal(data, "correct_key")
        # With wrong key, should fail or return wrong data
        try:
            result = SecretsModule.unseal(sealed, "wrong_key")
            assert result != data  # If it doesn't raise, data should be wrong
        except Exception:
            pass  # Exception is acceptable


class TestTokenGeneration:
    def test_generate_token(self):
        token = SecretsModule.generate_token(32)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_generate_token_unique(self):
        t1 = SecretsModule.generate_token()
        t2 = SecretsModule.generate_token()
        assert t1 != t2


class TestFromEnv:
    def test_from_env_exists(self):
        os.environ["TEST_SECRET_XYZ"] = "my_secret"
        try:
            result = SecretsModule.from_env("TEST_SECRET_XYZ")
            # Returns a dict with metadata
            assert result["value"] == "my_secret"
            assert result["tainted"] is True
        finally:
            del os.environ["TEST_SECRET_XYZ"]

    def test_from_env_missing_required(self):
        with pytest.raises(Exception):
            SecretsModule.from_env("NONEXISTENT_SECRET_VAR_12345", required=True)

    def test_from_env_missing_optional(self):
        result = SecretsModule.from_env("NONEXISTENT_SECRET_VAR_12345", required=False)
        assert result["value"] is None


class TestAuditLog:
    def test_audit_log(self):
        store = SecretsModule.create_store()
        SecretsModule.put(store, "x", "y")
        SecretsModule.get(store, "x")
        log = SecretsModule.audit_log(store)
        assert isinstance(log, list)
        assert len(log) >= 2  # put + get
