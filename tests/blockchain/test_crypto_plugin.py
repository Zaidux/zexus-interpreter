import pytest

from src.zexus.blockchain.crypto import CryptoPlugin, CRYPTO_AVAILABLE


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography dependency is required")
class TestCryptoAddressPrefix:
    def test_derive_address_default_prefix(self):
        priv, pub = CryptoPlugin.generate_keypair("ECDSA")
        address = CryptoPlugin.derive_address(pub)
        assert address.startswith("0x")
        assert len(address) == 42
        assert all(c in "0123456789abcdef" for c in address[2:])

    def test_derive_address_custom_prefix_per_call(self):
        _priv, pub = CryptoPlugin.generate_keypair("ECDSA")
        address = CryptoPlugin.derive_address(pub, prefix="Zx01")
        assert address.startswith("Zx01")
        assert len(address) == len("Zx01") + 40
        assert all(c in "0123456789abcdef" for c in address[len("Zx01"):])

    def test_derive_address_custom_default_prefix(self):
        _priv, pub = CryptoPlugin.generate_keypair("ECDSA")
        prev_prefix = CryptoPlugin.get_address_prefix()
        try:
            CryptoPlugin.set_address_prefix("OK")
            address = CryptoPlugin.derive_address(pub)
            assert address.startswith("OK")
            assert len(address) == len("OK") + 40
        finally:
            CryptoPlugin.set_address_prefix(prev_prefix)

    def test_derive_address_rejects_empty_prefix(self):
        _priv, pub = CryptoPlugin.generate_keypair("ECDSA")
        with pytest.raises(ValueError):
            CryptoPlugin.derive_address(pub, prefix="")

    def test_set_address_prefix_rejects_empty(self):
        with pytest.raises(ValueError):
            CryptoPlugin.set_address_prefix("")

