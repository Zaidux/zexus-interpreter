"""Tests for blockchain mempool, gas oracle, and ABI encoder."""

import pytest
from src.zexus.blockchain.mempool import PriorityMempool, GasPriceOracle, ABIEncoder


class TestPriorityMempool:
    def _make_tx(self, sender="0xAAA", nonce=0, gas_price=100, gas_limit=21000, tx_hash=None):
        """Create a mock transaction-like object."""
        class MockTx:
            def __init__(self, s, n, gp, gl, h):
                self.sender = s
                self.nonce = n
                self.gas_price = gp
                self.gas_limit = gl
                self.tx_hash = h or f"0x{s}{n}{gp}"
                self.hash = self.tx_hash

            def to_dict(self):
                return {
                    "sender": self.sender,
                    "nonce": self.nonce,
                    "gas_price": self.gas_price,
                    "gas_limit": self.gas_limit,
                    "hash": self.hash,
                    "tx_hash": self.tx_hash,
                }

        return MockTx(sender, nonce, gas_price, gas_limit, tx_hash)

    def test_create_mempool(self):
        mp = PriorityMempool()
        assert mp.size == 0

    def test_add_transaction(self):
        mp = PriorityMempool()
        tx = self._make_tx()
        mp.add(tx)
        assert mp.size == 1

    def test_get_pending_by_gas_price(self):
        mp = PriorityMempool()
        mp.add(self._make_tx(sender="0xA", nonce=0, gas_price=100, tx_hash="h1"))
        mp.add(self._make_tx(sender="0xB", nonce=0, gas_price=200, tx_hash="h2"))
        mp.add(self._make_tx(sender="0xC", nonce=0, gas_price=50, tx_hash="h3"))
        
        pending = mp.get_pending(gas_limit=100000)
        assert len(pending) == 3
        # Should be sorted by gas price descending
        gas_prices = [tx.gas_price for tx in pending]
        assert gas_prices == sorted(gas_prices, reverse=True)

    def test_remove(self):
        mp = PriorityMempool()
        tx = self._make_tx(tx_hash="removeme")
        mp.add(tx)
        mp.remove(tx.tx_hash)
        assert mp.size == 0

    def test_clear(self):
        mp = PriorityMempool()
        mp.add(self._make_tx(tx_hash="a"))
        mp.add(self._make_tx(sender="0xB", tx_hash="b"))
        mp.clear()
        assert mp.size == 0

    def test_evict_below(self):
        mp = PriorityMempool()
        mp.add(self._make_tx(sender="0xA", gas_price=100, tx_hash="h1"))
        mp.add(self._make_tx(sender="0xB", gas_price=50, tx_hash="h2"))
        mp.add(self._make_tx(sender="0xC", gas_price=200, tx_hash="h3"))
        mp.evict_below(min_gas_price=100)
        assert mp.size == 2  # 50 gas price tx evicted

    def test_stats(self):
        mp = PriorityMempool()
        mp.add(self._make_tx(sender="0xA", gas_price=100, tx_hash="h1"))
        mp.add(self._make_tx(sender="0xB", gas_price=200, tx_hash="h2"))
        stats = mp.stats()
        assert stats["size"] == 2
        assert stats["min_gas_price"] == 100
        assert stats["max_gas_price"] == 200

    def test_is_full(self):
        mp = PriorityMempool(max_size=2)
        mp.add(self._make_tx(sender="0xA", tx_hash="h1"))
        mp.add(self._make_tx(sender="0xB", tx_hash="h2"))
        assert mp.is_full is True


class TestGasPriceOracle:
    def test_create(self):
        oracle = GasPriceOracle()
        assert oracle.base_fee > 0

    def test_suggest_gas_price(self):
        oracle = GasPriceOracle(base_fee=1000)
        suggestion = oracle.suggest_gas_price()
        assert "base_fee" in suggestion
        assert "priority_fee" in suggestion
        assert "max_fee" in suggestion
        assert suggestion["base_fee"] == 1000

    def test_update_increases_fee(self):
        oracle = GasPriceOracle(base_fee=1000, target_gas_used=5_000_000)
        initial = oracle.base_fee
        # Simulate full block (above target)
        oracle.update(9_000_000)
        assert oracle.base_fee > initial

    def test_update_decreases_fee(self):
        oracle = GasPriceOracle(base_fee=1000, target_gas_used=5_000_000)
        initial = oracle.base_fee
        # Simulate empty block (below target)
        oracle.update(1_000_000)
        assert oracle.base_fee < initial

    def test_estimate_fee(self):
        oracle = GasPriceOracle(base_fee=100)
        estimate = oracle.estimate_fee(21000)
        assert isinstance(estimate, (int, float))
        assert estimate > 0

    def test_history(self):
        oracle = GasPriceOracle()
        oracle.update(5_000_000)
        oracle.update(8_000_000)
        assert len(oracle.history) >= 2


class TestABIEncoder:
    def test_function_selector(self):
        selector = ABIEncoder.function_selector("transfer", ["address", "uint256"])
        assert isinstance(selector, bytes)
        assert len(selector) == 4

    def test_encode_params_uint(self):
        encoded = ABIEncoder.encode_params(["uint256"], [42])
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

    def test_encode_params_bool(self):
        encoded = ABIEncoder.encode_params(["bool"], [True])
        assert isinstance(encoded, bytes)

    def test_encode_params_address(self):
        encoded = ABIEncoder.encode_params(["address"], ["0x1234567890abcdef1234567890abcdef12345678"])
        assert isinstance(encoded, bytes)

    def test_encode_function_call(self):
        encoded = ABIEncoder.encode_function_call(
            "transfer", 
            ["address", "uint256"],
            ["0x1234567890abcdef1234567890abcdef12345678", 1000]
        )
        assert isinstance(encoded, bytes)
        assert len(encoded) >= 4

    def test_decode_params_uint(self):
        encoded = ABIEncoder.encode_params(["uint256"], [42])
        decoded = ABIEncoder.decode_params(["uint256"], encoded)
        assert decoded == [42]

    def test_decode_params_bool(self):
        encoded = ABIEncoder.encode_params(["bool"], [True])
        decoded = ABIEncoder.decode_params(["bool"], encoded)
        assert decoded == [True]

    def test_roundtrip_multiple_params(self):
        types = ["uint256", "bool", "uint256"]
        values = [100, True, 200]
        encoded = ABIEncoder.encode_params(types, values)
        decoded = ABIEncoder.decode_params(types, encoded)
        assert decoded == values
