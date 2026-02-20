"""
Tests for the real blockchain infrastructure:
  - chain.py (Block, Chain, Mempool)
  - consensus.py (PoW, PoA, PoS)
  - network.py (Message, PeerInfo — unit-level, no real sockets)
  - node.py (BlockchainNode integration)
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
import pytest

from src.zexus.blockchain.chain import (
    Block, BlockHeader, Chain, Mempool, Transaction, TransactionReceipt,
)
from src.zexus.blockchain.consensus import (
    ProofOfWork, ProofOfAuthority, ProofOfStake, create_consensus,
)
from src.zexus.blockchain.network import Message, MessageType, PeerInfo
from src.zexus.blockchain.node import BlockchainNode, NodeConfig


# ── Test key pair (secp256k1) for signing transactions in tests ──
try:
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.backends import default_backend
    _test_priv = ec.generate_private_key(ec.SECP256K1(), default_backend())
    _test_pub = _test_priv.public_key()
    # Hex-encoded 32-byte scalar
    TEST_PRIVATE_KEY = format(
        _test_priv.private_numbers().private_value, '064x'
    )
    from cryptography.hazmat.primitives import serialization
    TEST_PUBLIC_KEY = _test_pub.public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    ).hex()
except ImportError:
    TEST_PRIVATE_KEY = "a" * 64
    TEST_PUBLIC_KEY = "04" + "a" * 128


def _sign_tx(tx: Transaction) -> Transaction:
    """Helper to sign a transaction with the test private key."""
    tx.sign(TEST_PRIVATE_KEY)
    return tx


# ═══════════════════════════════════════════════════════════════════════════
# Block / BlockHeader tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBlockHeader:
    def test_compute_hash_deterministic(self):
        h = BlockHeader(height=1, timestamp=1000.0, prev_hash="abc")
        assert h.compute_hash() == h.compute_hash()

    def test_different_nonces_give_different_hashes(self):
        h1 = BlockHeader(height=1, timestamp=1000.0, nonce=0)
        h2 = BlockHeader(height=1, timestamp=1000.0, nonce=1)
        assert h1.compute_hash() != h2.compute_hash()

    def test_hash_is_hex_sha256(self):
        h = BlockHeader(height=0, timestamp=0.0)
        hsh = h.compute_hash()
        assert len(hsh) == 64
        int(hsh, 16)  # Should not raise


class TestTransaction:
    def test_compute_hash(self):
        tx = Transaction(sender="alice", recipient="bob", value=100, timestamp=1000.0)
        h = tx.compute_hash()
        assert len(h) == 64
        assert tx.tx_hash == h

    def test_sign_and_verify(self):
        tx = Transaction(sender="alice", recipient="bob", value=50, timestamp=1000.0)
        tx.compute_hash()
        sig = tx.sign(TEST_PRIVATE_KEY)
        assert len(sig) > 0
        assert tx.verify(TEST_PUBLIC_KEY)

    def test_to_dict(self):
        tx = Transaction(sender="alice", recipient="bob", value=42)
        d = tx.to_dict()
        assert d["sender"] == "alice"
        assert d["value"] == 42


class TestBlock:
    def test_compute_hash(self):
        b = Block()
        b.header.height = 1
        b.header.timestamp = 1000.0
        h = b.compute_hash()
        assert len(h) == 64
        assert b.hash == h

    def test_compute_tx_root_empty(self):
        b = Block()
        root = b.compute_tx_root()
        assert len(root) == 64

    def test_compute_tx_root_with_txs(self):
        b = Block()
        tx1 = Transaction(sender="a", recipient="b", value=1, timestamp=1.0)
        tx1.compute_hash()
        tx2 = Transaction(sender="c", recipient="d", value=2, timestamp=2.0)
        tx2.compute_hash()
        b.transactions = [tx1, tx2]
        root = b.compute_tx_root()
        assert len(root) == 64

    def test_to_dict_and_from_dict_roundtrip(self):
        b = Block()
        b.header.height = 5
        b.header.timestamp = 42.0
        b.header.miner = "miner1"
        tx = Transaction(sender="a", recipient="b", value=10, timestamp=1.0)
        tx.compute_hash()
        b.transactions = [tx]
        b.receipts = [TransactionReceipt(tx_hash=tx.tx_hash, status=1)]
        b.compute_hash()

        d = b.to_dict()
        b2 = Block.from_dict(d)
        assert b2.hash == b.hash
        assert b2.header.height == 5
        assert len(b2.transactions) == 1
        assert b2.transactions[0].sender == "a"


# ═══════════════════════════════════════════════════════════════════════════
# Mempool tests
# ═══════════════════════════════════════════════════════════════════════════

class TestMempool:
    def test_add_and_size(self):
        mp = Mempool()
        tx = Transaction(sender="alice", recipient="bob", value=10, timestamp=1.0, nonce=0)
        tx.compute_hash()
        assert mp.add(tx)
        assert mp.size == 1

    def test_reject_duplicate(self):
        mp = Mempool()
        tx = Transaction(sender="alice", recipient="bob", value=10, timestamp=1.0, nonce=0)
        tx.compute_hash()
        mp.add(tx)
        assert not mp.add(tx)
        assert mp.size == 1

    def test_reject_low_nonce(self):
        mp = Mempool()
        tx1 = Transaction(sender="alice", recipient="bob", value=10, timestamp=1.0, nonce=1)
        tx1.compute_hash()
        mp.add(tx1)
        tx2 = Transaction(sender="alice", recipient="bob", value=5, timestamp=2.0, nonce=0)
        tx2.compute_hash()
        assert not mp.add(tx2)

    def test_max_size(self):
        mp = Mempool(max_size=2)
        for i in range(3):
            tx = Transaction(sender="alice", recipient="bob", value=i, timestamp=float(i), nonce=i)
            tx.compute_hash()
            mp.add(tx)
        assert mp.size == 2

    def test_get_pending_gas_ordering(self):
        mp = Mempool()
        tx_low = Transaction(sender="a", recipient="b", value=1, gas_price=1, timestamp=1.0, nonce=0)
        tx_low.compute_hash()
        tx_high = Transaction(sender="c", recipient="d", value=1, gas_price=10, timestamp=2.0, nonce=0)
        tx_high.compute_hash()
        mp.add(tx_low)
        mp.add(tx_high)
        pending = mp.get_pending()
        assert pending[0].gas_price >= pending[-1].gas_price

    def test_remove(self):
        mp = Mempool()
        tx = Transaction(sender="alice", recipient="bob", value=10, timestamp=1.0, nonce=0)
        tx.compute_hash()
        mp.add(tx)
        removed = mp.remove(tx.tx_hash)
        assert removed is not None
        assert mp.size == 0

    def test_clear(self):
        mp = Mempool()
        for i in range(5):
            tx = Transaction(sender="s", recipient="r", value=i, timestamp=float(i), nonce=i)
            tx.compute_hash()
            mp.add(tx)
        mp.clear()
        assert mp.size == 0


# ═══════════════════════════════════════════════════════════════════════════
# Chain tests
# ═══════════════════════════════════════════════════════════════════════════

class TestChain:
    def test_create_genesis(self):
        chain = Chain(chain_id="test")
        genesis = chain.create_genesis(miner="miner0")
        assert chain.height == 0
        assert genesis.header.height == 0
        assert genesis.hash
        assert chain.tip == genesis

    def test_cannot_create_double_genesis(self):
        chain = Chain(chain_id="test")
        chain.create_genesis()
        with pytest.raises(RuntimeError):
            chain.create_genesis()

    def test_initial_balances(self):
        chain = Chain(chain_id="test")
        chain.create_genesis(initial_balances={"alice": 1000, "bob": 500})
        assert chain.get_account("alice")["balance"] == 1000
        assert chain.get_account("bob")["balance"] == 500

    def test_add_valid_block(self):
        chain = Chain(chain_id="test")
        chain.create_genesis()
        chain.difficulty = 0  # No PoW for testing

        block = Block()
        block.header.height = 1
        block.header.prev_hash = chain.tip.hash
        block.header.timestamp = time.time()
        block.header.miner = "miner"
        block.compute_hash()

        success, err = chain.add_block(block)
        assert success, err
        assert chain.height == 1

    def test_reject_wrong_prev_hash(self):
        chain = Chain(chain_id="test")
        chain.create_genesis()
        chain.difficulty = 0

        block = Block()
        block.header.height = 1
        block.header.prev_hash = "wrong_hash"
        block.header.timestamp = time.time()
        block.compute_hash()

        success, err = chain.add_block(block)
        assert not success
        assert "prev_hash" in err

    def test_reject_wrong_height(self):
        chain = Chain(chain_id="test")
        chain.create_genesis()
        chain.difficulty = 0

        block = Block()
        block.header.height = 5  # Wrong
        block.header.prev_hash = chain.tip.hash
        block.header.timestamp = time.time()
        block.compute_hash()

        success, err = chain.add_block(block)
        assert not success
        assert "height" in err

    def test_reject_old_timestamp(self):
        chain = Chain(chain_id="test")
        chain.create_genesis()
        chain.difficulty = 0

        block = Block()
        block.header.height = 1
        block.header.prev_hash = chain.tip.hash
        block.header.timestamp = 0  # Before genesis
        block.compute_hash()

        success, err = chain.add_block(block)
        assert not success
        assert "timestamp" in err

    def test_get_block_by_height(self):
        chain = Chain(chain_id="test")
        genesis = chain.create_genesis()
        assert chain.get_block(0) == genesis

    def test_get_block_by_hash(self):
        chain = Chain(chain_id="test")
        genesis = chain.create_genesis()
        assert chain.get_block(genesis.hash) == genesis

    def test_validate_chain(self):
        chain = Chain(chain_id="test")
        chain.create_genesis()
        chain.difficulty = 0

        for i in range(1, 5):
            b = Block()
            b.header.height = i
            b.header.prev_hash = chain.tip.hash
            b.header.timestamp = time.time() + i
            b.header.miner = "miner"
            b.compute_hash()
            chain.add_block(b)

        valid, err = chain.validate_chain()
        assert valid, err

    def test_persistent_storage(self):
        tmpdir = tempfile.mkdtemp()
        try:
            # Create and populate chain
            chain1 = Chain(chain_id="persist-test", data_dir=tmpdir)
            chain1.create_genesis(initial_balances={"alice": 1000})
            chain1.difficulty = 0

            b = Block()
            b.header.height = 1
            b.header.prev_hash = chain1.tip.hash
            b.header.timestamp = time.time()
            b.header.miner = "m"
            b.compute_hash()
            chain1.add_block(b)
            chain1.close()

            # Reload from disk
            chain2 = Chain(chain_id="persist-test", data_dir=tmpdir)
            assert chain2.height == 1
            assert chain2.get_account("alice")["balance"] == 1000
            chain2.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_chain_info(self):
        chain = Chain(chain_id="info-test")
        chain.create_genesis()
        info = chain.get_chain_info()
        assert info["chain_id"] == "info-test"
        assert info["height"] == 0
        assert info["total_blocks"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Consensus tests
# ═══════════════════════════════════════════════════════════════════════════

class TestProofOfWork:
    def test_seal_and_verify(self):
        chain = Chain(chain_id="pow-test")
        chain.create_genesis(miner="miner", initial_balances={"alice": 100000})
        chain.difficulty = 0
        mp = Mempool()
        tx = Transaction(sender="alice", recipient="bob", value=100, nonce=0,
                          gas_limit=21000, gas_price=1, timestamp=time.time())
        tx.compute_hash()
        mp.add(tx)

        pow = ProofOfWork(difficulty=1)
        block = pow.seal(chain, mp, "miner")
        assert block is not None
        assert block.hash.startswith("0")
        assert pow.verify(block, chain)

    def test_verify_rejects_bad_hash(self):
        pow = ProofOfWork(difficulty=1)
        chain = Chain(chain_id="test")
        chain.create_genesis()
        b = Block()
        b.hash = "fffff"  # Does not start with 0
        b.header.difficulty = 1
        assert not pow.verify(b, chain)


class TestProofOfAuthority:
    def test_seal_correct_turn(self):
        chain = Chain(chain_id="poa-test")
        chain.create_genesis(miner="v1", initial_balances={"alice": 100000})
        chain.difficulty = 0
        mp = Mempool()
        tx = Transaction(sender="alice", recipient="bob", value=50, nonce=0,
                          gas_limit=21000, gas_price=1, timestamp=time.time())
        tx.compute_hash()
        mp.add(tx)

        poa = ProofOfAuthority(validators=["v1", "v2", "v3"])
        # Height 1 → validator index 1 → "v2"
        block = poa.seal(chain, mp, "v2")
        assert block is not None
        assert poa.verify(block, chain)

    def test_seal_wrong_turn(self):
        chain = Chain(chain_id="poa-test")
        chain.create_genesis(miner="v1", initial_balances={"alice": 100000})
        chain.difficulty = 0
        mp = Mempool()

        poa = ProofOfAuthority(validators=["v1", "v2"])
        # Height 1 → should be v2's turn
        block = poa.seal(chain, mp, "v1")
        assert block is None

    def test_seal_not_a_validator(self):
        chain = Chain(chain_id="poa-test")
        chain.create_genesis()
        mp = Mempool()
        poa = ProofOfAuthority(validators=["v1"])
        block = poa.seal(chain, mp, "unknown")
        assert block is None

    def test_add_remove_validator(self):
        poa = ProofOfAuthority()
        poa.add_validator("v1")
        poa.add_validator("v2")
        assert len(poa.validators) == 2
        poa.remove_validator("v1")
        assert poa.validators == ["v2"]


class TestProofOfStake:
    def test_seal_with_stake(self):
        chain = Chain(chain_id="pos-test")
        chain.create_genesis(miner="v1", initial_balances={"alice": 100000})
        chain.difficulty = 0
        mp = Mempool()
        tx = Transaction(sender="alice", recipient="bob", value=10, nonce=0,
                          gas_limit=21000, gas_price=1, timestamp=time.time())
        tx.compute_hash()
        mp.add(tx)

        pos = ProofOfStake(min_stake=100)
        pos.stake("v1", 5000)
        # v1 is the only eligible validator, so it should be selected
        block = pos.seal(chain, mp, "v1")
        assert block is not None
        assert pos.verify(block, chain)

    def test_seal_insufficient_stake(self):
        chain = Chain(chain_id="pos-test")
        chain.create_genesis()
        mp = Mempool()
        pos = ProofOfStake(min_stake=1000)
        pos.stake("v1", 500)  # Below min
        block = pos.seal(chain, mp, "v1")
        assert block is None

    def test_stake_and_unstake(self):
        pos = ProofOfStake(min_stake=100)
        pos.stake("v1", 1000)
        assert pos.stakes["v1"] == 1000
        pos.unstake("v1", 500)
        assert pos.stakes["v1"] == 500
        pos.unstake("v1", 500)
        assert "v1" not in pos.stakes

    def test_deterministic_selection(self):
        pos = ProofOfStake(min_stake=100)
        pos.stake("v1", 1000)
        pos.stake("v2", 2000)
        pos.stake("v3", 3000)
        # Same inputs → same output
        s1 = pos._select_validator("abc123", 10)
        s2 = pos._select_validator("abc123", 10)
        assert s1 == s2


class TestCreateConsensus:
    def test_create_pow(self):
        engine = create_consensus("pow", difficulty=2)
        assert isinstance(engine, ProofOfWork)

    def test_create_poa(self):
        engine = create_consensus("poa", validators=["v1"])
        assert isinstance(engine, ProofOfAuthority)

    def test_create_pos(self):
        engine = create_consensus("pos", min_stake=500)
        assert isinstance(engine, ProofOfStake)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            create_consensus("unknown_algo")


# ═══════════════════════════════════════════════════════════════════════════
# Network / Message tests (no real sockets)
# ═══════════════════════════════════════════════════════════════════════════

class TestMessage:
    def test_encode_decode_roundtrip(self):
        msg = Message(type=MessageType.PING, payload={"data": 42}, sender="node1")
        raw = msg.encode()
        msg2 = Message.decode(raw)
        assert msg2.type == MessageType.PING
        assert msg2.payload["data"] == 42
        assert msg2.sender == "node1"

    def test_unique_nonces(self):
        m1 = Message(type="a")
        m2 = Message(type="a")
        assert m1.nonce != m2.nonce

    def test_timestamp_set(self):
        msg = Message(type="test")
        assert msg.timestamp > 0


class TestPeerInfo:
    def test_address(self):
        p = PeerInfo(host="192.168.1.1", port=30303)
        assert p.address == "192.168.1.1:30303"

    def test_defaults(self):
        p = PeerInfo()
        assert p.reputation == 100
        assert not p.connected


# ═══════════════════════════════════════════════════════════════════════════
# Node integration tests (no network, just chain + mempool + consensus)
# ═══════════════════════════════════════════════════════════════════════════

class TestBlockchainNode:
    def _make_node(self, **kw) -> BlockchainNode:
        config = NodeConfig(
            chain_id="test-node",
            consensus="pow",
            consensus_params={"difficulty": 1},
            miner_address="miner1",
            mining_enabled=False,
            **kw,
        )
        node = BlockchainNode(config)
        node.chain.create_genesis(
            miner="miner1",
            initial_balances={"alice": 1_000_000, "bob": 500_000},
        )
        return node

    def test_submit_transaction(self):
        node = self._make_node()
        tx = node.create_transaction("alice", "bob", 100)
        result = node.submit_transaction(tx)
        assert result["success"]
        assert node.mempool.size == 1

    def test_submit_insufficient_balance(self):
        node = self._make_node()
        tx = node.create_transaction("alice", "bob", 99_999_999)
        result = node.submit_transaction(tx)
        assert not result["success"]
        assert "insufficient" in result["error"]

    def test_get_balance(self):
        node = self._make_node()
        assert node.get_balance("alice") == 1_000_000
        assert node.get_balance("unknown") == 0

    def test_get_nonce(self):
        node = self._make_node()
        assert node.get_nonce("alice") == 0

    def test_fund_account(self):
        node = self._make_node()
        node.fund_account("charlie", 5000)
        assert node.get_balance("charlie") == 5000

    def test_mine_block_sync(self):
        node = self._make_node()
        tx = node.create_transaction("alice", "bob", 100)
        _sign_tx(tx)
        node.submit_transaction(tx)
        block = node.mine_block_sync()
        assert block is not None
        assert node.chain.height == 1

    def test_mine_multiple_blocks(self):
        node = self._make_node()
        # Pin difficulty so auto-adjustment doesn't cause PoW to fail
        node.chain.difficulty = 0
        for i in range(3):
            tx = node.create_transaction("alice", "bob", 10, gas_limit=21000)
            _sign_tx(tx)
            node.submit_transaction(tx)
            block = node.mine_block_sync()
            assert block is not None
        assert node.chain.height == 3

    def test_get_chain_info(self):
        node = self._make_node()
        info = node.get_chain_info()
        assert info["chain_id"] == "test-node"
        assert info["height"] == 0
        assert info["consensus"] == "pow"

    def test_get_block(self):
        node = self._make_node()
        b = node.get_block(0)
        assert b is not None
        assert b["header"]["height"] == 0

    def test_get_latest_block(self):
        node = self._make_node()
        b = node.get_latest_block()
        assert b is not None
        assert b["header"]["height"] == 0

    def test_validate_chain(self):
        node = self._make_node()
        result = node.validate_chain()
        assert result["valid"]

    def test_export_chain(self):
        node = self._make_node()
        export = node.export_chain()
        assert len(export) == 1
        assert export[0]["header"]["height"] == 0

    def test_node_with_poa(self):
        node = BlockchainNode(NodeConfig(
            chain_id="poa-test",
            consensus="poa",
            consensus_params={"validators": ["v1", "v2"]},
            miner_address="v2",
        ))
        node.chain.create_genesis(miner="v1", initial_balances={"alice": 100000})
        node.chain.difficulty = 0

        tx = node.create_transaction("alice", "bob", 50)
        _sign_tx(tx)
        node.submit_transaction(tx)
        # Height 1 → v2's turn
        block = node.mine_block_sync()
        assert block is not None
        assert block.header.miner == "v2"

    def test_node_with_pos(self):
        config = NodeConfig(
            chain_id="pos-test",
            consensus="pos",
            consensus_params={"min_stake": 100},
            miner_address="v1",
        )
        node = BlockchainNode(config)
        node.chain.create_genesis(miner="v1", initial_balances={"alice": 100000})
        node.chain.difficulty = 0

        # Stake
        node.consensus_engine.stake("v1", 5000)

        tx = node.create_transaction("alice", "bob", 50)
        _sign_tx(tx)
        node.submit_transaction(tx)
        # v1 is the only eligible validator
        block = node.mine_block_sync()
        assert block is not None

    def test_repr(self):
        node = self._make_node()
        r = repr(node)
        assert "test-node" in r
        assert "height=0" in r
