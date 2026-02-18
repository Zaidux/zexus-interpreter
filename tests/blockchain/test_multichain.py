"""
Tests for the Zexus multichain infrastructure:
  - MerkleProofEngine (generate/verify proofs)
  - CrossChainMessage (serialization, hashing)
  - BridgeRelay (header tracking, message verification, replay protection)
  - ChainRouter (register, connect, send, flush, relay, inbox)
  - BridgeContract (lock-and-mint, burn-and-release, edge cases)
  - BlockchainNode.bridge_to integration
"""

import copy
import hashlib
import json
import pytest
import tempfile

from src.zexus.blockchain.chain import Chain, Block, BlockHeader
from src.zexus.blockchain.node import BlockchainNode, NodeConfig
from src.zexus.blockchain.multichain import (
    MerkleProofEngine,
    CrossChainMessage,
    MessageStatus,
    BridgeRelay,
    ChainRouter,
    BridgeContract,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _make_chain(chain_id: str, tmp_path, fund: dict = None) -> Chain:
    """Create a chain with genesis block and optional funded accounts."""
    chain = Chain(chain_id=chain_id, data_dir=str(tmp_path / chain_id))
    chain.create_genesis()
    for addr, bal in (fund or {}).items():
        chain.get_account(addr)["balance"] = bal
    return chain


# ═══════════════════════════════════════════════════════════════════════
# MerkleProofEngine
# ═══════════════════════════════════════════════════════════════════════

class TestMerkleProofEngine:
    def test_single_leaf(self):
        leaves = ["aaa"]
        root = MerkleProofEngine.compute_root(leaves)
        proof = MerkleProofEngine.generate_proof(leaves, 0)
        assert MerkleProofEngine.verify_proof("aaa", proof, root)

    def test_two_leaves(self):
        leaves = ["aaa", "bbb"]
        root = MerkleProofEngine.compute_root(leaves)
        for i in range(2):
            proof = MerkleProofEngine.generate_proof(leaves, i)
            assert MerkleProofEngine.verify_proof(leaves[i], proof, root)

    def test_odd_leaves(self):
        leaves = ["a", "b", "c"]
        root = MerkleProofEngine.compute_root(leaves)
        for i in range(3):
            proof = MerkleProofEngine.generate_proof(leaves, i)
            assert MerkleProofEngine.verify_proof(leaves[i], proof, root)

    def test_many_leaves(self):
        leaves = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(17)]
        root = MerkleProofEngine.compute_root(leaves)
        for i in range(len(leaves)):
            proof = MerkleProofEngine.generate_proof(leaves, i)
            assert MerkleProofEngine.verify_proof(leaves[i], proof, root)

    def test_wrong_leaf_fails(self):
        leaves = ["aaa", "bbb", "ccc"]
        root = MerkleProofEngine.compute_root(leaves)
        proof = MerkleProofEngine.generate_proof(leaves, 0)
        assert not MerkleProofEngine.verify_proof("zzz", proof, root)

    def test_empty_leaves(self):
        root = MerkleProofEngine.compute_root([])
        assert len(root) == 64  # sha256 hex

    def test_invalid_index_returns_empty_proof(self):
        assert MerkleProofEngine.generate_proof(["a", "b"], -1) == []
        assert MerkleProofEngine.generate_proof(["a", "b"], 5) == []
        assert MerkleProofEngine.generate_proof([], 0) == []


# ═══════════════════════════════════════════════════════════════════════
# CrossChainMessage
# ═══════════════════════════════════════════════════════════════════════

class TestCrossChainMessage:
    def test_compute_hash_deterministic(self):
        msg = CrossChainMessage(
            msg_id="abc", nonce=0, source_chain="a", dest_chain="b",
            sender="alice", payload={"x": 1}, timestamp=1000.0,
        )
        assert msg.compute_hash() == msg.compute_hash()

    def test_different_payloads_different_hashes(self):
        base = dict(msg_id="abc", nonce=0, source_chain="a",
                    dest_chain="b", sender="alice", timestamp=1000.0)
        m1 = CrossChainMessage(**base, payload={"x": 1})
        m2 = CrossChainMessage(**base, payload={"x": 2})
        assert m1.compute_hash() != m2.compute_hash()

    def test_round_trip_serialization(self):
        msg = CrossChainMessage(
            msg_id="test123", nonce=5, source_chain="chain-a",
            dest_chain="chain-b", sender="alice",
            payload={"action": "mint", "amount": 100},
            block_height=10, block_hash="deadbeef",
            merkle_root="cafebabe",
            merkle_proof=[("aaa", "L"), ("bbb", "R")],
            timestamp=12345.0, status=MessageStatus.CONFIRMED,
        )
        d = msg.to_dict()
        restored = CrossChainMessage.from_dict(d)
        assert restored.msg_id == "test123"
        assert restored.nonce == 5
        assert restored.status == MessageStatus.CONFIRMED
        assert restored.merkle_proof == [("aaa", "L"), ("bbb", "R")]
        assert restored.payload == {"action": "mint", "amount": 100}


# ═══════════════════════════════════════════════════════════════════════
# BridgeRelay
# ═══════════════════════════════════════════════════════════════════════

class TestBridgeRelay:
    def _make_relay(self):
        return BridgeRelay(local_chain_id="dest")

    def test_submit_and_retrieve_header(self):
        relay = self._make_relay()
        h = {"height": 0, "hash": "aaa", "prev_hash": "", "extra_data": ""}
        assert relay.submit_header("source", h) is True
        assert relay.has_header("source", 0)
        assert relay.get_header("source", 0) == h

    def test_header_chain_linkage_enforced(self):
        relay = self._make_relay()
        relay.submit_header("source", {"height": 0, "hash": "a"})
        # Wrong prev_hash should be rejected
        ok = relay.submit_header("source", {
            "height": 1, "hash": "b", "prev_hash": "WRONG",
        })
        assert ok is False

    def test_latest_height(self):
        relay = self._make_relay()
        assert relay.latest_height("source") == -1
        relay.submit_header("source", {"height": 0, "hash": "a"})
        assert relay.latest_height("source") == 0
        relay.submit_header("source", {
            "height": 1, "hash": "b", "prev_hash": "a",
        })
        assert relay.latest_height("source") == 1

    def test_verify_message_bad_destination(self):
        relay = self._make_relay()
        msg = CrossChainMessage(dest_chain="WRONG")
        ok, reason = relay.verify_message(msg)
        assert ok is False
        assert "WRONG" in reason

    def test_verify_message_no_header(self):
        relay = self._make_relay()
        msg = CrossChainMessage(
            dest_chain="dest", source_chain="source", block_height=0,
        )
        ok, reason = relay.verify_message(msg)
        assert ok is False
        assert "No header" in reason

    def test_verify_message_block_hash_mismatch(self):
        relay = self._make_relay()
        relay.submit_header("source", {
            "height": 0, "hash": "correct_hash",
            "extra_data": "", "state_root": "",
        })
        msg = CrossChainMessage(
            dest_chain="dest", source_chain="source",
            block_height=0, block_hash="wrong_hash",
        )
        ok, reason = relay.verify_message(msg)
        assert ok is False
        assert "hash mismatch" in reason.lower()

    def test_replay_protection(self):
        relay = self._make_relay()
        # Build a valid message + proof
        leaf_hash = hashlib.sha256(b"test").hexdigest()
        root = MerkleProofEngine.compute_root([leaf_hash])

        relay.submit_header("source", {
            "height": 0, "hash": "blockhash",
            "extra_data": f"xchain_root:{root}",
        })

        msg = CrossChainMessage(
            msg_id="unique1", nonce=0,
            dest_chain="dest", source_chain="source",
            block_height=0, block_hash="blockhash",
            merkle_root=root,
            merkle_proof=MerkleProofEngine.generate_proof([leaf_hash], 0),
        )
        # Hack: override compute_hash to match leaf_hash
        msg.compute_hash = lambda: leaf_hash

        ok, reason = relay.verify_message(msg)
        assert ok is True
        relay.accept_message(msg)

        # Replay should fail
        ok2, reason2 = relay.verify_message(msg)
        assert ok2 is False
        assert "already processed" in reason2.lower()


# ═══════════════════════════════════════════════════════════════════════
# ChainRouter
# ═══════════════════════════════════════════════════════════════════════

class TestChainRouter:
    def test_register_and_connect(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")
        assert router.is_connected("alpha", "beta")
        assert router.is_connected("beta", "alpha")

    def test_duplicate_register_raises(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        router.register_chain("alpha", ca)
        with pytest.raises(ValueError, match="already registered"):
            router.register_chain("alpha", ca)

    def test_connect_unregistered_raises(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        router.register_chain("alpha", ca)
        with pytest.raises(ValueError, match="not registered"):
            router.connect("alpha", "ghost")

    def test_send_without_connection_raises(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        with pytest.raises(ValueError, match="No bridge"):
            router.send("alpha", "beta", "alice", {"x": 1})

    def test_send_assigns_nonce_and_id(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")

        m1 = router.send("alpha", "beta", "alice", {"x": 1})
        m2 = router.send("alpha", "beta", "alice", {"x": 2})
        assert m1.nonce == 0
        assert m2.nonce == 1
        assert m1.msg_id != m2.msg_id

    def test_flush_generates_proofs(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")

        router.send("alpha", "beta", "alice", {"action": "ping"})
        router.send("alpha", "beta", "bob", {"action": "pong"})
        flushed = router.flush_outbox("alpha")
        assert len(flushed) == 2
        for msg in flushed:
            assert msg.merkle_root != ""
            assert len(msg.merkle_proof) > 0
            assert msg.status == MessageStatus.RELAYED

    def test_send_and_relay_end_to_end(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")

        msg, ok, reason = router.send_and_relay(
            "alpha", "beta", "alice", {"action": "test"},
        )
        assert ok is True, f"Relay failed: {reason}"
        assert msg.status == MessageStatus.CONFIRMED
        inbox = router.get_inbox("beta")
        assert len(inbox) == 1
        assert inbox[0].msg_id == msg.msg_id

    def test_bidirectional_messaging(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")

        # alpha -> beta
        m1, ok1, _ = router.send_and_relay("alpha", "beta", "alice", {"dir": "a2b"})
        assert ok1

        # beta -> alpha
        m2, ok2, _ = router.send_and_relay("beta", "alpha", "bob", {"dir": "b2a"})
        assert ok2

        assert len(router.get_inbox("beta")) == 1
        assert len(router.get_inbox("alpha")) == 1

    def test_pop_inbox_clears(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")

        router.send_and_relay("alpha", "beta", "alice", {"x": 1})
        msgs = router.pop_inbox("beta")
        assert len(msgs) == 1
        assert len(router.get_inbox("beta")) == 0

    def test_router_info(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")

        info = router.get_router_info()
        assert "alpha" in info["chains"]
        assert "beta" in info["chains"]
        assert "beta" in info["connections"]["alpha"]

    def test_message_log_audit_trail(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")

        router.send_and_relay("alpha", "beta", "alice", {"a": 1})
        router.send_and_relay("alpha", "beta", "bob", {"b": 2})
        log = router.get_message_log()
        assert len(log) == 2
        assert log[0]["sender"] == "alice"
        assert log[1]["sender"] == "bob"


# ═══════════════════════════════════════════════════════════════════════
# BridgeContract — lock-and-mint / burn-and-release
# ═══════════════════════════════════════════════════════════════════════

class TestBridgeContract:
    def _setup_bridge(self, tmp_path, bal_a=1000, bal_b=0):
        router = ChainRouter()
        ca = _make_chain("src", tmp_path, fund={"alice": bal_a})
        cb = _make_chain("dst", tmp_path, fund={"bob": bal_b})
        router.register_chain("src", ca)
        router.register_chain("dst", cb)
        router.connect("src", "dst")
        bridge = BridgeContract(router, "src", "dst")
        return router, ca, cb, bridge

    def test_lock_and_mint_basic(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        receipt = bridge.lock_and_mint(sender="alice", amount=100)
        assert receipt["success"] is True
        assert receipt["amount"] == 100
        # Source balance reduced
        assert ca.get_account("alice")["balance"] == 900
        # Dest balance credited
        assert cb.get_account("alice")["balance"] == 100
        # TVL
        assert bridge.total_value_locked == 100
        assert bridge.total_minted == 100

    def test_lock_and_mint_to_different_recipient(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        receipt = bridge.lock_and_mint(sender="alice", amount=50, recipient="bob")
        assert receipt["success"] is True
        assert ca.get_account("alice")["balance"] == 950
        assert cb.get_account("bob")["balance"] == 50

    def test_lock_insufficient_balance(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path, bal_a=10)
        receipt = bridge.lock_and_mint(sender="alice", amount=100)
        assert receipt["success"] is False
        assert "Insufficient" in receipt["error"]
        # No state change
        assert ca.get_account("alice")["balance"] == 10
        assert bridge.total_value_locked == 0

    def test_lock_zero_amount(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        receipt = bridge.lock_and_mint(sender="alice", amount=0)
        assert receipt["success"] is False

    def test_lock_negative_amount(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        receipt = bridge.lock_and_mint(sender="alice", amount=-5)
        assert receipt["success"] is False

    def test_burn_and_release_basic(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        # First lock-and-mint
        bridge.lock_and_mint(sender="alice", amount=200)
        assert ca.get_account("alice")["balance"] == 800
        assert cb.get_account("alice")["balance"] == 200

        # Now burn-and-release back
        receipt = bridge.burn_and_release(sender="alice", amount=80)
        assert receipt["success"] is True
        assert receipt["amount"] == 80
        # Source gets funds back
        assert ca.get_account("alice")["balance"] == 880
        # Dest balance reduced
        assert cb.get_account("alice")["balance"] == 120
        # TVL reduced
        assert bridge.total_value_locked == 120
        assert bridge.total_minted == 120

    def test_burn_insufficient_minted_balance(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        bridge.lock_and_mint(sender="alice", amount=50)
        receipt = bridge.burn_and_release(sender="alice", amount=100)
        assert receipt["success"] is False
        assert "Insufficient minted" in receipt["error"]

    def test_burn_to_different_recipient(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        bridge.lock_and_mint(sender="alice", amount=100)
        receipt = bridge.burn_and_release(sender="alice", amount=30, recipient="charlie")
        assert receipt["success"] is True
        assert ca.get_account("charlie")["balance"] == 30

    def test_multiple_transfers(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        bridge.lock_and_mint(sender="alice", amount=100)
        bridge.lock_and_mint(sender="alice", amount=200)
        assert bridge.total_value_locked == 300
        assert ca.get_account("alice")["balance"] == 700
        assert cb.get_account("alice")["balance"] == 300

        bridge.burn_and_release(sender="alice", amount=150)
        assert bridge.total_value_locked == 150
        assert ca.get_account("alice")["balance"] == 850

    def test_bridge_info(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        bridge.lock_and_mint(sender="alice", amount=100)
        info = bridge.get_bridge_info()
        assert info["total_value_locked"] == 100
        assert info["total_minted"] == 100
        assert info["source_chain"] == "src"
        assert info["dest_chain"] == "dst"

    def test_tx_log(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        bridge.lock_and_mint(sender="alice", amount=100)
        bridge.burn_and_release(sender="alice", amount=50)
        log = bridge.get_tx_log()
        assert len(log) == 2
        assert log[0]["action"] == "lock_and_mint"
        assert log[1]["action"] == "burn_and_release"

    def test_escrow_balance_tracking(self, tmp_path):
        router, ca, cb, bridge = self._setup_bridge(tmp_path)
        bridge.lock_and_mint(sender="alice", amount=100)
        assert bridge.get_escrow_balance("src", "alice") == 100
        assert bridge.get_minted_balance("alice") == 100

        bridge.burn_and_release(sender="alice", amount=40)
        assert bridge.get_escrow_balance("src", "alice") == 60
        assert bridge.get_minted_balance("alice") == 60


# ═══════════════════════════════════════════════════════════════════════
# Edge cases: Merkle proof tampering / nonce manipulation / state safety
# ═══════════════════════════════════════════════════════════════════════

class TestMultichainEdgeCases:
    def test_tampered_proof_rejected(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")

        msg = router.send("alpha", "beta", "alice", {"x": 1})
        flushed = router.flush_outbox("alpha")
        assert len(flushed) == 1

        # Tamper with the proof
        flushed[0].merkle_proof = [("fake_hash", "L")]
        results = router.relay(flushed)
        _, ok, reason = results[0]
        assert ok is False
        assert "proof" in reason.lower() or "failed" in reason.lower()

    def test_nonce_replay_rejected(self, tmp_path):
        router = ChainRouter()
        ca = _make_chain("alpha", tmp_path)
        cb = _make_chain("beta", tmp_path)
        router.register_chain("alpha", ca)
        router.register_chain("beta", cb)
        router.connect("alpha", "beta")

        # Send and relay a valid message
        msg, ok, _ = router.send_and_relay("alpha", "beta", "alice", {"x": 1})
        assert ok is True

        # Attempt to re-relay with same nonce (simulate replay attack)
        relay = router.get_relay("beta")
        ok2, reason2 = relay.verify_message(msg)
        assert ok2 is False  # Already processed

    def test_three_chain_triangle(self, tmp_path):
        """Three chains connected in a triangle can all communicate."""
        router = ChainRouter()
        ca = _make_chain("A", tmp_path, fund={"alice": 1000})
        cb = _make_chain("B", tmp_path, fund={"bob": 1000})
        cc = _make_chain("C", tmp_path, fund={"carol": 1000})
        router.register_chain("A", ca)
        router.register_chain("B", cb)
        router.register_chain("C", cc)
        router.connect("A", "B")
        router.connect("B", "C")
        router.connect("A", "C")

        # A -> B
        m1, ok1, _ = router.send_and_relay("A", "B", "alice", {"to": "B"})
        assert ok1
        # B -> C
        m2, ok2, _ = router.send_and_relay("B", "C", "bob", {"to": "C"})
        assert ok2
        # C -> A
        m3, ok3, _ = router.send_and_relay("C", "A", "carol", {"to": "A"})
        assert ok3

        assert len(router.get_inbox("B")) == 1
        assert len(router.get_inbox("C")) == 1
        assert len(router.get_inbox("A")) == 1

    def test_bridge_across_three_chains(self, tmp_path):
        """Two bridges: A<->B and B<->C. Transfer A→B→C."""
        router = ChainRouter()
        ca = _make_chain("A", tmp_path, fund={"alice": 500})
        cb = _make_chain("B", tmp_path)
        cc = _make_chain("C", tmp_path)
        router.register_chain("A", ca)
        router.register_chain("B", cb)
        router.register_chain("C", cc)
        router.connect("A", "B")
        router.connect("B", "C")

        bridge_ab = BridgeContract(router, "A", "B")
        bridge_bc = BridgeContract(router, "B", "C")

        # A -> B: lock 200 on A, mint 200 on B
        r1 = bridge_ab.lock_and_mint(sender="alice", amount=200)
        assert r1["success"]
        assert cb.get_account("alice")["balance"] == 200

        # B -> C: lock 100 on B (alice's minted tokens), mint 100 on C
        # We need to use B<->C bridge directly — alice has 200 on B
        r2 = bridge_bc.lock_and_mint(sender="alice", amount=100)
        assert r2["success"]
        assert cc.get_account("alice")["balance"] == 100
        assert cb.get_account("alice")["balance"] == 100  # 200 - 100

    def test_concurrent_bridges_same_chain(self, tmp_path):
        """Two bridges from the same source chain to different destinations."""
        router = ChainRouter()
        ca = _make_chain("hub", tmp_path, fund={"alice": 1000})
        cb = _make_chain("spoke1", tmp_path)
        cc = _make_chain("spoke2", tmp_path)
        router.register_chain("hub", ca)
        router.register_chain("spoke1", cb)
        router.register_chain("spoke2", cc)
        router.connect("hub", "spoke1")
        router.connect("hub", "spoke2")

        b1 = BridgeContract(router, "hub", "spoke1")
        b2 = BridgeContract(router, "hub", "spoke2")

        r1 = b1.lock_and_mint(sender="alice", amount=300)
        r2 = b2.lock_and_mint(sender="alice", amount=200)

        assert r1["success"]
        assert r2["success"]
        assert ca.get_account("alice")["balance"] == 500
        assert cb.get_account("alice")["balance"] == 300
        assert cc.get_account("alice")["balance"] == 200


# ═══════════════════════════════════════════════════════════════════════
# Node integration — bridge_to helper
# ═══════════════════════════════════════════════════════════════════════

class TestNodeBridgeIntegration:
    def test_bridge_to_creates_bridge(self, tmp_path):
        node_a = BlockchainNode(NodeConfig(
            chain_id="net-a", data_dir=str(tmp_path / "a"),
            miner_address="miner_a",
        ))
        node_b = BlockchainNode(NodeConfig(
            chain_id="net-b", data_dir=str(tmp_path / "b"),
            miner_address="miner_b",
        ))
        # Nodes need genesis blocks for bridging (normally created in start())
        node_a.chain.create_genesis(miner="miner_a")
        node_b.chain.create_genesis(miner="miner_b")
        # Fund alice on node_a
        node_a.fund_account("alice", 1000)

        bridge = node_a.bridge_to(node_b)
        receipt = bridge.lock_and_mint(sender="alice", amount=100)
        assert receipt["success"] is True
        assert node_a.get_balance("alice") == 900
        assert node_b.get_balance("alice") == 100

    def test_bridge_to_with_shared_router(self, tmp_path):
        from src.zexus.blockchain.multichain import ChainRouter

        router = ChainRouter()
        na = BlockchainNode(NodeConfig(
            chain_id="x", data_dir=str(tmp_path / "x"),
            miner_address="mx",
        ))
        nb = BlockchainNode(NodeConfig(
            chain_id="y", data_dir=str(tmp_path / "y"),
            miner_address="my",
        ))
        na.chain.create_genesis(miner="mx")
        nb.chain.create_genesis(miner="my")
        na.fund_account("alice", 500)

        bridge = na.bridge_to(nb, router=router)
        assert "x" in router.chain_ids
        assert "y" in router.chain_ids
        assert bridge.lock_and_mint("alice", 100)["success"]
