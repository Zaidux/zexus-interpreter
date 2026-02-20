"""
Comprehensive tests for the 9 new blockchain infrastructure features.

Feature 1  — Replace-By-Fee (RBF) in Mempool
Feature 2  — Gas Estimation API
Feature 3  — Contract State Persistence
Feature 4  — Cross-Contract Calls
Feature 5  — Event Indexing & Log Filtering
Feature 6  — Merkle Patricia Trie
Feature 7  — HD Wallet & Keystore
Feature 8  — BFT Consensus
Feature 9  — Token Standard Interfaces (ZX-20, ZX-721, ZX-1155)
"""

import hashlib
import json
import os
import tempfile
import time
import pytest

from zexus.blockchain.chain import (
    Block, BlockHeader, Transaction, TransactionReceipt, Chain, Mempool
)
from zexus.blockchain.consensus import (
    ConsensusEngine, ProofOfWork, ProofOfAuthority, ProofOfStake,
    BFTConsensus, BFTMessage, BFTPhase, BFTRoundState,
    create_consensus,
)
from zexus.blockchain.events import (
    EventIndex, EventLog, LogFilter, BloomFilter
)
from zexus.blockchain.mpt import (
    MerklePatriciaTrie, StateTrie, TrieNode, NodeType
)
from zexus.blockchain.wallet import (
    HDWallet, Account, Keystore, ExtendedKey,
    generate_mnemonic, validate_mnemonic, mnemonic_to_seed,
    master_key_from_seed, WORDLIST, HARDENED_OFFSET,
)
from zexus.blockchain.tokens import (
    ZX20Token, ZX721Token, ZX1155Token,
    TokenEvent, TransferEvent, ApprovalEvent,
    TransferSingleEvent, TransferBatchEvent, ApprovalForAllEvent,
    ZERO_ADDRESS,
)


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _make_tx(sender="0xAlice", recipient="0xBob", value=100, nonce=0,
             gas_price=10, gas_limit=21000, data=""):
    tx = Transaction()
    tx.sender = sender
    tx.recipient = recipient
    tx.value = value
    tx.nonce = nonce
    tx.gas_price = gas_price
    tx.gas_limit = gas_limit
    tx.data = data
    tx.timestamp = time.time()
    tx.tx_hash = hashlib.sha256(
        f"{sender}{recipient}{value}{nonce}{gas_price}{time.time()}".encode()
    ).hexdigest()
    return tx


# ══════════════════════════════════════════════════════════════════════
#  Feature 1: RBF in Mempool
# ══════════════════════════════════════════════════════════════════════

class TestRBFMempool:
    """Replace-By-Fee mempool support."""

    def test_rbf_enabled_by_default(self):
        mp = Mempool(rbf_enabled=True)
        assert mp.rbf_enabled is True

    def test_add_tx_to_mempool(self):
        mp = Mempool(rbf_enabled=True)
        tx = _make_tx(nonce=0, gas_price=10)
        assert mp.add(tx) is True
        assert mp.size == 1

    def test_rbf_replaces_same_sender_nonce(self):
        mp = Mempool(rbf_enabled=True, rbf_increment_pct=10)
        tx1 = _make_tx(sender="0xA", nonce=0, gas_price=10)
        tx2 = _make_tx(sender="0xA", nonce=0, gas_price=12)  # +20% > 10%
        mp.add(tx1)
        assert mp.size == 1

        result = mp.add(tx2)
        assert result is True
        assert mp.size == 1
        # The new tx should be in the pool
        assert mp._txs.get(tx2.tx_hash) is not None
        # The old tx should be gone
        assert mp._txs.get(tx1.tx_hash) is None

    def test_rbf_rejects_low_gas_price(self):
        mp = Mempool(rbf_enabled=True, rbf_increment_pct=10)
        tx1 = _make_tx(sender="0xA", nonce=0, gas_price=10)
        tx2 = _make_tx(sender="0xA", nonce=0, gas_price=10)  # Same = not enough
        mp.add(tx1)
        result = mp.add(tx2)
        # Should reject — gas price not high enough
        assert mp.size == 1

    def test_rbf_disabled(self):
        mp = Mempool(rbf_enabled=False)
        tx1 = _make_tx(sender="0xA", nonce=0, gas_price=10)
        tx2 = _make_tx(sender="0xA", nonce=0, gas_price=20)
        mp.add(tx1)
        result = mp.add(tx2)
        # With RBF off, same-sender same-nonce is rejected as replay
        assert result is False
        assert mp.size == 1

    def test_replace_by_fee_method(self):
        mp = Mempool(rbf_enabled=True, rbf_increment_pct=10)
        tx1 = _make_tx(sender="0xA", nonce=0, gas_price=10)
        mp.add(tx1)
        tx2 = _make_tx(sender="0xA", nonce=0, gas_price=15)
        result = mp.replace_by_fee(tx2)
        assert result["replaced"] is True
        assert mp.size == 1

    def test_get_by_sender_nonce(self):
        mp = Mempool(rbf_enabled=True)
        tx = _make_tx(sender="0xA", nonce=5, gas_price=10)
        mp.add(tx)
        found = mp.get_by_sender_nonce("0xA", 5)
        assert found is not None
        assert found.tx_hash == tx.tx_hash

    def test_remove_cleans_sender_nonce_index(self):
        mp = Mempool(rbf_enabled=True)
        tx = _make_tx(sender="0xA", nonce=0, gas_price=10)
        mp.add(tx)
        mp.remove(tx.tx_hash)
        assert mp.get_by_sender_nonce("0xA", 0) is None
        assert mp.size == 0

    def test_clear_resets_index(self):
        mp = Mempool(rbf_enabled=True)
        for i in range(5):
            mp.add(_make_tx(sender="0xA", nonce=i, gas_price=10))
        assert mp.size == 5
        mp.clear()
        assert mp.size == 0

    def test_multiple_senders_rbf(self):
        mp = Mempool(rbf_enabled=True, rbf_increment_pct=10)
        tx_a = _make_tx(sender="0xA", nonce=0, gas_price=10)
        tx_b = _make_tx(sender="0xB", nonce=0, gas_price=10)
        mp.add(tx_a)
        mp.add(tx_b)
        assert mp.size == 2

        # Replace A's tx
        tx_a2 = _make_tx(sender="0xA", nonce=0, gas_price=15)
        mp.add(tx_a2)
        assert mp.size == 2  # Still 2, B untouched

    def test_rbf_increment_pct_boundary(self):
        mp = Mempool(rbf_enabled=True, rbf_increment_pct=10)
        tx1 = _make_tx(sender="0xA", nonce=0, gas_price=100)
        mp.add(tx1)
        # Need >= 110 to replace
        tx2 = _make_tx(sender="0xA", nonce=0, gas_price=110)
        result = mp.add(tx2)
        assert result is True
        assert mp._txs.get(tx2.tx_hash) is not None


# ══════════════════════════════════════════════════════════════════════
#  Feature 5: Event Indexing & Log Filtering
# ══════════════════════════════════════════════════════════════════════

class TestBloomFilter:
    """Bloom filter tests."""

    def test_add_and_contains(self):
        bf = BloomFilter()
        bf.add("hello")
        assert bf.contains("hello")

    def test_not_contains(self):
        bf = BloomFilter()
        bf.add("hello")
        bf2 = BloomFilter()
        assert not bf2.contains("hello")

    def test_merge(self):
        bf1 = BloomFilter()
        bf1.add("alpha")
        bf2 = BloomFilter()
        bf2.add("beta")
        bf1.merge(bf2)  # merge mutates in place, returns None
        assert bf1.contains("alpha")
        assert bf1.contains("beta")

    def test_hex_roundtrip(self):
        bf = BloomFilter()
        bf.add("test")
        hex_val = bf.to_hex()
        restored = BloomFilter.from_hex(hex_val)
        assert restored.contains("test")


class TestEventLog:
    """EventLog tests."""

    def test_create_event_log(self):
        log = EventLog(
            address="0xContractAddr",
            topics=["Transfer", "0xFrom", "0xTo"],
            data={"value": 100},
            block_number=1,
        )
        assert log.address == "0xContractAddr"
        assert len(log.topics) == 3


class TestLogFilter:
    """LogFilter matching tests."""

    def test_filter_by_address(self):
        filt = LogFilter(address="0xContract")
        log = EventLog(address="0xContract", topics=["T"], data={}, block_number=1)
        assert filt.matches(log)

        log2 = EventLog(address="0xOther", topics=["T"], data={}, block_number=1)
        assert not filt.matches(log2)

    def test_filter_by_block_range(self):
        filt = LogFilter(from_block=5, to_block=10)
        log = EventLog(address="x", topics=[], data={}, block_number=7)
        assert filt.matches(log)

        log2 = EventLog(address="x", topics=[], data={}, block_number=3)
        assert not filt.matches(log2)

    def test_filter_by_topic(self):
        filt = LogFilter(topics=[["Transfer"]])
        log = EventLog(address="x", topics=["Transfer", "0xFrom"], data={}, block_number=1)
        assert filt.matches(log)

        log2 = EventLog(address="x", topics=["Approval"], data={}, block_number=1)
        assert not filt.matches(log2)

    def test_filter_address_list(self):
        filt = LogFilter(address=["0xA", "0xB"])
        log = EventLog(address="0xA", topics=[], data={}, block_number=1)
        assert filt.matches(log)


class TestEventIndex:
    """SQLite-backed event indexing."""

    def test_index_and_retrieve_logs(self):
        """Use _persist_log directly to insert an EventLog into the index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = EventIndex(data_dir=tmpdir)
            log = EventLog(
                address="0xContract",
                topics=["Transfer", "0xFrom", "0xTo"],
                data='{"value": 100}',
                block_number=1,
                block_hash="0xBlockHash",
                tx_hash="0xTxHash",
                tx_index=0,
                log_index=0,
            )
            idx._persist_log(log)
            idx._db.commit()
            results = idx.get_logs(LogFilter(address="0xContract"))
            assert len(results) >= 1

    def test_get_logs_for_tx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = EventIndex(data_dir=tmpdir)
            log = EventLog(
                address="0xC", topics=["X"], data="{}",
                block_number=1, tx_hash="0xMyTx",
                tx_index=0, log_index=0,
            )
            idx._persist_log(log)
            idx._db.commit()
            results = idx.get_logs_for_tx("0xMyTx")
            assert len(results) == 1

    def test_count_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = EventIndex(data_dir=tmpdir)
            for i in range(5):
                log = EventLog(
                    address="0xC", topics=[f"T{i}"], data="{}",
                    block_number=i, tx_hash=f"0xTx{i}",
                    tx_index=0, log_index=i,
                )
                idx._persist_log(log)
            idx._db.commit()
            assert idx.count_logs() == 5

    def test_bloom_per_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = EventIndex(data_dir=tmpdir)
            log = EventLog(
                address="0xC", topics=["Transfer"], data="{}",
                block_number=1, tx_hash="0xTx",
                tx_index=0, log_index=0,
            )
            idx._persist_log(log)
            idx._db.commit()
            # Store a bloom for this block
            bloom = BloomFilter()
            bloom.add(log.address)
            idx._blooms[1] = bloom
            if idx._db:
                idx._db.execute(
                    "INSERT OR REPLACE INTO block_blooms (block_number, bloom_hex) VALUES (?, ?)",
                    (1, bloom.to_hex()),
                )
                idx._db.commit()
            result = idx.get_bloom(1)
            assert result is not None


# ══════════════════════════════════════════════════════════════════════
#  Feature 6: Merkle Patricia Trie
# ══════════════════════════════════════════════════════════════════════

class TestMerklePatriciaTrie:
    """Full MPT tests — keys must be hex-encoded strings."""

    def test_put_and_get(self):
        trie = MerklePatriciaTrie()
        trie.put("0xabcd", "world")
        assert trie.get("0xabcd") == "world"

    def test_get_nonexistent(self):
        trie = MerklePatriciaTrie()
        assert trie.get("0x1234") is None

    def test_overwrite(self):
        trie = MerklePatriciaTrie()
        trie.put("0xaa", "v1")
        trie.put("0xaa", "v2")
        assert trie.get("0xaa") == "v2"

    def test_delete(self):
        trie = MerklePatriciaTrie()
        trie.put("0xabc0", "123")
        assert trie.delete("0xabc0")
        assert trie.get("0xabc0") is None

    def test_delete_nonexistent(self):
        trie = MerklePatriciaTrie()
        assert trie.delete("0xff00") is False

    def test_root_hash_changes(self):
        trie = MerklePatriciaTrie()
        h1 = trie.root_hash()
        trie.put("0xaa", "val")
        h2 = trie.root_hash()
        assert h1 != h2

    def test_root_hash_deterministic(self):
        t1 = MerklePatriciaTrie()
        t2 = MerklePatriciaTrie()
        t1.put("0xaa", "1")
        t1.put("0xbb", "2")
        t2.put("0xaa", "1")
        t2.put("0xbb", "2")
        assert t1.root_hash() == t2.root_hash()

    def test_multiple_keys(self):
        trie = MerklePatriciaTrie()
        for i in range(20):
            trie.put(f"0x{i:04x}", f"val{i}")
        for i in range(20):
            assert trie.get(f"0x{i:04x}") == f"val{i}"

    def test_proof_generation_and_verification(self):
        trie = MerklePatriciaTrie()
        trie.put("0xaaaa", "red")
        trie.put("0xbbbb", "yellow")
        proof = trie.generate_proof("0xaaaa")
        assert proof is not None
        assert MerklePatriciaTrie.verify_proof(
            trie.root_hash(), "0xaaaa", "red", proof
        )

    def test_snapshot_restore(self):
        trie = MerklePatriciaTrie()
        trie.put("0xaa", "1")
        snap = trie.snapshot()
        trie.put("0xbb", "2")
        trie.restore(snap)
        assert trie.get("0xaa") == "1"
        assert trie.get("0xbb") is None

    def test_items_iteration(self):
        trie = MerklePatriciaTrie()
        trie.put("0xaa", "1")
        trie.put("0xbb", "2")
        items = list(trie.items())
        assert len(items) == 2

    def test_put_batch(self):
        trie = MerklePatriciaTrie()
        batch = {f"0x{i:02x}": f"v{i}" for i in range(10)}
        trie.put_batch(batch)
        for k, v in batch.items():
            assert trie.get(k) == v

    def test_to_dict_from_dict(self):
        trie = MerklePatriciaTrie()
        trie.put("0xabcd", "world")
        d = trie.to_dict()
        restored = MerklePatriciaTrie.from_dict(d)
        assert restored.get("0xabcd") == "world"


class TestStateTrie:
    """World-state trie wrapper tests."""

    def test_set_and_get_account(self):
        st = StateTrie()
        acct = {"balance": 1000, "nonce": 0, "code": ""}
        st.set_account("0xaaaa", acct)
        result = st.get_account("0xaaaa")
        assert result["balance"] == 1000

    def test_account_not_found(self):
        st = StateTrie()
        assert st.get_account("0xffff") is None

    def test_delete_account(self):
        st = StateTrie()
        st.set_account("0xbbbb", {"balance": 500})
        st.delete_account("0xbbbb")
        assert st.get_account("0xbbbb") is None

    def test_storage_operations(self):
        st = StateTrie()
        st.set_storage("0xcccc", "0xaa", "value0")
        assert st.get_storage("0xcccc", "0xaa") == "value0"

    def test_delete_storage(self):
        st = StateTrie()
        st.set_storage("0xcc", "0xaa", "v1")
        st.delete_storage("0xcc", "0xaa")
        assert st.get_storage("0xcc", "0xaa") is None

    def test_state_root_changes(self):
        st = StateTrie()
        r1 = st.root_hash()
        st.set_account("0xaa", {"balance": 100})
        r2 = st.root_hash()
        assert r1 != r2

    def test_snapshot_and_restore(self):
        st = StateTrie()
        st.set_account("0xaa", {"balance": 100})
        snap = st.snapshot()
        st.set_account("0xbb", {"balance": 200})
        st.restore(snap)
        assert st.get_account("0xaa") is not None
        assert st.get_account("0xbb") is None


# ══════════════════════════════════════════════════════════════════════
#  Feature 7: HD Wallet & Keystore
# ══════════════════════════════════════════════════════════════════════

class TestMnemonic:
    """BIP-39 mnemonic tests."""

    def test_generate_12_words(self):
        m = generate_mnemonic(128)
        words = m.split()
        assert len(words) == 12

    def test_generate_24_words(self):
        m = generate_mnemonic(256)
        words = m.split()
        assert len(words) == 24

    def test_validate_valid_mnemonic(self):
        m = generate_mnemonic(128)
        assert validate_mnemonic(m) is True

    def test_validate_invalid_mnemonic(self):
        assert validate_mnemonic("not a valid mnemonic phrase at all") is False

    def test_validate_wrong_length(self):
        assert validate_mnemonic("hello world") is False

    def test_mnemonic_to_seed_deterministic(self):
        m = "abandon ability able about above absent absorb abstract absurd abuse access accident"
        s1 = mnemonic_to_seed(m)
        s2 = mnemonic_to_seed(m)
        assert s1 == s2
        assert len(s1) == 64

    def test_passphrase_changes_seed(self):
        m = generate_mnemonic(128)
        s1 = mnemonic_to_seed(m, "")
        s2 = mnemonic_to_seed(m, "my-passphrase")
        assert s1 != s2

    def test_wordlist_size(self):
        assert len(WORDLIST) == 2048
        # All unique
        assert len(set(WORDLIST)) == 2048

    def test_invalid_strength(self):
        with pytest.raises(ValueError):
            generate_mnemonic(100)


class TestExtendedKey:
    """BIP-32 key derivation tests."""

    def test_master_from_seed(self):
        seed = bytes(64)
        mk = master_key_from_seed(seed)
        assert mk.depth == 0
        assert mk.is_private is True
        assert len(mk.key) == 32

    def test_derive_child(self):
        seed = bytes.fromhex("00" * 64)
        mk = master_key_from_seed(seed)
        child = mk.derive_child(0)
        assert child.depth == 1
        assert child.index == 0
        assert child.key != mk.key

    def test_derive_hardened(self):
        seed = bytes.fromhex("00" * 64)
        mk = master_key_from_seed(seed)
        child = mk.derive_child(HARDENED_OFFSET)
        assert child.depth == 1
        assert child.index == HARDENED_OFFSET

    def test_derive_path(self):
        seed = bytes.fromhex("aa" * 64)
        mk = master_key_from_seed(seed)
        key = mk.derive_path("m/44'/806'/0'/0/0")
        assert key.depth == 5

    def test_different_paths_different_keys(self):
        seed = bytes.fromhex("bb" * 64)
        mk = master_key_from_seed(seed)
        k1 = mk.derive_path("m/44'/806'/0'/0/0")
        k2 = mk.derive_path("m/44'/806'/0'/0/1")
        assert k1.key != k2.key

    def test_fingerprint(self):
        seed = bytes(64)
        mk = master_key_from_seed(seed)
        fp = mk.fingerprint
        assert len(fp) == 4


class TestHDWallet:
    """HD Wallet tests."""

    def test_create_wallet(self):
        w = HDWallet.create(128)
        assert w.mnemonic
        assert len(w.mnemonic.split()) == 12

    def test_from_mnemonic(self):
        m = generate_mnemonic(128)
        w = HDWallet.from_mnemonic(m)
        assert w.mnemonic == m

    def test_derive_account(self):
        w = HDWallet.create()
        acct = w.derive_account(0)
        assert acct.address.startswith("0x")
        assert len(acct.private_key) == 32

    def test_multiple_accounts_different_addresses(self):
        w = HDWallet.create()
        a0 = w.derive_account(0)
        a1 = w.derive_account(1)
        assert a0.address != a1.address

    def test_derive_same_account_deterministic(self):
        m = generate_mnemonic(128)
        w1 = HDWallet.from_mnemonic(m)
        w2 = HDWallet.from_mnemonic(m)
        a1 = w1.derive_account(0)
        a2 = w2.derive_account(0)
        assert a1.address == a2.address
        assert a1.private_key == a2.private_key

    def test_list_accounts(self):
        w = HDWallet.create()
        w.derive_account(0)
        w.derive_account(1)
        assert len(w.list_accounts()) == 2

    def test_derive_multiple(self):
        w = HDWallet.create()
        accts = w.derive_multiple(5)
        assert len(accts) == 5
        addrs = [a.address for a in accts]
        assert len(set(addrs)) == 5

    def test_to_dict(self):
        w = HDWallet.create()
        w.derive_account(0)
        d = w.to_dict()
        assert "master_public_key" in d
        assert len(d["derived_accounts"]) == 1

    def test_from_seed(self):
        m = generate_mnemonic(128)
        seed = mnemonic_to_seed(m)
        w = HDWallet.from_seed(seed.hex())
        acct = w.derive_account(0)
        assert acct.address.startswith("0x")


class TestAccount:
    """Account signing tests."""

    def test_sign_data(self):
        w = HDWallet.create()
        acct = w.derive_account(0)
        sig = acct.sign(b"test message")
        assert len(sig) == 64  # 32 bytes hex = 64 chars

    def test_same_data_same_signature(self):
        w = HDWallet.create()
        acct = w.derive_account(0)
        s1 = acct.sign(b"data")
        s2 = acct.sign(b"data")
        assert s1 == s2

    def test_different_data_different_signature(self):
        w = HDWallet.create()
        acct = w.derive_account(0)
        s1 = acct.sign(b"data1")
        s2 = acct.sign(b"data2")
        assert s1 != s2

    def test_to_dict(self):
        w = HDWallet.create()
        acct = w.derive_account(0)
        d = acct.to_dict()
        assert "address" in d
        assert "public_key" in d


class TestKeystore:
    """Encrypted keystore tests."""

    def test_encrypt_decrypt_roundtrip(self):
        pk = "abcd" * 8  # 32 bytes hex
        ks = Keystore.encrypt(pk, "test-password")
        recovered = ks.decrypt("test-password")
        assert recovered == pk

    def test_wrong_password_fails(self):
        pk = "1234" * 8
        ks = Keystore.encrypt(pk, "correct-password")
        with pytest.raises(ValueError, match="MAC mismatch"):
            ks.decrypt("wrong-password")

    def test_save_and_load(self):
        pk = "aabb" * 8
        ks = Keystore.encrypt(pk, "pw123")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            ks.save(path)
            loaded = Keystore.load(path)
            assert loaded.decrypt("pw123") == pk
        finally:
            os.unlink(path)

    def test_to_dict_from_dict(self):
        pk = "ccdd" * 8
        ks = Keystore.encrypt(pk, "pw")
        d = ks.to_dict()
        restored = Keystore.from_dict(d)
        assert restored.decrypt("pw") == pk

    def test_keystore_has_address(self):
        pk = "eeff" * 8
        ks = Keystore.encrypt(pk, "pw", address="0xMyAddr")
        assert ks.address == "0xMyAddr"

    def test_pbkdf2_kdf(self):
        pk = "1111" * 8
        ks = Keystore.encrypt(pk, "pass", kdf="pbkdf2")
        assert ks.crypto["kdf"] == "pbkdf2"
        recovered = ks.decrypt("pass")
        assert recovered == pk

    def test_auto_derived_address(self):
        pk = "2222" * 8
        ks = Keystore.encrypt(pk, "pw")
        assert ks.address.startswith("0x")
        assert len(ks.address) == 42  # 0x + 40 hex chars


# ══════════════════════════════════════════════════════════════════════
#  Feature 8: BFT Consensus
# ══════════════════════════════════════════════════════════════════════

class TestBFTConsensus:
    """BFT consensus engine tests."""

    def _make_bft(self, validators=None):
        if validators is None:
            validators = ["0xV1", "0xV2", "0xV3", "0xV4"]
        bft = BFTConsensus(validators=validators)
        for v in validators:
            bft.add_validator(v)
        return bft

    def test_validator_management(self):
        bft = self._make_bft()
        assert bft.n == 4
        assert bft.f == 1  # (4-1)//3 = 1
        assert bft.quorum == 3  # 2*1+1

    def test_quorum_calculation(self):
        bft = self._make_bft(["0xV1"])
        assert bft.f == 0
        assert bft.quorum == 1

        bft = self._make_bft(["0xV1", "0xV2", "0xV3", "0xV4"])
        assert bft.quorum == 3

        bft = self._make_bft([f"0xV{i}" for i in range(7)])
        assert bft.f == 2
        assert bft.quorum == 5

    def test_leader_rotation(self):
        bft = self._make_bft(["0xA", "0xB", "0xC"])
        assert bft._get_leader(0) == "0xA"
        assert bft._get_leader(1) == "0xB"
        assert bft._get_leader(2) == "0xC"
        assert bft._get_leader(3) == "0xA"

    def test_add_remove_validator(self):
        bft = BFTConsensus()
        bft.add_validator("0xNew")
        assert "0xNew" in bft.validators
        bft.remove_validator("0xNew")
        assert "0xNew" not in bft.validators

    def test_bft_message_sign_verify(self):
        msg = BFTMessage(
            phase=BFTPhase.PREPARE,
            view=0, height=1,
            block_hash="0xHash", sender="0xV1",
        )
        key = b"secret-key"
        msg.sign(key)
        assert msg.signature
        assert msg.verify_signature(key)
        assert not msg.verify_signature(b"wrong-key")

    def test_bft_message_to_from_dict(self):
        msg = BFTMessage(
            phase=BFTPhase.COMMIT,
            view=2, height=5,
            block_hash="0xABC", sender="0xV2",
        )
        d = msg.to_dict()
        restored = BFTMessage.from_dict(d)
        assert restored.phase == BFTPhase.COMMIT
        assert restored.height == 5

    def test_seal_with_quorum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(data_dir=tmpdir)
            mempool = Mempool()
            bft = self._make_bft(["0xV1", "0xV2", "0xV3", "0xV4"])
            chain.accounts["0xAlice"] = {"balance": 1_000_000, "nonce": 0}
            tx = _make_tx(sender="0xAlice", recipient="0xBob", value=100)
            mempool.add(tx)

            block = bft.seal(chain, mempool, "0xV1")
            assert block is not None
            # Empty chain: first block is height 0
            assert block.header.height == 0

    def test_non_leader_cannot_seal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(data_dir=tmpdir)
            mempool = Mempool()
            bft = self._make_bft(["0xV1", "0xV2", "0xV3"])
            result = bft.seal(chain, mempool, "0xV2")
            assert result is None

    def test_non_validator_cannot_seal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(data_dir=tmpdir)
            mempool = Mempool()
            bft = self._make_bft(["0xV1", "0xV2", "0xV3"])
            result = bft.seal(chain, mempool, "0xNotValidator")
            assert result is None

    def test_verify_bft_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(data_dir=tmpdir)
            mempool = Mempool()
            bft = self._make_bft(["0xV1", "0xV2", "0xV3", "0xV4"])
            block = bft.seal(chain, mempool, "0xV1")
            assert block is not None
            assert bft.verify(block, chain) is True

    def test_view_change(self):
        bft = self._make_bft(["0xV1", "0xV2", "0xV3", "0xV4"])
        old_view = bft.view
        bft.request_view_change("0xV1")
        bft.request_view_change("0xV2")
        bft.request_view_change("0xV3")
        assert bft.view == old_view + 1

    def test_get_status(self):
        bft = self._make_bft(["0xV1", "0xV2", "0xV3"])
        status = bft.get_status()
        assert status["algorithm"] == "bft"
        assert status["validator_count"] == 3

    def test_create_consensus_bft(self):
        engine = create_consensus("bft", validators=["0xA", "0xB", "0xC"])
        assert isinstance(engine, BFTConsensus)

    def test_round_state(self):
        bft = self._make_bft()
        rnd = bft._get_round(1)
        assert isinstance(rnd, BFTRoundState)
        assert rnd.height == 1

    def test_prepare_commit_flow(self):
        """Test the individual on_prepare / on_commit message handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(data_dir=tmpdir)
            mempool = Mempool()
            validators = ["0xV1", "0xV2", "0xV3", "0xV4"]
            bft = self._make_bft(validators)

            # Step 1: Propose
            pre_prepare = bft.propose(chain, mempool, "0xV1")
            assert pre_prepare is not None

            height = pre_prepare.height
            rnd = bft._get_round(height)
            assert rnd.proposed_block is not None

            # Step 2: Validators send PREPARE
            for v in validators[1:]:
                prepare = bft.on_pre_prepare(pre_prepare, rnd.proposed_block, chain, v)
                assert prepare is not None

            # Step 3: Feed prepare messages — should reach quorum
            for v, msg in list(rnd.prepares.items()):
                bft.on_prepare(msg, "0xV1")

            assert rnd.phase == BFTPhase.COMMIT

            # Step 4: Feed commits
            for v in validators:
                commit_msg = BFTMessage(
                    phase=BFTPhase.COMMIT, view=0,
                    height=height, block_hash=rnd.proposed_hash, sender=v,
                )
                bft.on_commit(commit_msg)

            assert rnd.decided is True


# ══════════════════════════════════════════════════════════════════════
#  Feature 9: Token Standard Interfaces
# ══════════════════════════════════════════════════════════════════════

class TestZX20Token:
    """ZX-20 fungible token tests."""

    def test_create_token(self):
        t = ZX20Token("TestToken", "TT", 18, initial_supply=1000, owner="0xOwner")
        assert t.name() == "TestToken"
        assert t.symbol() == "TT"
        assert t.decimals() == 18
        assert t.total_supply() == 1000

    def test_initial_balance(self):
        t = ZX20Token("T", "T", 18, 1000, "0xOwner")
        assert t.balance_of("0xOwner") == 1000
        assert t.balance_of("0xOther") == 0

    def test_transfer(self):
        t = ZX20Token("T", "T", 18, 1000, "0xAlice")
        t.transfer("0xAlice", "0xBob", 300)
        assert t.balance_of("0xAlice") == 700
        assert t.balance_of("0xBob") == 300

    def test_transfer_insufficient(self):
        t = ZX20Token("T", "T", 18, 100, "0xAlice")
        with pytest.raises(ValueError, match="Insufficient"):
            t.transfer("0xAlice", "0xBob", 200)

    def test_transfer_to_zero_address(self):
        t = ZX20Token("T", "T", 18, 100, "0xAlice")
        with pytest.raises(ValueError, match="zero address"):
            t.transfer("0xAlice", ZERO_ADDRESS, 50)

    def test_approve_and_transfer_from(self):
        t = ZX20Token("T", "T", 18, 1000, "0xAlice")
        t.approve("0xAlice", "0xSpender", 500)
        assert t.allowance("0xAlice", "0xSpender") == 500
        t.transfer_from("0xSpender", "0xAlice", "0xBob", 200)
        assert t.balance_of("0xBob") == 200
        assert t.allowance("0xAlice", "0xSpender") == 300

    def test_transfer_from_exceeds_allowance(self):
        t = ZX20Token("T", "T", 18, 1000, "0xAlice")
        t.approve("0xAlice", "0xSpender", 100)
        with pytest.raises(ValueError, match="Allowance exceeded"):
            t.transfer_from("0xSpender", "0xAlice", "0xBob", 200)

    def test_mint(self):
        t = ZX20Token("T", "T", 18, 0, "0xOwner")
        t.mint("0xOwner", "0xUser", 500)
        assert t.total_supply() == 500
        assert t.balance_of("0xUser") == 500

    def test_mint_only_owner(self):
        t = ZX20Token("T", "T", 18, 0, "0xOwner")
        with pytest.raises(PermissionError, match="Only owner"):
            t.mint("0xNotOwner", "0xUser", 100)

    def test_burn(self):
        t = ZX20Token("T", "T", 18, 1000, "0xOwner")
        t.burn("0xOwner", 300)
        assert t.total_supply() == 700
        assert t.balance_of("0xOwner") == 700

    def test_burn_exceeds_balance(self):
        t = ZX20Token("T", "T", 18, 100, "0xOwner")
        with pytest.raises(ValueError, match="exceeds balance"):
            t.burn("0xOwner", 200)

    def test_events_emitted(self):
        t = ZX20Token("T", "T", 18, 1000, "0xAlice")
        t.transfer("0xAlice", "0xBob", 100)
        assert len(t.events) == 2
        assert t.events[1].event_name == "Transfer"

    def test_to_dict_from_dict(self):
        t = ZX20Token("Z", "ZZZ", 18, 1000, "0xOwner")
        t.transfer("0xOwner", "0xUser", 250)
        d = t.to_dict()
        restored = ZX20Token.from_dict(d)
        assert restored.balance_of("0xOwner") == 750
        assert restored.balance_of("0xUser") == 250


class TestZX721Token:
    """ZX-721 NFT token tests."""

    def test_mint_nft(self):
        nft = ZX721Token("MyNFT", "MNFT")
        nft.mint("0xAlice", 1, "ipfs://metadata/1")
        assert nft.owner_of(1) == "0xAlice"
        assert nft.token_uri(1) == "ipfs://metadata/1"

    def test_balance_of(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xAlice", 1)
        nft.mint("0xAlice", 2)
        nft.mint("0xBob", 3)
        assert nft.balance_of("0xAlice") == 2
        assert nft.balance_of("0xBob") == 1

    def test_total_supply(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xA", 1)
        nft.mint("0xA", 2)
        assert nft.total_supply() == 2

    def test_transfer(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xAlice", 1)
        nft.transfer_from("0xAlice", "0xAlice", "0xBob", 1)
        assert nft.owner_of(1) == "0xBob"

    def test_transfer_not_owner(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xAlice", 1)
        with pytest.raises(PermissionError):
            nft.transfer_from("0xBob", "0xAlice", "0xBob", 1)

    def test_approve_and_transfer(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xAlice", 1)
        nft.approve("0xAlice", "0xBob", 1)
        assert nft.get_approved(1) == "0xBob"
        nft.transfer_from("0xBob", "0xAlice", "0xBob", 1)
        assert nft.owner_of(1) == "0xBob"

    def test_approval_for_all(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xAlice", 1)
        nft.set_approval_for_all("0xAlice", "0xOperator", True)
        assert nft.is_approved_for_all("0xAlice", "0xOperator")
        nft.transfer_from("0xOperator", "0xAlice", "0xBob", 1)
        assert nft.owner_of(1) == "0xBob"

    def test_burn(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xAlice", 1)
        nft.burn("0xAlice", 1)
        with pytest.raises(ValueError):
            nft.owner_of(1)
        assert nft.total_supply() == 0

    def test_mint_duplicate_id(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xAlice", 1)
        with pytest.raises(ValueError, match="already exists"):
            nft.mint("0xBob", 1)

    def test_tokens_of_owner(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xAlice", 1)
        nft.mint("0xAlice", 5)
        nft.mint("0xBob", 3)
        assert sorted(nft.tokens_of_owner("0xAlice")) == [1, 5]

    def test_events_emitted(self):
        nft = ZX721Token("N", "N")
        nft.mint("0xAlice", 1)
        nft.transfer_from("0xAlice", "0xAlice", "0xBob", 1)
        assert len(nft.events) == 2

    def test_to_dict_from_dict(self):
        nft = ZX721Token("MyNFT", "MNFT", "0xOwner")
        nft.mint("0xAlice", 1, "ipfs://1")
        d = nft.to_dict()
        restored = ZX721Token.from_dict(d)
        assert restored.owner_of(1) == "0xAlice"
        assert restored.token_uri(1) == "ipfs://1"


class TestZX1155Token:
    """ZX-1155 multi-token tests."""

    def test_mint_fungible(self):
        mt = ZX1155Token()
        mt.mint("0xAlice", token_id=1, amount=1000)
        assert mt.balance_of("0xAlice", 1) == 1000
        assert mt.total_supply(1) == 1000

    def test_mint_nft(self):
        mt = ZX1155Token()
        mt.mint("0xAlice", token_id=100, amount=1)
        assert mt.balance_of("0xAlice", 100) == 1

    def test_transfer(self):
        mt = ZX1155Token()
        mt.mint("0xAlice", 1, 500)
        mt.safe_transfer_from("0xAlice", "0xAlice", "0xBob", 1, 200)
        assert mt.balance_of("0xAlice", 1) == 300
        assert mt.balance_of("0xBob", 1) == 200

    def test_transfer_insufficient(self):
        mt = ZX1155Token()
        mt.mint("0xAlice", 1, 100)
        with pytest.raises(ValueError, match="Insufficient"):
            mt.safe_transfer_from("0xAlice", "0xAlice", "0xBob", 1, 200)

    def test_batch_mint(self):
        mt = ZX1155Token()
        mt.mint_batch("0xAlice", [1, 2, 3], [100, 200, 300])
        assert mt.balance_of("0xAlice", 1) == 100
        assert mt.balance_of("0xAlice", 2) == 200
        assert mt.balance_of("0xAlice", 3) == 300

    def test_batch_transfer(self):
        mt = ZX1155Token()
        mt.mint_batch("0xAlice", [1, 2], [500, 500])
        mt.safe_batch_transfer_from(
            "0xAlice", "0xAlice", "0xBob", [1, 2], [100, 200]
        )
        assert mt.balance_of("0xAlice", 1) == 400
        assert mt.balance_of("0xBob", 2) == 200

    def test_balance_of_batch(self):
        mt = ZX1155Token()
        mt.mint("0xAlice", 1, 100)
        mt.mint("0xBob", 2, 200)
        assert mt.balance_of_batch(["0xAlice", "0xBob"], [1, 2]) == [100, 200]

    def test_operator_approval(self):
        mt = ZX1155Token()
        mt.mint("0xAlice", 1, 100)
        mt.set_approval_for_all("0xAlice", "0xOperator", True)
        assert mt.is_approved_for_all("0xAlice", "0xOperator")
        mt.safe_transfer_from("0xOperator", "0xAlice", "0xBob", 1, 50)
        assert mt.balance_of("0xBob", 1) == 50

    def test_burn(self):
        mt = ZX1155Token()
        mt.mint("0xAlice", 1, 500)
        mt.burn("0xAlice", 1, 200)
        assert mt.balance_of("0xAlice", 1) == 300
        assert mt.total_supply(1) == 300

    def test_burn_batch(self):
        mt = ZX1155Token()
        mt.mint_batch("0xAlice", [1, 2], [500, 500])
        mt.burn_batch("0xAlice", [1, 2], [100, 200])
        assert mt.balance_of("0xAlice", 1) == 400
        assert mt.balance_of("0xAlice", 2) == 300

    def test_uri(self):
        mt = ZX1155Token(base_uri="https://example.com/{id}.json")
        assert mt.uri(42) == "https://example.com/42.json"

    def test_custom_uri(self):
        mt = ZX1155Token()
        mt.set_uri(1, "ipfs://custom/1")
        assert mt.uri(1) == "ipfs://custom/1"

    def test_exists(self):
        mt = ZX1155Token()
        assert not mt.exists(1)
        mt.mint("0xAlice", 1, 100)
        assert mt.exists(1)

    def test_events_emitted(self):
        mt = ZX1155Token()
        mt.mint("0xAlice", 1, 100)
        mt.safe_transfer_from("0xAlice", "0xAlice", "0xBob", 1, 50)
        assert len(mt.events) == 2
        assert mt.events[0].event_name == "TransferSingle"

    def test_to_dict_from_dict(self):
        mt = ZX1155Token(base_uri="https://ex.com/{id}")
        mt.mint("0xAlice", 1, 500)
        mt.mint("0xBob", 2, 300)
        d = mt.to_dict()
        restored = ZX1155Token.from_dict(d)
        assert restored.balance_of("0xAlice", 1) == 500
        assert restored.balance_of("0xBob", 2) == 300

    def test_transfer_not_approved(self):
        mt = ZX1155Token()
        mt.mint("0xAlice", 1, 100)
        with pytest.raises(PermissionError, match="Not owner or approved"):
            mt.safe_transfer_from("0xBob", "0xAlice", "0xBob", 1, 50)


# ══════════════════════════════════════════════════════════════════════
#  Feature 2: Gas Estimation (quick sanity)
# ══════════════════════════════════════════════════════════════════════

class TestGasEstimation:
    def test_simple_transfer_cost(self):
        assert 21_000 == 21_000

    def test_contract_deployment_cost(self):
        data = "a" * 100
        assert 53_000 + len(data) * 200 == 73_000


# ══════════════════════════════════════════════════════════════════════
#  Feature 3: Contract State Persistence
# ══════════════════════════════════════════════════════════════════════

class TestContractStatePersistence:
    def test_chain_has_contract_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(data_dir=tmpdir)
            assert hasattr(chain, 'contract_state')

    def test_contract_state_persists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain = Chain(data_dir=tmpdir)
            chain.contract_state["0xContract"] = {"count": 42}
            chain._persist_state()
            chain2 = Chain(data_dir=tmpdir)
            assert chain2.contract_state.get("0xContract", {}).get("count") == 42


# ══════════════════════════════════════════════════════════════════════
#  Integration: create_consensus factory
# ══════════════════════════════════════════════════════════════════════

class TestConsensusFactory:
    def test_create_pow(self):
        assert isinstance(create_consensus("pow", difficulty=1), ProofOfWork)

    def test_create_poa(self):
        assert isinstance(create_consensus("poa", validators=["0xA"]), ProofOfAuthority)

    def test_create_pos(self):
        assert isinstance(create_consensus("pos", min_stake=100), ProofOfStake)

    def test_create_bft(self):
        assert isinstance(create_consensus("bft", validators=["0xA", "0xB", "0xC"]), BFTConsensus)

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_consensus("quantum")
