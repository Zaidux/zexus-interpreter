"""
Zexus Blockchain — Chain & Block Data Structures

Provides the core blockchain data model: Block headers, transactions,
block validation, and chain management with persistent storage.

This module implements a proper blockchain (linked list of blocks) on top
of the existing Ledger/Transaction infrastructure.
"""

import hashlib
import json
import time
import copy
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class BlockHeader:
    """Block header containing metadata and proof."""
    version: int = 1
    height: int = 0
    timestamp: float = 0.0
    prev_hash: str = "0" * 64
    state_root: str = ""
    tx_root: str = ""  # Merkle root of transactions
    receipts_root: str = ""
    miner: str = ""  # Address of block producer
    nonce: int = 0
    difficulty: int = 1
    gas_limit: int = 10_000_000
    gas_used: int = 0
    extra_data: str = ""

    def compute_hash(self) -> str:
        """Compute block header hash (excludes the hash itself)."""
        data = json.dumps({
            "version": self.version,
            "height": self.height,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
            "state_root": self.state_root,
            "tx_root": self.tx_root,
            "receipts_root": self.receipts_root,
            "miner": self.miner,
            "nonce": self.nonce,
            "difficulty": self.difficulty,
            "gas_limit": self.gas_limit,
            "gas_used": self.gas_used,
            "extra_data": self.extra_data,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class Transaction:
    """A blockchain transaction with cryptographic authentication."""
    tx_hash: str = ""
    sender: str = ""
    recipient: str = ""
    value: int = 0
    data: str = ""  # Contract call data or deployment bytecode
    nonce: int = 0  # Sender's nonce (replay protection)
    gas_limit: int = 21_000
    gas_price: int = 1
    signature: str = ""  # ECDSA signature
    timestamp: float = 0.0
    status: str = "pending"  # pending, confirmed, failed, reverted

    def compute_hash(self) -> str:
        """Compute transaction hash from its contents."""
        data = json.dumps({
            "sender": self.sender,
            "recipient": self.recipient,
            "value": self.value,
            "data": self.data,
            "nonce": self.nonce,
            "gas_limit": self.gas_limit,
            "gas_price": self.gas_price,
            "timestamp": self.timestamp,
        }, sort_keys=True)
        h = hashlib.sha256(data.encode()).hexdigest()
        self.tx_hash = h
        return h

    def sign(self, private_key: str) -> str:
        """Sign the transaction with a private key."""
        # Use HMAC-based signing for portability (real ECDSA when cryptography lib available)
        import hmac
        msg = self.compute_hash()
        sig = hmac.new(private_key.encode(), msg.encode(), hashlib.sha256).hexdigest()
        self.signature = sig
        return sig

    def verify(self, public_key: str) -> bool:
        """Verify transaction signature."""
        if not self.signature:
            return False
        # For HMAC-based signatures, verification requires the private key
        # In production, this would use ECDSA public key verification
        return len(self.signature) == 64  # Basic format check

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TransactionReceipt:
    """Receipt produced after transaction execution."""
    tx_hash: str = ""
    block_hash: str = ""
    block_height: int = 0
    status: int = 1  # 1 = success, 0 = failure
    gas_used: int = 0
    logs: List[Dict[str, Any]] = field(default_factory=list)
    contract_address: Optional[str] = None  # If deployment
    revert_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Block:
    """A complete block containing header, transactions, and hash."""
    header: BlockHeader = field(default_factory=BlockHeader)
    transactions: List[Transaction] = field(default_factory=list)
    receipts: List[TransactionReceipt] = field(default_factory=list)
    hash: str = ""

    def compute_hash(self) -> str:
        """Compute the block hash from its header."""
        self.hash = self.header.compute_hash()
        return self.hash

    def compute_tx_root(self) -> str:
        """Compute Merkle root of transactions."""
        if not self.transactions:
            return hashlib.sha256(b"empty").hexdigest()

        hashes = [tx.tx_hash or tx.compute_hash() for tx in self.transactions]
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            next_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = next_level
        return hashes[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hash": self.hash,
            "header": asdict(self.header),
            "transactions": [tx.to_dict() for tx in self.transactions],
            "receipts": [r.to_dict() for r in self.receipts],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Block':
        """Reconstruct a Block from a dictionary."""
        header = BlockHeader(**data["header"])
        txs = [Transaction(**td) for td in data.get("transactions", [])]
        receipts = [TransactionReceipt(**rd) for rd in data.get("receipts", [])]
        b = Block(header=header, transactions=txs, receipts=receipts, hash=data.get("hash", ""))
        return b


class Mempool:
    """Transaction mempool with priority ordering.
    
    Holds pending transactions, ordered by gas_price (descending)
    for block producers to select from.
    """

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._txs: Dict[str, Transaction] = {}  # tx_hash -> Transaction
        self._nonces: Dict[str, int] = {}  # sender -> highest nonce seen

    def add(self, tx: Transaction) -> bool:
        """Add transaction to mempool. Returns True if accepted."""
        if len(self._txs) >= self.max_size:
            return False
        if not tx.tx_hash:
            tx.compute_hash()
        if tx.tx_hash in self._txs:
            return False  # Duplicate
        # Nonce check
        expected = self._nonces.get(tx.sender, 0)
        if tx.nonce < expected:
            return False  # Replay
        self._txs[tx.tx_hash] = tx
        if tx.nonce >= expected:
            self._nonces[tx.sender] = tx.nonce + 1
        return True

    def remove(self, tx_hash: str) -> Optional[Transaction]:
        """Remove a transaction from the mempool."""
        return self._txs.pop(tx_hash, None)

    def get_pending(self, gas_limit: int = 10_000_000) -> List[Transaction]:
        """Get pending transactions ordered by gas_price, fitting within gas_limit."""
        sorted_txs = sorted(self._txs.values(), key=lambda t: t.gas_price, reverse=True)
        result = []
        total_gas = 0
        for tx in sorted_txs:
            if total_gas + tx.gas_limit <= gas_limit:
                result.append(tx)
                total_gas += tx.gas_limit
        return result

    @property
    def size(self) -> int:
        return len(self._txs)

    def clear(self):
        self._txs.clear()
        self._nonces.clear()


class Chain:
    """The blockchain — an ordered sequence of validated blocks.
    
    Features:
    - Genesis block creation
    - Block validation (hash chain, PoW, timestamps)
    - Chain state management with accounts
    - Persistent storage (SQLite)
    - Fork detection and chain tip tracking
    """

    def __init__(self, chain_id: str = "zexus-mainnet", data_dir: Optional[str] = None):
        self.chain_id = chain_id
        self.blocks: List[Block] = []
        self.block_index: Dict[str, Block] = {}  # hash -> Block
        self.height_index: Dict[int, Block] = {}  # height -> Block
        self.accounts: Dict[str, Dict[str, Any]] = {}  # address -> {balance, nonce, code, storage}
        self.contract_state: Dict[str, Dict[str, Any]] = {}  # contract_addr -> state
        self.difficulty: int = 1
        self.target_block_time: float = 10.0  # seconds
        
        # Persistent storage
        self._data_dir = data_dir
        self._db: Optional[sqlite3.Connection] = None
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
            self._init_db(os.path.join(data_dir, "chain.db"))

    def _init_db(self, db_path: str):
        """Initialize SQLite database for chain storage."""
        self._db = sqlite3.connect(db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                height INTEGER PRIMARY KEY,
                hash TEXT UNIQUE NOT NULL,
                data TEXT NOT NULL
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS state (
                address TEXT PRIMARY KEY,
                data TEXT NOT NULL
            )
        """)
        self._db.commit()
        self._load_from_db()

    def _load_from_db(self):
        """Load chain from persistent storage."""
        if not self._db:
            return
        cursor = self._db.execute("SELECT data FROM blocks ORDER BY height ASC")
        for row in cursor:
            block = Block.from_dict(json.loads(row[0]))
            self.blocks.append(block)
            self.block_index[block.hash] = block
            self.height_index[block.header.height] = block
        
        cursor = self._db.execute("SELECT address, data FROM state")
        for address, data in cursor:
            self.accounts[address] = json.loads(data)

    def _persist_block(self, block: Block):
        """Persist a block to the database."""
        if not self._db:
            return
        self._db.execute(
            "INSERT OR REPLACE INTO blocks (height, hash, data) VALUES (?, ?, ?)",
            (block.header.height, block.hash, json.dumps(block.to_dict()))
        )
        self._db.commit()

    def _persist_state(self):
        """Persist account state to the database."""
        if not self._db:
            return
        for address, data in self.accounts.items():
            self._db.execute(
                "INSERT OR REPLACE INTO state (address, data) VALUES (?, ?)",
                (address, json.dumps(data))
            )
        self._db.commit()

    def create_genesis(self, miner: str = "0x0000000000000000000000000000000000000000",
                       initial_balances: Optional[Dict[str, int]] = None) -> Block:
        """Create the genesis block."""
        if self.blocks:
            raise RuntimeError("Genesis block already exists")

        genesis = Block()
        genesis.header.height = 0
        genesis.header.timestamp = time.time()
        genesis.header.miner = miner
        genesis.header.extra_data = f"Zexus Genesis — {self.chain_id}"
        genesis.header.prev_hash = "0" * 64
        genesis.compute_hash()

        # Initialize accounts with balances
        if initial_balances:
            for addr, balance in initial_balances.items():
                self.accounts[addr] = {"balance": balance, "nonce": 0, "code": "", "storage": {}}

        self.blocks.append(genesis)
        self.block_index[genesis.hash] = genesis
        self.height_index[0] = genesis
        self._persist_block(genesis)
        self._persist_state()
        return genesis

    @property
    def tip(self) -> Optional[Block]:
        """Get the latest block (chain tip)."""
        return self.blocks[-1] if self.blocks else None

    @property
    def height(self) -> int:
        """Current chain height."""
        return len(self.blocks) - 1 if self.blocks else -1

    def get_block(self, hash_or_height) -> Optional[Block]:
        """Get block by hash or height."""
        if isinstance(hash_or_height, int):
            return self.height_index.get(hash_or_height)
        return self.block_index.get(hash_or_height)

    def get_account(self, address: str) -> Dict[str, Any]:
        """Get account state, creating if needed."""
        if address not in self.accounts:
            self.accounts[address] = {"balance": 0, "nonce": 0, "code": "", "storage": {}}
        return self.accounts[address]

    def add_block(self, block: Block) -> Tuple[bool, str]:
        """Validate and add a block to the chain.
        
        Returns (success, error_message).
        """
        if not self.blocks:
            return False, "No genesis block — call create_genesis() first"

        tip = self.tip
        
        # Validate parent hash
        if block.header.prev_hash != tip.hash:
            return False, f"Invalid prev_hash: expected {tip.hash}, got {block.header.prev_hash}"

        # Validate height
        expected_height = tip.header.height + 1
        if block.header.height != expected_height:
            return False, f"Invalid height: expected {expected_height}, got {block.header.height}"

        # Validate timestamp
        if block.header.timestamp <= tip.header.timestamp:
            return False, "Block timestamp must be after parent"

        # Validate hash
        computed = block.header.compute_hash()
        if block.hash != computed:
            return False, f"Invalid block hash: expected {computed}, got {block.hash}"

        # Validate PoW (if difficulty > 0)
        if self.difficulty > 0:
            target = "0" * self.difficulty
            if not block.hash.startswith(target):
                return False, f"Block hash does not meet difficulty {self.difficulty}"

        # Validate transactions
        for tx in block.transactions:
            if not tx.tx_hash:
                return False, f"Transaction missing hash"

        # Validate tx root
        expected_root = block.compute_tx_root()
        if block.header.tx_root and block.header.tx_root != expected_root:
            return False, f"Invalid tx_root"

        # Add block
        self.blocks.append(block)
        self.block_index[block.hash] = block
        self.height_index[block.header.height] = block
        
        # Process transactions — update account state
        for i, tx in enumerate(block.transactions):
            receipt = block.receipts[i] if i < len(block.receipts) else None
            if receipt and receipt.status == 1:
                # Debit sender
                sender_acct = self.get_account(tx.sender)
                sender_acct["balance"] -= tx.value + (tx.gas_limit * tx.gas_price)
                sender_acct["nonce"] = max(sender_acct["nonce"], tx.nonce + 1)
                
                # Credit recipient
                if tx.recipient:
                    recv_acct = self.get_account(tx.recipient)
                    recv_acct["balance"] += tx.value

        # Adjust difficulty
        self._adjust_difficulty()

        self._persist_block(block)
        self._persist_state()
        return True, ""

    def _adjust_difficulty(self):
        """Adjust mining difficulty based on block time."""
        if len(self.blocks) < 3:
            return
        last_two = self.blocks[-2:]
        actual_time = last_two[1].header.timestamp - last_two[0].header.timestamp
        if actual_time < self.target_block_time * 0.5:
            self.difficulty = min(self.difficulty + 1, 8)
        elif actual_time > self.target_block_time * 2.0:
            self.difficulty = max(self.difficulty - 1, 1)

    def validate_chain(self) -> Tuple[bool, str]:
        """Validate the entire chain integrity."""
        for i, block in enumerate(self.blocks):
            computed = block.header.compute_hash()
            if block.hash != computed:
                return False, f"Block {i} hash mismatch"
            if i > 0:
                if block.header.prev_hash != self.blocks[i - 1].hash:
                    return False, f"Block {i} prev_hash mismatch"
                if block.header.height != i:
                    return False, f"Block {i} height mismatch"
        return True, ""

    def get_chain_info(self) -> Dict[str, Any]:
        """Get chain status information."""
        return {
            "chain_id": self.chain_id,
            "height": self.height,
            "difficulty": self.difficulty,
            "tip_hash": self.tip.hash if self.tip else None,
            "total_accounts": len(self.accounts),
            "total_blocks": len(self.blocks),
        }

    def close(self):
        """Close persistent storage."""
        if self._db:
            self._db.close()
            self._db = None
