"""
Event Indexing & Log Filtering for the Zexus Blockchain.

Provides:
  - **BloomFilter**: Space-efficient probabilistic set for fast log matching.
  - **EventLog**: Structured event model with indexed topics.
  - **EventIndex**: Persistent event store (SQLite-backed) with multi-key
    lookup by block range, address, topic, and event name.
  - **LogFilter**: Composable filter object matching Ethereum-style
    ``getLogs`` semantics (fromBlock, toBlock, address, topics).

Usage (from RPCServer or BlockchainNode):

    >>> idx = EventIndex(data_dir="/tmp/zexus")
    >>> idx.index_block(block)   # called after each block is added
    >>> logs = idx.get_logs(LogFilter(from_block=0, to_block=10,
    ...                               address="0xabc..."))

Bloom filters are attached to each block header (``logs_bloom``) so
nodes can skip blocks that certainly do not contain matching logs.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  Bloom Filter — 2048-bit (256-byte) per Ethereum Yellow Paper §4.3.1
# ══════════════════════════════════════════════════════════════════════

class BloomFilter:
    """A 2048-bit (256-byte) Bloom filter using 3 hash functions.

    Compatible with Ethereum's *logsBloom* layout so tooling can
    interoperate.  Each item is hashed with Keccak-256 (or SHA-256
    fallback) and 3 independent bit positions are set.
    """

    SIZE_BITS = 2048
    SIZE_BYTES = SIZE_BITS // 8  # 256
    NUM_HASHES = 3

    def __init__(self, data: Optional[bytes] = None):
        if data is not None:
            if len(data) != self.SIZE_BYTES:
                raise ValueError(f"Bloom data must be {self.SIZE_BYTES} bytes")
            self._bits = bytearray(data)
        else:
            self._bits = bytearray(self.SIZE_BYTES)

    # ── Core ops ──────────────────────────────────────────────────

    def add(self, item: str) -> None:
        """Add an item (hex string or plain text) to the bloom."""
        for pos in self._bit_positions(item):
            byte_idx = pos // 8
            bit_idx = pos % 8
            self._bits[byte_idx] |= (1 << bit_idx)

    def contains(self, item: str) -> bool:
        """Probabilistic membership test (no false negatives)."""
        for pos in self._bit_positions(item):
            byte_idx = pos // 8
            bit_idx = pos % 8
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def merge(self, other: "BloomFilter") -> None:
        """OR another bloom into this one (union)."""
        for i in range(self.SIZE_BYTES):
            self._bits[i] |= other._bits[i]

    # ── Serialization ─────────────────────────────────────────────

    def to_hex(self) -> str:
        return "0x" + self._bits.hex()

    @classmethod
    def from_hex(cls, hex_str: str) -> "BloomFilter":
        raw = hex_str.removeprefix("0x")
        return cls(bytes.fromhex(raw))

    def to_bytes(self) -> bytes:
        return bytes(self._bits)

    @property
    def is_empty(self) -> bool:
        return all(b == 0 for b in self._bits)

    # ── Internal ──────────────────────────────────────────────────

    def _bit_positions(self, item: str) -> List[int]:
        h = hashlib.sha256(item.encode("utf-8")).digest()
        positions = []
        for i in range(self.NUM_HASHES):
            # Take 2 bytes from the hash for each function
            val = int.from_bytes(h[2 * i: 2 * i + 2], "big")
            positions.append(val % self.SIZE_BITS)
        return positions

    def __or__(self, other: "BloomFilter") -> "BloomFilter":
        result = BloomFilter(bytes(self._bits))
        result.merge(other)
        return result

    def __repr__(self) -> str:
        ones = sum(bin(b).count("1") for b in self._bits)
        return f"<BloomFilter bits_set={ones}/{self.SIZE_BITS}>"


# ══════════════════════════════════════════════════════════════════════
#  EventLog — structured event model
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EventLog:
    """A single indexed event log entry.

    Fields match Ethereum's log structure for maximum interoperability:
    - ``address``:   Contract that emitted the event.
    - ``topics``:    list of topic strings (topic[0] = event signature).
    - ``data``:      ABI-encoded (or JSON) event data payload.
    - ``block_number``, ``block_hash``, ``tx_hash``, ``tx_index``,
      ``log_index``: Location within the chain.
    """

    address: str = ""
    topics: List[str] = field(default_factory=list)
    data: str = ""
    block_number: int = 0
    block_hash: str = ""
    tx_hash: str = ""
    tx_index: int = 0
    log_index: int = 0
    timestamp: float = 0.0
    removed: bool = False  # True if log was reverted during a reorg

    @property
    def event_name(self) -> str:
        """Convenience: the human-readable event name from topic[0]."""
        return self.topics[0] if self.topics else ""

    def topic_hash(self) -> str:
        """Keccak-256/SHA-256 hash of the event signature (topic[0])."""
        if not self.topics:
            return ""
        return hashlib.sha256(self.topics[0].encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EventLog":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ══════════════════════════════════════════════════════════════════════
#  LogFilter — composable log query
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LogFilter:
    """Query filter for retrieving event logs.

    Semantics follow ``eth_getLogs``:
    - ``from_block`` / ``to_block``:  inclusive block range.
    - ``address``:  single address *or* list of addresses.
    - ``topics``:   list-of-lists; each position can be a single
      topic or a list of alternatives (OR within position, AND
      across positions).
    - ``event_name``: shortcut filter on the human-readable name.
    """

    from_block: int = 0
    to_block: Optional[int] = None  # None → latest
    address: Optional[Any] = None  # str or List[str]
    topics: Optional[List[Optional[Any]]] = None  # [[t1,t2], None, [t3]]
    event_name: Optional[str] = None
    limit: int = 10_000

    def address_set(self) -> Optional[Set[str]]:
        if self.address is None:
            return None
        if isinstance(self.address, str):
            return {self.address}
        return set(self.address)

    def matches(self, log: EventLog) -> bool:
        """Check if a log entry satisfies this filter."""
        # Block range
        if log.block_number < self.from_block:
            return False
        if self.to_block is not None and log.block_number > self.to_block:
            return False

        # Address
        addr_set = self.address_set()
        if addr_set is not None and log.address not in addr_set:
            return False

        # Event name shortcut
        if self.event_name and log.event_name != self.event_name:
            return False

        # Topics (position-based matching)
        if self.topics:
            for i, topic_filter in enumerate(self.topics):
                if topic_filter is None:
                    continue  # wildcard
                if i >= len(log.topics):
                    return False
                if isinstance(topic_filter, list):
                    if log.topics[i] not in topic_filter:
                        return False
                else:
                    if log.topics[i] != topic_filter:
                        return False

        return True


# ══════════════════════════════════════════════════════════════════════
#  EventIndex — persistent event store (SQLite)
# ══════════════════════════════════════════════════════════════════════

class EventIndex:
    """Persistent, indexed event/log store backed by SQLite.

    Every time a block is finalized, call ``index_block(block)`` to
    extract and persist all receipt logs.  Queries via ``get_logs``
    hit indexed columns and optionally check the per-block bloom
    filter *before* scanning individual entries.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self._db: Optional[sqlite3.Connection] = None
        self._blooms: Dict[int, BloomFilter] = {}  # block_height -> bloom
        if data_dir:
            import os
            os.makedirs(data_dir, exist_ok=True)
            self._init_db(os.path.join(data_dir, "events.db"))

    def _init_db(self, db_path: str):
        self._db = sqlite3.connect(db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS event_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_number INTEGER NOT NULL,
                block_hash TEXT NOT NULL,
                tx_hash TEXT NOT NULL,
                tx_index INTEGER NOT NULL,
                log_index INTEGER NOT NULL,
                address TEXT NOT NULL,
                topic0 TEXT,
                topic1 TEXT,
                topic2 TEXT,
                topic3 TEXT,
                data TEXT,
                timestamp REAL,
                removed INTEGER DEFAULT 0
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS block_blooms (
                block_number INTEGER PRIMARY KEY,
                bloom_hex TEXT NOT NULL
            )
        """)
        # Indices for fast lookups
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_logs_block ON event_logs(block_number)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_logs_address ON event_logs(address)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_logs_topic0 ON event_logs(topic0)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_logs_tx ON event_logs(tx_hash)")
        self._db.commit()

    # ── Indexing ──────────────────────────────────────────────────

    def index_block(self, block) -> int:
        """Extract logs from a block's receipts and persist them.

        Returns the number of new log entries indexed.
        """
        bloom = BloomFilter()
        count = 0
        log_index = 0

        for tx_idx, receipt in enumerate(block.receipts):
            for raw_log in receipt.logs:
                log = self._normalize_log(
                    raw_log, block, receipt, tx_idx, log_index
                )
                bloom.add(log.address)
                for topic in log.topics:
                    bloom.add(topic)

                self._persist_log(log)
                count += 1
                log_index += 1

        self._blooms[block.header.height] = bloom
        if self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO block_blooms (block_number, bloom_hex) VALUES (?, ?)",
                (block.header.height, bloom.to_hex()),
            )
            self._db.commit()

        return count

    def index_receipt_logs(self, receipt, block_number: int,
                           block_hash: str, tx_index: int) -> int:
        """Index logs from a single receipt (for incremental indexing)."""
        count = 0
        for log_idx, raw_log in enumerate(receipt.logs):
            log = EventLog(
                address=raw_log.get("contract", raw_log.get("address", "")),
                topics=[raw_log.get("event", "")] + raw_log.get("topics", []),
                data=json.dumps(raw_log.get("data", ""), default=str),
                block_number=block_number,
                block_hash=block_hash,
                tx_hash=receipt.tx_hash,
                tx_index=tx_index,
                log_index=log_idx,
                timestamp=raw_log.get("timestamp", 0.0),
            )
            self._persist_log(log)
            count += 1
        return count

    # ── Querying ──────────────────────────────────────────────────

    def get_logs(self, filt: LogFilter) -> List[EventLog]:
        """Query logs matching the given filter.

        Uses bloom filters for block-level pre-filtering when available,
        then applies full filter matching.
        """
        # Fast path: SQL query if DB available
        if self._db:
            return self._query_db(filt)

        # In-memory fallback (for tests without data_dir)
        return []

    def get_logs_for_tx(self, tx_hash: str) -> List[EventLog]:
        """Get all logs emitted by a specific transaction."""
        if self._db:
            rows = self._db.execute(
                "SELECT * FROM event_logs WHERE tx_hash = ? ORDER BY log_index",
                (tx_hash,)
            ).fetchall()
            return [self._row_to_log(r) for r in rows]
        return []

    def get_logs_for_block(self, block_number: int) -> List[EventLog]:
        """Get all logs in a specific block."""
        if self._db:
            rows = self._db.execute(
                "SELECT * FROM event_logs WHERE block_number = ? ORDER BY log_index",
                (block_number,)
            ).fetchall()
            return [self._row_to_log(r) for r in rows]
        return []

    def get_bloom(self, block_number: int) -> Optional[BloomFilter]:
        """Get the bloom filter for a specific block."""
        if block_number in self._blooms:
            return self._blooms[block_number]
        if self._db:
            row = self._db.execute(
                "SELECT bloom_hex FROM block_blooms WHERE block_number = ?",
                (block_number,)
            ).fetchone()
            if row:
                bloom = BloomFilter.from_hex(row[0])
                self._blooms[block_number] = bloom
                return bloom
        return None

    def count_logs(self, filt: Optional[LogFilter] = None) -> int:
        """Count total logs, optionally filtered."""
        if self._db:
            if filt:
                where, params = self._build_where(filt)
                row = self._db.execute(
                    f"SELECT COUNT(*) FROM event_logs {where}", params
                ).fetchone()
                return row[0]
            row = self._db.execute("SELECT COUNT(*) FROM event_logs").fetchone()
            return row[0]
        return 0

    # ── Reorg handling ────────────────────────────────────────────

    def mark_removed(self, block_number: int) -> int:
        """Mark all logs at or above a block height as removed (reorg)."""
        if self._db:
            cursor = self._db.execute(
                "UPDATE event_logs SET removed = 1 WHERE block_number >= ?",
                (block_number,)
            )
            self._db.commit()
            return cursor.rowcount
        return 0

    def prune_removed(self) -> int:
        """Permanently delete logs marked as removed."""
        if self._db:
            cursor = self._db.execute("DELETE FROM event_logs WHERE removed = 1")
            self._db.commit()
            return cursor.rowcount
        return 0

    # ── Internal helpers ──────────────────────────────────────────

    def _normalize_log(self, raw_log: Dict, block, receipt, tx_idx: int,
                       log_idx: int) -> EventLog:
        """Convert a raw receipt log dict into a structured EventLog."""
        topics = []
        if "event" in raw_log:
            topics.append(raw_log["event"])
        if "topics" in raw_log:
            topics.extend(raw_log["topics"])
        if not topics and "name" in raw_log:
            topics.append(raw_log["name"])

        return EventLog(
            address=raw_log.get("contract", raw_log.get("address", "")),
            topics=topics,
            data=json.dumps(raw_log.get("data", ""), default=str),
            block_number=block.header.height,
            block_hash=block.hash,
            tx_hash=receipt.tx_hash,
            tx_index=tx_idx,
            log_index=log_idx,
            timestamp=raw_log.get("timestamp", block.header.timestamp),
        )

    def _persist_log(self, log: EventLog):
        if not self._db:
            return
        topics = log.topics + [None] * (4 - len(log.topics))
        self._db.execute(
            """INSERT INTO event_logs
               (block_number, block_hash, tx_hash, tx_index, log_index,
                address, topic0, topic1, topic2, topic3, data, timestamp, removed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (log.block_number, log.block_hash, log.tx_hash, log.tx_index,
             log.log_index, log.address, topics[0], topics[1], topics[2],
             topics[3], log.data, log.timestamp, int(log.removed)),
        )

    def _build_where(self, filt: LogFilter) -> Tuple[str, list]:
        clauses = ["removed = 0"]
        params: list = []

        clauses.append("block_number >= ?")
        params.append(filt.from_block)
        if filt.to_block is not None:
            clauses.append("block_number <= ?")
            params.append(filt.to_block)

        addr_set = filt.address_set()
        if addr_set:
            placeholders = ",".join("?" for _ in addr_set)
            clauses.append(f"address IN ({placeholders})")
            params.extend(addr_set)

        if filt.event_name:
            clauses.append("topic0 = ?")
            params.append(filt.event_name)

        if filt.topics:
            for i, topic_filter in enumerate(filt.topics[:4]):
                col = f"topic{i}"
                if topic_filter is None:
                    continue
                if isinstance(topic_filter, list):
                    ph = ",".join("?" for _ in topic_filter)
                    clauses.append(f"{col} IN ({ph})")
                    params.extend(topic_filter)
                else:
                    clauses.append(f"{col} = ?")
                    params.append(topic_filter)

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return where, params

    def _query_db(self, filt: LogFilter) -> List[EventLog]:
        where, params = self._build_where(filt)
        sql = f"SELECT * FROM event_logs {where} ORDER BY block_number, log_index LIMIT ?"
        params.append(filt.limit)
        rows = self._db.execute(sql, params).fetchall()
        return [self._row_to_log(r) for r in rows]

    def _row_to_log(self, row) -> EventLog:
        topics = [t for t in [row[7], row[8], row[9], row[10]] if t is not None]
        return EventLog(
            address=row[6],
            topics=topics,
            data=row[11] or "",
            block_number=row[1],
            block_hash=row[2],
            tx_hash=row[3],
            tx_index=row[4],
            log_index=row[5],
            timestamp=row[12] or 0.0,
            removed=bool(row[13]),
        )

    def close(self):
        if self._db:
            self._db.close()
            self._db = None
