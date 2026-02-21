"""
Zexus Blockchain — Pluggable Storage Backends

Provides an abstract StorageBackend interface and concrete implementations
for SQLite, LevelDB, and RocksDB.  This replaces the previous hard-coded
SQLite persistence in ``chain.py`` and allows operators to choose the
backend that best fits their deployment — SQLite for development and
lightweight nodes, LevelDB/RocksDB for high-throughput production chains.

Usage::

    # Development / quick start — zero config (default)
    backend = get_storage_backend("sqlite", db_path="/data/chain.db")

    # Production — LevelDB
    backend = get_storage_backend("leveldb", db_path="/data/chaindb")

    # Production — RocksDB (best write-throughput)
    backend = get_storage_backend("rocksdb", db_path="/data/chaindb")

Pass the backend to ``Chain(storage=backend)``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("zexus.blockchain.storage")


# ── Abstract interface ─────────────────────────────────────────────────

class StorageBackend(ABC):
    """Minimal key-value style interface for blockchain persistence.

    The blockchain uses three *namespaces* (logical tables):

    * ``blocks``         — height ↦ serialised block JSON
    * ``state``          — address ↦ account JSON
    * ``contract_state`` — address ↦ contract state JSON

    Implementations MUST support these three namespaces and provide
    atomic writes (``commit``).
    """

    NAMESPACES = ("blocks", "state", "contract_state")

    # -- core operations -------------------------------------------------

    @abstractmethod
    def get(self, namespace: str, key: str) -> Optional[str]:
        """Retrieve a value by key from *namespace*, or ``None``."""

    @abstractmethod
    def put(self, namespace: str, key: str, value: str) -> None:
        """Write *value* under *key* in *namespace*.

        Implementations may buffer the write until ``commit()`` is called.
        """

    @abstractmethod
    def delete(self, namespace: str, key: str) -> None:
        """Remove *key* from *namespace*."""

    @abstractmethod
    def has(self, namespace: str, key: str) -> bool:
        """Return ``True`` if *key* exists in *namespace*."""

    @abstractmethod
    def iterate(self, namespace: str) -> Iterator[Tuple[str, str]]:
        """Yield all ``(key, value)`` pairs in *namespace* in
        insertion / key order."""

    @abstractmethod
    def iterate_sorted(self, namespace: str) -> Iterator[Tuple[str, str]]:
        """Yield all ``(key, value)`` pairs sorted numerically if
        applicable (e.g. block heights)."""

    @abstractmethod
    def commit(self) -> None:
        """Flush any buffered writes to durable storage."""

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the backend."""

    # -- convenience helpers (overridable) --------------------------------

    def get_json(self, namespace: str, key: str) -> Optional[Any]:
        raw = self.get(namespace, key)
        if raw is None:
            return None
        return json.loads(raw)

    def put_json(self, namespace: str, key: str, obj: Any) -> None:
        self.put(namespace, key, json.dumps(obj, default=str))

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── SQLite backend (default) ───────────────────────────────────────────

class SQLiteBackend(StorageBackend):
    """SQLite-based storage — the battle-tested default.

    Best for: development, testnets, single-node setups, light nodes.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # better concurrent reads
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()
        logger.info("SQLite storage backend opened: %s", db_path)

    def _init_tables(self):
        for ns in self.NAMESPACES:
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS [{ns}] (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
        self._conn.commit()

    # -- interface -------------------------------------------------------

    def get(self, namespace: str, key: str) -> Optional[str]:
        row = self._conn.execute(
            f"SELECT value FROM [{namespace}] WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, namespace: str, key: str, value: str) -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO [{namespace}] (key, value) VALUES (?, ?)",
            (key, value),
        )

    def delete(self, namespace: str, key: str) -> None:
        self._conn.execute(
            f"DELETE FROM [{namespace}] WHERE key = ?", (key,)
        )

    def has(self, namespace: str, key: str) -> bool:
        row = self._conn.execute(
            f"SELECT 1 FROM [{namespace}] WHERE key = ? LIMIT 1", (key,)
        ).fetchone()
        return row is not None

    def iterate(self, namespace: str) -> Iterator[Tuple[str, str]]:
        cursor = self._conn.execute(f"SELECT key, value FROM [{namespace}]")
        yield from cursor

    def iterate_sorted(self, namespace: str) -> Iterator[Tuple[str, str]]:
        # For blocks, key is height (numeric string) — CAST allows proper sort
        cursor = self._conn.execute(
            f"SELECT key, value FROM [{namespace}] ORDER BY CAST(key AS INTEGER) ASC"
        )
        yield from cursor

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
            logger.info("SQLite storage backend closed")


# ── LevelDB backend ───────────────────────────────────────────────────

class LevelDBBackend(StorageBackend):
    """LevelDB-based storage — optimised for read-heavy workloads.

    Best for: production validator nodes, archive nodes (read-heavy).

    Requires the ``plyvel`` package::

        pip install plyvel
    """

    def __init__(self, db_path: str):
        try:
            import plyvel
        except ImportError:
            raise ImportError(
                "LevelDB backend requires the 'plyvel' package.  "
                "Install it with:  pip install plyvel"
            )
        self._db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self._dbs: Dict[str, Any] = {}
        for ns in self.NAMESPACES:
            self._dbs[ns] = plyvel.DB(
                os.path.join(db_path, ns), create_if_missing=True
            )
        logger.info("LevelDB storage backend opened: %s", db_path)

    def _encode(self, s: str) -> bytes:
        return s.encode("utf-8")

    def _decode(self, b: bytes) -> str:
        return b.decode("utf-8")

    def get(self, namespace: str, key: str) -> Optional[str]:
        val = self._dbs[namespace].get(self._encode(key))
        return self._decode(val) if val is not None else None

    def put(self, namespace: str, key: str, value: str) -> None:
        self._dbs[namespace].put(self._encode(key), self._encode(value))

    def delete(self, namespace: str, key: str) -> None:
        self._dbs[namespace].delete(self._encode(key))

    def has(self, namespace: str, key: str) -> bool:
        return self._dbs[namespace].get(self._encode(key)) is not None

    def iterate(self, namespace: str) -> Iterator[Tuple[str, str]]:
        for k, v in self._dbs[namespace]:
            yield self._decode(k), self._decode(v)

    def iterate_sorted(self, namespace: str) -> Iterator[Tuple[str, str]]:
        # LevelDB iterates in lexicographic order by default.
        # For numeric keys (block heights) we zero-pad on write, or
        # fall back to in-memory sort for safety.
        pairs = list(self.iterate(namespace))
        try:
            pairs.sort(key=lambda kv: int(kv[0]))
        except ValueError:
            pairs.sort()
        yield from pairs

    def commit(self) -> None:
        pass  # LevelDB writes are durable by default

    def close(self) -> None:
        for db in self._dbs.values():
            db.close()
        self._dbs.clear()
        logger.info("LevelDB storage backend closed")


# ── RocksDB backend ───────────────────────────────────────────────────

class RocksDBBackend(StorageBackend):
    """RocksDB-based storage — high write-throughput for busy chains.

    Best for: production mining/validator nodes with high TPS.

    Requires the ``python-rocksdb`` (or ``rocksdb``) package::

        pip install python-rocksdb
    """

    def __init__(self, db_path: str):
        try:
            import rocksdb  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "RocksDB backend requires the 'python-rocksdb' package.  "
                "Install it with:  pip install python-rocksdb"
            )
        self._db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self._rocksdb = rocksdb
        opts = rocksdb.Options()
        opts.create_if_missing = True
        opts.max_open_files = 512
        opts.write_buffer_size = 64 * 1024 * 1024   # 64 MB
        opts.max_write_buffer_number = 3
        opts.target_file_size_base = 64 * 1024 * 1024

        self._dbs: Dict[str, Any] = {}
        for ns in self.NAMESPACES:
            ns_path = os.path.join(db_path, ns)
            os.makedirs(ns_path, exist_ok=True)
            self._dbs[ns] = rocksdb.DB(ns_path, opts)
        logger.info("RocksDB storage backend opened: %s", db_path)

    def _encode(self, s: str) -> bytes:
        return s.encode("utf-8")

    def _decode(self, b: bytes) -> str:
        return b.decode("utf-8")

    def get(self, namespace: str, key: str) -> Optional[str]:
        val = self._dbs[namespace].get(self._encode(key))
        return self._decode(val) if val is not None else None

    def put(self, namespace: str, key: str, value: str) -> None:
        self._dbs[namespace].put(self._encode(key), self._encode(value))

    def delete(self, namespace: str, key: str) -> None:
        self._dbs[namespace].delete(self._encode(key))

    def has(self, namespace: str, key: str) -> bool:
        return self._dbs[namespace].get(self._encode(key)) is not None

    def iterate(self, namespace: str) -> Iterator[Tuple[str, str]]:
        it = self._dbs[namespace].iteritems()
        it.seek_to_first()
        for k, v in it:
            yield self._decode(k), self._decode(v)

    def iterate_sorted(self, namespace: str) -> Iterator[Tuple[str, str]]:
        pairs = list(self.iterate(namespace))
        try:
            pairs.sort(key=lambda kv: int(kv[0]))
        except ValueError:
            pairs.sort()
        yield from pairs

    def commit(self) -> None:
        pass  # RocksDB writes are durable by default

    def close(self) -> None:
        # python-rocksdb DBs are closed via garbage collection
        self._dbs.clear()
        logger.info("RocksDB storage backend closed")


# ── In-memory backend (testing) ────────────────────────────────────────

class MemoryBackend(StorageBackend):
    """Ephemeral in-memory backend — useful for tests and benchmarks."""

    def __init__(self):
        self._data: Dict[str, Dict[str, str]] = {ns: {} for ns in self.NAMESPACES}

    def get(self, namespace: str, key: str) -> Optional[str]:
        return self._data[namespace].get(key)

    def put(self, namespace: str, key: str, value: str) -> None:
        self._data[namespace][key] = value

    def delete(self, namespace: str, key: str) -> None:
        self._data[namespace].pop(key, None)

    def has(self, namespace: str, key: str) -> bool:
        return key in self._data[namespace]

    def iterate(self, namespace: str) -> Iterator[Tuple[str, str]]:
        yield from self._data[namespace].items()

    def iterate_sorted(self, namespace: str) -> Iterator[Tuple[str, str]]:
        pairs = list(self._data[namespace].items())
        try:
            pairs.sort(key=lambda kv: int(kv[0]))
        except ValueError:
            pairs.sort()
        yield from pairs

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self._data.clear()


# ── Backend factory ────────────────────────────────────────────────────

_BACKENDS = {
    "sqlite": SQLiteBackend,
    "leveldb": LevelDBBackend,
    "rocksdb": RocksDBBackend,
    "memory": MemoryBackend,
}


def get_storage_backend(name: str, **kwargs) -> StorageBackend:
    """Instantiate a storage backend by name.

    Parameters
    ----------
    name : str
        One of ``"sqlite"``, ``"leveldb"``, ``"rocksdb"``, ``"memory"``.
    **kwargs
        Forwarded to the backend constructor (e.g. ``db_path``).

    Returns
    -------
    StorageBackend
    """
    name = name.lower().strip()
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown storage backend '{name}'. "
            f"Available: {', '.join(sorted(_BACKENDS))}"
        )
    return cls(**kwargs)


def register_backend(name: str, cls: type) -> None:
    """Register a custom storage backend class."""
    if not issubclass(cls, StorageBackend):
        raise TypeError(f"{cls} is not a StorageBackend subclass")
    _BACKENDS[name.lower()] = cls
