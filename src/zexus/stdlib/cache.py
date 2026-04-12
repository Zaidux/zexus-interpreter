"""Cache module for Zexus standard library."""

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional


class CacheModule:
    """Provides LRU and TTL cache implementations."""

    @staticmethod
    def create(capacity: int = 128) -> dict:
        """Create a new LRU cache with the given maximum capacity.

        Returns a dict-like object containing the ordered storage
        and metadata for tracking hits and misses.
        """
        return {
            "_type": "lru",
            "_data": OrderedDict(),
            "_capacity": capacity,
            "_hits": 0,
            "_misses": 0,
        }

    @staticmethod
    def get(cache: dict, key: str, default: Any = None) -> Any:
        """Get an item from the LRU cache.

        Moves the accessed key to the end (most recently used).
        Returns default if the key is not found.
        """
        data = cache["_data"]
        if key in data:
            data.move_to_end(key)
            cache["_hits"] += 1
            return data[key]
        cache["_misses"] += 1
        return default

    @staticmethod
    def put(cache: dict, key: str, value: Any) -> None:
        """Put an item into the LRU cache.

        If the key already exists, its value is updated and it is
        moved to the most-recently-used position. If the cache is
        at capacity, the least-recently-used item is evicted.
        """
        data = cache["_data"]
        if key in data:
            data.move_to_end(key)
            data[key] = value
        else:
            if len(data) >= cache["_capacity"]:
                data.popitem(last=False)
            data[key] = value

    @staticmethod
    def delete(cache: dict, key: str) -> bool:
        """Remove an item from the cache.

        Returns True if the key was present, False otherwise.
        """
        data = cache["_data"]
        if key in data:
            del data[key]
            return True
        return False

    @staticmethod
    def clear(cache: dict) -> None:
        """Clear all items from the cache and reset statistics."""
        cache["_data"].clear()
        cache["_hits"] = 0
        cache["_misses"] = 0

    @staticmethod
    def size(cache: dict) -> int:
        """Return the current number of items in the cache."""
        return len(cache["_data"])

    @staticmethod
    def keys(cache: dict) -> List[str]:
        """Return all keys currently stored in the cache."""
        return list(cache["_data"].keys())

    @staticmethod
    def has(cache: dict, key: str) -> bool:
        """Check whether a key exists in the cache."""
        return key in cache["_data"]

    @staticmethod
    def stats(cache: dict) -> Dict[str, Any]:
        """Return hit/miss statistics for the cache."""
        hits = cache["_hits"]
        misses = cache["_misses"]
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "total": total,
            "hit_rate": hits / total if total > 0 else 0.0,
            "size": len(cache["_data"]),
            "capacity": cache["_capacity"],
        }

    # ------------------------------------------------------------------
    # TTL (Time-To-Live) Cache
    # ------------------------------------------------------------------

    @staticmethod
    def create_ttl(capacity: int = 128, ttl: int = 300) -> dict:
        """Create a new TTL cache.

        Each entry expires after *ttl* seconds unless a per-key TTL is
        provided when inserting. Expired entries are lazily evicted on
        access and eagerly evicted when the cache is at capacity.
        """
        return {
            "_type": "ttl",
            "_data": OrderedDict(),
            "_expiry": {},
            "_capacity": capacity,
            "_default_ttl": ttl,
            "_hits": 0,
            "_misses": 0,
        }

    @staticmethod
    def _is_expired(cache: dict, key: str) -> bool:
        """Check whether a TTL cache entry has expired."""
        return time.time() > cache["_expiry"].get(key, 0)

    @staticmethod
    def _evict_expired(cache: dict) -> None:
        """Remove all expired entries from a TTL cache."""
        now = time.time()
        expired = [k for k, exp in cache["_expiry"].items() if now > exp]
        for k in expired:
            cache["_data"].pop(k, None)
            cache["_expiry"].pop(k, None)

    @staticmethod
    def get_ttl(cache: dict, key: str, default: Any = None) -> Any:
        """Get an item from the TTL cache.

        Returns *default* if the key does not exist or has expired.
        Expired entries are removed on access.
        """
        data = cache["_data"]
        if key in data:
            if CacheModule._is_expired(cache, key):
                del data[key]
                del cache["_expiry"][key]
                cache["_misses"] += 1
                return default
            data.move_to_end(key)
            cache["_hits"] += 1
            return data[key]
        cache["_misses"] += 1
        return default

    @staticmethod
    def put_ttl(cache: dict, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put an item into the TTL cache.

        Uses the cache's default TTL if *ttl* is not specified.
        Evicts expired entries first, then evicts the least-recently-used
        entry if the cache is still at capacity.
        """
        if ttl is None:
            ttl = cache["_default_ttl"]

        data = cache["_data"]

        if key in data:
            data.move_to_end(key)
            data[key] = value
            cache["_expiry"][key] = time.time() + ttl
        else:
            CacheModule._evict_expired(cache)
            if len(data) >= cache["_capacity"]:
                evicted_key, _ = data.popitem(last=False)
                cache["_expiry"].pop(evicted_key, None)
            data[key] = value
            cache["_expiry"][key] = time.time() + ttl


# Export functions for easy access
create = CacheModule.create
get = CacheModule.get
put = CacheModule.put
delete = CacheModule.delete
clear = CacheModule.clear
size = CacheModule.size
keys = CacheModule.keys
has = CacheModule.has
stats = CacheModule.stats
create_ttl = CacheModule.create_ttl
get_ttl = CacheModule.get_ttl
put_ttl = CacheModule.put_ttl
