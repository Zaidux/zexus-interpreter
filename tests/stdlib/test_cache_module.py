"""Tests for stdlib CacheModule."""

import time
import pytest
from src.zexus.stdlib.cache import CacheModule


class TestLRUCache:
    def test_create_default(self):
        c = CacheModule.create()
        assert c is not None

    def test_create_with_capacity(self):
        c = CacheModule.create(capacity=5)
        assert c is not None

    def test_put_and_get(self):
        c = CacheModule.create(capacity=10)
        CacheModule.put(c, "key1", "value1")
        assert CacheModule.get(c, "key1") == "value1"

    def test_get_missing_returns_default(self):
        c = CacheModule.create()
        assert CacheModule.get(c, "missing") is None
        assert CacheModule.get(c, "missing", "fallback") == "fallback"

    def test_has(self):
        c = CacheModule.create()
        CacheModule.put(c, "x", 42)
        assert CacheModule.has(c, "x") is True
        assert CacheModule.has(c, "y") is False

    def test_delete(self):
        c = CacheModule.create()
        CacheModule.put(c, "a", 1)
        CacheModule.delete(c, "a")
        assert CacheModule.has(c, "a") is False

    def test_clear(self):
        c = CacheModule.create()
        CacheModule.put(c, "a", 1)
        CacheModule.put(c, "b", 2)
        CacheModule.clear(c)
        assert CacheModule.size(c) == 0

    def test_size(self):
        c = CacheModule.create()
        assert CacheModule.size(c) == 0
        CacheModule.put(c, "a", 1)
        assert CacheModule.size(c) == 1

    def test_keys(self):
        c = CacheModule.create()
        CacheModule.put(c, "a", 1)
        CacheModule.put(c, "b", 2)
        assert set(CacheModule.keys(c)) == {"a", "b"}

    def test_eviction(self):
        c = CacheModule.create(capacity=2)
        CacheModule.put(c, "a", 1)
        CacheModule.put(c, "b", 2)
        CacheModule.put(c, "c", 3)  # should evict "a"
        assert CacheModule.has(c, "a") is False
        assert CacheModule.has(c, "b") is True
        assert CacheModule.has(c, "c") is True

    def test_lru_ordering(self):
        c = CacheModule.create(capacity=2)
        CacheModule.put(c, "a", 1)
        CacheModule.put(c, "b", 2)
        CacheModule.get(c, "a")  # access "a" so "b" is now LRU
        CacheModule.put(c, "c", 3)  # should evict "b"
        assert CacheModule.has(c, "a") is True
        assert CacheModule.has(c, "b") is False

    def test_stats(self):
        c = CacheModule.create()
        CacheModule.put(c, "x", 1)
        CacheModule.get(c, "x")  # hit
        CacheModule.get(c, "y")  # miss
        stats = CacheModule.stats(c)
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1

    def test_overwrite_value(self):
        c = CacheModule.create()
        CacheModule.put(c, "k", "old")
        CacheModule.put(c, "k", "new")
        assert CacheModule.get(c, "k") == "new"
        assert CacheModule.size(c) == 1


class TestTTLCache:
    def test_create_ttl(self):
        c = CacheModule.create_ttl(capacity=10, ttl=1)
        assert c is not None

    def test_put_and_get_ttl(self):
        c = CacheModule.create_ttl(ttl=10)
        CacheModule.put_ttl(c, "k", "v")
        assert CacheModule.get_ttl(c, "k") == "v"

    def test_ttl_expiry(self):
        c = CacheModule.create_ttl(ttl=0.1)
        CacheModule.put_ttl(c, "k", "v")
        time.sleep(0.2)
        assert CacheModule.get_ttl(c, "k") is None

    def test_custom_ttl_per_key(self):
        c = CacheModule.create_ttl(ttl=10)
        CacheModule.put_ttl(c, "short", "v", ttl=0.1)
        CacheModule.put_ttl(c, "long", "v", ttl=10)
        time.sleep(0.2)
        assert CacheModule.get_ttl(c, "short") is None
        assert CacheModule.get_ttl(c, "long") == "v"
