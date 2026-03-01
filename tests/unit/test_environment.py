"""
Unit tests for src/zexus/environment.py — Environment and _EnvironmentValuesProxy.

Tests the core runtime environment: variable scoping, module imports,
exports, nested scope resolution, and mapping protocol compliance.
"""

import pytest
from src.zexus.environment import Environment, _EnvironmentValuesProxy


# ---------------------------------------------------------------------------
# Basic store operations
# ---------------------------------------------------------------------------

class TestBasicOperations:
    def test_set_and_get(self):
        env = Environment()
        env.set("x", 42)
        assert env.get("x") == 42

    def test_get_missing_returns_default(self):
        env = Environment()
        assert env.get("missing") is None
        assert env.get("missing", "fallback") == "fallback"

    def test_set_overwrite(self):
        env = Environment()
        env.set("x", 1)
        env.set("x", 2)
        assert env.get("x") == 2

    def test_set_none_explicitly(self):
        env = Environment()
        env.set("x", None)
        assert "x" in env
        assert env.get("x") is None

    def test_bracket_set_get(self):
        env = Environment()
        env["a"] = 10
        assert env["a"] == 10

    def test_contains(self):
        env = Environment()
        env.set("x", 1)
        assert "x" in env
        assert "y" not in env

    def test_len(self):
        env = Environment()
        assert len(env) == 0
        env.set("a", 1)
        env.set("b", 2)
        assert len(env) == 2

    def test_iter(self):
        env = Environment()
        env.set("a", 1)
        env.set("b", 2)
        assert set(env) == {"a", "b"}

    def test_keys(self):
        env = Environment()
        env.set("x", 1)
        env.set("y", 2)
        assert set(env.keys()) == {"x", "y"}

    def test_items(self):
        env = Environment()
        env.set("x", 1)
        assert dict(env.items()) == {"x": 1}

    def test_copy(self):
        env = Environment()
        env.set("a", 1)
        c = env.copy()
        assert isinstance(c, dict)
        assert c == {"a": 1}
        c["a"] = 99
        assert env.get("a") == 1  # original not mutated


class TestUpdate:
    def test_update_from_dict(self):
        env = Environment()
        env.update({"a": 1, "b": 2})
        assert env.get("a") == 1
        assert env.get("b") == 2

    def test_update_none(self):
        env = Environment()
        env.update(None)  # should not raise
        assert len(env) == 0

    def test_update_from_environment(self):
        src = Environment()
        src.set("x", 10)
        dst = Environment()
        dst.update(src)
        assert dst.get("x") == 10


class TestSetdefault:
    def test_setdefault_missing(self):
        env = Environment()
        result = env.setdefault("x", 42)
        assert result == 42
        assert env.get("x") == 42

    def test_setdefault_existing(self):
        env = Environment()
        env.set("x", 1)
        result = env.setdefault("x", 99)
        assert result == 1
        assert env.get("x") == 1


# ---------------------------------------------------------------------------
# Scope chain (outer)
# ---------------------------------------------------------------------------

class TestScopeChain:
    def test_lookup_in_outer(self):
        outer = Environment()
        outer.set("x", 100)
        inner = Environment(outer=outer)
        assert inner.get("x") == 100

    def test_inner_shadows_outer(self):
        outer = Environment()
        outer.set("x", 100)
        inner = Environment(outer=outer)
        inner.set("x", 200)
        assert inner.get("x") == 200
        assert outer.get("x") == 100

    def test_contains_checks_outer(self):
        outer = Environment()
        outer.set("x", 1)
        inner = Environment(outer=outer)
        assert "x" in inner

    def test_deep_scope_chain(self):
        """Environment.get() traverses the outer chain.
        The middle env (e2) needs a variable so that the chain is
        exercised without short-circuiting on len==0."""
        e1 = Environment()
        e1.set("a", 1)
        e2 = Environment(outer=e1)
        e2.set("b", 2)  # exercise real chain traversal
        e3 = Environment(outer=e2)
        # Lookup goes: e3 → e2 → e1
        assert e3.get("a") == 1
        assert e3.get("b") == 2


# ---------------------------------------------------------------------------
# Assign (reassignment)
# ---------------------------------------------------------------------------

class TestAssign:
    def test_assign_existing_local(self):
        env = Environment()
        env.set("x", 1)
        env.assign("x", 2)
        assert env.get("x") == 2

    def test_assign_creates_if_missing(self):
        env = Environment()
        env.assign("y", 42)
        assert env.get("y") == 42

    def test_assign_updates_outer(self):
        outer = Environment()
        outer.set("x", 10)
        inner = Environment(outer=outer)
        inner.assign("x", 20)
        assert outer.get("x") == 20
        # inner should NOT have the variable in its own store
        assert "x" not in inner.store


class TestHasVariable:
    def test_local(self):
        env = Environment()
        env.set("x", 1)
        assert env._has_variable("x") is True

    def test_outer(self):
        outer = Environment()
        outer.set("x", 1)
        inner = Environment(outer=outer)
        assert inner._has_variable("x") is True

    def test_missing(self):
        env = Environment()
        assert env._has_variable("nope") is False


# ---------------------------------------------------------------------------
# Module system
# ---------------------------------------------------------------------------

class TestModules:
    def test_import_module(self):
        env = Environment()
        mod = Environment()
        mod.set("greet", "hello")
        env.import_module("mymod", mod)
        assert env.get("mymod.greet") == "hello"

    def test_contains_dotted_name(self):
        env = Environment()
        mod = Environment()
        mod.set("x", 1)
        env.import_module("m", mod)
        assert "m.x" in env

    def test_set_dotted_creates_module(self):
        env = Environment()
        env.set("ns.val", 42)
        assert "ns" in env.modules
        assert env.modules["ns"].get("val") == 42

    def test_set_dotted_existing_module(self):
        env = Environment()
        mod = Environment()
        mod.set("a", 1)
        env.import_module("mod", mod)
        env.set("mod.b", 2)
        assert mod.get("b") == 2


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

class TestExports:
    def test_export_stores_in_both(self):
        env = Environment()
        env.export("x", 42)
        assert env.get("x") == 42
        assert env.exports["x"] == 42

    def test_get_exports_returns_copy(self):
        env = Environment()
        env.export("a", 1)
        exports = env.get_exports()
        exports["a"] = 999
        assert env.exports["a"] == 1  # not mutated


# ---------------------------------------------------------------------------
# Values proxy
# ---------------------------------------------------------------------------

class TestValuesProxy:
    def test_callable_returns_store_values(self):
        env = Environment()
        env.set("x", 1)
        env.set("y", 2)
        vals = env.values()
        assert set(vals) == {1, 2}

    def test_contains_checks_exports(self):
        env = Environment()
        env.export("pub", 10)
        env.set("priv", 20)
        assert "pub" in env.values
        assert "priv" not in env.values

    def test_iter_over_exports(self):
        env = Environment()
        env.export("a", 1)
        env.export("b", 2)
        assert set(iter(env.values)) == {"a", "b"}

    def test_len_counts_exports(self):
        env = Environment()
        env.export("a", 1)
        assert len(env.values) == 1

    def test_keys_and_items(self):
        env = Environment()
        env.export("k", "v")
        assert list(env.values.keys()) == ["k"]
        assert list(env.values.items()) == [("k", "v")]


# ---------------------------------------------------------------------------
# Debug logging
# ---------------------------------------------------------------------------

class TestDebug:
    def test_enable_disable_debug(self):
        env = Environment()
        assert env._debug is False
        env.enable_debug()
        assert env._debug is True
        env.disable_debug()
        assert env._debug is False

    def test_debug_log_silent_when_disabled(self, capsys):
        env = Environment()
        env.debug_log("should not print")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_debug_log_prints_when_enabled(self, capsys):
        env = Environment()
        env.enable_debug()
        env.debug_log("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out
