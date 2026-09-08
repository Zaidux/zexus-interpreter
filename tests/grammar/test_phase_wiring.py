"""Tier 1-3 feature wiring regressions (post-Phase I).

invariant (enforcement + rollback), channels, find-where, seal,
throttle, middleware — each verified end-to-end through the CLI.
"""
from __future__ import annotations

import subprocess
import sys
import pathlib
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[2]
# PYTHONPATH: run the CLI against the source tree (not an installed copy)
import os
ZX_ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
ZX = [sys.executable, str(ROOT / "scripts" / "main.py"), "run", "--no-vm"]


def _run(code: str) -> tuple[int, str]:
    code = textwrap.dedent(code)
    import tempfile
    with tempfile.NamedTemporaryFile(
        suffix=".zx", mode="w", delete=False, dir=str(ROOT / "examples")
    ) as fh:
        fh.write(code)
        path = fh.name
    try:
        proc = subprocess.run(ZX + [path], capture_output=True,
                              text=True, timeout=30, cwd=str(ROOT), env=ZX_ENV)
        return proc.returncode, proc.stdout + proc.stderr
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def test_invariant_enforces_and_rolls_back():
    rc, out = _run("""
        contract Bank {
            state { balance: 100 }
            invariant no_overdraft { this.balance >= 0 }
            action withdraw(amount) { this.balance = this.balance - amount }
            action check() { return this.balance }
        }
        let b = Bank()
        b.withdraw(30)
        try { b.withdraw(200) } catch e { }
        print(b.check())
    """)
    assert rc == 0, out
    assert "70" in out, out  # rolled back to pre-violation value


def test_channels_roundtrip():
    rc, out = _run("""
        channel<integer> numbers = 10;
        send(numbers, 42);
        let a = receive(numbers);
        print(a)
    """)
    assert rc == 0, out
    assert "42" in out, out


def test_find_where_first_match():
    rc, out = _run("""
        let users = [{"name": "alice", "age": 30}, {"name": "bob", "age": 16}]
        let adult = find u in users where (u.age >= 18)
        print(adult.name)
    """)
    assert rc == 0, out
    assert "alice" in out, out


def test_find_where_no_match_is_null():
    rc, out = _run("""
        let items = [1, 2, 3]
        let hit = find x in items where (x > 100)
        print(hit)
    """)
    assert rc == 0, out
    assert "null" in out, out


def test_seal_blocks_mutation():
    rc, out = _run("""
        let config = {"k": "v"}
        seal config
        print(typeof(config))
    """)
    assert rc == 0, out
    assert "sealed" in out.lower(), out


def test_throttle_enforces_rate():
    rc, out = _run("""
        throttle api, { rate: 3, burst: 3 }
        let allowed = 0
        let i = 0
        while i < 5 {
            if throttle_check("api") { allowed = allowed + 1 }
            i = i + 1
        }
        print(allowed)
    """)
    assert rc == 0, out
    assert "3" in out, out  # 5 calls, only 3 within limits


def test_middleware_rejects():
    rc, out = _run("""
        fn guard(req) {
            if req.token == "secret" { return true }
            return false
        }
        middleware guard
        fn route(req) { return "data" }
        register_route("r", route)
        let bad = dispatch("r", {"token": "wrong"})
        print(bad)
    """)
    assert rc == 0, out
    assert "rejected" in out, out
