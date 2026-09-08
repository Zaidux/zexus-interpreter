# Zexus Programming Language

<div align="center">

![Zexus Logo](https://img.shields.io/badge/Zexus-v2.0.0-FF6B35?style=for-the-badge)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-Zaidux/zexus--interpreter-181717?style=for-the-badge&logo=github)](https://github.com/Zaidux/zexus-interpreter)
[![Tests](https://img.shields.io/badge/tests-2656%20passing-brightgreen.svg?style=for-the-badge)](.github/workflows/tests.yml)

**A security-first programming language for exploits, defense, and crypto.**

Safe by default, dangerous only on demand.

[Quick Start](QUICK_START.md) • [Grammar Spec](GRAMMAR.md) • [Roadmap](ROADMAP.md) • [Examples](examples/)

</div>

---

## What Zexus is

Zexus is a tree-walking interpreted language with an optional bytecode VM,
designed for security work in three categories:

| Category | What you get |
|---|---|
| **Exploits / DAST / bug bounty** | `bytes` with raw `\xNN` escapes, hex literals, `match`, the `pentest`/`netsec`/`payloads` modules (recon, fingerprinting, content-verified probing, structured findings) |
| **Defense / SAST / audit** | capability-gated builtins (`grant` / `revoke`), taint-tracked strings, sandboxed execution, contract `invariant` blocks, the `sast`/`audit` modules |
| **Crypto / contracts / chain** | `contract` / `state` / `action` / `require` / `emit`, checked-by-default integer arithmetic, real secp256k1/Keccak-256/AES-GCM via the `crypto` module (Rust-accelerated when the core is built) |

Plus everything a general-purpose language needs for backend work: HTTP
client with cookie surfaces, JSON, sockets, 4 database drivers, string
methods, pattern matching, ranges.

**One grammar.** [GRAMMAR.md](GRAMMAR.md) is the contract: one canonical
form per construct (~35 keywords), a full legacy→canonical migration
table, and parse errors are always fatal — statements are never silently
dropped.

## Installation

```bash
pip install zexus==2.0.0        # from PyPI
npm install -g zexus@2.0.0      # npm wrapper (requires python3 + pip)
```

From source:

```bash
git clone https://github.com/Zaidux/zexus-interpreter.git
cd zexus-interpreter
pip install -e .
```

Optional native acceleration (sha256/keccak hot paths, batch signature
verification):

```bash
cd rust_core && pip install maturin && maturin build --release
```

Everything works in pure Python without it — the extension is an
accelerator, never a dependency.

## Quick start

```zexus
// hello.zx
grant self network

use "netsec"

let headers = security_headers("https://example.com")
print(headers.missing)
```

```bash
zx run hello.zx
```

More: [QUICK_START.md](QUICK_START.md) · working examples in
[examples/](examples/) (recon tool, JSON API server, crypto, contracts).

## Safety model

- **Capability gates by default**: `http_get`, `file_read_text`, sockets
  and friends are denied until you `grant self network` (or `io_full`,
  etc.). `revoke` withdraws the same set. Undeclared operations fail
  loudly.
- **Checked arithmetic**: integer overflow traps; `wrapping_add` and
  friends are the explicit opt-out.
- **Taint-tracked strings**: sanitization status propagates through
  concatenation; `sanitize(x, "sql")` marks context.
- **Mock crypto is gated**: non-PEM keys raise instead of producing
  forgeable signatures (`ZEXUS_ALLOW_MOCK_CRYPTO=1` for tests only).

## Execution engines

Two engines, one behavior — enforced by a differential CI suite
([tests/grammar/test_differential.py](tests/grammar/test_differential.py)):
the same program must produce identical output on the tree-walk
evaluator and the bytecode VM, or the build fails.

Native tiers: **Rust core** (when built) → **pure Python**. The C/C++
extension layer was removed in v2.0.0.

## Honest status

- ✅ Language core: variables, functions, control flow, `match`, ranges,
  `bytes`, string methods, maps/lists, contracts with state/actions,
  modules (`use`), capability grants — all differential-tested
- ✅ Security modules: netsec (DNS/TLS/headers/certs/ports), pentest
  (fingerprint/probe/findings), payloads, sast, audit
- ✅ Crypto: SHA-2, Keccak-256, secp256k1 ECDSA, AES-256-GCM (Rust core
  accelerates hashing when built)
- ⚠️ Blockchain: the devnet/chain machinery is real but young; treat as
  experimental
- ⚠️ Known open issues: anonymous `action()` closures don't parse as
  expressions (V-004/R-029); the error model (errors-as-values vs
  exceptions) differs between engines in one documented case — both
  tracked as xfails in the differential suite, see [ROADMAP.md](ROADMAP.md)

No inflated benchmarks here. Performance work is Phase G; when there are
numbers, they will be measured and reproducible.

## Documentation

| Doc | Status |
|---|---|
| [GRAMMAR.md](GRAMMAR.md) | **Canonical** — the language contract |
| [QUICK_START.md](QUICK_START.md) | **Current** — verified commands |
| [ROADMAP.md](ROADMAP.md) | Phase tracker (A–F complete, I in progress) |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [ZEXUS_GUIDE.md](ZEXUS_GUIDE.md) | Tutorial — being regenerated against v2 (Phase I) |
| [examples/](examples/) | Runnable programs (recon, API server, crypto, contracts) |

## Development

```bash
pip install -e . && pip install pytest
python -m pytest tests/ -q            # full suite
python -m pytest tests/grammar/ -q    # grammar + differential parity
```

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
