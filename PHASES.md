# Zexus Blockchain — Rust Migration Phases

> Roadmap for moving contract execution from Python to Rust to achieve 20K-50K real-contract TPS.

## Current Baseline (Options 2 + 3 — Complete)

| Metric | Value |
|--------|-------|
| Sustained TPS (rate-limited) | **10,000+** ✅ |
| Burst TPS (Rust batched-GIL) | **133,138** |
| Burst TPS (Multiprocess 4w) | **43,000** |
| Error rate | 0% |
| Execution modes | 3-tier: multiprocess → Rust batched-GIL → Python threads |

---

## Phase 0 — Enable Bytecode Compilation for Contracts ✅ COMPLETE
**Status:** Complete (2026-02-21)  
**Effort:** ~1 day  
**Risk:** Low  

### Goal
Contracts currently execute via tree-walk interpretation (`use_vm=False` in `contract_vm.py`).
The bytecode VM (`vm/vm.py`, 4,830 lines, ~50 opcodes) exists but is **not used** for
contract execution. Phase 0 enables and validates the bytecode path.

### Results
- Added `use_bytecode_vm` feature flag to `ContractVM.__init__()` (default `False`)
- Bytecoded execution with automatic fallback to tree-walk on failure
- Shared gas metering, builtins, and blockchain state between contract VM and evaluator VM
- Execution stats tracking (`get_vm_execution_stats()`)
- **17/17 test cases passed** — all contract patterns work under bytecoded mode
- **100% bytecode rate** — zero fallbacks to tree-walk

### Benchmark (Python bytecode VM vs tree-walk)

| Loop iterations | Tree-walk | Bytecoded | Speedup |
|-----------------|-----------|-----------|---------|
| 100 | 9.0ms | 6.2ms | 1.45x |
| 500 | 35.8ms | 26.1ms | 1.37x |
| 1,000 | 51.9ms | 49.1ms | 1.06x |
| 2,000 | 140.0ms | 108.6ms | 1.29x |
| 5,000 | 298.0ms | 335.7ms | 0.89x |

Python bytecoded is modestly faster. The real gains come in Phase 2 (Rust bytecode interpreter).

### Files Changed
- `src/zexus/blockchain/contract_vm.py` — `use_bytecode_vm` flag, rewritten `_execute_action()`, `get_vm_execution_stats()`
- `tests/blockchain/test_phase0_bytecode_vm.py` — 17 test cases covering arithmetic, conditionals, loops, state ops, strings, functions, collections, error handling, stats, benchmarks

---

## Phase 1 — Binary Bytecode Format ✅ COMPLETE
**Status:** Complete (2026-02-22)  
**Effort:** ~1 day  
**Risk:** Low  

### Goal
Replace Python tuple-based opcodes `(opcode_name, operand)` with a compact binary format
that Rust can consume without Python interop overhead.

### Results
- Defined `.zxc` binary bytecode specification: 16-byte header (magic `ZXC\x00`, version, flags, counts), typed constant pool (9 tags + OPAQUE), variable-width instructions (5 operand types), CRC32 checksum
- Python serializer/deserializer: `serialize()`, `deserialize()`, `save_zxc()`, `load_zxc()`
- Multi-bytecode container: `serialize_multi()`, `deserialize_multi()` for file-level caching
- Co-located `.zxc` helper: `zxc_path_for()`, `is_zxc_fresh()` for automatic module caching
- Rust deserializer: GIL-free `deserialize_zxc()` + `RustBytecodeReader` PyO3 class with `deserialize()`, `validate()`, `header_info()` methods
- Wired into VM module loading (co-located `.zxc` files), disk cache (`.zxc` replaces pickle), contract VM (per-action `.zxc` caching)
- **43/43 test cases passed** — round-trip, checksums, file I/O, Rust cross-validation, compiler integration, multi-container, cache integration
- **1621 total tests pass** (zero regressions)

### Benchmarks

| Metric | Value |
|--------|-------|
| Binary size vs Python | **5% (20:1 compression)** |
| Serialize (7001 instrs) | 8.5ms |
| Deserialize (7001 instrs) | 20.2ms |
| Rust validate | GIL-free ✅ |
| Rust deserialize | GIL-free ✅ |

### Files Added/Changed
- `src/zexus/vm/binary_bytecode.py` — Python serializer/deserializer (~660 lines)
- `rust_core/src/binary_bytecode.rs` — Rust GIL-free deserializer (~544 lines)
- `rust_core/src/lib.rs` — registered `binary_bytecode` module
- `src/zexus/vm/cache.py` — disk persistence now uses `.zxc` binary format
- `src/zexus/vm/vm.py` — module loading checks co-located `.zxc` first
- `src/zexus/blockchain/contract_vm.py` — per-action `.zxc` caching
- `tests/vm/test_binary_bytecode.py` — 43 comprehensive tests

---

## Phase 2 — Rust Bytecode Interpreter ⬅️ NEXT ⭐ PRIMARY GOAL
**Status:** Not Started  
**Effort:** 4-6 weeks  
**Risk:** Medium  

### Goal
Port the bytecode dispatch loop from Python `vm/vm.py` to Rust. This is where 80%+ of
CPU time is spent during contract execution. Expected **10-50× per-contract speedup.**

### Tasks
- [ ] Implement Rust stack machine (`rust_core/src/vm.rs`)
- [ ] Port all ~50 opcodes (arithmetic, control flow, data, blockchain-specific)
- [ ] Implement Rust value type system (matching Python's dynamic types)
- [ ] Handle blockchain opcodes: `STATE_READ`, `STATE_WRITE`, `HASH_BLOCK`, `GAS_CHARGE`, `EMIT`, `CALL_CONTRACT`
- [ ] Implement gas metering in Rust (per-opcode costs)
- [ ] Create Python↔Rust state bridge for `STATE_READ`/`STATE_WRITE` callbacks
- [ ] Comprehensive opcode-level test suite
- [ ] Integration test with real contracts
- [ ] Benchmark against Python VM

### Success Criteria
- All contracts execute correctly under Rust VM
- 10-50× per-contract throughput improvement
- Gas metering matches Python VM within 5% tolerance
- Aggregate TPS: 5,000-15,000 with real contracts

---

## Phase 3 — Rust Gas Metering + State Adapter
**Status:** Not Started  
**Effort:** 1-2 weeks  
**Risk:** Low-Medium  

### Goal
Move `ContractStateAdapter` to Rust so state reads/writes don't cross the GIL boundary
during execution. Move gas metering logic to Rust for tighter integration.

### Tasks
- [ ] Implement `RustStateAdapter` with in-memory state cache
- [ ] Batch state flushes to Python storage layer
- [ ] Move gas calculation tables to Rust
- [ ] Implement gas refund logic in Rust
- [ ] Integration test with Phase 2 Rust VM

### Success Criteria
- State operations are 5-10× faster (no per-op GIL crossing)
- Gas metering is exact (no Python↔Rust discrepancies)
- Additional 2-3× aggregate TPS improvement

---

## Phase 4 — Rust ContractVM Orchestration
**Status:** Not Started  
**Effort:** 2-3 weeks  
**Risk:** Medium  

### Goal
Port the `ContractVM` orchestration layer (reentrancy guards, environment setup,
receipt generation, state commit/rollback) to Rust.

### Tasks
- [ ] Implement Rust `ContractVM` struct
- [ ] Port reentrancy detection
- [ ] Port environment + builtins setup
- [ ] Port transaction context management
- [ ] Port state commit/rollback logic
- [ ] Port `ContractExecutionReceipt` generation
- [ ] Cross-contract call handling in Rust

### Success Criteria
- Full contract execution lifecycle in Rust (no Python except storage backend)
- Eliminates Python orchestration overhead
- All existing contracts pass

---

## Phase 5 — Eliminate GIL Callback in Batch Executor
**Status:** Not Started  
**Effort:** ~1 week  
**Risk:** Low  

### Goal
With Phases 2-4 complete, the Rust batch executor no longer needs `Python::with_gil()`
callbacks. Remove the GIL acquisition entirely for pure-Rust end-to-end execution.

### Tasks
- [ ] Update `executor.rs` to call Rust VM directly instead of Python callback
- [ ] Remove `vm_callback` parameter from `execute_batch()`
- [ ] Benchmark GIL-free batch execution
- [ ] Verify Rayon parallelism scales linearly without GIL contention

### Success Criteria
- Zero GIL acquisitions during batch execution
- Near-linear scaling with CPU cores
- Aggregate TPS: 20,000-50,000 with real contracts

---

## Phase 6 — Rust Builtins
**Status:** Not Started  
**Effort:** 2-3 weeks  
**Risk:** Low  

### Goal
Port contract builtins (`crypto_hash`, `verify_signature`, `emit`, `log`, `transfer`,
`balance_of`, etc.) from Python to Rust. Currently these are the remaining Python calls
during contract execution.

### Tasks
- [ ] Audit all builtins used in contracts
- [ ] Port crypto builtins (already have Rust hasher + signature modules)
- [ ] Port state builtins (`transfer`, `balance_of`, `state_read`, `state_write`)
- [ ] Port event builtins (`emit`, `log`)
- [ ] Port utility builtins (`block_number`, `timestamp`, `caller`, `origin`)
- [ ] Wire into Rust VM opcode handlers

### Success Criteria
- All builtins execute in pure Rust
- Completes the full-Rust execution pipeline
- Target: up to 50,000 TPS with real contracts

---

## Deferred — Full Tree-Walk Evaluator Port
**Status:** Deferred (not recommended now)  
**Effort:** 8-12 weeks  
**Risk:** High  

### Why Deferred
The tree-walk evaluator (`evaluator/core.py`) handles 117+ AST node types with Python
closures and dynamic dispatch. Porting it is extremely complex and **unnecessary** once
contracts use the bytecode path. The tree-walk evaluator remains for REPL, scripting,
and non-contract use cases.

---

## Impact on .zx Contract Authors

| Aspect | Impact |
|--------|--------|
| Contract source code | **No changes** — same Zexus syntax |
| Contract semantics | **Identical** — same behavior |
| Deployment workflow | **Unchanged** — deploy, call, query |
| Gas costs | **Recalibration needed** — Rust is faster, so per-opcode gas ratios shift |
| Debug stack traces | **Different format** — Rust traces vs Python traces |
| Existing deployed contracts | **Fully compatible** — re-compilation is automatic |

---

## Timeline Summary

| Phase | Effort | Cumulative TPS (Real Contracts) |
|-------|--------|--------------------------------|
| Current (Options 2+3) | Done | 10,000+ |
| Phase 0 | ~1 week | 10,000+ (validation only) |
| Phase 1 | 1-2 weeks | 10,000+ (format only) |
| Phase 2 | 4-6 weeks | 5,000-15,000 |
| Phase 3 | 1-2 weeks | 10,000-20,000 |
| Phase 4 | 2-3 weeks | 15,000-30,000 |
| Phase 5 | ~1 week | 20,000-50,000 |
| Phase 6 | 2-3 weeks | up to 50,000 |
| **Total** | **~13-18 weeks** | **20,000-50,000** |

---

*Last updated: 2026-02-21*
