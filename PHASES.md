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

## Phase 2 — Rust Bytecode Interpreter ✅ COMPLETE
**Status:** Complete (2026-02-23)  
**Effort:** ~1 day  
**Risk:** Medium  

### Goal
Port the bytecode dispatch loop from Python `vm/vm.py` to Rust. This is where 80%+ of
CPU time is spent during contract execution. Expected **10-50× per-contract speedup.**

### Results
- Implemented complete Rust stack-machine bytecode interpreter (`rust_core/src/rust_vm.rs`, ~1,600 lines)
- `ZxValue` enum with all runtime types: Null, Bool, Int(i64), Float(f64), Str, List, Map, PyObj
- `Op` enum mapping all ~50 opcodes with `from_u16()` conversion and per-opcode `gas_cost()`
- `RustVM` struct: stack machine with full execution loop handling:
  - Stack ops: LOAD_CONST, LOAD_NAME, STORE_NAME, STORE_FUNC, POP, DUP
  - Arithmetic: ADD, SUB, MUL, DIV, MOD, POW, NEG (with int/float promotion, string concat/repeat, list concat)
  - Comparison: EQ, NEQ, LT, GT, LTE, GTE (with mixed-type comparison)
  - Logical: AND, OR, NOT
  - Control flow: JUMP, JUMP_IF_FALSE, JUMP_IF_TRUE, RETURN
  - Collections: BUILD_LIST, BUILD_MAP, BUILD_SET, INDEX, SLICE, GET_ATTR
  - Blockchain: STATE_READ, STATE_WRITE, TX_BEGIN, TX_COMMIT, TX_REVERT, GAS_CHARGE, REQUIRE, HASH_BLOCK, MERKLE_ROOT, VERIFY_SIGNATURE, LEDGER_APPEND, EMIT_EVENT, REGISTER_EVENT, AUDIT_LOG, RESTRICT_ACCESS
  - Exceptions: SETUP_TRY, POP_TRY, THROW
  - I/O: PRINT
  - Markers: NOP, PARALLEL_START, PARALLEL_END
- Function calls (CALL_NAME, CALL_TOP, CALL_METHOD, CALL_BUILTIN, CALL_FUNC_CONST) signal `NeedsPythonFallback` — Phase 3+ will inline common builtins
- `RustVMExecutor` PyO3 class with `execute(data, env, state, gas_limit)` and `benchmark(data, iterations, gas_limit)` methods
- Gas metering matching Python VM per-opcode costs
- Nested transaction support with snapshot/rollback
- **70/70 test cases passed** covering all opcode categories, gas metering, exception handling, loops, fibonacci, contract simulation
- **1692 total tests pass** (zero regressions from Phase 0 + Phase 1)

### Benchmarks

| Metric | Value |
|--------|-------|
| Rust VM throughput | **22 MIPS** (million instructions/sec) |
| Python VM throughput | ~1.1 MIPS |
| **Speedup** | **20.7×** |
| n=1K loop (100 iters) | 75 ms |
| n=10K loop (100 iters) | 805 ms |
| n=100K loop (10 iters) | 764 ms |
| Peak throughput | 20.7 MIPS |

### Architecture
```
Python Contract VM ──► .zxc bytes ──► Rust RustVMExecutor
                                          │
                                     deserialize_zxc()  (GIL-free)
                                          │
                                     RustVM::execute()  (pure Rust)
                                          │
                                     result dict ──► Python
```

### Files Added/Changed
- `rust_core/src/rust_vm.rs` — Complete Rust bytecode interpreter (~1,600 lines)
- `rust_core/src/lib.rs` — Registered `rust_vm` module + `RustVMExecutor` class
- `rust_core/Cargo.toml` — No new dependencies (uses existing sha2, hex)
- `tests/vm/test_rust_vm.py` — 70 comprehensive tests
- `benchmark_rust_vm.py` — Benchmark script
- `bench_quick.py` — Quick throughput benchmark
- `bench_vs.py` — Python vs Rust comparison benchmark

---

## Phase 3 — Adaptive VM Routing + Rust State Adapter ✅ COMPLETE
**Status:** Complete (2026-02-24)  
**Effort:** ~1 day  
**Risk:** Low  

### Goal
Connect the Rust VM to the Python VM with an adaptive execution strategy:
when the VM detects a program with ≥10,000 operations, it automatically
delegates to the Rust VM for faster execution. Also implement a Rust-side
state cache (`RustStateAdapter`) to minimise GIL crossings for state ops.

### Results
- **Adaptive VM routing** in Python VM (`vm.py`):
  - Added `_RUST_VM_AVAILABLE` flag + `RustVMExecutor` import at module level
  - Added `_rust_vm_threshold` (default 10,000, configurable via `ZEXUS_RUST_VM_THRESHOLD` env var)
  - Added `_execute_via_rust_vm()` method: serializes bytecode → .zxc, executes in Rust, bridges env + state + gas back to Python
  - Injection point: `_run_stack_bytecode_sync()` — Rust VM check sits right after the Cython fastops check
  - Transparent fallback: if Rust VM signals `needs_fallback` (e.g. for CALL_NAME), Python VM handles it seamlessly
  - Runtime toggle: `vm._rust_vm_enabled = False` to disable

- **Contract VM integration** (`contract_vm.py`):
  - Added `_try_rust_vm_execution()` method to `ContractVM`
  - Rust VM tier inserted before Phase 0 bytecoded execution for large contracts
  - Shares gas metering bridge (Rust gas_used → Python gas_metering.gas_used)
  - State changes from Rust merge back into `ContractStateAdapter`
  - Stats tracking: `rust_executions`, `rust_fallbacks`, `rust_rate` in `get_vm_execution_stats()`

- **RustStateAdapter** (`rust_core/src/state_adapter.rs`, ~280 lines):
  - PyO3 class with in-memory `HashMap<String, StateValue>` cache
  - `load_from_dict()` — bulk load from Python dict (warm cache)
  - `get()` / `set()` / `contains()` / `delete()` — cache-local operations
  - `flush_dirty()` — return only modified keys as Python dict (batch flush)
  - `tx_begin()` / `tx_commit()` / `tx_revert()` — snapshot-based nested transactions
  - Stats: `cache_hits`, `cache_writes`, `tx_depth`
  - 100K entries: load=45ms, 10K reads=13ms, 10K writes=11ms, flush=4ms

- **Gas metering bridge**:
  - Python passes remaining gas budget to Rust VM
  - Rust VM's `gas_used` bridges back to Python `GasMetering.gas_used`
  - `OutOfGas` errors from Rust raise properly in Python
  - Handles `GasMetering.remaining()` as callable method (not property)

- **48/48 test cases passed** covering:
  - RustStateAdapter: get/set, bulk load, transactions, dirty tracking, nested dicts/lists, 10K entries
  - Adaptive routing: threshold detection, env override, runtime toggle, Rust/Python switching
  - Gas bridge: limit enforcement, out-of-gas, usage tracking, bridge-back
  - Fallback: CALL_NAME triggers fallback, Python handles it, errors fall through
  - Integration: accumulated stats, env persistence, benchmark method, serialization overhead
  - Performance: 100K state entries, serialization timing
- **1740 total tests pass** (zero regressions from Phase 0 + Phase 1 + Phase 2)

### Benchmarks

| Program Size | Python VM | Adaptive | Direct Rust | Adaptive Speedup | Rust Speedup |
|-------------|-----------|----------|-------------|-----------------|--------------|
| 202 ops | 1.1ms | 0.8ms | 0.02ms | 1.4× | 55× |
| 2,002 ops | 7.5ms | 8.8ms | 0.18ms | 0.8× (below threshold) | 42× |
| 10,002 ops | 39.3ms | 33.8ms | 0.85ms | 1.2× | 46× |
| 20,002 ops | 82.9ms | 66.0ms | 3.5ms | 1.3× | 24× |
| 100,002 ops | 404.9ms | 417.3ms | 8.2ms | 1.0× | 49× |

**Note:** Adaptive path includes serialization overhead (~1ms/1K instrs). For pre-compiled
.zxc contracts (Phase 1 caching), serialization is skipped, yielding speedups closer to
the "Direct Rust" column. The RustStateAdapter provides additional speedup for
state-heavy contracts by eliminating per-operation GIL crossings.

| RustStateAdapter | Load | Read (10K) | Write (10K) | Flush |
|-----------------|------|-----------|------------|-------|
| 1,000 keys | 0.4ms | 0.4ms | 0.8ms | 0.3ms |
| 10,000 keys | 5.7ms | 4.5ms | 15.5ms | 7.6ms |
| 100,000 keys | 45.0ms | 12.9ms | 10.7ms | 4.4ms |

### Architecture
```
Python VM (vm.py)
    │
    ├── len(instructions) < 10,000 → Python sync dispatch
    │
    └── len(instructions) ≥ 10,000 → Rust VM adaptive path
            │
            serialize(bytecode) → .zxc bytes
            │
            RustVMExecutor.execute(zxc, env, state, gas_limit)
            │
            ├── Success → return result, bridge env+state+gas back
            ├── NeedsFallback → fall through to Python VM
            └── OutOfGas → raise OutOfGasError

Contract VM (contract_vm.py)
    │
    ├── Phase 3: _try_rust_vm_execution()  [Rust VM for large contracts]
    ├── Phase 1: .zxc cache lookup
    ├── Phase 0: bytecoded execution
    └── Fallback: tree-walk interpreter
```

### Files Added/Changed
- `src/zexus/vm/vm.py` — Rust VM import, threshold config, `_execute_via_rust_vm()`, `get_rust_vm_stats()`, adaptive routing in `_run_stack_bytecode_sync()`
- `src/zexus/blockchain/contract_vm.py` — Rust VM import, `_try_rust_vm_execution()`, stats integration
- `rust_core/src/state_adapter.rs` — `RustStateAdapter` PyO3 class (~280 lines)
- `rust_core/src/lib.rs` — Registered `state_adapter` module + `RustStateAdapter` class
- `tests/vm/test_phase3_adaptive.py` — 48 comprehensive tests
- `bench_phase3.py` — Performance benchmark script

---

## Phase 4 — Rust ContractVM Orchestration + Gas Optimizations ✅ COMPLETE
**Status:** Complete (2026-02-25)  
**Effort:** ~1 day  
**Risk:** Medium  

### Goal
Port the `ContractVM` orchestration layer (reentrancy guards, environment setup,
receipt generation, state commit/rollback) to Rust. Additionally, analyze and optimize
gas costs across both VMs to prevent cost inflation from the integrated VM pipeline.

### Results — Gas Optimizations
- **Gas analysis**: Benchmarked 7 contract patterns (token_transfer, dex_swap, nft_mint,
  staking_rewards, governance_vote, compute_loop, batch_transfers)
- **STATE_WRITE cost reduced**: 50 → 30 gas (RustStateAdapter caching makes writes memory-local)
- **STORE_FUNC aligned**: Python 5 → 3 (matched to Rust)
- **Rust gas discount**: 0.6× multiplier (40% cheaper in Rust, reflecting hardware-level efficiency)
- **Dynamic scaling**: BUILD_LIST (+1/element), BUILD_MAP (+2/pair), BUILD_SET (+1/element)
- **Static call optimization**: Read-only calls use `enable_gas_light = True` for reduced metering

### Gas Savings Summary

| Operation | Before | After | Savings |
|-----------|--------|-------|---------|
| STATE_WRITE | 50 gas | 30 gas | 40% |
| STORE_FUNC (Python) | 5 gas | 3 gas | 40% |
| Rust discount | 1.0× | 0.6× | 40% |
| Token transfer (Rust) | ~267 gas | ~160 gas | 40% |
| DEX swap (Rust) | ~645 gas | ~387 gas | 40% |

### Results — Rust ContractVM Orchestration
- **`RustContractVM`** PyO3 class (~633 lines, `contract_vm.rs`):
  - `execute_contract()`: Full lifecycle — reentrancy guard, call-depth check, state snapshot,
    VM creation with gas_discount, execution, state diff computation, receipt generation
  - `execute_batch()`: Sequential batch execution from list of call dicts
  - `get_stats()` / `reset_stats()`: Comprehensive stats tracking
  - Constructor: `RustContractVM(gas_discount=0.6, default_gas_limit=10_000_000, max_call_depth=10)`

- **Reentrancy detection**: Tracks executing contracts in a `HashSet<String>`, blocks re-entry
- **Call-depth enforcement**: Configurable `max_call_depth` (default 10), blocks deep nesting
- **State snapshot/rollback**: Clones state before execution, restores on failure
- **State diff computation**: Compares pre/post state to generate `state_changes` map
- **Receipt generation**: Success/error receipts with gas_used, gas_saved, state_changes,
  instructions_executed, output, revert_reason, needs_fallback

- **Python ContractVM integration** (`contract_vm.py`):
  - Phase 4 tier inserted in `execute_contract()` before Python execution path
  - `_try_rust_contract_vm()` method: compiles bytecode → .zxc, serializes state/env/args,
    calls `RustContractVM.execute_contract()`, merges state back, handles fallback
  - Stats tracking: `rust_contract_vm_available`, `rust_contract_vm_stats`

- **30/30 test cases passed** covering:
  - Import, init, gas_discount, stats, is_executing
  - Gas discount verification, STATE_WRITE cost, STORE_FUNC alignment
  - Simple execution, state write, state read/write, require failure
  - Reentrancy guard, call depth, out-of-gas, stats tracking, reset, receipt fields
  - Batch execution (success + mixed failures)
  - High-volume stress (10K transfers), state-heavy batch (1K writes), gas savings measurement
  - ContractVM integration (availability, stats)
  - Performance (>50K TPS simple, >50K TPS batch)
- **1769 total tests pass** (zero regressions from Phase 0-3)

### Benchmarks

| Metric | Value |
|--------|-------|
| Simple execution TPS | **>100,000** |
| Batch execution TPS | **>100,000** |
| 10K transfer stress | **>200,000 TPS** |
| Avg gas per transfer (Rust, discounted) | **~4 gas** |
| Gas savings vs full price | **~40%** |

### Architecture
```
ContractVM.execute_contract()
    │
    ├── Phase 4: _try_rust_contract_vm()
    │       │
    │       compile → .zxc → RustContractVM.execute_contract()
    │       │
    │       ├── Success → merge state, return receipt
    │       ├── NeedsFallback → fall through to Phase 3/0
    │       └── Error → rollback state, return error receipt
    │
    ├── Phase 3: _try_rust_vm_execution()  [large contracts]
    ├── Phase 1: .zxc cache lookup
    ├── Phase 0: bytecoded execution
    └── Fallback: tree-walk interpreter
```

### Files Added/Changed
- `rust_core/src/contract_vm.rs` — `RustContractVM` PyO3 class (~633 lines)
- `rust_core/src/rust_vm.rs` — Gas discount field, `consume_gas_dynamic()`, STATE_WRITE=30, pub helpers
- `rust_core/src/lib.rs` — Registered `RustContractVM` class
- `src/zexus/vm/gas_metering.py` — STATE_WRITE=30, STORE_FUNC=3
- `src/zexus/blockchain/contract_vm.py` — Phase 4 tier, `_try_rust_contract_vm()`, stats integration
- `tests/vm/test_phase4_contract_vm.py` — 30 comprehensive tests
- `bench_gas_stress.py` — Gas stress benchmark (7 contract patterns)

---

## Phase 5 — Eliminate GIL Callback in Batch Executor ⬅️ NEXT
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
| Phase 2 | ~1 day | 5,000-15,000 (20× speedup) |
| Phase 3 | ~1 day | 10,000-20,000 |
| Phase 4 | ~1 day | 15,000-30,000 |
| Phase 5 | ~1 week | 20,000-50,000 |
| Phase 6 | 2-3 weeks | up to 50,000 |
| **Total** | **~13-18 weeks** | **20,000-50,000** |

---

*Last updated: 2026-02-25*
