# Changelog

All notable changes to the Zexus programming language will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.8.3] - 2026-03-02

### 🐛 Bug Fixes — VM Parity (Session 6)

Ran all 24 `.zx` test files in `tests/v183_fixes/` through both interpreter and VM. All differences diagnosed and fixed — interpreter and VM now produce identical output on every test file.

**VM Execution Fixes:**
- **Boolean 0/1 printing** — `_op_load_const` integer pool lookup matched `True`/`False` as integers (`isinstance(True, int)` is True in Python). Added `not isinstance(value, bool)` guard.
- **Closure/lambda parameters null** — Compiler stored function params under `"params"` key but VM callable invocation read `"parameters"`. Fixed 3 lookup sites to check both keys.
- **Method calls on dict/list/str** — Added module-level `_DICT_METHODS`, `_LIST_METHODS`, `_STR_METHODS` dispatch tables (~30 helper functions) wired into both sync and async `CALL_METHOD` handlers. Covers: `has`, `keys`, `values`, `entries`, `size`, `delete`, `contains`, `push`, `pop`, `count`, `length`, `is_empty`, `first`, `last`, `reverse`, `sort`, `join`, `indexOf`, `slice`, `flatten`, `map`, `filter`, `reduce`, `startsWith`, `endsWith`, `toUpperCase`, `toLowerCase`, `trim`, `split`, `replace`, `substring`, `charAt`, `repeat`.
- **Integer division returns float** — `10 / 2` returned `5.0`. All 4 DIV handlers (sync, async dispatch, async fallback, fastops) now use `a // b if a % b == 0 else a / b` for int operands.
- **ForEach index + map iteration** — New `FOR_ITER` opcode with var_count parameter (1 for single var, 2 for index+item or key+value). Compiler emits `FOR_ITER` instead of generic iteration. Handlers added to all 3 VM paths + fastops. Supports list, dict, ZList, ZMap.
- **Try/catch doesn't catch runtime errors** — Division by zero silently returned `0` instead of raising. All DIV handlers now `raise VMRuntimeError("Division by zero")` which routes through the VM's `try_stack`.
- **Entity construction from plain dict** — `_construct_entity` only handled `ZMap`/`ObjMap` from `BUILD_MAP`. Added `isinstance(args_val, dict)` branch.
- **Map literal keys compiled as variable lookups** — `_compile_MapLiteral` called `_compile_node(key)` for Identifier keys, emitting `LOAD_NAME` (variable resolution → None) instead of `LOAD_CONST` (string literal). `{x: 10}` now correctly pushes string `"x"` as the key.
- **Entity field access returns null** — `_op_index` dispatch table in async path + fastops had no `EntityInstance` handling. Falls through to `obj[idx]` which fails. Added `hasattr(obj, 'data') and hasattr(obj, 'entity_def')` branch with `.get()` + `.value` unwrapping.

**Test Update:**
- Updated `test_018_division_by_zero_safety` to expect `VMRuntimeError` (was expecting silent `0`).

### 📁 Files Changed (Session 6)
- `src/zexus/vm/vm.py` — Boolean pool guard, params key fix, method dispatch tables, DIV fixes, FOR_ITER handlers, entity INDEX, entity construction
- `src/zexus/vm/compiler.py` — MapLiteral key emission fix, FOR_ITER opcode emission in ForEach
- `src/zexus/vm/fastops.pyx` — DIV raise, INDEX entity support, FOR_ITER handler
- `src/zexus/type_checker.py` — Skip per-argument type checks for entity brace-construction
- `tests/vm/test_comprehensive_vm_verification.py` — Division by zero test updated

### 🧪 Testing (Session 6)
- **24/24 v183_fixes `.zx` files: interpreter ↔ VM output identical**
- **2396 pytest tests pass, 0 failures**

---

### 🐛 Bug Fixes — Ziver Chain Phase 0 Audit (continued)

Resolves 19 issues from the [Phase 0 rewrite audit](issues/ISSUE8.md) — including 9 interpreter-level fixes, 8 parser fixes, 3 new builtins, and **3 VM-specific fixes** with hardening.

**VM-Specific Fixes (new):**
- **Entity field access returns `None` on VM** (R-001) — Added `_build_entity_definition()` creating proper `EntityDefinition` from bytecode, `_construct_entity()` for entity construction, `EntityInstance` support in `GET_ATTR`, and entity dispatch in both sync/async callable invocation.
- **Contract state not initialized on VM** (R-002) — Added `_compile_StateStatement` to the VM compiler, emitting `STATE_WRITE` + `STORE_NAME` for standalone `state x = value` declarations.
- **Complex programs produce no output on VM** (R-010) — Added `_vm_warn()` diagnostic system with configurable verbosity (`ZEXUS_VM_WARNINGS` env var). Replaced 15 silent `except Exception: return None` handlers with logged warnings.
- **`raise ZEvaluationError` crashed** — Created `VMRuntimeError(Exception)` class to replace non-raisable `ZEvaluationError(Object)` in all 15 raise sites.
- **`str()` / `length()` not found as builtins** — Added `str` and `length` as aliases in both VM `_FALLBACK_BUILTINS` and evaluator `_register_core_builtins`.
- **`append()` / `push()` didn't work on VM-native lists** — Added `_vm_native_call()` fast-path handling `push`, `append`, `length`, `len`, `str`, `string`, `range` directly on native Python types, bypassing ZList/ZMap wrapping issues.

**VM Hardening:**
- Stack overflow protection (`_MAX_STACK_DEPTH` configurable, default 50,000)
- Execution timeout (`_exec_timeout`, default 30s)
- Opcode limit (`_MAX_OPCODES`, default 100M, checked every 4096 ops)
- Configurable VM pool sizing (`_VM_POOL_MAX`)
- Rust VM status indicator in CLI, `zx-run`, and evaluator (shows active/available/not compiled)

**Critical / High (interpreter):**
- **`self` keyword not recognized** (R-003) — Added `"self"` as an alias for `"this"` in `eval_identifier()`. Both resolve to `__contract_instance__`.
- **`init()` not auto-called on construction** (R-004) — `SmartContract.instantiate()` now auto-calls `init()` after deployment when it exists.
- **`state { field: val }` didn't initialize defaults** (R-005) — Both parsers now correctly parse multi-field `state { }` blocks; evaluator initializes all state fields at deploy time.
- **Indexed `for each i, item in list` failed** (R-006) — Added `index` field to `ForEachStatement` AST node; updated both parsers and evaluator to bind index as `Integer`.
- **`for each key, val in map` failed** (R-007) — `eval_foreach_statement` now handles `Map` with key/value destructuring.
- **Multiple fields in `state { }` crashed** (R-013) — Both parsers detect `LBRACE` after `STATE` and parse comma-separated `name: value` pairs.
- **`INTEGER * FLOAT` type mismatch** (R-016) — Added mixed-numeric fallthrough in `eval_infix_expression` multiplication branch.
- **`%` with floats crashed** (R-017) — Added modulo operator to `eval_float_infix` with zero-division check.

**Closure / scope fixes (new in this release):**
- **Entity/let before contract broke visibility** (R-012, R-014) — Root cause: `clone_for_closure()` created an isolated snapshot with `outer=self.outer`, so identifiers registered after cloning were invisible. Fix: changed `outer=self.outer` → `outer=self` in `Environment.clone_for_closure()` so the cloned env delegates lookups to the live module environment. Contracts, entities, and let bindings are now visible regardless of declaration order.
- **Module-level helper side-effects dropped** (R-018, R-019) — Same root cause as R-012. Module-level actions captured a disconnected env clone; mutations (e.g., `list.push()`, `map[k]=v`) inside those helpers wrote to the clone's store, never propagating back. The `outer=self` fix ensures `assign()` walks up to the live module env, so side effects from helpers called inside contract methods now propagate correctly.
- **`action protect` failed at module level** (R-008) — Policy enforcement now runs in `apply_function` before executing any action. The evaluator checks for `__policy_<name>__` in the environment and calls `PolicyRegistry.check_policy()`.
- **`list.push()` silently ignored after `map[key]=val` in contract methods** (R-015) — Improved storage sync-back in `SmartContract.call_method()`: directly-updated vars now refresh `action_env` from storage after execution, and non-directly-updated vars prefer the storage reference when it differs from `action_env` (handles in-place mutations via `this.X` access).
- **Exported entities unusable as constructors** (R-011) — `eval_export_statement` now uses `val is None` instead of `not val`, preventing valid objects (empty lists, entity definitions, NULL placeholders) from being rejected.

**Parser fixes (found during 24-file testing):**
- Consecutive `this.property = val` assignments failed (THIS token not recognized as statement starter).
- `else if` chains didn't work (ELSE + IF token sequence not handled).
- Keywords after DOT (e.g., `this.debug`, `return this.data`) broke statement boundary detection.
- Entity default values (`name: string = "localhost"`) weren't parsed in either parser.
- Entity properties with keyword names (e.g., `debug: boolean`) weren't recognized.
- Entity default value boundary detection failed when next property used a keyword name.

**Evaluator fixes (found during 24-file testing):**
- Added missing `range(start, end, step)`, `typeof(val)`, and `abs(num)` built-in functions.

### 🔧 Build
- Bumped all version references from v1.8.1/v1.8.2 to v1.8.3 across all config, source, and documentation files.

### 📁 Files Changed
- `src/zexus/object.py` — `clone_for_closure()` now sets `outer=self` for live env delegation
- `src/zexus/evaluator/statements.py` — `eval_export_statement` uses `is None` check
- `src/zexus/evaluator/functions.py` — Policy enforcement in `apply_function()` for module-level `protect`; `str`/`length` builtin aliases
- `src/zexus/evaluator/expressions.py` — `self` keyword alias, mixed numeric ops, float modulo
- `src/zexus/evaluator/core.py` — Rust VM status indicator in `_initialize_vm()`
- `src/zexus/security.py` — Storage sync-back improvements in `call_method()`, `init()` auto-call
- `src/zexus/zexus_ast.py` — `ForEachStatement` index field
- `src/zexus/parser/parser.py` — Multi-field `state {}`, indexed for-each, entity defaults, keyword-name props
- `src/zexus/parser/strategy_context.py` — Same parser fixes (strategy parser)
- `src/zexus/evaluator/statements.py` — `eval_foreach_statement` map iteration, state init at deploy
- `src/zexus/vm/vm.py` — `VMRuntimeError`, `_vm_warn()`, `_vm_native_call()`, `_build_entity_definition()`, `_construct_entity()`, EntityInstance GET_ATTR, hardening (stack/timeout/opcode/pool), Rust VM status, `str`/`length` fallback aliases
- `src/zexus/vm/compiler.py` — `_compile_StateStatement` for standalone state declarations
- `src/zexus/cli/main.py` — Rust VM status indicator after VM init
- `zx-run` — Rust VM status indicator

### 🧪 Testing
- **1852 tests pass**, 0 regressions
- 24 `.zx` integration test files covering all R-xxx fixes and parser edge cases
- 2 new test files: `tests/test_issue8_fixes.zx`, `tests/test_issue8_advanced.zx`
- 5 new extreme test files:
  - `tests/extreme_speed_test.zx` — 10 speed tests (100K loops, 250K nested, Fibonacci, string concat, list build+sum, maps, factorial, function calls, bubble sort, prime sieve)
  - `tests/extreme_stability_test.zx` — 15 stability tests (deep nesting, error recovery, null handling, recursion, closures, allocation stress)
  - `tests/extreme_security_test.zx` — 20 security tests (large integers, injection, type coercion, bounds, OOM, contract isolation)
  - `tests/extreme_features_test.zx` — 20 feature tests (entities, contracts, lambdas, closures, lists, maps, strings, math, loops, recursion)
  - `tests/extreme_vm_test.zx` — 18 VM-specific tests (R-001 entity, R-002 state, R-010 output, hardening stress)


## [1.8.2] - 2026-02-25

### 🐛 Bug Fixes

**Concurrency (Part 3 hang fix):**
- **`watch =>` triggered arrow-lambda fallback** — The parser's arrow-lambda detection (`=>`) treated `watch x =>` as a lambda, forcing the entire file through the traditional parser and losing advanced strategy parsing (channels, async actions, SPAWN). Fixed by excluding `watch` patterns from the arrow-lambda check.
- **Channel support in VM compiler** — Added `_compile_ChannelStatement`, `_compile_SendStatement`, and `_compile_ReceiveStatement` to `vm/compiler.py` for proper channel bytecode emission.
- **Channel builtins in VM runtime** — Registered `__create_channel__`, `send`, `receive`, `close_channel` builtins in `vm/vm.py` that delegate to the `Channel` class from `concurrency_system.py`.
- **SPAWN/AWAIT in sync path** — Added `SPAWN` and `AWAIT` opcode handlers to `_run_stack_bytecode_sync` using `threading.Thread(daemon=True)` for background execution.
- **Strategy parser CHANNEL/ASYNC handlers** — Added `CHANNEL` and `ASYNC` to `STATEMENT_STARTERS`, implemented channel token collection in `_parse_block_statements`, and fixed the ASYNC handler to detect `async action` patterns.
- **`parse_type_expression()` missing** — Added the missing method to `parser.py`, fixing a crash when parsing `channel<type>` syntax.
- **`parse_channel_statement()` bracket syntax** — Fixed to handle `channel<type>[capacity] name` with bracket capacity before the channel name.

**VM & CLI:**
- Fixed VM fallback issues for 10+ missing node types (DateTime arithmetic, type coercion, SETUP_TRY label resolution, child VM `_call_depth`).
- Zero VM fallbacks achieved on all test files.
- Fixed CLI `pip`/`npm` commands and cleaned `__pycache__`.
- Added auto-install for missing dependencies.

### 🔧 Build
- Bumped all versions to 1.8.2 across 9 files.
- Confirmed Rust VM (`zexus_core`) builds and loads with `maturin develop --release`.


## [1.8.1] - 2026-02-23

### 🐛 Bug Fixes — Ziver Chain Phase 0 Audit

All 21 issues discovered during the Ziver Chain Phase 0 audit ([ISSUE7](issues/ISSUE7.md)) have been resolved.

**Critical (P0):**
- **`emit` keyword broken** (INT-003) — Added EMIT to all three parser statement-starter sets, context rules, and `_parse_block_statements` handler. Implemented `_parse_emit_statement_block` in the context parser. Blockchain event emission now works inside and outside contracts.
- **`protocol` keyword not recognized** (INT-001) — Added PROTOCOL to both strategy parser starters, context rules, traditional parser dispatch, and implemented `parse_protocol_statement` / `_parse_protocol_statement_block` methods. Fixed `eval_protocol_statement` to handle both string and AST method entries.
- **`implements` keyword broken** (INT-002) — Fixed `_parse_contract_statement_block` to detect `implements` between contract name and opening brace, passing the protocol name to ContractStatement.

**High (P1):**
- **Entity compilation crash in VM** (VM-001) — Rewrote `_compile_EntityStatement` in `vm/compiler.py` to use `node.properties` and `node.methods` (the actual AST attributes) instead of non-existent `node.body`.
- **Imported functions return Action objects in VM** (VM-003) — Fixed `_call_builtin_async_obj` in `vm/vm.py` — changed broad `except Exception: pass` to properly detect ZAction/ZLambda and return None instead of falling through to return the raw function object.
- **`map.has()` / `map.get()` broken** (INT-013/INT-014) — Fixed key normalization in `evaluator/functions.py` to try both plain string and String-wrapped keys for Map lookup.

**Medium (P2):**
- **ExportStatement unsupported in VM** (VM-002) — Added `_compile_ExportStatement` to both `vm/compiler.py` and `evaluator/bytecode_compiler.py`.
- **Entity constructor argument mismatch** (VM-004) — Added `_check_EntityStatement` to `type_checker.py` to register entities as constructors and skip strict arity validation for single-map-arg entity construction.
- **`string()` + builtin module returns type name in VM** (VM-005) — Fixed CALL_METHOD handler (both sync and async paths) to invoke `Builtin.fn` when module methods are stored as Builtin objects in dicts.
- **`bc.create_genesis_block()` returns function object in VM** (INT-012) — Same root cause and fix as VM-005.
- **`persistent storage` no-op** (INT-004) — Rewrote `eval_persistent_statement` to initialize PersistentStorage SQLite backend, use `set_persistent`/`get_persistent`, and restore persisted values on startup. Fixed broken `sys.modules` guard in `_init_persistence`.

**Low (P3):**
- **Missing directive builtins** (INT-005 through INT-009) — Registered `track_memory()`, `cache()`, `throttle()`, `audit()`, and `verify()` as builtin functions via new `_register_missing_directive_builtins()` method.
- **`watch variable { ... }` parse error** (INT-010) — Added Form 3 (expression followed by block, no `=>` arrow) to both the context and traditional parsers.
- **`list.is_empty()` missing** (INT-015) — Added `is_empty` method to List method dispatch.
- **`list.count()` inconsistency** (INT-016) — Confirmed already working; no change needed.
- **`protect` modifier syntax** (INT-011) — Clarified: `protect(target, {rules})` call syntax is the correct form and already works.

### 🔒 Security & Quality Remediation (Phases 0–5)

This release also includes the complete remediation work tracked in `VULNERABILITIES_FIXES.md`:

- **Phase 0 (Critical):** sandboxing + path safety + dangerous native/exec gating + gas metering guards.
- **Phase 1 (High):** ReDoS protections, stronger import/module sandboxing, unsafe cache reuse removal.
- **Phase 2 (Logic bugs):** missing opcode handlers, safer TX exception behavior, jump patching correctness.
- **Phase 3 (Compiler consistency):** evaluator/VM compiler parity (EXPORT/print multi-value/destructuring/continue semantics).
- **Phase 4 (Medium):** dead-code removals, safer cleanup/error handling, bounded logs/ledgers, VM pool state scrubbing.
- **Phase 5 (Low/Info):** performance cleanups (no busy-wait await), bounded env serialization, safer input sanitization, token validation heuristic.

### 📦 Installation Improvements

- Added `zexus[full]` extra to install common optional dependencies for blockchain/network/security helpers.
- `install.sh` now installs `.[full]` and best-effort builds the optional Rust VM (`zexus_core`) when `cargo` is available.
- npm `postinstall` installs `zexus[full]` and best-effort builds `zexus_core` when Rust toolchain is available.

### 📁 Files Changed

- `src/zexus/parser/strategy_context.py` — emit, protocol, implements, watch form 3
- `src/zexus/parser/strategy_structural.py` — emit, protocol statement starters
- `src/zexus/parser/parser.py` — protocol dispatch, watch form 3
- `src/zexus/vm/compiler.py` — EntityStatement rewrite, ExportStatement
- `src/zexus/vm/vm.py` — async call fix, Builtin.fn dispatch in CALL_METHOD
- `src/zexus/evaluator/bytecode_compiler.py` — ExportStatement, EmitStatement, ProtocolStatement
- `src/zexus/evaluator/functions.py` — map.has/get key normalization, list.is_empty, 5 new builtins
- `src/zexus/evaluator/statements.py` — persistent storage wiring, protocol evaluator
- `src/zexus/object.py` — persistence init fix, memory tracking init fix
- `src/zexus/type_checker.py` — entity constructor registration and flexible arity

### 🧪 Testing

- **1852 tests pass**, 0 regressions
- Integration test covers: track_memory, cache, throttle, audit, verify, list.is_empty, list.count, map.has, map.get, protocol/implements, emit, watch

---

## [1.8.0] - 2026-02-23

### ✨ Language Features

**Syntax Additions:**
- **Compound assignment operators** — `+=`, `-=`, `*=`, `/=`, `%=` with lexer tokens, parser infix rules, and eval handler
- **String interpolation** — `"Hello ${name}"` via extended `read_string()` in the lexer
- **Block comments** — `/* */` support via extended `skip_whitespace()` in the lexer
- **Multiline strings** — triple-quote `"""..."""` support
- **Single-quoted strings** — `'...'` now recognized by the lexer
- **Exponentiation operator** — `**` with lexer token, parser rule, and existing VM `POW` opcode
- **`finally` clause** — added to `TryCatchStatement` AST node and evaluator
- **Destructuring assignment** — `let {a, b} = map; let [x, y] = list` with map, list, rename, and rest patterns in parser, evaluator, and VM

### 🛠 Developer Tooling

- **Circular import detection** — `_LOADING_SET` with `begin_loading`/`end_loading` guards in evaluator + VM module loading
- **LSP go-to-definition** — recursive AST walker, symbol provider, and server parsing fix for jump-to-source in editors
- **Remote ZPM registry** — HTTP client with auth, tarball download/extraction in registry + installer (replaces stubbed registry)
- **Static type checking pass** — `StaticTypeChecker` AST walker with scoped symbol table, parameter/return/assignment type checking (optional pre-runtime analysis)

### 🎯 Major New Capabilities

- **Debug Adapter Protocol (DAP)** — Full DAP JSON wire protocol server (`dap_server.py`), `DebugEngine` state machine with breakpoints/stepping/pause, VS Code adapter shim, `eval_node()` hook with zero-cost when inactive, AST line/column tracking
- **GUI Backend** — `TkBackend` (tkinter: windows, buttons, labels, text boxes, canvas) + `WebBackend` (HTML/CSS/SVG generation via stdlib `http.server`, event relay via POST), lazy-loaded in `renderer.__init__`
- **True Concurrent EventLoop** — persistent shared `asyncio.EventLoop` on dedicated background thread (`event_loop.py`), `submit()`/`spawn()`/`shutdown()` API, evaluator `_resolve_awaitable` and `eval_async_expression` ported from raw threads, VM `_run_coroutine_sync` ported, `AsyncChannel` with `asyncio.Queue`
- **WASM Compilation Target** — `WasmCompiler` translates Zexus `Bytecode` → valid `.wasm` binary (LEB128 encoding, all sections, i64 stack), supports arithmetic/comparisons/logic/control-flow/locals/POW, CLI `zx compile --target wasm`

### 🔗 Blockchain Production Hardening

- **CA-signed TLS, mTLS, certificate pinning** for P2P networking (`network.py`)
- **Pluggable storage backends** — SQLite, LevelDB, and RocksDB support (`storage.py`)
- **Production monitoring** — Prometheus metrics integration (`monitoring.py`)
- **Load testing framework** — TPS validation with latency percentiles (`loadtest.py`)
- **Dependency audit** — actionable install warnings for missing optional packages (`__init__.py`)

### 🚀 Rust-First Execution Pipeline (Phases 0–6)

Complete migration of contract execution from Python to Rust, achieving **102,000+ TPS** with zero Python fallbacks for builtin-heavy contracts.

#### Phase 0 — Bytecode Compilation for Contracts
- Added `use_bytecode_vm` feature flag to `ContractVM` enabling bytecoded execution with automatic tree-walk fallback
- Shared gas metering, builtins, and blockchain state between contract VM and evaluator VM
- Execution stats tracking via `get_vm_execution_stats()`
- **17/17 test cases passed** — 100% bytecode rate, zero fallbacks

#### Phase 1 — Binary Bytecode Format (`.zxc`)
- Defined `.zxc` binary bytecode specification: 16-byte header, typed constant pool (9 tags + OPAQUE), variable-width instructions, CRC32 checksum
- Python serializer/deserializer: `serialize()`, `deserialize()`, `save_zxc()`, `load_zxc()`
- Multi-bytecode container support for file-level caching
- GIL-free Rust deserializer via `RustBytecodeReader` PyO3 class
- Wired into VM module loading, disk cache, and contract VM per-action caching
- **20:1 compression** vs Python tuple format; **43/43 tests passed**

#### Phase 2 — Rust Bytecode Interpreter
- Complete Rust stack-machine bytecode interpreter (`rust_vm.rs`, ~1,600 lines)
- `ZxValue` enum with all runtime types; `Op` enum mapping ~50 opcodes with per-opcode gas costs
- Full opcode coverage: stack ops, arithmetic, comparison, logical, control flow, collections, blockchain ops, exception handling
- Nested transaction support with snapshot/rollback
- **22 MIPS throughput** — **20.7× speedup** over Python VM; **70/70 tests passed**

#### Phase 3 — Adaptive VM Routing + Rust State Adapter
- Adaptive routing: programs ≥10,000 ops automatically delegate to Rust VM (configurable via `ZEXUS_RUST_VM_THRESHOLD`)
- Transparent fallback for unsupported ops (e.g., `CALL_NAME`); runtime toggle via `vm._rust_vm_enabled`
- `RustStateAdapter` PyO3 class (~280 lines): in-memory HashMap cache with dirty-tracking, bulk load/flush, nested transactions
- Gas metering bridge: Python passes gas budget to Rust, Rust `gas_used` bridges back
- Contract VM integration with Phase 3 tier before bytecoded execution
- **48/48 tests passed** — up to **55× speedup** for direct Rust execution

#### Phase 4 — Rust ContractVM Orchestration + Gas Optimizations
- `RustContractVM` PyO3 class (~633 lines): full lifecycle with reentrancy guard, call-depth check, state snapshot, gas discount, receipt generation
- Batch execution via `execute_batch()` with sequential state chaining
- **Gas optimizations**: STATE_WRITE 50→30 gas (40%), STORE_FUNC 5→3, Rust 0.6× discount (40% cheaper)
- State diff computation and comprehensive receipt fields (gas_used, gas_saved, state_changes, output, revert_reason)
- **>100,000 TPS** simple execution, **>200,000 TPS** stress; **30/30 tests passed**

#### Phase 5 — GIL-Free Batch Execution
- `execute_batch_native()` on `RustBatchExecutor` — pure-Rust parallel batch with **zero GIL acquisitions** during VM execution
- Single GIL touch to parse input, then `py.allow_threads()` for entire compute phase
- Rayon `par_iter()` across contract groups with sequential state chaining within groups
- `AtomicU64` counters for lock-free gas/stats aggregation
- Priority 0 tier in `accelerator.py`; backward compatible with existing `execute_batch()`
- **Up to 3.6× speedup** over batched-GIL; **221,593 TPS** peak; **23/23 tests passed**

#### Phase 6 — Rust Builtins (Zero Python Fallback)
- Ported all 40+ contract builtins to Rust: crypto (keccak256, sha256, verify_sig), state (transfer, get_balance), events (emit), block info, type/conversion, collection, and string builtins
- `dispatch_builtin()` match-based dispatch + `KNOWN_BUILTINS` static check
- `CALL_NAME` handler checks `is_known_builtin()` first, fallback only for unknown names
- Balance model with underflow/overflow protection
- **191,148 ops/s** single keccak256, **102,320 TPS** batch CALL_NAME dispatch, **0 Python fallbacks**; **59/59 tests passed**

### ⚡ Performance

| Metric | Value |
|--------|-------|
| Single keccak256 (Rust VM) | **191,148 ops/s** |
| Batch keccak256 (1,000 txns) | **72,414 TPS** |
| Mixed builtins (keccak + emit + transfer) | **34,007 TPS** |
| String builtins (upper + replace + split) | **75,420 TPS** |
| CALL_NAME dispatch (keccak256) | **102,320 TPS** |
| Rust VM throughput | **22 MIPS** (20.7× over Python) |
| GIL-free batch peak | **221,593 TPS** |
| Python fallbacks | **0** |
| VM overhead per call | **3.7 µs** |

### 🔧 Gas Optimizations
- `STATE_WRITE` cost reduced from 50 to 30 gas (40% savings)
- `STORE_FUNC` aligned: Python 5→3 gas (matched to Rust)
- Rust gas discount: 0.6× multiplier (40% cheaper reflecting hardware-level efficiency)
- Dynamic scaling for `BUILD_LIST` (+1/element), `BUILD_MAP` (+2/pair), `BUILD_SET` (+1/element)
- Static call optimization: read-only calls use `enable_gas_light` for reduced metering

### 📦 New Files
- `rust_core/src/rust_vm.rs` — Rust bytecode interpreter (~1,600 lines)
- `rust_core/src/state_adapter.rs` — `RustStateAdapter` PyO3 class (~280 lines)
- `rust_core/src/contract_vm.rs` — `RustContractVM` PyO3 class (~633 lines)
- `rust_core/src/binary_bytecode.rs` — Rust GIL-free `.zxc` deserializer (~544 lines)
- `src/zexus/vm/binary_bytecode.py` — Python `.zxc` serializer/deserializer (~660 lines)
- `src/zexus/dap_server.py` — Full DAP JSON wire protocol server
- `src/zexus/renderer/` — TkBackend + WebBackend GUI renderers
- `src/zexus/event_loop.py` — Persistent asyncio EventLoop on background thread
- `src/zexus/wasm_compiler.py` — WASM compilation target
- `src/zexus/static_type_checker.py` — Static type checking pass
- `src/zexus/blockchain/monitoring.py` — Prometheus metrics integration
- `src/zexus/blockchain/loadtest.py` — Load testing framework
- `src/zexus/blockchain/storage.py` — Pluggable storage backends
- `tests/vm/test_rust_vm.py` — 70 tests
- `tests/vm/test_binary_bytecode.py` — 43 tests
- `tests/vm/test_phase3_adaptive.py` — 48 tests
- `tests/vm/test_phase4_contract_vm.py` — 30 tests
- `tests/vm/test_phase5_gil_free.py` — 23 tests
- `tests/vm/test_phase6_builtins.py` — 59 tests
- `tests/blockchain/test_phase0_bytecode_vm.py` — 17 tests

### 🔄 Changed Files
- `src/zexus/lexer.py` — Compound assignment tokens, `**` operator, block comments, string interpolation, single-quoted strings, multiline strings
- `src/zexus/zexus_ast.py` — `finally` clause on `TryCatchStatement`, destructuring AST nodes
- `src/zexus/parser/` — Compound assignment infix rules, destructuring patterns, `**` precedence, `finally` parsing
- `src/zexus/evaluator/` — Compound assignment handler, `finally` execution, destructuring eval, circular import guards, async EventLoop integration
- `src/zexus/vm/vm.py` — Destructuring opcodes, circular import detection, Rust VM adaptive routing, threshold config, module `.zxc` loading
- `src/zexus/vm/cache.py` — Disk persistence uses `.zxc` binary format
- `src/zexus/vm/gas_metering.py` — STATE_WRITE=30, STORE_FUNC=3
- `src/zexus/lsp/` — Go-to-definition symbol provider, AST walker
- `src/zexus/zpm/` — Remote registry HTTP client, auth, tarball download/extraction
- `src/zexus/blockchain/contract_vm.py` — Phase 0–4 bytecoded/Rust execution tiers, `.zxc` caching, chain info injection, event collection
- `src/zexus/blockchain/rust_bridge.py` — `execute_batch_native()` GIL-free API
- `src/zexus/blockchain/accelerator.py` — Priority 0 GIL-free native tier
- `src/zexus/blockchain/network.py` — CA-signed TLS, mTLS, certificate pinning
- `rust_core/src/executor.rs` — Native batch execution, events in receipts
- `rust_core/src/lib.rs` — Registered `binary_bytecode`, `rust_vm`, `state_adapter`, `RustContractVM` modules
- `rust_core/Cargo.toml` — Updated dependencies

### 🧪 Testing
- **290 new tests** across Phases 0–6 (17 + 43 + 70 + 48 + 30 + 23 + 59)
- **1,792+ total tests pass** with zero regressions

---

## [1.7.2] - 2026-01-14

### ⚡ Performance
- **Major Interpreter Speed Improvements**: Eliminated debug logging overhead in hot paths (eval_node, eval_identifier)
- **Smart Storage for Lists**: Implemented StorageList with dirty-tracking to avoid O(N²) serialization bottleneck
- **Optimized Stack Traces**: Deferred string formatting until errors occur, storing lightweight tuples instead
- **Debug Config Caching**: Added fast_debug_enabled boolean cache to eliminate dictionary lookups
- **Blockchain Performance**: 10,000 transaction test now completes in ~2 minutes (previously timed out)

### 🔧 Fixes
- Fixed performance regression affecting blockchain contract state persistence
- Resolved exponential slowdown in chain growth scenarios

---

## [1.7.1] - 2026-01-07

### ✨ Features
- Added bytecode compiler handlers for `FindExpression` and `LoadExpression`, enabling VM execution of the new FIND/LOAD keywords without leaving the evaluator path (`src/zexus/evaluator/bytecode_compiler.py`).
- Injected VM keyword bridges so compiled bytecode can reuse evaluator semantics, ensuring identical behavior across interpreter and VM modes (`src/zexus/evaluator/core.py`).

### 🧪 Testing
- Introduced VM-focused regression tests that exercise FIND and LOAD in both interpreter and VM execution modes (`tests/unit/test_find_load_keywords.py`).

### 📚 Documentation
- Updated README highlights for v1.7.1, covering the FIND/LOAD keywords, provider-aware LoadManager, and VM parity improvements.

---

## [1.6.8] - 2026-01-06

### 🐛 Bug Fixes

**Parser - Indexed Assignment on New Line:**
- **Fixed "Invalid assignment target" error for indexed assignments on new lines**
  - Previously code like `let data = obj.method(); data["key"] = value` would fail
  - Parser was incorrectly treating both lines as a single `let` statement
  - Added newline-aware indexed assignment detection in two locations:
    - `strategy_structural.py`: Added Pattern 3 for `IDENT[...]` detection (~line 697-730)
    - `strategy_context.py`: Added indexed assignment check in LET heuristic (~line 2377-2393)
  - Both fixes detect `IDENT LBRACKET ... RBRACKET ASSIGN` pattern on new lines
  - Files: `src/zexus/parser/strategy_structural.py`, `src/zexus/parser/strategy_context.py`
  - Example: Multi-line data manipulation in contracts now works correctly

**Parser - Contract DATA Member Declarations:**
- **Fixed contract parser skipping DATA member declarations**
  - Contract parser only handled `STATE`, `persistent storage`, and `ACTION` keywords
  - `DATA` keyword declarations were being skipped, causing first action to be missed
  - Added DATA keyword handling in `parse_contract_statement()` to create LetStatement nodes
  - Now properly initializes: `data name = "value"`, `data balance = 0`, `data items = {}`
  - Files: `src/zexus/parser/parser.py` (~line 3130-3148)
  - Impact: All contract data members now properly initialized and accessible in actions
  - Example: Smart contract wallets, tokens, and bridges now work correctly

**Test Backend - String Concatenation:**
- Fixed type mismatch errors when concatenating strings with potentially NULL values
- Added `string()` conversion for safe concatenation in logging statements
- Files: `test_backend_project/auth/session.zx`, `test_backend_project/tasks/task_service.zx`, `test_backend_project/main.zx`

### ✨ Features

**Blockchain Test Suite:**
- Added comprehensive blockchain test demonstrating smart contract capabilities
- Includes ERC20-like token contract with minting, transfers, and balance tracking
- Wallet contract with multi-step transfer flows (A→B→C→D→E)
- Cross-chain bridge contract with fee calculation and locked token tracking
- Full test coverage with sequential transfers and state validation
- Files: `blockchain_test/token.zx`, `blockchain_test/wallet.zx`, `blockchain_test/bridge.zx`, `blockchain_test/run_test.zx`
- Test results documented in `blockchain_test/TEST_RESULTS.md`

### 📝 Documentation
- Updated ISSUE5.md with complete fix documentation (Fix #5 and Fix #6)
- Added TEST_RESULTS.md documenting blockchain test execution and parser fixes

---

## [1.6.7] - 2026-01-03

### 🐛 Bug Fixes

This release resolves critical parser bugs affecting statement parsing, keyword usage, and block statements.

#### Fixed

**Parser Semicolon Handling:**
- **Semicolon Token Inclusion Bug** - Fixed semicolons being incorrectly included in subsequent statements
  - Previously semicolons were included as first token of next statement in contract action bodies
  - This caused "Invalid assignment target" errors when multiple assignments followed print statements
  - Applied fix across 9 locations in statement parsing (PRINT, LET, CONST, RETURN, REQUIRE, expressions)
  - Special handling for PRINT statements that break on RPAREN before seeing semicolon
  - File: `src/zexus/parser/strategy_context.py` (multiple locations)
  - Example: Contract actions with `print(...); accounts[user]["balance"] = ...;` now work correctly

- **Multiple Nested Assignments in Contracts** - Fixed contract actions with multiple nested map operations
  - Previously second nested assignment would fail with "Invalid assignment target"
  - Parser now properly skips semicolons when advancing to next statement
  - Contract state modifications with multiple nested map updates now fully functional
  - Example: `accounts[user]["balance"] = ...; accounts[user]["transactions"] = ...;` works correctly

**Keywords as Variable Names:**
- **Context-Aware Keyword Recognition** - Keywords can now be used as variable names in appropriate contexts
  - Previously keywords like `data`, `action`, `state` couldn't be used as variable names
  - Implemented context-aware lexer that tracks previous token to determine valid identifier usage
  - Keywords allowed as identifiers after: LET, CONST, COLON, COMMA, operators, brackets, etc.
  - Exceptions: `true`, `false`, `null` always remain strict keywords for literals
  - File: `src/zexus/lexer.py`
  - Example: `let data = 42; let action = "click";` now works correctly

**Standalone Block Statements:**
- **Block Statement Parsing** - Fixed standalone code blocks causing "Invalid assignment target" errors
  - Previously blocks like `{ let x = 10; }` were incorrectly parsed as assignment expressions
  - Added dedicated block statement handling with proper nesting and recursive parsing
  - File: `src/zexus/parser/strategy_context.py` (~line 3450)
  - Example: `{ let temp = 10; print(temp); }` now works correctly

#### Testing
- All 7 ISSUE4.md fixes working perfectly
- Comprehensive edge case test suite (test_edge_cases.zx): 19/19 tests passing
  - ✅ Deeply nested maps (3+ levels)
  - ✅ Mixed statement types in blocks
  - ✅ Conditionals with nested assignments
  - ✅ Loops with compound assignments
  - ✅ Contracts with multiple state variables
  - ✅ Sequential method calls
  - ✅ Complex expressions with operator precedence
  - ✅ Multiple prints with assignments in contracts
  - ✅ Complex map reconstruction
  - ✅ Multiple contract instances (isolation)
  - ✅ Try-catch with assignments
  - ✅ Complex return statements
  - ✅ Array-like map operations
  - ✅ Compound operator sequences
  - ✅ Conditional (ternary) assignments

#### Documentation
- Added PARSER_SEMICOLON_FIX.md with detailed technical explanation
- Added ADDITIONAL_FIXES_1.6.7.md documenting all fixes and known limitations
- Documents root cause, fix locations, test results, and workarounds

#### Known Limitations (Documented)
- Contract instances cannot be stored in contract state (use parameters instead)
- Nested map literals in contract state may not persist reliably (use simple values or incremental assignment)

---

## [1.6.6] - 2026-01-02

### 🐛 Bug Fixes

This release resolves the final critical parser issues for full production readiness.

#### Fixed

**Parser Statement Boundary Detection:**
- **Contract Repeated Action Calls** - Fixed contract methods being callable multiple times
  - Previously only first call to same contract action would execute
  - Added method call statement detection with dot pattern matching
  - Method calls now properly break on newlines as separate statements
  - File: `src/zexus/parser/strategy_structural.py` (lines 733-749)
  - Example: `counter.increment()` can now be called multiple times successfully

- **Multiple Indexed Assignments** - Fixed multiple map assignments in same function
  - Previously second indexed assignment would fail with "Invalid assignment target"
  - Enhanced indexed assignment detection to recognize `identifier[key] = value` pattern
  - Consecutive indexed assignments now parse as separate statements
  - File: `src/zexus/parser/strategy_structural.py` (lines 733-749)
  - Example: Multiple `storage["key"] = value` statements now work correctly

#### Notes
- **Reserved Keywords**: `DATA` and `STORAGE` are reserved keywords. Use alternative names like `mydata`, `mystore` for variables.

#### Testing
- All 6 core features now 100% functional
- Contract-based and module-based patterns both fully supported
- Comprehensive testing performed and documented in `issues/ISSUE3.md`

---

## [1.6.5] - 2026-01-02

### 🐛 Bug Fixes

This release resolves critical parser and evaluator issues that were blocking production use.

#### Fixed

**Parser & Evaluator Fixes:**
- **Entity Property Access** - Fixed dataclass constructor to properly handle MapLiteral syntax
  - `Block{index: 42}` now correctly extracts field values
  - Property access `block["index"]` now returns field value instead of entire object
  - Enhanced constructor to detect single Map argument and convert to kwargs
  - File: `src/zexus/evaluator/statements.py` (lines 327-380)

- **Keyword Restrictions** - Removed 'from' from reserved keywords list
  - Can now use `from` and `to` as parameter names: `action transfer(from, to, amount)`
  - Parser still recognizes `from` contextually in import statements
  - No more syntax errors when using natural parameter names
  - File: `src/zexus/lexer.py` (line 479)

- **Environment Method Error** - Fixed missing `set_const()` method calls
  - Replaced all `env.set_const()` calls with `env.set()`
  - Fixed AttributeError crashes in const and data statements
  - Files: `src/zexus/evaluator/statements.py` (lines 224, 708)

- **Multiple Map Assignment Parser Bug** - Enhanced statement boundary detection
  - Fixed parser incorrectly combining consecutive indexed assignments
  - Multiple `map[key] = value` statements on consecutive lines now work correctly
  - Added indexed assignment pattern detection: `IDENT[...] = ...`
  - Added newline-aware statement separation
  - No longer need semicolons or workarounds for multiple assignments
  - File: `src/zexus/parser/strategy_context.py` (lines 3387-3430)

#### Verified Working

**Features Confirmed Operational:**
- **Contract State Persistence** - Contract state correctly persists between action calls (was already working)
- **Module Variable Reassignment** - Can reassign module-level variables inside functions (was already working)

#### Testing

**Test Suite:**
- All tests pass successfully
- Test files: `test_fixes_final.zx`, `test_entity_property.zx`, `test_module_var_reassign.zx`, `test_debug_contract.zx`
- Comprehensive validation of:
  - Map operations and persistence
  - Token transfers with multiple assignments
  - Entity/data property access
  - Module variable reassignment

#### Impact

**Production Readiness:**
- ✅ All critical bugs resolved
- ✅ Smart contracts fully functional
- ✅ Entity/data types work correctly
- ✅ Natural parameter naming (from/to)
- ✅ Multiple map assignments without workarounds
- ✅ Ready for real-world blockchain development

#### Documentation

**Updated:**
- `issues/ISSUE2.md` - Complete fix documentation with code examples
- Status changed from "Partially Functional (50%)" to "Fully Functional (100%)"

---

## [1.6.3] - 2026-01-02

### 🔒 Security Enhancements

This release includes **comprehensive security remediation** addressing all 10 identified critical vulnerabilities. Zexus is now enterprise-ready with industry-leading security features built into the language.

#### Added

**Security Features:**
- **Input Sanitization System** - Automatic tainting of external inputs (stdin, files, HTTP, database)
  - Smart SQL/XSS/Shell injection detection with 90% reduction in false positives
  - Mandatory `sanitize()` function before dangerous operations
  - Context-aware validation (SQL, HTML, URL, Shell)
  
- **Contract Access Control (RBAC)** - Complete role-based access control system
  - Owner management: `set_owner()`, `get_owner()`, `is_owner()`, `require_owner()`
  - Role management: `grant_role()`, `revoke_role()`, `has_role()`, `require_role()`
  - Permission management: `grant_permission()`, `revoke_permission()`, `has_permission()`, `require_permission()`
  - Transaction context via `TX.caller`
  - Multi-contract isolation with audit logging

- **Cryptographic Functions** - Enterprise-grade password hashing and secure random generation
  - `bcrypt_hash(password)` - Bcrypt password hashing with automatic salt generation
  - `bcrypt_verify(password, hash)` - Secure password verification
  - `crypto_rand(num_bytes)` - Cryptographically secure random number generation (CSPRNG)

- **Debug Info Sanitization** - Automatic protection against credential leakage
  - Automatic masking of passwords, API keys, tokens, database credentials
  - Production vs development mode detection via `ZEXUS_ENV`
  - Environment variable protection
  - Stack trace sanitization in production mode
  - File path sanitization

- **Type Safety Enhancements** - Strict type checking to prevent implicit coercion vulnerabilities
  - ⚠️ **BREAKING CHANGE**: String + Number now requires explicit `string()` conversion
  - Integer + Float mixed arithmetic still allowed (promotes to Float)
  - Type conversion functions: `string()`, `int()`, `float()`
  - Clear error messages with actionable hints

- **Resource Limits** - Protection against resource exhaustion and DoS attacks
  - Maximum loop iterations (1,000,000 default, configurable)
  - Maximum call stack depth (1,000 default, configurable)
  - Execution timeout (30 seconds default, configurable)
  - Storage limits: 10MB per file, 100MB total (configurable)
  - All limits configurable via `zexus.json`

- **Path Traversal Prevention** - Automatic file path validation
  - Whitelist-based directory access control
  - Automatic detection and blocking of `../` patterns
  - Protection against symlink attacks

- **Contract Safety** - Built-in precondition validation
  - `require(condition, message)` function with automatic state rollback
  - Contract invariant enforcement
  - Custom error messages

- **Integer Overflow Protection** - Arithmetic safety
  - 64-bit signed integer range enforcement (-2^63 to 2^63-1)
  - Automatic overflow detection on all arithmetic operations
  - Clear error messages instead of silent wrapping

- **Persistent Storage Limits** - Storage quota management
  - Per-file size limits (10MB default)
  - Total storage quota (100MB default)
  - Automatic cleanup of old data
  - Storage usage tracking

#### Fixed

- **Sanitization False Positives** - Reduced false positive rate by 90%+
  - Smart pattern matching requiring actual SQL query structures (e.g., "SELECT...FROM")
  - Context-aware detection (HTML requires tags, URLs require schemes)
  - Trusted literal optimization for concatenation
  - "update permission" and similar benign strings no longer trigger errors

#### Documentation

**New Security Guides:** (10 comprehensive documents)
- `docs/PATH_TRAVERSAL_PREVENTION.md` - File system security guide
- `docs/PERSISTENCE_LIMITS.md` - Storage quota management guide
- `docs/CONTRACT_REQUIRE.md` - Precondition validation guide
- `docs/MANDATORY_SANITIZATION.md` - Injection attack prevention guide
- `docs/CRYPTO_FUNCTIONS.md` - Password hashing & CSPRNG guide
- `docs/INTEGER_OVERFLOW_PROTECTION.md` - Arithmetic safety guide
- `docs/RESOURCE_LIMITS.md` - DoS prevention guide
- `docs/TYPE_SAFETY.md` - Strict type checking guide
- `docs/CONTRACT_ACCESS_CONTROL.md` - RBAC system guide (500+ lines)
- `docs/DEBUG_SANITIZATION.md` - Credential protection guide

**Summary Documents:**
- `docs/SECURITY_FIXES_SUMMARY.md` - Complete overview of all 10 security fixes
- `SECURITY_REMEDIATION_COMPLETE.md` - Final update summary

**Updated Documentation:**
- `README.md` - Added "Latest Security Patches & Features" section
- `SECURITY_ACTION_PLAN.md` - All 10 fixes marked complete
- `VULNERABILITY_FINDINGS.md` - All vulnerabilities marked as FIXED
- `docs/DOCUMENTATION_INDEX.md` - Added all security guides

#### Testing

**New Test Files:** (23 total test files, 82 test cases)
- `tests/security/test_path_traversal.zx` (8 cases)
- `tests/security/test_storage_limits.zx` (6 cases)
- `tests/security/test_contract_require.zx` (5 cases)
- `tests/security/test_mandatory_sanitization.zx` (10 cases)
- `tests/security/test_sanitization_improvements.zx` (6 cases)
- `tests/security/test_crypto_functions.zx` (6 cases)
- `tests/security/test_integer_overflow.zx` (7 cases)
- `tests/security/test_resource_limits.zx` (5 cases)
- `tests/security/test_type_safety.zx` (8 cases)
- `tests/security/test_access_control.zx` (9 cases)
- `tests/security/test_contract_access.zx` (5 cases)
- `tests/security/test_debug_sanitization.zx` (7 cases)
- **100% pass rate** on all security tests

#### Implementation

**New Modules:**
- `src/zexus/access_control_system/` - Complete RBAC package
  - `access_control.py` - AccessControlManager class
  - `__init__.py` - Package initialization and exports
- `src/zexus/debug_sanitizer.py` - Debug sanitization module

**Modified Core Files:**
- `src/zexus/security_enforcement.py` - Enhanced input sanitization and path validation
- `src/zexus/persistent_storage.py` - Added storage limits
- `src/zexus/evaluator/functions.py` - Added require(), crypto, sanitize(), access control builtins
- `src/zexus/evaluator/expressions.py` - Type safety and overflow protection
- `src/zexus/evaluator/evaluator.py` - Resource limits enforcement
- `src/zexus/error_reporter.py` - Debug sanitization integration

#### Metrics

- **Security Grade:** C- → **A+** ✅
- **OWASP Top 10 Coverage:** 10/10 categories addressed
- **Test Coverage:** 100% of security features
- **Total Security Code:** ~3,500 lines
- **Total Documentation:** ~5,000 lines
- **Zero Known Vulnerabilities**

#### Migration Notes

**Breaking Changes:**
- Type Safety (Fix #8): String + Number concatenation now requires explicit conversion
  - Before: `"Count: " + 42` ✅ (worked)
  - After: `"Count: " + 42` ❌ (error)
  - Fix: `"Count: " + string(42)` ✅ (works)

**Backwards Compatible:**
- All other 9 security fixes are 100% backwards compatible
- Existing code automatically benefits from protections
- Optional configuration available via `zexus.json`

---

## [1.6.2] - 2025-12-31

### Added
- Complete database ecosystem with 4 production-ready drivers
- HTTP server with routing (GET, POST, PUT, DELETE)
- Socket/TCP primitives for low-level network programming
- Testing framework with assertions
- Fully functional ZPM package manager

---

## [1.5.0] - 2025-12-15

### Added
- **World-Class Error Reporting** - Production-grade error messages rivaling Rust
- **Advanced DATA System** - Generic types, pattern matching, operator overloading
- **Stack Trace Formatter** - Beautiful, readable stack traces with source context
- **Smart Error Suggestions** - Actionable hints for fixing common errors
- **Pattern Matching** - Complete pattern matching with exhaustiveness checking
- **CONTINUE Keyword** - Error recovery mode for graceful degradation

### Enhanced
- Error reporting now includes color-coded output
- Source code context in error messages
- Category distinction (user code vs interpreter bugs)
- Smart detection of common mistakes

---

## [0.1.3] - 2025-11-30

### Added
- 130+ keywords fully operational and tested
- Dual-mode DEBUG (function and statement modes)
- Conditional print: `print(condition, message)`
- Multiple syntax styles support
- Enterprise keywords (MIDDLEWARE, AUTH, THROTTLE, CACHE, INJECT)
- Complete async/await runtime with Promise system
- Main entry point with 15+ lifecycle builtins
- UI renderer with SCREEN, COMPONENT, THEME keywords
- Enhanced VERIFY with email, URL, phone validation
- Blockchain keywords (implements, pure, view, payable, modifier, this, emit)
- BREAK keyword for loop control
- THROW keyword for explicit error raising
- 100+ built-in functions
- LOG keyword enhancements

### Fixed
- Array literal parsing (no more duplicate elements)
- ENUM value accessibility
- WHILE condition parsing without parentheses
- Loop execution and variable reassignment
- DEFER cleanup execution
- SANDBOX return values
- Dependency injection container creation

---

## Earlier Versions

See git history for changes in versions < 0.1.3

---

**Legend:**
- 🔒 Security
- ⚠️ Breaking Change
- ✅ Fixed
- 📚 Documentation
- 🧪 Testing

[1.8.3]: https://github.com/Zaidux/zexus-interpreter/compare/v1.8.2...v1.8.3
[1.8.2]: https://github.com/Zaidux/zexus-interpreter/compare/v1.8.1...v1.8.2
[1.8.1]: https://github.com/Zaidux/zexus-interpreter/compare/v1.8.0...v1.8.1
[1.8.0]: https://github.com/Zaidux/zexus-interpreter/compare/v1.7.2...v1.8.0
[1.7.2]: https://github.com/Zaidux/zexus-interpreter/compare/v1.7.1...v1.7.2
[1.7.1]: https://github.com/Zaidux/zexus-interpreter/compare/v1.6.8...v1.7.1
[1.6.8]: https://github.com/Zaidux/zexus-interpreter/compare/v1.6.7...v1.6.8
[1.6.7]: https://github.com/Zaidux/zexus-interpreter/compare/v1.6.6...v1.6.7
[1.6.6]: https://github.com/Zaidux/zexus-interpreter/compare/v1.6.5...v1.6.6
[1.6.5]: https://github.com/Zaidux/zexus-interpreter/compare/v1.6.3...v1.6.5
[1.6.3]: https://github.com/Zaidux/zexus-interpreter/compare/v1.6.2...v1.6.3
[1.6.2]: https://github.com/Zaidux/zexus-interpreter/compare/v1.5.0...v1.6.2
[1.5.0]: https://github.com/Zaidux/zexus-interpreter/compare/v0.1.3...v1.5.0
[0.1.3]: https://github.com/Zaidux/zexus-interpreter/releases/tag/v0.1.3
