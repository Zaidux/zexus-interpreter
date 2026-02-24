# Zexus v1.8.1 — Vulnerability Fixes Roadmap

**Created:** 2026-02-23  
**Audit scope:** Interpreter (evaluator layer, 7 files) + VM runtime & compilers (5 files)  
**Total findings:** 69  

---

## Phase 0 — CRITICAL (13 findings)

| ID | Area | File | Finding | Status |
|----|------|------|---------|--------|
| C1 | Interp | `evaluator/functions.py` ~L2654 | `exec()` in `_eval_file` — arbitrary Python code execution when evaluating `.py` files | ✅ Done |
| C2 | Interp | `evaluator/functions.py` ~L2680 | `subprocess.run(['node', ...])` for `.js` files — arbitrary Node.js execution | ✅ Done |
| C3 | Interp | `evaluator/statements.py` ~L3327 | `ctypes.CDLL(node.library_name)` in `eval_native_statement` — arbitrary native code loading | ✅ Done |
| C4 | Interp | `object.py` ~L599 | `File` class has zero path traversal protection — read/write any file on system | ✅ Done |
| C5 | Interp | `persistence.py` ~L155 | `scope_id` path traversal in `PersistentStorage` — attacker-controlled DB path | ✅ Done |
| C6 | Interp | `security.py` | `contract_id` path traversal in `ContractStorage` — same issue | ✅ Done |
| C7 | Interp | `persistence.py` | `delete_persistent_scope` — path traversal enables arbitrary file deletion | ✅ Done |
| C8 | Interp | `evaluator/functions.py` ~L4303 | `daemonize()` builtin forks the interpreter — background daemon creation | ✅ Done |
| C9 | Interp | `evaluator/functions.py` | `env_set` allows arbitrary env var manipulation (PATH, LD_PRELOAD injection) | ✅ Done |
| C10 | VM | `vm/binary_bytecode.py` ~L394 | `pickle.loads()` on untrusted `.zxc` bytecode — RCE via crafted cache files | ✅ Done |
| C11 | VM | `vm/vm.py` ~L3388 | `READ` opcode opens any file path with no sandboxing | ✅ Done |
| C12 | VM | `vm/vm.py` ~L4250 | `WRITE` opcode writes to any path with no restriction | ✅ Done |
| C13 | VM | `vm/vm.py` ~L3519 | Fast-loop path bypasses gas metering entirely — infinite loops run forever | ✅ Done |

---

## Phase 1 — HIGH (14 findings)

| ID | Area | File | Finding | Status |
|----|------|------|---------|--------|
| H1 | Interp | `security.py` ~L528 | Broken CIDR validation — uses string prefix match instead of proper network math | ✅ Done |
| H2 | Interp | `security.py` | User-controlled regex in `emit_event` — ReDoS vector | ✅ Done |
| H3 | Interp | `evaluator/expressions.py` ~L373 | Unbounded string repetition "x" * n — memory exhaustion | ✅ Done |
| H4 | Interp | `evaluator/functions.py` ~L4109 | `exit_program()` calls `sys.exit()` directly | ✅ Done |
| H5 | Interp | `evaluator/statements.py` ~L2233 | `__import__('time')` inline pattern instead of module-level import | ✅ Done |
| H6 | VM | `vm/vm.py` ~L1151 | `importlib.import_module()` with user-controlled module path — full host access | ✅ Done |
| H7 | VM | `vm/vm.py` ~L1000 | Path traversal in module import resolution (`use "../../etc/secrets.zx"`) | ✅ Done |
| H8 | VM | `vm/vm.py` ~L2756 | Unbounded stack growth — no maximum stack depth | ✅ Done |
| H9 | VM | `vm/vm.py` ~L2580 | Unbounded recursion via `_invoke_callable_sync` — no call-depth limit | ✅ Done |
| H10 | VM | `vm/vm.py` ~L3394 | Bare `except:` in `_op_read` catches `SystemExit`/`KeyboardInterrupt` | ✅ Done |
| H11 | VM | `evaluator/bytecode_compiler.py` ~L1195 | Bare `except:` in `_try_constant_operation` catches everything | ✅ Done |
| H12 | VM | `vm/vm.py` (90+ instances) | ~90 silent `except Exception: pass` blocks mask real bugs | ⏸ Deferred |
| H13 | VM | `vm/vm.py` ~L1543 | `_ensure_recursion_headroom` actively increases `sys.setrecursionlimit` to 5000 | ✅ Done |
| H14 | VM | `vm/cache.py` ~L509 | `pickle.dump()`/`pickle.load()` in bytecode cache — RCE via cache poisoning | ✅ Done |

---

## Phase 2 — LOGIC BUGS (8 findings)

| ID | Area | File | Finding | Status |
|----|------|------|---------|--------|
| L1 | VM | `vm/vm.py` | `JUMP_IF_TRUE` opcode not handled in any execution path — falls through silently | ✅ Done |
| L2 | VM | `vm/vm.py` | `AND` / `OR` opcodes not handled in sync or async loops — only register variants work | ✅ Done |
| L3 | VM | `evaluator/bytecode_compiler.py` ~L598 | `RequireStatement` emits broken bytecode — `JUMP_IF_TRUE` with `None` operand, never patched | ✅ Done |
| L4 | VM | `evaluator/bytecode_compiler.py` ~L571 | `TxStatement` has no exception-safety — crashes leave transactions uncommitted/unreverted | ✅ Done |
| L5 | VM | `vm/vm.py` ~L2020 | `MOD` opcode doesn't unwrap `.value` in sync path (unlike ADD/SUB/MUL/DIV) | ✅ Done |
| L6 | VM | `vm/vm.py` | `POW` opcode same issue — no `.value` unwrapping in sync path | ✅ Done |
| L7 | VM | `vm/vm.py` ~L4781 | `_call_builtin_async_obj` returns the exception object itself as a value | ✅ Done |
| L8 | VM | `vm/vm.py` ~L1603 | `collect_garbage(force=True)` deletes all user variables that don't start with `_` | ✅ Done |

---

## Phase 3 — COMPILER INCONSISTENCIES (7 findings)

| ID | Area | Finding | Status |
|----|------|---------|--------|
| I1 | Compilers | VM compiler has `ForStatement`; evaluator compiler does not | ✅ Done |
| I2 | Compilers | Evaluator compiler supports ~18 statement types the VM compiler does not (`NativeStatement`, `DeferStatement`, `PatternStatement`, `LambdaExpression`, `SpawnExpression`, `EmitStatement`, `ProtocolStatement`, etc.) | ✅ Done |
| I3 | Compilers | `PropertyAccessExpression` emits different opcodes (`INDEX` vs `GET_ATTR`) between compilers | ✅ Done |
| I4 | Compilers | `PrintStatement` multi-value support only in VM compiler | ✅ Done |
| I5 | Compilers | `LetStatement` destructuring only in VM compiler | ✅ Done |
| I6 | Compilers | `ContinueStatement` semantics differ (jump-to-loop-start vs custom `CONTINUE` opcode) | ✅ Done |
| I7 | Compilers | `EXPORT` opcode not handled in sync execution path | ✅ Done |

---

## Phase 4 — MEDIUM (15 findings)

| ID | Area | File | Finding | Status |
|----|------|------|---------|--------|
| M1 | Interp | `security.py` ~L659 | Bare `except:` in `SmartContract.__del__` | ✅ Done |
| M2 | Interp | `security.py` | ~10 silent `except Exception: pass` in `emit_event` | ✅ Done |
| M3 | Interp | `security.py` / `persistence.py` | `STORAGE_DIR`/`PERSISTENCE_DIR` created at import time | ✅ Done |
| M4 | Interp | `evaluator/core.py` ~L500 | Duplicate isinstance fallback chain duplicating dispatch table | ✅ Done |
| M5 | Interp | `object.py` / `security.py` | Duplicate `EntityDefinition`/`EntityInstance` definitions | ✅ Done |
.| M6 | Interp | `evaluator/statements.py` ~L4195 | Duplicate `eval_channel/send/receive/atomic_statement` defs — first set is dead code | ✅ Done |
| M7 | Interp | `evaluator/functions.py` ~L2696 | Triple definition of `_require` builtin — first two are dead code | ✅ Done |
| M8 | Interp | `evaluator/core.py` ~L99 | `_ensure_recursion_headroom` silently raises Python recursion limit | ✅ Done |
| M9 | Interp | `object.py` ~L732 | `lock_file` has no timeout — potential hang forever | ✅ Done |
| M10 | Interp | `evaluator/statements.py` | Sandbox `except Exception: pass` swallows VFS init errors | ✅ Done |
| M11 | Interp | `security.py` | Thread-local evaluator creation with no cleanup | ✅ Done |
| M12 | VM | `vm/vm.py` ~L2406 vs ~L4676 | `_LEDGER` has 10K cap sync but unbounded async — memory exhaustion | ✅ Done |
| M13 | VM | `vm/vm.py` ~L4586 | `_audit_log` list grows without bound — memory exhaustion | ✅ Done |
| M14 | VM | `vm/vm.py` ~L672 | VM pool doesn't clear `env`/`_closure_cells`/`_name_cache` — stale data leak | ✅ Done |
| M15 | VM | `vm/vm.py` ~L1910 | Stack underflow silently returns `None` vs async path raises `IndexError` — inconsistent | ✅ Done |

---

## Phase 5 — LOW / INFO (12 findings)

| ID | Area | File | Finding | Status |
|----|------|------|---------|--------|
| LI1 | Interp | `persistence.py` | `PersistentStorage.clear()` updates stats before deletion — stale immediately | ✅ Done |
| LI2 | Interp | `persistence.py` | `PersistentStorage.get()` calls `_update_usage_stats()` on every read — doubles cost | ✅ Done |
| LI3 | Interp | `persistence.py` | `EntityInstance` deserialization loses methods and computed properties | ✅ Done |
| LI4 | Interp | `evaluator/expressions.py` ~L150 | `eval_identifier` imports `traceback` on every miss (should be module-level) | ✅ Done |
| LI5 | Interp | `evaluator/functions.py` ~L2810 | Builtins dict has duplicate keys: `"require"`, `"random"`, `"input"`, `"sleep"` appear twice | ✅ Done |
| LI6 | Interp | `evaluator/expressions.py` ~L880 | `eval_await_expression` busy-waits with `time.sleep(0.001)` spin loop | ✅ Done |
| LI7 | Interp | `evaluator/core.py` | `_env_to_dict` walks outer environments without depth limit | ✅ Done |
| LI8 | Interp | `evaluator/functions.py` | `memory_stats()` fallback calls `gc.get_objects()` — extremely slow, meaningless result | ✅ Done |
| LI9 | Interp | `evaluator/functions.py` ~L5000 | `sanitize_input` regex-based SQL keyword stripping is bypassable and corrupts data | ✅ Done |
| LI10 | Interp | `security.py` | `AuthConfig.validate_token` unconditionally returns `True` | ✅ Done |
| LI11 | VM | `vm/vm.py` | `POW` allows exponent DoS (`10 ** 10000000`) — no guard on magnitude | ✅ Done |
| LI12 | VM | `evaluator/bytecode_compiler.py` | `_compile_PropertyAccessExpression` defined twice — first is dead code | ✅ Done |

---

## Severity Summary

| Phase | Severity | Count |
|-------|----------|-------|
| 0 | CRITICAL | 13 |
| 1 | HIGH | 14 |
| 2 | LOGIC BUG | 8 |
| 3 | COMPILER INCONSISTENCY | 7 |
| 4 | MEDIUM | 15 |
| 5 | LOW / INFO | 12 |
| | **Total** | **69** |

---

## Fix Strategy

- **Phase 0 (Critical):** Path sanitization helpers, sandbox enforcement, pickle removal, gas metering fix
- **Phase 1 (High):** CIDR validation, ReDoS protection, resource limits, exception narrowing
- **Phase 2 (Logic Bugs):** Missing opcode handlers, broken bytecode emission, value unwrapping
- **Phase 3 (Compiler Inconsistencies):** Missing statement compilers, unified opcode semantics
- **Phase 4 (Medium):** Dead code removal, error handling improvements, resource caps
- **Phase 5 (Low/Info):** Performance optimizations, stub completions, minor cleanups
