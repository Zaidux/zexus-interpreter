# Zexus v1.8.0 — VM & Interpreter Issues

> Discovered during Ziver Chain Phase 0 audit (Feb 23, 2026).
> ~~Target fix version: **v1.8.1**~~
> **All issues resolved in v1.8.1** (Feb 23, 2026)
> Tested with: `zexus 1.8.0` via `pip install zexus`, running on Ubuntu 24.04

---

## Bytecode VM Issues

### VM-001: EntityStatement compilation crash ✅ Fixed in v1.8.1
- **Severity:** High
- **Trigger:** Any file containing an `entity` declaration
- **Error:** `Bytecode compilation error: Failed to compile Program: Failed to compile EntityStatement: 'EntityStatement' object has no attribute 'body'`
- **Workaround:** Falls back to interpreter automatically
- **Fix:** Rewrote `_compile_EntityStatement` in `vm/compiler.py` to use `node.properties` and `node.methods` instead of non-existent `node.body`.
- **Repro:**
```zexus
entity Point {
    x: integer
    y: integer
}
let p = Point{x: 3, y: 4}
```

### VM-002: ExportStatement not supported in bytecode compiler ✅ Fixed in v1.8.1
- **Severity:** Medium
- **Trigger:** Any file with `export contract` or `export action`
- **Error:** `VM fallback: Node type 'ExportStatement' is not currently supported by the bytecode compiler.`
- **Workaround:** Falls back to interpreter
- **Fix:** Added `_compile_ExportStatement` to both `vm/compiler.py` and `evaluator/bytecode_compiler.py`.
- **Repro:**
```zexus
export contract MyContract {
    data val = 42
    action get_val() { return val }
}
```

### VM-003: Imported functions resolve to Action objects instead of being called ✅ Fixed in v1.8.1
- **Severity:** High
- **Trigger:** Calling a function imported from another `.zx` file in VM mode
- **Error:** `add result: action(a, b) { ... }` — prints the function object instead of calling it
- **Workaround:** Falls back to interpreter when VM error occurs
- **Fix:** Fixed `_call_builtin_async_obj` in `vm/vm.py` — changed broad `except Exception: pass` to properly handle ZAction/ZLambda objects instead of falling through to return the raw function object.
- **Repro:**
```zexus
# mylib.zx
action add(a, b) { return a + b }
export { add }

# main.zx
use "mylib.zx"
let result = add(10, 20)
print("result: " + string(result))  # VM prints: "action(a, b) { ... }"
```

### VM-004: Entity constructor argument count mismatch ✅ Fixed in v1.8.1
- **Severity:** Low
- **Trigger:** Entity with named-field construction `Entity{field: value}`
- **Error:** `[TypeError] 'Point' expects 0 argument(s) but got 1`
- **Note:** Only in bytecode validation pass; interpreter handles it correctly
- **Fix:** Added `_check_EntityStatement` to `type_checker.py` to register entity constructors, and modified `_check_call` to skip strict arity validation for entity constructors called with a single map argument.
- **Repro:**
```zexus
entity Point {
    x: integer
    y: integer
}
let p = Point{x: 3, y: 4}  # TypeError in validation
```

### VM-005: `string()` + builtin module returns type name instead of value ✅ Fixed in v1.8.1
- **Severity:** Medium
- **Trigger:** `string(dt.now())` or `string(math.max(3,7))` in VM mode
- **Error:** Prints `<built-in function: >` or `None` instead of the actual value
- **Workaround:** Interpreter handles these correctly
- **Fix:** Fixed CALL_METHOD handler in `vm/vm.py` (both sync and async paths) to detect `Builtin` objects stored in module dicts and invoke their inner `.fn` callable instead of treating them as non-callable values.
- **Repro:**
```zexus
use "datetime" as dt
let now = dt.now()
print("DateTime: " + string(now))  # VM: "<built-in function: >"
```

---

## Interpreter Issues

### INT-001: `protocol` keyword not recognized ✅ Fixed in v1.8.1
- **Severity:** High
- **Trigger:** Using `protocol` to define an interface
- **Error:** `Identifier 'protocol' not found`
- **Impact:** Cannot use protocol/implements pattern at all
- **Fix:** Added PROTOCOL to both strategy parser statement starters, context rules, and `_parse_block_statements` handler. Added `parse_protocol_statement` to the traditional parser dispatch. Added `_parse_protocol_statement_block` to context parser. Fixed `eval_protocol_statement` to handle both string and AST method entries.
- **Repro:**
```zexus
protocol Greetable {
    action greet() -> string
}
contract Greeter implements Greetable {
    data name = "World"
    action greet() { return "Hello, " + name }
}
```

### INT-002: `implements` keyword not recognized ✅ Fixed in v1.8.1
- **Severity:** High  
- **Trigger:** Using `implements` on a contract (consequence of INT-001)
- **Error:** Parse error when `protocol` is not recognized
- **Note:** Tied to INT-001; fixing `protocol` should fix this too
- **Fix:** Fixed `_parse_contract_statement_block` in `strategy_context.py` to detect the `implements` keyword between contract name and `{`, passing `implements=Identifier(name)` to the ContractStatement constructor.

### INT-003: `emit` not recognized outside or inside contracts ✅ Fixed in v1.8.1
- **Severity:** Critical
- **Trigger:** Using `emit EventName(args)` anywhere
- **Error:** `Identifier 'emit' not found`
- **Impact:** Cannot emit blockchain events — core blockchain feature is broken
- **Fix:** Added EMIT to `STATEMENT_STARTERS` in both `strategy_context.py` and `strategy_structural.py`, added to `context_rules` dict, added handler in `_parse_block_statements`, and implemented `_parse_emit_statement_block` method.
- **Repro (outside contract):**
```zexus
emit TestEvent("hello", 42)
```
- **Repro (inside contract):**
```zexus
contract Token {
    data balances = {}
    action transfer(to, amount) {
        emit Transfer("0xALICE", to, amount)  # ERROR: 'emit' not found
        return true
    }
}
let t = Token()
t.transfer("0xBOB", 100)
```

### INT-004: `persistent storage` declarations not functional ✅ Fixed in v1.8.1
- **Severity:** Medium
- **Trigger:** Using `persistent storage varname: type` in a contract
- **Behavior:** Parses without error but doesn't provide any persistence; behaves identically to `data`
- **Impact:** No actual on-chain persistence — purely cosmetic keyword
- **Fix:** Rewrote `eval_persistent_statement` in `statements.py` to initialise the `PersistentStorage` SQLite backend on the environment (deriving scope from contract name), use `set_persistent`/`get_persistent` for storage, and restore previously-persisted values on startup. Also fixed `_init_persistence` and `enable_memory_tracking` in `object.py` — removed broken `if 'zexus.persistence' in sys.modules` guard that prevented the import from ever succeeding.
- **Repro:**
```zexus
contract Store {
    persistent storage items: map
    action init() { items = {} }
    action set(k, v) { items[k] = v }
    action get(k) { return items[k] }
}
```

### INT-005: `track_memory()` not recognized ✅ Fixed in v1.8.1
- **Severity:** Low
- **Trigger:** Calling `track_memory()` at module top level
- **Error:** `Identifier 'track_memory' not found`
- **Impact:** Memory tracking feature not available
- **Fix:** Registered `track_memory` as a builtin function in `_register_missing_directive_builtins()`.

### INT-006: `cache()` not recognized ✅ Fixed in v1.8.1
- **Severity:** Low
- **Trigger:** Calling `cache("name", {ttl: 300})` at module top level
- **Error:** `Identifier 'cache' not found`
- **Impact:** Cache directive feature not available
- **Fix:** Registered `cache` as a builtin function with TTL option support.

### INT-007: `throttle()` not recognized ✅ Fixed in v1.8.1
- **Severity:** Low
- **Trigger:** Calling `throttle("name", {requests_per_minute: 100})`
- **Error:** `Identifier 'throttle' not found`
- **Impact:** Rate-limiting directive feature not available
- **Fix:** Registered `throttle` as a builtin function with `requests_per_minute` option support.

### INT-008: `audit()` not recognized ✅ Fixed in v1.8.1
- **Severity:** Low
- **Trigger:** Calling `audit("event", {data})` anywhere
- **Error:** `Identifier 'audit' not found`
- **Impact:** Audit trail feature not available
- **Fix:** Registered `audit` as a builtin function that logs events to an `__audit_trail__` list in the environment.

### INT-009: `verify()` not recognized as a standalone function ✅ Fixed in v1.8.1
- **Severity:** Low
- **Trigger:** Calling `verify(condition, "message")` (not `require`)
- **Error:** `Identifier 'verify' not found`
- **Note:** `require(condition, "message")` works correctly; `verify` is the missing alias
- **Fix:** Registered `verify` as a builtin function (alias for `require` semantics — throws on falsy condition).

### INT-010: `watch` blocks not recognized ✅ Fixed in v1.8.1
- **Severity:** Low
- **Trigger:** Reactive `watch variable { ... }` blocks
- **Error:** Parse error
- **Impact:** Reactive programming feature not available
- **Fix:** Added Form 3 (`watch expr { ... }` without `=>` arrow) to both the context parser's `_parse_watch_statement` and the traditional parser's `parse_watch_statement`. Previously only `watch { ... }` and `watch expr => { ... }` were supported.
- **Repro:**
```zexus
contract Tracker {
    data count = 0
    watch count {
        print("Count changed to: " + string(count))
    }
}
```

### INT-011: `protect` action modifier not recognized — Clarified in v1.8.1
- **Severity:** Low
- **Trigger:** Using `action protect method_name()` syntax
- **Error:** Parse error or ignored
- **Impact:** Cannot mark actions as protected/internal
- **Note:** The `protect` statement uses function-call syntax `protect(target, {rules})`, not an action modifier. Both parsers and the evaluator fully support `protect(target, {rate_limit: 100, auth_required: true})`. This is a documentation/expectation issue, not a code bug.

### INT-012: `bc.create_genesis_block()` returns function object without call ✅ Fixed in v1.8.1
- **Severity:** Medium
- **Trigger:** Calling `bc.create_genesis_block()` and inspecting result in VM mode
- **Behavior:** VM returns `<built-in function: >` instead of executing
- **Note:** Interpreter mode returns the correct block map
- **Workaround:** Use interpreter mode (automatic fallback)
- **Fix:** Same root cause as VM-005 — CALL_METHOD handler now properly invokes `Builtin.fn` for module method results stored in dicts.

### INT-013: `map.has()` method not available ✅ Fixed in v1.8.1
- **Severity:** Medium
- **Trigger:** Calling `.has("key")` on a map
- **Error:** Method not found or no-op
- **Workaround:** Use `map["key"] != null` instead
- **Fix:** Fixed key normalization in `evaluator/functions.py` — `map.has()` and `map.get()` now try both plain string key and String-wrapped key for lookup, handling the case where Map.pairs uses String object keys.
- **Repro:**
```zexus
let m = {"a": 1}
print(m.has("a"))  # Now returns true
```

### INT-014: `map.get(key, default)` method not available ✅ Fixed in v1.8.1
- **Severity:** Medium
- **Trigger:** Calling `.get("key", default_value)` on a map
- **Fix:** Same key normalization fix as INT-013. `map.get("key", default)` now works correctly.

### INT-015: `list.is_empty()` method not available ✅ Fixed in v1.8.1
- **Severity:** Low
- **Trigger:** Calling `.is_empty()` on a list
- **Workaround:** Use `len(list) == 0`
- **Fix:** Added `is_empty` method to the List method dispatch in `evaluator/functions.py`. Returns `true` if list has zero elements, `false` otherwise.

### INT-016: `list.count()` vs `len()` inconsistency ✅ Already working
- **Severity:** Low
- **Trigger:** `.count()` may not work on all list types
- **Workaround:** Always use `len(list)` which works reliably
- **Note:** Investigation confirmed `list.count()` already works correctly (returns `Integer(len(elements))`). No fix needed.

---

## Resolution Summary (v1.8.1)

| Issue | Status | Priority | Fix Location |
|---|---|---|---|
| VM-001 | ✅ Fixed | P1 | `vm/compiler.py` |
| VM-002 | ✅ Fixed | P2 | `vm/compiler.py`, `evaluator/bytecode_compiler.py` |
| VM-003 | ✅ Fixed | P1 | `vm/vm.py` |
| VM-004 | ✅ Fixed | P2 | `type_checker.py` |
| VM-005 | ✅ Fixed | P2 | `vm/vm.py` |
| INT-001 | ✅ Fixed | P0 | `parser/parser.py`, `parser/strategy_context.py`, `parser/strategy_structural.py` |
| INT-002 | ✅ Fixed | P0 | `parser/strategy_context.py` |
| INT-003 | ✅ Fixed | P0 | `parser/strategy_context.py`, `parser/strategy_structural.py` |
| INT-004 | ✅ Fixed | P2 | `evaluator/statements.py`, `object.py` |
| INT-005 | ✅ Fixed | P3 | `evaluator/functions.py` |
| INT-006 | ✅ Fixed | P3 | `evaluator/functions.py` |
| INT-007 | ✅ Fixed | P3 | `evaluator/functions.py` |
| INT-008 | ✅ Fixed | P3 | `evaluator/functions.py` |
| INT-009 | ✅ Fixed | P3 | `evaluator/functions.py` |
| INT-010 | ✅ Fixed | P3 | `parser/parser.py`, `parser/strategy_context.py` |
| INT-011 | 📝 Clarified | P3 | Documentation — `protect()` uses call syntax, not modifier |
| INT-012 | ✅ Fixed | P2 | `vm/vm.py` (same fix as VM-005) |
| INT-013 | ✅ Fixed | P1 | `evaluator/functions.py` |
| INT-014 | ✅ Fixed | P1 | `evaluator/functions.py` |
| INT-015 | ✅ Fixed | P3 | `evaluator/functions.py` |
| INT-016 | ✅ Already working | P3 | No change needed |

**Total: 20/21 issues fixed, 1 clarified (INT-011 was a documentation gap, not a bug)**

---

## What Works Correctly (v1.8.1)

All features from v1.8.0 continue to work, plus the following now work correctly:

| Feature | Status |
|---|---|
| `entity` declarations + field construction | ✅ Works (interpreter + VM) |
| `export contract` / `export action` | ✅ Works (interpreter + VM) |
| `protocol` / `implements` | ✅ Works |
| `emit EventName(args)` | ✅ Works |
| `persistent storage` in contracts | ✅ Works (SQLite-backed) |
| `map.has(key)` | ✅ Works |
| `map.get(key, default)` | ✅ Works |
| `list.is_empty()` | ✅ Works |
| `list.count()` | ✅ Works |
| `track_memory()` | ✅ Works |
| `cache(name, options)` | ✅ Works |
| `throttle(name, options)` | ✅ Works |
| `audit(event, data)` | ✅ Works |
| `verify(condition, message)` | ✅ Works |
| `watch variable { ... }` | ✅ Works |
| `protect(target, rules)` | ✅ Works |
| `string(dt.now())` in VM | ✅ Works |
| `bc.create_genesis_block()` in VM | ✅ Works |
| Imported functions in VM | ✅ Works |
| `contract` with `data` and `action` | ✅ Works |
| `use "file.zx"` file imports | ✅ Works |
| `use "subdir/file.zx"` subdirectory imports | ✅ Works |
| `use "crypto" as crypto` built-in module | ✅ Works |
| `use "blockchain" as bc` built-in module | ✅ Works |
| `use "datetime" as dt` built-in module | ✅ Works |
| `use "math" as math` built-in module | ✅ Works |
| `use "json" as json` built-in module | ✅ Works |
| `hash(data, algorithm)` global | ✅ Works |
| `keccak256(data)` global | ✅ Works (with pycryptodome) |
| `generateKeypair("ECDSA")` global | ✅ Works (with cryptography) |
| `signature(data, key, algo)` global | ✅ Works |
| `verify_sig(data, sig, pubkey, algo)` global | ✅ Works |
| `deriveAddress(pubkey)` global | ✅ Works |
| `require(condition, message)` | ✅ Works |
| `try { } catch (e) { }` | ✅ Works |
| `for each item in list { }` | ✅ Works |
| `for each key, val in map { }` | ✅ Works |
| `this.*` inside contracts | ✅ Works |
| `bc.create_genesis_block()` | ✅ Works |
| `bc.create_transaction(from, to, amt)` | ✅ Works |
| `bc.calculate_merkle_root(hashes)` | ✅ Works |
| `bc.create_block(idx, ts, data, prev, nonce)` | ✅ Works |
| `bc.hash_block(block)` | ✅ Works |
| `bc.validate_chain(blocks)` | ✅ Works |
| `crypto.keccak256(data)` | ✅ Works |
| `crypto.sha256(data)` | ✅ Works |
| `crypto.generate_keypair(algo)` | ✅ Works |
| `crypto.secp256k1_sign(data, key)` | ✅ Works |
| `crypto.verify_signature(data, sig, key)` | ✅ Works |
| `crypto.calculate_merkle_root(hashes)` | ✅ Works |
| `json.stringify(obj)` | ✅ Works |
| `dt.now()` | ✅ Works |
| `math.max(a, b)` | ✅ Works |
| `len(list)` | ✅ Works |
| `string(value)` | ✅ Works |
| String slicing `s[0:5]` | ✅ Works |
| List concatenation `list + [item]` | ✅ Works |
| Map bracket access `m["key"]` | ✅ Works |
| Map bracket assignment `m["key"] = val` | ✅ Works |

---

## Required pip Dependencies

```bash
pip install zexus cryptography pycryptodome
```

Without `cryptography`: `generateKeypair()` and `signature()` fail.
Without `pycryptodome`: `keccak256()` fails (SHA3-256 is NOT a substitute).