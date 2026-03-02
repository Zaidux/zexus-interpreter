# Zexus v1.8.3 — Issues Found During Phase 0 Rewrite

> This file tracks issues discovered while rewriting Ziver-Chain source files for Zexus v1.8.3.
> Updated as issues are found and resolved.

## Known Quirks (from initial testing)

| ID | Category | Description | Severity | Status |
|----|----------|-------------|----------|--------|
| Q-001 | map.has() | Returns `None` instead of `true/false` | Minor | ✅ **Fixed in v1.8.3** |
| Q-002 | list.is_empty() | Returns `None` instead of `true/false` | Minor | ✅ **Fixed in v1.8.3** |
| Q-003 | bc.createChain() | Returns `false` | Needs investigation |**✅ fixed**|

### Q-001 Fix Details
**File:** `src/zexus/object.py` (Map.has method, ~line 244)  
**Problem:** `Map.has()` only checked `key in self.pairs` directly, failing when keys were stored in normalized form (e.g., via `_normalize_key`).  
**Fix:** Updated `has()` to try direct lookup, then String-wrapped key, then normalized key — matching the same strategy used by `Map.get()`.  
**Example (now works):**
```zx
let m = {"hello": 1}
print(m.has("hello"))  // true
```

### Q-002 Fix Details
**File:** `src/zexus/object.py` (List class, added `is_empty()` method)  
**Problem:** No `is_empty()` method existed on the `List` class. The evaluator's method dispatch (functions.py) handled it, but calling it on the object directly returned `None`.  
**Fix:** Added `is_empty()` method to the `List` class that returns `Boolean(len(self.elements) == 0)`.  
**Example (now works):**
```zx
let items = []
print(items.is_empty())  // true
items.push(1)
print(items.is_empty())  // false
```

## Issues Found During Phase 0 Rewrite

| ID | Category | Description | Severity | Status |
|----|----------|-------------|----------|--------|
| R-001 | VM / Entity | Entity field access returns `None` on VM (works in interpreter) | **Critical** | ✅ **Fixed in v1.8.3** — Added `_build_entity_definition()`, `_construct_entity()`, EntityInstance GET_ATTR support |
| R-002 | VM / Contract | Contract state not initialized on VM; `this.val` is `null` even with `state { val: 0 }` | **Critical** | ✅ **Fixed in v1.8.3** — Added `_compile_StateStatement` to compiler for standalone state declarations |
| R-003 | Interpreter / self | `self` keyword not recognized in interpreter — must use `this` | **High** | ✅ **Fixed in v1.8.3** |
| R-004 | Interpreter / init | `init()` is NOT auto-called when constructing contract with `Contract{}` | **High** | ✅ **Fixed in v1.8.3** |
| R-005 | Interpreter / state | `state { val: 0 }` does NOT initialize fields to default values | **High** | ✅ **Fixed in v1.8.3** |
| R-006 | Interpreter / for-each | `for each i, item in list` (indexed) fails: "Identifier 'i' not found" | **High** | ✅ **Fixed in v1.8.3** |
| R-007 | Interpreter / for-each | `for each key, val in map` fails: "ForEach expects List" | **High** | ✅ **Fixed in v1.8.3** |
| R-008 | Interpreter / protect | `action protect` on module-level actions fails: "Identifier not found" | Medium | ✅ **Fixed in v1.8.3** — Policy enforcement added to `apply_function` |
| R-009 | Interpreter / nested assign | `m["a"]["b"] = val` fails: "Invalid assignment target" | Medium | ✅ **Verified working in v1.8.3** |
| R-010 | VM / output | Some complex programs compile to bytecode but produce no output | Medium | ✅ **Fixed in v1.8.3** — Added `_vm_warn()` diagnostic system, replaced silent `except: return None` with logged warnings |
| R-011 | Import / entity | Exported entities from other files can't be used as constructors in importing file | Medium | ✅ **Fixed in v1.8.3** — `export` now uses `is None` check instead of `not val` |
| R-012 | Interpreter / order | Entity declarations before a contract make the contract identifier invisible ("Identifier not found") | **High** | ✅ **Fixed in v1.8.3** — Closure env fix (`outer=self`) |
| R-013 | Interpreter / state | Multiple fields in `state { }` causes "Invalid assignment target" — only ONE field allowed | **Critical** | ✅ **Fixed in v1.8.3** |
| R-014 | Interpreter / order | `let` variable declarations before a contract also break contract visibility | **High** | ✅ **Fixed in v1.8.3** — Same closure env fix as R-012 |
| R-015 | Interpreter / push+map | After `map[key] = val` in a contract method, subsequent `list.push()` is silently ignored | **High** | ⚠️ **Partially fixed in v1.8.3** — Module-level state sync fixed (R-018), but local-variable push after map bracket-assign still drops in both contract methods AND module-level actions |
| R-016 | Interpreter / types | `INTEGER * FLOAT` causes "Type mismatch" — no implicit coercion | **High** | ✅ **Fixed in v1.8.3** |
| R-017 | Interpreter / modulo | `%` operator doesn't work with floats ("Unknown float operator: %") | Medium | ✅ **Fixed in v1.8.3** |
| R-018 | Interpreter / contract-call | Module-level helper functions that modify module-level state (`push`, `map[k]=v`) have their side-effects silently ignored when called FROM a contract method | **High** | ✅ **Fixed in v1.8.3** — `clone_for_closure` now references live env |
| R-019 | Interpreter / push-in-func | `push()` / `map[k]=v` inside a module-level `action` called from init or action-chain sometimes silently fails | **High** | ✅ **Fixed in v1.8.3** — Same closure env fix as R-018 |
| R-020 | Interpreter / chained-assign | `self.field[key] = val` causes "Invalid assignment target" — chained property + bracket assign on contract state | **High** | Use intermediate var: `let f = self.field; f[key] = val` |
| R-021 | Interpreter / operators | `+=` operator silent no-op (value unchanged); `not` keyword "Identifier not found" | **High** | Use `x = x + 1` for +=; use `!flag` for negation |
| R-022 | Interpreter / state-init | `state { items: [], data: {} }` — list/map initial values become `null`; only primitives work | **High** | Store lists/maps as module-level `let` vars; state fields primitives only |
| R-023 | Interpreter / methods | `.slice()`, `.contains()` (string), `.remove()`/`.delete()` (map), `.filter()`/`.sort()` (callback) — all "Method not supported" | Medium | Manual while/for-each loops |
| R-024 | Interpreter / range-loop | `for i in 0..N {}` — "Identifier 'i' not found" inside body | Medium | Use `while` loop with counter |
| R-025 | Interpreter / builtins | `cache_set()`/`cache_get()` standalone not found; `math.floor()` not found | Low | Use maps for cache; `int()` for floor |

### R-003 Fix Details
**File:** `src/zexus/evaluator/expressions.py` (eval_identifier, ~line 92)  
**Problem:** Only the `this` keyword was recognized for contract self-reference. `self` was treated as a regular identifier.  
**Fix:** Added `"self"` as an alias for `"this"` in `eval_identifier()` — both now resolve to the contract instance via `__contract_instance__`.  
**Example (now works):**
```zx
contract Counter {
    state count = 0
    action increment() {
        self.count = self.count + 1  // 'self' now works like 'this'
    }
}
```

### R-004 Fix Details
**File:** `src/zexus/security.py` (SmartContract.instantiate, ~line 1466)  
**Problem:** `SmartContract.instantiate()` created new instances and deployed them, but never called `init()` automatically.  
**Fix:** After deployment, `instantiate()` now checks if an `init` action exists and auto-calls it with any provided arguments.  
**Example (now works):**
```zx
contract Wallet {
    state balance = 0
    action init() { this.balance = 100 }
    action getBalance() { return this.balance }
}
let w = Wallet()        // init() is now auto-called
print(w.getBalance())   // 100
```

### R-005 Fix Details
**Files:** `src/zexus/parser/parser.py`, `src/zexus/parser/strategy_context.py`  
**Problem:** The `state { field: value }` block syntax was not parsed correctly by the strategy context parser, causing fields to remain uninitialized.  
**Fix:** Both parsers now correctly parse multi-field `state { }` blocks (see R-013 fix), and the evaluator initializes all declared state fields to their default values during contract deployment.  
**Example (now works):**
```zx
contract Token {
    state {
        balance: 100,
        name: "ZToken"
    }
    action getBalance() { return this.balance }  // returns 100
}
```

### R-006 Fix Details
**Files:** `src/zexus/zexus_ast.py`, `src/zexus/parser/parser.py`, `src/zexus/parser/strategy_context.py`, `src/zexus/evaluator/statements.py`  
**Problem:** `ForEachStatement` AST node only supported a single iterator variable. The parser only parsed `for each item in list`, not `for each i, item in list`.  
**Fix:**  
1. Added optional `index` field to `ForEachStatement` AST node.  
2. Updated both parsers (standard and strategy context) to detect a comma after the first identifier and parse a second variable.  
3. Updated `eval_foreach_statement` to bind the index variable (as `Integer`) during iteration.  
**Example (now works):**
```zx
let items = [10, 20, 30]
for each i, item in items {
    print("index: " + string(i) + " value: " + string(item))
}
// Output: index: 0 value: 10, index: 1 value: 20, index: 2 value: 30
```

### R-007 Fix Details
**File:** `src/zexus/evaluator/statements.py` (eval_foreach_statement, ~line 1226)  
**Problem:** `eval_foreach_statement` only checked for `List` type and returned `EvaluationError("ForEach expects List")` for Maps.  
**Fix:** Added a `Map` branch that iterates over `iterable.pairs.items()`. With two variables (`key, val`), both are bound. With one variable, the key is bound.  
**Example (now works):**
```zx
let m = {"a": 1, "b": 2, "c": 3}
for each key, val in m {
    print(key + ": " + string(val))
}
// Output: a: 1, b: 2, c: 3
```

### R-009 Verification
**Status:** Already working — no code change needed.  
**Verification:** The parser correctly recognizes nested `PropertyAccessExpression` chains as valid assignment targets. The evaluator resolves the intermediate object (e.g., `m["a"]`) and sets the property on it.  
**Example (works):**
```zx
let m = {"a": {"b": 0}}
m["a"]["b"] = 42
print(m["a"]["b"])  // 42
```

### R-013 Fix Details
**Files:** `src/zexus/parser/parser.py` (~line 4084), `src/zexus/parser/strategy_context.py` (~line 1719)  
**Problem:** The `state { field1: val1, field2: val2 }` block syntax was not supported. The parser expected `state IDENT = value` format only. With `state {`, it saw `{` instead of an identifier and failed.  
**Fix:** Both parsers now detect `LBRACE` after `STATE` and parse the block as a series of `name: value` pairs separated by commas, creating multiple `StateStatement`/`AstNodeShim` entries.  
**Example (now works):**
```zx
contract Token {
    state {
        balance: 100,
        owner: "alice",
        paused: false
    }
    action getBalance() { return this.balance }  // 100
    action getOwner() { return this.owner }      // "alice"
}
```

### R-016 Fix Details
**File:** `src/zexus/evaluator/expressions.py` (eval_infix_expression, ~line 391)  
**Problem:** In `eval_infix_expression`, the `elif operator == "*":` branch only handled string repetition (`String * Integer`). Mixed numeric multiplication (`Integer * Float` or `Float * Integer`) entered this branch but matched no inner condition, returning `None`.  
**Fix:** Added a fallthrough case: `elif isinstance(left, (Integer, Float)) and isinstance(right, (Integer, Float))` that performs `float(left.value) * float(right.value)` and returns a `Float`.  
**Example (now works):**
```zx
let result = 5 * 2.5
print(result)  // 12.5
```

### R-017 Fix Details
**File:** `src/zexus/evaluator/expressions.py` (eval_float_infix, ~line 278)  
**Problem:** `eval_float_infix` handled `+`, `-`, `*`, `/`, comparison operators, and `**`, but not `%` (modulo). Any `Float % Float` returned `EvaluationError("Unknown float operator: %")`.  
**Fix:** Added `elif operator == "%":` branch with zero-division check, returning `Float(left_val % right_val)`.  
**Example (now works):**
```zx
let remainder = 10.5 % 3.0
print(remainder)  // 1.5
```

## Working Feature Reference (v1.8.3 interpreter mode)

| Feature | Status | Notes |
|---------|--------|-------|
| `use "module" as alias` | ✅ Works | Built-in modules: crypto, datetime, math, json, blockchain |
| `use "file.zx" as alias` | ✅ Works | Relative to current file |
| `entity Name { field: default }` | ✅ Works | Field access works in interpreter |
| `Entity{ field: value }` construction | ✅ Works | In same file only |
| `contract Name { state {...} }` | ✅ Works | Must use `this` or `self`; `state { }` block now supports multiple fields (v1.8.3) |
| `protocol Name { ... }` | ✅ Works | |
| `contract X implements Y` | ✅ Works | |
| `self` keyword in contracts | ✅ Works | **Fixed v1.8.3** — `self` is now an alias for `this` |
| `persistent storage field: val` | ✅ Works | Must init manually |
| `emit EventName(args)` | ✅ Works | Inside contracts |
| `verify(cond, msg)` | ✅ Works | Global function |
| `audit(event, data)` | ✅ Works | Global function |
| `track_memory()` | ✅ Works | |
| `cache("name", opts)` | ✅ Works | |
| `throttle("name", opts)` | ✅ Works | |
| `watch variable { ... }` | ✅ Parses | May not fire reactively |
| `action protect` inside contracts | ✅ Works | |
| `for each item in list` | ✅ Works | Single variable only |
| `for each i, item in list` | ✅ Works | **Fixed v1.8.3** — indexed for-each |
| `for each key, val in map` | ✅ Works | **Fixed v1.8.3** — map iteration |
| `while cond { ... }` | ✅ Works | Use for indexed/map iteration |
| `try/catch/throw` | ✅ Works | |
| `match val { ... }` | ✅ Works | |
| `null ?? default` | ✅ Works | |
| `let/const` | ✅ Works | |
| `list.push(val)` | ✅ Works | |
| `map["key"]` bracket access | ✅ Works | Use bracket-built maps preferred |
| `map.dot` access | ⚠️ Varies | Works on entities, unreliable on plain maps |
| `len()`, `string()`, `print()` | ✅ Works | |
| `crypto.*` | ✅ Works | generateKeypair, keccak256, sign |
| `datetime.now().timestamp()` | ✅ Works | |
| `math.max/min` | ✅ Works | |
| `export { ... }` | ✅ Works | Functions and variables export; entities partially |
| `action name() -> type` | ✅ Works | Return type annotations |

## Rewrite Rules (Phase 0)

1. ~~Use `this` instead of `self` everywhere in contracts~~ → **Fixed v1.8.3**: Both `this` and `self` now work
2. Always include `init()` with manual state initialization
3. ~~After `Contract{}`, always call `.init()`~~ → **Fixed v1.8.3**: `init()` is now auto-called on construction
4. ~~Use `while` loops for indexed iteration and map iteration~~ → **Fixed v1.8.3**: `for each i, item in list` and `for each key, val in map` now work
5. Track map keys in a parallel list for iteration → **No longer needed for maps** (v1.8.3 supports map iteration)
6. Use `action protect` only inside contracts, not at module level
7. Remove phantom imports: `../database/postgres`, `../ai/zaie_engine`, `../middleware/security_middleware`
8. Remove `inject`, `middleware()`, `register_dependency()` (untested/unavailable)
9. Replace DB calls with in-memory maps
10. ~~Use factory functions for cross-file entity usage~~ → **Fixed v1.8.3**: Export check uses `is None` instead of `not val`
11. Prefer `--no-vm` execution until VM stabilizes
12. ~~Declare all entities AFTER contracts (entity before contract breaks contract visibility — R-012)~~ → **Fixed v1.8.3**: Closure env now references live module env
13. ~~Only ONE field in `state { }` — use module-level `let` vars for additional state (R-013)~~ → **Fixed v1.8.3**: Multiple fields in `state { }` now work
14. ~~Declare module-level `let` vars AFTER contract — contract can forward-reference them (R-014)~~ → **Fixed v1.8.3**: Same closure env fix as R-012
15. ~~In contract methods: always `list.push()` BEFORE `map[key] = val` (R-015)~~ → **Partially fixed v1.8.3**: Module-level state sync works, but **local-variable push after local map bracket-assign still drops** — keep push-before-map order for local vars
16. ~~Convert integers to float before multiplying with floats: `(int_val + 0.0) * float_val` (R-016)~~ → **Fixed v1.8.3**: Implicit Integer/Float coercion now works for `*`
17. ~~Avoid `%` with floats; simulate modulo or convert first (R-017)~~ → **Fixed v1.8.3**: `%` now works with floats
18. ~~State-modifying helpers called from contract methods have side-effects silently dropped — call them at module level instead (R-018)~~ → **Fixed v1.8.3**: `clone_for_closure` now references live env
19. ~~Use list/map **literals** for bulk initialization instead of loops of `push()`/`map[k]=v` (R-019)~~ → **Fixed v1.8.3**: Same closure env fix as R-018
20. Preferred file layout order: `imports → constants → protocol → helper actions → contract → module-level let vars → module-level helpers → entities → test → exports`
21. Use intermediate variable for `self.field[key] = val` — do `let f = self.field; f[key] = val` (R-020)
22. `+=` operator is broken — use `x = x + 1` instead (R-021)
23. `not` keyword not recognized — use `!` for negation (R-021)
24. State fields cannot be initialized with `[]` or `{}` — they become `null`. Use primitive defaults in `state {}`, store lists/maps as module-level `let` vars (R-022)
25. `.slice()` not supported on lists — use manual while-loop copy
26. `.contains()` not supported on strings — use `==` comparison
27. `.remove()` / `.delete()` not supported on maps — set key to `null` or rebuild map
28. `.filter()` / `.sort()` with callbacks not supported — "Identifier 'action' not found" — use manual loops
29. `for i in 0..N` range loop broken — "Identifier 'i' not found" — use `while` loop with counter
30. `cache_set()` / `cache_get()` standalone functions not found — use maps as cache
31. `math.floor()` not found — use `int()` for truncation
32. Skip separate key-tracking lists for maps — iterate maps directly with `for each key, val in map` (R-015 still drops pushes interleaved with map bracket-assigns even on different variables)

## Phase 0 Rewrite Progress

| File | Lines (original) | Status | Notes |
|------|-----------------|--------|-------|
| src/core/block.zx | ~440 | ✅ Done | Genesis, block creation, validation, merkle root |
| src/core/crypto.zx | ~370 | ✅ Done | Quantum-resistant crypto, keypairs, signing |
| src/core/consensus.zx | ~420 | ✅ Done | PoS consensus, validators, rewards, slashing |
| src/core/state.zx | ~600 | ✅ Done | Self-evolving state, parameters, snapshots, rollback |
| src/core/zvm.zx | ~1674 | ✅ Done | Ziver Virtual Machine — deploy, execute, upgrade, security, metrics |
| src/core/social_capital.zx | ~1238→~530 | ✅ Done | 15/15 tests pass |
| src/core/seb_defi.zx | ~1213 | ❌ Not started | SEB DeFi protocols |
| src/network/network.zx | ~1819 | ❌ Not started | P2P networking |
| src/network/p2p.zx | ~2870 | ❌ Not started | Peer discovery |
| src/rpc/server.zx | ~2753 | ❌ Not started | JSON-RPC server |
| src/rpc/websocket.zx | ~3006 | ❌ Not started | WebSocket server |
| src/runtime/contract_runtime.zx | ~583 | ❌ Not started | Contract execution |
| src/runtime/state_manager.zx | ~462 | ❌ Not started | Runtime state management |
| src/contracts/token_standard.zx | ~198 | ❌ Not started | Token standard |
| src/contracts/wallet.zx | ~239 | ❌ Not started | Wallet |
| src/contracts/bridge.zx | ~174 | ❌ Not started | Bridge |
| src/contracts/test_contract.zx | ~20 | ❌ Not started | Test contract |
| src/main.zx | ~263 | ❌ Not started | Entry point |

---

## additional fixes made:

Created and ran 24 test files covering all fixes and edge cases. Found and fixed 8 additional bugs during testing:

Parser fixes (commits 1635369 through 985f081):

Consecutive this.property = val assignments failed in strategy parser (THIS token not recognized as statement starter)
else if chains didn't work (ELSE + IF token sequence not handled)
Keywords after DOT (e.g., this.debug, return this.data) broke statement boundary detection
Entity default values (name: string = "localhost") weren't parsed in either parser
Entity properties with keyword names (e.g., debug: boolean) weren't recognized
Entity default value boundary detection failed when next property used a keyword name
Evaluator fixes:

Added missing range(start, end, step), typeof(val), and abs(num) built-in functions
Test results: 24 .zx test files all passing, 300 unit tests passing, 0 CodeQL alerts.

---
*Last updated: 2026-03-02 — v1.8.3, zvm.zx rewrite in progress, R-021 through R-025 new*