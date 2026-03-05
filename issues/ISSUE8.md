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
| R-024 | Interpreter / range-loop | `for i in 0..N {}` and `for x in list {}` — "Identifier not found" inside body. Only `for each x in list` syntax works | Medium | Use `while` loop with counter, or `for each x in list` |
| R-025 | Interpreter / builtins | `cache_set()`/`cache_get()` standalone not found; `math.floor()` not found | Low | Use maps for cache; `int()` for floor |
| R-026 | Interpreter / entity-defaults | Entity fields not explicitly set during construction become `null` instead of declared defaults — `Foo{name: "test"}` leaves `value`, `weight` etc. as `null` | **High** | Always specify ALL entity fields during construction |

| R-027 | Interpreter / match | `match val { case "x" { } }` — case body NEVER executes (string, integer, variable). Match parses but silently produces no result. | **Critical** | Use `if/elif` chains instead |
| R-028 | Interpreter / self-call | `self.method()` fire-and-forget (no return capture) — only the FIRST call executes per contract action, subsequent calls silently dropped | **Critical** | Always capture return: `let _ = self.method()` |
| R-029 | Interpreter / anon-action | Anonymous `action()` stored in map key — calling `map["key"](args)` returns "Not a function" | **High** | Use named module-level actions instead of anonymous closures in maps |
| R-030 | Interpreter / closure-maps | Too many anonymous multi-line `action()` closures assigned to map keys inside a function body causes "Identifier not found" parser confusion in large files | **High** | Use named module-level helper actions instead |
| R-031 | Interpreter / string-methods | String methods missing: `.toLowerCase()`, `.toUpperCase()`, `.substring()`, `.indexOf()`, `.split()`, `.trim()`, `.replace()`, `.startsWith()`, `.endsWith()` — all return "Method not supported for STRING" | **High** | Use manual comparison, `string()` conversion, or workaround with character iteration |
| R-032 | Interpreter / list-methods | List methods missing: `.join()`, `.reverse()`, `.includes()`, `.pop()`, `.indexOf()`, `.concat()` — all return "Method not supported for LIST" | **High** | Use `for each` loops with manual accumulation; use `+` operator for list concat |
| R-033 | Interpreter / string-concat | `"str" + 42` or `"str" + true` or `"str" + null` — "Type mismatch: cannot add STRING and INTEGER/BOOLEAN/NULL" | Medium | Always wrap with `string()`: `"str" + string(42)` |
| R-034 | Interpreter / map-delete | `map.delete(key)` and `map.remove(key)` — "Method not supported for MAP" | Medium | Rebuild map without key using `for each`, or set value to `null` |
| R-035 | Interpreter / map-merge | `map1 + map2` — "Type error: cannot add MAP and MAP" | Medium | Use `for each k, v in map2 { map1[k] = v }` to merge |
| R-036 | Interpreter / bool-cast | `bool(1)` — "Identifier 'bool' not found" | Low | Use `val == true`, `val != false`, or `!!val` pattern |
| R-037 | Interpreter / map-numeric-key | `map.has(0)` returns false even when key `0` exists; `map[0]` still retrieves value | Medium | Avoid numeric map keys, or use `string(n)` as key |
| R-038 | Interpreter / primitive-mutation | Module-level primitive (`let x = 0`) mutated via helper function called from contract → changes silently lost (closure copies primitive) | **High** | Use module-level maps instead: `let _state = {}; _state["counter"] = 0` — map mutations propagate |
| R-039 | Security / url-strings | String containing `"http://"` in construction (e.g. `"http://0.0.0.0:"`) triggers Zexus security error — flagged as unsanitized URL injection | Medium | Omit URL scheme from strings; store host/port separately |

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

## VM Mode Issues (v1.8.3)

> Tested with `zx run FILE` (no `--no-vm` flag). VM compiles to bytecode, then runs on Python VM.
> "Rust VM: not compiled — using Python VM" is expected.
> Testing conducted 2026-03-05 — 86 tests (VT1–VT90) across 9 test batches.

### VM Issue Summary

| ID | Category | Description | Severity | Interp? |
|----|----------|-------------|----------|---------|
| V-001 | VM / Contract State | `instance.property` always returns `null` — direct property access broken | **Critical** | ✅ Works |
| V-002 | VM / Contract State | Map/list state does NOT persist between contract method calls (only primitives persist via methods) | **Critical** | ✅ Works |
| V-003 | VM / Modules | All built-in modules return null/empty: `crypto.sha256()`, `datetime.now()`, `json.encode/decode`, `math.abs/max/min` | **Critical** | ✅ Works |
| V-004 | VM / Closures | Anonymous `action()` closures return null: `let f = action(a,b) { return a+b }; f(3,4)` = null | **High** | ❌ Also broken (R-029) |
| V-005 | VM / Enums | `Color.Red` returns `null` instead of `0` | **High** | ✅ Works (returns 0) |
| V-006 | VM / Defer | `defer { ... }` block never executes — silently dropped | **High** | ✅ Works |
| V-007 | VM / For-In | `for x in list` doesn't iterate (sum stays 0, no error). Only `for each x in list` works | Medium | ❌ Also broken (crashes) |
| V-008 | VM / List Concat | `[1,2] + [3,4]` returns `len=2` instead of `4` — list concatenation via `+` broken | **High** | ✅ Works (len=4) |
| V-009 | VM / Map Return | Map returned from contract method has `keys()=0` but bracket access `map["key"]` still works | Medium | ✅ Works |
| V-010 | VM / Module from Contract | Calling module-level actions from contract throws "Not a function: Map object" | **Critical** | ✅ Works |

### V-001 Details — Contract Direct Property Access Returns Null
**Severity:** Critical  
**Symptom:** On VM, `instance.property` ALWAYS returns `null`, even after `init()` assigns the value.
```zx
contract Foo { state { x: 0 } action init() { self.x = 42 } }
let foo = Foo{}
print(foo.x)            // VM: null  ❌  (Interpreter: 42 ✅)
```
**However:** Accessor methods DO work for primitives:
```zx
print(foo.get_x())       // VM: 42  ✅
```
**State persistence between methods (primitives only):**
```zx
let _ = foo.set_x(99)
print(foo.get_x())       // VM: 99  ✅  (state persists between calls via methods)
print(foo.x)             // VM: null ❌  (direct access still broken)
```
**Workaround:** Always use getter/setter methods. Never access contract properties directly.

### V-002 Details — Map/List State Doesn't Persist on VM
**Severity:** Critical  
**Symptom:** Primitive state (int, string, bool) persists between method calls via accessor methods. But map and list state does NOT persist — getter methods return `null`.
```zx
contract MapHolder {
    state { data: {} }
    action init() { self.data = { "key1": "val1" } }
    action get(k) { return self.data[k] }
}
let mh = MapHolder{}
print(mh.get("key1"))    // VM: null ❌ (Interpreter: "val1" ✅)
```
**Workaround:** For VM, store maps/lists as module-level variables instead of contract state.

### V-003 Details — Built-in Modules Return Null/Empty on VM
**Severity:** Critical  
**Symptom:** All `use "module"` imports compile but module method calls return null or empty string.
```zx
use "crypto"
print(crypto.sha256("hello"))   // VM: ""    ❌  (Interpreter: hash string ✅)
use "datetime"
print(datetime.now())            // VM: null  ❌  (Interpreter: timestamp ✅)
use "json"
print(json.encode({a: 1}))      // VM: ""    ❌  (Interpreter: JSON string ✅)
use "math"
print(math.abs(-5))              // VM: null  ❌  (Interpreter: 5 ✅)
```
**Workaround:** Use `--no-vm` (interpreter mode) when built-in modules are needed.

### V-005 Details — Enums Return Null on VM
**Severity:** High  
**Symptom:** Enum member access returns `null` instead of the ordinal integer.
```zx
enum Color { Red, Green, Blue }
let c = Color.Red
print(string(c))   // VM: null  ❌  (Interpreter: 0 ✅)
```
**Workaround:** Use constants instead of enums on VM: `const COLOR_RED = 0`

### V-006 Details — Defer Block Never Executes on VM
**Severity:** High  
**Symptom:** `defer { ... }` block inside an action is silently ignored — the deferred code never runs.
```zx
action cleanup() {
    defer { print("cleaning up") }   // VM: NEVER PRINTS ❌
    print("doing work")
    return "done"
}
let r = cleanup()   // prints "doing work" only, no "cleaning up"
```
**Workaround:** Call cleanup code explicitly before return instead of using defer.

### V-008 Details — List Concatenation Returns Wrong Length on VM
**Severity:** High  
**Symptom:** `list1 + list2` returns a list with length of first list only, not the combined length.
```zx
let a = [1, 2]
let b = [3, 4]
let c = a + b
print(len(c))   // VM: 2  ❌  (Interpreter: 4 ✅)
```
**Workaround:** Use `.push()` in a loop to combine lists:
```zx
let c = []
for each item in a { c.push(item) }
for each item in b { c.push(item) }
```

### VM Working Feature Reference

| Feature | VM | Interp | Notes |
|---------|------|--------|-------|
| `print()` / `debug()` | ✅ | ✅ | |
| `let` / `const` | ✅ | ✅ | |
| `if / elif / else` | ✅ | ✅ | |
| `while` / `break` / `continue` | ✅ | ✅ | |
| `for each item in list` | ✅ | ✅ | |
| `for each i, item in list` | ✅ | ✅ | Indexed |
| `for each key, val in map` | ✅ | ✅ | |
| `for x in list` (no "each") | ❌ | ❌ | Silent 0 on VM, crash on interp |
| `try / catch / throw` | ✅ | ✅ | |
| `match / case` | ❌ | ❌ | R-027: broken on both |
| `ternary ? :` | ✅ | ✅ | Nested works too |
| Module-level `action()` | ✅ | ✅ | |
| Recursion | ✅ | ✅ | factorial(10) works |
| Higher-order named actions | ✅ | ✅ | Pass named action as arg |
| Anonymous `action()` closures | ❌ | ❌ | V-004 / R-029 |
| `entity` | ✅ | ✅ | Field access works on both |
| `protocol` / `implements` | ✅ | ✅ | |
| `contract` basic | ⚠️ | ✅ | VM: no direct state access (V-001) |
| Contract direct property access | ❌ | ✅ | V-001: always null on VM |
| Contract accessor methods (primitives) | ✅ | ✅ | get/set pattern works |
| Contract accessor methods (maps/lists) | ❌ | ✅ | V-002: null on VM |
| Contract pure compute methods | ✅ | ✅ | No state, just args + return |
| Contract self-calls from methods | ✅ | ⚠️ | Works on VM; interp needs `let _=` (R-028) |
| Module action from contract | ❌ | ✅ | V-010: "Not a function" on VM |
| `enum` | ❌ | ✅ | V-005: null on VM |
| `defer` | ❌ | ✅ | V-006: never fires on VM |
| `use "crypto"` | ❌ | ✅ | V-003 |
| `use "datetime"` | ❌ | ✅ | V-003 |
| `use "json"` | ❌ | ✅ | V-003 |
| `use "math"` | ❌ | ✅ | V-003 |
| String methods (upper/lower/split/replace/contains) | ✅ | ❌ | VM has them! Interp missing (R-031) |
| `typeof()` | ✅ | ✅ | |
| `len()` / `string()` / `int()` / `float()` | ✅ | ✅ | |
| Map bracket access / keys / values | ✅ | ✅ | |
| List push / pop / index access | ✅ | ✅ | |
| `list + list` concatenation | ❌ | ✅ | V-008: wrong length on VM |
| `map + map` merge | ❌ | ❌ | R-035: broken on both |
| Large loops (1000+) | ✅ | ✅ | |
| Nested loops | ✅ | ✅ | |
| Large maps/lists (100+) | ✅ | ✅ | |
| Float arithmetic | ✅ | ✅ | Minor precision (normal) |
| Boolean logic (`&&` `||` `!`) | ✅ | ✅ | |
| Comparison operators | ✅ | ✅ | `==`, `!=`, `>`, `<`, `>=`, `<=` |
| `trail()` | ✅ | ✅ | No crash, logging |
| String escape sequences (`\n`, `\t`) | ✅ | ✅ | |
| `int()` / `float()` type casting | ✅ | ✅ | From strings too |
| Map from action return | ✅ | ✅ | keys/bracket access correct |
| List from action return | ✅ | ✅ | len/index access correct |
| Map returned from contract | ⚠️ | ✅ | V-009: keys()=0 but bracket works |

### Key VM vs Interpreter Differences

**VM advantages:**
- String methods work (`.upper()`, `.lower()`, `.contains()`, `.split()`, `.replace()`)
- List `.pop()` works
- No R-028 self-call issue (self-calls work without capture)

**VM disadvantages:**
- ALL built-in modules broken (crypto, datetime, json, math)
- Contract direct state access always null
- Map/list state doesn't persist in contracts
- Enums broken
- Defer broken
- List concatenation returns wrong length
- Closures return null
- Module actions from contract throw "Not a function"

**Recommendation:** Use interpreter mode (`--no-vm`) for production code. VM is unsuitable for any contract-heavy or module-dependent code.

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
| `try/catch/throw` | ✅ Works | try/catch catches runtime errors; throw works |
| `match val { ... }` | ❌ **Broken** | **R-027**: Case bodies never execute. Use `if/elif` chains instead |
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
| `self.method()` fire-and-forget | ⚠️ **Buggy** | **R-028**: Only FIRST `self.method()` per action executes if return not captured. Use `let _ = self.method()` |
| `self.method()` with return capture | ✅ Works | `let r = self.method()` — all calls execute correctly |
| `string + non-string` auto-coercion | ❌ Broken | **R-033**: Must use `string()` wrapper: `"val=" + string(42)` |
| String methods (toLowerCase, split, etc.) | ❌ Missing | **R-031**: No string methods on interpreter. **Note:** VM has `.upper()`, `.lower()`, `.contains()`, `.split()`, `.replace()` |
| List methods (join, pop, indexOf, etc.) | ❌ Missing | **R-032**: Only `.push()`, `.is_empty()` work. Use loops |
| `list + list` concatenation | ✅ Works | `[1,2] + [3,4]` returns `[1,2,3,4]` |
| `map.delete()` / `map.remove()` | ❌ Missing | **R-034**: Rebuild map without key, or set to null |
| `map + map` merge | ❌ Missing | **R-035**: Use `for each` loop to copy keys |
| Anonymous `action()` in map | ❌ Broken | **R-029**: "Not a function" when calling. Use named module-level actions |
| `bool()` cast | ❌ Missing | **R-036**: Use truthy comparisons instead |
| Map numeric keys with `.has()` | ⚠️ **Buggy** | **R-037**: `.has(0)` returns false; bracket access `map[0]` works. Use string keys |
| Module-level primitive from contract | ⚠️ **Buggy** | **R-038**: Primitive mutation via helper function silently lost. Use maps |
| Module-level map from contract | ✅ Works | `_map["key"] = val` from contract methods and init works |
| `string[0]` char access | ✅ Works | Bracket indexing on strings works |
| `typeof(val)` | ✅ Works | Returns: integer, float, string, boolean, null, list, map |
| `break` / `continue` in loops | ✅ Works | Both work in `while` and `for each` |
| Ternary `cond ? a : b` | ✅ Works | Conditional expression works |
| `elif` keyword | ✅ Works | Works with 16+ branches, from module-level and contract |
| Deep nested returns | ✅ Works | `return` from nested `if` blocks inside contract works |
| `debug()` inside contracts | ✅ Works | Both `print()` and `debug()` work inside contract methods |

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
29. `for i in 0..N` range loop and `for x in list` both broken — use `while` loop with counter, or `for each x in list` (R-024, V-007)
30. `cache_set()` / `cache_get()` standalone functions not found — use maps as cache
31. `math.floor()` not found — use `int()` for truncation
32. Skip separate key-tracking lists for maps — iterate maps directly with `for each key, val in map` (R-015 still drops pushes interleaved with map bracket-assigns even on different variables)
33. Always specify ALL entity fields during construction — unset fields become `null` even if entity declares defaults (R-026)
34. `match` is completely broken — always use `if/elif` chains for dispatch/pattern matching (R-027)
35. Always capture `self.method()` return values: `let _ = self.method()` — fire-and-forget calls after the first are silently dropped (R-028)
36. Never use anonymous `action()` closures in map keys — use named module-level actions (R-029, R-030)
37. Always wrap non-strings with `string()` before concatenation: `"val=" + string(42)` (R-033)
38. No string methods available — no `.split()`, `.trim()`, `.toLowerCase()` etc. Use manual workarounds (R-031)
39. No list `.pop()`, `.join()`, `.reverse()`, `.includes()`, `.indexOf()` — only `.push()` and `.is_empty()` work. Use `+` for list concat (R-032)
40. Can't delete map keys — rebuild map without key or set to null (R-034)
41. Can't merge maps with `+` — use `for each` to copy keys (R-035)
42. Avoid numeric map keys — `.has(0)` returns false. Use `string(n)` as key (R-037)
43. Module-level primitives can't be reliably mutated from contract via helper functions — use module-level maps with string keys instead (R-038)
44. Avoid URL scheme strings like `"http://"` — triggers Zexus security system (R-039)

## Phase 0 Rewrite Progress

| File | Lines (original) | Status | Notes |
|------|-----------------|--------|-------|
| src/core/block.zx | ~440 | ✅ Done | Genesis, block creation, validation, merkle root |
| src/core/crypto.zx | ~370 | ✅ Done | Quantum-resistant crypto, keypairs, signing |
| src/core/consensus.zx | ~420 | ✅ Done | PoS consensus, validators, rewards, slashing |
| src/core/state.zx | ~600 | ✅ Done | Self-evolving state, parameters, snapshots, rollback |
| src/core/zvm.zx | ~1674 | ✅ Done | Ziver Virtual Machine — deploy, execute, upgrade, security, metrics |
| src/core/social_capital.zx | ~1238→~530 | ✅ Done | 15/15 tests pass |
| src/core/seb_defi.zx | ~1213→~750 | ✅ Done | 18/18 tests pass |
| src/network/network.zx | ~1819→~1139 | ✅ Done | 18/18 tests pass |
| src/network/p2p.zx | ~2870→1425 | ✅ Done | 20/20 tests pass |
| src/rpc/server.zx | ~2753 | 🔧 In Progress | Rewritten, blocked by R-028 (self-call from init). Workaround: register methods at module level |
| src/rpc/websocket.zx | ~3006 | ❌ Not started | WebSocket server |
| src/runtime/contract_runtime.zx | ~583 | ❌ Not started | Contract execution |
| src/runtime/state_manager.zx | ~462 | ❌ Not started | Runtime state management |
| src/contracts/token_standard.zx | ~198 | ❌ Not started | Token standard |
| src/contracts/wallet.zx | ~239 | ❌ Not started | Wallet |
| src/contracts/bridge.zx | ~174 | ❌ Not started | Bridge |
| src/contracts/test_contract.zx | ~20 | ❌ Not started | Test contract |
| src/main.zx | ~263 | ❌ Not started | Entry point |

---

### R-027 Details — `match` Statement Broken
**Severity:** Critical  
**Symptom:** `match val { case "x" { print("found") } default { print("default") } }` — NO case body ever executes, not even default. Match runs silently with no output. Tested with strings, integers, and variable references — all fail.  
**Workaround:** Use `if/elif` chains:
```zx
if val == "a" { ... }
elif val == "b" { ... }
else { ... }
```
**Verified:** `if/elif` works correctly with 16+ branches, from both module level and contract methods.

### R-028 Details — `self.method()` Fire-and-Forget Only First Executes
**Severity:** Critical  
**Symptom:** When calling `self.method()` multiple times within a contract action WITHOUT capturing the return value, only the FIRST call executes. Subsequent calls are silently dropped.
```zx
action do_all() {
    self.set_a()   // ✅ Executes
    self.set_b()   // ❌ Silently dropped
    self.set_c()   // ❌ Silently dropped
}
```
**Workaround:** Always capture the return value, even if unused:
```zx
action do_all() {
    let _ = self.set_a()   // ✅ Executes
    let _2 = self.set_b()  // ✅ Executes
    let _3 = self.set_c()  // ✅ Executes
}
```
**Verified:** With `let r = self.method()`, ALL calls execute correctly including state mutations.  
**Note:** Self-call CHAINS (self.a() → self.b() → self.c()) also only execute first level. Module-level function calls from contract are NOT affected — they all execute.

### R-029/R-030 Details — Anonymous Actions in Maps
**Severity:** High  
**Symptom (R-029):** Assigning `handlers["key"] = action(x) { return x }` and then calling `handlers["key"]("arg")` throws "Not a function".  
**Symptom (R-030):** In large files, having too many anonymous multi-line `action()` closures assigned to map keys inside a function body causes "Identifier not found" parser errors for unrelated identifiers.  
**Workaround:** Define all handler functions as named module-level actions and reference them by name.

### R-031 Details — String Methods Missing
**Severity:** High  
**All missing:** `.toLowerCase()`, `.toUpperCase()`, `.substring()`, `.indexOf()`, `.split()`, `.trim()`, `.replace()`, `.startsWith()`, `.endsWith()` — all throw "Method not supported for STRING"  
**Available:** `len(str)`, `string(val)`, `str[0]` (char access by index)  
**Workarounds:**
- Use `==` comparison instead of `.startsWith()`/`.endsWith()`
- Use character-level iteration for search
- Use `string()` for type conversion

### R-032 Details — List Methods Missing
**Severity:** High  
**All missing:** `.join()`, `.reverse()`, `.includes()`, `.pop()`, `.indexOf()`, `.concat()` — all throw "Method not supported for LIST"  
**Available:** `.push(val)`, `.is_empty()`, `len(list)`, `list[i]` (index access), `list + list` (concatenation via `+`)  
**Workarounds:**
- Use `+` for concatenation: `[1,2] + [3,4]`
- Use `for each` with accumulator for join/includes/indexOf
- Use `while` loop for reverse

### R-033 Details — String Concatenation Type Mismatch
**Severity:** Medium  
**Symptom:** `"val=" + 42` throws "Type mismatch: cannot add STRING and INTEGER". Same for BOOLEAN and NULL.  
**Workaround:** Always use `string()`: `"val=" + string(42)`, `"flag=" + string(true)`

### R-037 Details — Map Numeric Keys
**Severity:** Medium  
**Symptom:** `map[0] = "zero"; map.has(0)` returns `false`. But `map[0]` retrieves "zero" correctly.  
**Workaround:** Use string keys: `map[string(0)] = "zero"; map.has(string(0))` → `true`

### R-038 Details — Module-Level Primitive Mutation from Contract
**Severity:** High  
**Symptom:** A module-level `let x = 0` modified by a helper function (`action inc() { x = x + 1 }`) called from a contract method — the change is silently lost (closure creates a copy of the primitive).  
**Note:** Direct assignment from contract (`_val = 42`) DOES work. Module-level MAP mutations also work (maps are reference types). Only primitives via intermediate helper functions fail.  
**Workaround:** Store counters/flags in a module-level map:
```zx
let _state = {}
_state["counter"] = 0
action inc() { _state["counter"] = _state["counter"] + 1 }  // ✅ Works from contract
```

### R-039 Details — URL String Security Flag
**Severity:** Medium  
**Symptom:** Constructing a string containing `"http://"` (e.g., `"http://0.0.0.0:" + string(port)`) triggers Zexus security system — "unsanitized URL injection risk".  
**Workaround:** Omit URL scheme; store host:port without protocol prefix.

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

## Changelog

- **2026-03-05** — v1.8.3, Issues R-027 through R-039 discovered via systematic interpreter testing (48 tests, T1–T48)
- **2026-03-05** — v1.8.3, VM mode systematic testing: 86 tests (VT1–VT90) across 9 batches. Discovered 10 VM-specific issues (V-001 through V-010). VM is significantly more broken than interpreter — NOT suitable for production contract-heavy code.

---
*Last updated: 2026-03-05 — v1.8.3, Issues R-027 through R-039 discovered via systematic testing (server.zx rewrite in progress, 9/17 done)*