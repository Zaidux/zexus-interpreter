# Zexus v1.8.0 — VM & Interpreter Issues

> Discovered during Ziver Chain Phase 0 audit (Feb 23, 2026).
> Target fix version: **v1.8.1**
> Tested with: `zexus 1.8.0` via `pip install zexus`, running on Ubuntu 24.04

---

## Bytecode VM Issues

### VM-001: EntityStatement compilation crash
- **Severity:** High
- **Trigger:** Any file containing an `entity` declaration
- **Error:** `Bytecode compilation error: Failed to compile Program: Failed to compile EntityStatement: 'EntityStatement' object has no attribute 'body'`
- **Workaround:** Falls back to interpreter automatically
- **Repro:**
```zexus
entity Point {
    x: integer
    y: integer
}
let p = Point{x: 3, y: 4}
```

### VM-002: ExportStatement not supported in bytecode compiler
- **Severity:** Medium
- **Trigger:** Any file with `export contract` or `export action`
- **Error:** `VM fallback: Node type 'ExportStatement' is not currently supported by the bytecode compiler.`
- **Workaround:** Falls back to interpreter
- **Repro:**
```zexus
export contract MyContract {
    data val = 42
    action get_val() { return val }
}
```

### VM-003: Imported functions resolve to Action objects instead of being called
- **Severity:** High
- **Trigger:** Calling a function imported from another `.zx` file in VM mode
- **Error:** `add result: action(a, b) { ... }` — prints the function object instead of calling it
- **Workaround:** Falls back to interpreter when VM error occurs
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

### VM-004: Entity constructor argument count mismatch
- **Severity:** Low
- **Trigger:** Entity with named-field construction `Entity{field: value}`
- **Error:** `[TypeError] 'Point' expects 0 argument(s) but got 1`
- **Note:** Only in bytecode validation pass; interpreter handles it correctly
- **Repro:**
```zexus
entity Point {
    x: integer
    y: integer
}
let p = Point{x: 3, y: 4}  # TypeError in validation
```

### VM-005: `string()` + builtin module returns type name instead of value
- **Severity:** Medium
- **Trigger:** `string(dt.now())` or `string(math.max(3,7))` in VM mode
- **Error:** Prints `<built-in function: >` or `None` instead of the actual value
- **Workaround:** Interpreter handles these correctly
- **Repro:**
```zexus
use "datetime" as dt
let now = dt.now()
print("DateTime: " + string(now))  # VM: "<built-in function: >"
```

---

## Interpreter Issues

### INT-001: `protocol` keyword not recognized
- **Severity:** High
- **Trigger:** Using `protocol` to define an interface
- **Error:** `Identifier 'protocol' not found`
- **Impact:** Cannot use protocol/implements pattern at all
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

### INT-002: `implements` keyword not recognized
- **Severity:** High  
- **Trigger:** Using `implements` on a contract (consequence of INT-001)
- **Error:** Parse error when `protocol` is not recognized
- **Note:** Tied to INT-001; fixing `protocol` should fix this too

### INT-003: `emit` not recognized outside or inside contracts
- **Severity:** Critical
- **Trigger:** Using `emit EventName(args)` anywhere
- **Error:** `Identifier 'emit' not found`
- **Impact:** Cannot emit blockchain events — core blockchain feature is broken
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

### INT-004: `persistent storage` declarations not functional
- **Severity:** Medium
- **Trigger:** Using `persistent storage varname: type` in a contract
- **Behavior:** Parses without error but doesn't provide any persistence; behaves identically to `data`
- **Impact:** No actual on-chain persistence — purely cosmetic keyword
- **Repro:**
```zexus
contract Store {
    persistent storage items: map
    action init() { items = {} }
    action set(k, v) { items[k] = v }
    action get(k) { return items[k] }
}
```

### INT-005: `track_memory()` not recognized
- **Severity:** Low
- **Trigger:** Calling `track_memory()` at module top level
- **Error:** `Identifier 'track_memory' not found`
- **Impact:** Memory tracking feature not available

### INT-006: `cache()` not recognized
- **Severity:** Low
- **Trigger:** Calling `cache("name", {ttl: 300})` at module top level
- **Error:** `Identifier 'cache' not found`
- **Impact:** Cache directive feature not available

### INT-007: `throttle()` not recognized
- **Severity:** Low
- **Trigger:** Calling `throttle("name", {requests_per_minute: 100})`
- **Error:** `Identifier 'throttle' not found`
- **Impact:** Rate-limiting directive feature not available

### INT-008: `audit()` not recognized
- **Severity:** Low
- **Trigger:** Calling `audit("event", {data})` anywhere
- **Error:** `Identifier 'audit' not found`
- **Impact:** Audit trail feature not available

### INT-009: `verify()` not recognized as a standalone function
- **Severity:** Low
- **Trigger:** Calling `verify(condition, "message")` (not `require`)
- **Error:** `Identifier 'verify' not found`
- **Note:** `require(condition, "message")` works correctly; `verify` is the missing alias

### INT-010: `watch` blocks not recognized
- **Severity:** Low
- **Trigger:** Reactive `watch variable { ... }` blocks
- **Error:** Parse error
- **Impact:** Reactive programming feature not available
- **Repro:**
```zexus
contract Tracker {
    data count = 0
    watch count {
        print("Count changed to: " + string(count))
    }
}
```

### INT-011: `protect` action modifier not recognized
- **Severity:** Low
- **Trigger:** Using `action protect method_name()` syntax
- **Error:** Parse error or ignored
- **Impact:** Cannot mark actions as protected/internal

### INT-012: `bc.create_genesis_block()` returns function object without call
- **Severity:** Medium
- **Trigger:** Calling `bc.create_genesis_block()` and inspecting result in VM mode
- **Behavior:** VM returns `<built-in function: >` instead of executing
- **Note:** Interpreter mode returns the correct block map
- **Workaround:** Use interpreter mode (automatic fallback)

### INT-013: `map.has()` method not available
- **Severity:** Medium
- **Trigger:** Calling `.has("key")` on a map
- **Error:** Method not found or no-op
- **Workaround:** Use `map["key"] != null` instead
- **Repro:**
```zexus
let m = {"a": 1}
print(m.has("a"))  # Fails or returns unexpected result
# Workaround:
if m["a"] != null { print("has key a") }
```

### INT-014: `map.get(key, default)` method not available
- **Severity:** Medium
- **Trigger:** Calling `.get("key", default_value)` on a map
- **Workaround:** Use null check pattern:
```zexus
let val = m["key"]
if val == null { val = default_value }
```

### INT-015: `list.is_empty()` method not available
- **Severity:** Low
- **Trigger:** Calling `.is_empty()` on a list
- **Workaround:** Use `len(list) == 0`

### INT-016: `list.count()` vs `len()` inconsistency
- **Severity:** Low
- **Trigger:** `.count()` may not work on all list types
- **Workaround:** Always use `len(list)` which works reliably

---

## What Works Correctly (v1.8.0)

For reference, these features work as expected:

| Feature | Status |
|---|---|
| `entity` declarations + field construction | ✅ Works (interpreter) |
| `contract` with `data` and `action` | ✅ Works |
| `export contract` / `export action` | ✅ Works (interpreter) |
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
| `bc.create_genesis_block()` | ✅ Works (interpreter) |
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

---

## Suggested Priority for v1.8.1

| Priority | Issue | Impact |
|---|---|---|
| **P0** | INT-003: `emit` broken | Cannot build any event-driven blockchain |
| **P0** | INT-001/002: `protocol`/`implements` broken | Cannot define interfaces |
| **P1** | VM-001: Entity compilation crash | Forces interpreter fallback on all files with entities |
| **P1** | VM-003: Imported functions not called | Forces interpreter fallback on all multi-file projects |
| **P1** | INT-013/014: `map.has()`/`map.get()` missing | Very common patterns, need workarounds everywhere |
| **P2** | VM-002: ExportStatement not compiled | Forces fallback on exported modules |
| **P2** | INT-004: `persistent storage` no-op | Cosmetic only, but misleading |
| **P2** | VM-004: Entity constructor validation | False positive error in validation |
| **P3** | INT-005–011: Missing directives | `track_memory`, `cache`, `throttle`, `audit`, `verify`, `watch`, `protect` |