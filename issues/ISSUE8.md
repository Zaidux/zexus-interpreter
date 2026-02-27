# Zexus v1.8.2 — Issues Found During Phase 0 Rewrite

> This file tracks issues discovered while rewriting Ziver-Chain source files for Zexus v1.8.2.
> Updated as issues are found.

## Known Quirks (from initial testing)

| ID | Category | Description | Severity | Workaround |
|----|----------|-------------|----------|------------|
| Q-001 | map.has() | Returns `None` instead of `true/false` | Minor | Use `map.get(key, null) != null` or bracket access |
| Q-002 | list.is_empty() | Returns `None` instead of `true/false` | Minor | Use `len(list) == 0` |
| Q-003 | bc.createChain() | Returns `false` | Needs investigation | May require config params |

## Issues Found During Phase 0 Rewrite

| ID | Category | Description | Severity | Workaround |
|----|----------|-------------|----------|------------|
| R-001 | VM / Entity | Entity field access returns `None` on VM (works in interpreter) | **Critical** | Use `--no-vm` or let fallback handle it |
| R-002 | VM / Contract | Contract state not initialized on VM; `this.val` is `null` even with `state { val: 0 }` | **Critical** | Use `--no-vm` + manual `init()` call |
| R-003 | Interpreter / self | `self` keyword not recognized in interpreter — must use `this` | **High** | Replace all `self.` with `this.` |
| R-004 | Interpreter / init | `init()` is NOT auto-called when constructing contract with `Contract{}` | **High** | Manually call `.init()` after creation |
| R-005 | Interpreter / state | `state { val: 0 }` does NOT initialize fields to default values | **High** | Always use `init()` to set state |
| R-006 | Interpreter / for-each | `for each i, item in list` (indexed) fails: "Identifier 'i' not found" | **High** | Use `while` loop with manual index counter |
| R-007 | Interpreter / for-each | `for each key, val in map` fails: "ForEach expects List" | **High** | Track keys in separate list; use while loop |
| R-008 | Interpreter / protect | `action protect` on module-level actions fails: "Identifier not found" | Medium | Use `protect` only inside contracts |
| R-009 | Interpreter / nested assign | `m["a"]["b"] = val` fails: "Invalid assignment target" | Medium | Use intermediate variable: `let inner = m["a"]; inner["b"] = val` |
| R-010 | VM / output | Some complex programs compile to bytecode but produce no output | Medium | Use `--no-vm` flag for reliable execution |
| R-011 | Import / entity | Exported entities from other files can't be used as constructors in importing file | Medium | Use factory functions that return map objects |
| R-012 | Interpreter / order | Entity declarations before a contract make the contract identifier invisible ("Identifier not found") | **High** | Declare all entities AFTER contracts, or use forward references (entities declared after contract can still be used inside it) |
| R-013 | Interpreter / state | Multiple fields in `state { }` causes "Invalid assignment target" — only ONE field allowed | **Critical** | Use single `state { placeholder: 0 }` + module-level `let` variables for additional state |
| R-014 | Interpreter / order | `let` variable declarations before a contract also break contract visibility | **High** | Declare module-level `let` variables AFTER contract; contract methods can forward-reference them |
| R-015 | Interpreter / push+map | After `map[key] = val` in a contract method, subsequent `list.push()` is silently ignored | **High** | Always do `list.push()` BEFORE `map[key] = val` in the same method, or split into separate calls |
| R-016 | Interpreter / types | `INTEGER * FLOAT` causes "Type mismatch" — no implicit coercion | **High** | Convert integer to float first: `(int_val + 0.0) * float_val` |
| R-017 | Interpreter / modulo | `%` operator doesn't work with floats ("Unknown float operator: %") | Medium | Simulate modulo: `a - (b * (a / b))` or convert to integers first |
| R-018 | Interpreter / contract-call | Module-level helper functions that modify module-level state (`push`, `map[k]=v`) have their side-effects silently ignored when called FROM a contract method | **High** | Move state-modifying logic entirely to module-level functions and call them directly instead of through the contract |
| R-019 | Interpreter / push-in-func | `push()` / `map[k]=v` inside a module-level `action` called from init or action-chain sometimes silently fails | **High** | Use list/map literals for initialization instead of push/map-assignment in function calls |

## Working Feature Reference (v1.8.2 interpreter mode)

| Feature | Status | Notes |
|---------|--------|-------|
| `use "module" as alias` | ✅ Works | Built-in modules: crypto, datetime, math, json, blockchain |
| `use "file.zx" as alias` | ✅ Works | Relative to current file |
| `entity Name { field: default }` | ✅ Works | Field access works in interpreter |
| `Entity{ field: value }` construction | ✅ Works | In same file only |
| `contract Name { state {...} }` | ✅ Works | Must use `this` not `self` |
| `protocol Name { ... }` | ✅ Works | |
| `contract X implements Y` | ✅ Works | |
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

1. Use `this` instead of `self` everywhere in contracts
2. Always include `init()` with manual state initialization
3. After `Contract{}`, always call `.init()` 
4. Use `while` loops for indexed iteration and map iteration
5. Track map keys in a parallel list for iteration
6. Use `action protect` only inside contracts, not at module level
7. Remove phantom imports: `../database/postgres`, `../ai/zaie_engine`, `../middleware/security_middleware`
8. Remove `inject`, `middleware()`, `register_dependency()` (untested/unavailable)
9. Replace DB calls with in-memory maps
10. Use factory functions for cross-file entity usage
11. Prefer `--no-vm` execution until VM stabilizes
12. Declare all entities AFTER contracts (entity before contract breaks contract visibility — R-012)
13. Only ONE field in `state { }` — use module-level `let` vars for additional state (R-013)
14. Declare module-level `let` vars AFTER contract — contract can forward-reference them (R-014)
15. In contract methods: always `list.push()` BEFORE `map[key] = val` (R-015)
16. Convert integers to float before multiplying with floats: `(int_val + 0.0) * float_val` (R-016)
17. Avoid `%` with floats; simulate modulo or convert first (R-017)
18. State-modifying helpers called from contract methods have side-effects silently dropped — call them at module level instead (R-018)
19. Use list/map **literals** for bulk initialization instead of loops of `push()`/`map[k]=v` (R-019)
20. Preferred file layout order: `imports → constants → protocol → helper actions → contract (single state field) → module-level let vars → module-level helpers → entities → test → exports`

## Phase 0 Rewrite Progress

| File | Lines (original) | Status | Notes |
|------|-----------------|--------|-------|
| src/core/block.zx | ~440 | ✅ Done | Genesis, block creation, validation, merkle root |
| src/core/crypto.zx | ~370 | ✅ Done | Quantum-resistant crypto, keypairs, signing |
| src/core/consensus.zx | ~420 | ✅ Done | PoS consensus, validators, rewards, slashing |
| src/core/state.zx | ~600 | ✅ Done | Self-evolving state, parameters, snapshots, rollback |
| src/core/zvm.zx | ~1674 | ❌ Not started | Ziver Virtual Machine |
| src/core/social_capital.zx | ~1238 | ❌ Not started | Social capital scoring |
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
*Last updated: 2026-02-26*