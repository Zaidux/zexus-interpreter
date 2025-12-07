# Zexus Strategic Upgrades — Blueprint

Purpose: refine depth over quantity. This blueprint turns your 10 strategic upgrades into a concrete implementation plan with APIs, AST changes, evaluator expectations and test ideas.

**Summary of Goals**
- Give keywords multiple semantic modes via contextual parsing and modifiers.
- Add a concise modifier system (`secure`, `async`, `inline`, `public`, `private`, `sealed`, `native`, `pure`).
- Introduce a plugin system for capability extension.
- Implement capability-based sandbox enforcement.
- Add a virtual filesystem and virtual memory layer for safe isolation.
- Introduce an internal unified type system (infrastructure and initial inference).
- Provide metaprogramming hooks (macros and AST transforms).
- Implement internal optimization passes: constant folding, dead-code elimination, bytecode, inline expansion.
- Document the language philosophy and integration guidelines.

---

**1) Multiple semantic modes for keywords**

Design:
- Do not add tokens. Allow statement forms to accept contextual mode.
- Parser rule approach: when parsing a statement starting with a keyword (e.g. `map`), inspect the next token(s) and choose an interpretation.

Examples & grammar sketches:
- `map data { key: value, ... }` -> Map literal / declaration
- `map (x -> x*2) over items` -> Functional mapping (map operator)
- `map screen { ... }` -> UI mapping (domain-specific) — resolved via plugin or current context
- `map route "/home" { ... }` -> Router mapping (plugin)

API decisions:
- `map` statement will produce different AST node subtypes: `MapLiteral`, `MapFunction`, `MapUI`, `MapRoute`.
- Add a small helper function `resolve_map_mode(peek_tokens, env)` in parser to decide which AST to build.

Tests:
- Validate `map` parses into correct AST given varied inputs.
- Evaluate mappings in interpreter with differing semantics (pure map returns new collection, route map registers a route handler in router plugin).

---

**2) Modifier system (single syntax for many behaviors)**

Design:
- Introduce a `Modifier` AST element list attached to declarations and actions.
- Grammar: `modifier := 'public' | 'private' | 'sealed' | 'async' | 'native' | 'inline' | 'secure' | 'pure'`
- Declaration syntax: `<mod>* <declaration>` where `<mod>*` is zero or more modifiers.

Example:
- `secure async action fetch_user { ... }`
- `public inline function add(a,b) { ... }`

AST changes:
- Add `Modifiers` attribute to `FunctionDeclaration`, `ActionDeclaration`, `EnumDeclaration`, `StreamDeclaration`, etc.
- Represent modifiers as set of strings in AST nodes: `node.modifiers = ['secure','async']`.

Parser changes:
- Add `parse_modifiers()` that consumes modifier tokens before parsing a declaration.
- Reuse same function in all parse_*_declaration() entry points.

Evaluator:
- Evaluator checks `node.modifiers` and adjusts behavior: e.g. `native` loads external library; `inline` marks function for later inline pass; `secure` triggers capability checks at call-time (or when binding to environment).

Tests:
- Ensure modifiers accepted in various orders.
- Ensure unknown modifier raises parse error.

---

**3) Plugin system**

Design goals:
- Small runtime API for plugins and a loader.
- Plugins register capabilities and extension points (functions, types, AST transformers, runtime hooks).
- Plugin files: `plugins/<name>.py` or dynamic Python package loaded by plugin manager.

API sketch (pseudo):
```
# zexus-plugin API
class Plugin:
    name: str
    def register(env, host_api):
        # host_api exposes runtime hooks: register_function, register_type, register_ast_macro, allow_capabilities
```

Loader API:
- `use plugin "network"` -> loads `plugins/network.py` and executes `register(env, host_api)`.
- Plugins declare required capabilities (e.g. `network` requires `network.connect` permission).

Example plugin usage in language:
- `use plugin "crypto"`
- call `crypto.hash("sha256", data)`

Security:
- Plugin registration can declare capabilities which then can be gated by sandbox policies.

Tests:
- Plugin loads and registers functions
- Plugin functions callable from Zexus code
- Plugin registration denied if not allowed in sandbox

---

**4) Capability-based security model**

Design:
- Abilities modeled as hierarchical capability strings: `filesystem.read`, `filesystem.write`, `network.connect`, `native.ffi`, `plugin.install`.
- Sandbox block API already exists. Expand to accept explicit allow/deny rules.

Runtime enforcement:
- Environment (`env`) gets a `capabilities` object detailing allowed capabilities.
- On operation, evaluator asks `env.capabilities.check('filesystem.write')` — throws `CapabilityError` if denied.

Policy evaluation process:
- Parse `sandbox with { allow X; deny Y; } { code }` into a `CapabilityPolicy` that composes policies with scoping.
- On entering sandbox block, create a child environment with computed capabilities.

Implementation notes:
- Implement `CapabilitySet` (bitset/Trie) for fast check.
- Provide admin function to audit current capability checks for debugging.

Tests:
- Attempt FS write inside restricted sandbox -> fail
- Allowed operations succeed

---

**5) Virtual filesystem & virtual memory layer**

Virtual FS (VFS):
- Provide `vfs` object in environment: `env.vfs` exposing `read`, `write`, `list`, `stat`, `delete`.
- Backed by in-memory store plus optional persister (file-backed store stored under `.zexus/vfs/` within repo) if allowed.

Virtual Memory:
- Buffer allocations done in VMM: `env.vmm.allocate(size)` returns handle or a `Buffer` object that operations reference.
- Ensure `BUFFER`, `NATIVE` and `SIMD` operate only on VMM buffers unless capability `native.ffi` granted.

Security:
- VFS operations require capabilities: `filesystem.read`, `filesystem.write` toggled by sandbox.
- By default, ephemeral in-memory store used to avoid host writes.

Tests:
- VFS read/write work with persisted and ephemeral modes.
- Native buffer cannot access host memory without `native.ffi` capability.

---

**6) Unified type system (initial phase)**

Scope for first iteration:
- Implement runtime `Type` objects and a `TypeChecker`/`TypeInferencer` with:
  - Primitive types: `Int`, `Float`, `String`, `Bool`, `Null`
  - Collection types: `List<T>`, `Map<K,V>`
  - Generic type placeholders and simple inference for `let` assignments
  - Union types for pattern matching: `Int | String`

API:
- `env.types.infer(expr)` returns a `Type` object
- `Type` objects implement `.is_assignable_from(other)`

Parser/AST:
- Allow optional type annotations: `let x: list<int> = [1,2,3]`

Benefits:
- Enables optimizations (monomorphic calls), better error messages, and plugin type integration.

Tests:
- Type inference for common expressions
- Type errors detected at parse/eager-check stage where possible

---

**7) Metaprogramming hooks**

Two-pronged approach:
1. Macros (compile-time)
   - `macro` keyword (or via plugin) to register macros
   - Macro expands into AST prior to evaluation
2. AST transforms
   - Plugins or user code can register AST transforms at parse/compile time

Safety:
- Macros only run at compile-phase, inside current capability policy (disallow macros that use `native.ffi` unless capability provided at compile time).

API examples:
```
macro repeat(n, code) {
  // returns AST that repeats `code` n times
}

// invocation
repeat(3) { print("hi") }
```

Implementation notes:
- Add `MacroRegistry` in parser pipeline
- Macro expansion happens in a separate `expand_macros(ast, env)` pass

Tests:
- Macro expansions produce expected AST
- Macro cannot escalate privileges

---

**8) Internal optimizations**

Pass pipeline (simple):
1. AST normalization
2. Constant folding
3. Dead code elimination
4. Inline expansion (guided by `inline` modifier)
5. Bytecode generation (optional second phase)

Implementation approach:
- Pass manager with ordered passes
- Each pass receives and returns AST; passes can annotate nodes for later passes
- Bytecode generator outputs a bytecode representation and a VM that executes bytecode. Start with interpreter fallback.

Tests:
- Ensure constant folding turns `3*4` into `12`
- DCE removes unreachable branches
- Inline expands small functions that are marked `inline` and referenced once

---

**9) Document the philosophy behind Zexus**

Create `ENHANCEMENT_PACKAGE/ZEXUS_PHILOSOPHY.md` containing:
- Language goals (embedded, safe, sandbox-first, UI-friendly, scripting)
- Token and modifier design philosophy
- Security-first rationale
- Plugin and capability model
- Migration guidelines for existing Zexus code

---

**10) Process, tests, CI, and rollout**

- Implement features incrementally, starting with: modifier system -> plugin loader -> capability model -> VFS/VMM -> types -> macros -> optimizations.
- Create unit tests and integration tests for each step and add to `pytest`.
- Add optional `make check`/`dev` scripts to run the new checks.

---

## Immediate next actions (I can start now)
1. Implement modifier parsing + AST integration and add a small test harness. This unlocks the rest.
2. Add `Modifier` representation to core AST file: `zexus_ast.py`.
3. Add `parse_modifiers()` helper to `parser.py` and update `parse_function` / `parse_action` / `parse_enum` entry points.

If you give me the go-ahead, I'll start by adding the `STRATEGIC_UPGRADES.md` (this file), writing the tracked todo list (done), and implementing the modifier parser + AST stub and a unit test that validates modifiers parse correctly.

If you prefer a different order, tell me which item to prioritize.
