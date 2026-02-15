# Zexus Interpreter — Complete Deep-Dive Report

> Generated: February 14, 2026 | Version: 1.7.2 | Codebase: ~40,000+ lines of Python

---

## Table of Contents

- [Architecture Overview](#1-architecture-overview)
- [What Zexus CAN Do](#2-what-zexus-can-do)
- [What Zexus CAN'T Do (Design Limitations)](#3-what-zexus-cant-do-design-limitations)
- [What Can Be Implemented (Priority Recommendations)](#4-what-can-be-implemented-priority-recommendations)
- [Developer Experience & Access](#5-developer-experience--access)
- [Security Posture](#6-security-posture)
- [Performance Characteristics](#7-performance-characteristics)
- [Known Bugs / Code Quality Issues](#8-known-bugs--code-quality-issues)
- [Lexer & Token System](#9-lexer--token-system)
- [Parser & AST](#10-parser--ast)
- [Evaluator & Object Model](#11-evaluator--object-model)
- [VM Subsystem](#12-vm-subsystem)
- [Security Subsystem](#13-security-subsystem)
- [Blockchain Subsystem](#14-blockchain-subsystem)
- [Ecosystem Components](#15-ecosystem-components)
- [Implementation Status Tracker](#16-implementation-status-tracker)

---

## 1. Architecture Overview

| Layer | Tech | Lines |
|-------|------|-------|
| **Lexer** | Hand-written scanner, 113 keywords, context-aware | ~700 |
| **Parser** | Pratt (TDOP) + multi-strategy (structural/context/recovery) | ~10,000+ |
| **AST** | 62 statement nodes, 28 expression nodes | ~1,500 |
| **Evaluator** | Tree-walking interpreter with mixin dispatch | ~11,000+ |
| **VM** | Stack + Register + Parallel modes, 80+ opcodes | ~4,500 |
| **JIT** | 3-tier (interpreted → bytecode → native Python) | ~500 |
| **Security** | Mandatory sanitization, capabilities, policy-as-code | ~3,000+ |
| **Blockchain** | Ledger, contracts, TX, crypto, gas metering | ~1,500+ |
| **Stdlib** | 20 modules (math, http, crypto, fs, db, etc.) | ~3,000+ |

**Total codebase**: ~40,000+ lines of Python

### Entry Points

| Binary | Description |
|--------|-------------|
| `zx` / `zexus` | Main interpreter CLI (`python3 -m zexus`) |
| `zpm` | Package manager CLI |
| `zx-run` | Run scripts |
| `zx-dev` | Development mode |
| `zx-deploy` | Deployment tool |
| `zpics` | Testing framework |

### CLI Commands

| Command | Description |
|---------|-------------|
| `zx run <file>` | Execute a Zexus program |
| `zx -r "<code>"` | Run inline Zexus code (like `python -c`) |
| `zx check <file>` | Check syntax with detailed validation |
| `zx validate <file>` | Validate and auto-fix syntax errors |
| `zx profile <file>` | Profile performance with time and memory tracking |
| `zx ast <file>` | Display Abstract Syntax Tree |
| `zx tokens <file>` | Show tokenization output |
| `zx repl` | Start interactive REPL |
| `zx init [name]` | Initialize new Zexus project |
| `zx debug <on\|off\|minimal\|status>` | Control debug logging |

---

## 2. What Zexus CAN Do

### Core Language
- Variables (`let`, `const`) with lexical scoping and const immutability
- Functions (`action`, `function`, `lambda`) with closures
- Control flow: `if/elif/else`, `while`, `for each`, `match/case`, `break`, `continue`
- Error handling: `try/catch/throw`
- Data types: int, float, string, bool, null, list, map, entity instances, promises
- Operators: arithmetic, comparison, logical (`and`/`or`/`&&`/`||`), ternary `?:`, nullish `??`
- Dual syntax: `{}` (universal) and `:` (tolerable/Python-like)
- 100+ built-in functions

### OOP & Type System
- **Entities** with typed properties, methods, inheritance (`extends`)
- **Data classes** with generics (`data Box<T>`), pattern matching, auto-generated `toString`/`toJSON`/`clone`/`hash`
- **Contracts** (Solidity-inspired smart contracts with storage, constructors, TX context)
- **Enums** and **interfaces/protocols**
- **Runtime type checking** with structural typing, nullable types, custom validators
- **Advanced generics** with bounds, variance, and union types

### Security (Built-in, Not Optional)
- **Mandatory string sanitization** — strings passing through SQL/HTML/URL/shell contexts must be explicitly sanitized
- **Capability-based access control** — fine-grained permission tokens (`core.language`, `core.io`, etc.)
- **Policy-as-code** — `protect`, `verify`, `restrict`, `seal`, `audit`, `sandbox` keywords
- **Audit logging** with JSONL persistence and query support
- **Entity-level access control** — `public`/`private`/`sealed`/`secure` modifiers
- **Input validation** — `validate`, `sanitize` as first-class keywords
- **Crypto built-ins** — bcrypt password hashing, SHA-256/512, HMAC, constant-time comparison

### Blockchain
- **Smart contracts** — Solidity-like `contract` blocks with storage vars, constructors, require/revert
- **Immutable ledger** — SHA-256 chained entries with versioning
- **Transaction context** — `TX.caller`, `TX.timestamp`, `TX.gas_remaining`, gas metering
- **Crypto primitives** — ECDSA (secp256k1), RSA-2048, SHA3, BLAKE2, Keccak-256, Merkle trees
- **Proof of Work** — standard and quantum-resistant (SHA3-based) variants
- **VM opcodes** for blockchain: `HASH_BLOCK`, `VERIFY_SIGNATURE`, `STATE_READ/WRITE`, `GAS_CHARGE`

### Performance
- **VM execution** — 3 modes (stack, register, parallel) with 80+ opcodes
- **JIT compilation** — hot path detection (100 exec threshold), compiles to Python bytecode
- **Bytecode optimizer** — constant folding, dead code elimination, peephole optimizations
- **C-level extensions** — `fastops.c`, `cabi.c` compiled to `.so` for critical paths
- **Memory pool** and **register allocator** for VM-level optimizations
- **Gas metering** for resource-bounded execution

### Ecosystem & Tooling
- **ZPM package manager** — init, install, uninstall, publish, search (4 built-in packages)
- **REPL** — interactive mode with multi-mode execution switching
- **LSP server** — completions (60+ keywords, 80+ builtins), hover info, built on pygls
- **Profiler** — per-function time/memory profiling with `tracemalloc`
- **Syntax validator** — auto-detection, validation, auto-fix of syntax issues
- **ZPICS testing** — snapshot-based regression system (parse + runtime)
- **Error reporting** — Rust-quality error messages with line/column, source context, suggestions

### I/O & Networking
- **File system** — full CRUD with path traversal protection
- **HTTP client** — GET/POST/PUT/DELETE with connection pooling, async variants
- **HTTP server** — routing with GET/POST/PUT/DELETE handlers
- **Socket/TCP** — low-level server/client creation
- **Database drivers** — SQLite, PostgreSQL, MySQL, MongoDB

### Concurrency & Async
- **`async`/`await`** — Python-generator-based coroutines with promise integration
- **Channels** — Go-style buffered/unbuffered with blocking/non-blocking send/receive
- **Event loop** — cooperative task scheduling with priority and dependency DAG
- **Reactive state** — `watch` keyword for automatic variable change reactions
- **Parallel VM mode** — opcodes for task spawning, barriers, atomic operations

### Advanced Features
- **Metaprogramming** — AST manipulation, macro system, code generators
- **Dependency injection** — `inject`, `register_dependency`, `mock_dependency`
- **Middleware** — function wrapping for auth, throttling, caching
- **UI rendering** — terminal-based screens, components, themes, canvas with Bresenham line drawing
- **Module system** — `use`/`import` with path resolution, caching, `zpm_modules/` support
- **Embedded code** — `embedded { }` blocks for foreign language interop
- **Decorators** — `@logged`, `@cached`, etc. via `@` syntax
- **FIND/LOAD** — declarative project search and provider-aware config loading
- **`continue` error recovery** — graceful degradation mode

---

## 3. What Zexus CAN'T Do (Design Limitations)

| Limitation | Reason | Implementable? |
|-----------|--------|----------------|
| **No compound assignment** (`+=`, `-=`, `*=`) | Not in lexer/parser | **Yes** — add tokens + parse rules |
| **No `++`/`--` operators** | Not in lexer | **Yes** — add to lexer + parser |
| **No string interpolation** (`${}`) | Lexer only handles plain strings | **Yes** — extend `read_string()` |
| **No single-quoted strings** | Lexer only recognizes `"..."` | **Yes** — extend `read_string()` |
| **No block comments** (`/* */`) | Only `#` and `//` line comments | **Yes** — extend lexer |
| **No bitwise operators** (`&`, `|`, `^`, `~`) | Not in token set | **Yes** — add tokens/precedence |
| **No exponentiation** (`**`) | Not in token set | **Yes** — add token, use `POW` opcode (already in VM) |
| **No spread operator** (`...`) | Not in lexer/parser | **Yes** — moderate effort |
| **No `finally` clause** in try/catch | AST has no `finally` block | **Yes** — add to AST + evaluator |
| **No static type checking** | Type system is runtime-only | **Possible** — add a type-checking pass on AST before evaluation |
| **No true parallelism** in EventLoop | Single-threaded cooperative | **Design choice** — Python GIL limits this; parallel VM uses threads but has limitations |
| **No real remote package registry** | ZPM registry is stubbed | **Yes** — needs HTTP implementation |
| **No circular import detection** | Module manager caches but doesn't track cycles | **Yes** — add visited-set during import |
| **No go-to-definition in LSP** | Stub implementation | **Yes** — needs AST symbol table |
| **No debugger/step-through** | No debug protocol | **Possible** — significant effort, needs DAP implementation |
| **No class-based OOP with `new`** | Entities/data serve this role | **Design choice** — entities are the OOP model |
| **No multiline strings/heredocs** | Lexer limitation | **Yes** — extend string parsing |
| **No destructuring assignment** | `let {a, b} = map` not parsed | **Yes** — add AST node + parser support |
| **No `switch`/`when`** | Uses `match/case` instead | **Design choice** — `match` is more powerful |
| **No real GUI rendering** | ASCII terminal only | **Possible** — could add Tk/Qt/web backend |

---

## 4. What Can Be Implemented (Priority Recommendations)

### Quick Wins (Low Effort, High Value)
1. **Compound assignment** (`+=`, `-=`, `*=`, `/=`, `%=`) — add 5 tokens + infix parse rule + eval handler
2. **String interpolation** — `"Hello ${name}"` → extend lexer's `read_string()`
3. **Block comments** (`/* */`) — extend `skip_whitespace()` in lexer
4. **`finally` clause** — add to `TryCatchStatement` AST node + evaluator
5. **Multiline strings** — triple-quote `"""..."""` style
6. **Single-quoted strings** — extend lexer
7. **Exponentiation** `**` — token exists in VM (`POW`), just need lexer + parser

### Medium Effort, High Impact
8. **Destructuring** — `let {a, b} = map; let [x, y] = list`
9. **Circular import detection** — visited set in module manager
10. **LSP go-to-definition** — build symbol table from AST
11. **Remote ZPM registry** — real HTTP-based package publishing/download
12. **Static type checking pass** — optional analysis before runtime

### Significant Effort
13. **Debug Adapter Protocol** — step-through debugging
14. **GUI backend** — Tk or web-based renderer
15. **True concurrent EventLoop** — asyncio integration
16. **WASM compilation target** — expose Zexus to browsers

---

## 5. Developer Experience & Access

| Feature | Status | Notes |
|---------|--------|-------|
| **CLI tools** | 7 commands (`zx`, `zexus`, `zpm`, `zx-run`, `zx-dev`, `zx-deploy`, `zpics`) | |
| **npm install** | Works (`npm i zexus`) | Requires Python 3.8+ as peer dep |
| **pip install** | Works (`pip install .`) | setup.py configured |
| **REPL** | Functional | `zx repl` with mode switching |
| **LSP** | Partial | Completions work, hover/def limited |
| **VS Code extension** | Present in workspace | Syntax highlighting + TextMate grammar |
| **Error messages** | Excellent | Rust-quality with suggestions |
| **Documentation** | 2680-line README | Comprehensive keyword reference |
| **Testing** | ZPICS snapshots + pytest | No built-in assertion DSL for users |
| **Profiling** | `zx profile` | Function-level time + memory |
| **Inline execution** | `zx -r "<code>"` | Like `python -c` |

---

## 6. Security Posture

### Strengths
- Mandatory string sanitization (can't accidentally inject SQL/XSS)
- Capability-based access control (deny-by-default possible)
- Path traversal protection on file operations
- Resource limiter (call depth, iteration, timeout)
- Gas metering for bounded execution
- Audit logging with persistence

### Risks
- Context detection is heuristic (false positives/negatives)
- Capability enforcement is runtime-only (no compile-time guarantee)
- `eval_file()` built-in can execute arbitrary code
- No formal security verification
- SQLite persistence has no encryption at rest
- `continue_on_error` mode could mask security failures

---

## 7. Performance Characteristics

### Fast Paths
- VM bytecode execution (stack mode) for arithmetic, loops, comparisons
- C extensions (`fastops.so`, `cabi.so`) for critical operations
- JIT for hot loops (100+ iterations triggers compilation)
- Constant deduplication in bytecode
- Module caching (import once)
- Connection pooling in HTTP client

### Slow Paths
- Tree-walking interpreter fallback for complex constructs (entities, contracts, for-each)
- Many AST node types unsupported by bytecode compiler
- JIT only handles ~20 basic opcodes
- Python GIL limits true parallelism
- No lazy evaluation
- No tail call optimization in interpreter (only claimed in compiler)

---

## 8. Known Bugs / Code Quality Issues

- Duplicate token definitions in `zexus_token.py` (harmless but sloppy)
- Duplicate `__repr__` in `WatchStatement` AST node
- Duplicate `mkdir` function in `src/zexus/stdlib/fs.py` (one incorrectly calls `os.remove`)
- `ecosystem.py` and `zpm/` duplicate package management concepts
- Context parser is 8,438 lines — organically grown, potentially fragile
- Lambda lookahead bounded to 300 chars — fails for very long parameter lists
- `satisfies_bounds()` in advanced types always returns `True` (stub)

---

## 9. Lexer & Token System

### 113 Keywords

| Category | Keywords |
|----------|----------|
| **Variables & Declarations** | `let`, `const`, `data` |
| **Control Flow** | `if`, `then`, `elif`, `else`, `while`, `for`, `each`, `in`, `return`, `break`, `continue`, `match`, `case`, `default`, `pattern` |
| **Functions** | `action`, `function`, `lambda` |
| **Error Handling** | `try`, `catch`, `throw` |
| **I/O & Debug** | `print`, `debug`, `log` |
| **Modules & Imports** | `use`, `find`, `load`, `exactly`, `embedded`, `export`, `import`, `external`, `as`, `module`, `package`, `using`, `inject` |
| **Literals** | `true`, `false`, `null` |
| **UI/Graphics** | `screen`, `component`, `theme`, `color`, `canvas`, `graphics`, `animation`, `clock` |
| **Async & Concurrency** | `async`, `await`, `channel`, `send`, `receive`, `atomic` |
| **Events & Reactive** | `event`, `emit`, `stream`, `watch` |
| **OOP & Types** | `enum`, `protocol`, `interface`, `type_alias`, `implements`, `this` |
| **Security** | `entity`, `verify`, `contract`, `protect`, `seal`, `audit`, `restrict`, `sandbox`, `trail`, `capability`, `grant`, `revoke`, `validate`, `sanitize` |
| **Access Modifiers** | `public`, `private`, `sealed`, `secure`, `pure`, `view`, `payable`, `modifier`, `native`, `inline` |
| **Blockchain** | `ledger`, `state`, `revert`, `limit`, `persistent`, `storage`, `require` |
| **Backend** | `middleware`, `auth`, `throttle`, `cache` |
| **Performance** | `gc`, `buffer`, `simd`, `defer` |
| **Logical** | `and` (→ `&&`), `or` (→ `||`) |

### Operators Supported

| Type | Operators |
|------|-----------|
| Arithmetic | `+`, `-`, `*`, `/`, `%` |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| Logical | `&&` (also `and`), `\|\|` (also `or`), `!` |
| Assignment | `=` |
| Ternary | `? :` |
| Nullish coalescing | `??` |
| Stream/IO | `>>` (append), `<<` (import) |
| Member access | `.` |
| Decorator | `@` |

### NOT Supported (Pre-Implementation)
`+=`, `-=`, `*=`, `++`, `--`, bitwise (`&`, `|`, `^`, `~`), exponentiation (`**`), spread (`...`)

### Special Features
- Arrow functions: `=>` lexed as `LAMBDA`
- Two comment styles: `#` and `//` (no block comments yet)
- Embedded code blocks: `embedded { ... }` disables keyword resolution
- Contextual keywords: ~80 keywords can be used as identifiers when preceded by `LET`, `CONST`, `DOT`, etc.
- Line/column tracking on every token for error reporting

---

## 10. Parser & AST

### Parsing Model: Pratt Parser (Top-Down Operator Precedence)

- **11 precedence levels** from LOWEST to CALL
- **Prefix parse functions** for tokens that start expressions
- **Infix parse functions** for binary/postfix operations
- **Statement dispatch table** — O(1) keyword-to-parser lookup for ~70 statement types

### Multi-Strategy Parsing

1. **StructuralAnalyzer** — Pre-pass tokenizing stream into top-level blocks
2. **ContextStackParser** (8,438 lines) — Maintains context stack with ~30 context-type handlers
3. **ErrorRecoveryEngine** — Cooperates with above for graceful error recovery

### Statement Nodes (62 types)

| Category | AST Nodes |
|----------|-----------|
| **Variables** | `LetStatement`, `ConstStatement`, `ImmutableStatement` |
| **Data/Types** | `DataStatement`, `EnumStatement`, `EntityStatement`, `InterfaceStatement`, `TypeAliasStatement`, `ProtocolStatement` |
| **Control Flow** | `IfStatement`, `WhileStatement`, `ForEachStatement`, `ReturnStatement`, `ContinueStatement`, `BreakStatement`, `ThrowStatement`, `TryCatchStatement`, `PatternStatement` |
| **Functions** | `ActionStatement`, `FunctionStatement`, `ExactlyStatement`, `PureFunctionStatement`, `ModifierDeclaration` |
| **Modules** | `UseStatement`, `FromStatement`, `ExportStatement`, `ModuleStatement`, `PackageStatement`, `UsingStatement` |
| **I/O & Debug** | `PrintStatement`, `DebugStatement`, `LogStatement`, `ImportLogStatement`, `TrailStatement` |
| **Security** | `VerifyStatement`, `ProtectStatement`, `SealStatement`, `AuditStatement`, `RestrictStatement`, `SandboxStatement`, `CapabilityStatement`, `GrantStatement`, `RevokeStatement`, `ValidateStatement`, `SanitizeStatement`, `InjectStatement` |
| **Concurrency** | `ChannelStatement`, `SendStatement`, `ReceiveStatement`, `AtomicStatement`, `StreamStatement`, `WatchStatement`, `EmitStatement` |
| **Blockchain** | `ContractStatement`, `LedgerStatement`, `StateStatement`, `RequireStatement`, `RevertStatement`, `LimitStatement`, `TxStatement`, `PersistentStatement` |
| **Performance** | `NativeStatement`, `GCStatement`, `InlineStatement`, `BufferStatement`, `SIMDStatement`, `DeferStatement` |
| **UI/Graphics** | `ScreenStatement`, `ColorStatement`, `CanvasStatement`, `GraphicsStatement`, `AnimationStatement`, `ClockStatement`, `ComponentStatement`, `ThemeStatement` |
| **Other** | `ExpressionStatement`, `BlockStatement`, `ExternalDeclaration`, `EmbeddedCodeStatement`, `MiddlewareStatement`, `AuthStatement`, `ThrottleStatement`, `CacheStatement`, `VisibilityModifier` |

### Expression Nodes (28 types)

| Category | AST Nodes |
|----------|-----------|
| **Literals** | `IntegerLiteral`, `FloatLiteral`, `StringLiteral`, `Boolean`, `NullLiteral`, `ListLiteral`, `MapLiteral`, `EmbeddedLiteral` |
| **Identifiers** | `Identifier`, `ThisExpression` |
| **Operations** | `PrefixExpression`, `InfixExpression`, `TernaryExpression`, `NullishExpression`, `AssignmentExpression` |
| **Calls/Access** | `CallExpression`, `MethodCallExpression`, `PropertyAccessExpression`, `SliceExpression` |
| **Functions** | `ActionLiteral`, `LambdaExpression`, `AsyncExpression`, `AwaitExpression` |
| **Pattern Matching** | `MatchExpression`, `ConstructorPattern`, `VariablePattern`, `WildcardPattern`, `LiteralPattern` |
| **Special** | `FindExpression`, `LoadExpression`, `FileImportExpression`, `IfExpression` |
| **Blockchain** | `TXExpression`, `HashExpression`, `SignatureExpression`, `VerifySignatureExpression`, `GasExpression` |

---

## 11. Evaluator & Object Model

### Data Types

| Type | Class | Notes |
|------|-------|-------|
| **Integer** | `Integer(value)` | Arbitrary precision, overflow protection at 4096 bits |
| **Float** | `Float(value)` | Python float |
| **Boolean** | `Boolean(value)` | Singletons `TRUE`/`FALSE` |
| **Null** | `Null()` | Singleton `NULL` |
| **String** | `String(value, sanitized_for, is_trusted)` | Tracks trust/sanitization status for security |
| **List** | `List(elements)` | Mutable array with `get`/`set`/`append`/`extend` |
| **Map** | `Map(pairs)` | Dict-like, sealed-key protection |
| **Action** | `Action(parameters, body, env)` | User-defined function with closure |
| **LambdaFunction** | `LambdaFunction(parameters, body, env)` | Anonymous function |
| **Builtin** | `Builtin(fn, name)` | Native Python wrapper |
| **ReturnValue** | `ReturnValue(value)` | Propagates `return` through call stack |
| **Promise** | `Promise(executor)` | PENDING/FULFILLED/REJECTED with `then`/`catch`/`finally` |
| **Coroutine** | `Coroutine(generator, action)` | Python generator wrapper for async |
| **EntityDefinition** | `EntityDefinition(name, properties, parent)` | Struct/class template with inheritance |
| **EntityInstance** | `EntityInstance(entity_def, values)` | Instance of entity |
| **DateTime** | `DateTime(timestamp)` | Unix timestamp wrapper |
| **EmbeddedCode** | `EmbeddedCode(name, language, code)` | Foreign-language code blocks |
| **ContractReference** | `ContractReference(address)` | Lazy reference to deployed contract |

### Architecture: Tree-Walk Interpreter with VM Hybrid

Mixin-based composition:
```
Evaluator(ExpressionEvaluatorMixin, StatementEvaluatorMixin, FunctionEvaluatorMixin)
```

- Precomputed dispatch table mapping 35+ AST node types to handlers
- Optional VM bytecode compilation for hot paths with automatic fallback
- File-level bytecode caching

### Built-in Functions (100+)

| Category | Functions |
|----------|-----------|
| **Date/Time** | `now`, `timestamp`, `time` |
| **Math** | `random`, `sqrt`, `to_hex`, `from_hex` |
| **Type Conversion** | `string`, `int`, `float` |
| **String** | `uppercase`, `lowercase`, `split`, `len` |
| **List/Collection** | `len`, `first`, `rest`, `push`, `append`, `extend`, `sort`, `slice`, `reduce`, `map`, `filter` |
| **I/O** | `input`, `print`, `file_read_text`, `file_write_text`, `file_exists`, `file_read_json`, `file_write_json` |
| **File System** | `fs_is_file`, `fs_is_dir`, `fs_mkdir`, `fs_remove`, `fs_rename`, `fs_copy` |
| **Networking** | `http_get`, `http_post`, `http_put`, `http_delete`, `http_server`, `socket_create_server` |
| **Database** | `sqlite_connect`, `postgres_connect`, `mysql_connect`, `mongo_connect` |
| **Security/Crypto** | `hash_password`, `verify_password`, `crypto_random`, `constant_time_compare` |
| **Persistence** | `persist_set`, `persist_get` |

### Environments & Scoping

- Lexical scoping via linked-list chain (each `Environment` has an `outer` pointer)
- `const` immutability enforced via `const_vars` set
- Closures capture reference to defining environment
- Exports dict for module `export` statements
- Reactive watchers for `watch` feature
- Persistent storage and memory tracking

---

## 12. VM Subsystem

### Three Execution Modes

| Mode | Description |
|------|-------------|
| **STACK** | Default stack-based execution |
| **REGISTER** | Opcodes 200–241 operate on numbered registers |
| **PARALLEL** | Multi-core with opcodes 300–309 |
| **AUTO** | Runtime selects best mode |

### Opcode Categories (80+)

| Range | Category | Examples |
|-------|----------|----------|
| 1–6 | Stack | `LOAD_CONST`, `LOAD_NAME`, `STORE_NAME`, `POP`, `DUP` |
| 10–16 | Arithmetic | `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `POW`, `NEG` |
| 20–25 | Comparison | `EQ`, `NEQ`, `LT`, `GT`, `LTE`, `GTE` |
| 30–32 | Logic | `AND`, `OR`, `NOT` |
| 40–43 | Control Flow | `JUMP`, `JUMP_IF_FALSE`, `JUMP_IF_TRUE`, `RETURN` |
| 50–54 | Calls | `CALL_NAME`, `CALL_FUNC_CONST`, `CALL_TOP`, `CALL_BUILTIN`, `CALL_METHOD` |
| 60–65 | Collections | `BUILD_LIST`, `BUILD_MAP`, `INDEX`, `SLICE`, `GET_ATTR` |
| 70–72 | Async | `SPAWN`, `AWAIT`, `SPAWN_CALL` |
| 80–81 | Events | `REGISTER_EVENT`, `EMIT_EVENT` |
| 90–91 | Modules | `IMPORT`, `EXPORT` |
| 110–119 | Blockchain | `HASH_BLOCK`, `VERIFY_SIGNATURE`, `STATE_READ/WRITE`, `GAS_CHARGE` |
| 130–137 | Security | `REQUIRE`, `DEFINE_CONTRACT`, `DEFINE_ENTITY`, `AUDIT_LOG` |
| 140–142 | Exception | `SETUP_TRY`, `POP_TRY`, `THROW` |
| 200–241 | Register | Full ALU mirrored to register form |
| 250–252 | I/O | `PRINT`, `READ`, `WRITE` |
| 300–309 | Parallel | `PARALLEL_START`, `BARRIER`, `SPAWN_TASK`, `ATOMIC_ADD` |

### JIT: Three-Tier Compilation

| Tier | Name | Description |
|------|------|-------------|
| 0 | INTERPRETED | Direct AST walking |
| 1 | BYTECODE | Stack-based VM execution |
| 2 | JIT_NATIVE | JIT-compiled Python code for hot paths |

Hot path threshold: 100 executions. JIT handles ~20 basic opcodes only.

---

## 13. Security Subsystem

### Security Enforcement
- **Mandatory sanitization** for SQL, HTML, URL, and shell contexts
- **Trusted strings** — literal strings marked `is_trusted`
- **Concatenation checks** — both operands verified in sensitive contexts
- **Heuristic-based context detection** (regex pattern matching)

### Capability System
- **4 levels**: DENY, RESTRICTED, ALLOWED, UNRESTRICTED
- **Policies**: AllowAll (dev), DenyAll (sandbox), Selective (production)
- **Base capabilities** always available: `core.language`, `core.control`, `core.math`, `core.strings`
- **Audit log** records every capability request

### Policy Engine
- **Rule types**: VerifyRule (boolean), RestrictRule (field-level)
- **Enforcement levels**: strict (fail-fast), warn, audit, permissive
- **Middleware chain** wraps protected functions

---

## 14. Blockchain Subsystem

### Ledger
- Immutable, versioned state storage with SHA-256 chaining
- Multi-ledger management via `LedgerManager`
- Permanent lock support

### Transaction Context
- `TX.caller`, `TX.timestamp`, `TX.block_hash`
- Gas metering with configurable limit (default: 1M)
- Revert with reason tracking

### Crypto
- Hashing: SHA-256/512, SHA3, BLAKE2, Keccak-256
- Key generation: ECDSA (secp256k1) and RSA-2048
- Signing/verification (requires `cryptography` library)

### Limitation
This is a **local simulation** — no networking, no consensus, no peer discovery.

---

## 15. Ecosystem Components

### ZPM Package Manager
- 4 built-in packages: `std`, `crypto`, `web`, `blockchain`
- Remote registry **stubbed** (not yet implemented)
- No transitive dependency resolution
- No integrity verification

### Standard Library (20 modules)
- **Math**: 50+ functions (trig, log, factorial, gcd, lcm, random)
- **HTTP**: Connection pooling, thread-pool async
- **Crypto**: 10+ hash algorithms, HMAC, PBKDF2, secure random
- **Blockchain**: Quantum-resistant hashing, PoW, address generation
- **File System**: Full CRUD with path traversal protection
- **Databases**: SQLite, PostgreSQL, MySQL, MongoDB drivers

### LSP Server
- Completions: 60+ keywords, 80+ built-in signatures
- Hover: ~10 entries (minimal)
- Go-to-definition: **not implemented**
- Document symbols: **not implemented**

### Profiler
- Per-function time + memory profiling
- Uses `tracemalloc`
- No flame graphs, no sampling profiler

### Concurrency System
- Go-style channels (buffered/unbuffered)
- Blocking/non-blocking send/receive with timeout
- No select/multi-channel wait

### Type System
- Runtime-only checking
- Structural typing for objects
- Generic array types, nullable types
- No union types, no type inference, no static checking

---

## 16. Implementation Status Tracker

### Quick Wins
| # | Feature | Status |
|---|---------|--------|
| 1 | Compound assignment (`+=`, `-=`, `*=`, `/=`, `%=`) | ⬜ Not started |
| 2 | String interpolation (`"Hello ${name}"`) | ⬜ Not started |
| 3 | Block comments (`/* */`) | ⬜ Not started |
| 4 | `finally` clause in try/catch | ⬜ Not started |
| 5 | Multiline strings (`"""..."""`) | ⬜ Not started |
| 6 | Single-quoted strings | ⬜ Not started |
| 7 | Exponentiation `**` operator | ⬜ Not started |

### Medium Effort
| # | Feature | Status |
|---|---------|--------|
| 8 | Destructuring assignment | ⬜ Not started |
| 9 | Circular import detection | ⬜ Not started |
| 10 | LSP go-to-definition | ⬜ Not started |
| 11 | Remote ZPM registry | ⬜ Not started |
| 12 | Static type checking pass | ⬜ Not started |

### Significant Effort
| # | Feature | Status |
|---|---------|--------|
| 13 | Debug Adapter Protocol | ⬜ Not started |
| 14 | GUI backend | ⬜ Not started |
| 15 | True concurrent EventLoop | ⬜ Not started |
| 16 | WASM compilation target | ⬜ Not started |

---

*This document is maintained as the source of truth for Zexus capabilities, limitations, and implementation roadmap.*
