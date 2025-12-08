# Zexus Language Philosophy & Design

## Core Principles

### 1. **Explicit Security by Default**

Zexus operates on the principle that security should be **opt-in to permissions**, not opt-out of restrictions. Every execution context begins in a restricted sandbox with minimal capabilities, requiring explicit grants to access system resources.

**Implementation:**
- **Capability-Based Security Model**: Fine-grained access control through explicit capability grants
- **Default Deny Policy**: Untrusted code starts with `DenyAllPolicy` - zero capabilities by default
- **Explicit Declarations**: Plugins declare required capabilities upfront for transparency
- **Audit Trail**: Every capability access is logged with requester, capability, and timestamp

**Trade-off:** Lower performance for untrusted code (validation overhead), but guaranteed isolation and auditability. Trusted code can use `AllowAllPolicy` for full performance.

```python
# Untrusted plugin - restricted by default
capability_manager.grant_capabilities(
    "untrusted_plugin", 
    ["core.language", "core.math"]  # Only essential capabilities
)

# Trusted plugin - can enable full capabilities
capability_manager.set_policy("trusted_plugin", AllowAllPolicy())
```

---

### 2. **Non-Invasive Extensibility**

Zexus plugins operate as **first-class coexistents**, not third-class citizens. The plugin system is designed to extend language capabilities without modifying core evaluator logic.

**Implementation:**
- **Hook System**: Plugins register hooks for specific evaluation events (action call, function definition, etc.)
- **Minimal Core**: Core evaluator remains clean; plugins transparently inject behavior
- **Return Value Preservation**: All hooks respect return values - hooks can intercept or pass-through
- **Composability**: Multiple plugins can register for the same hook - execution follows registration order

**Design Pattern:**
```python
# Plugin registers hook once during initialization
plugin_mgr.register_hook(
    "action.call",
    plugin_name="logging",
    callback=lambda ctx: log_action_call(ctx)
)

# Hook fires transparently during evaluation
action = evaluator.evaluate_action(ast_node)  # Automatically triggers hooks
```

**Philosophy:** The core language remains a clean, minimal specification. All advanced features (logging, profiling, validation, etc.) are plugins that could be removed without breaking the language.

---

### 3. **Type Safety Through Voluntary Adoption**

Zexus is **dynamically typed by default**, but provides a **comprehensive optional type system** for code that needs stronger guarantees.

**Implementation:**
- **Runtime Type Checking**: Types are validated at evaluation time, not compile time
- **Type Inference**: Automatic type detection from values and usage patterns
- **Gradual Typing**: Mix typed and untyped code freely
- **Function Signatures**: Optional parameter and return type validation

**Philosophy:**
```python
# Dynamic typing (always available)
x = 42  # Type inferred as INT
greet("Alice")  # Runtime type validation on call

# Optional static checking
def add(a: int, b: int) -> int:
    return a + b

# Type system validates during evaluation
add(1, 2)        # ✅ OK
add("1", "2")    # ❌ TypeError at evaluation time
```

---

### 4. **Performance Through Explicit Optimization**

Zexus assumes **correctness first, performance second**. Optimizations are applied selectively based on profiling and explicit hints.

**Implementation:**
- **Bytecode Compilation**: Explicit compilation phase with optimization passes
- **Constant Folding**: Fold constant expressions at compile time (5 + 3 → 8)
- **Dead Code Elimination**: Remove unreachable code after returns/breaks
- **Function Inlining**: Inline small hot functions based on usage patterns
- **Execution Profiling**: Track real execution to identify optimization targets

**Philosophy:**
```python
# Performance hints for hot code
@inline
def hot_loop():
    # This function is inlined to reduce call overhead
    pass

@optimize  
def critical_path():
    # Compiler applies aggressive optimization passes
    pass

# Profiling-guided optimization
optimizer.get_hot_functions()   # Functions called >10 times
optimizer.get_slow_functions()  # Functions taking >1.0s total
```

**Trade-off:** Additional compiler complexity, but straightforward mental model - optimize what's actually slow, not what might be.

---

### 5. **Modular Code Organization**

Code is organized through **modifiers and metadata**, not enforced access levels.

**Implementation:**
- **Semantic Modifiers**: PUBLIC, PRIVATE, SEALED, ASYNC, NATIVE, INLINE, SECURE, PURE
- **Metadata Preservation**: Modifiers are preserved in AST and accessible to tools
- **Convention Over Enforcement**: Tools and plugins respect modifiers through convention
- **Tool Integration**: IDE plugins, linters, and documentation tools use modifier metadata

**Philosophy:**
```python
@public
action greet(name: string) -> string:
    return "Hello, " + name

@private
action _internal_helper():
    # Private by convention - enforced through tools, not language
    pass

@sealed
action final_method():
    # Cannot be overridden in subclasses (enforced at definition time)
    pass
```

**Flexibility:** Allows language to remain simple while giving tools the information they need for proper IDE support, documentation, and enforcement.

---

## Architecture Principles

### 6. **Composable Sandboxing**

Security is provided through **layered, composable sandboxes** rather than all-or-nothing restrictions.

**Layers:**
1. **Capability System**: What operations are allowed
2. **Virtual Filesystem**: Which files can be accessed (with READ/WRITE/EXECUTE modes)
3. **Memory Quotas**: How much memory can be allocated
4. **Type System**: What operations are type-safe

**Example:**
```python
# Build a secure sandbox for untrusted code
sandbox = (SandboxBuilder()
    .with_capability_set("read_only")  # Only read operations
    .with_memory_quota(50 * 1024 * 1024)  # 50MB max
    .with_filesystem_mount("/data", "/", FileAccessMode.READ)
    .with_type_checking(strict=True)
    .build()
)

# Execute untrusted code safely
result = sandbox.execute(untrusted_plugin)
```

---

### 7. **Metaprogramming for Code Generation**

Zexus provides **compile-time code transformation** through the metaprogramming system, enabling domain-specific optimizations without language changes.

**Implementation:**
- **AST Manipulation**: Macros and transformers work directly on AST
- **Reflection API**: Runtime introspection of functions and objects
- **Code Generators**: Programmatic code generation from specifications
- **Custom Compiler Passes**: Plugins can register optimization passes

**Philosophy:**
```python
# Macro for compile-time transformation
@once
action singleton():
    # This action runs only once, subsequent calls return cached result
    pass

# Reflection for metaprogramming
sig = ReflectionAPI.get_signature(function)
members = ReflectionAPI.get_members(object)
source = ReflectionAPI.get_source(function)
```

---

## Design Trade-offs

### Explicit vs. Implicit

**Zexus chooses: EXPLICIT**

```python
# ❌ Implicit: What does @optimize do exactly?
@optimize
action foo() { }

# ✅ Explicit: Clear intention and scope
@optimize(constant_folding=true, inlining=true)
action foo() { }
```

**Rationale:** Developers should understand what code is doing. Documentation over magic.

---

### Strict vs. Permissive Security

**Zexus chooses: STRICT DEFAULT, PERMISSIVE ON DEMAND**

```python
# Untrusted: Denied by default
policy = DenyAllPolicy()  # No access to anything

# Trusted: Allowed by default
policy = AllowAllPolicy()  # Full access

# Selective: Only specified capabilities
policy = SelectivePolicy(allowed=["io.read", "core.math"])
```

**Rationale:** Better to start restricted and grant permissions than to discover security holes later.

---

### Performance vs. Safety

**Zexus chooses: SAFETY FIRST, OPTIMIZE LATER**

- All code types are checked (even "trusted" code)
- All capabilities are logged
- All filesystem access is sandboxed
- Optimizations are explicit and measurable

**Rationale:** Security and correctness bugs are expensive. Performance can be added incrementally.

---

## Extension Philosophy

### Plugins Are First-Class

Plugins in Zexus are **part of the language specification**, not an afterthought. Key principles:

1. **Declare Requirements**: Plugins explicitly declare capabilities and hooks they need
2. **Transparent Integration**: Hooks fire transparently - evaluator doesn't know plugins exist
3. **Composable**: Multiple plugins can work together without conflict
4. **Verifiable**: All plugin actions are auditable through capability logs

### Built-in Plugin Ecosystem

Zexus provides 5 built-in plugins:

1. **JSON Plugin**: Serialize/deserialize JSON data
2. **Logging Plugin**: Structured logging with levels (DEBUG, INFO, WARN, ERROR)
3. **Crypto Plugin**: Cryptographic operations (hashing, signing, key generation)
4. **Validation Plugin**: Schema validation and data sanitization
5. **Collections Plugin**: Extended data structures (Set, Map, etc.)

---

## Future Extension Points

These systems provide hooks for future phases:

### Phase 9: Advanced Type System
- Generic types: `Array<T>`, `Map<K, V>`
- Trait/interface system: Structural subtyping
- Union types: `int | string | null`

### Phase 10: Ecosystem Features
- Package management hooks
- Dependency resolution
- Plugin marketplace integration
- Performance profiling hooks

---

## Summary

Zexus is designed as a **secure, extensible, performant language** where:

- **Security** is the default (capability-based, sandboxed)
- **Extensibility** is non-invasive (plugin hooks, metaprogramming)
- **Performance** is measured (profiling-guided optimization)
- **Correctness** is prioritized (type system, capability validation)
- **Transparency** is enforced (explicit declarations, audit trails)

The philosophy enables **safe execution of untrusted code** while maintaining **clean language design** and **measurable performance improvements**.
