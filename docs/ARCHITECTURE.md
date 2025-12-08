# Zexus Language Architecture - Strategic Enhancement Overview

## Executive Summary

Zexus has been enhanced with two foundational systems that enable safe, extensible growth:

1. **Modifier System** - Semantic tagging for code
2. **Plugin System** - Non-invasive extensibility framework

These systems work together to support all future upgrades (capability-based security, virtual filesystem, type system, metaprogramming, optimizations).

---

## Architecture Layers

```
┌─────────────────────────────────────────┐
│         User Applications               │
│    (Use plugins, modifiers, features)   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       Plugin System (Phase 2)            │
│  PluginManager, Hooks, Capabilities     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│    Modifier System (Phase 1) ✓           │
│  Semantic tagging, evaluator flags      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│     Evaluator (Core Language)            │
│  Statements, Expressions, Functions     │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│     Parser + Lexer (Core Language)       │
│  Token Recognition, AST Construction    │
└─────────────────────────────────────────┘
```

---

## Phase 1: Modifier System Architecture

### Token Layer
```
Lexer Input: "public async action foo() { ... }"
              ↓
Lexer Output: [PUBLIC, ASYNC, ACTION, IDENT("foo"), LPAREN, ...]
              ↓
Parser Recognition: ModifierToken types identified
```

**Token Constants:**
- `PUBLIC`, `PRIVATE`, `SEALED` - Visibility modifiers
- `ASYNC`, `INLINE`, `NATIVE` - Execution hints
- `SECURE`, `PURE` - Capability/side-effect markers

### Parser Layer
```
parse_statement() {
  modifiers = _parse_modifiers()     # Collect: [PUBLIC, ASYNC]
  node = parse_action_declaration()  # Parse: ActionStatement
  attach_modifiers(node, modifiers)  # Attach: node.modifiers = ["public", "async"]
  return node
}
```

**Key Methods:**
- `_parse_modifiers()` - Consume modifier tokens until non-modifier
- `attach_modifiers(node, mods)` - Non-invasive attribute attachment
- All statement parsers use this pattern

### AST Layer
```python
# Before modification
class ActionStatement:
    def __init__(self, name, parameters, body):
        self.name = name
        self.parameters = parameters
        self.body = body

# After modifier attachment
action_node.modifiers = ["public", "async", "inline"]
```

**Design:** Modifiers attached post-construction via `setattr()` for non-invasiveness

### Evaluator Layer
```python
def eval_action_statement(self, node, env, stack_trace):
    action = Action(node.parameters, node.body, env)
    
    modifiers = getattr(node, 'modifiers', [])
    
    # Set flags based on modifiers
    if 'inline' in modifiers:
        action.is_inlined = True
    if 'async' in modifiers:
        action.is_async = True
    if 'secure' in modifiers:
        action.is_secure = True
    if 'pure' in modifiers:
        action.is_pure = True
    if 'native' in modifiers:
        action.is_native = True
    
    # Public actions auto-export
    if 'public' in modifiers:
        env.export(node.name.value, action)
    
    env.set(node.name.value, action)
    return NULL
```

**Key Behavior:**
- Modifiers enable Action properties for optimization/security
- `public` triggers export mechanism
- Modifiers checked at runtime for semantics

---

## Phase 2: Plugin System Architecture

### Core Components

#### 1. Plugin Metadata (`PluginMetadata` dataclass)
```python
@dataclass
class PluginMetadata:
    name: str                           # e.g., "json"
    version: str                        # e.g., "1.0.0"
    author: str                         # Author name
    description: str                    # What plugin does
    requires: List[str]                 # ["io.read"]
    provides: List[str]                 # ["json.parse", "json.stringify"]
    hooks: List[str]                    # ["type_validator"]
    config: Dict[str, Any]              # Configuration schema
```

#### 2. Plugin Manager (`PluginManager` class)
```python
class PluginManager:
    def load_plugin(path, config) → PluginMetadata
    def register_hook(name, handler, plugin_name, priority)
    def call_hooks(name, *args, **kwargs) → Any
    def check_capability(capability) → bool
    def grant_capability(capability)
    def get_capabilities() → [str]
    def get_loaded_plugins() → [str]
```

**Responsibilities:**
- Plugin lifecycle management (load, init, cleanup)
- Hook registration and execution
- Capability tracking
- Configuration injection

#### 3. Hook System
```
Hook Registration:
  Plugin A registers handler for "pre_eval" hook
              ↓
  PluginManager stores: hooks["pre_eval"] = [
    Hook(name="pre_eval", handler=<func>, plugin_name="A", priority=10),
    Hook(name="pre_eval", handler=<func>, plugin_name="B", priority=5)
  ]

Hook Execution:
  call_hooks("pre_eval", node, context)
              ↓
  Execute high priority first (Plugin A)
              ↓
  Execute lower priority (Plugin B)
              ↓
  Return transformed result
```

**Six Hook Types:**
1. **import_resolver** - Module path resolution
2. **type_validator** - Type checking logic
3. **pre_eval** - AST transformation before evaluation
4. **post_eval** - Result inspection/modification
5. **security_check** - Capability enforcement
6. **optimizer** - Performance optimization

#### 4. Plugin Global Object (`PluginGlobalObject`)

Available to plugin code as the `plugin` global:

```zexus
plugin.register_hook(name, handler)
plugin.grant_capability(capability)
plugin.has_capability(capability)
plugin.metadata()
plugin.get_hooks()
plugin.get_capabilities()
plugin.get_loaded_plugins()
plugin.load(name, config)
```

### Plugin Loading Flow

```
File System
    ↓
plugin.zx file
    ↓
Parse @plugin metadata
    ↓
Validate requirements (capabilities available?)
    ↓
Register hooks
    ↓
Add to loaded_plugins
    ↓
Return PluginMetadata
```

### Capability Flow

```
Plugin declares in metadata:
  provides: ["json.parse", "json.stringify"]
              ↓
PluginManager.load_plugin() adds to capabilities
              ↓
Code that requires ["json.parse"] can check:
  plugin.has_capability("json.parse")
              ↓
Transitive dependency checking (Phase 3)
              ↓
Sandbox policy enforcement (Phase 3)
```

### Integration Points

#### With Evaluator
```python
# In evaluator/core.py eval_node()
def eval_node(self, node, env):
    # Pre-evaluation hook
    node = self.plugin_manager.call_hooks("pre_eval", node, env)
    
    # Standard evaluation
    result = self._eval_node_impl(node, env)
    
    # Post-evaluation hook
    result = self.plugin_manager.call_hooks("post_eval", result, node, env)
    
    return result
```

#### With Parser
```python
# In parser/parser.py parse_expression()
def parse_expression(self):
    # Custom syntax hook
    expr = self.plugin_manager.call_hooks("pre_parse", self.cur_token)
    if expr is not None:
        return expr
    
    # Default parsing
    return self._parse_expression_impl()
```

#### With Module System
```python
# In module_manager.py resolve_module()
def resolve_module(self, name):
    # Plugin-based module resolution
    path = self.plugin_manager.call_hooks("import_resolver", name)
    if path is not None:
        return self.load_module(path)
    
    # Default resolution
    return self._resolve_default(name)
```

---

## Builtin Plugins Overview

### 1. JSON Plugin
- **Capabilities:** `json.parse`, `json.stringify`
- **Functions:** `parse()`, `stringify()`, `pretty()`
- **Use Case:** Data serialization

### 2. Logging Plugin
- **Capabilities:** `logging.debug`, `.info`, `.warn`, `.error`
- **Configuration:** level, format, output
- **Use Case:** Structured logging

### 3. Crypto Plugin
- **Capabilities:** `crypto.sha256`, `.sha512`, `.hmac`, `.random`
- **Functions:** Cryptographic operations
- **Use Case:** Security operations

### 4. Validation Plugin
- **Capabilities:** `validation.email`, `.url`, `.phone`
- **Hooks:** `type_validator`
- **Use Case:** Data validation

### 5. Collections Plugin
- **Capabilities:** `collections.map`, `.filter`, `.reduce`, `.zip`, `.group`
- **Functions:** Functional programming utilities
- **Use Case:** Data transformation

---

## Design Principles

### 1. Non-Invasiveness
- Modifiers attached post-construction
- Plugins loaded dynamically
- Hooks optional (no handlers = no effect)
- Core evaluator unchanged

### 2. Safety
- Capabilities declared explicitly
- Hooks execute in defined context
- Timeouts prevent runaway handlers
- Errors in hooks caught, don't crash runtime

### 3. Extensibility
- New hooks can be added without breaking existing plugins
- Plugins can provide other plugins as hooks
- Configuration schema enables future validation

### 4. Performance
- Hook registration O(1)
- Hook execution O(n) where n = registered handlers
- Priority-based execution allows critical hooks first
- No reflection overhead (direct function calls)

---

## Integration with Future Phases

### Phase 3: Capability-Based Security
```
Plugin provides ["crypto.hash"]
              ↓
Code requires ["crypto.hash"]
              ↓
Evaluator calls security_check hook
              ↓
Sandbox policy allows/denies
              ↓
Access granted/error raised
```

### Phase 4: Virtual Filesystem
```
Plugin hook transforms I/O operations
              ↓
Path like "/app/data.txt"
              ↓
pre_eval hook sandboxes to "/app/plugins/myplugin/"
              ↓
IO operation proceeds with sandboxed path
```

### Phase 5: Type System
```
Type annotation: action foo(x: int) { ... }
              ↓
type_validator hook checks type at runtime/compile
              ↓
Custom validators from validation plugin
              ↓
Compile-time or runtime type error
```

### Phase 6: Metaprogramming
```
Plugin registers pre_eval hook
              ↓
AST transformation (e.g., macro expansion)
              ↓
Code generated/modified before evaluation
              ↓
Result evaluates transformed code
```

### Phase 7: Optimizations
```
Optimizer hook receives AST
              ↓
Plugin applies optimization (constant folding, inlining, etc.)
              ↓
Returns optimized AST
              ↓
Evaluator processes optimized code
```

---

## File Structure

```
zexus-interpreter/
├── src/zexus/
│   ├── plugin_system.py          # PluginManager, PluginMetadata, hooks
│   ├── builtin_plugins.py        # 5 builtin plugins
│   ├── parser/
│   │   └── parser.py             # Modified for modifiers
│   ├── evaluator/
│   │   ├── statements.py         # Modified eval_action_statement
│   │   └── core.py               # Will integrate hooks
│   └── ...
├── docs/
│   ├── MODIFIERS.md              # Modifier system user guide
│   ├── PLUGIN_SYSTEM.md          # Complete plugin design
│   ├── PLUGIN_QUICK_REFERENCE.md # Quick reference
│   └── ...
├── test_plugin_system.py         # 13 unit tests
└── PHASE_1_2_SUMMARY.md          # This session's work
```

---

## Next Steps

### Immediate (Phase 3)
- [ ] Implement `CapabilityManager` class
- [ ] Add `security_check` hook calls in evaluator
- [ ] Create audit logging for capabilities
- [ ] Design base capability set (io, crypto, network, etc.)
- [ ] Build sandbox policy system
- [ ] Test capability enforcement

### Short Term (Phase 4-5)
- [ ] Virtual filesystem abstraction
- [ ] Unified type system (initial)
- [ ] Type validator integration

### Medium Term (Phase 6-7)
- [ ] Metaprogramming hooks
- [ ] Optimization framework
- [ ] Bytecode compilation

### Long Term (Phase 8+)
- [ ] Plugin marketplace
- [ ] Advanced type system (generics, traits)
- [ ] Performance profiling

---

## Summary

The Zexus language has been elevated from a simple interpreter to an **extensible, semantic, plugin-based language system**. 

**Completed:**
- Modifiers enable semantic depth
- Plugins enable safe extensibility
- Hooks enable interception/transformation
- Capabilities enable fine-grained control

**Ready for:**
- Real-world plugin ecosystems
- Security-critical applications (via capabilities)
- Advanced language features (type system, metaprogramming)
- Performance optimization

The foundation is solid. The language is ready for depth.
