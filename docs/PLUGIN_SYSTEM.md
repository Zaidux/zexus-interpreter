# Zexus Plugin System Design

## Overview

The Zexus plugin system leverages the modifier infrastructure to enable third-party extensions without compromising core language integrity. Plugins are self-contained Zexus modules that define capabilities and hooks, allowing users to extend the language with custom syntax handlers, builtin functions, security policies, and runtime behaviors.

## Core Concepts

### Plugins as Modules

A plugin is a `.zx` module that:
1. Declares capabilities via `@plugin` directive
2. Exports hook handlers (functions with specific signatures)
3. Registers with the runtime via `plugin.register()`
4. Can depend on other plugins with `requires` metadata

```zexus
@plugin {
  name: "json",
  version: "1.0.0",
  requires: [],
  provides: ["json.parse", "json.stringify"]
}

public action parse(input) {
  # Parse JSON string
  # Implementation using core language
}

public action stringify(obj) {
  # Convert object to JSON
  # Implementation using core language
}
```

### Capability Model

Each plugin declares **capabilities** it provides, forming a directed dependency graph:

```
core (always available)
  ├── json.parse
  ├── json.stringify
  ├── crypto.hash
  └── crypto.sign
```

Code that uses a capability must either:
- Explicitly declare dependency in its `@requires` metadata
- Import the plugin module
- Be explicitly granted the capability by sandboxing policy

### Hook System

Plugins can intercept language operations via hooks:

```zexus
# Register a custom import resolver
plugin.register_hook("import_resolver", action(path, ctx) {
  # Custom logic to resolve import paths
  ret custom_path;
});

# Register a custom type validator
plugin.register_hook("type_validator", action(value, type_spec) {
  # Validate value against custom type specification
  ret is_valid;
});

# Register a pre-evaluation handler
plugin.register_hook("pre_eval", action(node, ctx) {
  # Inspect or transform AST node before evaluation
  ret modified_node;
});
```

Available hooks:
- **import_resolver**: Customize module resolution
- **type_validator**: Implement custom type checking
- **pre_eval**: Transform AST nodes before evaluation
- **post_eval**: Inspect/modify evaluated results
- **security_check**: Enforce capability-based access control
- **optimizer**: Register optimization passes

## Plugin API

### `plugin` Global Object

The `plugin` object provides the following interface:

```zexus
# Metadata and registration
plugin.metadata()           # Returns {name, version, requires, provides}
plugin.register_hook(name, handler)     # Register a hook handler
plugin.grant_capability(cap)            # Request/declare capability
plugin.has_capability(cap)              # Check if capability available

# Introspection
plugin.get_hooks()                      # List registered hooks
plugin.get_capabilities()               # List available capabilities
plugin.get_loaded_plugins()             # List active plugins

# Module helpers
plugin.load(name)                       # Load plugin module
plugin.require_capability(caps)         # Declare required capabilities
```

### Hook Signatures

Each hook type has a standardized signature:

```zexus
# import_resolver(path: string, context: object) → string
# Resolve import path to actual module location
action handle_import(path, ctx) {
  ret ctx.default_resolver(path);
}

# type_validator(value: any, spec: object) → bool
# Validate value against type specification
action handle_type(value, spec) {
  ret typeof(value) == spec.type;
}

# pre_eval(node: ASTNode, context: object) → ASTNode
# Transform AST before evaluation
action handle_pre_eval(node, ctx) {
  ret node;  # Return modified or original node
}

# post_eval(result: any, node: ASTNode, context: object) → any
# Inspect/modify evaluation result
action handle_post_eval(result, node, ctx) {
  ret result;
}

# security_check(capability: string, context: object) → bool
# Check if capability access is allowed
action handle_security(cap, ctx) {
  ret true;  # Allow by default, deny raises error
}

# optimizer(ast: AST, context: object) → AST
# Perform optimization on AST
action handle_optimizer(ast, ctx) {
  ret ast;  # Return optimized AST
}
```

## Plugin Metadata

Every plugin must define metadata:

```zexus
@plugin {
  name: "string",              # Plugin identifier (required)
  version: "semver",           # Plugin version (required)
  author: "string",            # Plugin author
  description: "string",       # What this plugin does
  requires: ["capability"],    # Capabilities this plugin needs
  provides: ["capability"],    # Capabilities this plugin exports
  hooks: ["hook_name"],        # Hooks this plugin registers
  config: {                    # Optional configuration schema
    field: { type: "string", default: "value" }
  }
}
```

## Plugin Loading & Management

### Runtime Plugin Loader

The `PluginManager` class handles plugin lifecycle:

```python
class PluginManager:
    """Manage plugin loading, registration, and dependency resolution."""
    
    def __init__(self):
        self.loaded_plugins = {}
        self.hooks = defaultdict(list)
        self.capabilities = {}
        
    def load_plugin(self, module_path: str) -> dict:
        """Load and initialize a plugin module."""
        # 1. Parse plugin metadata
        # 2. Validate dependencies
        # 3. Register hooks
        # 4. Track capabilities
        
    def register_hook(self, hook_name: str, handler: Action):
        """Register a hook handler."""
        
    def call_hooks(self, hook_name: str, *args, **kwargs):
        """Execute all registered handlers for a hook."""
        
    def check_capability(self, capability: str) -> bool:
        """Check if capability is available."""
        
    def grant_capability(self, capability: str):
        """Grant a capability to the environment."""
```

### Plugin Discovery

Plugins are discovered from:
1. Builtin plugins (shipped with Zexus)
2. `$ZEXUS_PLUGINS` environment variable paths
3. `~/.zexus/plugins/` directory
4. Project `.zexus/plugins/` directory
5. Explicit imports with `use plugin@version`

```zexus
# Explicit plugin loading
use plugin("json", { version: "1.0.0" });

# Load with specific capabilities
use plugin("crypto", { requires: ["crypto.hash"] });
```

### Configuration

Plugins can accept configuration:

```zexus
@plugin {
  name: "logger",
  config: {
    level: { type: "string", default: "info" },
    output: { type: "string", default: "stdout" }
  }
}

action log(msg) {
  # Use configured level and output
}
```

Configuration is passed during plugin load:

```python
manager.load_plugin("logger", config={
    "level": "debug",
    "output": "/var/log/app.log"
})
```

## Example Plugins

### 1. JSON Plugin

```zexus
@plugin {
  name: "json",
  version: "1.0.0",
  provides: ["json.parse", "json.stringify"],
  description: "JSON serialization support"
}

public action parse(input) {
  # Parse JSON string into Zexus object
  # Implementation: use builtin string parsing
}

public action stringify(obj) {
  # Convert Zexus object to JSON string
  # Implementation: walk object tree
}
```

### 2. Validation Plugin

```zexus
@plugin {
  name: "validator",
  version: "1.0.0",
  provides: ["validation"],
  hooks: ["type_validator"]
}

action register_validators() {
  plugin.register_hook("type_validator", action(value, spec) {
    cond {
      spec.type == "email" ? validate_email(value);
      spec.type == "url" ? validate_url(value);
      spec.type == "phone" ? validate_phone(value);
      else ? false;
    }
  });
}

public action validate_email(addr) {
  ret addr ~matches /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
}
```

### 3. Logging Plugin (with Config)

```zexus
@plugin {
  name: "logger",
  version: "2.0.0",
  provides: ["logging"],
  config: {
    level: { type: "string", default: "info" },
    format: { type: "string", default: "json" }
  }
}

let LEVEL = { debug: 0, info: 1, warn: 2, error: 3 };

public action log(level, msg) {
  let config_level = plugin.config.level;
  if LEVEL[level] >= LEVEL[config_level] {
    cond {
      plugin.config.format == "json" ?
        stdout(json.stringify({ level, msg, ts: now() }));
      else ?
        stdout("[" ++ level ++ "] " ++ msg);
    }
  }
}
```

### 4. Custom Syntax Plugin (via Optimizer Hook)

```zexus
@plugin {
  name: "pattern_matching",
  version: "1.0.0",
  provides: ["pattern_matching"],
  hooks: ["pre_eval"]
}

action register_transformer() {
  plugin.register_hook("pre_eval", action(node, ctx) {
    # Transform custom syntax into standard Zexus
    # e.g., convert `match expr { pattern => result }`
    # into nested conditionals
    ret transform_pattern_match(node);
  });
}
```

## Integration Points

### 1. Evaluator Integration

Hooks are called from key evaluator points:

```python
# In evaluator/core.py
def eval_node(self, node, env):
    # Call pre_eval hooks
    node = self.plugin_manager.call_hooks("pre_eval", node, env)
    
    # Evaluate
    result = self._eval_node_impl(node, env)
    
    # Call post_eval hooks
    result = self.plugin_manager.call_hooks("post_eval", result, node, env)
    
    return result
```

### 2. Parser Integration

Custom syntax can be registered via hooks:

```python
# In parser/parser.py
def parse_expression(self):
    # Call pre_parse hooks (for custom syntax)
    expr = self.plugin_manager.call_hooks("pre_parse", self.cur_token)
    if expr is not None:
        return expr
    
    # Default parsing
    return self._parse_expression_impl()
```

### 3. Module System Integration

Plugins participate in module loading:

```python
# In module_manager.py
def resolve_module(self, name):
    # Call import_resolver hooks
    path = self.plugin_manager.call_hooks("import_resolver", name)
    if path is not None:
        return self.load_module(path)
    
    # Default resolution
    return self._resolve_default(name)
```

## Security Considerations

### Capability-Based Access Control

Plugins must declare required capabilities. The runtime enforces:
1. **Capability Grant**: Only code with granted capability can invoke it
2. **Transitive Closure**: Granting a capability grants all transitive dependencies
3. **Audit Trail**: All capability uses are logged

```zexus
@requires {
  capabilities: ["crypto.hash", "io.read"]
}

action process_file(path) {
  let content = io.read(path);  # Requires io.read
  let hash = crypto.hash(content);  # Requires crypto.hash
  ret hash;
}
```

### Sandbox Policy

Plugins can define sandbox policies:

```python
# Default deny all, explicitly allow needed
policy = SandboxPolicy(default_allow=False)
policy.allow_capability("json.parse")
policy.allow_capability("json.stringify")

manager.load_plugin("json", sandbox_policy=policy)
```

### Hook Safety

Hooks are executed in restricted context:
- Limited execution time (configurable timeout)
- Memory quotas
- No access to parent environment unless explicitly granted
- Changes to environment are isolated (can be rolled back)

## Rollout Strategy

### Phase 1: Core Infrastructure ✓ (Modifier System)
- Modifiers enable semantic extension
- Foundation for capabilities

### Phase 2: Plugin API (Current)
- Define `plugin` global object
- Implement `PluginManager` class
- Create hook system
- Example builtin plugins (json, logging)

### Phase 3: Capability System
- Implement capability tracking
- Build capability-based security model
- Integrate with evaluator

### Phase 4: Advanced Features
- Custom syntax via hooks
- Type system integration
- Performance optimization hooks

### Phase 5: Plugin Ecosystem
- Plugin repository/registry
- Version management
- Dependency resolution

## Testing Strategy

```python
# Test plugin loading
def test_load_simple_plugin():
    manager = PluginManager()
    plugin = manager.load_plugin("plugins/json.zx")
    assert plugin.metadata.name == "json"

# Test hook registration
def test_register_hook():
    manager = PluginManager()
    handler = create_test_handler()
    manager.register_hook("pre_eval", handler)
    assert len(manager.hooks["pre_eval"]) == 1

# Test capability checking
def test_capability_check():
    manager = PluginManager()
    manager.grant_capability("json.parse")
    assert manager.check_capability("json.parse")

# Test sandbox isolation
def test_sandbox_isolation():
    policy = SandboxPolicy(default_allow=False)
    # Load plugin with restricted capabilities
    # Verify it cannot access forbidden operations
```

## References

**Related Systems:**
- Modifiers (Phase 1): Define `public`, `private`, `sealed` for plugin exports
- Capabilities (Phase 3): Fine-grained access control for plugins
- Metaprogramming (Phase 5): AST hooks for syntax extension

**Inspiration:**
- Node.js/npm: Plugin discovery and dependency management
- Lua plugins: Lightweight, embedded extension model
- Wasm plugins: Sandboxed execution with clear boundaries
- Python importlib: Hook-based import system

---

## Next Steps

1. **Implement PluginManager** class with core methods
2. **Define plugin metadata format** and parser
3. **Create hook system** with execution pipeline
4. **Build example plugins** (json, logging, validation)
5. **Integrate with evaluator** at key points
6. **Add capability model** (see Phase 3)
7. **Create plugin documentation** for users
