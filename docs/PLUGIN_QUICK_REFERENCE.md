# Zexus Plugin System Quick Reference

## Quick Start

### Creating a Simple Plugin

```zexus
@plugin {
  name: "my_plugin",
  version: "1.0.0",
  provides: ["myfeature.operation"],
  description: "My first plugin"
}

public action my_operation(input) {
  ret input ++ "_processed";
}
```

### Using a Plugin

```zexus
use plugin("my_plugin");

action main() {
  let result = my_plugin.my_operation("hello");
  ret result;
}
```

---

## Plugin Metadata

```zexus
@plugin {
  name: "string",           # Required: unique identifier
  version: "semver",        # Required: version number
  author: "string",         # Optional: author name
  description: "string",    # Optional: what it does
  requires: ["capability"], # Optional: capabilities needed
  provides: ["capability"], # Optional: capabilities provided
  hooks: ["hook_name"],     # Optional: hooks this plugin registers
  config: {                 # Optional: configuration schema
    setting: { type: "string", default: "value" }
  }
}
```

---

## Hook System

### Registering a Hook

```zexus
action setup_hooks() {
  plugin.register_hook("pre_eval", action(node, ctx) {
    # Inspect/transform AST node before evaluation
    ret node;  # Return modified or original node
  });
}

# Call setup_hooks() when plugin initializes
setup_hooks();
```

### Available Hooks

| Hook | Signature | Purpose |
|------|-----------|---------|
| `import_resolver` | `(path: string, ctx) → string` | Customize module resolution |
| `type_validator` | `(value: any, spec: object) → bool` | Implement custom type checking |
| `pre_eval` | `(node: ASTNode, ctx) → ASTNode` | Transform AST before evaluation |
| `post_eval` | `(result: any, node: ASTNode, ctx) → any` | Inspect/modify eval results |
| `security_check` | `(capability: string, ctx) → bool` | Enforce capability access |
| `optimizer` | `(ast: AST, ctx) → AST` | Performance optimization |

---

## Plugin Global API

```zexus
# Registration
plugin.register_hook(name, handler)      # Register a hook handler
plugin.grant_capability(capability)      # Declare a capability

# Checking
plugin.has_capability(capability)        # Check if capability available
plugin.check_security(capability)        # Security check (raises on deny)

# Introspection
plugin.metadata()                        # Get current plugin metadata
plugin.get_hooks()                       # List all registered hooks
plugin.get_capabilities()                # List available capabilities
plugin.get_loaded_plugins()              # List loaded plugin names

# Loading
plugin.load(name)                        # Load another plugin
plugin.load(name, { config: {...} })     # Load with configuration
```

---

## Builtin Plugins

### JSON Plugin

```zexus
use plugin("json");

let obj = json.parse("{\"key\": \"value\"}");
let str = json.stringify(obj);
let pretty = json.pretty(obj);
```

### Logging Plugin

```zexus
use plugin("logging", {
  config: {
    level: "debug",
    format: "json",
    output: "stderr"
  }
});

let log = logging;
log.debug("Debug message");
log.info("Info message");
log.warn("Warning message");
log.error("Error message");
```

### Crypto Plugin

```zexus
use plugin("crypto");

let hash = crypto.sha256("input");
let hash512 = crypto.sha512("input");
let hmac = crypto.hmac("secret", "message", "sha256");
let random = crypto.random(32);  # 32 bytes of random data
```

### Validation Plugin

```zexus
use plugin("validation");

if validation.validate_email("user@example.com") {
  # Email is valid
}

if validation.validate_url("https://example.com") {
  # URL is valid
}

if validation.validate_phone("+1234567890") {
  # Phone is valid
}
```

### Collections Plugin

```zexus
use plugin("collections");

let nums = [1, 2, 3, 4, 5];
let doubled = collections.map(nums, action(x) { ret x * 2; });
let evens = collections.filter(nums, action(x) { ret x % 2 == 0; });
let sum = collections.reduce(nums, action(acc, x) { ret acc + x; }, 0);

let pairs = collections.zip([1, 2, 3], ["a", "b", "c"]);
# Result: [[1, "a"], [2, "b"], [3, "c"]]

let grouped = collections.group(
  [
    {type: "A", val: 1},
    {type: "B", val: 2},
    {type: "A", val: 3}
  ],
  action(item) { ret item.type; }
);
# Result: {A: [...], B: [...]}
```

---

## Examples

### Example 1: Custom Logging Plugin

```zexus
@plugin {
  name: "custom_logger",
  version: "1.0.0",
  provides: ["logging.custom"],
  config: {
    prefix: { type: "string", default: "[APP]" }
  }
}

public action log(msg) {
  let prefix = plugin.config.prefix;
  stdout(prefix ++ " " ++ msg);
}
```

### Example 2: Hook-Based Validator

```zexus
@plugin {
  name: "strict_types",
  version: "1.0.0",
  hooks: ["type_validator"]
}

action register_validators() {
  plugin.register_hook("type_validator", action(value, spec) {
    # Strict type checking
    cond {
      spec.type == "int" ? typeof(value) == "number";
      spec.type == "string" ? typeof(value) == "string";
      spec.type == "array" ? typeof(value) == "array";
      else ? true;
    }
  });
}

register_validators();
```

### Example 3: Plugin with Dependencies

```zexus
@plugin {
  name: "data_processor",
  version: "1.0.0",
  requires: ["json.parse", "crypto.hash"],
  provides: ["processor.secure_parse"]
}

public action secure_parse(json_str) {
  # Requires json plugin
  let obj = json.parse(json_str);
  
  # Requires crypto plugin
  let hash = crypto.sha256(json_str);
  
  ret {
    data: obj,
    hash: hash
  };
}
```

---

## Configuration

### Passing Configuration

```python
# In Python/evaluator code
manager.load_plugin("logging", config={
    "level": "debug",
    "output": "/var/log/app.log"
})
```

### Using Configuration in Plugin

```zexus
@plugin {
  config: {
    timeout: { type: "number", default: 5000 },
    retries: { type: "number", default: 3 }
  }
}

action main() {
  let timeout = plugin.config.timeout;    # 5000
  let retries = plugin.config.retries;    # 3
}
```

---

## Capabilities

### Declaring Dependencies

```zexus
@requires {
  capabilities: ["crypto.hash", "io.read"]
}

action process_secure_file(path) {
  let content = io.read(path);
  let hash = crypto.hash(content);
  ret hash;
}
```

### Checking at Runtime

```zexus
public action safe_hash(input) {
  if !plugin.has_capability("crypto.hash") {
    ret error("Crypto not available");
  }
  ret crypto.hash(input);
}
```

---

## Best Practices

### 1. Clear Exports
Always mark exported actions with `public`:

```zexus
public action exported_feature() { }
action internal_helper() { }  # Not exported
```

### 2. Declare All Dependencies
List all capabilities your plugin needs:

```zexus
@plugin {
  requires: ["crypto.hash", "io.read"],
  provides: ["secure.process"]
}
```

### 3. Graceful Degradation
Check capabilities at runtime:

```zexus
public action process(data) {
  if plugin.has_capability("crypto.hash") {
    ret hash_data(data);
  } else {
    ret data;  # Fallback
  }
}
```

### 4. Configuration Defaults
Provide sensible defaults:

```zexus
@plugin {
  config: {
    timeout: { type: "number", default: 5000 },
    retries: { type: "number", default: 3 },
    cache: { type: "bool", default: true }
  }
}
```

### 5. Clear Metadata
Use descriptive names and versions:

```zexus
@plugin {
  name: "json_processor",  # Clear, specific name
  version: "1.0.0",        # Semantic versioning
  description: "JSON parsing with validation",
  author: "Your Name"
}
```

---

## Troubleshooting

### Plugin Not Loading
- Check that plugin file exists and is valid Zexus code
- Verify `@plugin` metadata is correct
- Check required capabilities are available

### Hook Not Called
- Verify hook is registered with `plugin.register_hook()`
- Check hook name matches (case-sensitive)
- Ensure handler function has correct signature

### Capability Error
- Check plugin `requires` lists all needed capabilities
- Verify capabilities are provided by other loaded plugins
- Use `plugin.get_capabilities()` to debug

---

## See Also

- **Full Documentation:** `docs/PLUGIN_SYSTEM.md`
- **Modifier System:** `docs/MODIFIERS.md`
- **Source Code:** `src/zexus/plugin_system.py`
- **Builtin Plugins:** `src/zexus/builtin_plugins.py`
