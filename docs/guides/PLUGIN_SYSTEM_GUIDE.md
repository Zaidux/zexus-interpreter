# Zexus Plugin System - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How to Extend Zexus Without New Keywords](#how-to-extend-without-keywords)
4. [Plugin Components](#plugin-components)
5. [Event System](#event-system)
6. [Hook System](#hook-system)
7. [Capability System](#capability-system)
8. [Creating Custom Plugins](#creating-custom-plugins)
9. [Examples](#examples)
10. [Best Practices](#best-practices)

---

## Overview

The Zexus plugin system allows you to **extend the language's functionality without adding new keywords**. Instead of modifying the core language, you can:

✅ Add new built-in functions  
✅ Hook into language events (function calls, variable access, etc.)  
✅ Create middleware for security and validation  
✅ Implement domain-specific features  
✅ Add external library integrations  

**Philosophy:** "Extend, don't modify" - Keep the language core simple and stable

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Zexus Core                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Lexer      │→ │   Parser     │→ │  Evaluator   │  │
│  └──────────────┘  └──────────────┘  └──────┬───────┘  │
│                                               │          │
└───────────────────────────────────────────────┼──────────┘
                                                │
                                                ↓
                    ┌───────────────────────────────────┐
                    │      Plugin System Layer          │
                    ├───────────────────────────────────┤
                    │  • Event Emitter                  │
                    │  • Hook Registry                  │
                    │  • Capability Manager             │
                    │  • Plugin Manager                 │
                    └───────────┬───────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ↓               ↓               ↓
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │   Plugin A   │ │   Plugin B   │ │   Plugin C   │
        │  (Logging)   │ │  (Caching)   │ │  (Security)  │
        └──────────────┘ └──────────────┘ └──────────────┘
```

---

## How to Extend Zexus Without New Keywords

### Method 1: Add Built-in Functions

Instead of adding keywords, add functions to the global environment.

**Example: Adding HTTP support without `http` keyword**

```python
# In Python (plugin implementation)
from zexus.object import Function, String

def http_get(args):
    """Built-in function for HTTP GET requests"""
    if len(args) != 1:
        return Error("http_get expects 1 argument: url")
    
    url = args[0].value
    response = requests.get(url)
    return String(response.text)

# Register as built-in
environment.set("httpGet", Function(http_get))
```

**Usage in Zexus:**
```zexus
// No new keyword needed! Just use the function
let response = httpGet("https://api.example.com/data");
print("Response: " + response);
```

---

### Method 2: Use the Event System

Hook into language events without modifying the interpreter.

**Example: Auto-logging without `log` keyword**

```zexus
// Register event listener
on("function_call", function(event) {
    print("Function called: " + event.name);
    print("Arguments: " + event.args);
});

// Now all function calls are automatically logged
let result = add(5, 3);  // Logs: "Function called: add"
```

---

### Method 3: Use Hooks for Middleware

Wrap existing functionality with hooks.

**Example: Auto-validation without `validate` keyword**

```zexus
// Add validation hook
beforeFunctionCall(function(funcName, args) {
    if (funcName == "divide") {
        if (args[1] == 0) {
            throw("Division by zero prevented!");
        }
    }
});

// Now division is automatically safe
let result = divide(10, 2);  // Works
let bad = divide(10, 0);     // Prevented by hook
```

---

### Method 4: Capability-Based Features

Use the capability system for feature control.

**Example: File access control without `allow` keyword**

```zexus
// Check capabilities instead of keywords
function safeReadFile(path) {
    if (!hasCapability("io.read")) {
        throw("Missing io.read capability");
    }
    return readFile(path);
}

// Use like normal function
let content = safeReadFile("data.txt");
```

---

## Plugin Components

### 1. Plugin Registration

```zexus
// Define a plugin (in Zexus code)
let myPlugin = {
    name: "MyPlugin",
    version: "1.0.0",
    
    init: function() {
        print("Plugin initialized");
        registerBuiltins();
    },
    
    cleanup: function() {
        print("Plugin cleaned up");
    }
};

// Register the plugin
registerPlugin(myPlugin);
```

### 2. Python-Side Plugin

```python
# src/plugins/my_plugin.py

class MyPlugin:
    """Custom Zexus plugin"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.name = "MyPlugin"
        self.version = "1.0.0"
    
    def register_builtins(self, env):
        """Add custom built-in functions"""
        from zexus.object import Function, Integer
        
        def custom_function(args):
            # Your implementation
            return Integer(42)
        
        env.set("customFunc", Function(custom_function))
    
    def register_hooks(self):
        """Register event hooks"""
        self.evaluator.on("function_call", self.on_function_call)
    
    def on_function_call(self, event):
        """Handle function call events"""
        print(f"[Plugin] Function {event.name} called")

# Register with evaluator
from plugins.my_plugin import MyPlugin
plugin = MyPlugin(evaluator)
plugin.register_builtins(environment)
plugin.register_hooks()
```

---

## Event System

The event system allows plugins to listen for language events.

### Available Events

| Event Name | Triggered When | Data Available |
|------------|---------------|----------------|
| `function_call` | Function is called | name, args, caller |
| `variable_set` | Variable is assigned | name, value, scope |
| `variable_get` | Variable is accessed | name, scope |
| `module_loaded` | Module is imported | name, path |
| `error` | Runtime error occurs | message, stack |
| `custom:*` | Custom event | plugin-defined |

### Event API

```zexus
// Emit an event
emit("custom:data_loaded", { size: 1024, time: 500 });

// Listen for events
on("custom:data_loaded", function(data) {
    print("Data loaded: " + data.size + " bytes");
    print("Time taken: " + data.time + "ms");
});

// One-time listener
once("custom:startup", function() {
    print("Application started");
});

// Remove listener
let listener = on("custom:event", handler);
off("custom:event", listener);
```

---

## Hook System

Hooks allow middleware-style interception.

### Hook Types

#### Before Hooks
Run before the operation:

```zexus
beforeFunctionCall(function(name, args) {
    print("About to call: " + name);
    // Can modify args
    return args;
});
```

#### After Hooks
Run after the operation:

```zexus
afterFunctionCall(function(name, args, result) {
    print("Function " + name + " returned: " + result);
    // Can modify result
    return result;
});
```

#### Around Hooks
Wrap the entire operation:

```zexus
aroundFunctionCall(function(name, args, proceed) {
    print("Before: " + name);
    let result = proceed(args);  // Call original
    print("After: " + name);
    return result;
});
```

### Hook Examples

**Performance Monitoring:**
```zexus
afterFunctionCall(function(name, args, result) {
    if (name == "slowFunction") {
        recordMetric("slowFunction_calls", 1);
    }
    return result;
});
```

**Automatic Caching:**
```zexus
let cache = {};

aroundFunctionCall(function(name, args, proceed) {
    if (name == "expensiveComputation") {
        let key = name + "_" + args[0];
        if (cache[key]) {
            print("Cache hit!");
            return cache[key];
        }
        let result = proceed(args);
        cache[key] = result;
        return result;
    }
    return proceed(args);
});
```

**Input Validation:**
```zexus
beforeFunctionCall(function(name, args) {
    if (name == "processUser") {
        if (!args[0] || !args[0].email) {
            throw("User must have email");
        }
    }
    return args;
});
```

---

## Capability System

Control feature access without keywords.

### Checking Capabilities

```zexus
function secureOperation() {
    if (!hasCapability("io.write")) {
        throw("Missing io.write capability");
    }
    // Perform operation
}
```

### Requesting Capabilities

```zexus
// At module/plugin level
requireCapability("network.http");
requireCapability("io.read");

// Then use features
let data = httpGet("https://api.com");
```

### Granting Capabilities

```python
# In Python plugin
def grant_capabilities(env, capabilities):
    """Grant capabilities to environment"""
    env.capabilities = capabilities

# Usage
grant_capabilities(env, ["io.read", "io.write", "network.http"])
```

---

## Creating Custom Plugins

### Full Plugin Example: Database Plugin

**File: `src/plugins/database_plugin.py`**

```python
import sqlite3
from zexus.object import Function, String, Integer, Hash, Error

class DatabasePlugin:
    """SQLite database plugin for Zexus"""
    
    def __init__(self):
        self.connections = {}
    
    def register(self, env):
        """Register all database functions"""
        
        # db_connect(path)
        def db_connect(args):
            if len(args) != 1:
                return Error("db_connect expects 1 arg: path")
            path = args[0].value
            conn = sqlite3.connect(path)
            conn_id = id(conn)
            self.connections[conn_id] = conn
            return Integer(conn_id)
        
        # db_query(conn_id, sql)
        def db_query(args):
            if len(args) != 2:
                return Error("db_query expects 2 args")
            conn_id = args[0].value
            sql = args[1].value
            
            if conn_id not in self.connections:
                return Error("Invalid connection")
            
            conn = self.connections[conn_id]
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            # Convert to Zexus list of maps
            result = []
            for row in rows:
                result.append(Hash({
                    String("data"): String(str(row))
                }))
            return result
        
        # db_close(conn_id)
        def db_close(args):
            if len(args) != 1:
                return Error("db_close expects 1 arg")
            conn_id = args[0].value
            
            if conn_id in self.connections:
                self.connections[conn_id].close()
                del self.connections[conn_id]
            return None
        
        # Register functions
        env.set("dbConnect", Function(db_connect))
        env.set("dbQuery", Function(db_query))
        env.set("dbClose", Function(db_close))
```

**Usage in Zexus:**

```zexus
// No new keywords! Just use the plugin functions
let conn = dbConnect("users.db");
let users = dbQuery(conn, "SELECT * FROM users");

for (let i = 0; i < users.length; i = i + 1) {
    print("User: " + users[i].data);
}

dbClose(conn);
```

---

### Plugin Example: HTTP Client

**File: `src/plugins/http_plugin.py`**

```python
import requests
from zexus.object import Function, String, Hash, Error

class HttpPlugin:
    def register(self, env):
        def http_get(args):
            if len(args) < 1:
                return Error("httpGet expects: url, [headers]")
            
            url = args[0].value
            headers = {}
            
            if len(args) > 1:
                # Parse headers from Zexus map
                headers_map = args[1]
                for key, value in headers_map.pairs.items():
                    headers[key.value] = value.value
            
            try:
                response = requests.get(url, headers=headers)
                return Hash({
                    String("status"): Integer(response.status_code),
                    String("body"): String(response.text),
                    String("headers"): String(str(response.headers))
                })
            except Exception as e:
                return Error(f"HTTP error: {str(e)}")
        
        def http_post(args):
            # Similar implementation
            pass
        
        env.set("httpGet", Function(http_get))
        env.set("httpPost", Function(http_post))
```

**Usage:**

```zexus
let response = httpGet("https://api.github.com/users/octocat", {
    "User-Agent": "Zexus",
    "Accept": "application/json"
});

if (response.status == 200) {
    print("Success: " + response.body);
} else {
    print("Error: " + response.status);
}
```

---

## Examples

### Example 1: Logging Plugin

```zexus
// logging_plugin.zx

let LoggingPlugin = {
    name: "Logging",
    levels: { debug: 0, info: 1, warn: 2, error: 3 },
    currentLevel: 1,
    
    init: function() {
        print("[LoggingPlugin] Initialized");
        
        // Register logging functions
        this.registerLoggers();
        
        // Hook all function calls for debug logging
        afterFunctionCall(function(name, args, result) {
            if (LoggingPlugin.currentLevel <= 0) {
                print("[DEBUG] " + name + " returned: " + result);
            }
        });
    },
    
    registerLoggers: function() {
        // These become built-in-like functions
        logDebug = function(msg) {
            if (LoggingPlugin.currentLevel <= 0) {
                print("[DEBUG] " + msg);
            }
        };
        
        logInfo = function(msg) {
            if (LoggingPlugin.currentLevel <= 1) {
                print("[INFO] " + msg);
            }
        };
        
        logWarn = function(msg) {
            if (LoggingPlugin.currentLevel <= 2) {
                print("[WARN] " + msg);
            }
        };
        
        logError = function(msg) {
            print("[ERROR] " + msg);
        };
    },
    
    setLevel: function(level) {
        this.currentLevel = level;
        logInfo("Log level set to: " + level);
    }
};

// Initialize
LoggingPlugin.init();

// Usage:
logDebug("Application starting");
logInfo("User logged in");
logWarn("High memory usage");
logError("Database connection failed");
```

---

### Example 2: Validation Plugin

```zexus
// validation_plugin.zx

let ValidationPlugin = {
    rules: {},
    
    addRule: function(funcName, validator) {
        this.rules[funcName] = validator;
        
        // Hook the function
        beforeFunctionCall(function(name, args) {
            if (ValidationPlugin.rules[name]) {
                let isValid = ValidationPlugin.rules[name](args);
                if (!isValid) {
                    throw("Validation failed for: " + name);
                }
            }
            return args;
        });
    }
};

// Add validation rules
ValidationPlugin.addRule("createUser", function(args) {
    let user = args[0];
    return user.email && user.name;
});

ValidationPlugin.addRule("divide", function(args) {
    return args[1] != 0;
});

// Now functions are automatically validated
createUser({ email: "test@example.com", name: "Test" });  // ✓
createUser({ name: "Test" });  // ✗ Validation failed

divide(10, 2);  // ✓
divide(10, 0);  // ✗ Validation failed
```

---

### Example 3: Caching Plugin

```zexus
// caching_plugin.zx

let CachingPlugin = {
    cache: {},
    hits: 0,
    misses: 0,
    
    memoize: function(funcName) {
        aroundFunctionCall(function(name, args, proceed) {
            if (name == funcName) {
                let key = name + "_" + args.join("_");
                
                if (CachingPlugin.cache[key]) {
                    CachingPlugin.hits = CachingPlugin.hits + 1;
                    print("[CACHE HIT] " + funcName);
                    return CachingPlugin.cache[key];
                } else {
                    CachingPlugin.misses = CachingPlugin.misses + 1;
                    print("[CACHE MISS] " + funcName);
                    let result = proceed(args);
                    CachingPlugin.cache[key] = result;
                    return result;
                }
            }
            return proceed(args);
        });
    },
    
    stats: function() {
        print("Cache hits: " + this.hits);
        print("Cache misses: " + this.misses);
        let hitRate = (this.hits * 100) / (this.hits + this.misses);
        print("Hit rate: " + hitRate + "%");
    },
    
    clear: function() {
        this.cache = {};
        this.hits = 0;
        this.misses = 0;
        print("Cache cleared");
    }
};

// Enable caching for expensive functions
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

CachingPlugin.memoize("fibonacci");

let result = fibonacci(10);  // Slow, caches results
let result2 = fibonacci(10); // Fast, from cache

CachingPlugin.stats();
// Output:
// Cache hits: 8
// Cache misses: 11
// Hit rate: 42%
```

---

## Best Practices

### 1. **Namespace Your Functions**
```zexus
// ❌ Bad - pollutes global namespace
function get() { }
function set() { }

// ✅ Good - use prefixes or objects
function dbGet() { }
function dbSet() { }

// or
let db = {
    get: function() { },
    set: function() { }
};
```

### 2. **Check Capabilities Early**
```zexus
function readConfig() {
    // Check first
    if (!hasCapability("io.read")) {
        throw("Missing io.read capability");
    }
    
    // Then use
    return readFile("config.json");
}
```

### 3. **Use Events for Loose Coupling**
```zexus
// ❌ Bad - tight coupling
function saveUser(user) {
    database.save(user);
    emailService.sendWelcome(user);  // Direct dependency
    analytics.trackSignup(user);     // Direct dependency
}

// ✅ Good - loose coupling via events
function saveUser(user) {
    database.save(user);
    emit("user:created", user);  // Other systems listen
}

// Separate listeners
on("user:created", function(user) {
    emailService.sendWelcome(user);
});

on("user:created", function(user) {
    analytics.trackSignup(user);
});
```

### 4. **Handle Errors Gracefully**
```zexus
function safeHttpGet(url) {
    try {
        return httpGet(url);
    } catch (error) {
        logError("HTTP request failed: " + error);
        return { status: 500, body: "" };
    }
}
```

### 5. **Document Your Plugin API**
```zexus
/**
 * HTTP Plugin - Simple HTTP client
 * 
 * Functions:
 *   httpGet(url, headers?) -> { status, body, headers }
 *   httpPost(url, data, headers?) -> { status, body }
 * 
 * Requires capabilities:
 *   - network.http
 * 
 * Events emitted:
 *   - http:request { url, method }
 *   - http:response { status, time }
 */
let HttpPlugin = { ... };
```

---

## Summary

### Adding Functionality Without Keywords

| What You Want | How to Do It | Example |
|---------------|--------------|---------|
| New operation | Add built-in function | `httpGet(url)` |
| Middleware | Use hooks | `beforeFunctionCall(...)` |
| Feature control | Use capabilities | `hasCapability("io.read")` |
| Communication | Use events | `on("data:loaded", ...)` |
| Domain features | Create plugin object | `db.query(...)` |

### Key Takeaway

**You don't need to modify the Zexus core or add keywords!** The plugin system provides all the extensibility you need through:

1. **Built-in functions** - Add new operations
2. **Events** - Communicate between components  
3. **Hooks** - Intercept and modify behavior
4. **Capabilities** - Control feature access
5. **Plugins** - Package it all together

This keeps the language **simple, stable, and extensible**.

---

## Next Steps

Now that you understand the plugin system, you can:

1. Create custom plugins for your domain
2. Extend Zexus with external libraries (HTTP, databases, crypto)
3. Add middleware for logging, validation, security
4. Build reusable plugin packages for the ecosystem

**What functionality would you like to add?** Let me know and I can help design the plugin!
