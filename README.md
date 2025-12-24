# Zexus Programming Language

<div align="center">

![Zexus Logo](https://img.shields.io/badge/Zexus-v0.1.3-FF6B35?style=for-the-badge)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-Zaidux/zexus--interpreter-181717?style=for-the-badge&logo=github)](https://github.com/Zaidux/zexus-interpreter)

**A modern, security-first programming language with built-in blockchain support, VM-accelerated execution, advanced memory management, and policy-as-code**

[What's New](#-whats-new-in-v013) • [Features](#-key-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Keywords](#-complete-keyword-reference) • [Documentation](#-documentation) • [Examples](#-examples) • [Troubleshooting](#-getting-help--troubleshooting)

</div>

---

## 📋 Table of Contents

- [What is Zexus?](#-what-is-zexus)
- [What's New](#-whats-new-in-v013)
- [Key Features](#-key-features)
  - [VM-Accelerated Performance](#-vm-accelerated-performance-new)
  - [Security & Policy-as-Code](#-security--policy-as-code--verify-enhanced)
  - [Blockchain Support](#️-native-blockchain-support)
  - [Persistent Memory](#-persistent-memory-management)
  - [Dependency Injection](#-dependency-injection--testing)
  - [Reactive State](#-reactive-state-management)
  - [Advanced Features](#-advanced-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Examples](#-examples)
- [Complete Feature Reference](#-complete-feature-reference)
- [Complete Keyword Reference](#-complete-keyword-reference)
- [Built-in Functions](#built-in-functions-100)
- [CLI Commands](#-cli-commands)
- [Architecture](#️-architecture)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Testing](#-testing)
- [Getting Help & Troubleshooting](#-getting-help--troubleshooting)
- [License](#-license)
- [Roadmap](#️-roadmap)
- [Project Stats](#-project-stats)



---

## 🎯 What is Zexus?

Zexus is a next-generation, general-purpose programming language designed for security-conscious developers who need:

- **🔐 Policy-as-code** - Declarative security rules and access control
- **⚡ VM-Accelerated Execution** - Hybrid interpreter/compiler with bytecode VM
- **⛓️ Built-in Blockchain** - Native smart contracts and DApp primitives  
- **💾 Persistent Memory** - Cross-session data with automatic leak detection
- **🔌 Dependency Injection** - Powerful DI system with mocking for testing
- **👀 Reactive State** - WATCH for automatic state change reactions
- **🎭 Flexible Syntax** - Support for both universal (`{}`) and tolerant (`:`) styles
- **📦 Package Manager** - ZPM for dependency management
- **🚀 Main Entry Point** - Python-style `if __name__ == "__main__"` pattern support
- **🎨 UI Rendering** - Built-in screen, component, and theme system
- **🔒 Enterprise Features** - Middleware, authentication, throttling, and caching

---

## 🎉 What's New in v0.1.3

### Recent Enhancements

✅ **130+ Keywords Fully Operational** - All core language features tested and verified  
✅ **Dual-Mode DEBUG** - Function mode (`debug(x)`) and statement mode (`debug x;`)  
✅ **Multiple Syntax Styles** - `let x = 5`, `let x : 5`, `let x : int = 5` all supported  
✅ **Enterprise Keywords** - MIDDLEWARE, AUTH, THROTTLE, CACHE, INJECT fully functional  
✅ **Async/Await Runtime** - Complete Promise-based async system with context propagation  
✅ **Main Entry Point** - 15+ builtins for program lifecycle management  
✅ **UI Renderer** - SCREEN, COMPONENT, THEME keywords with 120+ tests  
✅ **Enhanced VERIFY** - Email, URL, phone validation, pattern matching, database checks  
✅ **Blockchain Keywords** - implements, pure, view, payable, modifier, this, emit  
✅ **100+ Built-in Functions** - Comprehensive standard library

### Bug Fixes & Improvements

✅ Fixed array literal parsing (no more duplicate elements)  
✅ Fixed ENUM value accessibility  
✅ Fixed WHILE condition parsing without parentheses  
✅ Fixed loop execution and variable reassignment  
✅ Fixed DEFER cleanup execution  
✅ Fixed SANDBOX return values  
✅ Fixed dependency injection container creation  
✅ Added tolerance blocks for REQUIRE  
✅ Improved error messages and debugging output

---

## ✨ Key Features

### ⚡ VM-Accelerated Performance (NEW!)

Zexus now includes a sophisticated Virtual Machine for optimized execution:

```zexus
# Automatically optimized via VM
let sum = 0
let i = 0
while (i < 1000) {
    sum = sum + i
    i = i + 1
}
# ↑ This loop executes 2-10x faster via bytecode!
```

**VM Features:**
- ✅ Stack-based bytecode execution
- ✅ Automatic optimization for loops and math-heavy code
- ✅ Async/await support (SPAWN, AWAIT opcodes)
- ✅ Function call optimization
- ✅ Collection operations (lists, maps)
- ✅ Event system
- ✅ Module imports
- ✅ Smart fallback to interpreter for unsupported features

[Learn more about VM integration →](VM_INTEGRATION_SUMMARY.md)

### 🔐 Security & Policy-as-Code (✨ VERIFY Enhanced!)
```zexus
# Define security policies declaratively
protect(transfer_funds, {
    rate_limit: 100,
    auth_required: true,
    require_https: true,
    allowed_ips: ["10.0.0.0/8"]
}, "strict")

# Enhanced runtime verification with custom logic
verify is_email(email) {
    log_error("Invalid email attempt");
    block_submission();
}

# Access control with blocking
verify userRole == "admin" {
    log_unauthorized_access(user);
    block_request();
}

# Database and environment verification
verify:db userId exists_in "users", "User not found"
verify:env "API_KEY" is_set, "API_KEY not configured"

# Data constraints
restrict(amount, {
    range: [0, 10000],
    type: "integer"
})
```
**NEW**: VERIFY now includes email/URL/phone validation, pattern matching, database checks, environment variables, input sanitization, and custom logic blocks! [See VERIFY Guide →](docs/VERIFY_ENHANCEMENT_GUIDE.md)

### ⛓️ Native Blockchain Support
```zexus
# Smart contracts made easy
contract Token {
    persistent storage balances: Map<Address, integer>
    
    action transfer(from: Address, to: Address, amount: integer) {
        require(balances[from] >= amount, "Insufficient balance")
        balances[from] = balances[from] - amount
        balances[to] = balances.get(to, 0) + amount
        emit Transfer(from, to, amount)
    }
}
```

### 💾 Persistent Memory Management
```zexus
# Store data across program runs
persist_set("user_preferences", preferences)
let prefs = persist_get("user_preferences")

# Automatic memory tracking
track_memory()  # Detects leaks automatically
```

### 🔌 Dependency Injection & Testing
```zexus
# Register dependencies
register_dependency("database", ProductionDB())

# Inject at runtime
inject database

# Mock for testing
test_mode(true)
mock_dependency("database", MockDB())
```

### 👀 Reactive State Management
```zexus
# Watch variables for changes
let count = 0
watch count {
    print("Count changed to: " + string(count))
}

count = 5  # Automatically triggers watch callback
```

### 🚀 Advanced Features

- **Multi-strategy parsing**: Tolerates syntax variations
- **Hybrid execution**: Auto-selects interpreter or compiler/VM
- **Type safety**: Strong typing with inference
- **Pattern matching**: Powerful match expressions
- **Async/await**: Built-in concurrency primitives
- **Module system**: Import/export with access control
- **Rich built-ins**: 100+ built-in functions
- **Plugin system**: Extensible architecture
- **Advanced types**: Entities, Contracts, Enums, Protocols
- **Syntax flexibility**: Multiple syntax styles (`:` and `=` for assignments)
- **130+ keywords**: Comprehensive language features
- **Main entry point**: Run/execute patterns like Python's `if __name__ == "__main__"`

---

## 🔍 Why Choose Zexus?

### Language Comparison

| Feature | Zexus | Python | Solidity | Rust | TypeScript |
|---------|-------|--------|----------|------|------------|
| **Blockchain Native** | ✅ Built-in | ❌ Libraries | ✅ Native | ❌ Libraries | ❌ Libraries |
| **Policy-as-Code** | ✅ Native | ⚠️ Manual | ⚠️ Modifiers | ❌ None | ❌ None |
| **VM Execution** | ✅ Hybrid | ✅ Bytecode | ✅ EVM | ✅ Native | ⚠️ V8/Node |
| **Type Safety** | ✅ Strong+Inference | ⚠️ Optional | ✅ Strong | ✅ Strong | ✅ Strong |
| **Async/Await** | ✅ Native | ✅ Native | ❌ None | ✅ Native | ✅ Native |
| **Dependency Injection** | ✅ Built-in | ⚠️ Libraries | ❌ None | ⚠️ Manual | ⚠️ Libraries |
| **Reactive State** | ✅ WATCH | ⚠️ Libraries | ❌ None | ⚠️ Libraries | ⚠️ Libraries |
| **Memory Tracking** | ✅ Automatic | ⚠️ Manual | ⚠️ Gas-based | ✅ Ownership | ⚠️ Manual |
| **Security Features** | ✅✅✅ Extensive | ⚠️ Libraries | ⚠️ Limited | ✅ Safe | ⚠️ Libraries |
| **Syntax Flexibility** | ✅ Multiple styles | ✅ PEP-8 | ✅ Solidity | ✅ Strict | ✅ Strict |
| **Learning Curve** | 🟢 Easy | 🟢 Easy | 🟡 Medium | 🔴 Hard | 🟡 Medium |

### Use Zexus When You Need

✅ **Smart contracts without EVM complexity** - Cleaner syntax than Solidity  
✅ **Security-first development** - Built-in policy enforcement  
✅ **Rapid prototyping with production-ready features** - Faster than Rust  
✅ **Cross-platform blockchain apps** - No separate contracts needed  
✅ **Enterprise features out-of-the-box** - DI, middleware, auth, caching  
✅ **Reactive applications** - Built-in WATCH for state management  
✅ **Memory-safe applications** - Automatic leak detection  

### Zexus = Python's Ease + Solidity's Blockchain + Rust's Safety

```zexus
# Python-like simplicity
let users = []
for each user in get_users() {
    print(user.name)
}

# Solidity-like contracts
contract Token {
    persistent storage balances: Map<Address, integer>
    action payable transfer(to, amount) { ... }
}

# Rust-like safety
verify balance >= amount {
    log_error("Insufficient balance")
    revert("Not enough funds")
}
```

---

## 📦 Installation

### Quick Install (Recommended)

```bash
pip install zexus
```

**Includes:**
- `zx` - Main Zexus CLI
- `zpm` - Zexus Package Manager

### From Source

```bash
git clone https://github.com/Zaidux/zexus-interpreter.git
cd zexus-interpreter
pip install -e .
```

### Verify Installation

```bash
zx --version   # Should show: Zexus v0.1.0
zpm --version  # Should show: ZPM v0.1.0
```

---

## 🚀 Quick Start

### 1. Hello World

```zexus
# hello.zx
let name = "World"
print("Hello, " + name + "!")
```

Run it:
```bash
zx run hello.zx
```

### 2. Interactive REPL

```bash
zx repl
```

```zexus
>> let x = 10 + 5
>> print(x * 2)
30
```

### 3. Create a Project

```bash
zx init my-app
cd my-app
zx run main.zx
```

---

## 💡 Examples

### Example 1: Secure API with Policy-as-Code

```zexus
entity ApiRequest {
    endpoint: string,
    method: string,
    user_id: integer
}

action handle_request(request: ApiRequest) -> string {
    # Verify authentication
    verify(request.user_id > 0)
    
    # Restrict input
    restrict(request.method, {
        allowed: ["GET", "POST", "PUT", "DELETE"]
    })
    
    return "Request handled successfully"
}

# Protect the endpoint
protect(handle_request, {
    rate_limit: 100,
    auth_required: true,
    require_https: true
}, "strict")
```

### Example 2: Blockchain Token

```zexus
contract ERC20Token {
    persistent storage total_supply: integer
    persistent storage balances: Map<Address, integer>
    
    action constructor(initial_supply: integer) {
        total_supply = initial_supply
        balances[msg.sender] = initial_supply
    }
    
    action transfer(to: Address, amount: integer) -> boolean {
        require(balances[msg.sender] >= amount, "Insufficient balance")
        balances[msg.sender] = balances[msg.sender] - amount
        balances[to] = balances.get(to, 0) + amount
        emit Transfer(msg.sender, to, amount)
        return true
    }
    
    action balance_of(account: Address) -> integer {
        return balances.get(account, 0)
    }
}
```

### Example 3: Reactive State Management

```zexus
# E-commerce cart with reactive updates
let cart_items = []
let cart_total = 0

watch cart_items {
    # Recalculate total when cart changes
    cart_total = cart_items.reduce(
        initial: 0,
        transform: total + item.price
    )
    print("Cart updated! New total: $" + string(cart_total))
}

# Add items (automatically triggers watch)
cart_items.push({name: "Laptop", price: 999})
cart_items.push({name: "Mouse", price: 29})
```

### Example 4: VM-Optimized Computation

```zexus
# Fibonacci with automatic VM optimization
action fibonacci(n: integer) -> integer {
    if n <= 1 {
        return n
    }
    
    let a = 0
    let b = 1
    let i = 2
    
    while (i <= n) {
        let temp = a + b
        a = b
        b = temp
        i = i + 1
    }
    
    return b
}

# VM automatically compiles this for faster execution
let result = fibonacci(100)
print(result)
```

### Example 5: Main Entry Point Pattern

```zexus
# Similar to Python's if __name__ == "__main__"
action main() {
    print("Running main program")
    let result = process_data()
    print("Result: " + string(result))
}

# Only runs if this is the main module
if is_main() {
    run(main)
}
```

### Example 6: Middleware & Enterprise Features

```zexus
# Define authentication middleware
middleware("auth", action(req, res) {
    if !req.has_token {
        return {status: 401, message: "Unauthorized"}
    }
    return true
})

# Configure authentication
auth {
    provider: "oauth2",
    scopes: ["read", "write"],
    token_expiry: 3600
}

# Apply rate limiting
throttle(api_endpoint, {
    requests_per_minute: 100,
    burst: 20
})

# Enable caching
cache(expensive_query, {
    ttl: 300,
    strategy: "lru"
})
```

### Example 7: Concurrency with Channels

```zexus
# Create typed channel
channel<integer> numbers

# Producer
action producer() {
    for each i in range(0, 10) {
        send(numbers, i)
        sleep(0.1)
    }
    close_channel(numbers)
}

# Consumer
action consumer() {
    while true {
        let value = receive(numbers)
        if value == null {
            break
        }
        print("Received: " + string(value))
    }
}

# Run concurrently
async producer()
async consumer()
```

---

## 📚 Complete Feature Reference

### Core Language Features

#### Variables & Constants
```zexus
# Multiple syntax options supported
let mutable_var = 42            # Standard assignment
let mutable_var : 42            # Colon syntax (tolerant style)
let typed_var : int = 42        # With type annotation
const IMMUTABLE = 3.14159       # Immutable constant
```

**Variable Scoping**: Zexus uses function-level scoping (not block-level). Variables can only be shadowed within function boundaries.

#### Data Types
- **Primitives**: Integer, Float, String, Boolean, Null
- **Collections**: List, Map, Set
- **Advanced**: Entity, Contract, Action, Lambda
- **Special**: DateTime, File, Math

#### Functions
```zexus
action greet(name: string) -> string {
    return "Hello, " + name
}

# Lambda functions
let double = lambda(x) { x * 2 }

# Deferred cleanup (executes on scope exit)
defer {
    cleanup_resources()
}
```

#### Debugging
```zexus
# DUAL-MODE DEBUG:
# Function mode - returns value, usable in expressions
let x = debug(42)           # Outputs: [DEBUG] 42, x = 42

# Statement mode - logs with metadata
debug myVariable;           # Outputs: 🔍 DEBUG: <value> with context

# Other debug tools
debug_log("message", context)
debug_trace()               # Stack trace
```

#### Control Flow
```zexus
# Conditionals
if condition {
    # code
} elif other_condition {
    # code
} else {
    # code
}

# Loops
while condition {
    # code
}

for each item in collection {
    # code
}

# Pattern Matching
match value {
    case 1: print("One")
    case 2: print("Two")
    default: print("Other")
}
```

#### Entities & Contracts
```zexus
entity User {
    name: string,
    age: integer,
    email: string
}

contract MyContract {
    persistent storage state: integer
    
    action update(new_value: integer) {
        state = new_value
    }
}
```

### Advanced Features

#### 🔐 Security Features

**PROTECT** - Policy-as-code security:
```zexus
protect(function_name, {
    rate_limit: 100,              # Max calls per minute
    auth_required: true,          # Require authentication
    require_https: true,          # HTTPS only
    allowed_ips: ["10.0.0.0/8"], # IP allowlist
    blocked_ips: ["192.168.1.100"], # IP blocklist
    log_access: true              # Audit logging
}, "strict")  # Enforcement mode: strict, warn, log
```

**VERIFY** - Runtime assertions:
```zexus
verify(user.is_admin)
verify(amount > 0 and amount < 1000)
```

**RESTRICT** - Input validation:
```zexus
restrict(input_value, {
    type: "string",
    min_length: 5,
    max_length: 100,
    pattern: "^[a-zA-Z0-9]+$",
    range: [0, 100],              # For numbers
    allowed: ["GET", "POST"]      # Enum values
})
```

**SEAL** - Immutable objects:
```zexus
seal(config)  # Make config immutable
```

**SANDBOX** - Isolated execution:
```zexus
sandbox {
    # Code runs in restricted environment
    # Limited file system, network access
}
```

**TRAIL** - Audit logging:
```zexus
trail(operation, "user_action", {
    user_id: user.id,
    action: "transfer",
    amount: 1000
})
```

#### 💾 Persistence & Memory

**Persistent Storage:**
```zexus
persist_set("key", value)
let value = persist_get("key")
persist_clear("key")
let all_keys = persist_list()
```

**Memory Tracking:**
```zexus
track_memory()              # Enable tracking
let stats = memory_stats()  # Get statistics
```

#### 🔌 Dependency Injection

**Register Dependencies:**
```zexus
register_dependency("logger", FileLogger("/var/log/app.log"))
register_dependency("database", PostgresDB("localhost:5432"))
```

**Inject Dependencies:**
```zexus
inject logger
inject database

action save_user(user: Entity) {
    logger.info("Saving user: " + user.name)
    database.insert("users", user)
}
```

**Mocking for Tests:**
```zexus
test_mode(true)
mock_dependency("logger", MockLogger())
mock_dependency("database", MockDB())

# Now all injected dependencies use mocks
```

#### 👀 Reactive State (WATCH)

```zexus
let counter = 0

watch counter {
    print("Counter changed to: " + string(counter))
    # Can trigger other logic
    if counter > 10 {
        send_alert()
    }
}

counter = counter + 1  # Triggers watch
```

#### ⛓️ Blockchain Features

**Transactions:**
```zexus
let tx = transaction({
    from: sender_address,
    to: recipient_address,
    value: 100,
    data: "0x1234"
})
```

**Events:**
```zexus
emit Transfer(from, to, amount)
```

**Smart Contract Primitives:**
```zexus
require(condition, "Error message")  # Revert if false
assert(condition)                     # Always check
revert("Reason")                      # Explicit revert
let balance = balance_of(address)
```

**Cryptographic Functions:**
```zexus
let hash = keccak256(data)
let sig = signature(data, private_key)
let valid = verify_sig(data, sig, public_key)
```

#### 🔄 Concurrency

**Async/Await:**
```zexus
async action fetch_data(url: string) -> string {
    let response = await http_get(url)
    return response.body
}

let data = await fetch_data("https://api.example.com/data")
```

**Channels:**
```zexus
channel messages

# Send
messages.send("Hello")

# Receive
let msg = messages.receive()
```

**Atomic Operations:**
```zexus
atomic {
    # Thread-safe operations
    counter = counter + 1
}
```

#### 📦 Module System

```zexus
# Export from module
export action public_function() {
    return "accessible"
}

private action internal_function() {
    return "not exported"
}

# Import in another file
use {public_function} from "mymodule"

# Import with alias
use {public_function as pf} from "mymodule"

# Import entire module
use * from "utilities"
```

#### 🎨 Pattern Matching

```zexus
match response_code {
    case 200: print("Success")
    case 404: print("Not Found")
    case 500: print("Server Error")
    case x where x >= 400 and x < 500: print("Client Error")
    default: print("Unknown status")
}

# Pattern matching with destructuring
match request {
    case {method: "GET", path: p}: handle_get(p)
    case {method: "POST", body: b}: handle_post(b)
    default: handle_other()
}
```

#### 🔧 Advanced Types

**Enums:**
```zexus
enum Status {
    PENDING,
    ACTIVE,
    COMPLETED,
    CANCELLED
}

let status = Status.ACTIVE
```

**Protocols (Interfaces):**
```zexus
protocol Serializable {
    action serialize() -> string
    action deserialize(data: string) -> Entity
}
```

**Type Aliases:**
```zexus
type_alias UserId = integer
type_alias UserMap = Map<UserId, User>
```

### Built-in Functions (100+)

#### I/O Functions
```zexus
print(value)                    # Print without newline
println(value)                  # Print with newline
input(prompt)                   # Get user input
read_text(path)                 # Read text file
write_text(path, content)       # Write text file
```

#### Type Conversion
```zexus
string(value)                   # Convert to string
int(value)                      # Convert to integer
float(value)                    # Convert to float
bool(value)                     # Convert to boolean
```

#### Collection Operations
```zexus
len(collection)                 # Length/size
list(items...)                  # Create list
map(pairs...)                   # Create map
set(items...)                   # Create set
range(start, end, step)         # Generate range
```

#### Functional Programming
```zexus
filter(collection, predicate)   # Filter elements
map(collection, transform)      # Transform elements
reduce(collection, fn, initial) # Reduce to single value
sort(collection, comparator)    # Sort elements
reverse(collection)             # Reverse order
```

#### String Operations
```zexus
join(array, separator)          # Join strings
split(string, delimiter)        # Split string
replace(string, old, new)       # Replace substring
uppercase(string)               # Convert to uppercase
lowercase(string)               # Convert to lowercase
trim(string)                    # Remove whitespace
substring(string, start, end)   # Extract substring
```

#### Math Operations
```zexus
abs(number)                     # Absolute value
ceil(number)                    # Ceiling
floor(number)                   # Floor
round(number, decimals)         # Round
min(numbers...)                 # Minimum
max(numbers...)                 # Maximum
sum(numbers)                    # Sum
sqrt(number)                    # Square root
pow(base, exponent)             # Power
random()                        # Random number
random(max)                     # Random 0 to max
random(min, max)                # Random in range
```

#### Date & Time
```zexus
now()                           # Current datetime
timestamp()                     # Unix timestamp
```

#### File I/O
```zexus
file_read_text(path)            # Read text file
file_write_text(path, content)  # Write text file
file_exists(path)               # Check if file exists
file_read_json(path)            # Read JSON file
file_write_json(path, data)     # Write JSON file
file_append(path, content)      # Append to file
file_list_dir(path)             # List directory
```

#### Persistence
```zexus
persist_set(key, value)         # Store persistent data
persist_get(key)                # Retrieve persistent data
persist_clear(key)              # Delete persistent data
persist_list()                  # List all keys
```

#### Memory Management
```zexus
track_memory()                  # Enable memory tracking
memory_stats()                  # Get memory statistics
```

#### Security & Policy
```zexus
protect(function, policy, mode) # Apply security policy
verify(condition)               # Runtime verification
restrict(value, constraints)    # Validate input
create_policy(rules)            # Create custom policy
enforce_policy(policy, value)   # Apply policy
```

#### Dependency Injection
```zexus
register_dependency(name, impl) # Register dependency
inject_dependency(name)         # Inject dependency
mock_dependency(name, mock)     # Mock for testing
test_mode(enabled)              # Enable/disable test mode
```

#### Concurrency & Channels
```zexus
channel<type> name              # Create typed channel
send(channel, value)            # Send to channel
receive(channel)                # Receive from channel
close_channel(channel)          # Close channel
atomic { }                      # Atomic operation block
```
emit(event, ...args)            # Emit event
require(condition, message)     # Assert with revert
assert(condition)               # Assert
balance_of(address)             # Get balance
transfer(to, amount)            # Transfer value
hash(data)                      # Hash data
keccak256(data)                 # Keccak-256 hash
signature(data, key)            # Sign data
verify_sig(data, sig, key)      # Verify signature
```

#### Renderer (UI)
```zexus
define_screen(name, props)      # Define UI screen
define_component(name, props)   # Define component
render_screen(name)             # Render screen
set_theme(theme)                # Set UI theme
create_canvas(width, height)    # Create drawing canvas
draw_line(canvas, x1, y1, x2, y2) # Draw line
draw_text(canvas, text, x, y)   # Draw text
```

#### Debug & Development
```zexus
debug(value)                    # Debug function (returns value)
debug value;                    # Debug statement (logs with metadata)
debug_log(message, context)     # Debug logging
debug_trace()                   # Stack trace
is_main()                       # Check if module is main entry point
exit_program(code)              # Exit with status code
module_info()                   # Get module metadata
```

#### Main Entry Point Features
```zexus
run(task_fn)                    # Execute task function
execute(fn)                     # Execute function immediately
is_main()                       # True if current module is main
exit_program(code)              # Exit with status code
on_start(fn)                    # Register startup handler
on_exit(fn)                     # Register cleanup handler
signal_handler(signal, fn)      # Handle OS signals
schedule(fn, delay)             # Schedule delayed execution
sleep(seconds)                  # Sleep for duration
daemonize(fn)                   # Run as background daemon
watch_and_reload(path)          # Auto-reload on file changes
get_module_name()               # Get current module name
get_module_path()               # Get current module path
list_imports()                  # List imported modules
get_exported_names()            # List exported names
```

#### Validation & Verification
```zexus
is_email(string)                # Validate email format
is_url(string)                  # Validate URL format
is_phone(string)                # Validate phone format
is_numeric(string)              # Check if numeric
is_alpha(string)                # Check if alphabetic
is_alphanumeric(string)         # Check if alphanumeric
matches_pattern(str, pattern)   # Regex pattern matching
password_strength(password)     # Check password strength
sanitize_input(text, type)      # Sanitize user input
validate_length(str, min, max)  # Validate string length
env_get(name)                   # Get environment variable
env_set(name, value)            # Set environment variable
env_exists(name)                # Check if env var exists
```

---

## 📖 Complete Keyword Reference

Zexus supports **130+ keywords** organized into functional categories:

### Core Language Keywords

#### Variable Declaration & Constants
- **`let`** - Mutable variable declaration (supports `=` and `:` syntax)
- **`const`** - Immutable constant declaration
- **`immutable`** - Mark variable as permanently immutable

#### Control Flow
- **`if`** / **`elif`** / **`else`** - Conditional execution
- **`while`** - While loop
- **`for`** / **`each`** / **`in`** - For-each iteration
- **`match`** / **`case`** / **`default`** - Pattern matching
- **`break`** / **`continue`** - Loop control
- **`return`** - Return from function

#### Functions & Actions
- **`action`** - Define action (Zexus function)
- **`function`** - Define function
- **`lambda`** - Anonymous function
- **`defer`** - Deferred cleanup execution

#### Types & Structures
- **`entity`** - Define data structure
- **`enum`** - Define enumeration
- **`protocol`** / **`interface`** - Define interface
- **`type_alias`** - Create type alias
- **`implements`** - Implement protocol

### Module System Keywords

- **`use`** - Import modules/symbols
- **`import`** - Alternative import syntax
- **`export`** - Export symbols
- **`module`** - Define module
- **`package`** - Define package/namespace
- **`from`** - Import from specific module
- **`external`** - Declare external function

### Security & Policy Keywords

#### Policy Enforcement
- **`protect`** - Apply security policy to function
- **`verify`** - Runtime verification with custom logic
- **`restrict`** - Input validation and constraints
- **`require`** - Assert condition (with tolerance blocks)
- **`assert`** - Always-check assertion

#### Access Control & Isolation
- **`seal`** - Make object immutable
- **`sandbox`** - Isolated execution environment
- **`audit`** - Compliance logging
- **`trail`** - Event tracking and audit trails
- **`capability`** - Define capability
- **`grant`** / **`revoke`** - Capability management

#### Data Validation
- **`validate`** - Schema validation
- **`sanitize`** - Input sanitization

### Blockchain Keywords

#### Smart Contracts
- **`contract`** - Define smart contract
- **`state`** - Mutable contract state
- **`ledger`** - Immutable ledger
- **`persistent`** / **`storage`** - Persistent storage
- **`tx`** - Transaction context
- **`gas`** - Gas tracking
- **`limit`** - Gas/resource limits

#### Cryptography
- **`hash`** - Cryptographic hashing
- **`signature`** - Digital signatures
- **`verify_sig`** - Signature verification

#### Contract Features
- **`emit`** - Emit event
- **`event`** - Event type
- **`revert`** - Revert transaction
- **`this`** - Current contract reference

### Modifiers

#### Visibility
- **`public`** - Public visibility (auto-export)
- **`private`** - Private/module-only visibility

#### Contract Modifiers
- **`pure`** / **`view`** - Read-only functions
- **`payable`** - Can receive value
- **`modifier`** - Define function modifier
- **`sealed`** - Prevent override
- **`secure`** - Security flag

### Concurrency & Async Keywords

- **`async`** - Async function
- **`await`** - Await promise/coroutine
- **`channel`** - Create channel
- **`send`** / **`receive`** - Channel operations
- **`atomic`** - Atomic operation block
- **`stream`** - Event streaming
- **`watch`** - Reactive state monitoring

### Error Handling Keywords

- **`try`** / **`catch`** - Exception handling
- **`throw`** - Throw exception
- **`finally`** - Cleanup block

### Performance Optimization Keywords

- **`native`** - Native C/C++ FFI
- **`inline`** - Function inlining hint
- **`gc`** - Garbage collection control
- **`buffer`** - Memory buffer operations
- **`simd`** - SIMD vector operations

### Advanced Language Features

- **`pattern`** - Pattern matching blocks
- **`exactly`** - Exact matching block
- **`embedded`** - Embed foreign language code
- **`using`** - Resource management

### Renderer/UI Keywords

- **`screen`** - Define UI screen
- **`component`** - Define UI component
- **`theme`** - Theme declaration
- **`canvas`** - Canvas for drawing
- **`graphics`** - Graphics context
- **`animation`** - Animation definition
- **`clock`** - Timing/clock
- **`color`** - Color definition

### Enterprise Features

- **`middleware`** - Request/response middleware
- **`auth`** - Authentication configuration
- **`throttle`** - Rate limiting
- **`cache`** - Caching directive
- **`inject`** - Dependency injection

### Special Keywords

- **`true`** / **`false`** - Boolean literals
- **`null`** - Null value
- **`map`** - Map/object literal

### Reserved Transaction Context

- **`TX`** - Global transaction context object with properties:
  - `TX.caller` - Transaction sender
  - `TX.value` - Sent value
  - `TX.timestamp` - Block timestamp
  - `TX.block_hash` - Current block hash
  - `TX.gas_used` - Gas consumed
  - `TX.gas_remaining` - Gas remaining
  - `TX.gas_limit` - Gas limit

[Complete keyword testing documentation →](docs/KEYWORD_TESTING_MASTER_LIST.md)

---

## 🎮 CLI Commands

### Zexus CLI (`zx`)

```bash
# Execution
zx run program.zx              # Run a program
zx run --debug program.zx      # Run with debugging
zx repl                        # Start interactive REPL

# Analysis
zx check program.zx            # Check syntax
zx validate program.zx         # Validate and auto-fix
zx ast program.zx              # Show AST
zx tokens program.zx           # Show tokens

# Project Management
zx init my-project             # Create new project
zx test                        # Run tests

# Configuration
zx debug on                    # Enable debugging
zx debug off                   # Disable debugging
```

**Advanced Options:**
```bash
# Syntax style
zx --syntax-style=universal run program.zx
zx --syntax-style=tolerable run program.zx
zx --syntax-style=auto run program.zx    # Auto-detect (default)

# Execution mode
zx --execution-mode=interpreter run program.zx
zx --execution-mode=compiler run program.zx
zx --execution-mode=auto run program.zx  # Auto-select (default)

# VM control
zx --use-vm run program.zx               # Use VM when beneficial (default)
zx --no-vm run program.zx                # Disable VM
```

### Package Manager (`zpm`)

```bash
# Initialize
zpm init                       # Create new project

# Install packages
zpm install                    # Install all from zexus.json
zpm install std                # Install specific package
zpm install web@0.2.0          # Install specific version
zpm install testing -D         # Install as dev dependency

# Manage packages
zpm list                       # List installed packages
zpm search <query>             # Search for packages
zpm uninstall <package>        # Remove a package
zpm clean                      # Remove zpm_modules/

# Publishing
zpm info                       # Show project info
zpm publish                    # Publish to registry
```

---

## 🏗️ Architecture

```
zexus-interpreter/
├── src/zexus/                  # Core interpreter
│   ├── lexer.py               # Tokenization
│   ├── parser/                # Parsing (multi-strategy)
│   │   ├── parser.py          # Main parser
│   │   └── ...                # Parser utilities
│   ├── evaluator/             # Evaluation engine
│   │   ├── core.py            # Main evaluator
│   │   ├── bytecode_compiler.py  # VM bytecode compiler
│   │   ├── expressions.py     # Expression evaluation
│   │   ├── statements.py      # Statement evaluation
│   │   └── functions.py       # Function handling & builtins
│   ├── vm/                    # Virtual Machine
│   │   ├── vm.py              # VM execution engine
│   │   ├── bytecode.py        # Bytecode definitions
│   │   └── jit.py             # JIT compilation
│   ├── compiler/              # Compiler frontend
│   │   ├── __init__.py        # Compiler main
│   │   ├── parser.py          # Production parser
│   │   ├── semantic.py        # Semantic analysis
│   │   └── bytecode.py        # Bytecode generation
│   ├── object.py              # Object system
│   ├── zexus_ast.py           # AST definitions
│   ├── persistence.py         # Persistent storage
│   ├── policy_engine.py       # Security policies
│   ├── dependency_injection.py # DI system
│   ├── blockchain/            # Blockchain features
│   │   ├── transaction.py     # Transaction handling
│   │   ├── crypto.py          # Cryptography
│   │   └── ...                # Other blockchain features
│   ├── security.py            # Security features
│   ├── module_manager.py      # Module system
│   └── ...                    # Other components
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── examples/              # Example programs
├── docs/                       # Documentation
│   ├── features/              # Feature docs
│   ├── guides/                # User guides
│   └── api/                   # API reference
├── syntaxes/                  # Syntax highlighting
├── zpm_modules/               # Installed packages
└── examples/                  # Example programs
```

### Execution Flow

```
Source Code (.zx)
       ↓
   [Lexer]  → Tokens
       ↓
   [Parser] → AST
       ↓
  [Evaluator] ←→ [Bytecode Compiler]
       ↓              ↓
 Direct Eval    [VM Execution]
       ↓              ↓
    Result  ←─────────┘
```

---

## 📖 Documentation

### Complete Documentation

- **[Feature Guide](docs/ADVANCED_FEATURES_IMPLEMENTATION.md)** - Complete feature reference
- **[Developer Guide](src/README.md)** - Internal architecture and API
- **[Documentation Index](docs/INDEX.md)** - All documentation organized
- **[Quick Start](docs/QUICK_START.md)** - Getting started tutorial
- **[Architecture](docs/ARCHITECTURE.md)** - System design
- **[Philosophy](docs/PHILOSOPHY.md)** - Design principles

### Keyword & Syntax Documentation

- **[Keyword Testing Master List](docs/KEYWORD_TESTING_MASTER_LIST.md)** - Complete keyword reference with 130+ keywords
- **[Blockchain Keywords](docs/BLOCKCHAIN_KEYWORDS.md)** - Smart contract keywords (implements, pure, view, payable, modifier, this, emit)
- **[Advanced Keywords](docs/keywords/ADVANCED_KEYWORDS.md)** - Advanced language features
- **[Modifiers](docs/MODIFIERS.md)** - Function and access modifiers

### Language Features by Category

#### Core Language
- **[LET](docs/keywords/LET.md)** - Variable declaration (multiple syntax styles)
- **[CONST](docs/keywords/CONST.md)** - Constant declaration
- **[ACTION/FUNCTION/LAMBDA/RETURN](docs/keywords/ACTION_FUNCTION_LAMBDA_RETURN.md)** - Function definitions
- **[IF/ELIF/ELSE](docs/keywords/IF_ELIF_ELSE.md)** - Conditional execution
- **[WHILE/FOR/EACH/IN](docs/keywords/WHILE_FOR_EACH_IN.md)** - Loops and iteration
- **[PRINT/DEBUG](docs/keywords/PRINT_DEBUG.md)** - Output and debugging

#### Module System
- **[MODULE_SYSTEM](docs/keywords/MODULE_SYSTEM.md)** - USE, IMPORT, EXPORT, MODULE, PACKAGE
- **[Main Entry Point](docs/MAIN_ENTRY_POINT.md)** - run, execute, is_main patterns

#### Async & Concurrency
- **[ASYNC/AWAIT](docs/keywords/ASYNC_AWAIT.md)** - Asynchronous programming
- **[ASYNC_CONCURRENCY](docs/keywords/ASYNC_CONCURRENCY.md)** - Channels, send, receive, atomic

#### Events & Reactive
- **[EVENTS_REACTIVE](docs/keywords/EVENTS_REACTIVE.md)** - Event system
- **[WATCH](docs/COMMAND_watch.md)** - Reactive state management

#### Security Features
- **[SECURITY](docs/keywords/SECURITY.md)** - Security features overview
- **[RESTRICT](docs/COMMAND_restrict.md)** - Input validation
- **[SANDBOX](docs/COMMAND_sandbox.md)** - Isolated execution
- **[AUDIT](docs/COMMAND_audit.md)** - Compliance logging
- **[TRAIL](docs/COMMAND_trail.md)** - Event tracking

#### Performance
- **[PERFORMANCE](docs/keywords/PERFORMANCE.md)** - Performance features
- **[NATIVE](docs/COMMAND_native.md)** - C/C++ FFI
- **[INLINE](docs/COMMAND_inline.md)** - Function inlining
- **[GC](docs/COMMAND_gc.md)** - Garbage collection control
- **[BUFFER](docs/COMMAND_buffer.md)** - Memory buffers
- **[SIMD](docs/COMMAND_simd.md)** - SIMD operations

#### Advanced Features
- **[DEFER](docs/COMMAND_defer.md)** - Deferred cleanup
- **[PATTERN](docs/COMMAND_pattern.md)** - Pattern matching
- **[ENUM](docs/COMMAND_enum.md)** - Enumerations
- **[STREAM](docs/COMMAND_stream.md)** - Event streaming

#### Blockchain & State
- **[BLOCKCHAIN_STATE](docs/keywords/BLOCKCHAIN_STATE.md)** - State management
- **[Error Handling](docs/keywords/ERROR_HANDLING.md)** - TRY/CATCH/REQUIRE/REVERT

#### Renderer/UI
- **[RENDERER_UI](docs/keywords/RENDERER_UI.md)** - UI and rendering system

### Specific Features

- **[VM Integration](VM_INTEGRATION_SUMMARY.md)** - Virtual Machine details
- **[VM Quick Reference](VM_QUICK_REFERENCE.md)** - VM API and usage
- **[Blockchain](docs/BLOCKCHAIN_FEATURES.md)** - Smart contracts and DApps
- **[Security](docs/SECURITY_FEATURES.md)** - Security features guide
- **[Concurrency](docs/CONCURRENCY.md)** - Async/await and channels
- **[Module System](docs/MODULE_SYSTEM.md)** - Import/export system
- **[Plugin System](docs/PLUGIN_SYSTEM.md)** - Extending Zexus
- **[ZPM Guide](docs/ZPM_GUIDE.md)** - Package manager
- **[Performance](docs/PERFORMANCE_FEATURES.md)** - Optimization features

### Command Documentation

Each advanced feature has detailed documentation:
- [PROTECT](docs/COMMAND_protect.md) - Security policies
- [WATCH](docs/COMMAND_watch.md) - Reactive state
- [RESTRICT](docs/COMMAND_restrict.md) - Input validation
- [SANDBOX](docs/COMMAND_sandbox.md) - Isolated execution
- [TRAIL](docs/COMMAND_trail.md) - Audit logging
- [DEFER](docs/COMMAND_defer.md) - Deferred execution
- [PATTERN](docs/COMMAND_pattern.md) - Pattern matching
- And many more in [docs/](docs/)

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

---

## 🧪 Testing

### Run Test Suite

```bash
# Unit tests
pytest tests/unit/

# Integration tests
cd tests/integration
zx run test_builtins_simple.zx
zx run test_advanced_features_complete.zx

# VM integration tests
python test_vm_integration.py
```

---

## 💡 Best Practices

### Code Organization

```zexus
# Use modules for organization
module UserManagement {
    export action createUser(name, email) { ... }
    export action deleteUser(id) { ... }
    
    private action hashPassword(password) { ... }
}

# Import only what you need
use {createUser, deleteUser} from "UserManagement"
```

### Security First

```zexus
# Always validate inputs
action processPayment(amount, recipient) {
    # Validate amount
    verify amount > 0, "Amount must be positive"
    restrict(amount, {
        type: "integer",
        range: [1, 1000000]
    })
    
    # Validate recipient
    verify is_email(recipient), "Invalid email"
    
    # Sanitize inputs
    let clean_recipient = sanitize(recipient, "email")
    
    # Apply security policies
    protect(processPayment, {
        auth_required: true,
        rate_limit: 10,
        log_access: true
    }, "strict")
}
```

### Error Handling

```zexus
# Use try-catch for error recovery
try {
    let data = file_read_json("config.json")
    process_config(data)
} catch (error) {
    # Fallback to defaults
    let data = get_default_config()
    debug_log("Using default config", {error: error})
}

# Use defer for cleanup
action process_file(path) {
    let handle = open_file(path)
    defer {
        close_file(handle)  # Always executes
    }
    
    # Process file...
    return result
}
```

### Performance Optimization

```zexus
# Use native for CPU-intensive tasks
native action calculate_hash(data: string) -> string {
    source: "crypto.cpp"
    function: "sha256_hash"
}

# Mark read-only functions as pure
action pure calculate_total(items) {
    return reduce(items, lambda(sum, item) { sum + item.price }, 0)
}

# Use inline for small frequently-called functions
inline action square(x) {
    return x * x
}
```

### Async Patterns

```zexus
# Use async/await for I/O operations
async action fetch_user_data(user_id) {
    let profile = await http_get("/users/" + user_id)
    let posts = await http_get("/users/" + user_id + "/posts")
    
    return {profile: profile, posts: posts}
}

# Use channels for producer-consumer patterns
channel<Task> work_queue

action producer() {
    for each task in pending_tasks {
        send(work_queue, task)
    }
    close_channel(work_queue)
}
```

### Testing with Dependency Injection

```zexus
# Production code
register_dependency("database", ProductionDB())

action saveUser(user) {
    inject database
    database.insert("users", user)
}

# Test code
test_mode(true)
mock_dependency("database", MockDB())
# Now saveUser() uses mocks
```

### Smart Contract Best Practices

```zexus
# Use modifiers for reusable guards
contract Vault {
    state owner
    
    modifier onlyOwner {
        require(TX.caller == owner, "Not authorized")
    }
    
    action payable withdraw(amount) modifier onlyOwner {
        require(balance >= amount, "Insufficient balance")
        transfer(TX.caller, amount)
        emit Withdrawal(TX.caller, amount)
    }
}
```

### Code Style Guidelines

1. **Naming**: `snake_case` for variables/functions, `PascalCase` for types
2. **Indentation**: 4 spaces (not tabs)
3. **Comments**: Use `#` for single-line comments
4. **Functions**: Keep under 50 lines when possible
5. **Error Messages**: Be descriptive and actionable

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Community Contributors** - Thank you for your support!
- **Open Source Libraries** - Built with Python, Click, and Rich
- **Inspiration** - From modern languages like Rust, Python, Solidity, TypeScript, and Go

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Zaidux/zexus-interpreter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Zaidux/zexus-interpreter/discussions)
- **Email**: zaidux@example.com

---

## 🗺️ Roadmap

### Completed ✅
- [x] Core interpreter with hybrid execution
- [x] VM-accelerated bytecode execution with JIT compilation
- [x] 130+ language keywords fully implemented and tested
- [x] Policy-as-code (PROTECT/VERIFY/RESTRICT)
- [x] Enhanced VERIFY with validation builtins (email, URL, phone, patterns)
- [x] Persistent memory management with leak detection
- [x] Dependency injection system with auto-container creation
- [x] Reactive state (WATCH) for automatic change reactions
- [x] Blockchain primitives and smart contracts
- [x] Blockchain modifiers (pure, view, payable, modifier, this, implements, emit)
- [x] Module system (USE, EXPORT, MODULE, PACKAGE) with access control
- [x] Package manager (ZPM) for dependency management
- [x] 100+ built-in functions across all categories
- [x] Advanced types (entities, contracts, protocols, enums, type_alias)
- [x] Security features (sandbox, seal, trail, audit, capability, grant/revoke)
- [x] Concurrency primitives (async/await with Promises, channels, send/receive, atomic)
- [x] Main entry point system (run, execute, is_main, exit_program, on_start/on_exit)
- [x] Enterprise features (middleware, auth, throttle, cache, inject)
- [x] UI rendering system (screen, component, theme, canvas)
- [x] Performance optimization (native, inline, gc, buffer, simd)
- [x] Advanced features (defer, pattern, stream, exactly, embedded)
- [x] Dual-mode DEBUG (function and statement modes)
- [x] Multiple syntax styles (`:` and `=` for assignments)
- [x] Tolerance blocks for enhanced REQUIRE
- [x] Comprehensive test suite (1175+ tests)

### In Progress 🚧
- [ ] VS Code extension with full IntelliSense
- [ ] Language Server Protocol (LSP)
- [ ] Standard library expansion (fs, http, json, datetime)
- [ ] Debugger integration
- [ ] Performance profiling tools

### Planned 🎯
- [ ] WASM compilation target
- [ ] JIT compilation for hot paths
- [ ] Official package registry
- [ ] CI/CD templates
- [ ] Docker images
- [ ] Production monitoring tools

### Future Enhancements 🚀
- [ ] GPU acceleration for SIMD operations
- [ ] Distributed computing primitives
- [ ] Native mobile app support
- [ ] WebAssembly interop
- [ ] Advanced static analysis

---

## 📊 Project Stats

- **Language**: Python 3.8+
- **Version**: 0.1.3 (Alpha)
- **Lines of Code**: ~50,000+
- **Keywords**: 130+ language keywords
- **Built-in Functions**: 100+ built-in functions
- **Documentation Pages**: 100+
- **Test Cases**: 1175+ comprehensive tests
- **Features**: 100+ language features
- **Supported Platforms**: Linux, macOS, Windows

---

## ❓ Getting Help & Troubleshooting

### Common Issues

#### "Identifier not found" errors
- Check variable spelling and case sensitivity
- Ensure variable is declared in current or parent scope
- Remember: Zexus uses function-level scoping (not block-level)
- Variables declared in blocks persist in function scope

#### Import/Module errors
- Use `use {symbol} from "module"` syntax for imports
- Check that module file exists and has `.zx` extension
- Ensure exported symbols are marked with `export` keyword
- Use `zpm install` to install package dependencies

#### Syntax errors
- Zexus supports multiple syntax styles: `let x = 5` or `let x : 5`
- Ensure proper braces `{}` for blocks
- Use `;` for statement termination (optional in some contexts)
- Check for unmatched parentheses, brackets, or braces

#### Performance issues
- Enable VM execution for compute-heavy code (default: auto)
- Use `--use-vm` flag for explicit VM mode
- Consider using `native` keyword for C/C++ FFI
- Profile with `memory_stats()` to check for leaks

#### Blockchain/Contract issues
- Remember `TX` is a global context object (uppercase)
- Use `persistent storage` for contract state
- Mark value-receiving functions as `payable`
- Use `pure` or `view` for read-only functions

### Documentation Quick Links

- **Beginner**: Start with [Quick Start Guide](docs/QUICK_START.md)
- **Keywords**: See [Keyword Master List](docs/KEYWORD_TESTING_MASTER_LIST.md)
- **Examples**: Check [examples/](examples/) directory
- **API Reference**: Browse [docs/](docs/) for detailed docs
- **Advanced**: Read [Advanced Features Guide](docs/ADVANCED_FEATURES_IMPLEMENTATION.md)

### Debug Tools

```zexus
# Enable detailed debugging
debug myVariable;              # Logs with context

# Check execution context
print(is_main())              # Am I the main module?
print(get_module_name())      # Current module name
print(module_info())          # Module metadata

# Memory debugging
track_memory()                # Enable tracking
print(memory_stats())         # Check for leaks

# AST/Token inspection
# Run: zx ast program.zx
# Run: zx tokens program.zx
```

### Getting Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Zaidux/zexus-interpreter/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/Zaidux/zexus-interpreter/discussions)
- **Documentation**: [Browse complete docs](docs/)
- **Examples**: [See working code samples](examples/)

### Community & Ecosystem

#### Official Resources
- **GitHub Repository**: [Zaidux/zexus-interpreter](https://github.com/Zaidux/zexus-interpreter)
- **Documentation Site**: [docs/](docs/)
- **VS Code Extension**: [.vscode/extensions/zexus-language/](.vscode/extensions/zexus-language/)
- **Syntax Highlighting**: [syntaxes/](syntaxes/)

#### Standard Library Packages
- **zexus-blockchain**: Blockchain utilities and helpers
- **zexus-network**: HTTP, WebSocket, and networking
- **zexus-math**: Advanced mathematical operations
- **zexus-stdlib**: Standard library modules

Install packages with ZPM:
```bash
zpm install zexus-blockchain
zpm install zexus-network
zpm install zexus-math
```

#### Learning Resources
- **Quick Start**: [docs/QUICK_START.md](docs/QUICK_START.md)
- **Examples Directory**: [examples/](examples/)
- **Test Suite**: [tests/](tests/) - 1175+ working examples
- **Keyword Testing**: [docs/KEYWORD_TESTING_MASTER_LIST.md](docs/KEYWORD_TESTING_MASTER_LIST.md)
- **Feature Guides**: [docs/features/](docs/features/)

#### Development Tools
- **CLI**: `zx` for running programs
- **Package Manager**: `zpm` for dependencies
- **REPL**: Interactive shell with `zx repl`
- **AST Inspector**: `zx ast program.zx`
- **Token Viewer**: `zx tokens program.zx`
- **Validator**: `zx validate program.zx`

---

<div align="center">

**Made with ❤️ by the Zexus Team**

[⭐ Star us on GitHub](https://github.com/Zaidux/zexus-interpreter) | [📖 Read the Docs](docs/) | [🐛 Report Bug](https://github.com/Zaidux/zexus-interpreter/issues) | [💡 Request Feature](https://github.com/Zaidux/zexus-interpreter/issues/new)

</div>
