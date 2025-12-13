# Zexus Programming Language

<div align="center">

![Zexus Logo](https://img.shields.io/badge/Zexus-v0.1.0-FF6B35?style=for-the-badge)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![GitHub](https://img.shields.io/badge/GitHub-Zaidux/zexus--interpreter-181717?style=for-the-badge&logo=github)](https://github.com/Zaidux/zexus-interpreter)

**A modern, security-first programming language with built-in blockchain support, advanced memory management, and policy-as-code**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 🎯 What is Zexus?

Zexus is a next-generation programming language designed for security-conscious developers who need:
- **Policy-as-code** for declarative security rules
- **Built-in blockchain** primitives and smart contracts
- **Persistent memory** with automatic leak detection
- **Dependency injection** and module mocking for testing
- **Reactive state management** with WATCH
- **Hybrid execution** (interpreter + compiler)
- **Flexible syntax** supporting both universal and tolerant styles

## ✨ Features

### 🔐 Security & Policy-as-Code
```zexus
# Define security policies declaratively
protect(transfer_funds, {
    rate_limit: 100,
    auth_required: true,
    require_https: true,
    allowed_ips: ["10.0.0.0/8"]
}, "strict")

# Runtime verification
verify(user.is_authenticated and user.has_permission("transfer"))

# Data constraints
restrict(amount, {
    range: [0, 10000],
    type: "integer"
})
```

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
- **Hybrid execution**: Auto-selects interpreter or compiler
- **Type safety**: Strong typing with inference
- **Pattern matching**: Powerful match expressions
- **Async/await**: Built-in concurrency
- **Module system**: Import/export with access control
- **Rich built-ins**: 50+ built-in functions

## 📦 Installation

### Quick Install (Recommended)

```bash
pip install zexus
```

**After installation**, you'll see documentation links pointing to the comprehensive feature guide.

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

## 🚀 Quick Start

### 1. Create Your First Program

```bash
# Create a new project
zx init my-app

# Or create a file manually
cat > hello.zx << 'EOF'
# Simple Zexus program
let name = "World"
print("Hello, " + name + "!")
EOF
```

### 2. Run It

```bash
zx run hello.zx
```

### 3. Explore More Examples

```zexus
# Variables and types
let count = 42
const PI = 3.14159
let items = [1, 2, 3, 4, 5]
let user = {name: "Alice", age: 30}

# Functions
action greet(name: string) -> string {
    return "Hello, " + name
}

# Loops and conditionals
for each item in items {
    if item > 2 {
        print(item)
    }
}

# Pattern matching
match response_code {
    case 200: print("Success")
    case 404: print("Not Found")
    case 500: print("Server Error")
    default: print("Unknown")
}
```

## 📚 Documentation

### Core Docs
- **[Feature Guide](docs/features/ADVANCED_FEATURES_IMPLEMENTATION.md)** - Complete feature reference
- **[Developer Guide](src/README.md)** - Internal architecture and API
- **[Plugin System](docs/guides/PLUGIN_SYSTEM_GUIDE.md)** - Extending Zexus
- **[Blockchain Guide](docs/features/BLOCKCHAIN_IMPLEMENTATION.md)** - Smart contracts and DApps

### Language Reference
- **Syntax**: Both universal (`{}`) and tolerant (`:`) styles supported
- **Types**: Integer, Float, String, Boolean, List, Map, Set, Entity, Contract
- **Keywords**: `let`, `const`, `action`, `entity`, `contract`, `if`, `for`, `while`, `match`, `watch`, `protect`, `inject`
- **Built-ins**: [50+ functions](docs/api/builtins.md) including blockchain, security, and persistence

### Quick Links
- **[Installation Guide](#-installation)**
- **[Examples](#-examples)**
- **[CLI Commands](#-cli-commands)**
- **[Contributing](docs/CONTRIBUTING.md)**

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

### Example 4: Dependency Injection

```zexus
# Production code
register_dependency("logger", FileLogger("/var/log/app.log"))
register_dependency("database", PostgresDB("localhost:5432"))

inject logger
inject database

action save_user(user: Entity) {
    logger.info("Saving user: " + user.name)
    database.insert("users", user)
}

# Test code
test_mode(true)
mock_dependency("logger", MockLogger())
mock_dependency("database", MockDB())

# Now save_user uses mocks automatically
save_user({name: "Test User", email: "test@example.com"})
```

### Example 5: Package Management

```bash
# Initialize project
zpm init

# Install packages
zpm install std crypto web

# Use in code
```

```zexus
use {map, filter} from "std"
use {encrypt, decrypt} from "crypto"
use {Server, Router} from "web"

let app = Server()
app.listen(8080)
```

## 🎮 CLI Commands

### Zexus CLI (`zx`)

```bash
# Run a program
zx run program.zx

# Check syntax
zx check program.zx

# Validate and auto-fix
zx validate program.zx

# Show AST
zx ast program.zx

# Show tokens
zx tokens program.zx

# Start REPL
zx repl

# Initialize project
zx init my-project

# Debug mode
zx debug on
zx run program.zx
zx debug off

# Help and options
zx --help
zx run --help
```

### Package Manager (`zpm`)

```bash
# Initialize project
zpm init

# Install packages
zpm install              # Install all from zexus.json
zpm install std          # Install specific package
zpm install web@0.2.0    # Install specific version
zpm install testing -D   # Install as dev dependency

# Manage packages
zpm list                 # List installed packages
zpm search <query>       # Search for packages
zpm uninstall <package>  # Remove a package
zpm clean                # Remove zpm_modules/

# Publishing
zpm info                 # Show project info
zpm publish              # Publish to registry

# Help
zpm --help
```

See **[ZPM Guide](docs/ZPM_GUIDE.md)** for complete package manager documentation.

### Advanced Options

```bash
# Syntax style
zx --syntax-style=universal run program.zx
zx --syntax-style=tolerable run program.zx
zx --syntax-style=auto run program.zx  # Auto-detect (default)

# Execution mode
zx --execution-mode=interpreter run program.zx
zx --execution-mode=compiler run program.zx
zx --execution-mode=auto run program.zx  # Auto-select (default)

# Advanced parsing
zx --advanced-parsing run program.zx  # Enable multi-strategy parsing

# Debug
zx --debug run program.zx  # Enable debug logging
```

## 🏗️ Architecture

```
zexus-interpreter/
├── src/zexus/              # Core interpreter
│   ├── lexer.py           # Tokenization
│   ├── parser/            # Parsing (traditional + advanced)
│   ├── evaluator/         # Evaluation engine
│   ├── object.py          # Object system
│   ├── persistence.py     # Memory persistence
│   ├── policy_engine.py   # Security policies
│   ├── dependency_injection.py  # DI system
│   └── blockchain/        # Blockchain features
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── examples/          # Example programs
├── docs/                   # Documentation
│   ├── features/          # Feature docs
│   ├── guides/            # User guides
│   └── api/               # API reference
├── syntaxes/              # Syntax highlighting
└── scripts/               # Development scripts
```

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Community Contributors** - Thank you for your support!
- **Open Source Libraries** - Built with Python, Click, and Rich
- **Inspiration** - From modern languages like Rust, Python, Solidity, and TypeScript

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Zaidux/zexus-interpreter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Zaidux/zexus-interpreter/discussions)
- **Email**: zaidux@example.com

## 🗺️ Roadmap

- [x] Core interpreter with hybrid execution
- [x] Policy-as-code (PROTECT/VERIFY/RESTRICT)
- [x] Persistent memory management
- [x] Dependency injection system
- [x] Reactive state (WATCH)
- [x] Blockchain primitives
- [ ] VS Code extension with full IntelliSense
- [ ] Standard library expansion
- [ ] Package manager (ZPM)
- [ ] WASM compilation target
- [ ] Language Server Protocol (LSP)
- [ ] Debugger integration

---

<div align="center">

**Made with ❤️ by the Zexus Team**

[⭐ Star us on GitHub](https://github.com/Zaidux/zexus-interpreter) | [📖 Read the Docs](docs/) | [🐛 Report Bug](https://github.com/Zaidux/zexus-interpreter/issues)

</div>
