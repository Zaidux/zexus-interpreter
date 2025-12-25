# Zexus Documentation Index

Welcome to the Zexus Programming Language documentation! This index helps you navigate all available documentation.

## 🚀 Getting Started

- **[README](../README.md)** - Project overview, installation, quick start
- **[Installation Guide](#installation)** - Detailed installation instructions
- **[Quick Start Tutorial](#quick-start)** - Your first Zexus program
- **[CLI Reference](#cli-reference)** - Command-line interface guide

## 📚 Core Documentation

### Language Features
- **[Advanced Features Implementation](keywords/features/ADVANCED_FEATURES_IMPLEMENTATION.md)** - Complete feature reference including:
  - Persistent Memory Management
  - Policy-as-Code (PROTECT/VERIFY/RESTRICT)
  - Dependency Injection & Mocking
  - All built-in functions
  
- **[Blockchain Implementation](keywords/features/BLOCKCHAIN_IMPLEMENTATION.md)** - Smart contracts, transactions, events
- **[WATCH Feature](WATCH_FEATURE.md)** - Reactive state management
- **[Plugin System Guide](guides/PLUGIN_SYSTEM_GUIDE.md)** - Extending Zexus

### Development Guides
- **[Developer Guide](../src/README.md)** - Internal architecture, module structure, testing
- **[Security Best Practices](../src/README.md#security-best-practices-for-zexus)** - Defense-in-depth security model

### Status & History
- **[Status](STATUS.md)** - Current project status
- **[Phase Summaries](keywords/features/)** - Development phase documentation

## 🎯 Quick Reference

### Installation

```bash
# From PyPI (recommended)
pip install zexus

# From source
git clone https://github.com/Zaidux/zexus-interpreter.git
cd zexus-interpreter
pip install -e .
```

### CLI Reference

```bash
# Run programs
zx run program.zx                    # Execute a program
zx run --zexus                       # Show all commands
zx run --syntax-style=universal      # Use strict syntax
zx run --execution-mode=compiler     # Force compiler mode

# Development tools
zx check program.zx                  # Check syntax
zx validate program.zx               # Auto-fix syntax
zx ast program.zx                    # Show AST
zx tokens program.zx                 # Show tokens

# Interactive
zx repl                              # Start REPL
zx init my-app                       # Create project

# Debug
zx debug on                          # Enable debugging
zx debug off                         # Disable debugging
```

### Language Syntax

#### Variables
```zexus
let count = 42                       # Mutable variable
const PI = 3.14159                   # Immutable constant
```

#### Functions
```zexus
action greet(name: string) -> string {
    return "Hello, " + name
}
```

#### Entities & Contracts
```zexus
entity User {
    name: string,
    age: integer
}

contract Token {
    persistent storage balances: Map<Address, integer>
    
    action transfer(to: Address, amount: integer) {
        # Implementation
    }
}
```

#### Security
```zexus
protect(sensitive_function, {
    rate_limit: 100,
    auth_required: true
}, "strict")

verify(user.is_authenticated)

restrict(amount, {range: [0, 1000]})
```

#### Reactive State
```zexus
watch variable {
    # Callback when variable changes
}
```

#### Dependency Injection
```zexus
register_dependency("db", ProductionDB())
inject db
```

## 📖 Complete Feature List

### Core Language
- Variables: `let`, `const`
- Functions: `action`
- Types: Integer, Float, String, Boolean, List, Map, Set
- Control Flow: `if`, `else`, `for`, `while`, `match`
- Operators: Arithmetic, logical, comparison, pipeline (`|>`)

### Advanced Features
- **Entities**: Structured data types
- **Contracts**: Persistent smart contracts
- **Modules**: Import/export system with access control
- **Pattern Matching**: Powerful `match` expressions
- **Async/Await**: Built-in concurrency
- **WATCH**: Reactive state management
- **PROTECT**: Policy-as-code security
- **INJECT**: Dependency injection
- **Persistent Storage**: Cross-session data

### Blockchain
- Transactions
- Events & Emissions
- Address types
- Balance queries
- Smart contract deployment

### Security
- Authentication requirements
- Rate limiting
- HTTPS enforcement
- IP allowlists/blocklists
- Input validation & sanitization
- Audit logging

### Built-in Functions (50+)

**I/O**: `print`, `println`, `input`
**Type Conversion**: `string`, `int`, `float`, `bool`
**Collections**: `len`, `list`, `map`, `set`, `range`
**Functional**: `filter`, `reduce`, `sort`, `reverse`
**String**: `join`, `split`, `replace`, `uppercase`, `lowercase`, `trim`
**Math**: `abs`, `ceil`, `floor`, `round`, `min`, `max`, `sum`, `sqrt`, `pow`
**Memory**: `persist_set`, `persist_get`, `persist_clear`, `track_memory`
**Policy**: `create_policy`, `enforce_policy`, `verify_condition`
**DI**: `register_dependency`, `inject_dependency`, `mock_dependency`, `test_mode`
**Blockchain**: `transaction`, `emit`, `require`, `assert`, `balance`, `transfer`

## 🔧 Development

### Project Structure
```
zexus-interpreter/
├── src/zexus/          # Core implementation
├── tests/              # Test suite
├── docs/               # Documentation (you are here)
├── syntaxes/           # Syntax highlighting
└── scripts/            # Development scripts
```

### Running Tests
```bash
cd tests/integration
zx run test_builtins_simple.zx
zx run test_advanced_features_complete.zx
```

### Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Zaidux/zexus-interpreter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Zaidux/zexus-interpreter/discussions)

## 📄 License

MIT License - see [LICENSE](../LICENSE) file for details.

---

**Last Updated**: December 13, 2025  
**Version**: 1.5.0
