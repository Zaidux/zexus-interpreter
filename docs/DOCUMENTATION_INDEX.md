# Zexus Documentation Index

Complete documentation for the Zexus programming language.

## 🚀 Getting Started

- [README](../README.md) - Main project overview
- [Quick Start Guide](QUICK_START.md) - Get up and running quickly
- [Installation Guide](../README.md#installation) - How to install Zexus

## 📚 Core Documentation

### Language Features
- [Architecture](ARCHITECTURE.md) - System design and components
- [Philosophy](PHILOSOPHY.md) - Design principles and goals
- [Advanced Features](features/ADVANCED_FEATURES_IMPLEMENTATION.md) - Complete feature reference
- [Keyword Reference](KEYWORD_TESTING_MASTER_LIST.md) - All 130+ keywords documented
- [Module System](MODULE_SYSTEM.md) - Import/export and package management
- [Error Reporting](ERROR_REPORTING.md) - World-class error messages

### Development Tools

#### VS Code Extension
- [Extension README](../vscode-extension/README.md) - Full-featured IDE support
- Features: IntelliSense, debugging, profiling, syntax highlighting

#### Language Server Protocol (LSP)
- [LSP Guide](lsp/LSP_GUIDE.md) - Complete LSP documentation
- Features: Completion, hover, go-to-definition, diagnostics
- Real-time code intelligence

#### Performance Profiling
- [Profiler Guide](profiler/PROFILER_GUIDE.md) - Performance analysis tools
- Execution time profiling
- Memory profiling
- Hotspot detection

### Standard Library
- [Standard Library Overview](stdlib/README.md) - All stdlib modules
- [fs Module](stdlib/FS_MODULE.md) - File system operations (coming soon)
- [http Module](stdlib/HTTP_MODULE.md) - HTTP client (coming soon)
- [json Module](stdlib/JSON_MODULE.md) - JSON parsing/serialization (coming soon)
- [datetime Module](stdlib/DATETIME_MODULE.md) - Date/time operations (coming soon)
- [crypto Module](stdlib/CRYPTO_MODULE.md) - Cryptographic functions
- [blockchain Module](stdlib/BLOCKCHAIN_MODULE.md) - Blockchain utilities

## 🔑 Language Features by Category

### Core Language
- [Variables & Constants](keywords/LET.md) - let, const, immutable
- [Functions](keywords/ACTION_FUNCTION_LAMBDA_RETURN.md) - action, function, lambda
- [Control Flow](keywords/IF_ELIF_ELSE.md) - if, elif, else
- [Loops](keywords/WHILE_FOR_EACH_IN.md) - while, for, each, in
- [I/O](keywords/PRINT_DEBUG.md) - print, debug, log

### Advanced Features
- [Pattern Matching](keywords/DATA.md) - match, case, pattern
- [Data Classes](keywords/DATA.md) - Generic types and inheritance
- [Error Handling](keywords/ERROR_HANDLING.md) - try, catch, finally, continue
- [Async/Await](keywords/ASYNC_AWAIT.md) - Asynchronous programming
- [Modules](keywords/MODULE_SYSTEM.md) - use, import, export

### Security & Policy
- [Security Features](SECURITY_FEATURES.md) - Overview of security
- [PROTECT](keywords/COMMAND_protect.md) - Policy-as-code
- [VERIFY](keywords/COMMAND_verify.md) - Runtime verification
- [RESTRICT](keywords/COMMAND_restrict.md) - Input validation
- [SANDBOX](keywords/COMMAND_sandbox.md) - Isolated execution

### Blockchain
- [Blockchain Features](BLOCKCHAIN_FEATURES.md) - Smart contracts and DApps
- [Blockchain Keywords](BLOCKCHAIN_KEYWORDS.md) - contract, pure, view, payable
- [Blockchain State](keywords/BLOCKCHAIN_STATE.md) - State management

### Performance
- [VM Integration](../VM_INTEGRATION_STATUS.md) - Virtual machine details
- [VM Quick Reference](../VM_QUICK_REFERENCE.md) - VM API and usage
- [Performance Features](PERFORMANCE_FEATURES.md) - Optimization tools
- [NATIVE](keywords/COMMAND_native.md) - C/C++ FFI
- [INLINE](keywords/COMMAND_inline.md) - Function inlining

### Concurrency
- [Concurrency Guide](CONCURRENCY.md) - Async/await and channels
- [ASYNC/AWAIT](keywords/ASYNC_AWAIT.md) - Asynchronous functions
- [Channels](keywords/ASYNC_CONCURRENCY.md) - Channel communication
- [ATOMIC](keywords/ASYNC_CONCURRENCY.md) - Atomic operations

### Other Features
- [Watch/Reactive](keywords/COMMAND_watch.md) - Reactive state management
- [Dependency Injection](keywords/ADVANCED_KEYWORDS.md) - DI system
- [Persistence](PERSISTENCE.md) - Cross-session data storage
- [Plugin System](PLUGIN_SYSTEM.md) - Extending Zexus
- [Renderer/UI](keywords/RENDERER_UI.md) - UI and rendering

## 📖 Guides & Tutorials

### User Guides
- [Quick Start](QUICK_START.md) - Getting started tutorial
- [Main Entry Point](MAIN_ENTRY_POINT.md) - Program lifecycle management
- [Watch Feature](WATCH_FEATURE.md) - Reactive programming
- [VERIFY Enhancement Guide](VERIFY_ENHANCEMENT_GUIDE.md) - Advanced validation

### Developer Guides
- [Developer Guide](../src/README.md) - Internal architecture and API
- [Parser Optimization](PARSER_OPTIMIZATION_GUIDE.md) - Parser performance
- [Plugin System Guide](guides/PLUGIN_SYSTEM_GUIDE.md) - Creating plugins
- [Plugin Quick Reference](PLUGIN_QUICK_REFERENCE.md) - Plugin API

## 🔧 Technical Documentation

### Architecture
- [Architecture](ARCHITECTURE.md) - System design
- [Execution Flow](../README.md#execution-flow) - How code is executed
- [Hybrid Orchestrator](../src/README.md) - Interpreter/compiler selection

### Parser & Lexer
- [Parser Optimization](PARSER_OPTIMIZATION_GUIDE.md) - Parser internals
- [Strategy Recovery](../src/zexus/strategy_recovery.py) - Error recovery

### Security
- [Security Features](SECURITY_FEATURES.md) - Security overview
- [Capability System](../src/zexus/capability_system.py) - Capabilities
- [Policy Engine](../src/zexus/policy_engine.py) - Policy enforcement

## 📦 Package Management

- [ZPM Guide](ZPM_GUIDE.md) - Zexus Package Manager
- [Module System](MODULE_SYSTEM.md) - Importing and exporting

## 🧪 Testing

- [Testing Guide](../tests/README.md) - Running tests (if exists)
- [Keyword Testing](KEYWORD_TESTING_MASTER_LIST.md) - Comprehensive keyword tests

## 🤝 Contributing

- [Contributing Guide](CONTRIBUTING.md) - How to contribute (if exists)
- [Code of Conduct](CODE_OF_CONDUCT.md) - Community guidelines (if exists)

## 📝 Changelog & Roadmap

- [Changelog](../CHANGELOG.md) - Version history
- [Roadmap](../README.md#roadmap) - Future plans
- [Future Implementations](future_implentations.md) - Upcoming features

## 🆘 Help & Support

- [Getting Help](../README.md#getting-help--troubleshooting) - Troubleshooting
- [FAQ](FAQ.md) - Frequently asked questions (if exists)
- [GitHub Issues](https://github.com/Zaidux/zexus-interpreter/issues) - Report bugs
- [Discussions](https://github.com/Zaidux/zexus-interpreter/discussions) - Ask questions

## 📊 Reference

### Complete Keyword List
- [Keyword Master List](KEYWORD_TESTING_MASTER_LIST.md) - All 130+ keywords
- [Modifiers](MODIFIERS.md) - Function and access modifiers
- [Advanced Keywords](keywords/ADVANCED_KEYWORDS.md) - Advanced features

### Built-in Functions
- [Built-in Functions](../README.md#built-in-functions-100) - All 100+ builtins
- [Main Entry Point Functions](MAIN_ENTRY_POINT.md) - Lifecycle functions

## 🔗 External Links

- [GitHub Repository](https://github.com/Zaidux/zexus-interpreter)
- [Issue Tracker](https://github.com/Zaidux/zexus-interpreter/issues)
- [Discussions](https://github.com/Zaidux/zexus-interpreter/discussions)

---

## Quick Navigation by Topic

### I want to...
- **Get started**: [Quick Start](QUICK_START.md)
- **Learn the basics**: [README](../README.md)
- **Use VS Code**: [VS Code Extension](../vscode-extension/README.md)
- **Profile my code**: [Profiler Guide](profiler/PROFILER_GUIDE.md)
- **Use the standard library**: [Stdlib Overview](stdlib/README.md)
- **Build a smart contract**: [Blockchain Features](BLOCKCHAIN_FEATURES.md)
- **Secure my app**: [Security Features](SECURITY_FEATURES.md)
- **Optimize performance**: [VM Integration](../VM_INTEGRATION_STATUS.md)
- **Contribute**: [Architecture](ARCHITECTURE.md) and [Developer Guide](../src/README.md)

---

Last updated: 2025-12-25
