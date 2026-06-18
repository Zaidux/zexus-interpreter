# Contributing to Zexus

Thank you for your interest in contributing to the Zexus programming language! This guide will get you started.

## Getting Started

### Prerequisites

- **Python 3.8+** — for the interpreter and compiler
- **Rust** (optional) — for the high-performance VM core (`rust_core/`)
- **Node.js** (optional) — for the VS Code extension

### Setup

```bash
git clone https://github.com/Zaidux/zexus-interpreter.git
cd zexus-interpreter

# Install Python dependencies
pip install -r requirements.txt

# Run the interpreter
python zexus.py examples/hello.zx

# Run in REPL mode
python zexus.py --repl
```

### Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/keyword_tests/ -v
python -m pytest tests/edge_cases/ -v
python -m pytest tests/golden/ -v
python -m pytest tests/builtin_modules/ -v
```

## Project Structure

```
zexus-interpreter/
├── src/                   # Core interpreter source
│   ├── lexer/             # Tokenization
│   ├── parser/            # AST construction
│   ├── compiler/          # Bytecode compilation
│   ├── vm/                # Virtual machine execution
│   ├── interpreter/       # Tree-walking interpreter
│   ├── runtime/           # Runtime environment
│   └── stdlib/            # Standard library modules
├── rust_core/             # Rust VM implementation (performance-critical)
├── blockchain_test/       # Blockchain feature tests
├── docs/                  # Documentation (see docs/INDEX.md)
│   ├── keywords/          # Per-keyword documentation
│   ├── packages/          # Package documentation
│   ├── stdlib/            # Standard library docs
│   └── profiler/          # Profiler documentation
├── examples/              # Example Zexus programs
├── tests/                 # Test suite
├── zpm_modules/           # Zexus Package Manager modules
├── vscode-extension/      # VS Code language support
├── scripts/               # Build and utility scripts
└── zexus.py               # Main entry point
```

## Areas for Contribution

### Language Features
- New keywords and syntax constructs
- Standard library functions and modules
- Pattern matching improvements
- Type system enhancements

### Performance
- VM bytecode optimizations
- Rust core improvements
- Memory management enhancements
- Profiler accuracy

### Blockchain
- Smart contract features
- New blockchain primitives
- Security auditing tools

### Tooling
- VS Code extension improvements
- Package manager (ZPM) features
- CLI improvements
- Debugger support

### Documentation
- Keyword documentation in `docs/keywords/`
- Examples for complex features
- Tutorial writing
- Translation

### Testing
- Edge case tests
- Golden tests (expected output verification)
- Benchmark tests
- Fuzzing

## Branch Naming

- `feat/<description>` — new language features or tools
- `fix/<description>` — bug fixes
- `docs/<description>` — documentation
- `perf/<description>` — performance improvements
- `test/<description>` — test additions

## Resources

- [Quick Start Guide](QUICK_START.md) — get running in 5 minutes
- [Language Guide](ZEXUS_GUIDE.md) — learn the language
- [Language Rules](ZEXUS_RULES.md) — formal specification
- [Documentation Index](docs/INDEX.md) — all docs in one place
- [Changelog](CHANGELOG.md) — version history

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
