```markdown
# ~/zexus-interpreter/README.md
# Zexus Programming Language

<p align="center">
  <strong>🚀 Declarative, intent-based programming for the modern web</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/zexus-lang/">
    <img src="https://img.shields.io/badge/version-0.1.0-00FF80" alt="Version">
  </a>
  <a href="https://python.org/">
    <img src="https://img.shields.io/badge/python-3.8+-blue" alt="Python version">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  </a>
</p>

## ✨ Features

- **Declarative Syntax** - Describe what you want, not how to do it
- **Multi-paradigm** - Functional, imperative, and declarative programming  
- **Built-in Blockchain** - Native Ziver Chain integration
- **Modern Tooling** - REPL, syntax checking, beautiful CLI

## 🚀 Quick Start

### Installation

```bash
# Install from current directory
pip install .

# Or install in development mode
pip install -e .
```

Usage

```bash
# Run a Zexus program
zx run examples/hello_world.zx

# Start REPL
zx repl

# Check syntax
zx check my_program.zx

# Create new project
zx init my-project
```

Hello World

Create hello.zx:

```zexus
let message = "Hello, Zexus!"
print message

action greet(name: text):
    print "Hello, " + name + "!"

greet("World")
```

📖 Examples

See the examples/ directory for:

· hello_world.zx - Basic syntax
· blockchain_demo.zx - Ziver Chain integration
· web_app.zx - UI development

🛠️ Development

```bash
# Run tests
pytest tests/

# Debug parser
zx ast my_file.zx

# See tokens
zx tokens my_file.zx
```

📄 License

Apache 2.0 - See LICENSE file for details.

```
