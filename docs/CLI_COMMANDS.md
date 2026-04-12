# Zexus CLI Commands Reference

All available commands for the `zx` command-line tool (v1.8.4).

---

## Installation

```bash
pip install zexus
```

Or install from source:

```bash
git clone https://github.com/Zaidux/zexus-interpreter
cd zexus-interpreter
pip install -e ".[dev]"
```

---

## Commands

### `zx run <file>`

Run a Zexus program.

```bash
zx run program.zx
zx run --no-vm program.zx          # Disable VM, use interpreter only
zx run --vm-mode stack program.zx   # Force stack-based VM
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--use-vm / --no-vm` | Enable/disable VM execution | Enabled |
| `--vm-mode [auto\|stack\|register\|parallel]` | VM execution mode | auto |
| `--no-optimize` | Disable bytecode optimizations | Off |
| `--precompile-modules` | Pre-parse and cache imports | Off |

**Status:** ✅ Works correctly

---

### `zx -r "<code>"`

Execute inline Zexus code directly (like `python -c`).

```bash
zx -r 'print("Hello!")'
zx -r 'let x = 10; print(x * 2)'
```

**Status:** ✅ Works correctly

---

### `zx repl`

Start the interactive Read-Eval-Print Loop.

```bash
zx repl
```

**Status:** ✅ Works (interactive mode)

---

### `zx check <file>`

Check syntax of a Zexus file with detailed validation.

```bash
zx check program.zx
```

**Status:** ✅ Works correctly

---

### `zx validate <file>`

Validate and attempt to auto-fix Zexus syntax.

```bash
zx validate program.zx
```

**Status:** ⚠️ Validation works but the auto-fix summary may produce an `'applied_fixes'` error. Syntax checking itself is functional.

---

### `zx ast <file>`

Display the Abstract Syntax Tree of a Zexus program.

```bash
zx ast program.zx
```

**Status:** ✅ Works correctly

---

### `zx tokens <file>`

Show the tokenization output of a Zexus file.

```bash
zx tokens program.zx
```

**Status:** ✅ Works correctly

---

### `zx profile <file>`

Profile performance of a Zexus program with timing and memory data.

```bash
zx profile program.zx
zx profile --no-memory program.zx     # Without memory profiling
zx profile --top 10 program.zx        # Show top 10 functions
zx profile --json-output stats.json program.zx  # Export as JSON
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--memory / --no-memory` | Enable memory profiling | Enabled |
| `--top N` | Show top N functions | 20 |
| `--json-output FILE` | Save profile data as JSON | None |

**Status:** ✅ Works correctly

---

### `zx init`

Initialize a new Zexus project with recommended structure.

```bash
zx init
```

**Status:** ✅ Works correctly

---

### `zx compile <file>`

Compile a Zexus file to a target format.

```bash
zx compile program.zx
```

**Status:** ✅ Works correctly

---

### `zx debug <on|off|minimal|status>`

Control persistent debug logging.

```bash
zx debug on         # Enable debug output
zx debug off        # Disable debug output
zx debug minimal    # Minimal debug output
zx debug status     # Show current debug settings
```

**Status:** ✅ Works correctly

---

### `zx kernel`

Show Zexus kernel status and registered domains.

```bash
zx kernel
```

**Status:** ✅ Works correctly

---

## Global Options

These options can be used with any command:

| Option | Description |
|--------|-------------|
| `--version` | Show Zexus version |
| `--syntax-style [universal\|tolerable\|auto]` | Syntax style mode |
| `--advanced-parsing` | Enable multi-strategy parsing (default: on) |
| `--execution-mode [interpreter\|compiler\|auto]` | Execution engine |
| `--debug [on\|off\|minimal\|full\|none]` | Debug logging level |
| `--no-debug` | Disable debug logging |
| `--zexus` | Show all available Zexus commands |
| `--help` | Show help |

---

## Quick Examples

```bash
# Run a program
zx run hello.zx

# Run inline code
zx -r 'print("Hello, Zexus!")'

# Check syntax without running
zx check myfile.zx

# Show tokens
zx tokens myfile.zx

# Show AST
zx ast myfile.zx

# Profile performance
zx profile heavy_computation.zx

# Start REPL
zx repl

# Initialize new project
zx init
```
