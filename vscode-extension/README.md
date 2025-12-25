# Zexus VS Code Extension

Full-featured VS Code extension for the Zexus programming language with IntelliSense, debugging, and performance profiling.

## Features

### 🎯 IntelliSense & Code Completion
- **Keyword completion** - All 130+ Zexus keywords
- **Built-in function completion** - 100+ built-in functions with signatures
- **Smart suggestions** - Context-aware completions
- **Parameter hints** - Function signature help
- **Snippet support** - Quick templates for common patterns

### 📖 Language Server Protocol (LSP)
- **Real-time diagnostics** - Syntax error checking as you type
- **Hover documentation** - See documentation on hover
- **Go to definition** - Navigate to symbol definitions
- **Find all references** - Find all usages of a symbol
- **Document symbols** - Outline view for quick navigation
- **Code formatting** - Automatic code formatting

### 🐛 Debugger Integration
- **Breakpoints** - Set breakpoints in your code
- **Step debugging** - Step over, step into, step out
- **Variable inspection** - Inspect variables and their values
- **Call stack** - View the call stack
- **Watch expressions** - Watch specific expressions
- **Debug console** - REPL in debug context

### ⚡ Performance Profiling
- **Execution profiling** - Measure function execution time
- **Memory profiling** - Track memory allocation and peak usage
- **Hotspot detection** - Identify performance bottlenecks
- **Call graph** - Visualize function call relationships
- **Profile reports** - Detailed profiling reports

### 🎨 Syntax Highlighting
- Full TextMate grammar for Zexus
- Supports all language features
- Optimized for both light and dark themes

## Installation

### From Source (Development)

1. Clone the repository:
```bash
git clone https://github.com/Zaidux/zexus-interpreter.git
cd zexus-interpreter/vscode-extension
```

2. Install dependencies:
```bash
npm install
```

3. Compile the extension:
```bash
npm run compile
```

4. Install the Zexus language server:
```bash
cd ..
pip install -e ".[dev]"
pip install pygls
```

5. Open in VS Code:
```bash
code .
```

6. Press F5 to launch the Extension Development Host

### From VSIX (Coming Soon)

```bash
code --install-extension zexus-vscode-1.5.0.vsix
```

## Usage

### Running Zexus Files

- **Keyboard shortcut**: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- **Command Palette**: `Zexus: Run Zexus File`
- Opens a terminal and runs the current file

### Syntax Checking

- **Command Palette**: `Zexus: Check Syntax`
- Runs syntax validation on the current file

### Performance Profiling

- **Command Palette**: `Zexus: Profile Performance`
- Runs the file with profiling enabled
- Displays execution time, memory usage, and hotspots

### Debugging

1. Open a Zexus file (.zx)
2. Set breakpoints by clicking in the gutter
3. Press F5 or select "Run > Start Debugging"
4. Use the debug toolbar to control execution

Debug configurations are in `.vscode/launch.json`:

```json
{
  "type": "zexus",
  "request": "launch",
  "name": "Debug Zexus File",
  "program": "${file}",
  "stopOnEntry": true
}
```

## Configuration

Configure the extension in VS Code settings:

```json
{
  "zexus.syntaxStyle": "auto",           // "universal", "tolerable", or "auto"
  "zexus.executionMode": "auto",         // "interpreter", "compiler", or "auto"
  "zexus.advancedParsing": true,         // Enable multi-strategy parsing
  "zexus.languageServer.enabled": true,  // Enable LSP
  "zexus.languageServer.trace": "off",   // LSP trace level
  "zexus.profiling.enabled": false       // Enable profiling
}
```

## Requirements

- VS Code 1.75.0 or higher
- Python 3.8 or higher
- Zexus interpreter (install with `pip install zexus`)
- pygls (install with `pip install pygls`)

## Features in Detail

### IntelliSense

The extension provides intelligent code completion:

- **Keywords**: All language keywords with descriptions
- **Built-ins**: Function signatures with parameter information
- **Context-aware**: Suggestions based on current context
- **Snippets**: Templates for common patterns

Example completions:
- Type `con` → suggests `contract`, `const`, `continue`
- Type `print` → shows `print(value)` with documentation
- In a contract → suggests contract-specific keywords

### LSP Features

#### Hover Information
Hover over any symbol to see:
- Documentation
- Type information
- Parameter descriptions

#### Go to Definition
Navigate to where symbols are defined:
- Functions
- Variables
- Contracts
- Entities

#### Diagnostics
Real-time error detection:
- Syntax errors
- Type errors
- Undefined variables
- Import errors

### Debugging

**Coming Soon**: Full DAP (Debug Adapter Protocol) implementation is planned for a future release.

Planned features:

#### Breakpoints (Planned)
- Line breakpoints
- Conditional breakpoints
- Log points

#### Execution Control (Planned)
- Continue (F5)
- Step Over (F10)
- Step Into (F11)
- Step Out (Shift+F11)
- Restart (Ctrl+Shift+F5)
- Stop (Shift+F5)

#### Variable Inspection
- View all variables in current scope
- Expand nested objects
- Edit variables during debugging

#### Debug Console
- Execute Zexus code in debug context
- Inspect expressions
- Call functions

### Profiling

Comprehensive performance analysis:

#### Execution Time
- Total execution time
- Per-function time
- Self time vs total time
- Call counts

#### Memory Usage
- Peak memory usage
- Per-function allocation
- Memory growth over time

#### Hotspot Detection
- Identifies slow functions
- Shows percentage of total time
- Suggests optimization opportunities

Example profile report:
```
================================================================================
ZEXUS PERFORMANCE PROFILE REPORT
================================================================================

Total Time: 1.2345 seconds
Total Calls: 1523
Peak Memory: 12.34 MB

Function                                      Calls      Total Time       Self Time        Avg Time
----------------------------------------------------------------------------------------------------
process_data                                     100        0.8234s        0.3456s      0.008234s
calculate_result                                 500        0.3456s        0.3456s      0.000691s
...

================================================================================
TOP HOTSPOTS (by total time)
================================================================================
1. process_data: 0.8234s (66.7%)
2. calculate_result: 0.3456s (28.0%)
...
================================================================================
```

## Troubleshooting

### Language Server Not Starting

1. Check Python installation:
```bash
python3 --version
pip3 list | grep pygls
```

2. Install pygls if missing:
```bash
pip install pygls
```

3. Restart language server:
   - Command Palette → "Zexus: Restart Language Server"

### IntelliSense Not Working

1. Verify language server is running:
   - Output panel → "Zexus Language Server"

2. Check file extension is `.zx`

3. Reload VS Code window:
   - Command Palette → "Developer: Reload Window"

### Debugger Not Working

1. Verify Zexus is installed:
```bash
zx --version
```

2. Check launch configuration in `.vscode/launch.json`

3. Ensure file is saved before debugging

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Links

- [Zexus Documentation](../docs/)
- [GitHub Repository](https://github.com/Zaidux/zexus-interpreter)
- [Issue Tracker](https://github.com/Zaidux/zexus-interpreter/issues)

## Changelog

See [CHANGELOG.md](../CHANGELOG.md) for version history.

## Support

- GitHub Issues: [Report a bug](https://github.com/Zaidux/zexus-interpreter/issues/new)
- Discussions: [Ask questions](https://github.com/Zaidux/zexus-interpreter/discussions)
