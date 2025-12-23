# Main Entry Point Examples

This directory contains examples demonstrating Zexus's main entry point pattern, similar to Python's `if __name__ == "__main__":`.

## Examples

### 1. `main_entry_point.zx`
Demonstrates the basic `__MODULE__` pattern. Shows how to write a module that can be both imported and executed directly.

**Run it:**
```bash
zx run examples/main_entry_point.zx
```

**Expected output:**
```
🚀 Running as main program!
Testing utility functions:
Hello, World!
5 + 3 = 8
4 * 6 = 24
✅ Main execution complete!
```

### 2. `import_main_module.zx`
Imports the `main_entry_point.zx` module and uses its functions. Notice that the main block from `main_entry_point.zx` does NOT execute when imported.

**Run it:**
```bash
zx run examples/import_main_module.zx
```

### 3. `server_with_run.zx`
Demonstrates using `run()` to create a long-running server that handles requests periodically.

**Run it:**
```bash
zx run examples/server_with_run.zx
```

**Note:** Press Ctrl+C to stop the server.

### 4. `simple_run_example.zx`
Simple demonstration of the `run()` function with a callback.

**Run it:**
```bash
zx run examples/simple_run_example.zx
```

**Note:** Press Ctrl+C to stop.

### 5. `is_main_helper.zx`
Shows using the `is_main()` helper function as an alternative to checking `__MODULE__`.

**Run it:**
```bash
zx run examples/is_main_helper.zx
```

### 6. `exit_program_example.zx`
Demonstrates using `exit_program()` to terminate with specific exit codes.

**Run it:**
```bash
zx run examples/exit_program_example.zx
```

## New Features

### `__MODULE__` Variable
- Automatically set to `"__main__"` when file is run directly
- Set to module path when file is imported
- Use to distinguish between direct execution and module import

### `run()` / `execute()` Functions
- Keep program running until interrupted
- Optional callback for periodic execution
- Proper signal handling (Ctrl+C)

### `is_main()` Helper
- Returns `true` if running as main module
- More readable than `__MODULE__ == "__main__"`

### `exit_program()` Function
- Exit with specific exit code
- Clean program termination

## Documentation

See [MAIN_ENTRY_POINT.md](../docs/MAIN_ENTRY_POINT.md) for complete documentation.
