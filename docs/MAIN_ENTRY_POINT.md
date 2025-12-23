# Main Entry Point Pattern in Zexus

## Overview

Zexus now supports a pattern similar to Python's `if __name__ == "__main__":` that allows developers to:

1. **Distinguish between direct execution and module import**: Write code that only runs when a file is executed directly, not when imported as a module
2. **Keep programs running continuously**: Use `run()` or `execute()` to create long-running programs like servers or event loops
3. **Control program lifecycle**: Use helper functions like `is_main()` and `exit_program()` for better control

This feature was implemented to address the common need for:
- Creating reusable modules with testable entry points
- Building servers and daemons that run continuously
- Writing scripts that behave differently when run vs. imported

---

## Features

### 1. `__MODULE__` Special Variable

The `__MODULE__` variable is automatically set by the Zexus runtime:
- **`"__main__"`** when a file is executed directly
- **`<module_path>`** when a file is imported as a module

```zexus
if __MODULE__ == "__main__" {
    # This code only runs when file is executed directly
    print("Running as main program!")
} else {
    print("Loaded as module: " + __MODULE__)
}
```

### 2. `run()` Function

Keeps the program running until interrupted (Ctrl+C). Useful for servers, event loops, or long-running programs.

**Signatures:**
```zexus
run()                     # Keep running indefinitely
run(callback)             # Call callback repeatedly (1 second interval)
run(callback, interval)   # Call callback every N seconds
```

**Parameters:**
- `callback` (optional): Function to execute repeatedly
- `interval` (optional): Number of seconds between executions (default: 1.0)

**Example:**
```zexus
action server_tick() {
    print("Server is running...")
}

if __MODULE__ == "__main__" {
    print("Starting server...")
    run(server_tick, 5)  # Call every 5 seconds
}
```

### 3. `execute()` Function

Alias for `run()` - same functionality, different name.

```zexus
if __MODULE__ == "__main__" {
    execute()  # Keep program alive
}
```

### 4. `is_main()` Helper

Returns `true` if the current module is the main entry point, `false` otherwise.

```zexus
if is_main() {
    print("Running as main!")
}
```

This is equivalent to `__MODULE__ == "__main__"` but more readable.

### 5. `exit_program()` Function

Exit the program with an optional exit code.

**Signatures:**
```zexus
exit_program()      # Exit with code 0 (success)
exit_program(code)  # Exit with specified code
```

**Example:**
```zexus
if error_occurred {
    print("Fatal error!")
    exit_program(1)  # Exit with error code
}
```

---

## Use Cases

### Use Case 1: Reusable Library with Examples

Create a module that provides functions and can also demonstrate them:

```zexus
# File: math_utils.zx

action square(n) {
    return n * n
}

action cube(n) {
    return n * n * n
}

export square
export cube

# Demo code that only runs when file is executed directly
if __MODULE__ == "__main__" {
    print("=== Math Utils Demo ===")
    print("Square of 5: " + string(square(5)))
    print("Cube of 3: " + string(cube(3)))
}
```

When run directly:
```bash
$ zx run math_utils.zx
=== Math Utils Demo ===
Square of 5: 25
Cube of 3: 27
```

When imported:
```zexus
use "./math_utils.zx" as math
print(math.square(10))  # Demo doesn't run
```

### Use Case 2: Long-Running Server

```zexus
# File: server.zx

let request_count = 0

action handle_request() {
    request_count = request_count + 1
    print("Handled request #" + string(request_count))
}

if __MODULE__ == "__main__" {
    print("🌐 Server starting...")
    print("Press Ctrl+C to stop")
    
    # Process requests every 2 seconds
    run(handle_request, 2)
}
```

### Use Case 3: Configuration Module

```zexus
# File: config.zx

const API_KEY = "secret_key_123"
const MAX_RETRIES = 3
const TIMEOUT = 30

export API_KEY
export MAX_RETRIES  
export TIMEOUT

# Validation that only runs when executed directly
if __MODULE__ == "__main__" {
    print("✓ Configuration loaded")
    print("API Key: " + API_KEY)
    print("Max Retries: " + string(MAX_RETRIES))
    print("Timeout: " + string(TIMEOUT))
}
```

### Use Case 4: Test Runner

```zexus
# File: tests.zx

action test_addition() {
    let result = 2 + 2
    if result == 4 {
        print("✓ Addition test passed")
        return true
    }
    print("✗ Addition test failed")
    return false
}

action test_multiplication() {
    let result = 3 * 4
    if result == 12 {
        print("✓ Multiplication test passed")
        return true
    }
    print("✗ Multiplication test failed")
    return false
}

export test_addition
export test_multiplication

# Run all tests when executed directly
if __MODULE__ == "__main__" {
    print("Running tests...")
    let passed = 0
    let total = 2
    
    if test_addition() {
        passed = passed + 1
    }
    
    if test_multiplication() {
        passed = passed + 1
    }
    
    print("")
    print("Results: " + string(passed) + "/" + string(total) + " tests passed")
    
    if passed == total {
        exit_program(0)
    } else {
        exit_program(1)
    }
}
```

---

## Implementation Details

### How `__MODULE__` is Set

1. **Direct execution** (via `zx run file.zx`):
   - CLI sets `__MODULE__ = "__main__"` in the root environment
   
2. **Module import** (via `use "./file.zx"`):
   - Evaluator sets `__MODULE__ = <module_path>` when loading the module
   - This allows the module to know how it was loaded

### Signal Handling in `run()`

The `run()` function properly handles interrupt signals:
- **SIGINT** (Ctrl+C): Graceful shutdown
- **SIGTERM**: Graceful shutdown
- Cleanup code executes before exit

### Exit Codes

Standard exit codes:
- **0**: Success
- **1**: Generic error
- **2+**: Application-specific errors

---

## Best Practices

### 1. Always Export Reusable Functions

```zexus
# ✅ Good: Export functions for reuse
action helper() {
    return "helper"
}

export helper

if __MODULE__ == "__main__" {
    helper()
}
```

```zexus
# ❌ Bad: No exports, can't be imported
action helper() {
    return "helper"
}

helper()  # Runs even when imported!
```

### 2. Use Descriptive Main Blocks

```zexus
# ✅ Good: Clear main entry point
if __MODULE__ == "__main__" {
    print("Starting application...")
    main_function()
}
```

```zexus
# ❌ Bad: Code runs on import
print("Starting application...")
main_function()
```

### 3. Provide Usage Examples in Main Block

```zexus
# ✅ Good: Self-documenting with examples
export process_data

if __MODULE__ == "__main__" {
    print("Example usage:")
    let result = process_data([1, 2, 3])
    print("Result: " + string(result))
}
```

### 4. Use run() for Long-Running Processes

```zexus
# ✅ Good: Proper event loop
if __MODULE__ == "__main__" {
    run(poll_for_updates, 10)
}
```

```zexus
# ❌ Bad: Busy waiting
if __MODULE__ == "__main__" {
    while true {
        poll_for_updates()
        # No sleep, wastes CPU!
    }
}
```

---

## Comparison with Other Languages

### Python

**Python:**
```python
def main():
    print("Main function")

if __name__ == "__main__":
    main()
```

**Zexus:**
```zexus
action main() {
    print("Main function")
}

if __MODULE__ == "__main__" {
    main()
}
```

### Node.js

**Node.js:**
```javascript
function main() {
    console.log("Main function");
}

if (require.main === module) {
    main();
}
```

**Zexus:**
```zexus
action main() {
    print("Main function")
}

if __MODULE__ == "__main__" {
    main()
}
```

---

## Complete Examples

See the `examples/` directory for working examples:
- `examples/main_entry_point.zx` - Basic __MODULE__ usage
- `examples/import_main_module.zx` - Importing a module with main block
- `examples/server_with_run.zx` - Long-running server example
- `examples/simple_run_example.zx` - Simple run() usage
- `examples/is_main_helper.zx` - Using is_main() helper
- `examples/exit_program_example.zx` - Exit code handling

---

## API Reference

### `__MODULE__` (Variable)
- **Type**: String
- **Values**: `"__main__"` or module path
- **Description**: Indicates execution context
- **Usage**: `if __MODULE__ == "__main__" { ... }`

### `run([callback, interval])`
- **Parameters**:
  - `callback` (optional): Function to call repeatedly
  - `interval` (optional): Seconds between calls (default: 1.0)
- **Returns**: null
- **Throws**: EvaluationError on invalid arguments
- **Description**: Keeps program running until interrupted

### `execute([callback, interval])`
- **Alias**: Same as `run()`

### `is_main()`
- **Parameters**: None
- **Returns**: Boolean
- **Description**: Returns true if running as main module

### `exit_program([code])`
- **Parameters**:
  - `code` (optional): Integer exit code (default: 0)
- **Returns**: Never (exits process)
- **Description**: Terminates program with exit code

---

## Troubleshooting

### Problem: Main block runs when imported

**Cause**: Not using `if __MODULE__ == "__main__"` check

**Solution**:
```zexus
# Wrap execution code in main block
if __MODULE__ == "__main__" {
    # Your code here
}
```

### Problem: run() exits immediately

**Cause**: No callback provided and program completes

**Solution**: Provide a callback function:
```zexus
run(my_callback, 1)
```

### Problem: Can't stop program with Ctrl+C

**Cause**: Not using `run()`, using infinite while loop

**Solution**: Use `run()` which handles signals properly:
```zexus
# Replace this:
while true { }

# With this:
run()
```

---

## Future Enhancements

Planned features:
- `__PACKAGE__` variable for package identification
- `reload()` function for hot module reloading
- `schedule()` for cron-like scheduling
- `daemon()` for background processes
- Better module introspection

---

**Related Documentation:**
- [Module System](MODULE_SYSTEM.md)
- [Builtins Reference](../docs/BUILTINS.md)
- [CLI Usage](../README.md#cli-usage)

**Last Updated**: December 23, 2025
