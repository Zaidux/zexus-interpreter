# Main Entry Point Feature Implementation Summary

## Overview

Successfully implemented a Python-like `if __name__ == "__main__":` pattern for Zexus, along with program lifecycle management features.

**Implementation Date:** December 23, 2025

---

## Features Implemented

### 1. ✅ `__MODULE__` Special Variable

A special variable that tracks execution context:
- Set to `"__main__"` when file is executed directly via CLI
- Set to module path when file is imported via `use` statement
- Automatically managed by the runtime

**Implementation Location:**
- [src/zexus/cli/main.py](src/zexus/cli/main.py#L172) - Sets `__MODULE__ = "__main__"` for direct execution
- [src/zexus/evaluator/statements.py](src/zexus/evaluator/statements.py#L490-L491) - Sets `__MODULE__ = <path>` for imports

### 2. ✅ `run()` Function

Keeps program running until interrupted (Ctrl+C).

**Signatures:**
```zexus
run()                     # Keep alive indefinitely
run(callback)             # Execute callback every 1 second
run(callback, interval)   # Execute callback every N seconds
```

**Features:**
- Proper signal handling (SIGINT, SIGTERM)
- Graceful shutdown
- Customizable execution interval
- Error handling for callback failures

**Implementation Location:**
- [src/zexus/evaluator/functions.py](src/zexus/evaluator/functions.py#L1032-L1089) - Function implementation

### 3. ✅ `execute()` Function

Alias for `run()` - same functionality.

**Implementation Location:**
- [src/zexus/evaluator/functions.py](src/zexus/evaluator/functions.py#L1091-L1100) - Function implementation

### 4. ✅ `is_main()` Helper

Returns `true` if current module is the main entry point.

**Implementation Location:**
- [src/zexus/evaluator/functions.py](src/zexus/evaluator/functions.py#L1102-L1116) - Function implementation

### 5. ✅ `exit_program()` Function

Exit program with optional exit code.

**Signature:**
```zexus
exit_program()      # Exit with code 0
exit_program(code)  # Exit with specified code
```

**Implementation Location:**
- [src/zexus/evaluator/functions.py](src/zexus/evaluator/functions.py#L1118-L1132) - Function implementation

---

## Files Modified

### Core Implementation

1. **[src/zexus/evaluator/functions.py](src/zexus/evaluator/functions.py)**
   - Added `_register_main_entry_point_builtins()` method
   - Registered 4 new builtin functions
   - ~140 lines of new code

2. **[src/zexus/cli/main.py](src/zexus/cli/main.py)**
   - Added `String` import
   - Set `__MODULE__` variable for direct execution
   - 2 lines modified

3. **[src/zexus/evaluator/statements.py](src/zexus/evaluator/statements.py)**
   - Set `__MODULE__` variable when loading modules (2 locations)
   - 4 lines added

### Documentation

4. **[docs/MAIN_ENTRY_POINT.md](docs/MAIN_ENTRY_POINT.md)** (NEW)
   - Comprehensive feature documentation
   - API reference
   - Use cases and examples
   - Best practices
   - Troubleshooting guide

5. **[examples/MAIN_ENTRY_POINT_EXAMPLES.md](examples/MAIN_ENTRY_POINT_EXAMPLES.md)** (NEW)
   - Examples overview
   - Quick start guide

### Examples

6. **[examples/main_entry_point.zx](examples/main_entry_point.zx)** (NEW)
   - Basic `__MODULE__` pattern
   - Exportable functions with demo

7. **[examples/import_main_module.zx](examples/import_main_module.zx)** (NEW)
   - Demonstrates module import behavior

8. **[examples/server_with_run.zx](examples/server_with_run.zx)** (NEW)
   - Long-running server example

9. **[examples/simple_run_example.zx](examples/simple_run_example.zx)** (NEW)
   - Basic `run()` usage

10. **[examples/is_main_helper.zx](examples/is_main_helper.zx)** (NEW)
    - Using `is_main()` helper

11. **[examples/exit_program_example.zx](examples/exit_program_example.zx)** (NEW)
    - Exit code handling

---

## Usage Examples

### Basic Main Entry Point

```zexus
# Define reusable functions
action greet(name) {
    return "Hello, " + name + "!"
}

export greet

# Only run when executed directly
if __MODULE__ == "__main__" {
    print(greet("World"))
}
```

### Long-Running Server

```zexus
action handle_request() {
    print("Processing request...")
}

if __MODULE__ == "__main__" {
    print("Server starting...")
    run(handle_request, 2)  # Every 2 seconds
}
```

### Using Helper Function

```zexus
if is_main() {
    print("Running as main!")
}
```

---

## Technical Details

### Signal Handling

The `run()` function properly handles:
- **SIGINT** (Ctrl+C) - Graceful shutdown
- **SIGTERM** - Graceful shutdown
- Cleanup in `finally` block

### Module Loading Behavior

1. **Direct execution:**
   ```bash
   zx run file.zx  # __MODULE__ = "__main__"
   ```

2. **Module import:**
   ```zexus
   use "./file.zx"  # __MODULE__ = "./file.zx" in imported module
   ```

### Environment Variable Scope

- `__MODULE__` is set in each module's environment
- Not inherited from outer scopes
- Fresh value for each module load

---

## Testing

### Manual Testing Performed

✅ Direct execution sets `__MODULE__ = "__main__"`
✅ Module import sets `__MODULE__ to module path
✅ `run()` keeps program alive
✅ `run(callback)` executes callback periodically
✅ Ctrl+C gracefully shuts down `run()`
✅ `is_main()` returns correct boolean
✅ `exit_program()` terminates with exit code

### Test Files

All example files in `examples/` directory serve as integration tests.

To run tests:
```bash
zx run examples/main_entry_point.zx
zx run examples/import_main_module.zx
zx run examples/is_main_helper.zx
zx run examples/exit_program_example.zx
# For interactive tests:
zx run examples/server_with_run.zx  # Press Ctrl+C to stop
zx run examples/simple_run_example.zx  # Press Ctrl+C to stop
```

---

## Compatibility

### Python Equivalent

**Python:**
```python
if __name__ == "__main__":
    main()
```

**Zexus:**
```zexus
if __MODULE__ == "__main__" {
    main()
}
```

### Node.js Equivalent

**Node.js:**
```javascript
if (require.main === module) {
    main();
}
```

**Zexus:**
```zexus
if __MODULE__ == "__main__" {
    main()
}
```

---

## Benefits

1. **Code Reusability**: Write modules that can be both imported and executed
2. **Testability**: Add test code that only runs during direct execution
3. **Documentation**: Include usage examples in the main block
4. **Servers & Daemons**: Easy to create long-running processes
5. **Better Structure**: Clear separation between library code and entry points

---

## Future Enhancements

Potential future additions:
- `__PACKAGE__` for package identification
- `reload()` for hot module reloading  
- `schedule()` for cron-like scheduling
- `daemon()` for background processes
- Better module introspection APIs

---

## Related Documentation

- [Module System Documentation](docs/MODULE_SYSTEM.md)
- [Main Entry Point Guide](docs/MAIN_ENTRY_POINT.md)
- [Examples Guide](examples/MAIN_ENTRY_POINT_EXAMPLES.md)
- [README](README.md)

---

## Summary

Successfully implemented a comprehensive main entry point pattern for Zexus that matches the developer experience of Python's `if __name__ == "__main__":` while adding additional features like `run()` for long-running processes. The implementation is clean, well-documented, and includes 6 working examples.

**Total Lines Added:** ~700 (code + documentation + examples)
**Files Modified:** 3
**Files Created:** 8
**Status:** ✅ Complete and Tested
