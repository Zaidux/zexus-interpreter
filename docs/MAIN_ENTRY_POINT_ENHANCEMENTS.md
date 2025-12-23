# Main Entry Point Enhancements

This document describes the enhanced main entry point features in Zexus, including command-line arguments, module introspection, lifecycle hooks, and advanced execution control.

## Table of Contents

1. [Module Context Variables](#module-context-variables)
2. [Command-Line Arguments](#command-line-arguments)
3. [Enhanced run() Function](#enhanced-run-function)
4. [Lifecycle Hooks](#lifecycle-hooks)
5. [Signal Handling](#signal-handling)
6. [Module Introspection](#module-introspection)
7. [Best Practices](#best-practices)

## Module Context Variables

Zexus provides several built-in variables that give you information about the current module and its execution context.

### `__MODULE__`
The name of the current module. When a file is run directly, this is set to `"__main__"`. When a file is imported, this contains the module path.

```zexus
if __MODULE__ == "__main__":
    print("Running as main program")
else:
    print("Module imported from: " + __MODULE__)
```

### `__file__` and `__FILE__`
The absolute path to the current file.

```zexus
print("Current file: " + __file__)
print("Same thing: " + __FILE__)  # Alternative name
```

### `__DIR__`
The directory containing the current file.

```zexus
print("Current directory: " + __DIR__)
config_path = __DIR__ + "/config.json"
```

### `__PACKAGE__`
The package name (first directory component of the file path). Empty string if not in a package.

```zexus
print("Package: " + __PACKAGE__)

# Example: if file is src/myapp/main.zx
# __PACKAGE__ will be "src"
```

## Command-Line Arguments

Access command-line arguments passed to your script using `__ARGS__` or `__ARGV__`.

### `__ARGS__` and `__ARGV__`
A list containing all command-line arguments passed to the script.

**Running a script:**
```bash
zx run script.zx arg1 arg2 arg3
```

**Accessing arguments:**
```zexus
# Get all arguments
print("Arguments: " + str(__ARGS__))

# Access specific arguments
if len(__ARGS__) > 0:
    first_arg = __ARGS__[0]
    print("First argument: " + first_arg)

# Iterate through arguments
for arg in __ARGV__:  # ARGV is an alias for ARGS
    print("Argument: " + arg)
```

**Example: Script with configuration**
```zexus
# config_loader.zx
if __MODULE__ == "__main__":
    if len(__ARGS__) == 0:
        print("Usage: zx run config_loader.zx <config_file>")
        exit_program(1)
    
    config_file = __ARGS__[0]
    print("Loading config from: " + config_file)
    # Load and process config...
```

## Enhanced run() Function

The `run()` function keeps your program running continuously. It has been enhanced with support for callback arguments and better control.

### Basic Usage
```zexus
if is_main():
    run()  # Keep running forever
```

### With Callback
```zexus
func process_queue():
    print("Processing queue...")
    # Do work here

if is_main():
    run(process_queue)  # Call process_queue every second
```

### With Interval
```zexus
func check_status():
    print("Status check")

if is_main():
    run(check_status, 0.5)  # Run every 500ms (0.5 seconds)
```

### With Callback Arguments
```zexus
func process_requests(port, host):
    print("Processing on " + host + ":" + str(port))
    # Server logic here

if is_main():
    # Get port and host from command-line args or use defaults
    port = 8080
    host = "localhost"
    
    if len(__ARGS__) >= 2:
        port = int(__ARGS__[0])
        host = __ARGS__[1]
    
    # Pass arguments to callback
    run(process_requests, 1.0, [port, host])
```

## Lifecycle Hooks

Register functions to run at program startup and shutdown.

### `on_start(callback)`
Register a function to run before the main run loop starts.

```zexus
func initialize():
    print("Initializing database...")
    # Setup database connection
    print("Database ready!")

on_start(initialize)

if is_main():
    run(process_requests)
```

### `on_exit(callback)`
Register a function to run when the program exits (including Ctrl+C).

```zexus
func cleanup():
    print("Closing connections...")
    # Close database, sockets, etc.
    print("Cleanup complete!")

on_exit(cleanup)

if is_main():
    run(server_loop)
```

### Multiple Hooks
You can register multiple hooks - they run in the order they were registered.

```zexus
on_start(initialize_database)
on_start(load_configuration)
on_start(start_background_tasks)

on_exit(stop_background_tasks)
on_exit(close_database)
on_exit(save_state)
```

## Signal Handling

Register custom handlers for system signals (like Ctrl+C).

### `signal_handler(signal_name, callback)`
Register a custom signal handler.

```zexus
func handle_interrupt(signal):
    print("Caught signal: " + signal)
    print("Performing emergency save...")
    # Save work before exiting

signal_handler("SIGINT", handle_interrupt)  # Ctrl+C
signal_handler("SIGTERM", handle_interrupt)  # kill command

if is_main():
    run(main_loop)
```

**Common signals:**
- `"SIGINT"` - Interrupt signal (Ctrl+C)
- `"SIGTERM"` - Termination request
- `"SIGHUP"` - Hangup (terminal closed)

## Module Introspection

Functions to inspect and learn about the current module.

### `module_info()`
Get a map with all module information.

```zexus
info = module_info()
print("Module: " + info["module"])
print("File: " + info["file"])
print("Directory: " + info["dir"])
print("Package: " + info["package"])
```

### `list_imports()`
Get a list of all imported modules.

```zexus
imports = list_imports()
print("Imported modules: " + str(imports))

for module in imports:
    print("- " + module)
```

### `get_exported_names()`
Get a list of all exported variables/functions in the current module.

```zexus
exports = get_exported_names()
print("Exported names: " + str(exports))

for name in exports:
    print("- " + name)
```

## Best Practices

### 1. Always Use `is_main()` Check
```zexus
# ✅ Good
if is_main():
    run()

# ❌ Avoid running code at module level
run()  # This will run even when imported!
```

### 2. Handle Command-Line Arguments Gracefully
```zexus
# ✅ Good - with validation
if is_main():
    if len(__ARGS__) < 2:
        print("Usage: script.zx <input> <output>")
        exit_program(1)
    
    input_file = __ARGS__[0]
    output_file = __ARGS__[1]
    process(input_file, output_file)
```

### 3. Always Register Cleanup Hooks
```zexus
# ✅ Good - cleanup is guaranteed
var db_connection = null

func setup():
    db_connection = connect_database()

func teardown():
    if db_connection != null:
        db_connection.close()

on_start(setup)
on_exit(teardown)
```

### 4. Use Descriptive Module Variables
```zexus
# ✅ Good - clear intent
var CONFIG_DIR = __DIR__ + "/config"
var DATA_DIR = __DIR__ + "/data"

# Build paths relative to module directory
func load_config(name):
    return read_file(CONFIG_DIR + "/" + name + ".json")
```

## Complete Example

Here's a complete server example using all the enhanced features:

```zexus
# server.zx - A complete server with lifecycle management

# Configuration from command-line or defaults
var port = 8080
var host = "localhost"

func initialize():
    print("🚀 Starting server...")
    print("Module: " + __file__)
    print("Directory: " + __DIR__)
    
    # Parse command-line arguments
    if len(__ARGS__) >= 1:
        port = int(__ARGS__[0])
    if len(__ARGS__) >= 2:
        host = __ARGS__[1]
    
    print("Configuration: " + host + ":" + str(port))

func cleanup():
    print("🛑 Shutting down server...")
    # Close connections, save state
    print("✅ Server stopped cleanly")

func handle_signal(sig):
    print("⚠️  Received " + sig + ", initiating graceful shutdown...")

func process_requests(p, h):
    print("⚡ Processing requests on " + h + ":" + str(p))
    # Server logic here

# Register lifecycle hooks
on_start(initialize)
on_exit(cleanup)
signal_handler("SIGINT", handle_signal)
signal_handler("SIGTERM", handle_signal)

# Run server if this is the main module
if is_main():
    print("Starting run loop...")
    run(process_requests, 1.0, [port, host])
else:
    print("Server module imported (not running)")
```

**Running the server:**
```bash
# Default configuration
zx run server.zx

# Custom port
zx run server.zx 3000

# Custom port and host
zx run server.zx 3000 0.0.0.0
```

## Function Reference

| Function | Signature | Description |
|----------|-----------|-------------|
| `run()` | `run([callback, interval, args])` | Keep program running with optional callback |
| `execute()` | `execute([callback, interval, args])` | Alias for `run()` |
| `is_main()` | `is_main() -> bool` | Check if module is main entry point |
| `exit_program()` | `exit_program([code])` | Exit with optional exit code |
| `on_start()` | `on_start(callback)` | Register startup hook |
| `on_exit()` | `on_exit(callback)` | Register shutdown hook |
| `signal_handler()` | `signal_handler(signal, callback)` | Register signal handler |
| `module_info()` | `module_info() -> map` | Get module metadata |
| `list_imports()` | `list_imports() -> list` | List imported modules |
| `get_exported_names()` | `get_exported_names() -> list` | List exported names |

## Variable Reference

| Variable | Type | Description |
|----------|------|-------------|
| `__MODULE__` | string | Module name or `"__main__"` |
| `__file__`, `__FILE__` | string | Absolute file path |
| `__DIR__` | string | Directory path |
| `__PACKAGE__` | string | Package name |
| `__ARGS__`, `__ARGV__` | list | Command-line arguments |
