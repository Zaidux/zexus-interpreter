# Zexus Module System Documentation

## Overview

The Zexus module system allows you to organize code into reusable modules and import them into other Zexus programs. This document describes the syntax, behavior, and implementation of the module system.

## Module System Features

- **Module Loading**: Load external Zexus files using the `use` statement
- **Exports**: Control which symbols are exported from a module using the `export` statement
- **Module Caching**: Modules are cached to avoid re-evaluation and improve performance
- **Circular Imports**: Handled gracefully via placeholder caching to prevent infinite recursion
- **Module Aliases**: Import modules with custom names using the `as` keyword

## Syntax

### Using a Module

#### Basic Import

```zexus
use "./path/to/module.zx"
```

This loads the module and makes all of its exports available in the current scope.

#### Import with Alias

```zexus
use "./path/to/module.zx" as mymodule
```

This imports the module but prefixes all exports with the alias name (reserved for future use).

### Exporting from a Module

#### Single Export

```zexus
let pi = 3.14159
export pi
```

This exports the symbol `pi` from the module, making it available to importers.

#### Multiple Exports

```zexus
let version = "1.0"
let author = "Zexus Labs"

export version
export author
```

Multiple `export` statements can be used to export different symbols.

## Module Search Paths

The module system searches for modules in the following locations (in order):

1. **Current directory** (where the importing script is located)
2. **`./zpm_modules`** (local module repository)
3. **`./lib`** (standard library directory)

Paths in `use` statements can be:
- Relative paths: `"./my_module.zx"`, `"../shared/utils.zx"`
- Absolute paths: `"/workspaces/modules/core.zx"`

## Circular Dependencies

The module system handles circular imports gracefully:

```zexus
// file: a.zx
let a = 1
use "./b.zx"
export a
export b

// file: b.zx
let b = 2
use "./a.zx"
export b
export a
```

When `a.zx` uses `b.zx`, which in turn uses `a.zx`, the system prevents infinite recursion by:

1. Creating a placeholder environment for the module BEFORE evaluating it
2. Caching the placeholder immediately
3. Evaluating the module code into the cached environment
4. This allows `b.zx` to access the partial state of `a.zx` during its evaluation

Both modules are eventually fully evaluated and their exports are accessible in the importer.

## Examples

### Example 1: Simple Module and Import

**math.zx:**
```zexus
let pi = 3.14159
let e = 2.71828

export pi
export e
```

**main.zx:**
```zexus
use "./math.zx"

print(pi)    // Output: 3.14159
print(e)     // Output: 2.71828
```

### Example 2: Module with Functions

**strings.zx:**
```zexus
action uppercase(s) {
    return s  // This is a placeholder
}

action lowercase(s) {
    return s  // This is a placeholder
}

export uppercase
export lowercase
```

**main.zx:**
```zexus
use "./strings.zx"

let text = "Hello"
print(uppercase(text))
```

### Example 3: Re-exports

**core.zx:**
```zexus
let version = "1.0"
export version
```

**extended.zx:**
```zexus
use "./core.zx"

let extra = "feature"

export version  // Re-export from core
export extra    // Export local symbol
```

**main.zx:**
```zexus
use "./extended.zx"

print(version)  // Output: "1.0"
print(extra)    // Output: "feature"
```

## Module Caching

Modules are cached in memory after their first evaluation. This provides two benefits:

1. **Performance**: Subsequent imports don't re-parse and re-evaluate the module
2. **Consistency**: All importers see the same module state (excluding isolated exports)

### Cache Invalidation

The cache can be cleared programmatically:

```python
from zexus.module_cache import invalidate_module, list_cached_modules

# Invalidate a specific module
invalidate_module("./path/to/module.zx")

# List all cached modules
cached = list_cached_modules()
print(cached)  # ["./math.zx", "./strings.zx", ...]
```

## Implementation Details

### Architecture

The module system consists of three main components:

1. **Lexer/Parser**: Recognizes `use` and `export` statements and produces AST nodes
   - `UseStatement`: Represents `use "path" [as alias]`
   - `ExportStatement`: Represents `export identifier`

2. **Module Cache** (`src/zexus/module_cache.py`):
   - Stores evaluated module environments
   - Normalizes and resolves module paths
   - Provides APIs to list and invalidate cached modules

3. **Evaluator** (`src/zexus/evaluator.py`):
   - Executes `UseStatement` nodes by:
     - Resolving the module path
     - Creating and caching a placeholder environment (breaks cycles)
     - Parsing and evaluating the module file
     - Extracting exports and wiring them into the importer's environment
     - Cleaning up placeholders on error

### Placeholder Caching for Circular Imports

When a module `a.zx` uses module `b.zx` which uses `a.zx`:

1. `main` starts importing `a.zx`
2. Evaluator creates `env_a = Environment()` and calls `cache_module("a.zx", env_a)`
3. While evaluating `a.zx`, it encounters `use "./b.zx"`
4. Evaluator creates `env_b = Environment()` and calls `cache_module("b.zx", env_b)`
5. While evaluating `b.zx`, it encounters `use "./a.zx"`
6. Evaluator calls `get_cached_module("a.zx")` and gets `env_a` (already cached!)
7. No infinite recursion occurs

### Export Resolution

When a module is imported:

1. The evaluator calls `env.get_exports()` to retrieve all exported symbols
2. Each exported symbol is set in the importer's environment using `env.set(name, value)`
3. This makes the exports directly accessible by name (without module prefix)

## Limitations and Known Issues

1. **Module aliases** (`as` keyword) are parsed but not fully implemented - all exports are direct
2. **Dynamic imports**: Imports must be static at module definition time (no runtime `use` statements)
3. **Lazy loading**: Modules are evaluated immediately on use (no lazy/deferred loading)
4. **Module reloading**: Once cached, modules cannot be reloaded without clearing the cache
5. **Selective imports**: The system imports all exports (no `from ... import ...` syntax yet)

## Future Enhancements

1. **Selective Imports**: Support `from "./module.zx" import symbol1, symbol2` syntax
2. **Module Reloading**: Add reload functionality for development workflows
3. **Namespaced Imports**: Use `as` keyword to create module namespaces
4. **Lazy Loading**: Defer module evaluation until first access
5. **Better Error Messages**: Include module path and line number in circular dependency errors
6. **Module Version Management**: Track module versions and dependencies
7. **Hot Module Replacement**: Support updating modules at runtime

## Testing

The module system includes comprehensive tests in `tests/test_module_system.py`:

- **test_simple_use_import**: Validates basic module import and export
- **test_circular_imports**: Validates that circular imports are handled without infinite recursion

Run tests with:
```bash
cd /workspaces/zexus-interpreter
PYTHONPATH=./src pytest tests/test_module_system.py -v
```

## Troubleshooting

### Module Not Found

**Error**: `FileNotFoundError: Module not found at path "./my_module.zx"`

**Solution**: Check that the path is relative to the current working directory or use an absolute path. The module search path must include one of:
- The current directory
- `./zpm_modules/`
- `./lib/`

### Circular Import Error

**Error**: Module evaluation seems to hang or produces undefined behavior

**Solution**: This should not occur with the current implementation. If you encounter this, check that the module files exist and are readable, and that there are no syntax errors.

### Export Not Found

**Error**: `NameError: Name 'symbol' is not defined`

**Solution**: Verify that:
1. The symbol is defined in the imported module
2. The symbol is explicitly exported with `export symbol`
3. The import statement syntax is correct

## See Also

- [Zexus Language Reference](./README.md)
- [Module Cache API](../src/zexus/module_cache.py)
- [Evaluator Implementation](../src/zexus/evaluator.py)
