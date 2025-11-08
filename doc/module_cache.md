# Module Caching System

## Overview

The Zexus interpreter now includes a thread-safe module caching system to improve performance when loading modules via USE statements. This system prevents redundant parsing and evaluation of previously loaded modules.

## Features

1. Thread Safety
   - Uses thread locks to ensure safe concurrent access
   - Prevents race conditions during module loading
   - Safe for multi-threaded environments

2. Path Resolution
   - Supports absolute paths
   - Handles relative paths from current directory
   - Checks zpm_modules directory automatically
   - Normalizes paths for consistent cache keys

3. Extension Support
   - Automatically tries multiple extensions (.zx, .zexus)
   - Handles extensionless module names
   - Maintains consistent module identity across imports

4. Caching Behavior
   - Stores fully evaluated module environments
   - Thread-safe cache access and updates
   - Option to clear cache when needed

## API

- `get_cached_module(module_path)`: Retrieve cached module if available
- `cache_module(module_path, module_env)`: Store module in cache
- `clear_module_cache()`: Clear the entire module cache
- `get_module_candidates(file_path)`: Get possible module file paths
- `normalize_path(path)`: Normalize path for cache keys

## Implementation Details

The caching system is implemented in `src/zexus/module_cache.py` and provides:

1. Thread-safe Dictionary
   - Global `_MODULE_CACHE` dictionary
   - Protected by `_MODULE_CACHE_LOCK`

2. Path Resolution Logic
   - Checks multiple possible locations
   - Handles various file extensions
   - Normalizes paths for consistency

3. Interface with Evaluator
   - Integrated with USE statement handling
   - Transparent to user code
   - Maintains module semantics