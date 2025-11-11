"""
Module cache for Zexus interpreter.

This module provides caching functionality for loaded Zexus modules to avoid
re-parsing and re-evaluating modules that have already been loaded.
"""

import os
import threading
from typing import Dict, Optional
from .object import Environment

_MODULE_CACHE: Dict[str, Environment] = {}
_MODULE_CACHE_LOCK = threading.Lock()

def get_cached_module(module_path: str) -> Optional[Environment]:
    """Get a cached module environment if available"""
    with _MODULE_CACHE_LOCK:
        return _MODULE_CACHE.get(module_path)

def cache_module(module_path: str, module_env: Environment) -> None:
    """Cache a loaded module environment"""
    with _MODULE_CACHE_LOCK:
        _MODULE_CACHE[module_path] = module_env

def clear_module_cache() -> None:
    """Clear the entire module cache"""
    with _MODULE_CACHE_LOCK:
        _MODULE_CACHE.clear()

def invalidate_module(module_path: str) -> None:
    """Invalidate a single module entry from the cache (if present)"""
    norm = normalize_path(module_path)
    with _MODULE_CACHE_LOCK:
        if norm in _MODULE_CACHE:
            del _MODULE_CACHE[norm]

def list_cached_modules() -> list[str]:
    """Return a list of normalized module paths currently cached"""
    with _MODULE_CACHE_LOCK:
        return list(_MODULE_CACHE.keys())

def get_module_candidates(file_path: str) -> list[str]:
    """Get candidate paths for a module, checking zpm_modules etc."""
    candidates = []
    if os.path.isabs(file_path):
        candidates.append(file_path)
    else:
        candidates.append(os.path.join(os.getcwd(), file_path))
        candidates.append(os.path.join(os.getcwd(), 'zpm_modules', file_path))

    # Try adding typical extensions
    return candidates + [c + ext for c in list(candidates) for ext in ['', '.zx', '.zexus']]

def normalize_path(path: str) -> str:
    """Normalize a path for consistent cache keys"""
    return os.path.abspath(os.path.expanduser(path))