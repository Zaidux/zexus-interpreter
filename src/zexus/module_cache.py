"""
Module cache for Zexus interpreter.

This module provides caching functionality for loaded Zexus modules to avoid
re-parsing and re-evaluating modules that have already been loaded.
"""

import os
import threading
from typing import Dict, Optional, Any, Tuple
from .object import Environment

# Module cache stores: (environment, bytecode, ast)
_MODULE_CACHE: Dict[str, Tuple[Environment, Any, Any]] = {}
_MODULE_CACHE_LOCK = threading.Lock()

# Contract AST cache by source hash
_CONTRACT_AST_CACHE: Dict[str, Any] = {}
_CONTRACT_AST_LOCK = threading.Lock()

def get_cached_module(module_path: str) -> Optional[Tuple[Environment, Any, Any]]:
    """Get a cached module (environment, bytecode, ast) if available"""
    with _MODULE_CACHE_LOCK:
        return _MODULE_CACHE.get(module_path)

def cache_module(module_path: str, module_env: Environment, bytecode: Any = None, ast: Any = None) -> None:
    """Cache a loaded module environment with optional bytecode and AST"""
    with _MODULE_CACHE_LOCK:
        _MODULE_CACHE[module_path] = (module_env, bytecode, ast)

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

def get_cached_contract_ast(source_hash: str) -> Optional[Any]:
    """Get a cached contract AST by source hash"""
    with _CONTRACT_AST_LOCK:
        return _CONTRACT_AST_CACHE.get(source_hash)

def cache_contract_ast(source_hash: str, ast: Any) -> None:
    """Cache a parsed contract AST"""
    with _CONTRACT_AST_LOCK:
        _CONTRACT_AST_CACHE[source_hash] = ast

def get_cached_contract_ast(source_hash: str) -> Optional[Any]:
    """Get a cached contract AST by source hash"""
    with _CONTRACT_AST_LOCK:
        return _CONTRACT_AST_CACHE.get(source_hash)

def cache_contract_ast(source_hash: str, ast: Any) -> None:
    """Cache a parsed contract AST"""
    with _CONTRACT_AST_LOCK:
        _CONTRACT_AST_CACHE[source_hash] = ast

def get_cached_contract_ast(source_hash: str) -> Optional[Any]:
    """Get a cached contract AST by source hash"""
    with _CONTRACT_AST_LOCK:
        return _CONTRACT_AST_CACHE.get(source_hash)

def cache_contract_ast(source_hash: str, ast: Any) -> None:
    """Cache a parsed contract AST"""
    with _CONTRACT_AST_LOCK:
        _CONTRACT_AST_CACHE[source_hash] = ast

def get_module_candidates(file_path: str, importer_file: str = None) -> list[str]:
    """Get candidate paths for a module, checking zpm_modules etc.
    
    Args:
        file_path: The module path to import
        importer_file: The absolute path of the file doing the importing (for relative imports)
    
    Returns:
        List of candidate absolute paths to check
    """
    candidates = []
    
    if os.path.isabs(file_path):
        # Absolute path - use as-is
        candidates.append(file_path)
    else:
        # Relative path - resolve based on importer's directory
        if importer_file and file_path.startswith('./'):
            # Relative to the importing file's directory
            importer_dir = os.path.dirname(importer_file)
            resolved_path = os.path.join(importer_dir, file_path[2:])  # Remove './'
            candidates.append(resolved_path)
            # Also consider project-root relative paths like "./tests/..."
            candidates.append(os.path.join(os.getcwd(), file_path[2:]))
        elif importer_file and file_path.startswith('../'):
            # Parent directory relative to importing file
            importer_dir = os.path.dirname(importer_file)
            resolved_path = os.path.join(importer_dir, file_path)
            candidates.append(resolved_path)
        else:
            # Relative to current working directory
            candidates.append(os.path.join(os.getcwd(), file_path))
        
        # Also check zpm_modules directory
        candidates.append(os.path.join(os.getcwd(), 'zpm_modules', file_path))

    # Try adding typical extensions (.zx, .zexus)
    extended_candidates = []
    for candidate in candidates:
        extended_candidates.append(candidate)
        if not candidate.endswith(('.zx', '.zexus')):
            extended_candidates.append(candidate + '.zx')
            extended_candidates.append(candidate + '.zexus')
    
    return extended_candidates

def normalize_path(path: str) -> str:
    """Normalize a path for consistent cache keys"""
    return os.path.abspath(os.path.expanduser(path))