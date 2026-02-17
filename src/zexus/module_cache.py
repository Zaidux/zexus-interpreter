"""
Module cache for Zexus interpreter.

This module provides caching functionality for loaded Zexus modules to avoid
re-parsing and re-evaluating modules that have already been loaded.
"""

import os
import hashlib
import threading
from typing import Dict, Optional, Any, Tuple, Set
from .object import Environment

# Module cache stores: (environment, bytecode, ast)
_MODULE_CACHE: Dict[str, Tuple[Environment, Any, Any]] = {}
_MODULE_CACHE_LOCK = threading.Lock()

# Circular-import detection: set of normalized paths currently being loaded
_LOADING_SET: Set[str] = set()
_LOADING_SET_LOCK = threading.Lock()

# Contract AST cache by source hash
_CONTRACT_AST_CACHE: Dict[str, Any] = {}
_CONTRACT_AST_LOCK = threading.Lock()


class CircularImportError(Exception):
    """Raised when a circular import dependency is detected."""
    def __init__(self, path: str, chain: Optional[list] = None):
        self.path = path
        self.chain = chain or []
        if chain:
            cycle = " -> ".join(chain + [path])
            msg = f"Circular import detected: {cycle}"
        else:
            msg = f"Circular import detected while loading: {path}"
        super().__init__(msg)


def begin_loading(module_path: str) -> None:
    """Mark *module_path* as currently being loaded.
    
    Raises ``CircularImportError`` if the module is already in the loading set
    (i.e. a circular dependency has been encountered).
    """
    norm = normalize_path(module_path)
    with _LOADING_SET_LOCK:
        if norm in _LOADING_SET:
            raise CircularImportError(norm, list(_LOADING_SET))
        _LOADING_SET.add(norm)


def end_loading(module_path: str) -> None:
    """Remove *module_path* from the loading set after it has finished loading."""
    norm = normalize_path(module_path)
    with _LOADING_SET_LOCK:
        _LOADING_SET.discard(norm)


def is_loading(module_path: str) -> bool:
    """Check whether *module_path* is currently being loaded."""
    norm = normalize_path(module_path)
    with _LOADING_SET_LOCK:
        return norm in _LOADING_SET

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
            # For bare imports (no ./ or ../ prefix), check relative to importer first
            if importer_file:
                importer_dir = os.path.dirname(importer_file)
                candidates.append(os.path.join(importer_dir, file_path))
            # Then relative to current working directory
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


# ---------------------------------------------------------------------------
# Module Pre-compilation
# ---------------------------------------------------------------------------

def _extract_import_paths(node) -> list:
    """Recursively walk an AST and return all import file paths."""
    from . import zexus_ast
    paths = []
    if isinstance(node, zexus_ast.Program):
        for stmt in getattr(node, 'statements', []):
            paths.extend(_extract_import_paths(stmt))
    elif isinstance(node, zexus_ast.UseStatement):
        fp = node.file_path
        if hasattr(fp, 'value'):
            fp = fp.value
        if isinstance(fp, str):
            paths.append(fp)
    elif isinstance(node, zexus_ast.FromStatement):
        fp = node.file_path
        if hasattr(fp, 'value'):
            fp = fp.value
        if isinstance(fp, str):
            paths.append(fp)
    elif isinstance(node, (zexus_ast.BlockStatement,)):
        for stmt in getattr(node, 'statements', []):
            paths.extend(_extract_import_paths(stmt))
    elif isinstance(node, zexus_ast.IfStatement):
        paths.extend(_extract_import_paths(node.consequence))
        if node.alternative:
            paths.extend(_extract_import_paths(node.alternative))
        for cond, body in getattr(node, 'elif_parts', []):
            paths.extend(_extract_import_paths(body))
    elif isinstance(node, (zexus_ast.WhileStatement, zexus_ast.ForEachStatement)):
        paths.extend(_extract_import_paths(getattr(node, 'body', None) or getattr(node, 'block', None)))
    elif isinstance(node, zexus_ast.FunctionStatement):
        paths.extend(_extract_import_paths(getattr(node, 'body', None)))
    elif isinstance(node, zexus_ast.ActionStatement):
        paths.extend(_extract_import_paths(getattr(node, 'body', None)))
    elif isinstance(node, zexus_ast.TryCatchStatement):
        paths.extend(_extract_import_paths(getattr(node, 'try_body', None)))
        paths.extend(_extract_import_paths(getattr(node, 'catch_body', None)))
        if getattr(node, 'finally_body', None):
            paths.extend(_extract_import_paths(node.finally_body))
    return [p for p in paths if p]


def precompile_modules(program, main_file: str, compile_bytecode: bool = True) -> dict:
    """Pre-compile all imported modules before execution.
    
    Walks the AST of *program*, resolves every ``use`` / ``from`` import,
    lexes + parses each module file, optionally compiles to bytecode, and
    stores everything in the global ``_MODULE_CACHE``.  Processes modules
    recursively so transitive dependencies are also cached.
    
    Args:
        program: The parsed AST (Program node) of the main file.
        main_file: Absolute path of the main source file.
        compile_bytecode: If True, also compile each module to VM bytecode.
    
    Returns:
        dict mapping normalised module path → (ast, bytecode_or_None).
    """
    from .lexer import Lexer
    from .parser import Parser
    from .stdlib_integration import is_stdlib_module
    from .builtin_modules import is_builtin_module

    compiler_mod = None
    if compile_bytecode:
        try:
            from .vm.compiler import compile_ast_to_bytecode
            compiler_mod = compile_ast_to_bytecode
        except Exception:
            compiler_mod = None

    results: Dict[str, Any] = {}
    visited: Set[str] = set()

    def _resolve_and_cache(import_path: str, importer_file: str):
        """Resolve a single import and cache it, then recurse."""
        # Skip stdlib / builtin modules — they aren't file-based
        if is_stdlib_module(import_path) or is_builtin_module(import_path):
            return

        candidates = get_module_candidates(import_path, importer_file)
        resolved = None
        for cand in candidates:
            norm = normalize_path(cand)
            if norm in visited:
                import warnings
                warnings.warn(
                    f"Circular import detected during pre-compilation: {import_path} "
                    f"(resolved to {norm})",
                    stacklevel=2,
                )
                return  # Already processed (or in progress — avoids cycles)
            if os.path.isfile(cand):
                resolved = cand
                break

        if resolved is None:
            return  # Unresolvable — runtime will report the error

        norm = normalize_path(resolved)
        if norm in visited:
            return
        visited.add(norm)

        # Already cached from a previous run?
        cached = get_cached_module(norm)
        if cached is not None:
            results[norm] = (cached[2], cached[1])  # (ast, bytecode)
            # Still recurse into dependencies of the cached module
            if cached[2] is not None:
                sub_paths = _extract_import_paths(cached[2])
                for sp in sub_paths:
                    _resolve_and_cache(sp, resolved)
            return

        # Read, lex, parse
        try:
            with open(resolved, 'r', encoding='utf-8') as f:
                source = f.read()
        except (OSError, IOError):
            return

        try:
            lexer = Lexer(source, filename=resolved)
            parser = Parser(lexer, 'universal', enable_advanced_strategies=True)
            mod_ast = parser.parse_program()
        except Exception:
            return  # Parse failure — runtime will handle

        # Optionally compile to bytecode
        mod_bytecode = None
        if compiler_mod is not None:
            try:
                mod_bytecode = compiler_mod(mod_ast, optimize=True)
            except Exception:
                pass  # Compilation failure is non-fatal; interpreter fallback

        # Cache a placeholder env + ast + bytecode so runtime finds them.
        # Mark the env as pre-compiled but not yet evaluated — the runtime
        # will execute the bytecode/AST to populate it on first use.
        mod_env = Environment()
        mod_env.set("__file__", resolved)
        mod_env.set("__MODULE__", os.path.splitext(os.path.basename(resolved))[0])
        mod_env._precompiled = True  # Marker for eval_use_statement
        cache_module(norm, mod_env, mod_bytecode, mod_ast)
        results[norm] = (mod_ast, mod_bytecode)

        # Recurse into sub-imports
        sub_paths = _extract_import_paths(mod_ast)
        for sp in sub_paths:
            _resolve_and_cache(sp, resolved)

    # Kick off from the main program's imports
    import_paths = _extract_import_paths(program)
    for ip in import_paths:
        _resolve_and_cache(ip, main_file)

    return results