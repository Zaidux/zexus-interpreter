# module_manager.py
import os
from pathlib import Path

class ModuleManager:
    def __init__(self, base_path=None):
        self.base_path = Path(base_path or os.getcwd())
        self.module_cache = {}
        self.search_paths = [
            self.base_path,
            self.base_path / "zpm_modules",  # Keep existing zpm_modules path
            self.base_path / "modules",
            self.base_path / "lib"
        ]
        self._debug = False
        self._max_find_results = 50

    def normalize_path(self, path):
        """Normalize a module path"""
        return str(Path(path).resolve()).replace("\\", "/").strip()

    def resolve_module_path(self, path, current_dir=""):
        """Resolve a module path to an absolute path"""
        # Support existing behavior
        if isinstance(current_dir, str) and current_dir:
            base = Path(current_dir)
        else:
            base = self.base_path

        try:
            if path.startswith("./"):
                # Relative path
                resolved = (base / path[2:]).resolve()
            elif path.startswith("/"):
                # Absolute path
                resolved = Path(path).resolve()
            else:
                # Try zpm_modules first (existing behavior)
                zpm_path = (self.base_path / "zpm_modules" / path).resolve()
                if zpm_path.exists():
                    return self.normalize_path(zpm_path)

                # Search in other paths
                for search_path in self.search_paths:
                    test_path = (search_path / path).resolve()
                    if test_path.exists():
                        return self.normalize_path(test_path)
                    
                    # Try with .zx extension
                    test_path_zx = (search_path / f"{path}.zx").resolve()
                    if test_path_zx.exists():
                        return self.normalize_path(test_path_zx)

                # Default to zpm_modules (maintain compatibility)
                resolved = zpm_path

            return self.normalize_path(resolved)
        except (TypeError, ValueError):
            return None

    def get_module(self, path):
        """Get a cached module or None"""
        return self.module_cache.get(self.normalize_path(path))

    def cache_module(self, path, module_env):
        """Cache a module environment"""
        self.module_cache[self.normalize_path(path)] = module_env

    def clear_cache(self):
        """Clear the module cache"""
        self.module_cache.clear()
        if self._debug:
            print("[MOD] Module cache cleared")

    def add_search_path(self, path):
        """Add a directory to module search paths"""
        path = Path(path).resolve()
        if path not in self.search_paths:
            self.search_paths.append(path)
            if self._debug:
                print(f"[MOD] Added search path: {path}")

    def enable_debug(self):
        """Enable debug logging"""
        self._debug = True

    def disable_debug(self):
        """Disable debug logging"""
        self._debug = False

    def find_files(self, pattern, current_dir=None, scope=None, max_results=None):
        """Search for files matching pattern within known search paths.

        Args:
            pattern: File name or relative pattern to search for.
            current_dir: Directory of the importing file (for relative priority).
            scope: Optional directory hint (absolute or relative) to limit search.
            max_results: Maximum matches to return (defaults to manager limit).

        Returns:
            List of normalized absolute paths matching the pattern.
        """
        max_results = max_results or self._max_find_results

        if not pattern:
            return []

        # Absolute pattern short-circuit
        try:
            pattern_path = Path(pattern)
        except TypeError:
            return []

        if pattern_path.is_absolute():
            if pattern_path.exists():
                return [self.normalize_path(pattern_path)]
            # Try with known extensions
            for ext in (".zx", ".zexus"):
                candidate = pattern_path.with_suffix(ext)
                if candidate.exists():
                    return [self.normalize_path(candidate)]
            return []

        roots = []

        def _append_root(path_candidate):
            if not path_candidate:
                return
            try:
                resolved = Path(path_candidate).resolve()
            except (OSError, RuntimeError):
                return
            if resolved not in roots and resolved.exists():
                roots.append(resolved)

        # Scope hint first (if provided)
        if scope:
            scope_path = Path(scope)
            if not scope_path.is_absolute() and current_dir:
                _append_root(Path(current_dir) / scope_path)
            _append_root(self.base_path / scope_path)
            if scope_path.is_absolute():
                _append_root(scope_path)

        # Current file directory has highest priority after scope
        if current_dir:
            _append_root(current_dir)

        # Standard search paths (project root, modules, lib, etc.)
        for search_path in self.search_paths:
            _append_root(search_path)

        # Deduplicate while preserving order
        seen = set()
        ordered_roots = []
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            ordered_roots.append(root)

        matches = []
        match_set = set()

        def _record(path_obj):
            normalized = self.normalize_path(path_obj)
            if normalized not in match_set:
                match_set.add(normalized)
                matches.append(normalized)

        # Exact relative matches via direct join first
        for root in ordered_roots:
            candidate = root / pattern_path
            if candidate.exists() and candidate.is_file():
                _record(candidate)
                if len(matches) >= max_results:
                    return matches
            else:
                for ext in (".zx", ".zexus"):
                    candidate_ext = candidate.with_suffix(ext)
                    if candidate_ext.exists() and candidate_ext.is_file():
                        _record(candidate_ext)
                        if len(matches) >= max_results:
                            return matches

        # Fallback to glob search across roots
        if pattern_path.name:
            search_pattern = str(pattern_path)
            basename_pattern = pattern_path.name
        else:
            search_pattern = pattern
            basename_pattern = pattern

        for root in ordered_roots:
            if len(matches) >= max_results:
                break
            try:
                iterator = root.rglob(search_pattern)
            except (ValueError, OSError):
                try:
                    iterator = root.rglob(basename_pattern)
                except (ValueError, OSError):
                    continue

            for found in iterator:
                if not found.is_file():
                    continue
                _record(found)
                if len(matches) >= max_results:
                    break

        return matches

# Create a default instance for backwards compatibility
_default_manager = ModuleManager()

# Expose existing API through default instance
def normalize_path(path):
    return _default_manager.normalize_path(path)

def resolve_module_path(path, current_dir=""):
    return _default_manager.resolve_module_path(path, current_dir)

def get_module(path):
    return _default_manager.get_module(path)

def cache_module(path, module_env):
    return _default_manager.cache_module(path, module_env)

def clear_cache():
    return _default_manager.clear_cache()


def find_files(pattern, current_dir=None, scope=None, max_results=None):
    return _default_manager.find_files(pattern, current_dir=current_dir, scope=scope, max_results=max_results)