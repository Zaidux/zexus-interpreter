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