# environment_manager.py

import os
from pathlib import Path
from .environment import Environment

class EnvironmentManager:
    def __init__(self, base_path=None):
        """Initialize the environment manager
        
        Args:
            base_path: Base path for resolving module paths. Defaults to cwd.
        """
        self.base_path = Path(base_path or os.getcwd())
        self.module_cache = {}
        self.search_paths = [
            self.base_path,
            self.base_path / "zpm_modules",  # Keep zpm_modules as primary module path
            self.base_path / "lib"
        ]
        self._debug = False
        
    def normalize_path(self, path):
        """Normalize a file path"""
        return str(Path(path).resolve())
        
    def resolve_module_path(self, file_path):
        """Resolve a module path to an absolute path
        
        Args:
            file_path: File path as a string (from UseStatement/FromStatement)
            
        Returns:
            Absolute Path object or None if not found
        """
        try:
            # Handle absolute paths
            if str(file_path).startswith("/"):
                path = Path(file_path)
                if path.exists():
                    return path.resolve()
                    
            # Handle relative paths
            if str(file_path).startswith("./") or str(file_path).startswith("../"):
                path = (self.base_path / file_path).resolve()
                if path.exists():
                    return path
                    
            # Search in module paths
            for search_path in self.search_paths:
                # Try exact path
                path = search_path / file_path
                if path.exists():
                    return path.resolve()
                    
                # Try with .zx extension
                path_with_ext = search_path / f"{file_path}.zx"
                if path_with_ext.exists():
                    return path_with_ext.resolve()
                    
                # Try with index.zx in directory
                path_index = search_path / file_path / "index.zx"
                if path_index.exists():
                    return path_index.resolve()
                    
        except Exception as e:
            if self._debug:
                print(f"[ENV] Error resolving module path '{file_path}': {e}")
            return None
            
        return None
        
    def get_module(self, file_path):
        """Get a cached module environment
        
        Args:
            file_path: Absolute path to module file
            
        Returns:
            Cached Environment or None
        """
        normalized = self.normalize_path(file_path)
        return self.module_cache.get(normalized)
        
    def cache_module(self, file_path, module_env):
        """Cache a module environment
        
        Args:
            file_path: Absolute path to module file
            module_env: Environment instance to cache
        """
        normalized = self.normalize_path(file_path)
        self.module_cache[normalized] = module_env
        
    def clear_cache(self):
        """Clear the module cache"""
        self.module_cache.clear()
        if self._debug:
            print("[ENV] Module cache cleared")
            
    def add_search_path(self, path):
        """Add a path to module search paths
        
        Args:
            path: Directory path to add to search paths
        """
        path = Path(path).resolve()
        if path not in self.search_paths:
            self.search_paths.append(path)
            if self._debug:
                print(f"[ENV] Added search path: {path}")
                
    def enable_debug(self):
        """Enable debug logging"""
        self._debug = True
        
    def disable_debug(self):
        """Disable debug logging"""
        self._debug = False