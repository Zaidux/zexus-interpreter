"""
Package Registry - Manages package discovery and metadata
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional


class PackageRegistry:
    """Package registry for discovering and managing packages"""
    
    def __init__(self, registry_url: str = None):
        self.registry_url = registry_url or os.environ.get(
            "ZPM_REGISTRY",
            "https://registry.zexus.dev"  # Default registry
        )
        
        # Local cache directory
        self.cache_dir = Path.home() / ".zpm" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Built-in packages
        self.builtin_packages = self._load_builtin_packages()
    
    def _load_builtin_packages(self) -> Dict:
        """Load built-in package definitions"""
        return {
            "std": {
                "name": "std",
                "version": "0.1.0",
                "description": "Zexus standard library",
                "type": "builtin",
                "files": []
            },
            "crypto": {
                "name": "crypto",
                "version": "0.1.0",
                "description": "Cryptography utilities",
                "type": "builtin",
                "files": []
            },
            "web": {
                "name": "web",
                "version": "0.1.0",
                "description": "Web framework for Zexus",
                "type": "builtin",
                "files": []
            },
            "blockchain": {
                "name": "blockchain",
                "version": "0.1.0",
                "description": "Blockchain utilities and helpers",
                "type": "builtin",
                "files": []
            }
        }
    
    def get_package(self, name: str, version: str = "latest") -> Optional[Dict]:
        """Get package metadata from registry"""
        # Check built-in packages first
        if name in self.builtin_packages:
            return self.builtin_packages[name]
        
        # Check cache
        cache_file = self.cache_dir / f"{name}-{version}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        # TODO: Fetch from remote registry
        # For now, return None if not found
        return None
    
    def search(self, query: str) -> List[Dict]:
        """Search for packages"""
        results = []
        
        # Search built-in packages
        for name, pkg in self.builtin_packages.items():
            if query.lower() in name.lower() or query.lower() in pkg.get("description", "").lower():
                results.append(pkg)
        
        # TODO: Search remote registry
        
        return results
    
    def publish_package(self, package_data: Dict, files: List[str]) -> bool:
        """Publish a package to registry"""
        # For now, just cache locally
        name = package_data["name"]
        version = package_data["version"]
        
        cache_file = self.cache_dir / f"{name}-{version}.json"
        with open(cache_file, "w") as f:
            json.dump(package_data, f, indent=2)
        
        print(f"✅ Package cached locally at {cache_file}")
        print(f"⚠️  Remote registry publication not yet implemented")
        
        # TODO: Upload to remote registry
        return True
    
    def get_versions(self, package: str) -> List[str]:
        """Get all available versions of a package"""
        if package in self.builtin_packages:
            return [self.builtin_packages[package]["version"]]
        
        # TODO: Query remote registry
        return []
