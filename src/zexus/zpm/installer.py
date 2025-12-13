"""
Package Installer - Handles package installation and dependencies
"""
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional


class PackageInstaller:
    """Handles package installation"""
    
    def __init__(self, install_dir: Path):
        self.install_dir = Path(install_dir)
        self.install_dir.mkdir(parents=True, exist_ok=True)
    
    def install(self, package_info: Dict) -> bool:
        """Install a package"""
        name = package_info["name"]
        version = package_info["version"]
        pkg_type = package_info.get("type", "normal")
        
        target_dir = self.install_dir / name
        
        # Check if already installed
        if target_dir.exists():
            existing_pkg = target_dir / "zexus.json"
            if existing_pkg.exists():
                with open(existing_pkg) as f:
                    existing_info = json.load(f)
                    if existing_info.get("version") == version:
                        print(f"ℹ️  {name}@{version} already installed")
                        return True
        
        # Create package directory
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # For built-in packages, create stub
        if pkg_type == "builtin":
            self._install_builtin(name, version, target_dir)
        else:
            # TODO: Download and extract package
            self._install_from_source(package_info, target_dir)
        
        return True
    
    def _install_builtin(self, name: str, version: str, target_dir: Path):
        """Install a built-in package"""
        # Create package.json
        pkg_json = {
            "name": name,
            "version": version,
            "type": "builtin",
            "main": "index.zx"
        }
        
        with open(target_dir / "zexus.json", "w") as f:
            json.dump(pkg_json, f, indent=2)
        
        # Create stub main file
        main_file = target_dir / "index.zx"
        main_file.write_text(f"""// {name} - Built-in Zexus package
// Version: {version}

// This is a built-in package provided by Zexus
// Functions are available globally when imported

export {{
    // Package exports will be defined here
}}
""")
    
    def _install_from_source(self, package_info: Dict, target_dir: Path):
        """Install package from source/tarball"""
        # TODO: Implement actual download and extraction
        
        # For now, create placeholder
        pkg_json = {
            "name": package_info["name"],
            "version": package_info["version"],
            "description": package_info.get("description", ""),
        }
        
        with open(target_dir / "zexus.json", "w") as f:
            json.dump(pkg_json, f, indent=2)
        
        main_file = target_dir / "index.zx"
        main_file.write_text(f"""// {package_info['name']}
// Placeholder - package installation from remote sources not yet implemented
""")
    
    def uninstall(self, package: str) -> bool:
        """Uninstall a package"""
        target_dir = self.install_dir / package
        
        if not target_dir.exists():
            print(f"⚠️  Package {package} not installed")
            return False
        
        shutil.rmtree(target_dir)
        return True
    
    def is_installed(self, package: str) -> bool:
        """Check if a package is installed"""
        return (self.install_dir / package).exists()
    
    def get_installed_version(self, package: str) -> Optional[str]:
        """Get installed version of a package"""
        pkg_json = self.install_dir / package / "zexus.json"
        if not pkg_json.exists():
            return None
        
        with open(pkg_json) as f:
            info = json.load(f)
            return info.get("version")
