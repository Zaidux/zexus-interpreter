"""
Package Manager Core - Main ZPM interface
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from .registry import PackageRegistry
from .installer import PackageInstaller
from .publisher import PackagePublisher


class PackageManager:
    """Main package manager interface"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.zpm_dir = self.project_root / "zpm_modules"
        self.config_file = self.project_root / "zexus.json"
        self.lock_file = self.project_root / "zexus-lock.json"
        
        self.registry = PackageRegistry()
        self.installer = PackageInstaller(self.zpm_dir)
        self.publisher = PackagePublisher(self.registry)
        
    def init(self, name: str = None, version: str = "1.5.0") -> Dict:
        """Initialize a new Zexus project with package.json"""
        if self.config_file.exists():
            print(f"⚠️  {self.config_file} already exists")
            return self.load_config()
        
        if not name:
            name = self.project_root.name
        
        config = {
            "name": name,
            "version": version,
            "description": "",
            "main": "main.zx",
            "dependencies": {},
            "devDependencies": {},
            "scripts": {
                "test": "zx run tests/test_*.zx",
                "build": "zx compile main.zx"
            },
            "author": "",
            "license": "MIT"
        }
        
        self.save_config(config)
        print(f"✅ Created {self.config_file}")
        print(f"📝 Edit {self.config_file} to customize your project")
        
        return config
    
    def install(self, package: str = None, dev: bool = False) -> bool:
        """Install a package or all packages from zexus.json"""
        if not package:
            # Install all dependencies
            return self.install_all()
        
        # Install specific package
        print(f"📦 Installing {package}...")
        
        # Parse package@version
        if "@" in package:
            name, version = package.split("@", 1)
        else:
            name, version = package, "latest"
        
        # Get package info from registry
        pkg_info = self.registry.get_package(name, version)
        if not pkg_info:
            print(f"❌ Package {name}@{version} not found")
            return False
        
        # Install package
        success = self.installer.install(pkg_info)
        if not success:
            return False
        
        # Update config
        config = self.load_config()
        dep_key = "devDependencies" if dev else "dependencies"
        if dep_key not in config:
            config[dep_key] = {}
        
        config[dep_key][name] = version
        self.save_config(config)
        
        # Update lock file
        self.update_lock_file()
        
        print(f"✅ Installed {name}@{version}")
        return True
    
    def install_all(self) -> bool:
        """Install all dependencies from zexus.json"""
        config = self.load_config()
        if not config:
            print("❌ No zexus.json found. Run 'zpm init' first.")
            return False
        
        dependencies = config.get("dependencies", {})
        dev_dependencies = config.get("devDependencies", {})
        all_deps = {**dependencies, **dev_dependencies}
        
        if not all_deps:
            print("✅ No dependencies to install")
            return True
        
        print(f"📦 Installing {len(all_deps)} package(s)...")
        
        success_count = 0
        for name, version in all_deps.items():
            pkg_info = self.registry.get_package(name, version)
            if pkg_info and self.installer.install(pkg_info):
                success_count += 1
            else:
                print(f"⚠️  Failed to install {name}@{version}")
        
        self.update_lock_file()
        print(f"✅ Installed {success_count}/{len(all_deps)} package(s)")
        return success_count == len(all_deps)
    
    def uninstall(self, package: str) -> bool:
        """Uninstall a package"""
        print(f"🗑️  Uninstalling {package}...")
        
        success = self.installer.uninstall(package)
        if not success:
            return False
        
        # Update config
        config = self.load_config()
        if package in config.get("dependencies", {}):
            del config["dependencies"][package]
        if package in config.get("devDependencies", {}):
            del config["devDependencies"][package]
        
        self.save_config(config)
        self.update_lock_file()
        
        print(f"✅ Uninstalled {package}")
        return True
    
    def list(self) -> List[Dict]:
        """List installed packages"""
        if not self.zpm_dir.exists():
            return []
        
        packages = []
        for item in self.zpm_dir.iterdir():
            if item.is_dir():
                pkg_json = item / "zexus.json"
                if pkg_json.exists():
                    with open(pkg_json) as f:
                        pkg_info = json.load(f)
                        packages.append({
                            "name": pkg_info.get("name", item.name),
                            "version": pkg_info.get("version", "unknown"),
                            "path": str(item)
                        })
        
        return packages
    
    def search(self, query: str) -> List[Dict]:
        """Search for packages in registry"""
        return self.registry.search(query)
    
    def publish(self) -> bool:
        """Publish current package to registry"""
        config = self.load_config()
        if not config:
            print("❌ No zexus.json found")
            return False
        
        return self.publisher.publish(self.project_root, config)
    
    def load_config(self) -> Optional[Dict]:
        """Load zexus.json"""
        if not self.config_file.exists():
            return None
        
        with open(self.config_file) as f:
            return json.load(f)
    
    def save_config(self, config: Dict):
        """Save zexus.json"""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def update_lock_file(self):
        """Update zexus-lock.json with installed packages"""
        packages = self.list()
        lock_data = {
            "lockfileVersion": 1,
            "packages": {
                pkg["name"]: {
                    "version": pkg["version"],
                    "path": pkg["path"]
                }
                for pkg in packages
            }
        }
        
        with open(self.lock_file, "w") as f:
            json.dump(lock_data, f, indent=2)
