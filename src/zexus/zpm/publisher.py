"""
Package Publisher - Handles package publishing to registry
"""
import os
import json
import tarfile
from pathlib import Path
from typing import Dict


class PackagePublisher:
    """Handles package publishing"""
    
    def __init__(self, registry):
        self.registry = registry
    
    def publish(self, project_root: Path, config: Dict) -> bool:
        """Publish a package"""
        name = config.get("name")
        version = config.get("version")
        
        if not name or not version:
            print("❌ Package name and version required")
            return False
        
        print(f"📤 Publishing {name}@{version}...")
        
        # Collect files
        files = self._collect_files(project_root, config)
        
        if not files:
            print("❌ No files to publish")
            return False
        
        # Create tarball
        tarball_path = self._create_tarball(project_root, name, version, files)
        
        # Publish to registry
        package_data = {
            **config,
            "tarball": str(tarball_path),
            "files": files
        }
        
        success = self.registry.publish_package(package_data, files)
        
        if success:
            print(f"✅ Published {name}@{version}")
            print(f"📦 Tarball: {tarball_path}")
        else:
            print(f"❌ Failed to publish {name}@{version}")
        
        return success
    
    def _collect_files(self, project_root: Path, config: Dict) -> list:
        """Collect files to include in package"""
        files = []
        
        # Include main file
        main_file = config.get("main", "main.zx")
        if (project_root / main_file).exists():
            files.append(main_file)
        
        # Include all .zx files (excluding node_modules equivalent)
        for zx_file in project_root.rglob("*.zx"):
            relative = zx_file.relative_to(project_root)
            # Exclude zpm_modules, tests, etc.
            if not any(part.startswith(".") or part in ["zpm_modules", "node_modules"] 
                      for part in relative.parts):
                files.append(str(relative))
        
        # Include package.json
        files.append("zexus.json")
        
        # Include README if exists
        for readme in ["README.md", "README.txt", "README"]:
            if (project_root / readme).exists():
                files.append(readme)
                break
        
        # Include LICENSE if exists
        if (project_root / "LICENSE").exists():
            files.append("LICENSE")
        
        return files
    
    def _create_tarball(self, project_root: Path, name: str, version: str, files: list) -> Path:
        """Create a tarball of the package"""
        tarball_name = f"{name}-{version}.tar.gz"
        tarball_path = project_root / tarball_name
        
        with tarfile.open(tarball_path, "w:gz") as tar:
            for file in files:
                file_path = project_root / file
                if file_path.exists():
                    tar.add(file_path, arcname=f"{name}/{file}")
        
        return tarball_path
