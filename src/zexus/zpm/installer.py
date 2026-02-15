"""
Package Installer - Handles package installation and dependencies
"""
import os
import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Optional


class PackageInstaller:
    """Handles package installation"""
    
    def __init__(self, install_dir: Path, registry=None):
        self.install_dir = Path(install_dir)
        self.install_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry  # PackageRegistry for tarball downloads
    
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
            self._install_from_source(package_info, target_dir)
        
        return True
    
    def _install_builtin(self, name: str, version: str, target_dir: Path):
        """Install a built-in package"""
        pkg_json = {
            "name": name,
            "version": version,
            "type": "builtin",
            "main": "index.zx"
        }
        
        with open(target_dir / "zexus.json", "w") as f:
            json.dump(pkg_json, f, indent=2)
        
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
        """Install package from a remote tarball or local path.

        If the package metadata contains a ``tarball`` key pointing to a
        local file, that tarball is extracted directly.  Otherwise the
        installer attempts to download the tarball from the registry via
        ``registry.download_tarball(name, version)``.
        """
        name = package_info["name"]
        version = package_info["version"]

        # 1. Check for a local tarball path in the metadata
        tarball_path = package_info.get("tarball")
        if tarball_path and os.path.isfile(tarball_path):
            self._extract_tarball(Path(tarball_path), target_dir, name)
            return

        # 2. Try to download from the remote registry
        if self.registry is not None:
            downloaded = self.registry.download_tarball(name, version)
            if downloaded and downloaded.exists():
                print(f"📦 Downloaded {name}@{version}")
                self._extract_tarball(downloaded, target_dir, name)
                return

        # 3. Fallback — create placeholder with metadata
        print(f"⚠️  Could not download {name}@{version} — creating placeholder")
        pkg_json = {
            "name": name,
            "version": version,
            "description": package_info.get("description", ""),
        }
        with open(target_dir / "zexus.json", "w") as f:
            json.dump(pkg_json, f, indent=2)
        main_file = target_dir / "index.zx"
        main_file.write_text(f'// {name}@{version} — placeholder (tarball not available)\n')

    @staticmethod
    def _extract_tarball(tarball_path: Path, target_dir: Path, package_name: str):
        """Extract a ``.tar.gz`` tarball into *target_dir*.

        Tarballs created by ``PackagePublisher`` contain files under a
        ``<package_name>/`` prefix.  We strip that prefix so the files
        land directly in *target_dir*.
        """
        with tarfile.open(tarball_path, "r:gz") as tar:
            # Security: filter out absolute paths and ..'s
            safe_members = []
            prefix = f"{package_name}/"
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    continue
                # Strip the package-name prefix from the archive path
                if member.name.startswith(prefix):
                    member.name = member.name[len(prefix):]
                elif member.name == package_name:
                    continue  # skip the bare directory entry
                safe_members.append(member)
            tar.extractall(path=target_dir, members=safe_members)
    
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
