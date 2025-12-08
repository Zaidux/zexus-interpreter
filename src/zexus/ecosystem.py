"""
Ecosystem features: package management, profiling hooks, marketplace integration.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib


class DependencyType(Enum):
    """Type of dependency."""
    RUNTIME = "runtime"
    DEVELOPMENT = "development"
    OPTIONAL = "optional"
    PEER = "peer"


@dataclass
class PackageVersion:
    """Package version information."""
    version: str
    released: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_compatible_with(self, required_version: str) -> bool:
        """Check version compatibility."""
        # Simplified: exact match or wildcard
        if required_version == "*":
            return True
        if required_version.startswith("^"):
            return self.version.startswith(required_version[1])
        return self.version == required_version


@dataclass
class PackageDependency:
    """Package dependency specification."""
    name: str
    version: str  # Version constraint (e.g., "^1.0.0", "1.2.3", "*")
    type: DependencyType = DependencyType.RUNTIME
    optional: bool = False
    
    def is_satisfied_by(self, installed_version: str) -> bool:
        """Check if installed version satisfies constraint."""
        if self.version == "*":
            return True
        if self.version.startswith("^"):
            return installed_version.startswith(self.version[1])
        return installed_version == self.version


@dataclass
class PackageMetadata:
    """Package metadata."""
    name: str
    version: str
    author: str = ""
    description: str = ""
    license: str = ""
    homepage: str = ""
    repository: str = ""
    keywords: List[str] = field(default_factory=list)
    dependencies: Dict[str, PackageDependency] = field(default_factory=dict)
    
    def get_hash(self) -> str:
        """Get package hash for integrity checking."""
        content = f"{self.name}@{self.version}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class PackageRegistry:
    """Central package registry."""
    
    def __init__(self):
        """Initialize package registry."""
        self.packages: Dict[str, Dict[str, PackageMetadata]] = {}  # name -> version -> metadata
        self.cache: Dict[str, bytes] = {}  # package_id -> bytecode
    
    def publish(self, metadata: PackageMetadata, bytecode: Optional[bytes] = None):
        """Publish a package."""
        if metadata.name not in self.packages:
            self.packages[metadata.name] = {}
        
        self.packages[metadata.name][metadata.version] = metadata
        
        if bytecode:
            package_id = f"{metadata.name}@{metadata.version}"
            self.cache[package_id] = bytecode
    
    def find_package(self, name: str, version: str) -> Optional[PackageMetadata]:
        """Find package by name and version."""
        if name not in self.packages:
            return None
        return self.packages[name].get(version)
    
    def find_compatible_version(self, name: str, constraint: str) -> Optional[str]:
        """Find compatible version for constraint."""
        if name not in self.packages:
            return None
        
        versions = list(self.packages[name].keys())
        
        # Simplified: match exact or take latest
        if constraint == "*":
            return versions[-1] if versions else None
        
        if constraint.startswith("^"):
            prefix = constraint[1]
            matching = [v for v in versions if v.startswith(prefix)]
            return matching[-1] if matching else None
        
        if constraint in versions:
            return constraint
        
        return None
    
    def resolve_dependencies(self, package: PackageMetadata,
                            resolved: Optional[Set[str]] = None) -> Dict[str, PackageMetadata]:
        """Resolve package dependencies recursively."""
        if resolved is None:
            resolved = set()
        
        result = {}
        package_id = f"{package.name}@{package.version}"
        
        if package_id in resolved:
            return result
        
        resolved.add(package_id)
        
        for dep_name, dep in package.dependencies.items():
            version = self.find_compatible_version(dep_name, dep.version)
            if version:
                dep_pkg = self.find_package(dep_name, version)
                if dep_pkg:
                    result[f"{dep_name}@{version}"] = dep_pkg
                    # Recursively resolve
                    result.update(self.resolve_dependencies(dep_pkg, resolved))
        
        return result
    
    def list_packages(self) -> List[str]:
        """List all package names."""
        return list(self.packages.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """List all versions of a package."""
        return list(self.packages.get(name, {}).keys())


class PackageManager:
    """Package management system."""
    
    def __init__(self, registry: Optional[PackageRegistry] = None):
        """Initialize package manager."""
        self.registry = registry or PackageRegistry()
        self.installed: Dict[str, str] = {}  # name -> version
        self.lock_file: Dict[str, Tuple[str, str]] = {}  # name -> (version, hash)
    
    def install(self, name: str, version: str = "*") -> bool:
        """Install package."""
        resolved_version = self.registry.find_compatible_version(name, version)
        if not resolved_version:
            return False
        
        package = self.registry.find_package(name, resolved_version)
        if not package:
            return False
        
        # Resolve and install dependencies
        deps = self.registry.resolve_dependencies(package)
        for dep_id, dep_pkg in deps.items():
            self.installed[dep_pkg.name] = dep_pkg.version
            self.lock_file[dep_pkg.name] = (dep_pkg.version, dep_pkg.get_hash())
        
        # Install package itself
        self.installed[name] = resolved_version
        self.lock_file[name] = (resolved_version, package.get_hash())
        
        return True
    
    def uninstall(self, name: str) -> bool:
        """Uninstall package."""
        if name in self.installed:
            del self.installed[name]
            if name in self.lock_file:
                del self.lock_file[name]
            return True
        return False
    
    def is_installed(self, name: str, version: Optional[str] = None) -> bool:
        """Check if package is installed."""
        if name not in self.installed:
            return False
        if version:
            return self.installed[name] == version
        return True
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Get all installed packages."""
        return self.installed.copy()
    
    def verify_integrity(self, name: str) -> bool:
        """Verify package integrity."""
        if name not in self.lock_file:
            return False
        
        version, expected_hash = self.lock_file[name]
        package = self.registry.find_package(name, version)
        
        if not package:
            return False
        
        return package.get_hash() == expected_hash


class PerformanceProfiler:
    """Performance profiler for functions."""
    
    def __init__(self):
        """Initialize profiler."""
        self.profiles: Dict[str, 'FunctionProfile'] = {}
        self.hooks: List[Callable] = []
    
    def profile_function(self, name: str) -> 'FunctionProfile':
        """Create profile for function."""
        if name not in self.profiles:
            self.profiles[name] = FunctionProfile(name)
        return self.profiles[name]
    
    def register_hook(self, hook: Callable):
        """Register profiling hook."""
        self.hooks.append(hook)
    
    def record_call(self, name: str, duration: float, args_size: int = 0, return_size: int = 0):
        """Record function call."""
        if name not in self.profiles:
            self.profile_function(name)
        
        profile = self.profiles[name]
        profile.record_call(duration, args_size, return_size)
        
        # Trigger hooks
        for hook in self.hooks:
            try:
                hook(profile, duration)
            except Exception:
                pass
    
    def get_profile(self, name: str) -> Optional['FunctionProfile']:
        """Get function profile."""
        return self.profiles.get(name)
    
    def get_slowest_functions(self, count: int = 10) -> List[Tuple[str, float]]:
        """Get slowest functions."""
        items = [(name, prof.total_time) for name, prof in self.profiles.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:count]
    
    def get_hottest_functions(self, count: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently called functions."""
        items = [(name, prof.call_count) for name, prof in self.profiles.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:count]


@dataclass
class FunctionProfile:
    """Profile for a single function."""
    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    total_args_size: int = 0
    total_return_size: int = 0
    
    def record_call(self, duration: float, args_size: int = 0, return_size: int = 0):
        """Record a call."""
        self.call_count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.total_args_size += args_size
        self.total_return_size += return_size
    
    def get_avg_time(self) -> float:
        """Get average call time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def get_avg_args_size(self) -> float:
        """Get average arguments size."""
        return self.total_args_size / self.call_count if self.call_count > 0 else 0.0


class PluginMarketplace:
    """Plugin marketplace for discovering and managing plugins."""
    
    def __init__(self):
        """Initialize marketplace."""
        self.plugins: Dict[str, 'MarketplacePlugin'] = {}
        self.categories: Dict[str, List[str]] = {}  # category -> plugin names
        self.ratings: Dict[str, Tuple[float, int]] = {}  # plugin_name -> (avg_rating, count)
    
    @dataclass
    class MarketplacePlugin:
        """Plugin listing in marketplace."""
        name: str
        version: str
        author: str
        description: str
        category: str
        requires_capabilities: Set[str] = field(default_factory=set)
        homepage: str = ""
        downloads: int = 0
        
        def __hash__(self):
            return hash(f"{self.name}@{self.version}")
    
    def publish_plugin(self, plugin: 'PluginMarketplace.MarketplacePlugin'):
        """Publish plugin to marketplace."""
        self.plugins[plugin.name] = plugin
        
        if plugin.category not in self.categories:
            self.categories[plugin.category] = []
        if plugin.name not in self.categories[plugin.category]:
            self.categories[plugin.category].append(plugin.name)
    
    def search_by_name(self, query: str) -> List['PluginMarketplace.MarketplacePlugin']:
        """Search plugins by name."""
        results = []
        for name, plugin in self.plugins.items():
            if query.lower() in name.lower():
                results.append(plugin)
        return results
    
    def search_by_category(self, category: str) -> List['PluginMarketplace.MarketplacePlugin']:
        """Search plugins by category."""
        names = self.categories.get(category, [])
        return [self.plugins[name] for name in names if name in self.plugins]
    
    def get_trending(self, count: int = 10) -> List['PluginMarketplace.MarketplacePlugin']:
        """Get trending plugins (most downloaded)."""
        plugins = sorted(
            self.plugins.values(),
            key=lambda p: p.downloads,
            reverse=True
        )
        return plugins[:count]
    
    def rate_plugin(self, name: str, rating: float) -> bool:
        """Rate a plugin (1-5 stars)."""
        if name not in self.plugins:
            return False
        
        if rating < 1 or rating > 5:
            return False
        
        if name not in self.ratings:
            self.ratings[name] = (rating, 1)
        else:
            avg, count = self.ratings[name]
            new_avg = (avg * count + rating) / (count + 1)
            self.ratings[name] = (new_avg, count + 1)
        
        return True
    
    def get_rating(self, name: str) -> Optional[Tuple[float, int]]:
        """Get plugin rating."""
        return self.ratings.get(name)


class EcosystemManager:
    """Central ecosystem manager."""
    
    def __init__(self):
        """Initialize ecosystem manager."""
        self.package_manager = PackageManager()
        self.profiler = PerformanceProfiler()
        self.marketplace = PluginMarketplace()
    
    def get_package_manager(self) -> PackageManager:
        """Get package manager."""
        return self.package_manager
    
    def get_profiler(self) -> PerformanceProfiler:
        """Get profiler."""
        return self.profiler
    
    def get_marketplace(self) -> PluginMarketplace:
        """Get marketplace."""
        return self.marketplace
    
    def get_ecosystem_stats(self) -> Dict[str, Any]:
        """Get ecosystem statistics."""
        packages = self.package_manager.get_installed_packages()
        
        total_calls = sum(p.call_count for p in self.profiler.profiles.values())
        total_time = sum(p.total_time for p in self.profiler.profiles.values())
        
        return {
            "installed_packages": len(packages),
            "total_function_calls": total_calls,
            "total_execution_time": total_time,
            "profiled_functions": len(self.profiler.profiles),
            "marketplace_plugins": len(self.marketplace.plugins),
            "plugin_categories": len(self.marketplace.categories)
        }


# Global ecosystem manager
_global_ecosystem = EcosystemManager()


def get_ecosystem() -> EcosystemManager:
    """Get global ecosystem manager."""
    return _global_ecosystem


def get_package_manager() -> PackageManager:
    """Get global package manager."""
    return _global_ecosystem.package_manager


def get_profiler() -> PerformanceProfiler:
    """Get global profiler."""
    return _global_ecosystem.profiler


def get_marketplace() -> PluginMarketplace:
    """Get global marketplace."""
    return _global_ecosystem.marketplace
