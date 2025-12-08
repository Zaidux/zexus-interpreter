"""
Tests for ecosystem features.
"""

import unittest
from datetime import datetime
from src.zexus.ecosystem import (
    DependencyType, PackageVersion, PackageDependency, PackageMetadata,
    PackageRegistry, PackageManager, PerformanceProfiler, FunctionProfile,
    PluginMarketplace, EcosystemManager,
    get_ecosystem, get_package_manager, get_profiler, get_marketplace
)


class TestPackageVersion(unittest.TestCase):
    """Test package version."""
    
    def test_create_version(self):
        """Test creating version."""
        version = PackageVersion("1.0.0", datetime.now())
        self.assertEqual(version.version, "1.0.0")
    
    def test_exact_version_match(self):
        """Test exact version matching."""
        version = PackageVersion("1.0.0", datetime.now())
        self.assertTrue(version.is_compatible_with("1.0.0"))
        self.assertFalse(version.is_compatible_with("1.0.1"))
    
    def test_wildcard_match(self):
        """Test wildcard version matching."""
        version = PackageVersion("1.0.0", datetime.now())
        self.assertTrue(version.is_compatible_with("*"))


class TestPackageDependency(unittest.TestCase):
    """Test package dependency."""
    
    def test_create_dependency(self):
        """Test creating dependency."""
        dep = PackageDependency("lodash", "^1.0.0", DependencyType.RUNTIME)
        self.assertEqual(dep.name, "lodash")
        self.assertEqual(dep.type, DependencyType.RUNTIME)
    
    def test_dependency_satisfaction(self):
        """Test dependency version satisfaction."""
        dep = PackageDependency("lodash", "^1.0.0")
        self.assertTrue(dep.is_satisfied_by("1.0.0"))
        self.assertTrue(dep.is_satisfied_by("1.5.0"))
        self.assertFalse(dep.is_satisfied_by("2.0.0"))
    
    def test_optional_dependency(self):
        """Test optional dependency."""
        dep = PackageDependency("optional-lib", "1.0", optional=True)
        self.assertTrue(dep.optional)


class TestPackageMetadata(unittest.TestCase):
    """Test package metadata."""
    
    def test_create_metadata(self):
        """Test creating metadata."""
        metadata = PackageMetadata(
            name="myapp",
            version="1.0.0",
            author="John Doe",
            description="Test app"
        )
        self.assertEqual(metadata.name, "myapp")
        self.assertEqual(metadata.version, "1.0.0")
    
    def test_package_hash(self):
        """Test package hash."""
        metadata = PackageMetadata("app", "1.0.0")
        hash1 = metadata.get_hash()
        
        metadata2 = PackageMetadata("app", "1.0.0")
        hash2 = metadata2.get_hash()
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)
    
    def test_add_dependency(self):
        """Test adding dependencies."""
        metadata = PackageMetadata("app", "1.0.0")
        dep = PackageDependency("lodash", "^1.0.0")
        metadata.dependencies["lodash"] = dep
        
        self.assertIn("lodash", metadata.dependencies)


class TestPackageRegistry(unittest.TestCase):
    """Test package registry."""
    
    def test_create_registry(self):
        """Test creating registry."""
        registry = PackageRegistry()
        self.assertEqual(len(registry.packages), 0)
    
    def test_publish_package(self):
        """Test publishing package."""
        registry = PackageRegistry()
        metadata = PackageMetadata("app", "1.0.0")
        
        registry.publish(metadata)
        self.assertIn("app", registry.packages)
    
    def test_find_package(self):
        """Test finding package."""
        registry = PackageRegistry()
        metadata = PackageMetadata("app", "1.0.0")
        registry.publish(metadata)
        
        found = registry.find_package("app", "1.0.0")
        self.assertIsNotNone(found)
        self.assertEqual(found.version, "1.0.0")
    
    def test_find_compatible_version(self):
        """Test finding compatible version."""
        registry = PackageRegistry()
        registry.publish(PackageMetadata("app", "1.0.0"))
        registry.publish(PackageMetadata("app", "1.5.0"))
        registry.publish(PackageMetadata("app", "2.0.0"))
        
        version = registry.find_compatible_version("app", "^1.0.0")
        self.assertIsNotNone(version)
        self.assertTrue(version.startswith("1"))
    
    def test_resolve_simple_dependency(self):
        """Test resolving simple dependency."""
        registry = PackageRegistry()
        
        # Create package with dependency
        dep = PackageDependency("lodash", "1.0.0")
        metadata = PackageMetadata("app", "1.0.0")
        metadata.dependencies["lodash"] = dep
        
        # Publish both
        registry.publish(metadata)
        registry.publish(PackageMetadata("lodash", "1.0.0"))
        
        deps = registry.resolve_dependencies(metadata)
        self.assertIn("lodash@1.0.0", deps)
    
    def test_list_packages(self):
        """Test listing packages."""
        registry = PackageRegistry()
        registry.publish(PackageMetadata("app1", "1.0.0"))
        registry.publish(PackageMetadata("app2", "1.0.0"))
        
        packages = registry.list_packages()
        self.assertEqual(len(packages), 2)
    
    def test_list_versions(self):
        """Test listing versions."""
        registry = PackageRegistry()
        registry.publish(PackageMetadata("app", "1.0.0"))
        registry.publish(PackageMetadata("app", "1.1.0"))
        registry.publish(PackageMetadata("app", "2.0.0"))
        
        versions = registry.list_versions("app")
        self.assertEqual(len(versions), 3)


class TestPackageManager(unittest.TestCase):
    """Test package manager."""
    
    def test_create_manager(self):
        """Test creating manager."""
        registry = PackageRegistry()
        manager = PackageManager(registry)
        self.assertEqual(len(manager.installed), 0)
    
    def test_install_package(self):
        """Test installing package."""
        registry = PackageRegistry()
        registry.publish(PackageMetadata("app", "1.0.0"))
        
        manager = PackageManager(registry)
        success = manager.install("app", "1.0.0")
        
        self.assertTrue(success)
        self.assertTrue(manager.is_installed("app", "1.0.0"))
    
    def test_install_with_dependencies(self):
        """Test installing with dependencies."""
        registry = PackageRegistry()
        
        # Create packages
        dep_pkg = PackageMetadata("lodash", "1.0.0")
        registry.publish(dep_pkg)
        
        app = PackageMetadata("app", "1.0.0")
        app.dependencies["lodash"] = PackageDependency("lodash", "1.0.0")
        registry.publish(app)
        
        manager = PackageManager(registry)
        success = manager.install("app", "1.0.0")
        
        self.assertTrue(success)
        self.assertTrue(manager.is_installed("lodash"))
    
    def test_uninstall_package(self):
        """Test uninstalling package."""
        registry = PackageRegistry()
        registry.publish(PackageMetadata("app", "1.0.0"))
        
        manager = PackageManager(registry)
        manager.install("app")
        self.assertTrue(manager.is_installed("app"))
        
        success = manager.uninstall("app")
        self.assertTrue(success)
        self.assertFalse(manager.is_installed("app"))
    
    def test_verify_integrity(self):
        """Test package integrity verification."""
        registry = PackageRegistry()
        metadata = PackageMetadata("app", "1.0.0")
        registry.publish(metadata)
        
        manager = PackageManager(registry)
        manager.install("app")
        
        valid = manager.verify_integrity("app")
        self.assertTrue(valid)


class TestFunctionProfile(unittest.TestCase):
    """Test function profile."""
    
    def test_create_profile(self):
        """Test creating profile."""
        profile = FunctionProfile("test_func")
        self.assertEqual(profile.call_count, 0)
    
    def test_record_call(self):
        """Test recording call."""
        profile = FunctionProfile("test_func")
        profile.record_call(0.001, 100, 50)
        
        self.assertEqual(profile.call_count, 1)
        self.assertAlmostEqual(profile.total_time, 0.001)
        self.assertEqual(profile.total_args_size, 100)
    
    def test_average_time(self):
        """Test average time calculation."""
        profile = FunctionProfile("test_func")
        profile.record_call(0.001)
        profile.record_call(0.003)
        
        avg = profile.get_avg_time()
        self.assertAlmostEqual(avg, 0.002)
    
    def test_min_max_time(self):
        """Test min/max tracking."""
        profile = FunctionProfile("test_func")
        profile.record_call(0.001)
        profile.record_call(0.005)
        profile.record_call(0.002)
        
        self.assertAlmostEqual(profile.min_time, 0.001)
        self.assertAlmostEqual(profile.max_time, 0.005)


class TestPerformanceProfiler(unittest.TestCase):
    """Test performance profiler."""
    
    def test_create_profiler(self):
        """Test creating profiler."""
        profiler = PerformanceProfiler()
        self.assertEqual(len(profiler.profiles), 0)
    
    def test_profile_function(self):
        """Test profiling function."""
        profiler = PerformanceProfiler()
        profile = profiler.profile_function("test")
        
        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "test")
    
    def test_record_call(self):
        """Test recording call."""
        profiler = PerformanceProfiler()
        profiler.record_call("test", 0.001)
        profiler.record_call("test", 0.002)
        
        profile = profiler.get_profile("test")
        self.assertEqual(profile.call_count, 2)
    
    def test_slowest_functions(self):
        """Test finding slowest functions."""
        profiler = PerformanceProfiler()
        profiler.record_call("fast", 0.001)
        profiler.record_call("slow", 0.100)
        profiler.record_call("slow", 0.050)
        
        slowest = profiler.get_slowest_functions(1)
        self.assertEqual(slowest[0][0], "slow")
    
    def test_hottest_functions(self):
        """Test finding hottest functions."""
        profiler = PerformanceProfiler()
        profiler.record_call("hot", 0.001)
        for _ in range(5):
            profiler.record_call("hot", 0.001)
        profiler.record_call("cold", 0.001)
        
        hottest = profiler.get_hottest_functions(1)
        self.assertEqual(hottest[0][0], "hot")
    
    def test_profiling_hook(self):
        """Test profiling hooks."""
        profiler = PerformanceProfiler()
        
        called = []
        hook = lambda p, d: called.append(True)
        profiler.register_hook(hook)
        
        profiler.record_call("test", 0.001)
        self.assertGreater(len(called), 0)


class TestPluginMarketplace(unittest.TestCase):
    """Test plugin marketplace."""
    
    def test_create_marketplace(self):
        """Test creating marketplace."""
        marketplace = PluginMarketplace()
        self.assertEqual(len(marketplace.plugins), 0)
    
    def test_publish_plugin(self):
        """Test publishing plugin."""
        marketplace = PluginMarketplace()
        plugin = marketplace.MarketplacePlugin(
            name="json-plugin",
            version="1.0.0",
            author="Author",
            description="JSON support",
            category="utilities"
        )
        
        marketplace.publish_plugin(plugin)
        self.assertIn("json-plugin", marketplace.plugins)
    
    def test_search_by_name(self):
        """Test searching by name."""
        marketplace = PluginMarketplace()
        plugin = marketplace.MarketplacePlugin(
            name="json-plugin",
            version="1.0.0",
            author="Author",
            description="JSON",
            category="utilities"
        )
        marketplace.publish_plugin(plugin)
        
        results = marketplace.search_by_name("json")
        self.assertEqual(len(results), 1)
    
    def test_search_by_category(self):
        """Test searching by category."""
        marketplace = PluginMarketplace()
        plugin = marketplace.MarketplacePlugin(
            name="test-plugin",
            version="1.0.0",
            author="Author",
            description="Test",
            category="testing"
        )
        marketplace.publish_plugin(plugin)
        
        results = marketplace.search_by_category("testing")
        self.assertEqual(len(results), 1)
    
    def test_rate_plugin(self):
        """Test rating plugin."""
        marketplace = PluginMarketplace()
        plugin = marketplace.MarketplacePlugin(
            name="test",
            version="1.0.0",
            author="Author",
            description="Test",
            category="utilities"
        )
        marketplace.publish_plugin(plugin)
        
        success = marketplace.rate_plugin("test", 4.5)
        self.assertTrue(success)
        
        rating, count = marketplace.get_rating("test")
        self.assertAlmostEqual(rating, 4.5)
        self.assertEqual(count, 1)
    
    def test_get_trending(self):
        """Test getting trending plugins."""
        marketplace = PluginMarketplace()
        
        for i in range(3):
            plugin = marketplace.MarketplacePlugin(
                name=f"plugin{i}",
                version="1.0.0",
                author="Author",
                description="Test",
                category="utilities"
            )
            plugin.downloads = i * 100
            marketplace.publish_plugin(plugin)
        
        trending = marketplace.get_trending(1)
        self.assertEqual(trending[0].name, "plugin2")


class TestEcosystemManager(unittest.TestCase):
    """Test ecosystem manager."""
    
    def test_create_ecosystem(self):
        """Test creating ecosystem."""
        ecosystem = EcosystemManager()
        self.assertIsNotNone(ecosystem.package_manager)
        self.assertIsNotNone(ecosystem.profiler)
        self.assertIsNotNone(ecosystem.marketplace)
    
    def test_get_ecosystem_stats(self):
        """Test getting ecosystem stats."""
        ecosystem = EcosystemManager()
        ecosystem.profiler.record_call("test", 0.001)
        
        stats = ecosystem.get_ecosystem_stats()
        self.assertIn("profiled_functions", stats)
        self.assertGreater(stats["profiled_functions"], 0)


class TestGlobalFunctions(unittest.TestCase):
    """Test global ecosystem functions."""
    
    def test_get_ecosystem_global(self):
        """Test getting global ecosystem."""
        ecosystem = get_ecosystem()
        self.assertIsNotNone(ecosystem)
    
    def test_get_package_manager_global(self):
        """Test getting global package manager."""
        pm = get_package_manager()
        self.assertIsNotNone(pm)
    
    def test_get_profiler_global(self):
        """Test getting global profiler."""
        profiler = get_profiler()
        self.assertIsNotNone(profiler)
    
    def test_get_marketplace_global(self):
        """Test getting global marketplace."""
        marketplace = get_marketplace()
        self.assertIsNotNone(marketplace)


if __name__ == "__main__":
    unittest.main()
