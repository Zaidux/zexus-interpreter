"""
Unit tests for the plugin system.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.plugin_system import PluginManager, PluginMetadata, PluginGlobalObject


def test_plugin_metadata():
    """Test plugin metadata creation and manipulation."""
    metadata = PluginMetadata(
        name="test_plugin",
        version="1.0.0",
        author="Test Author",
        description="A test plugin",
        requires=["core"],
        provides=["test.feature"],
        hooks=["pre_eval"]
    )
    
    assert metadata.name == "test_plugin"
    assert metadata.version == "1.0.0"
    assert "core" in metadata.requires
    assert "test.feature" in metadata.provides
    assert "pre_eval" in metadata.hooks
    print("✓ plugin_metadata test passed")


def test_metadata_from_dict():
    """Test creating metadata from dictionary."""
    data = {
        "name": "json",
        "version": "1.0.0",
        "provides": ["json.parse", "json.stringify"],
        "requires": [],
    }
    
    metadata = PluginMetadata.from_dict(data)
    assert metadata.name == "json"
    assert metadata.version == "1.0.0"
    assert "json.parse" in metadata.provides
    print("✓ metadata_from_dict test passed")


def test_plugin_manager_initialization():
    """Test that plugin manager initializes correctly."""
    manager = PluginManager()
    
    assert "core" in manager.capabilities
    assert len(manager.loaded_plugins) == 0
    assert len(manager.hooks) == 0
    print("✓ plugin_manager_initialization test passed")


def test_grant_capability():
    """Test granting capabilities."""
    manager = PluginManager()
    
    manager.grant_capability("json.parse")
    assert manager.check_capability("json.parse")
    
    manager.grant_capability("crypto.hash")
    assert manager.check_capability("crypto.hash")
    
    assert not manager.check_capability("nonexistent")
    print("✓ grant_capability test passed")


def test_register_hook():
    """Test hook registration."""
    manager = PluginManager()
    
    def test_handler(*args, **kwargs):
        return "test_result"
    
    manager.register_hook("pre_eval", test_handler, "test_plugin")
    
    hooks = manager.get_hooks("pre_eval")
    assert "pre_eval" in hooks
    assert "test_plugin" in hooks["pre_eval"]
    print("✓ register_hook test passed")


def test_call_hooks_no_handlers():
    """Test calling hooks with no handlers."""
    manager = PluginManager()
    
    result = manager.call_hooks("pre_eval", "input_value")
    assert result == "input_value"
    print("✓ call_hooks_no_handlers test passed")


def test_call_hooks_with_handler():
    """Test calling hooks with registered handler."""
    manager = PluginManager()
    
    def transform_handler(value):
        return value.upper() if isinstance(value, str) else value
    
    manager.register_hook("transform", transform_handler, "test_plugin")
    
    result = manager.call_hooks("transform", "hello")
    assert result == "HELLO"
    print("✓ call_hooks_with_handler test passed")


def test_call_hooks_multiple_handlers():
    """Test calling hooks with multiple handlers."""
    manager = PluginManager()
    
    call_order = []
    
    def handler1(value):
        call_order.append("handler1")
        return value + "_1"
    
    def handler2(value):
        call_order.append("handler2")
        return value + "_2"
    
    manager.register_hook("chain", handler1, "plugin1")
    manager.register_hook("chain", handler2, "plugin2")
    
    result = manager.call_hooks("chain", "value")
    # Should call both handlers in order
    assert len(call_order) == 2
    assert "handler1" in call_order
    assert "handler2" in call_order
    print("✓ call_hooks_multiple_handlers test passed")


def test_hook_priority():
    """Test that hooks are executed in priority order."""
    manager = PluginManager()
    
    execution_order = []
    
    def handler_low(value):
        execution_order.append("low")
        return value
    
    def handler_high(value):
        execution_order.append("high")
        return value
    
    # Register low priority first
    manager.register_hook("priority_test", handler_low, "plugin_low", priority=0)
    # Register high priority second
    manager.register_hook("priority_test", handler_high, "plugin_high", priority=10)
    
    manager.call_hooks("priority_test", "value")
    
    # High priority should execute first
    assert execution_order[0] == "high"
    assert execution_order[1] == "low"
    print("✓ hook_priority test passed")


def test_get_capabilities():
    """Test capability listing."""
    manager = PluginManager()
    
    manager.grant_capability("feature1")
    manager.grant_capability("feature2")
    manager.grant_capability("feature3")
    
    caps = manager.get_capabilities()
    assert "core" in caps
    assert "feature1" in caps
    assert "feature2" in caps
    assert "feature3" in caps
    print("✓ get_capabilities test passed")


def test_plugin_global_object():
    """Test the plugin global object interface."""
    manager = PluginManager()
    manager.grant_capability("core")
    
    # Create a plugin metadata
    metadata = PluginMetadata(
        name="test_plugin",
        version="1.0.0",
        provides=["test.feature"]
    )
    manager.loaded_plugins["test_plugin"] = metadata
    
    # Create plugin global
    plugin_obj = PluginGlobalObject(manager, "test_plugin")
    
    # Test metadata access
    meta = plugin_obj.metadata()
    assert meta.name == "test_plugin"
    
    # Test capability checking
    assert plugin_obj.has_capability("core")
    
    # Test hook registration
    def dummy_handler(*args, **kwargs):
        pass
    plugin_obj.register_hook("test_hook", dummy_handler)
    
    hooks = plugin_obj.get_hooks()
    assert "test_hook" in hooks
    print("✓ plugin_global_object test passed")


def test_has_plugin():
    """Test plugin existence checking."""
    manager = PluginManager()
    
    metadata = PluginMetadata(name="myplug", version="1.0.0")
    manager.loaded_plugins["myplug"] = metadata
    
    assert manager.has_plugin("myplug")
    assert not manager.has_plugin("nonexistent")
    print("✓ has_plugin test passed")


def test_get_loaded_plugins():
    """Test retrieving list of loaded plugins."""
    manager = PluginManager()
    
    manager.loaded_plugins["plugin1"] = PluginMetadata(name="plugin1", version="1.0.0")
    manager.loaded_plugins["plugin2"] = PluginMetadata(name="plugin2", version="1.0.0")
    
    plugins = manager.get_loaded_plugins()
    assert "plugin1" in plugins
    assert "plugin2" in plugins
    assert len(plugins) == 2
    print("✓ get_loaded_plugins test passed")


if __name__ == '__main__':
    try:
        test_plugin_metadata()
        test_metadata_from_dict()
        test_plugin_manager_initialization()
        test_grant_capability()
        test_register_hook()
        test_call_hooks_no_handlers()
        test_call_hooks_with_handler()
        test_call_hooks_multiple_handlers()
        test_hook_priority()
        test_get_capabilities()
        test_plugin_global_object()
        test_has_plugin()
        test_get_loaded_plugins()
        print("\n✅ All plugin system tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
