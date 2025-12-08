"""
Plugin system for Zexus interpreter.

Enables third-party extensions through hooks and capability declarations.
Plugins are self-contained modules that extend the language without
modifying core functionality.
"""

from typing import Dict, List, Callable, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import os
import sys


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    author: str = ""
    description: str = ""
    requires: List[str] = field(default_factory=list)  # Capabilities required
    provides: List[str] = field(default_factory=list)  # Capabilities provided
    hooks: List[str] = field(default_factory=list)  # Hook names registered
    config: Dict[str, Any] = field(default_factory=dict)  # Configuration schema

    @classmethod
    def from_dict(cls, data: dict) -> 'PluginMetadata':
        """Create metadata from dictionary."""
        return cls(
            name=data.get('name', ''),
            version=data.get('version', '1.0.0'),
            author=data.get('author', ''),
            description=data.get('description', ''),
            requires=data.get('requires', []),
            provides=data.get('provides', []),
            hooks=data.get('hooks', []),
            config=data.get('config', {})
        )


@dataclass
class Hook:
    """A registered hook handler."""
    name: str
    handler: Callable
    plugin_name: str
    priority: int = 0  # Higher priority executes first


class PluginManager:
    """
    Manages plugin loading, registration, and execution.
    
    Handles:
    - Plugin discovery and loading
    - Hook registration and execution
    - Capability tracking and validation
    - Dependency resolution
    - Sandbox enforcement
    """
    
    def __init__(self):
        """Initialize the plugin manager."""
        self.loaded_plugins: Dict[str, PluginMetadata] = {}
        self.hooks: Dict[str, List[Hook]] = defaultdict(list)
        self.capabilities: Set[str] = set()
        self.plugin_modules: Dict[str, Any] = {}  # Loaded plugin modules
        self.config: Dict[str, Dict[str, Any]] = {}  # Per-plugin config
        
        # Builtin capabilities always available
        self.capabilities.add("core")
        
    def load_plugin(self, module_path: str, config: Optional[Dict[str, Any]] = None) -> PluginMetadata:
        """
        Load a plugin from a module path.
        
        Args:
            module_path: Path to plugin module (.zx file or directory)
            config: Configuration dictionary for the plugin
            
        Returns:
            PluginMetadata object
            
        Raises:
            FileNotFoundError: If plugin module not found
            ValueError: If plugin metadata invalid or dependencies unmet
        """
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Plugin module not found: {module_path}")
        
        # Note: Actual implementation will parse .zx module
        # For now, this is the interface
        metadata = PluginMetadata(
            name="placeholder",
            version="1.0.0"
        )
        
        # Validate dependencies
        for required_cap in metadata.requires:
            if required_cap not in self.capabilities:
                raise ValueError(f"Plugin {metadata.name} requires unavailable capability: {required_cap}")
        
        # Store metadata and config
        self.loaded_plugins[metadata.name] = metadata
        if config:
            self.config[metadata.name] = config
        
        # Add provided capabilities
        for cap in metadata.provides:
            self.capabilities.add(cap)
        
        return metadata
    
    def register_hook(self, hook_name: str, handler: Callable, 
                     plugin_name: str, priority: int = 0) -> None:
        """
        Register a hook handler.
        
        Args:
            hook_name: Name of the hook (e.g., "pre_eval", "import_resolver")
            handler: Callable that handles the hook
            plugin_name: Name of plugin registering the hook
            priority: Execution priority (higher = earlier)
        """
        hook = Hook(name=hook_name, handler=handler, plugin_name=plugin_name, priority=priority)
        self.hooks[hook_name].append(hook)
        
        # Sort by priority (descending)
        self.hooks[hook_name].sort(key=lambda h: h.priority, reverse=True)
    
    def call_hooks(self, hook_name: str, *args, **kwargs) -> Any:
        """
        Call all registered handlers for a hook.
        
        Handlers are called in priority order (highest first).
        If a handler returns non-None, stop and return that value.
        Otherwise, return the first argument (usually the transformed input).
        
        Args:
            hook_name: Name of the hook to call
            *args: Arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
            
        Returns:
            Result from first non-None handler return, or first argument
        """
        if hook_name not in self.hooks:
            # No handlers registered, return first arg unchanged
            return args[0] if args else None
        
        result = args[0] if args else None
        
        for hook in self.hooks[hook_name]:
            try:
                # Call handler with arguments
                handler_result = hook.handler(*args, **kwargs)
                if handler_result is not None:
                    # Handler returned a value, update result
                    result = handler_result
                    # Update args[0] for next handler
                    if args:
                        args = (result,) + args[1:]
            except Exception as e:
                # Log hook error but continue
                print(f"Error in hook {hook_name} from {hook.plugin_name}: {e}", 
                      file=sys.stderr)
        
        return result
    
    def check_capability(self, capability: str) -> bool:
        """
        Check if a capability is available.
        
        Args:
            capability: Capability name to check
            
        Returns:
            True if capability is available, False otherwise
        """
        return capability in self.capabilities
    
    def grant_capability(self, capability: str) -> None:
        """
        Manually grant a capability.
        
        Args:
            capability: Capability name to grant
        """
        self.capabilities.add(capability)
    
    def has_plugin(self, plugin_name: str) -> bool:
        """Check if a plugin is loaded."""
        return plugin_name in self.loaded_plugins
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get metadata for a loaded plugin."""
        return self.loaded_plugins.get(plugin_name)
    
    def get_capabilities(self) -> List[str]:
        """Get list of available capabilities."""
        return sorted(list(self.capabilities))
    
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugin names."""
        return sorted(list(self.loaded_plugins.keys()))
    
    def get_hooks(self, hook_name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get registered hooks.
        
        Args:
            hook_name: If specified, return handlers for specific hook
            
        Returns:
            Dict of hook_name -> [plugin_names] or filtered results
        """
        if hook_name:
            return {hook_name: [h.plugin_name for h in self.hooks.get(hook_name, [])]}
        
        result = {}
        for name, hooks_list in self.hooks.items():
            result[name] = [h.plugin_name for h in hooks_list]
        return result


class PluginGlobalObject:
    """
    The `plugin` global object exposed to plugin code.
    
    Provides interface for:
    - Registering hooks
    - Declaring capabilities
    - Accessing metadata
    - Introspection
    """
    
    def __init__(self, manager: PluginManager, current_plugin: str):
        """
        Initialize the plugin global.
        
        Args:
            manager: PluginManager instance
            current_plugin: Name of the currently-executing plugin
        """
        self.manager = manager
        self.current_plugin = current_plugin
    
    def register_hook(self, hook_name: str, handler: Callable) -> None:
        """Register a hook handler for the current plugin."""
        self.manager.register_hook(hook_name, handler, self.current_plugin)
    
    def grant_capability(self, capability: str) -> None:
        """Declare that this plugin grants a capability."""
        self.manager.grant_capability(capability)
    
    def has_capability(self, capability: str) -> bool:
        """Check if a capability is available."""
        return self.manager.check_capability(capability)
    
    def get_hooks(self) -> Dict[str, List[str]]:
        """Get all registered hooks."""
        return self.manager.get_hooks()
    
    def get_capabilities(self) -> List[str]:
        """Get all available capabilities."""
        return self.manager.get_capabilities()
    
    def get_loaded_plugins(self) -> List[str]:
        """Get list of loaded plugins."""
        return self.manager.get_loaded_plugins()
    
    def metadata(self) -> PluginMetadata:
        """Get metadata for current plugin."""
        return self.manager.get_plugin_metadata(self.current_plugin)
    
    def load(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load another plugin by name.
        
        Returns:
            True if loaded, False if already loaded
        """
        if self.manager.has_plugin(plugin_name):
            return False
        
        # Would resolve plugin path and load
        # Simplified for now
        return True
