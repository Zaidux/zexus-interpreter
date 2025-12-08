"""
Integration layer connecting all 10 phases to the evaluator.

This module provides hooks and integration points for:
- Phase 1: Modifiers
- Phase 2: Plugin System
- Phase 3: Capability-Based Security
- Phase 4: Virtual Filesystem
- Phase 5: Type System
- Phase 6: Metaprogramming
- Phase 7: Optimization
- Phase 9: Advanced Types
- Phase 10: Ecosystem
"""

from typing import Any, Dict, Optional, List, Callable
from ..plugin_system import PluginManager
from ..capability_system import CapabilityManager, DenyAllPolicy
from ..virtual_filesystem import VirtualFileSystemManager, SandboxBuilder
from ..type_system import TypeChecker, TypeInferencer
from ..metaprogramming import MetaRegistry, apply_all_macros
from ..optimization import BytecodeCompiler, OptimizationFramework
from ..advanced_types import TraitRegistry
from ..ecosystem import EcosystemManager


class EvaluatorIntegration:
    """Central integration point for all 10 phases."""
    
    def __init__(self):
        """Initialize all phase systems."""
        self.plugin_manager: PluginManager = PluginManager()
        self.capability_manager: CapabilityManager = CapabilityManager()
        self.vfs_manager: VirtualFileSystemManager = VirtualFileSystemManager()
        self.type_checker: TypeChecker = TypeChecker()
        self.type_inferencer: TypeInferencer = TypeInferencer()
        self.meta_registry: MetaRegistry = MetaRegistry()
        self.optimizer: OptimizationFramework = OptimizationFramework()
        self.trait_registry: TraitRegistry = TraitRegistry()
        self.ecosystem: EcosystemManager = EcosystemManager()
        
        # Execution context
        self.current_sandbox: Optional[SandboxBuilder] = None
        self.current_capabilities: set = set()
    
    def setup_default_security(self):
        """Setup default security policy (untrusted code)."""
        from ..capability_system import AllowAllPolicy
        self.capability_manager.set_policy(DenyAllPolicy())
    
    def setup_trusted_execution(self):
        """Setup trusted execution context (no restrictions)."""
        from ..capability_system import AllowAllPolicy
        self.capability_manager.set_policy(AllowAllPolicy())


# Global integration instance
_global_integration: Optional[EvaluatorIntegration] = None


def get_integration() -> EvaluatorIntegration:
    """Get or create global integration instance."""
    global _global_integration
    if _global_integration is None:
        _global_integration = EvaluatorIntegration()
        _global_integration.setup_default_security()
    return _global_integration


# ============================================================================
# PHASE 1: MODIFIER SUPPORT
# ============================================================================

class ModifierHandler:
    """Handle modifier-based behavior during evaluation."""
    
    @staticmethod
    def apply_modifiers(node: Any, evaluator_method: Callable, *args, **kwargs) -> Any:
        """Apply modifier effects to evaluation."""
        if not hasattr(node, 'modifiers'):
            return evaluator_method(*args, **kwargs)
        
        modifiers = node.modifiers if hasattr(node, 'modifiers') else []
        result = evaluator_method(*args, **kwargs)
        
        # INLINE modifier: Mark for inlining
        if 'inline' in modifiers or 'INLINE' in modifiers:
            if hasattr(result, '__dict__'):
                result.is_inline = True
        
        # ASYNC modifier: Mark for async execution
        if 'async' in modifiers or 'ASYNC' in modifiers:
            if hasattr(result, '__dict__'):
                result.is_async = True
        
        # SECURE modifier: Require security check
        if 'secure' in modifiers or 'SECURE' in modifiers:
            if hasattr(result, '__dict__'):
                result.is_secure = True
        
        # PURE modifier: No side effects
        if 'pure' in modifiers or 'PURE' in modifiers:
            if hasattr(result, '__dict__'):
                result.is_pure = True
        
        return result


# ============================================================================
# PHASE 2: PLUGIN HOOKS
# ============================================================================

class PluginHookHandler:
    """Trigger plugin hooks at evaluation points."""
    
    @staticmethod
    def before_action_call(action_name: str, args: Dict[str, Any]):
        """Trigger hook before action call."""
        integration = get_integration()
        integration.plugin_manager.call_hooks(
            "action.before_call",
            {"action": action_name, "args": args}
        )
    
    @staticmethod
    def after_action_call(action_name: str, result: Any):
        """Trigger hook after action call."""
        integration = get_integration()
        integration.plugin_manager.call_hooks(
            "action.after_call",
            {"action": action_name, "result": result}
        )
    
    @staticmethod
    def on_function_definition(func_name: str, func_obj: Any):
        """Trigger hook on function definition."""
        integration = get_integration()
        integration.plugin_manager.call_hooks(
            "function.definition",
            {"name": func_name, "function": func_obj}
        )
    
    @staticmethod
    def on_variable_assignment(var_name: str, value: Any):
        """Trigger hook on variable assignment."""
        integration = get_integration()
        integration.plugin_manager.call_hooks(
            "variable.assignment",
            {"name": var_name, "value": value}
        )


# ============================================================================
# PHASE 3: CAPABILITY CHECKING
# ============================================================================

class CapabilityChecker:
    """Check capabilities during evaluation."""
    
    @staticmethod
    def check_io_read(context: str = "default") -> bool:
        """Check if IO read capability is available."""
        integration = get_integration()
        allowed, reason = integration.capability_manager.check_capability(
            context, "io.read"
        )
        return allowed
    
    @staticmethod
    def check_io_write(context: str = "default") -> bool:
        """Check if IO write capability is available."""
        integration = get_integration()
        allowed, reason = integration.capability_manager.check_capability(
            context, "io.write"
        )
        return allowed
    
    @staticmethod
    def check_network(context: str = "default") -> bool:
        """Check if network capability is available."""
        integration = get_integration()
        allowed, reason = integration.capability_manager.check_capability(
            context, "network.tcp"
        )
        return allowed
    
    @staticmethod
    def require_capability(capability: str, context: str = "default"):
        """Require a capability, raise if not available."""
        integration = get_integration()
        integration.capability_manager.require_capability(context, capability)


# ============================================================================
# PHASE 4: VIRTUAL FILESYSTEM
# ============================================================================

class VirtualFilesystemHandler:
    """Handle filesystem access through VFS."""
    
    @staticmethod
    def resolve_file_path(virtual_path: str, context: str = "default") -> Optional[str]:
        """Resolve virtual path to real path with access check."""
        integration = get_integration()
        vfs = integration.vfs_manager.get_sandbox_filesystem(context)
        
        if vfs is None:
            return virtual_path  # No VFS, use path directly
        
        try:
            return vfs.resolve_path(virtual_path)
        except PermissionError:
            return None
    
    @staticmethod
    def check_file_access(virtual_path: str, operation: str = "read", context: str = "default") -> bool:
        """Check if file access is allowed."""
        integration = get_integration()
        vfs = integration.vfs_manager.get_sandbox_filesystem(context)
        
        if vfs is None:
            return True  # No VFS, allow access
        
        operation_upper = operation.upper()
        if operation_upper == "READ":
            return vfs.can_read(virtual_path)
        elif operation_upper == "WRITE":
            return vfs.can_write(virtual_path)
        return False


# ============================================================================
# PHASE 5: TYPE SYSTEM
# ============================================================================

class TypeSystemHandler:
    """Handle type checking and inference."""
    
    @staticmethod
    def infer_type(value: Any) -> str:
        """Infer type of a value."""
        integration = get_integration()
        type_spec = integration.type_inferencer.infer_type(value)
        return str(type_spec)
    
    @staticmethod
    def check_type(value: Any, type_spec: Any) -> bool:
        """Check if value matches type specification."""
        integration = get_integration()
        matches, reason = integration.type_checker.check_type(value, type_spec)
        return matches
    
    @staticmethod
    def validate_call(func_sig: Any, args: Dict[str, Any]) -> bool:
        """Validate function call against signature."""
        if hasattr(func_sig, 'validate_call'):
            integration = get_integration()
            valid, errors = func_sig.validate_call(args, integration.type_checker)
            return valid
        return True


# ============================================================================
# PHASE 6: METAPROGRAMMING
# ============================================================================

class MetaprogrammingHandler:
    """Handle metaprogramming features."""
    
    @staticmethod
    def apply_macros(ast_node: Any) -> Any:
        """Apply registered macros to AST."""
        integration = get_integration()
        return integration.meta_registry.apply_macros(ast_node)
    
    @staticmethod
    def register_macro(name: str, pattern: str, transformer: Callable):
        """Register a macro."""
        from ..metaprogramming import Macro
        integration = get_integration()
        macro = Macro(name, pattern, transformer)
        integration.meta_registry.register_macro(macro)
    
    @staticmethod
    def reflect_on(obj: Any) -> Dict[str, Any]:
        """Get reflection metadata about object."""
        integration = get_integration()
        return integration.meta_registry.reflect(obj)


# ============================================================================
# PHASE 7: OPTIMIZATION
# ============================================================================

class OptimizationHandler:
    """Handle bytecode compilation and optimization."""
    
    @staticmethod
    def profile_function_call(func_name: str, duration: float):
        """Record function call for profiling."""
        integration = get_integration()
        integration.optimizer.compiler.compiled_functions.setdefault(func_name, None)
        profile = integration.optimizer.create_profile(func_name)
        profile.record_call(duration)


# ============================================================================
# PHASE 9: ADVANCED TYPES
# ============================================================================

class AdvancedTypeHandler:
    """Handle advanced type features."""
    
    @staticmethod
    def check_trait(obj: Any, trait_name: str) -> bool:
        """Check if object implements a trait."""
        integration = get_integration()
        trait = integration.trait_registry.get_trait(trait_name)
        
        if trait is None:
            return False
        
        valid, _ = trait.validate_implementation(obj)
        return valid


# ============================================================================
# PHASE 10: ECOSYSTEM
# ============================================================================

class EcosystemHandler:
    """Handle ecosystem features."""
    
    @staticmethod
    def install_package(name: str, version: str = "*") -> bool:
        """Install a package."""
        integration = get_integration()
        pm = integration.ecosystem.get_package_manager()
        return pm.install(name, version)
    
    @staticmethod
    def is_package_installed(name: str) -> bool:
        """Check if package is installed."""
        integration = get_integration()
        pm = integration.ecosystem.get_package_manager()
        return pm.is_installed(name)
    
    @staticmethod
    def get_marketplace_plugins(category: str = None) -> List[str]:
        """Get plugins from marketplace."""
        integration = get_integration()
        marketplace = integration.ecosystem.get_marketplace()
        
        if category:
            plugins = marketplace.search_by_category(category)
        else:
            plugins = marketplace.get_trending()
        
        return [p.name for p in plugins]


# ============================================================================
# COMPREHENSIVE EVALUATION CONTEXT
# ============================================================================

class EvaluationContext:
    """Complete evaluation context with all phases integrated."""
    
    def __init__(self, context_name: str = "default"):
        """Initialize evaluation context."""
        self.name = context_name
        self.integration = get_integration()
        
        # Phase handlers
        self.modifiers = ModifierHandler()
        self.plugins = PluginHookHandler()
        self.capabilities = CapabilityChecker()
        self.vfs = VirtualFilesystemHandler()
        self.types = TypeSystemHandler()
        self.metaprogramming = MetaprogrammingHandler()
        self.optimization = OptimizationHandler()
        self.advanced_types = AdvancedTypeHandler()
        self.ecosystem = EcosystemHandler()
    
    def setup_for_untrusted_code(self):
        """Setup context for untrusted code execution."""
        # Use deny-all policy
        self.integration.capability_manager.set_policy(DenyAllPolicy())
    
    def setup_for_trusted_code(self):
        """Setup context for trusted code execution."""
        from ..capability_system import AllowAllPolicy
        self.integration.capability_manager.set_policy(AllowAllPolicy())
    
    def __repr__(self) -> str:
        return f"EvaluationContext({self.name})"
