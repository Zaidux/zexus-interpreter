"""
Complexity & Large Project Management System for Zexus.

Provides features for organizing large, complex projects:
1. Formal Interfaces/Type Classes - Define contracts without implementation
2. Type Aliases - Simplify complex type signatures
3. Module/Package Scoping - Organize code into logical units with visibility control
4. Resource Management - Ensure proper cleanup of resources (RAII pattern)

These features enable:
- Better code organization and readability
- Type safety and correctness
- Prevention of naming conflicts
- Deterministic resource cleanup
- Clear separation of concerns
"""

from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class Visibility(Enum):
    """Visibility/access levels for module members."""
    PUBLIC = "public"           # Accessible from outside module
    INTERNAL = "internal"       # Only accessible within module
    PROTECTED = "protected"     # Accessible by submodules


@dataclass
class InterfaceMethod:
    """Definition of a method required by an interface."""
    name: str
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    description: str = ""
    
    def __repr__(self):
        params = ", ".join(self.parameters)
        return f"InterfaceMethod({self.name}({params}))"


@dataclass
class Interface:
    """
    Formal interface/type class defining a contract.
    
    An interface specifies what methods/properties an implementing type must have,
    without providing implementation details.
    
    Example:
        interface Drawable {
            draw(canvas);
            bounds();
        }
        
        interface Serializable {
            to_string();
            from_string(str);
        }
    """
    name: str
    methods: List[InterfaceMethod] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)  # name -> type
    extends: List[str] = field(default_factory=list)  # Other interfaces this extends
    documentation: str = ""
    generic_parameters: List[str] = field(default_factory=list)  # For generic interfaces
    
    def add_method(self, method: InterfaceMethod):
        """Add a method to the interface."""
        self.methods.append(method)
    
    def add_property(self, name: str, type_name: str):
        """Add a property to the interface."""
        self.properties[name] = type_name
    
    def get_all_methods(self) -> List[InterfaceMethod]:
        """Get all methods including inherited ones."""
        return self.methods.copy()
    
    def requires_implementation(self) -> Dict[str, Any]:
        """Return what needs to be implemented."""
        return {
            "methods": [m.name for m in self.methods],
            "properties": list(self.properties.keys())
        }
    
    def __repr__(self):
        return f"Interface({self.name}, methods={len(self.methods)}, properties={len(self.properties)})"


@dataclass
class TypeAlias:
    """
    Type alias providing a simpler name for complex types.
    
    Example:
        type_alias UserID = integer;
        type_alias Point = { x: float, y: float };
        type_alias Handler = function(request) -> response;
    """
    name: str
    base_type: str  # The actual type this is an alias for
    documentation: str = ""
    generic_parameters: List[str] = field(default_factory=list)
    
    def __repr__(self):
        return f"TypeAlias({self.name} = {self.base_type})"


@dataclass
class ModuleMember:
    """A member (function, variable, class) within a module."""
    name: str
    member_type: str  # "function", "class", "variable", "interface", etc.
    visibility: Visibility = Visibility.PUBLIC
    value: Any = None
    documentation: str = ""
    
    def __repr__(self):
        return f"ModuleMember({self.name}: {self.member_type}, visibility={self.visibility.value})"


class Module:
    """
    Logical code organization unit with visibility control.
    
    Modules provide:
    - Namespace separation (prevent naming conflicts)
    - Visibility control (public/internal/protected)
    - Clear organization of related functionality
    - Import/export boundaries
    
    Example:
        module database {
            // internal functions
            internal function connect_db(path) { ... }
            
            // public API
            public function query(sql) { ... }
            public function execute(sql, params) { ... }
        }
    """
    
    def __init__(self, name: str, parent: Optional['Module'] = None):
        self.name = name
        self.parent = parent
        self.members: Dict[str, ModuleMember] = {}
        self.submodules: Dict[str, 'Module'] = {}
        self.imports: List[str] = []
        self.exports: List[str] = []
        self.documentation = ""
    
    def get_full_name(self) -> str:
        """Get fully qualified module name."""
        if self.parent:
            return f"{self.parent.get_full_name()}.{self.name}"
        return self.name
    
    def add_member(self, member: ModuleMember):
        """Add a member to the module."""
        self.members[member.name] = member
    
    def get_member(self, name: str) -> Optional[ModuleMember]:
        """Get a member by name."""
        return self.members.get(name)
    
    def get(self, name: str):
        """Get a member's value by name (for compatibility with property access).
        Returns the value directly, not the ModuleMember wrapper."""
        member = self.get_member(name)
        if member:
            return member.value
        return None
    
    def type(self):
        """Return the type name for this object (for compatibility with evaluator)."""
        return f"Module[{self.name}]"
    
    def get_public_members(self) -> List[ModuleMember]:
        """Get all public members."""
        return [m for m in self.members.values() if m.visibility == Visibility.PUBLIC]
    
    def add_submodule(self, module: 'Module'):
        """Add a submodule."""
        module.parent = self
        self.submodules[module.name] = module
    
    def get_submodule(self, name: str) -> Optional['Module']:
        """Get a submodule by name."""
        return self.submodules.get(name)
    
    def add_import(self, module_path: str):
        """Add an import."""
        if module_path not in self.imports:
            self.imports.append(module_path)
    
    def add_export(self, name: str):
        """Mark a member as exported."""
        if name not in self.exports:
            self.exports.append(name)
    
    def can_access(self, member: ModuleMember, from_module: 'Module') -> bool:
        """Check if a module can access a member."""
        if member.visibility == Visibility.PUBLIC:
            return True
        
        if member.visibility == Visibility.INTERNAL:
            # Only accessible within same module
            return from_module.name == self.name
        
        if member.visibility == Visibility.PROTECTED:
            # Accessible within module and submodules
            return from_module.get_full_name().startswith(self.get_full_name())
        
        return False
    
    def __repr__(self):
        return f"Module({self.get_full_name()}, members={len(self.members)}, submodules={len(self.submodules)})"


class Package:
    """
    Top-level organization unit containing multiple modules.
    
    Packages represent a cohesive unit of functionality.
    
    Example:
        package myapp.database {
            module connection { ... }
            module query { ... }
            module migration { ... }
        }
    """
    
    def __init__(self, name: str):
        self.name = name
        self.modules: Dict[str, Module] = {}
        self.version = "0.0.1"
        self.documentation = ""
        self.dependencies: List[str] = []
    
    def add_module(self, module: Module):
        """Add a module to the package."""
        self.modules[module.name] = module
    
    def get_module(self, name: str) -> Optional[Module]:
        """Get a module by name."""
        return self.modules.get(name)
    
    def get(self, name: str):
        """Get a module or sub-package by name (for property access)."""
        return self.modules.get(name)
    
    def add_dependency(self, package_name: str):
        """Add a package dependency."""
        if package_name not in self.dependencies:
            self.dependencies.append(package_name)
    
    def __repr__(self):
        return f"Package({self.name}, modules={len(self.modules)}, version={self.version})"


class ResourceManager:
    """
    Manages resource lifecycle following RAII (Resource Acquisition Is Initialization) pattern.
    
    Ensures deterministic resource cleanup, even if exceptions occur.
    
    Example:
        using(file = open("/path/to/file.txt")) {
            content = file.read();
            // file is automatically closed here
        }
        
        using(connection = db.connect()) {
            result = connection.query("SELECT * FROM users");
            // connection is automatically closed here
        }
    """
    
    def __init__(self):
        self.resource_stack: List[Dict[str, Any]] = []
    
    def acquire_resource(self, resource_name: str, resource: Any, 
                        cleanup_fn: Callable[[Any], None]):
        """
        Acquire a resource with automatic cleanup.
        
        Args:
            resource_name: Name/identifier for the resource
            resource: The resource object
            cleanup_fn: Function to call when resource should be released
        """
        self.resource_stack.append({
            "name": resource_name,
            "resource": resource,
            "cleanup": cleanup_fn,
            "acquired": True
        })
        
        return resource
    
    def release_resource(self, resource_name: str):
        """Explicitly release a resource."""
        for i, entry in enumerate(self.resource_stack):
            if entry["name"] == resource_name:
                try:
                    entry["cleanup"](entry["resource"])
                    entry["acquired"] = False
                except Exception as e:
                    raise ResourceCleanupError(f"Error cleaning up {resource_name}: {e}")
                break
    
    def cleanup_all(self):
        """Clean up all acquired resources (called on exit)."""
        # Release in reverse order (LIFO)
        for entry in reversed(self.resource_stack):
            if entry["acquired"]:
                try:
                    entry["cleanup"](entry["resource"])
                    entry["acquired"] = False
                except Exception as e:
                    # Log but don't raise - ensure all resources are cleaned
                    print(f"Warning: Error cleaning up {entry['name']}: {e}")
    
    def get_resource(self, resource_name: str) -> Optional[Any]:
        """Get a currently acquired resource."""
        for entry in self.resource_stack:
            if entry["name"] == resource_name and entry["acquired"]:
                return entry["resource"]
        return None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up all resources."""
        self.cleanup_all()
        return False  # Don't suppress exceptions


class InterfaceValidator:
    """
    Validates that types properly implement interfaces.
    
    Checks:
    - All required methods are present
    - Method signatures match
    - All required properties exist
    """
    
    def __init__(self):
        self.interfaces: Dict[str, Interface] = {}
        self.implementations: Dict[str, Set[str]] = {}  # type_name -> interfaces
    
    def register_interface(self, interface: Interface):
        """Register an interface."""
        self.interfaces[interface.name] = interface
    
    def register_implementation(self, type_name: str, interface_name: str):
        """Register that a type implements an interface."""
        if type_name not in self.implementations:
            self.implementations[type_name] = set()
        self.implementations[type_name].add(interface_name)
    
    def validate_implementation(self, type_name: str, interface_name: str,
                              implementation: Dict[str, Any]) -> List[str]:
        """
        Validate that a type properly implements an interface.
        
        Args:
            type_name: Name of the type
            interface_name: Name of the interface
            implementation: The actual implementation (dict of methods/properties)
        
        Returns:
            List of errors (empty if valid)
        """
        interface = self.interfaces.get(interface_name)
        if not interface:
            return [f"Interface '{interface_name}' not found"]
        
        errors = []
        
        # Check methods
        for method in interface.methods:
            if method.name not in implementation:
                errors.append(f"Missing method: {method.name}")
            else:
                impl_method = implementation[method.name]
                # Basic signature checking
                if not callable(impl_method):
                    errors.append(f"'{method.name}' is not callable")
        
        # Check properties
        for prop_name, prop_type in interface.properties.items():
            if prop_name not in implementation:
                errors.append(f"Missing property: {prop_name}")
        
        return errors
    
    def get_interface_requirements(self, interface_name: str) -> Optional[Dict[str, Any]]:
        """Get what's required by an interface."""
        interface = self.interfaces.get(interface_name)
        if not interface:
            return None
        
        return interface.requires_implementation()


class TypeAliasRegistry:
    """Registry for type aliases."""
    
    def __init__(self):
        self.aliases: Dict[str, TypeAlias] = {}
    
    def register_alias(self, alias: TypeAlias):
        """Register a type alias.
        
        Allows re-registration of the same type alias name.
        This enables type aliases to be redefined in different scopes
        or updated during development/testing.
        """
        # Allow re-registration - just update the alias
        self.aliases[alias.name] = alias
    
    def get_alias(self, name: str) -> Optional[TypeAlias]:
        """Get a type alias by name."""
        return self.aliases.get(name)
    
    def resolve_type(self, type_name: str) -> str:
        """Resolve a type name to its actual type (following aliases)."""
        alias = self.aliases.get(type_name)
        if alias:
            # Recursively resolve in case of nested aliases
            return self.resolve_type(alias.base_type)
        return type_name
    
    def list_aliases(self) -> List[TypeAlias]:
        """List all registered aliases."""
        return list(self.aliases.values())


class ComplexityManager:
    """
    Central manager for project complexity features.
    
    Coordinates:
    - Interfaces and type classes
    - Type aliases
    - Modules and packages
    - Resource management
    - Visibility and access control
    """
    
    def __init__(self):
        self.root_module = Module("root")
        self.packages: Dict[str, Package] = {}
        self.interface_validator = InterfaceValidator()
        self.type_alias_registry = TypeAliasRegistry()
        self.resource_manager = ResourceManager()
    
    def create_package(self, name: str) -> Package:
        """Create a new package."""
        pkg = Package(name)
        self.packages[name] = pkg
        return pkg
    
    def get_package(self, name: str) -> Optional[Package]:
        """Get a package by name."""
        return self.packages.get(name)
    
    def create_module(self, name: str, parent: Optional[Module] = None) -> Module:
        """Create a new module."""
        module = Module(name, parent)
        return module
    
    def register_interface(self, interface: Interface):
        """Register an interface."""
        self.interface_validator.register_interface(interface)
    
    def register_type_alias(self, alias: TypeAlias):
        """Register a type alias."""
        self.type_alias_registry.register_alias(alias)
    
    def validate_interface_implementation(self, type_name: str, 
                                         interface_name: str,
                                         implementation: Dict[str, Any]) -> bool:
        """Validate interface implementation."""
        errors = self.interface_validator.validate_implementation(
            type_name, interface_name, implementation
        )
        return len(errors) == 0
    
    def get_implementation_errors(self, type_name: str,
                                 interface_name: str,
                                 implementation: Dict[str, Any]) -> List[str]:
        """Get list of implementation errors."""
        return self.interface_validator.validate_implementation(
            type_name, interface_name, implementation
        )


# Global instance
_complexity_manager = ComplexityManager()


def get_complexity_manager() -> ComplexityManager:
    """Get the global complexity manager instance."""
    return _complexity_manager


class ResourceCleanupError(Exception):
    """Exception raised when resource cleanup fails."""
    pass


class InterfaceImplementationError(Exception):
    """Exception raised when interface implementation is invalid."""
    pass


class ModuleAccessError(Exception):
    """Exception raised when accessing private/internal module members."""
    pass


# Example usage patterns
"""
# 1. Define an interface
interface Drawable {
    draw(canvas);
    get_bounds();
}

# 2. Implement the interface
class Circle {
    radius: float;
    
    public function draw(canvas) {
        canvas.draw_circle(this.radius);
    }
    
    public function get_bounds() {
        return { x: 0, y: 0, width: this.radius * 2, height: this.radius * 2 };
    }
}

# 3. Use type alias for clarity
type_alias UserID = integer;
type_alias Point = { x: float, y: float };

# 4. Organize with modules
module graphics {
    internal function initialize_graphics() { ... }
    
    public function render(drawable) {
        drawable.draw(current_canvas);
    }
}

# 5. Manage resources with using
using(file = open("data.txt")) {
    content = file.read();
    process(content);
    // file is automatically closed
}

using(conn = database.connect()) {
    results = conn.query("SELECT * FROM users");
    // connection is automatically closed
}
"""
