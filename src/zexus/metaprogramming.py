"""
Metaprogramming system for Zexus.

Provides AST manipulation hooks and code generation capabilities.
Enables macros, compile-time transformations, and reflection.
"""

from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import inspect


class MetaOperationType(Enum):
    """Types of metaprogramming operations."""
    MACRO = "macro"
    TRANSFORM = "transform"
    REFLECTION = "reflection"
    CODE_GEN = "code_gen"


@dataclass
class ASTNode:
    """Simplified AST node representation."""
    type: str  # Node type (statement, expression, etc.)
    value: Any = None  # Node value
    children: List['ASTNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def clone(self) -> 'ASTNode':
        """Create a deep copy of the node."""
        new_children = [child.clone() for child in self.children]
        return ASTNode(
            type=self.type,
            value=self.value,
            children=new_children,
            metadata=self.metadata.copy()
        )
    
    def find_nodes(self, node_type: str) -> List['ASTNode']:
        """Find all nodes of a given type."""
        results = []
        if self.type == node_type:
            results.append(self)
        for child in self.children:
            results.extend(child.find_nodes(node_type))
        return results
    
    def walk(self, visitor: Callable[['ASTNode'], None]):
        """Walk tree and call visitor for each node."""
        visitor(self)
        for child in self.children:
            child.walk(visitor)
    
    def replace_node(self, old: 'ASTNode', new: 'ASTNode') -> bool:
        """Replace a node in the tree."""
        for i, child in enumerate(self.children):
            if child is old:
                self.children[i] = new
                return True
            if child.replace_node(old, new):
                return True
        return False
    
    def __str__(self) -> str:
        """String representation."""
        if self.value:
            return f"{self.type}({self.value})"
        return f"{self.type}[{len(self.children)}]"


class Macro:
    """A macro for compile-time code transformation."""
    
    def __init__(self, name: str, pattern: str, transformer: Callable[[ASTNode], ASTNode]):
        """
        Initialize macro.
        
        Args:
            name: Macro name
            pattern: Pattern to match (e.g., "action:foo")
            transformer: Function that transforms matching AST
        """
        self.name = name
        self.pattern = pattern
        self.transformer = transformer
    
    def matches(self, node: ASTNode) -> bool:
        """Check if AST node matches macro pattern."""
        # Simple pattern matching
        if ":" in self.pattern:
            node_type, node_name = self.pattern.split(":", 1)
            return node.type == node_type and node.value == node_name
        return node.type == self.pattern
    
    def apply(self, node: ASTNode) -> ASTNode:
        """Apply macro transformation to node."""
        if self.matches(node):
            return self.transformer(node)
        return node


class MetaRegistry:
    """
    Registry for metaprogramming hooks.
    
    Manages macros, transformers, and code generators.
    """
    
    def __init__(self):
        """Initialize metaprogramming registry."""
        self.macros: Dict[str, Macro] = {}
        self.transformers: Dict[str, Callable] = {}
        self.generators: Dict[str, Callable] = {}
        self.reflection_hooks: List[Callable] = []
    
    def register_macro(self, macro: Macro):
        """Register a macro."""
        self.macros[macro.name] = macro
    
    def register_transformer(self, name: str, transformer: Callable[[ASTNode], ASTNode]):
        """Register an AST transformer."""
        self.transformers[name] = transformer
    
    def register_generator(self, name: str, generator: Callable[..., str]):
        """Register a code generator."""
        self.generators[name] = generator
    
    def register_reflection_hook(self, hook: Callable[[Any], Dict]):
        """Register a reflection hook."""
        self.reflection_hooks.append(hook)
    
    def apply_macros(self, ast: ASTNode) -> ASTNode:
        """Apply all registered macros to AST."""
        result = ast.clone()
        
        def apply_to_node(node: ASTNode) -> None:
            # Apply each macro to the node
            for macro in self.macros.values():
                if macro.matches(node):
                    transformed = macro.apply(node)
                    # Update the node in place with transformed metadata
                    node.metadata.update(transformed.metadata)
        
        # Walk tree and apply macros
        result.walk(apply_to_node)
        return result
    
    def apply_transformers(self, ast: ASTNode, name: str) -> Optional[ASTNode]:
        """Apply a specific transformer."""
        if name in self.transformers:
            return self.transformers[name](ast)
        return None
    
    def generate_code(self, name: str, *args, **kwargs) -> Optional[str]:
        """Generate code using a registered generator."""
        if name in self.generators:
            return self.generators[name](*args, **kwargs)
        return None
    
    def reflect(self, obj: Any) -> Dict[str, Any]:
        """Get metadata about an object via reflection."""
        info = {
            "type": type(obj).__name__,
            "callable": callable(obj),
            "methods": [],
            "attributes": []
        }
        
        # Get methods and attributes
        if hasattr(obj, "__dict__"):
            for name, value in obj.__dict__.items():
                if callable(value):
                    info["methods"].append(name)
                else:
                    info["attributes"].append({
                        "name": name,
                        "type": type(value).__name__
                    })
        
        # Apply reflection hooks
        for hook in self.reflection_hooks:
            try:
                extra_info = hook(obj)
                info.update(extra_info)
            except Exception:
                pass
        
        return info


class MacroBuilder:
    """Builder for creating macros."""
    
    def __init__(self, name: str, pattern: str):
        """Initialize macro builder."""
        self.name = name
        self.pattern = pattern
        self.transformations: List[Callable[[ASTNode], ASTNode]] = []
    
    def add_transformation(self, func: Callable[[ASTNode], ASTNode]) -> 'MacroBuilder':
        """Add a transformation step."""
        self.transformations.append(func)
        return self
    
    def replace_value(self, old_value: Any, new_value: Any) -> 'MacroBuilder':
        """Add a value replacement transformation."""
        def transform(node: ASTNode) -> ASTNode:
            if node.value == old_value:
                node.value = new_value
            return node
        return self.add_transformation(transform)
    
    def add_metadata(self, key: str, value: Any) -> 'MacroBuilder':
        """Add metadata to matched nodes."""
        def transform(node: ASTNode) -> ASTNode:
            node.metadata[key] = value
            return node
        return self.add_transformation(transform)
    
    def build(self) -> Macro:
        """Build the macro."""
        def combined_transformer(node: ASTNode) -> ASTNode:
            result = node.clone()
            for transform in self.transformations:
                result = transform(result)
            return result
        
        return Macro(self.name, self.pattern, combined_transformer)


# Common macro patterns

class CommonMacros:
    """Pre-defined common macros."""
    
    @staticmethod
    def once(pattern: str) -> Macro:
        """Macro that marks code to run only once."""
        return MacroBuilder(f"{pattern}_once", pattern) \
            .add_metadata("run_once", True) \
            .build()
    
    @staticmethod
    def inline(pattern: str) -> Macro:
        """Macro that marks code for inlining."""
        return MacroBuilder(f"{pattern}_inline", pattern) \
            .add_metadata("inline", True) \
            .build()
    
    @staticmethod
    def deprecated(pattern: str, message: str = "") -> Macro:
        """Macro that marks code as deprecated."""
        return MacroBuilder(f"{pattern}_deprecated", pattern) \
            .add_metadata("deprecated", True) \
            .add_metadata("deprecation_message", message) \
            .build()
    
    @staticmethod
    def optimize(pattern: str) -> Macro:
        """Macro that marks code for optimization."""
        return MacroBuilder(f"{pattern}_optimize", pattern) \
            .add_metadata("optimize", True) \
            .build()


class ReflectionAPI:
    """API for runtime reflection."""
    
    @staticmethod
    def get_signature(func: Callable) -> Dict[str, Any]:
        """Get function signature."""
        try:
            sig = inspect.signature(func)
            return {
                "params": list(sig.parameters.keys()),
                "return_annotation": str(sig.return_annotation) if sig.return_annotation else None
            }
        except Exception:
            return {"params": [], "return_annotation": None}
    
    @staticmethod
    def get_source(func: Callable) -> Optional[str]:
        """Get function source code."""
        try:
            return inspect.getsource(func)
        except Exception:
            return None
    
    @staticmethod
    def get_docstring(obj: Any) -> Optional[str]:
        """Get object docstring."""
        return inspect.getdoc(obj)
    
    @staticmethod
    def get_members(obj: Any) -> List[str]:
        """Get all members of an object."""
        try:
            return [name for name, _ in inspect.getmembers(obj)]
        except Exception:
            return []


# Global metaprogramming registry

_global_meta_registry = MetaRegistry()


def get_meta_registry() -> MetaRegistry:
    """Get the global metaprogramming registry."""
    return _global_meta_registry


def register_macro(macro: Macro):
    """Register a macro globally."""
    _global_meta_registry.register_macro(macro)


def apply_all_macros(ast: ASTNode) -> ASTNode:
    """Apply all registered macros."""
    return _global_meta_registry.apply_macros(ast)
