"""
Advanced type system with generics, traits, and union types.
"""

from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class TypeVariance(Enum):
    """Type parameter variance."""
    COVARIANT = "covariant"      # Producer type
    CONTRAVARIANT = "contravariant"  # Consumer type
    INVARIANT = "invariant"      # Bidirectional type


@dataclass
class TypeParameter:
    """Generic type parameter."""
    name: str
    bounds: Optional[List['TypeSpec']] = None  # Upper bounds
    variance: TypeVariance = TypeVariance.INVARIANT
    default: Optional['TypeSpec'] = None  # Default type
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TypeParameter):
            return False
        return self.name == other.name
    
    def satisfies_bounds(self, type_spec: 'TypeSpec') -> bool:
        """Check if type satisfies bounds."""
        if not self.bounds:
            return True
        # Simplified - would need full subtype checking
        return True


@dataclass
class GenericType:
    """Generic type with type parameters."""
    base_type: 'TypeSpec'
    type_params: List[TypeParameter] = field(default_factory=list)
    type_args: Dict[str, 'TypeSpec'] = field(default_factory=dict)
    
    def is_fully_specified(self) -> bool:
        """Check if all type parameters are bound."""
        return len(self.type_args) == len(self.type_params)
    
    def instantiate(self, type_args: Dict[str, 'TypeSpec']) -> 'TypeSpec':
        """Create concrete type from generic."""
        # Return instantiated type
        return self.base_type
    
    def __repr__(self) -> str:
        """String representation."""
        params = ", ".join(p.name for p in self.type_params)
        return f"Generic<{params}>"


@dataclass
class UnionType:
    """Union type (multiple possible types)."""
    types: Set['TypeSpec'] = field(default_factory=set)
    discriminator: Optional[str] = None  # Discriminator field for tagged unions
    
    def add_type(self, type_spec: 'TypeSpec'):
        """Add type to union."""
        self.types.add(type_spec)
    
    def is_member(self, type_spec: 'TypeSpec') -> bool:
        """Check if type is in union."""
        return type_spec in self.types
    
    def __repr__(self) -> str:
        """String representation."""
        type_names = " | ".join(str(t) for t in self.types)
        return f"({type_names})"


class Trait(ABC):
    """Base class for type traits."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Get trait name."""
        pass
    
    @abstractmethod
    def get_required_methods(self) -> Set[str]:
        """Get methods required to implement trait."""
        pass
    
    @abstractmethod
    def get_provided_methods(self) -> Dict[str, callable]:
        """Get default method implementations."""
        pass
    
    @abstractmethod
    def validate_implementation(self, obj: Any) -> Tuple[bool, List[str]]:
        """Validate that object implements trait."""
        pass


@dataclass
class StructuralTrait(Trait):
    """Trait based on object structure."""
    name: str
    required_methods: Set[str] = field(default_factory=set)
    required_attributes: Set[str] = field(default_factory=set)
    method_signatures: Dict[str, List['TypeSpec']] = field(default_factory=dict)
    default_impls: Dict[str, callable] = field(default_factory=dict)
    
    def get_name(self) -> str:
        """Get trait name."""
        return self.name
    
    def get_required_methods(self) -> Set[str]:
        """Get required methods."""
        return self.required_methods.copy()
    
    def get_provided_methods(self) -> Dict[str, callable]:
        """Get default implementations."""
        return self.default_impls.copy()
    
    def validate_implementation(self, obj: Any) -> Tuple[bool, List[str]]:
        """Check if object implements trait."""
        missing = []
        
        # Check methods
        for method in self.required_methods:
            if not hasattr(obj, method) or not callable(getattr(obj, method)):
                missing.append(f"Missing method: {method}")
        
        # Check attributes
        for attr in self.required_attributes:
            if not hasattr(obj, attr):
                missing.append(f"Missing attribute: {attr}")
        
        return len(missing) == 0, missing


@dataclass
class TraitImpl:
    """Trait implementation for a type."""
    type_name: str
    trait: Trait
    methods: Dict[str, callable] = field(default_factory=dict)
    
    def get_method(self, name: str) -> Optional[callable]:
        """Get method implementation."""
        if name in self.methods:
            return self.methods[name]
        return None
    
    def is_valid(self) -> bool:
        """Check if implementation is valid."""
        valid, _ = self.trait.validate_implementation(self)
        return valid


class SimpleIterableTrait(StructuralTrait):
    """Trait for iterable types."""
    
    def __init__(self):
        """Initialize iterable trait."""
        super().__init__(
            name="Iterable",
            required_methods={"iter", "next"},
            method_signatures={
                "iter": [],
                "next": []
            }
        )


class SimpleComparableTrait(StructuralTrait):
    """Trait for comparable types."""
    
    def __init__(self):
        """Initialize comparable trait."""
        super().__init__(
            name="Comparable",
            required_methods={"compare", "equals"},
            method_signatures={
                "compare": [],
                "equals": []
            }
        )


class SimpleCloneableTrait(StructuralTrait):
    """Trait for cloneable types."""
    
    def __init__(self):
        """Initialize cloneable trait."""
        super().__init__(
            name="Cloneable",
            required_methods={"clone"},
            method_signatures={
                "clone": []
            }
        )


class AdvancedTypeSpec:
    """Advanced type specification with generics and traits."""
    
    def __init__(self, base_type: str):
        """Initialize advanced type spec."""
        self.base_type = base_type
        self.generic: Optional[GenericType] = None
        self.union: Optional[UnionType] = None
        self.traits: List[Trait] = []
        self.nullable = False
        self.array_of: Optional['AdvancedTypeSpec'] = None
    
    def with_generic(self, type_params: List[TypeParameter],
                    type_args: Optional[Dict[str, 'AdvancedTypeSpec']] = None) -> 'AdvancedTypeSpec':
        """Add generic parameters."""
        self.generic = GenericType(
            base_type=self,
            type_params=type_params,
            type_args=type_args or {}
        )
        return self
    
    def with_union(self, types: List['AdvancedTypeSpec']) -> 'AdvancedTypeSpec':
        """Create union type."""
        self.union = UnionType()
        for t in types:
            self.union.add_type(t)
        return self
    
    def with_trait(self, trait: Trait) -> 'AdvancedTypeSpec':
        """Add trait requirement."""
        self.traits.append(trait)
        return self
    
    def make_nullable(self) -> 'AdvancedTypeSpec':
        """Make type nullable."""
        self.nullable = True
        return self
    
    def make_array(self) -> 'AdvancedTypeSpec':
        """Create array type."""
        arr = AdvancedTypeSpec(f"{self.base_type}[]")
        arr.array_of = self
        return arr
    
    def is_assignable_to(self, other: 'AdvancedTypeSpec') -> bool:
        """Check if assignable to another type."""
        # Base type check
        if self.base_type == other.base_type:
            return True
        
        # Nullable check
        if other.nullable and self.nullable:
            return True
        
        # Union check
        if other.union and self in other.union.types:
            return True
        
        return False
    
    def satisfies_traits(self, obj: Any) -> bool:
        """Check if object satisfies all traits."""
        for trait in self.traits:
            valid, _ = trait.validate_implementation(obj)
            if not valid:
                return False
        return True
    
    def __repr__(self) -> str:
        """String representation."""
        parts = [self.base_type]
        
        if self.generic:
            parts.append(str(self.generic))
        
        if self.union:
            parts.append(str(self.union))
        
        if self.traits:
            trait_names = ", ".join(t.get_name() for t in self.traits)
            parts.append(f"impl({trait_names})")
        
        if self.nullable:
            parts.append("?")
        
        return "".join(parts)


class TraitRegistry:
    """Registry for managing traits."""
    
    def __init__(self):
        """Initialize trait registry."""
        self.traits: Dict[str, Trait] = {}
        self.implementations: Dict[Tuple[str, str], TraitImpl] = {}  # (type, trait) -> impl
        self._register_default_traits()
    
    def _register_default_traits(self):
        """Register built-in traits."""
        self.register_trait("Iterable", SimpleIterableTrait())
        self.register_trait("Comparable", SimpleComparableTrait())
        self.register_trait("Cloneable", SimpleCloneableTrait())
    
    def register_trait(self, name: str, trait: Trait):
        """Register a trait."""
        self.traits[name] = trait
    
    def get_trait(self, name: str) -> Optional[Trait]:
        """Get trait by name."""
        return self.traits.get(name)
    
    def register_impl(self, type_name: str, trait_name: str, impl: TraitImpl):
        """Register trait implementation for a type."""
        self.implementations[(type_name, trait_name)] = impl
    
    def get_impl(self, type_name: str, trait_name: str) -> Optional[TraitImpl]:
        """Get trait implementation."""
        return self.implementations.get((type_name, trait_name))
    
    def get_type_traits(self, type_name: str) -> List[Trait]:
        """Get all traits implemented by a type."""
        trait_names = set()
        for (t_type, t_name), _ in self.implementations.items():
            if t_type == type_name:
                trait_names.add(t_name)
        
        return [self.traits[name] for name in trait_names if name in self.traits]


class GenericResolver:
    """Resolve generic type applications."""
    
    def __init__(self):
        """Initialize generic resolver."""
        self.instantiations: Dict[str, AdvancedTypeSpec] = {}
    
    def resolve(self, generic_type: 'AdvancedTypeSpec',
               type_args: Dict[str, 'AdvancedTypeSpec']) -> 'AdvancedTypeSpec':
        """Resolve generic type with type arguments."""
        if not generic_type.generic:
            return generic_type
        
        # Create instantiation key
        type_arg_strs = [str(t) for t in type_args.values()]
        key = f"{generic_type.base_type}[{','.join(type_arg_strs)}]"
        
        if key in self.instantiations:
            return self.instantiations[key]
        
        # Create concrete type
        concrete = AdvancedTypeSpec(key)
        self.instantiations[key] = concrete
        
        return concrete
    
    def check_type_bounds(self, type_param: TypeParameter,
                         type_arg: AdvancedTypeSpec) -> bool:
        """Check if type argument satisfies bounds."""
        if not type_param.bounds:
            return True
        
        for bound in type_param.bounds:
            # Simplified check
            if hasattr(bound, 'base_type') and hasattr(type_arg, 'base_type'):
                if bound.base_type == type_arg.base_type:
                    return True
        
        return len(type_param.bounds) == 0


# Global trait registry
_global_trait_registry = TraitRegistry()


def get_trait_registry() -> TraitRegistry:
    """Get global trait registry."""
    return _global_trait_registry


def create_generic_type(name: str, type_params: List[str]) -> AdvancedTypeSpec:
    """Create a generic type."""
    spec = AdvancedTypeSpec(name)
    params = [TypeParameter(p) for p in type_params]
    spec.with_generic(params)
    return spec


def create_union_type(types: List[AdvancedTypeSpec]) -> AdvancedTypeSpec:
    """Create a union type."""
    spec = AdvancedTypeSpec("union")
    spec.with_union(types)
    return spec
