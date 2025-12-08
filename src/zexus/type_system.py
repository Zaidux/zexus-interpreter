"""
Unified type system for Zexus.

Provides runtime type checking, type annotations, and custom type definitions.
Supports basic types, compound types, and user-defined types.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


class BaseType(Enum):
    """Base type constants."""
    NONE = "none"
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    ARRAY = "array"
    OBJECT = "object"
    ACTION = "action"
    ANY = "any"


@dataclass
class TypeSpec:
    """Type specification for values."""
    base_type: Union[BaseType, str]
    nullable: bool = False
    array_of: Optional['TypeSpec'] = None
    object_schema: Optional[Dict[str, 'TypeSpec']] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    
    def __str__(self) -> str:
        """String representation of type."""
        type_str = self.base_type.value if isinstance(self.base_type, BaseType) else self.base_type
        
        if self.array_of:
            type_str = f"[{self.array_of}]"
        
        if self.object_schema:
            fields = ", ".join(f"{k}:{v}" for k, v in self.object_schema.items())
            type_str = f"{{{fields}}}"
        
        if self.nullable:
            type_str = f"{type_str}?"
        
        return type_str
    
    def is_compatible_with(self, other: 'TypeSpec') -> bool:
        """Check if this type is compatible with another."""
        if other.base_type == BaseType.ANY:
            return True
        
        if self.base_type == BaseType.ANY:
            return True
        
        if self.base_type != other.base_type:
            return False
        
        if self.nullable and not other.nullable:
            return False
        
        return True


class TypeChecker:
    """
    Checks values against type specifications.
    
    Supports both built-in types and custom validators.
    """
    
    def __init__(self):
        """Initialize type checker."""
        self.custom_types: Dict[str, TypeSpec] = {}
        self.validators: Dict[str, Callable[[Any], bool]] = {}
    
    def register_type(self, name: str, type_spec: TypeSpec):
        """Register a custom type."""
        self.custom_types[name] = type_spec
    
    def register_validator(self, type_name: str, validator: Callable[[Any], bool]):
        """Register a custom validator function."""
        self.validators[type_name] = validator
    
    def get_type(self, name: str) -> Optional[TypeSpec]:
        """Get a registered type by name."""
        return self.custom_types.get(name)
    
    def check_type(self, value: Any, type_spec: TypeSpec) -> Tuple[bool, str]:
        """
        Check if value matches type specification.
        
        Returns:
            (matches: bool, reason: str)
        """
        # Handle None/null values
        if value is None:
            if type_spec.nullable:
                return True, "Value is null (allowed)"
            else:
                return False, f"Value is null but {type_spec} is not nullable"
        
        # Handle custom validators
        if type_spec.custom_validator:
            try:
                if type_spec.custom_validator(value):
                    return True, "Value passes custom validator"
                else:
                    return False, f"Value fails custom validator for {type_spec}"
            except Exception as e:
                return False, f"Error in custom validator: {e}"
        
        # Handle array types
        if type_spec.array_of:
            if not isinstance(value, list):
                return False, f"Expected array, got {type(value).__name__}"
            
            for i, item in enumerate(value):
                matches, reason = self.check_type(item, type_spec.array_of)
                if not matches:
                    return False, f"Item {i} in array: {reason}"
            
            return True, "All array items match type"
        
        # Handle object/dict types
        if type_spec.object_schema:
            if not isinstance(value, dict):
                return False, f"Expected object, got {type(value).__name__}"
            
            # Check required fields
            for field_name, field_type in type_spec.object_schema.items():
                if field_name not in value:
                    return False, f"Missing required field '{field_name}' of type {field_type}"
                
                matches, reason = self.check_type(value[field_name], field_type)
                if not matches:
                    return False, f"Field '{field_name}': {reason}"
            
            return True, "Object matches schema"
        
        # Check base types
        type_name = type_spec.base_type.value if isinstance(type_spec.base_type, BaseType) else type_spec.base_type
        return self._check_base_type(value, type_name)
    
    def _check_base_type(self, value: Any, type_name: str) -> Tuple[bool, str]:
        """Check value against base type."""
        if type_name == "any":
            return True, "Type is 'any'"
        
        if type_name == "bool":
            if isinstance(value, bool):
                return True, f"Value is bool"
            return False, f"Expected bool, got {type(value).__name__}"
        
        if type_name == "int":
            if isinstance(value, int) and not isinstance(value, bool):
                return True, f"Value is int"
            return False, f"Expected int, got {type(value).__name__}"
        
        if type_name == "float":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True, f"Value is numeric"
            return False, f"Expected float, got {type(value).__name__}"
        
        if type_name == "string":
            if isinstance(value, str):
                return True, f"Value is string"
            return False, f"Expected string, got {type(value).__name__}"
        
        if type_name == "array":
            if isinstance(value, list):
                return True, f"Value is array"
            return False, f"Expected array, got {type(value).__name__}"
        
        if type_name == "object":
            if isinstance(value, dict):
                return True, f"Value is object"
            return False, f"Expected object, got {type(value).__name__}"
        
        if type_name == "action":
            # Actions are callable-like objects
            return callable(value), f"Value is {'action' if callable(value) else 'not action'}"
        
        # Unknown type
        return False, f"Unknown type: {type_name}"
    
    def require_type(self, value: Any, type_spec: TypeSpec, context: str = "") -> bool:
        """
        Require that value matches type. Raises TypeError if not.
        
        Args:
            value: Value to check
            type_spec: Expected type
            context: Context description for error message
            
        Returns:
            True if value matches type
            
        Raises:
            TypeError: If type check fails
        """
        matches, reason = self.check_type(value, type_spec)
        if not matches:
            error_msg = f"Type mismatch{f' in {context}' if context else ''}: {reason}"
            raise TypeError(error_msg)
        return True


class TypeInferencer:
    """
    Infers types of values at runtime.
    
    Used when type annotations are not provided.
    """
    
    @staticmethod
    def infer_type(value: Any) -> TypeSpec:
        """Infer the type of a value."""
        if value is None:
            return TypeSpec(BaseType.NONE)
        
        if isinstance(value, bool):
            return TypeSpec(BaseType.BOOL)
        
        if isinstance(value, int):
            return TypeSpec(BaseType.INT)
        
        if isinstance(value, float):
            return TypeSpec(BaseType.FLOAT)
        
        if isinstance(value, str):
            return TypeSpec(BaseType.STRING)
        
        if isinstance(value, list):
            if len(value) == 0:
                return TypeSpec(BaseType.ARRAY, array_of=TypeSpec(BaseType.ANY))
            
            # Infer element type from first element (simplified)
            element_type = TypeInferencer.infer_type(value[0])
            return TypeSpec(BaseType.ARRAY, array_of=element_type)
        
        if isinstance(value, dict):
            schema = {}
            for key, val in value.items():
                schema[key] = TypeInferencer.infer_type(val)
            return TypeSpec(BaseType.OBJECT, object_schema=schema)
        
        if callable(value):
            return TypeSpec(BaseType.ACTION)
        
        return TypeSpec(BaseType.ANY)


class FunctionSignature:
    """Function signature with parameter and return types."""
    
    def __init__(self, name: str, parameters: Dict[str, TypeSpec], 
                 return_type: Optional[TypeSpec] = None):
        """
        Initialize function signature.
        
        Args:
            name: Function name
            parameters: Dict of param_name -> TypeSpec
            return_type: Expected return type (None = any)
        """
        self.name = name
        self.parameters = parameters
        self.return_type = return_type or TypeSpec(BaseType.ANY)
    
    def __str__(self) -> str:
        """String representation."""
        params = ", ".join(f"{k}:{v}" for k, v in self.parameters.items())
        return f"action {self.name}({params}) -> {self.return_type}"
    
    def validate_call(self, args: Dict[str, Any], checker: TypeChecker) -> Tuple[bool, List[str]]:
        """
        Validate arguments for a function call.
        
        Returns:
            (valid: bool, errors: [str])
        """
        errors = []
        
        # Check all parameters are provided
        for param_name, param_type in self.parameters.items():
            if param_name not in args:
                errors.append(f"Missing parameter '{param_name}' of type {param_type}")
            else:
                matches, reason = checker.check_type(args[param_name], param_type)
                if not matches:
                    errors.append(f"Parameter '{param_name}': {reason}")
        
        # Check no extra parameters
        for arg_name in args:
            if arg_name not in self.parameters:
                errors.append(f"Unexpected parameter '{arg_name}'")
        
        return len(errors) == 0, errors
    
    def validate_return(self, value: Any, checker: TypeChecker) -> Tuple[bool, str]:
        """Validate return value."""
        return checker.check_type(value, self.return_type)


# Standard type definitions

STANDARD_TYPES = {
    "bool": TypeSpec(BaseType.BOOL),
    "int": TypeSpec(BaseType.INT),
    "float": TypeSpec(BaseType.FLOAT),
    "string": TypeSpec(BaseType.STRING),
    "array": TypeSpec(BaseType.ARRAY),
    "object": TypeSpec(BaseType.OBJECT),
    "action": TypeSpec(BaseType.ACTION),
    "any": TypeSpec(BaseType.ANY),
}


def parse_type_annotation(annotation: str) -> Optional[TypeSpec]:
    """
    Parse a type annotation string.
    
    Examples:
        "int" -> int
        "string?" -> nullable string
        "[int]" -> array of int
        "{name:string, age:int}" -> object with schema
    """
    annotation = annotation.strip()
    
    # Check nullable
    nullable = annotation.endswith("?")
    if nullable:
        annotation = annotation[:-1].strip()
    
    # Check array
    if annotation.startswith("[") and annotation.endswith("]"):
        inner = annotation[1:-1].strip()
        inner_type = parse_type_annotation(inner)
        if inner_type:
            return TypeSpec(BaseType.ARRAY, nullable=nullable, array_of=inner_type)
        return None
    
    # Check object schema
    if annotation.startswith("{") and annotation.endswith("}"):
        # Simplified: assume simple fields like {name:string, age:int}
        inner = annotation[1:-1].strip()
        if not inner:
            return TypeSpec(BaseType.OBJECT, nullable=nullable, object_schema={})
        
        schema = {}
        for field_str in inner.split(","):
            if ":" not in field_str:
                return None
            field_name, field_type_str = field_str.split(":", 1)
            field_type = parse_type_annotation(field_type_str.strip())
            if field_type:
                schema[field_name.strip()] = field_type
        
        return TypeSpec(BaseType.OBJECT, nullable=nullable, object_schema=schema)
    
    # Check standard types
    if annotation in STANDARD_TYPES:
        return TypeSpec(
            STANDARD_TYPES[annotation].base_type,
            nullable=nullable
        )
    
    return None
