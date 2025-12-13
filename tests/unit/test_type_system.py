"""
Tests for type system.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.type_system import (
    TypeSpec, TypeChecker, TypeInferencer, FunctionSignature,
    BaseType, STANDARD_TYPES, parse_type_annotation
)


def test_base_types():
    """Test basic type specifications."""
    int_type = TypeSpec(BaseType.INT)
    assert str(int_type) == "int"
    
    string_type = TypeSpec(BaseType.STRING)
    assert str(string_type) == "string"
    
    bool_type = TypeSpec(BaseType.BOOL)
    assert str(bool_type) == "bool"
    
    print("✓ base_types test passed")


def test_nullable_types():
    """Test nullable type specifications."""
    nullable_int = TypeSpec(BaseType.INT, nullable=True)
    assert "?" in str(nullable_int)
    
    print("✓ nullable_types test passed")


def test_array_types():
    """Test array type specifications."""
    int_array = TypeSpec(BaseType.ARRAY, array_of=TypeSpec(BaseType.INT))
    assert "[" in str(int_array)
    assert "]" in str(int_array)
    
    print("✓ array_types test passed")


def test_object_types():
    """Test object type specifications."""
    schema = {
        "name": TypeSpec(BaseType.STRING),
        "age": TypeSpec(BaseType.INT)
    }
    obj_type = TypeSpec(BaseType.OBJECT, object_schema=schema)
    type_str = str(obj_type)
    assert "{" in type_str
    assert "}" in type_str
    
    print("✓ object_types test passed")


def test_type_compatibility():
    """Test type compatibility checking."""
    int_type = TypeSpec(BaseType.INT)
    any_type = TypeSpec(BaseType.ANY)
    string_type = TypeSpec(BaseType.STRING)
    
    # Any is compatible with everything
    assert any_type.is_compatible_with(int_type)
    
    # Matching types are compatible
    assert int_type.is_compatible_with(int_type)
    
    # Different types are not compatible
    assert not int_type.is_compatible_with(string_type)
    
    print("✓ type_compatibility test passed")


def test_type_checker_int():
    """Test type checker for integers."""
    checker = TypeChecker()
    int_type = TypeSpec(BaseType.INT)
    
    # Valid int
    matches, reason = checker.check_type(42, int_type)
    assert matches
    
    # Invalid - string
    matches, reason = checker.check_type("not an int", int_type)
    assert not matches
    
    # Invalid - bool (bool is not int in Zexus)
    matches, reason = checker.check_type(True, int_type)
    assert not matches
    
    print("✓ type_checker_int test passed")


def test_type_checker_string():
    """Test type checker for strings."""
    checker = TypeChecker()
    string_type = TypeSpec(BaseType.STRING)
    
    # Valid string
    matches, reason = checker.check_type("hello", string_type)
    assert matches
    
    # Invalid - int
    matches, reason = checker.check_type(42, string_type)
    assert not matches
    
    print("✓ type_checker_string test passed")


def test_type_checker_nullable():
    """Test nullable types."""
    checker = TypeChecker()
    nullable_int = TypeSpec(BaseType.INT, nullable=True)
    non_nullable_int = TypeSpec(BaseType.INT, nullable=False)
    
    # None should match nullable
    matches, reason = checker.check_type(None, nullable_int)
    assert matches
    
    # None should not match non-nullable
    matches, reason = checker.check_type(None, non_nullable_int)
    assert not matches
    
    print("✓ type_checker_nullable test passed")


def test_type_checker_array():
    """Test array type checking."""
    checker = TypeChecker()
    int_array = TypeSpec(BaseType.ARRAY, array_of=TypeSpec(BaseType.INT))
    
    # Valid array of ints
    matches, reason = checker.check_type([1, 2, 3], int_array)
    assert matches
    
    # Invalid - contains string
    matches, reason = checker.check_type([1, "two", 3], int_array)
    assert not matches
    
    # Invalid - not array
    matches, reason = checker.check_type(42, int_array)
    assert not matches
    
    print("✓ type_checker_array test passed")


def test_type_checker_object():
    """Test object type checking."""
    checker = TypeChecker()
    schema = {
        "name": TypeSpec(BaseType.STRING),
        "age": TypeSpec(BaseType.INT)
    }
    obj_type = TypeSpec(BaseType.OBJECT, object_schema=schema)
    
    # Valid object
    obj = {"name": "Alice", "age": 30}
    matches, reason = checker.check_type(obj, obj_type)
    assert matches
    
    # Invalid - wrong type for field
    obj = {"name": "Bob", "age": "thirty"}
    matches, reason = checker.check_type(obj, obj_type)
    assert not matches
    
    # Invalid - missing field
    obj = {"name": "Charlie"}
    matches, reason = checker.check_type(obj, obj_type)
    assert not matches
    
    print("✓ type_checker_object test passed")


def test_type_checker_require_type():
    """Test require_type raises TypeError on mismatch."""
    checker = TypeChecker()
    int_type = TypeSpec(BaseType.INT)
    
    # Should not raise for valid type
    checker.require_type(42, int_type)
    
    # Should raise for invalid type
    try:
        checker.require_type("not an int", int_type)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    
    print("✓ type_checker_require_type test passed")


def test_type_inferencer():
    """Test type inference."""
    # Infer int
    inferred = TypeInferencer.infer_type(42)
    assert inferred.base_type == BaseType.INT
    
    # Infer string
    inferred = TypeInferencer.infer_type("hello")
    assert inferred.base_type == BaseType.STRING
    
    # Infer bool
    inferred = TypeInferencer.infer_type(True)
    assert inferred.base_type == BaseType.BOOL
    
    # Infer array
    inferred = TypeInferencer.infer_type([1, 2, 3])
    assert inferred.base_type == BaseType.ARRAY
    
    # Infer object
    inferred = TypeInferencer.infer_type({"key": "value"})
    assert inferred.base_type == BaseType.OBJECT
    
    print("✓ type_inferencer test passed")


def test_function_signature():
    """Test function signatures."""
    params = {
        "name": TypeSpec(BaseType.STRING),
        "age": TypeSpec(BaseType.INT)
    }
    return_type = TypeSpec(BaseType.STRING)
    
    sig = FunctionSignature("greet", params, return_type)
    
    # Validate correct call
    args = {"name": "Alice", "age": 30}
    checker = TypeChecker()
    valid, errors = sig.validate_call(args, checker)
    assert valid
    assert len(errors) == 0
    
    # Validate incorrect call
    args = {"name": "Bob"}  # Missing age
    valid, errors = sig.validate_call(args, checker)
    assert not valid
    assert len(errors) > 0
    
    print("✓ function_signature test passed")


def test_function_signature_return_validation():
    """Test validating function return values."""
    sig = FunctionSignature(
        "get_value",
        {},
        TypeSpec(BaseType.INT)
    )
    
    checker = TypeChecker()
    
    # Valid return
    matches, reason = sig.validate_return(42, checker)
    assert matches
    
    # Invalid return
    matches, reason = sig.validate_return("not an int", checker)
    assert not matches
    
    print("✓ function_signature_return_validation test passed")


def test_parse_type_annotation():
    """Test parsing type annotations."""
    # Simple type
    t = parse_type_annotation("int")
    assert t.base_type == BaseType.INT
    
    # Nullable
    t = parse_type_annotation("string?")
    assert t.nullable
    
    # Array
    t = parse_type_annotation("[int]")
    assert t.base_type == BaseType.ARRAY
    
    # Object
    t = parse_type_annotation("{name:string, age:int}")
    assert t.base_type == BaseType.OBJECT
    assert "name" in t.object_schema
    assert "age" in t.object_schema
    
    print("✓ parse_type_annotation test passed")


def test_standard_types():
    """Test standard type definitions."""
    assert "int" in STANDARD_TYPES
    assert "string" in STANDARD_TYPES
    assert "bool" in STANDARD_TYPES
    assert "array" in STANDARD_TYPES
    assert "object" in STANDARD_TYPES
    assert "action" in STANDARD_TYPES
    
    print("✓ standard_types test passed")


def test_custom_validator():
    """Test custom type validators."""
    checker = TypeChecker()
    
    # Create type with custom validator
    is_positive = lambda x: isinstance(x, int) and x > 0
    positive_int = TypeSpec(BaseType.INT, custom_validator=is_positive)
    
    # Valid
    matches, reason = checker.check_type(5, positive_int)
    assert matches
    
    # Invalid
    matches, reason = checker.check_type(-5, positive_int)
    assert not matches
    
    print("✓ custom_validator test passed")


if __name__ == '__main__':
    try:
        test_base_types()
        test_nullable_types()
        test_array_types()
        test_object_types()
        test_type_compatibility()
        test_type_checker_int()
        test_type_checker_string()
        test_type_checker_nullable()
        test_type_checker_array()
        test_type_checker_object()
        test_type_checker_require_type()
        test_type_inferencer()
        test_function_signature()
        test_function_signature_return_validation()
        test_parse_type_annotation()
        test_standard_types()
        test_custom_validator()
        print("\n✅ All type system tests passed!")
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
