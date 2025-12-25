"""
Input Validation Module for Zexus Interpreter

Provides comprehensive input validation for all public APIs to ensure
robust error handling and prevent crashes.

Location: src/zexus/input_validation.py
"""


def validate_string_input(value, param_name="value", allow_empty=True, max_length=None):
    """
    Validate string input parameters.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter (for error messages)
        allow_empty: Whether empty strings are allowed
        max_length: Maximum allowed length (None for unlimited)
        
    Returns:
        str: The validated string value
        
    Raises:
        TypeError: If value is not a string
        ValueError: If validation constraints are violated
    """
    if not isinstance(value, str):
        raise TypeError(f"{param_name} must be a string, got {type(value).__name__}")
    
    if not allow_empty and len(value) == 0:
        raise ValueError(f"{param_name} cannot be empty")
    
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"{param_name} exceeds maximum length of {max_length}")
    
    return value


def validate_integer_input(value, param_name="value", min_value=None, max_value=None):
    """
    Validate integer input parameters.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter (for error messages)
        min_value: Minimum allowed value (None for unlimited)
        max_value: Maximum allowed value (None for unlimited)
        
    Returns:
        int: The validated integer value
        
    Raises:
        TypeError: If value is not an integer
        ValueError: If validation constraints are violated
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{param_name} must be an integer, got {type(value).__name__}")
    
    if min_value is not None and value < min_value:
        raise ValueError(f"{param_name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} must be <= {max_value}, got {value}")
    
    return value


def validate_number_input(value, param_name="value", min_value=None, max_value=None):
    """
    Validate numeric (int or float) input parameters.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter (for error messages)
        min_value: Minimum allowed value (None for unlimited)
        max_value: Maximum allowed value (None for unlimited)
        
    Returns:
        The validated numeric value
        
    Raises:
        TypeError: If value is not numeric
        ValueError: If validation constraints are violated
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{param_name} must be numeric, got {type(value).__name__}")
    
    if min_value is not None and value < min_value:
        raise ValueError(f"{param_name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} must be <= {max_value}, got {value}")
    
    return value


def validate_collection_input(value, param_name="value", min_length=None, max_length=None):
    """
    Validate collection (list, tuple, set) input parameters.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter (for error messages)
        min_length: Minimum allowed length (None for unlimited)
        max_length: Maximum allowed length (None for unlimited)
        
    Returns:
        The validated collection
        
    Raises:
        TypeError: If value is not a collection
        ValueError: If validation constraints are violated
    """
    if not hasattr(value, '__len__') or isinstance(value, (str, bytes)):
        raise TypeError(f"{param_name} must be a collection, got {type(value).__name__}")
    
    length = len(value)
    
    if min_length is not None and length < min_length:
        raise ValueError(f"{param_name} must have at least {min_length} elements, got {length}")
    
    if max_length is not None and length > max_length:
        raise ValueError(f"{param_name} must have at most {max_length} elements, got {length}")
    
    return value


def validate_index(index, collection, param_name="index"):
    """
    Validate that an index is within bounds for a collection.
    
    Args:
        index: The index to validate
        collection: The collection being indexed
        param_name: Name of the parameter (for error messages)
        
    Returns:
        int: The validated index
        
    Raises:
        TypeError: If index is not an integer
        IndexError: If index is out of bounds
    """
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"{param_name} must be an integer, got {type(index).__name__}")
    
    if not hasattr(collection, '__len__'):
        raise TypeError(f"Cannot index into {type(collection).__name__}")
    
    length = len(collection)
    
    # Handle negative indices
    if index < 0:
        index = length + index
    
    if index < 0 or index >= length:
        raise IndexError(f"{param_name} {index} is out of bounds for collection of length {length}")
    
    return index


def validate_file_path(path, param_name="path", must_exist=False, must_be_file=False):
    """
    Validate file path input.
    
    Args:
        path: The file path to validate
        param_name: Name of the parameter (for error messages)
        must_exist: Whether the path must exist
        must_be_file: Whether the path must be a file (not directory)
        
    Returns:
        str: The validated path
        
    Raises:
        TypeError: If path is not a string
        ValueError: If validation constraints are violated
    """
    import os
    
    if not isinstance(path, str):
        raise TypeError(f"{param_name} must be a string, got {type(path).__name__}")
    
    if len(path) == 0:
        raise ValueError(f"{param_name} cannot be empty")
    
    if must_exist and not os.path.exists(path):
        raise ValueError(f"{param_name} does not exist: {path}")
    
    if must_be_file and os.path.exists(path) and not os.path.isfile(path):
        raise ValueError(f"{param_name} must be a file, got directory: {path}")
    
    return path


def validate_enum_input(value, allowed_values, param_name="value"):
    """
    Validate that a value is one of a set of allowed values.
    
    Args:
        value: The value to validate
        allowed_values: Collection of allowed values
        param_name: Name of the parameter (for error messages)
        
    Returns:
        The validated value
        
    Raises:
        ValueError: If value is not in allowed_values
    """
    if value not in allowed_values:
        raise ValueError(
            f"{param_name} must be one of {allowed_values}, got {value}"
        )
    
    return value


def validate_not_none(value, param_name="value"):
    """
    Validate that a value is not None.
    
    Args:
        value: The value to validate
        param_name: Name of the parameter (for error messages)
        
    Returns:
        The validated value (not None)
        
    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(f"{param_name} cannot be None")
    
    return value


# Convenience validators for common patterns

def validate_positive_integer(value, param_name="value"):
    """Validate that value is a positive integer (> 0)."""
    return validate_integer_input(value, param_name, min_value=1)


def validate_non_negative_integer(value, param_name="value"):
    """Validate that value is a non-negative integer (>= 0)."""
    return validate_integer_input(value, param_name, min_value=0)


def validate_non_empty_string(value, param_name="value"):
    """Validate that value is a non-empty string."""
    return validate_string_input(value, param_name, allow_empty=False)


def validate_percentage(value, param_name="value"):
    """Validate that value is a number between 0 and 100."""
    return validate_number_input(value, param_name, min_value=0, max_value=100)
