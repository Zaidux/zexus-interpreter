"""
Data Validation & Sanitization System for Zexus.

Provides built-in validation and sanitization primitives to prevent
common security vulnerabilities like injection attacks, type mismatches, 
and malformed input.

Key Features:
- Schema validation (JSON-like schemas with type checking)
- Input sanitization (remove/escape dangerous characters)
- Common validators (email, URL, phone, IP, range, length)
- Custom validator registration
- Encoding-aware sanitization (HTML, SQL, JavaScript, URL)
"""

from typing import Any, Dict, List, Optional, Callable, Pattern, Union
import re
import json
import html
import urllib.parse
from abc import ABC, abstractmethod
from enum import Enum


class ValidationError(Exception):
    """Exception raised when validation fails."""
    def __init__(self, message: str, field: str = "", value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(f"Validation error in '{field}': {message}")


class SanitizationError(Exception):
    """Exception raised when sanitization fails."""
    pass


class Encoding(Enum):
    """Common encoding types for sanitization."""
    HTML = "html"           # HTML entity encoding
    URL = "url"             # URL percent encoding
    SQL = "sql"             # SQL escape sequences
    JAVASCRIPT = "javascript"  # JavaScript string escaping
    CSV = "csv"             # CSV field escaping
    NONE = "none"           # No encoding


class Validator(ABC):
    """Base class for custom validators."""
    
    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Return True if valid, False otherwise."""
        pass
    
    @abstractmethod
    def get_error_message(self) -> str:
        """Return human-readable error message."""
        pass


class RegexValidator(Validator):
    """Validate against a regex pattern."""
    
    def __init__(self, pattern: Union[str, Pattern], message: str = ""):
        self.pattern = pattern if isinstance(pattern, Pattern) else re.compile(pattern)
        self.message = message or f"Value does not match pattern: {pattern}"
    
    def validate(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        return self.pattern.match(str(value)) is not None
    
    def get_error_message(self) -> str:
        return self.message


class RangeValidator(Validator):
    """Validate numeric ranges."""
    
    def __init__(self, min_val: Optional[float] = None, 
                 max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value: Any) -> bool:
        try:
            num = float(value)
            if self.min_val is not None and num < self.min_val:
                return False
            if self.max_val is not None and num > self.max_val:
                return False
            return True
        except (TypeError, ValueError):
            return False
    
    def get_error_message(self) -> str:
        if self.min_val is not None and self.max_val is not None:
            return f"Value must be between {self.min_val} and {self.max_val}"
        elif self.min_val is not None:
            return f"Value must be >= {self.min_val}"
        elif self.max_val is not None:
            return f"Value must be <= {self.max_val}"
        return "Invalid range"


class LengthValidator(Validator):
    """Validate string/list length."""
    
    def __init__(self, min_len: Optional[int] = None, 
                 max_len: Optional[int] = None):
        self.min_len = min_len
        self.max_len = max_len
    
    def validate(self, value: Any) -> bool:
        try:
            length = len(value)
            if self.min_len is not None and length < self.min_len:
                return False
            if self.max_len is not None and length > self.max_len:
                return False
            return True
        except (TypeError, AttributeError):
            return False
    
    def get_error_message(self) -> str:
        if self.min_len is not None and self.max_len is not None:
            return f"Length must be between {self.min_len} and {self.max_len}"
        elif self.min_len is not None:
            return f"Length must be >= {self.min_len}"
        elif self.max_len is not None:
            return f"Length must be <= {self.max_len}"
        return "Invalid length"


class ChoiceValidator(Validator):
    """Validate against allowed choices."""
    
    def __init__(self, choices: List[Any]):
        self.choices = set(choices)
    
    def validate(self, value: Any) -> bool:
        return value in self.choices
    
    def get_error_message(self) -> str:
        return f"Value must be one of: {self.choices}"


class TypeValidator(Validator):
    """Validate type."""
    
    def __init__(self, expected_type: Union[type, tuple]):
        self.expected_type = expected_type
    
    def validate(self, value: Any) -> bool:
        return isinstance(value, self.expected_type)
    
    def get_error_message(self) -> str:
        return f"Value must be of type {self.expected_type}"


class CompositeValidator(Validator):
    """Combine multiple validators (all must pass)."""
    
    def __init__(self, validators: List[Validator]):
        self.validators = validators
    
    def validate(self, value: Any) -> bool:
        return all(v.validate(value) for v in self.validators)
    
    def get_error_message(self) -> str:
        return " AND ".join(v.get_error_message() for v in self.validators)


class ValidationSchema:
    """Define validation rules for data structures."""
    
    def __init__(self, rules: Dict[str, Union[Validator, type, List[Validator]]]):
        """
        Initialize schema with validation rules.
        
        Args:
            rules: Dict mapping field names to validators or types
                - str value: check isinstance(value, str)
                - Validator instance: use custom validator
                - List[Validator]: use CompositeValidator
        """
        self.rules: Dict[str, Validator] = {}
        
        for field, rule in rules.items():
            if isinstance(rule, Validator):
                self.rules[field] = rule
            elif isinstance(rule, type):
                self.rules[field] = TypeValidator(rule)
            elif isinstance(rule, list):
                self.rules[field] = CompositeValidator(rule)
            else:
                raise ValueError(f"Invalid rule for field {field}: {rule}")
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate data against schema."""
        for field, validator in self.rules.items():
            if field not in data:
                raise ValidationError(f"Missing required field: {field}", field)
            
            value = data[field]
            if not validator.validate(value):
                raise ValidationError(
                    validator.get_error_message(),
                    field,
                    value
                )
        
        return True
    
    def validate_partial(self, data: Dict[str, Any], 
                        required_fields: Optional[List[str]] = None) -> bool:
        """Validate only specified fields (partial validation)."""
        fields_to_check = required_fields or list(self.rules.keys())
        
        for field in fields_to_check:
            if field not in self.rules:
                continue
            
            if field not in data:
                raise ValidationError(f"Missing required field: {field}", field)
            
            value = data[field]
            validator = self.rules[field]
            if not validator.validate(value):
                raise ValidationError(
                    validator.get_error_message(),
                    field,
                    value
                )
        
        return True


class StandardValidators:
    """Collection of standard, reusable validators."""
    
    # Email validation
    EMAIL = RegexValidator(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "Invalid email format"
    )
    
    # URL validation
    URL = RegexValidator(
        r'^https?://[^\s/$.?#].[^\s]*$',
        "Invalid URL format"
    )
    
    # IPv4 address
    IPV4 = RegexValidator(
        r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
        "Invalid IPv4 address"
    )
    
    # IPv6 address
    IPV6 = RegexValidator(
        r'^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4})$',
        "Invalid IPv6 address"
    )
    
    # Phone number (basic US format)
    PHONE = RegexValidator(
        r'^(\+?1)?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
        "Invalid phone number"
    )
    
    # UUID
    UUID = RegexValidator(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        "Invalid UUID format"
    )
    
    # Alphanumeric only
    ALPHANUMERIC = RegexValidator(
        r'^[a-zA-Z0-9]+$',
        "Must contain only alphanumeric characters"
    )
    
    # Positive integer
    POSITIVE_INT = CompositeValidator([
        TypeValidator(int),
        RangeValidator(min_val=0)
    ])
    
    # Non-empty string
    NON_EMPTY_STRING = CompositeValidator([
        TypeValidator(str),
        LengthValidator(min_len=1)
    ])


class Sanitizer:
    """Sanitize untrusted input to prevent injection attacks."""
    
    # HTML injection patterns
    DANGEROUS_HTML_TAGS = {
        'script', 'iframe', 'embed', 'object', 'link', 'style', 'meta'
    }
    
    DANGEROUS_ATTRIBUTES = {
        'onclick', 'onload', 'onerror', 'onmouseover', 'onmouseout',
        'onchange', 'onfocus', 'onblur', 'onfocus', 'onsubmit'
    }
    
    # SQL keywords for detection
    SQL_KEYWORDS = {
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 
        'FROM', 'WHERE', 'OR', 'AND'
    }
    
    @staticmethod
    def sanitize_string(value: str, encoding: Encoding = Encoding.HTML) -> str:
        """
        Sanitize a string value.
        
        Args:
            value: String to sanitize
            encoding: Type of encoding to apply
        
        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            raise SanitizationError(f"Expected string, got {type(value)}")
        
        if encoding == Encoding.HTML:
            return Sanitizer._sanitize_html(value)
        elif encoding == Encoding.URL:
            return urllib.parse.quote(value, safe='')
        elif encoding == Encoding.SQL:
            return Sanitizer._sanitize_sql(value)
        elif encoding == Encoding.JAVASCRIPT:
            return Sanitizer._sanitize_javascript(value)
        elif encoding == Encoding.CSV:
            return Sanitizer._sanitize_csv(value)
        elif encoding == Encoding.NONE:
            return value
        else:
            raise SanitizationError(f"Unknown encoding: {encoding}")
    
    @staticmethod
    def _sanitize_html(value: str) -> str:
        """Remove dangerous HTML tags and attributes."""
        # Escape HTML entities
        value = html.escape(value, quote=True)
        
        # Additional cleanup for remaining tags
        value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE)
        
        return value
    
    @staticmethod
    def _sanitize_sql(value: str) -> str:
        """Escape SQL special characters."""
        # Escape single quotes by doubling them
        return value.replace("'", "''")
    
    @staticmethod
    def _sanitize_javascript(value: str) -> str:
        """Escape JavaScript special characters."""
        replacements = {
            '"': '\\"',
            "'": "\\'",
            "\n": "\\n",
            "\r": "\\r",
            "\t": "\\t",
            "\b": "\\b",
            "\f": "\\f"
        }
        
        for char, escaped in replacements.items():
            value = value.replace(char, escaped)
        
        return value
    
    @staticmethod
    def _sanitize_csv(value: str) -> str:
        """Escape CSV field."""
        if '"' in value or ',' in value or '\n' in value:
            value = '"' + value.replace('"', '""') + '"'
        return value
    
    @staticmethod
    def sanitize_dict(data: Dict[str, Any], 
                     encoding: Encoding = Encoding.HTML) -> Dict[str, Any]:
        """Sanitize all string values in a dictionary."""
        result = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = Sanitizer.sanitize_string(value, encoding)
            elif isinstance(value, dict):
                result[key] = Sanitizer.sanitize_dict(value, encoding)
            elif isinstance(value, list):
                result[key] = Sanitizer.sanitize_list(value, encoding)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def sanitize_list(data: List[Any], 
                     encoding: Encoding = Encoding.HTML) -> List[Any]:
        """Sanitize all string values in a list."""
        result = []
        
        for value in data:
            if isinstance(value, str):
                result.append(Sanitizer.sanitize_string(value, encoding))
            elif isinstance(value, dict):
                result.append(Sanitizer.sanitize_dict(value, encoding))
            elif isinstance(value, list):
                result.append(Sanitizer.sanitize_list(value, encoding))
            else:
                result.append(value)
        
        return result


class ValidationManager:
    """
    Central validation/sanitization manager for the interpreter.
    
    Provides a unified interface for:
    - Registering custom validators
    - Validating data against schemas
    - Sanitizing untrusted input
    - Tracking validation/sanitization history
    """
    
    def __init__(self):
        self.custom_validators: Dict[str, Validator] = {}
        self.schemas: Dict[str, ValidationSchema] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Initialize built-in schemas for common types
        self._register_builtin_schemas()
    
    def _register_builtin_schemas(self):
        """Register built-in schemas for common data types."""
        # String type schema
        self.register_schema("string", ValidationSchema({
            "_type": TypeValidator(str)
        }))
        
        # Integer type schema
        self.register_schema("integer", ValidationSchema({
            "_type": TypeValidator(int)
        }))
        
        # Number type schema (int or float)
        self.register_schema("number", ValidationSchema({
            "_type": TypeValidator((int, float))
        }))
        
        # Boolean type schema
        self.register_schema("boolean", ValidationSchema({
            "_type": TypeValidator(bool)
        }))
        
        # Email schema (uses standard validator)
        self.register_schema("email", ValidationSchema({
            "_value": StandardValidators.EMAIL
        }))
        
        # URL schema (uses standard validator)
        self.register_schema("url", ValidationSchema({
            "_value": StandardValidators.URL
        }))
        
        # Phone schema (uses standard validator)
        self.register_schema("phone", ValidationSchema({
            "_value": StandardValidators.PHONE
        }))
        
        # UUID schema (uses standard validator)
        self.register_schema("uuid", ValidationSchema({
            "_value": StandardValidators.UUID
        }))
        
        # IPv4 schema
        self.register_schema("ipv4", ValidationSchema({
            "_value": StandardValidators.IPV4
        }))
        
        # IPv6 schema
        self.register_schema("ipv6", ValidationSchema({
            "_value": StandardValidators.IPV6
        }))
    
    def register_validator(self, name: str, validator: Validator):
        """Register a custom validator."""
        self.custom_validators[name] = validator
    
    def get_validator(self, name: str) -> Optional[Validator]:
        """Get a registered validator by name."""
        return self.custom_validators.get(name)
    
    def register_schema(self, name: str, schema: ValidationSchema):
        """Register a validation schema."""
        self.schemas[name] = schema
    
    def get_schema(self, name: str) -> Optional[ValidationSchema]:
        """Get a registered schema by name."""
        return self.schemas.get(name)
    
    def validate(self, value: Any, validator_name: str) -> bool:
        """Validate a value using a registered validator."""
        validator = self.get_validator(validator_name)
        if not validator:
            raise ValueError(f"Unknown validator: {validator_name}")
        
        self._record_validation(validator_name, value, validator.validate(value))
        return validator.validate(value)
    
    def validate_schema(self, data: Dict[str, Any], schema_name: str) -> bool:
        """Validate data against a registered schema."""
        schema = self.get_schema(schema_name)
        if not schema:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        return schema.validate(data)
    
    def sanitize(self, value: str, encoding: Encoding = Encoding.HTML) -> str:
        """Sanitize a string value."""
        result = Sanitizer.sanitize_string(value, encoding)
        self._record_sanitization(value, result, encoding.value)
        return result
    
    def _record_validation(self, validator: str, value: Any, result: bool):
        """Record validation operation."""
        self.history.append({
            "type": "validation",
            "validator": validator,
            "value": str(value)[:100],  # Truncate long values
            "result": result
        })
    
    def _record_sanitization(self, original: str, sanitized: str, encoding: str):
        """Record sanitization operation."""
        self.history.append({
            "type": "sanitization",
            "encoding": encoding,
            "original": original[:50],
            "sanitized": sanitized[:50],
            "length": len(original)
        })
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get validation/sanitization history."""
        if limit:
            return self.history[-limit:]
        return self.history.copy()


# Global instance
_validation_manager = ValidationManager()


def get_validation_manager() -> ValidationManager:
    """Get the global validation manager instance."""
    return _validation_manager
