"""
Comprehensive tests for new security features in Zexus.

Tests cover:
1. Capability-based access control (CAPABILITY, GRANT, REVOKE)
2. Pure function enforcement (PURE, IMMUTABLE)
3. Data validation and sanitization (VALIDATE, SANITIZE)
"""

import pytest
from src.zexus.capability_system import (
    CapabilityManager, CapabilityLevel, SelectivePolicy, DenyAllPolicy
)
from src.zexus.purity_system import (
    PurityAnalyzer, PurityEnforcer, PurityViolationError, Immutability
)
from src.zexus.validation_system import (
    ValidationSchema, StandardValidators, Sanitizer, Encoding,
    ValidationError, ValidationManager, RegexValidator, RangeValidator
)


# ============================================================================
# CAPABILITY SYSTEM TESTS
# ============================================================================

class TestCapabilitySystem:
    """Test capability-based security system."""
    
    def test_capability_grant_and_check(self):
        """Test granting and checking capabilities."""
        manager = CapabilityManager()
        
        # Grant capabilities
        manager.grant_capability("user1", "io.read")
        manager.grant_capability("user1", "io.write")
        
        # Check capabilities
        assert manager.has_capability("user1", "io.read")
        assert manager.has_capability("user1", "io.write")
        assert not manager.has_capability("user1", "io.delete")
    
    def test_capability_policy_selective(self):
        """Test selective capability policy."""
        policy = SelectivePolicy(["io.read", "io.write"])
        manager = CapabilityManager(policy)
        
        assert manager.has_capability("any_user", "io.read")
        assert manager.has_capability("any_user", "io.write")
        assert not manager.has_capability("any_user", "io.delete")
    
    def test_capability_policy_deny_all(self):
        """Test deny-all policy."""
        policy = DenyAllPolicy()
        manager = CapabilityManager(policy)
        
        assert not manager.has_capability("any_user", "io.read")
        assert not manager.has_capability("any_user", "network.tcp")
    
    def test_capability_audit_log(self):
        """Test capability audit logging."""
        manager = CapabilityManager()
        manager.grant_capability("user1", "io.read")
        
        # Check capability (this logs)
        manager.check_capability("user1", "io.read")
        manager.check_capability("user1", "io.delete")  # Denied
        
        # Get audit log
        log = manager.get_audit_log()
        assert len(log) == 2
        assert log[0]["granted"] == True
        assert log[1]["granted"] == False
    
    def test_base_capabilities_always_available(self):
        """Test that base capabilities are always available."""
        manager = CapabilityManager(DenyAllPolicy())
        
        # Even with deny-all policy, base capabilities should be available
        assert manager.has_capability("user", "core.language")
        assert manager.has_capability("user", "core.math")
        assert manager.has_capability("user", "core.strings")
    
    def test_required_capabilities_validation(self):
        """Test validation of required capabilities."""
        manager = CapabilityManager()
        
        # Declare required capabilities
        manager.declare_required_capabilities("module1", ["io.read", "network.http"])
        
        # Check if all requirements are met
        valid, missing = manager.validate_requirements("module1")
        assert not valid  # Missing capabilities
        assert "io.read" in missing
        assert "network.http" in missing
        
        # Grant the capabilities
        manager.grant_capabilities("module1", ["io.read", "network.http"])
        
        # Now validation should pass
        valid, missing = manager.validate_requirements("module1")
        assert valid
        assert len(missing) == 0


# ============================================================================
# PURITY SYSTEM TESTS
# ============================================================================

class TestPuritySystem:
    """Test pure function enforcement."""
    
    def test_pure_function_detection(self):
        """Test detection of pure functions."""
        analyzer = PurityAnalyzer()
        
        # Pure function
        pure_code = """
        function add(a, b) {
            return a + b;
        }
        """
        sig = analyzer.analyze("add", pure_code, ["a", "b"])
        assert sig.actual_purity.name == "PURE"
        assert len(sig.side_effects) == 0
    
    def test_impure_function_detection_io(self):
        """Test detection of I/O operations (impure)."""
        analyzer = PurityAnalyzer()
        
        # Impure: performs I/O
        impure_code = """
        function read_file(path) {
            return open(path).read();
        }
        """
        sig = analyzer.analyze("read_file", impure_code, ["path"])
        assert sig.actual_purity.name in ["IMPURE", "RESTRICTED"]
        assert len(sig.side_effects) > 0
    
    def test_impure_function_detection_global_state(self):
        """Test detection of global state access (impure)."""
        analyzer = PurityAnalyzer()
        
        # Impure: accesses global state
        impure_code = """
        function update_global() {
            globals()['counter'] += 1;
            return globals()['counter'];
        }
        """
        sig = analyzer.analyze("update_global", impure_code, [])
        assert sig.actual_purity.name in ["IMPURE", "RESTRICTED"]
    
    def test_impure_function_detection_exception(self):
        """Test detection of exception throwing (side effect)."""
        analyzer = PurityAnalyzer()
        
        code = """
        function divide(a, b) {
            if b == 0:
                raise "Division by zero";
            return a / b;
        }
        """
        sig = analyzer.analyze("divide", code, ["a", "b"])
        assert len(sig.side_effects) > 0
    
    def test_immutability_marking(self):
        """Test marking objects as immutable."""
        immutability = Immutability()
        
        data = {"name": "Alice", "age": 30}
        immutability.mark_immutable(data)
        
        assert immutability.is_immutable(data)
    
    def test_immutability_nested_structures(self):
        """Test immutability on nested structures."""
        immutability = Immutability()
        
        data = {"user": {"name": "Alice", "age": 30}}
        immutability.mark_immutable(data)
        
        # Both outer and inner should be marked
        assert immutability.is_immutable(data)
        # Note: nested dict is also marked
        assert immutability.is_immutable(data["user"])


# ============================================================================
# VALIDATION SYSTEM TESTS
# ============================================================================

class TestValidationSystem:
    """Test data validation system."""
    
    def test_standard_email_validator(self):
        """Test email validation."""
        validator = StandardValidators.EMAIL
        
        assert validator.validate("user@example.com")
        assert not validator.validate("invalid-email")
        assert not validator.validate("@example.com")
    
    def test_standard_url_validator(self):
        """Test URL validation."""
        validator = StandardValidators.URL
        
        assert validator.validate("https://example.com")
        assert validator.validate("http://example.com/path")
        assert not validator.validate("not a url")
    
    def test_standard_ipv4_validator(self):
        """Test IPv4 validation."""
        validator = StandardValidators.IPV4
        
        assert validator.validate("192.168.1.1")
        assert validator.validate("10.0.0.1")
        assert not validator.validate("256.256.256.256")
        assert not validator.validate("invalid.ip")
    
    def test_range_validator(self):
        """Test range validation."""
        validator = RangeValidator(min_val=0, max_val=100)
        
        assert validator.validate(50)
        assert validator.validate(0)
        assert validator.validate(100)
        assert not validator.validate(-1)
        assert not validator.validate(101)
    
    def test_validation_schema(self):
        """Test validation schema."""
        schema = ValidationSchema({
            "name": str,
            "age": int,
            "email": StandardValidators.EMAIL
        })
        
        # Valid data
        valid_data = {
            "name": "Alice",
            "age": 30,
            "email": "alice@example.com"
        }
        assert schema.validate(valid_data)
        
        # Invalid data
        invalid_data = {
            "name": "Alice",
            "age": 30,
            "email": "invalid-email"
        }
        with pytest.raises(ValidationError):
            schema.validate(invalid_data)
    
    def test_missing_required_field(self):
        """Test validation of missing required field."""
        schema = ValidationSchema({
            "name": str,
            "age": int
        })
        
        incomplete_data = {"name": "Alice"}
        
        with pytest.raises(ValidationError) as exc_info:
            schema.validate(incomplete_data)
        
        assert "age" in str(exc_info.value)


# ============================================================================
# SANITIZATION SYSTEM TESTS
# ============================================================================

class TestSanitizationSystem:
    """Test input sanitization system."""
    
    def test_html_sanitization(self):
        """Test HTML entity sanitization."""
        dangerous = '<script>alert("XSS")</script>'
        sanitized = Sanitizer.sanitize_string(dangerous, Encoding.HTML)
        
        # Script tag should be escaped
        assert '<script>' not in sanitized
        assert '&lt;' in sanitized or '&#' in sanitized
    
    def test_url_sanitization(self):
        """Test URL encoding."""
        unsafe = 'hello world & special chars!'
        sanitized = Sanitizer.sanitize_string(unsafe, Encoding.URL)
        
        # Spaces and special chars should be encoded
        assert '%20' in sanitized or '+' in sanitized
        assert '%26' in sanitized  # &
    
    def test_sql_sanitization(self):
        """Test SQL escape."""
        dangerous = "user'; DROP TABLE users; --"
        sanitized = Sanitizer.sanitize_string(dangerous, Encoding.SQL)
        
        # Single quotes should be escaped
        assert "''" in sanitized
    
    def test_javascript_sanitization(self):
        """Test JavaScript escape."""
        dangerous = 'var x = "hello"; alert("test");'
        sanitized = Sanitizer.sanitize_string(dangerous, Encoding.JAVASCRIPT)
        
        # Quotes should be escaped
        assert '\\"' in sanitized or "\\'" in sanitized
    
    def test_csv_sanitization(self):
        """Test CSV field escaping."""
        dangerous = 'value with "quotes" and, commas'
        sanitized = Sanitizer.sanitize_string(dangerous, Encoding.CSV)
        
        # Should wrap in quotes and escape inner quotes
        assert sanitized.startswith('"')
        assert sanitized.endswith('"')
        assert '""' in sanitized
    
    def test_sanitize_dict(self):
        """Test sanitizing entire dictionary."""
        unsafe_data = {
            "name": '<script>alert("xss")</script>',
            "email": "user@example.com"
        }
        
        sanitized = Sanitizer.sanitize_dict(unsafe_data, Encoding.HTML)
        
        assert '<script>' not in sanitized["name"]
        assert sanitized["email"] == "user@example.com"
    
    def test_sanitize_nested_structure(self):
        """Test sanitizing nested structures."""
        unsafe_data = {
            "user": {
                "name": '<img src=x onerror="alert()">',
                "posts": [
                    {"title": '<script>alert("xss")</script>'}
                ]
            }
        }
        
        sanitized = Sanitizer.sanitize_dict(unsafe_data, Encoding.HTML)
        
        # Nested dangerous content should be sanitized
        assert '<img' not in sanitized["user"]["name"]
        assert '<script>' not in sanitized["user"]["posts"][0]["title"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSecurityIntegration:
    """Integration tests combining multiple security features."""
    
    def test_capability_with_validation(self):
        """Test using capabilities with validation."""
        # User with io.read capability
        manager = CapabilityManager()
        manager.grant_capability("user1", "io.read")
        
        # Validate file path
        schema = ValidationSchema({
            "path": StandardValidators.ALPHANUMERIC
        })
        
        file_data = {"path": "/path/to/file"}
        
        # Both capability check and validation should pass
        assert manager.has_capability("user1", "io.read")
        assert schema.validate(file_data)
    
    def test_purity_with_immutability(self):
        """Test pure functions with immutable data."""
        analyzer = PurityAnalyzer()
        immutability = Immutability()
        
        # Pure function with immutable inputs
        pure_code = """
        function sum(numbers) {
            total = 0;
            for num in numbers:
                total = total + num;
            return total;
        }
        """
        
        sig = analyzer.analyze("sum", pure_code, ["numbers"])
        
        # Immutable input list
        numbers = [1, 2, 3, 4, 5]
        immutability.mark_immutable(numbers)
        
        assert sig.actual_purity.name == "PURE"
        assert immutability.is_immutable(numbers)
    
    def test_sanitize_then_validate(self):
        """Test sanitizing input then validating it."""
        # Untrusted input
        user_input = '<img src=x onerror="alert()"> alice@example.com'
        
        # Sanitize
        sanitized = Sanitizer.sanitize_string(user_input, Encoding.HTML)
        
        # Then extract email part and validate
        email_part = "alice@example.com"
        validator = StandardValidators.EMAIL
        
        assert validator.validate(email_part)
        assert '<img' not in sanitized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
