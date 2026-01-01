# src/zexus/security_enforcement.py
"""
Security enforcement for Zexus language.

This module enforces mandatory sanitization in sensitive contexts.
It's NOT optional - security is built into the language.
"""

from .object import String, EvaluationError


class SecurityEnforcementError(Exception):
    """Raised when unsanitized input is used in sensitive context"""
    pass


class SensitiveContext:
    """Defines sensitive contexts that require sanitization"""
    
    SQL = 'sql'
    HTML = 'html'
    URL = 'url'
    SHELL = 'shell'
    
    # Patterns that indicate SQL context
    SQL_PATTERNS = [
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
        'ALTER', 'FROM', 'WHERE', 'JOIN', 'UNION'
    ]
    
    # Patterns that indicate HTML context
    HTML_PATTERNS = [
        '<html', '<div', '<span', '<script', '<body', '<head',
        'innerHTML', 'outerHTML'
    ]
    
    # Patterns that indicate URL context
    URL_PATTERNS = [
        'http://', 'https://', 'ftp://', '?', '&', 'url=', 'redirect='
    ]
    
    # Patterns that indicate shell context
    SHELL_PATTERNS = [
        'exec', 'system', 'shell', 'bash', 'sh', 'cmd', 'powershell'
    ]


def detect_sensitive_context(string_value):
    """
    Detect if a string is being used in a sensitive context.
    
    Returns the context type (sql, html, url, shell) or None.
    
    IMPORTANT: This now uses more sophisticated pattern matching to reduce
    false positives. We look for actual dangerous patterns, not just keywords.
    """
    if not isinstance(string_value, str):
        return None
    
    upper_value = string_value.upper()
    
    # Check for SQL context - require actual SQL query patterns, not just keywords
    # Look for patterns like "SELECT ... FROM", "WHERE ... =", etc.
    sql_query_indicators = [
        ('SELECT', 'FROM'),  # SELECT must be followed by FROM
        ('INSERT', 'INTO'),  # INSERT must be followed by INTO
        ('UPDATE', 'SET'),   # UPDATE must be followed by SET
        ('DELETE', 'FROM'),  # DELETE must be followed by FROM
        ('DROP', 'TABLE'),   # DROP must be followed by TABLE
        ('CREATE', 'TABLE'), # CREATE must be followed by TABLE
    ]
    
    for keyword1, keyword2 in sql_query_indicators:
        if keyword1 in upper_value and keyword2 in upper_value:
            # Found a real SQL query pattern
            return SensitiveContext.SQL
    
    # Single keywords alone are not enough - they could be normal text
    # Only trigger if we see SQL-like syntax patterns
    if ' WHERE ' in upper_value and ('=' in string_value or 'LIKE' in upper_value):
        return SensitiveContext.SQL
    
    # Check for HTML context - require actual HTML tags, not just keywords
    for pattern in SensitiveContext.HTML_PATTERNS:
        if pattern.lower() in string_value.lower():
            # Check if it's actually a tag (starts with <)
            if pattern.startswith('<') or 'innerHTML' in string_value or 'outerHTML' in string_value:
                return SensitiveContext.HTML
    
    # Check for URL context - require actual URL schemes or injection patterns
    url_indicators = ['http://', 'https://', 'ftp://']
    injection_indicators = ['url=', 'redirect=', 'goto=', 'next=']
    
    has_url_scheme = any(indicator in string_value.lower() for indicator in url_indicators)
    has_injection_param = any(indicator in string_value.lower() for indicator in injection_indicators)
    
    if has_url_scheme or (has_injection_param and ('?' in string_value or '&' in string_value)):
        return SensitiveContext.URL
    
    # Check for shell context - require actual command execution patterns
    shell_execution_funcs = ['exec(', 'system(', 'shell(', 'bash ', 'sh ', 'cmd ', 'powershell ']
    if any(pattern in string_value.lower() for pattern in shell_execution_funcs):
        return SensitiveContext.SHELL
    
    return None


def enforce_sanitization(string_obj, operation_context=None):
    """
    Enforce sanitization requirement for String objects in sensitive contexts.
    
    This is ALWAYS enforced - not optional. Security is built into the language.
    
    Args:
        string_obj: The String object to check
        operation_context: Optional explicit context (sql, html, url, shell)
        
    Raises:
        EvaluationError: If unsanitized input is used in sensitive context
    """
    if not isinstance(string_obj, String):
        return  # Not a string, nothing to enforce
    
    # If string is trusted (literal), no enforcement needed
    if string_obj.is_trusted:
        return
    
    # Detect context if not explicitly provided
    if operation_context is None:
        operation_context = detect_sensitive_context(string_obj.value)
    
    # If no sensitive context detected, allow
    if operation_context is None:
        return
    
    # Check if string is sanitized for this context
    if not string_obj.is_safe_for(operation_context):
        raise_sanitization_error(string_obj, operation_context)


def raise_sanitization_error(string_obj, context):
    """
    Raise a clear, helpful error message for unsanitized input.
    
    The error message guides developers to use the sanitize keyword.
    """
    context_name = context.upper()
    
    # Create helpful error message
    error_msg = f"""
🔒 SECURITY ERROR: Unsanitized input used in {context_name} context

The string value appears to be used in a {context_name} operation, but it has not been sanitized.
This could lead to {get_vulnerability_name(context)} vulnerabilities.

To fix this, sanitize the input before use:

    sanitize your_variable as {context}
    
Example:
    
    ❌ UNSAFE:
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    
    ✅ SAFE:
    sanitize user_input as {context}
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"

Security is mandatory in Zexus - this protection cannot be disabled.
"""
    
    raise SecurityEnforcementError(error_msg.strip())


def get_vulnerability_name(context):
    """Get the vulnerability name for a given context"""
    vuln_map = {
        SensitiveContext.SQL: "SQL Injection",
        SensitiveContext.HTML: "Cross-Site Scripting (XSS)",
        SensitiveContext.URL: "URL Injection / Open Redirect",
        SensitiveContext.SHELL: "Command Injection"
    }
    return vuln_map.get(context, "Injection")


def check_string_concatenation(left, right):
    """
    Check string concatenation for security issues.
    
    When concatenating strings, if the result would be used in a sensitive
    context, both operands must be sanitized or trusted.
    
    Improvements:
    - If BOTH operands are trusted (literals), the result is safe
    - Only check context on the final combined result
    - Reduce false positives from normal text operations
    """
    # If either operand is a String object, check sanitization
    left_is_string = isinstance(left, String)
    right_is_string = isinstance(right, String)
    
    if not (left_is_string or right_is_string):
        return  # Not string concatenation
    
    # OPTIMIZATION: If both are trusted literals, the concatenation is safe
    if (left_is_string and left.is_trusted) and (right_is_string and right.is_trusted):
        return  # Both sides are literals - safe!
    
    # Get the concatenated value for context detection
    left_val = left.value if left_is_string else str(left.inspect() if hasattr(left, 'inspect') else left)
    right_val = right.value if right_is_string else str(right.inspect() if hasattr(right, 'inspect') else right)
    combined = left_val + right_val
    
    # Detect if the combined string is in a sensitive context
    context = detect_sensitive_context(combined)
    
    if context is None:
        return  # No sensitive context detected
    
    # Check if both operands are safe for this context
    # NOTE: We only enforce if the string is NOT trusted AND NOT sanitized
    if left_is_string and not left.is_trusted and not left.is_safe_for(context):
        enforce_sanitization(left, context)
    
    if right_is_string and not right.is_trusted and not right.is_safe_for(context):
        enforce_sanitization(right, context)


def mark_as_trusted(string_obj):
    """
    Mark a string as trusted (from literal, not external input).
    
    This should be called when creating String objects from literals.
    """
    if isinstance(string_obj, String):
        string_obj.is_trusted = True
    return string_obj
