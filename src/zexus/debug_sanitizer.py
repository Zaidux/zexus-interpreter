"""
Zexus Debug Information Sanitizer

Security Fix #10: Debug Info Sanitization
Prevents sensitive information from leaking through debug output and error messages.
"""

import re
import os
from typing import Any, Dict, List, Optional


class DebugSanitizer:
    """
    Sanitizes debug information to prevent sensitive data leakage.
    
    Removes or masks:
    - Passwords and API keys
    - Database connection strings
    - File paths (optional, based on mode)
    - Environment variables
    - Stack traces (in production mode)
    """
    
    # Patterns for sensitive data
    SENSITIVE_PATTERNS = [
        # Passwords
        (re.compile(r'password\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), 'password=***'),
        (re.compile(r'passwd\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), 'passwd=***'),
        (re.compile(r'pwd\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), 'pwd=***'),
        
        # API Keys and Tokens
        (re.compile(r'api[_-]?key\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), 'api_key=***'),
        (re.compile(r'secret[_-]?key\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), 'secret_key=***'),
        (re.compile(r'auth[_-]?token\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), 'auth_token=***'),
        (re.compile(r'access[_-]?token\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), 'access_token=***'),
        (re.compile(r'bearer\s+([a-zA-Z0-9_\-\.]+)', re.IGNORECASE), 'bearer ***'),
        
        # Database credentials
        (re.compile(r'mysql://([^:]+):([^@]+)@', re.IGNORECASE), 'mysql://***:***@'),
        (re.compile(r'postgres://([^:]+):([^@]+)@', re.IGNORECASE), 'postgres://***:***@'),
        (re.compile(r'mongodb://([^:]+):([^@]+)@', re.IGNORECASE), 'mongodb://***:***@'),
        
        # Generic key=value patterns with sensitive keywords
        (re.compile(r'(private[_-]?key)\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), r'\1=***'),
        (re.compile(r'(encryption[_-]?key)\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), r'\1=***'),
        (re.compile(r'(client[_-]?secret)\s*[=:]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE), r'\1=***'),
    ]
    
    # Environment variables that should be masked
    SENSITIVE_ENV_VARS = {
        'PASSWORD', 'SECRET', 'TOKEN', 'KEY', 'API_KEY', 'DB_PASSWORD',
        'DATABASE_PASSWORD', 'MYSQL_PASSWORD', 'POSTGRES_PASSWORD',
        'MONGODB_PASSWORD', 'REDIS_PASSWORD', 'AWS_SECRET_ACCESS_KEY',
        'PRIVATE_KEY', 'ENCRYPTION_KEY', 'JWT_SECRET'
    }
    
    def __init__(self, production_mode: bool = None):
        """
        Initialize sanitizer
        
        Args:
            production_mode: If True, applies stricter sanitization.
                           If None, auto-detects from environment.
        """
        if production_mode is None:
            # Auto-detect from environment
            env_mode = os.environ.get('ZEXUS_ENV', 'development').lower()
            production_mode = env_mode in ('production', 'prod')
        
        self.production_mode = production_mode
    
    def sanitize_message(self, message: str) -> str:
        """
        Sanitize a message by removing sensitive information
        
        Args:
            message: The message to sanitize
            
        Returns:
            Sanitized message
        """
        if not isinstance(message, str):
            message = str(message)
        
        result = message
        
        # Apply all sensitive patterns
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            result = pattern.sub(replacement, result)
        
        # Sanitize file paths in production mode
        if self.production_mode:
            result = self._sanitize_file_paths(result)
        
        return result
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize a dictionary by masking sensitive values
        
        Args:
            data: Dictionary to sanitize
            
        Returns:
            Sanitized dictionary
        """
        if not isinstance(data, dict):
            return data
        
        result = {}
        for key, value in data.items():
            # Check if key indicates sensitive data
            key_lower = key.lower()
            is_sensitive = any(
                sensitive in key_lower 
                for sensitive in ['password', 'secret', 'token', 'key', 'api']
            )
            
            if is_sensitive:
                result[key] = '***'
            elif isinstance(value, dict):
                result[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                result[key] = self.sanitize_list(value)
            elif isinstance(value, str):
                result[key] = self.sanitize_message(value)
            else:
                result[key] = value
        
        return result
    
    def sanitize_list(self, data: List[Any]) -> List[Any]:
        """Sanitize a list by sanitizing each element"""
        if not isinstance(data, list):
            return data
        
        result = []
        for item in data:
            if isinstance(item, dict):
                result.append(self.sanitize_dict(item))
            elif isinstance(item, list):
                result.append(self.sanitize_list(item))
            elif isinstance(item, str):
                result.append(self.sanitize_message(item))
            else:
                result.append(item)
        
        return result
    
    def sanitize_stack_trace(self, stack_trace: str) -> str:
        """
        Sanitize stack trace information
        
        In production mode, removes internal file paths and limits detail.
        """
        if not self.production_mode:
            # In development, show full stack trace but sanitize sensitive data
            return self.sanitize_message(stack_trace)
        
        # In production, provide minimal stack trace
        lines = stack_trace.split('\n')
        sanitized_lines = []
        
        for line in lines:
            # Keep error messages but sanitize them
            if 'Error:' in line or 'Exception:' in line:
                sanitized_lines.append(self.sanitize_message(line))
            # Remove file system paths
            elif not line.strip().startswith('File '):
                sanitized_lines.append(line)
        
        return '\n'.join(sanitized_lines)
    
    def sanitize_environment(self, env_vars: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize environment variables
        
        Masks sensitive environment variables.
        """
        result = {}
        for key, value in env_vars.items():
            # Check if it's a sensitive environment variable
            key_upper = key.upper()
            is_sensitive = any(
                sensitive in key_upper 
                for sensitive in self.SENSITIVE_ENV_VARS
            )
            
            if is_sensitive:
                result[key] = '***'
            else:
                result[key] = value
        
        return result
    
    def _sanitize_file_paths(self, text: str) -> str:
        """
        Remove or generalize file paths in production mode
        
        Converts absolute paths to relative or generic paths.
        """
        # Replace home directory paths
        home = os.path.expanduser('~')
        text = text.replace(home, '~')
        
        # Replace absolute paths with relative
        import re
        text = re.sub(r'/[a-zA-Z0-9_\-/]+/([a-zA-Z0-9_\-\.]+\.zx)', r'./\1', text)
        
        return text
    
    def should_show_debug_info(self) -> bool:
        """
        Check if debug information should be shown
        
        Returns False in production mode.
        """
        return not self.production_mode


# Global sanitizer instance
_sanitizer = DebugSanitizer()


def get_sanitizer() -> DebugSanitizer:
    """Get the global debug sanitizer instance"""
    return _sanitizer


def set_production_mode(enabled: bool):
    """Set production mode globally"""
    global _sanitizer
    _sanitizer = DebugSanitizer(production_mode=enabled)


def sanitize_debug_output(message: str) -> str:
    """Quick function to sanitize debug output"""
    return _sanitizer.sanitize_message(message)


def sanitize_error_data(data: Any) -> Any:
    """Quick function to sanitize error data"""
    if isinstance(data, dict):
        return _sanitizer.sanitize_dict(data)
    elif isinstance(data, list):
        return _sanitizer.sanitize_list(data)
    elif isinstance(data, str):
        return _sanitizer.sanitize_message(data)
    return data
