"""
Zexus Error Reporting System

Provides clear, beginner-friendly error messages that distinguish between
user code errors and interpreter bugs.
"""

import sys
from typing import Optional, List, Dict, Any
from enum import Enum


class ErrorCategory(Enum):
    """Categories of errors in Zexus"""
    USER_CODE = "user_code"  # Error in user's Zexus code
    INTERPRETER = "interpreter"  # Bug in the Zexus interpreter


class ErrorSeverity(Enum):
    """Severity levels for errors"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ZexusError(Exception):
    """Base class for all Zexus errors"""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.USER_CODE,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        filename: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
        source_line: Optional[str] = None,
        suggestion: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.filename = filename or "<stdin>"
        self.line = line
        self.column = column
        self.source_line = source_line
        self.suggestion = suggestion
        self.error_code = error_code
    
    def format_error(self) -> str:
        """Format the error message for display"""
        parts = []
        
        # Header with error type and location
        location = f"{self.filename}"
        if self.line is not None:
            location += f":{self.line}"
            if self.column is not None:
                location += f":{self.column}"
        
        # Color codes (ANSI)
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        
        # Disable colors if not in terminal
        if not sys.stderr.isatty():
            RED = YELLOW = BLUE = CYAN = BOLD = RESET = ""
        
        if self.severity == ErrorSeverity.ERROR:
            severity_color = RED
            severity_text = "ERROR"
        elif self.severity == ErrorSeverity.WARNING:
            severity_color = YELLOW
            severity_text = "WARNING"
        else:
            severity_color = BLUE
            severity_text = "INFO"
        
        # Error header
        error_type = self.__class__.__name__
        if self.error_code:
            error_type = f"{error_type}[{self.error_code}]"
        
        parts.append(f"{BOLD}{severity_color}{severity_text}{RESET}: {BOLD}{error_type}{RESET}")
        parts.append(f"  {CYAN}→{RESET} {location}")
        parts.append("")
        
        # Show source code with pointer
        if self.source_line is not None and self.line is not None:
            line_num_width = len(str(self.line))
            line_num = f"{self.line}".rjust(line_num_width)
            
            parts.append(f"  {BLUE}{line_num} |{RESET} {self.source_line}")
            
            # Add pointer to the error location
            if self.column is not None:
                pointer_padding = " " * (line_num_width + 3 + self.column)
                parts.append(f"  {pointer_padding}{RED}^{RESET}")
            
            parts.append("")
        
        # Error message
        parts.append(f"  {self.message}")
        
        # Suggestion
        if self.suggestion:
            parts.append("")
            parts.append(f"  {YELLOW}💡 Suggestion:{RESET} {self.suggestion}")
        
        # Internal error note
        if self.category == ErrorCategory.INTERPRETER:
            parts.append("")
            parts.append(f"  {RED}⚠️  This is an internal interpreter error.{RESET}")
            parts.append(f"  {RED}   Please report this bug to the Zexus developers.{RESET}")
        
        parts.append("")
        return "\n".join(parts)


class SyntaxError(ZexusError):
    """Syntax errors in Zexus code"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="SYNTAX", **kwargs)


class NameError(ZexusError):
    """Name/identifier not found errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="NAME", **kwargs)


class TypeError(ZexusError):
    """Type-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="TYPE", **kwargs)


class ValueError(ZexusError):
    """Value-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="VALUE", **kwargs)


class AttributeError(ZexusError):
    """Attribute access errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="ATTR", **kwargs)


class IndexError(ZexusError):
    """Index out of bounds errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="INDEX", **kwargs)


class PatternMatchError(ZexusError):
    """Pattern matching errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="PATTERN", **kwargs)


class ImportError(ZexusError):
    """Module import errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="IMPORT", **kwargs)


class InterpreterError(ZexusError):
    """Internal interpreter errors (bugs in Zexus itself)"""
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.INTERPRETER
        super().__init__(
            f"Internal error: {message}",
            error_code="INTERNAL",
            **kwargs
        )


class ErrorReporter:
    """
    Centralized error reporting for the Zexus interpreter.
    Tracks source code context for better error messages.
    """
    
    def __init__(self):
        self.source_lines: Dict[str, List[str]] = {}
        self.current_file: Optional[str] = None
    
    def register_source(self, filename: str, source: str):
        """Register source code for a file"""
        self.source_lines[filename] = source.splitlines()
        self.current_file = filename
    
    def get_source_line(self, filename: Optional[str], line: int) -> Optional[str]:
        """Get a specific line from the source code"""
        if filename is None:
            filename = self.current_file
        
        if filename and filename in self.source_lines:
            lines = self.source_lines[filename]
            if 0 < line <= len(lines):
                return lines[line - 1]
        
        return None
    
    def report_error(
        self,
        error_class,
        message: str,
        line: Optional[int] = None,
        column: Optional[int] = None,
        filename: Optional[str] = None,
        suggestion: Optional[str] = None,
        **kwargs
    ) -> ZexusError:
        """
        Create and return a properly formatted error.
        
        Args:
            error_class: The error class to instantiate
            message: Error message
            line: Line number where error occurred
            column: Column number where error occurred
            filename: Filename (defaults to current file)
            suggestion: Helpful suggestion for fixing the error
            **kwargs: Additional arguments for the error class
        
        Returns:
            ZexusError instance ready to be raised
        """
        if filename is None:
            filename = self.current_file
        
        source_line = None
        if line is not None:
            source_line = self.get_source_line(filename, line)
        
        return error_class(
            message=message,
            filename=filename,
            line=line,
            column=column,
            source_line=source_line,
            suggestion=suggestion,
            **kwargs
        )
    
    def create_suggestion(self, error_type: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Generate helpful suggestions based on error type and context.
        
        Args:
            error_type: Type of error (e.g., "undefined_variable", "type_mismatch")
            context: Additional context for generating suggestions
        
        Returns:
            Helpful suggestion string or None
        """
        suggestions = {
            "undefined_variable": lambda ctx: (
                f"Did you mean '{ctx.get('similar')}'?" if ctx.get('similar')
                else "Make sure the variable is declared before using it."
            ),
            "type_mismatch": lambda ctx: (
                f"Expected {ctx.get('expected')}, got {ctx.get('actual')}. "
                f"Try converting the value or checking your types."
            ),
            "missing_semicolon": lambda ctx: (
                "Zexus statements don't require semicolons. Remove the semicolon."
            ),
            "wrong_indentation": lambda ctx: (
                "Zexus uses curly braces {{ }} for blocks, not indentation. "
                "Make sure your braces are balanced."
            ),
            "pattern_no_match": lambda ctx: (
                "Add a wildcard pattern '_' as the last case to handle all values, "
                "or ensure your patterns cover all possible cases."
            ),
            "generic_type_args": lambda ctx: (
                f"This generic type requires {ctx.get('expected')} type argument(s). "
                f"Use: {ctx.get('example')}"
            ),
        }
        
        suggestion_fn = suggestions.get(error_type)
        if suggestion_fn:
            return suggestion_fn(context)
        
        return None


# Global error reporter instance
_error_reporter = ErrorReporter()


def get_error_reporter() -> ErrorReporter:
    """Get the global error reporter instance"""
    return _error_reporter


def format_error(error: ZexusError) -> str:
    """Format a ZexusError for display"""
    return error.format_error()


def print_error(error: ZexusError):
    """Print a formatted error to stderr"""
    print(error.format_error(), file=sys.stderr)
