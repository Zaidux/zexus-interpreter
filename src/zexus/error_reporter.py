"""
Zexus Error Reporting System

Provides clear, beginner-friendly error messages that distinguish between
user code errors and interpreter bugs.

Security Fix #10: Debug info sanitization to prevent sensitive data leakage.
"""

import sys
from typing import Optional, List, Dict, Any, Sequence
from enum import Enum


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insert
                prev_row[j + 1] + 1,   # delete
                prev_row[j] + cost,     # substitute
            ))
        prev_row = curr_row
    return prev_row[-1]


def find_closest_match(name: str, candidates: Sequence[str], max_distance: int = 3) -> Optional[str]:
    """Find the closest matching name from a list of candidates.
    
    Uses Levenshtein distance for accurate "did you mean?" suggestions.
    Returns the best match or None if nothing is close enough.
    """
    if not candidates:
        return None
    best = None
    best_dist = max_distance + 1
    name_lower = name.lower()
    for candidate in candidates:
        # Quick exact/case check
        if candidate.lower() == name_lower:
            return candidate
        dist = _levenshtein_distance(name_lower, candidate.lower())
        if dist < best_dist:
            best_dist = dist
            best = candidate
    return best if best_dist <= max_distance else None


# Import debug sanitizer for Security Fix #10
try:
    from .debug_sanitizer import get_sanitizer
    _SANITIZER_AVAILABLE = True
except ImportError:
    _SANITIZER_AVAILABLE = False


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
        message = self.message
        # Security Fix #10: Sanitize error messages
        if _SANITIZER_AVAILABLE:
            sanitizer = get_sanitizer()
            message = sanitizer.sanitize_message(message)
        
        parts.append(f"  {message}")
        
        # Suggestion
        if self.suggestion:
            suggestion = self.suggestion
            # Sanitize suggestions too
            if _SANITIZER_AVAILABLE:
                suggestion = get_sanitizer().sanitize_message(suggestion)
            parts.append("")
            parts.append(f"  {YELLOW}💡 Suggestion:{RESET} {suggestion}")
        
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


class DivisionError(ZexusError):
    """Division by zero and modulo by zero errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="DIVISION", **kwargs)


class FormulaError(ZexusError):
    """Unknown or invalid formula/function errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="FORMULA", **kwargs)


class BraceMismatchError(ZexusError):
    """Unmatched or mismatched braces/brackets/parentheses"""
    def __init__(self, message: str, **kwargs):
        if 'suggestion' not in kwargs:
            kwargs['suggestion'] = (
                "Zexus uses curly braces { } for code blocks. "
                "Make sure every opening brace has a matching closing brace."
            )
        super().__init__(message, error_code="BRACE", **kwargs)


class ArgumentError(ZexusError):
    """Wrong number or type of function arguments"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="ARGS", **kwargs)


class NotCallableError(ZexusError):
    """Attempt to call something that is not a function"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CALL", **kwargs)


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
    
    def check_brace_balance(self, source: str, filename: Optional[str] = None) -> Optional[ZexusError]:
        """Check for unmatched braces, brackets, and parentheses.
        
        Returns a BraceMismatchError if imbalanced, or None if OK.
        """
        stack = []  # (char, line, col)
        pairs = {'{': '}', '[': ']', '(': ')'}
        closers = {'}': '{', ']': '[', ')': '('}
        in_string = False
        string_char = None
        in_comment = False
        in_line_comment = False
        
        lines = source.splitlines()
        for line_num, line_text in enumerate(lines, 1):
            i = 0
            while i < len(line_text):
                ch = line_text[i]
                
                # Handle line comments
                if in_line_comment:
                    break  # Rest of line is comment
                
                # Handle multi-line comments
                if in_comment:
                    if ch == '*' and i + 1 < len(line_text) and line_text[i + 1] == '/':
                        in_comment = False
                        i += 2
                        continue
                    i += 1
                    continue
                
                # Handle strings
                if in_string:
                    if ch == '\\':
                        i += 2  # Skip escaped char
                        continue
                    if ch == string_char:
                        in_string = False
                    i += 1
                    continue
                
                # Detect comment start
                if ch == '/' and i + 1 < len(line_text):
                    if line_text[i + 1] == '/':
                        in_line_comment = True
                        break
                    elif line_text[i + 1] == '*':
                        in_comment = True
                        i += 2
                        continue
                
                # Detect string start
                if ch in ('"', "'", '`'):
                    in_string = True
                    string_char = ch
                    i += 1
                    continue
                
                # Track braces/brackets/parens
                if ch in pairs:
                    stack.append((ch, line_num, i))
                elif ch in closers:
                    expected_opener = closers[ch]
                    if not stack:
                        return BraceMismatchError(
                            f"Unexpected closing '{ch}' with no matching opening '{expected_opener}'",
                            filename=filename or self.current_file,
                            line=line_num,
                            column=i,
                            source_line=line_text,
                        )
                    opener, open_line, open_col = stack.pop()
                    if opener != expected_opener:
                        return BraceMismatchError(
                            f"Mismatched brackets: opening '{opener}' at line {open_line} "
                            f"does not match closing '{ch}'",
                            filename=filename or self.current_file,
                            line=line_num,
                            column=i,
                            source_line=line_text,
                            suggestion=(
                                f"The '{opener}' opened at line {open_line}, column {open_col} "
                                f"expects a closing '{pairs[opener]}', but found '{ch}' instead."
                            ),
                        )
                
                i += 1
            
            in_line_comment = False  # Reset for next line
        
        if stack:
            opener, open_line, open_col = stack[-1]
            return BraceMismatchError(
                f"Unclosed '{opener}' — expected closing '{pairs[opener]}' before end of file",
                filename=filename or self.current_file,
                line=open_line,
                column=open_col,
                source_line=lines[open_line - 1] if open_line <= len(lines) else None,
                suggestion=(
                    f"Add a matching '{pairs[opener]}' to close the block that starts at "
                    f"line {open_line}, column {open_col}."
                ),
            )
        
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
        """
        suggestions = {
            "undefined_variable": lambda ctx: (
                f"Did you mean '{ctx.get('similar')}'?" if ctx.get('similar')
                else "Make sure the variable is declared before using it."
            ),
            "undefined_function": lambda ctx: (
                f"Did you mean '{ctx.get('similar')}'?" if ctx.get('similar')
                else "Make sure the function is defined before calling it. "
                     "Use 'action name() {{ }}' to define a function."
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
            "wrong_arg_count": lambda ctx: (
                f"Function '{ctx.get('name', '?')}' expects {ctx.get('expected')} "
                f"argument(s), but got {ctx.get('actual')}."
            ),
            "not_callable": lambda ctx: (
                f"'{ctx.get('name', 'value')}' is a {ctx.get('type', 'value')}, not a function. "
                "You can only call functions, actions, and lambdas."
            ),
            "unknown_formula": lambda ctx: (
                f"Unknown function or formula '{ctx.get('name')}'. "
                + (f"Did you mean '{ctx.get('similar')}'?" if ctx.get('similar') else
                   "Check the Zexus documentation for available built-in functions.")
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
