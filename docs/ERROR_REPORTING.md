# Zexus Error Reporting System

## Overview

We've implemented a comprehensive error reporting system for Zexus that provides:

- **Clear, beginner-friendly error messages**
- **Color-coded output with source code context**
- **Helpful suggestions for fixing errors**
- **Distinction between user code errors and interpreter bugs**

## Features

### 1. Error Categories

- **UserCode Errors**: Errors in the user's Zexus code (syntax, type, name errors)
- **Interpreter Errors**: Bugs in the Zexus interpreter itself

### 2. Error Types

- `SyntaxError`: Syntax errors in Zexus code
- `NameError`: Undefined identifiers
- `TypeError`: Type mismatches
- `ValueError`: Invalid values
- `AttributeError`: Attribute access errors
- `IndexError`: Index out of bounds
- `PatternMatchError`: Pattern matching errors
- `ImportError`: Module import errors
- `InterpreterError`: Internal interpreter bugs

### 3. Error Formatting

Errors are displayed with:
- Line and column numbers
- Source code snippet with pointer (^)
- Clear error message
- Helpful suggestion for fixing
- Color coding (red for errors, yellow for warnings)

## Examples

### Lexer Errors

#### Unterminated String
```zexus
let message = "Hello world
```

**Output:**
```
ERROR: SyntaxError[SYNTAX]
  → test.zx:1:16

  1 | let message = "Hello world
                      ^

  Unterminated string literal

  💡 Suggestion: Add a closing quote " to terminate the string.
```

#### Single & Instead of &&
```zexus
if (x & y) {
    print("test")
}
```

**Output:**
```
ERROR: SyntaxError[SYNTAX]
  → test.zx:1:8

  1 | if (x & y) {
              ^

  Unexpected character '&'

  💡 Suggestion: Did you mean '&&' for logical AND?
```

### Parser Errors

#### Missing Condition
```zexus
if {
    print("no condition!")
}
```

**Output:**
```
ERROR: SyntaxError[SYNTAX]
  → test.zx:1:4

  1 | if {
         ^

  Expected condition after 'if'

  💡 Suggestion: Add a condition expression: if (condition) { ... }
```

#### Missing Variable Name
```zexus
let = 42
```

**Output:**
```
ERROR: SyntaxError[SYNTAX]
  → test.zx:1:5

  1 | let = 42
          ^

  Expected variable name after 'let'

  💡 Suggestion: Use 'let' to declare a variable: let myVariable = value
```

## Usage

### In Code

```python
from src.zexus.error_reporter import get_error_reporter, SyntaxError, print_error

# Register source code
error_reporter = get_error_reporter()
error_reporter.register_source('myfile.zx', source_code)

# Create and raise an error
error = error_reporter.report_error(
    SyntaxError,
    "Invalid syntax",
    line=10,
    column=5,
    filename='myfile.zx',
    suggestion="Check your syntax and try again."
)
raise error

# Or print without raising
print_error(error)
```

### In Lexer

```python
from .error_reporter import get_error_reporter, SyntaxError as ZexusSyntaxError

# In lexer initialization
self.error_reporter = get_error_reporter()
self.error_reporter.register_source(filename, source_code)

# When creating errors
error = self.error_reporter.report_error(
    ZexusSyntaxError,
    "Unterminated string literal",
    line=self.line,
    column=self.column,
    filename=self.filename,
    suggestion="Add a closing quote \" to terminate the string."
)
raise error
```

### In Parser

```python
def _create_parse_error(self, message, suggestion=None, token=None):
    if token is None:
        token = self.cur_token
    
    return self.error_reporter.report_error(
        ZexusSyntaxError,
        message,
        line=getattr(token, 'line', None),
        column=getattr(token, 'column', None),
        filename=self.filename,
        suggestion=suggestion
    )

# Usage
error = self._create_parse_error(
    "Expected '}' to close map literal",
    suggestion="Make sure all opening braces { have matching closing braces }"
)
raise error
```

## Architecture

### Error Reporter

The `ErrorReporter` class tracks:
- Source code for each file
- Current file being processed
- Helps generate context-aware error messages

### Error Formatting

Errors are formatted with ANSI color codes for terminal display:
- **Red**: Errors
- **Yellow**: Warnings  
- **Blue**: Line numbers and info
- **Cyan**: Location indicators

Colors are automatically disabled when output is not to a terminal.

### Suggestions

The error reporter can generate context-aware suggestions based on error type:
- Undefined variable → suggest similar names
- Type mismatch → show expected vs actual
- Missing semicolon → explain Zexus doesn't need semicolons
- Pattern match failure → suggest adding wildcard

## Implementation Status

### ✅ Completed
- Error reporting infrastructure
- Lexer integration (unterminated strings, illegal characters)
- Parser integration (missing conditions, invalid syntax)
- Color-coded terminal output
- Source code context display
- Helpful suggestions

### 🚧 In Progress
- Evaluator integration (runtime errors)
- Stack trace formatting
- Error recovery strategies

### 📋 Planned
- IDE integration (LSP support)
- Error codes with documentation links
- Multilingual error messages
- Error statistics and analytics

## Benefits

### For Beginners
- **Clear messages**: No cryptic error codes
- **Visual context**: See exactly where the error is
- **Helpful hints**: Suggestions guide toward fixes
- **No confusion**: Distinguish code errors from interpreter bugs

### For Experienced Developers
- **Fast debugging**: Line/column precision
- **Stack traces**: When needed for complex errors
- **Color coding**: Quick visual parsing
- **Consistent format**: Easy to parse programmatically

## Comparison with Other Languages

### Better than Python
- ✅ Clearer indentation error messages
- ✅ Visual pointer to exact error location
- ✅ Helpful suggestions built-in
- ✅ Distinguishes user vs interpreter errors

### On Par with Rust
- ✅ Color-coded output
- ✅ Source code snippets
- ✅ Helpful suggestions
- ⚠️ Not yet: Error codes with docs links

### Better than JavaScript
- ✅ More informative messages
- ✅ Better formatting
- ✅ Context-aware suggestions

## Next Steps

1. Complete evaluator integration for runtime errors
2. Add stack trace formatting with call chain
3. Implement error recovery for better fault tolerance
4. Create error code catalog with detailed explanations
5. Add LSP support for IDE integration
