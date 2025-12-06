# const - Immutable Variable Declaration

## Overview

The `const` keyword declares immutable (constant) variables that cannot be reassigned after initialization. This is useful for configuration values, mathematical constants, and preventing accidental modifications.

**Status**: ✅ Implemented in v1.1.0

## Syntax

```zexus
const NAME = value;
const CONSTANT_NAME = expression;
```

## Parameters

- **NAME**: Identifier for the constant variable (must be unique in current scope)
- **value/expression**: Any valid Zexus expression to be assigned as the constant value

## Return Value

Returns `null` (statement, not an expression).

## Behavior

- The constant is initialized with the provided value
- Any attempt to reassign the constant raises an error
- Constants work with all Zexus types: primitives, objects, arrays, functions
- Scoping follows standard block scope rules

## Examples

### Example 1: Basic Numeric Constant

```zexus
const MAX_VALUE = 100;
const PI = 3.14159;

print(MAX_VALUE);  // Output: 100
print(PI);         // Output: 3.14159

// This will error:
MAX_VALUE = 200;   // Error: Cannot reassign const variable 'MAX_VALUE'
```

### Example 2: String Constants

```zexus
const API_BASE_URL = "https://api.example.com";
const API_KEY = "sk_live_abc123xyz";
const DEFAULT_TIMEOUT = 5000;

print(API_BASE_URL);  // Output: https://api.example.com
```

### Example 3: Object Constants

```zexus
const CONFIG = {
  debug: false,
  timeout: 5000,
  retries: 3,
  endpoints: {
    users: "/api/users",
    posts: "/api/posts"
  }
};

print(CONFIG.timeout);              // Output: 5000
print(CONFIG.endpoints.users);      // Output: /api/users

// Cannot reassign the object itself:
CONFIG = {};  // Error: Cannot reassign const variable 'CONFIG'
```

### Example 4: Array Constants

```zexus
const ALLOWED_ROLES = ["admin", "moderator", "user", "guest"];
const RGB_VALUES = [255, 128, 64];

print(ALLOWED_ROLES[0]);  // Output: admin
print(RGB_VALUES[2]);     // Output: 64

// Cannot reassign the array:
ALLOWED_ROLES = [];  // Error: Cannot reassign const variable 'ALLOWED_ROLES'
```

### Example 5: Function Constants

```zexus
const calculate_area = fn(width, height) {
  return width * height;
};

print(calculate_area(5, 10));  // Output: 50

// Cannot reassign the function:
calculate_area = fn() { return 0; };  // Error: Cannot reassign const variable
```

### Example 6: Scope Examples

```zexus
const GLOBAL_MAX = 1000;

if true {
  const LOCAL_MAX = 100;
  print(GLOBAL_MAX);  // Output: 1000 (inherited)
  print(LOCAL_MAX);   // Output: 100
}

// print(LOCAL_MAX);  // Error: LOCAL_MAX not defined (out of scope)

for i in 1..5 {
  const ITERATION = i;
  print(ITERATION);   // Prints: 1, 2, 3, 4, 5
}
```

### Example 7: Combined with seal

```zexus
const seal SECURE_CONFIG = {
  api_key: "secret_key",
  db_url: "mongodb://..."
};

// Cannot reassign (const):
SECURE_CONFIG = {};  // Error

// Cannot modify properties (seal):
SECURE_CONFIG.api_key = "new_key";  // Error
```

## Error Handling

### Reassignment Error

```zexus
const X = 10;
X = 20;  // Runtime Error: Cannot reassign const variable 'X'
```

### Missing Value

```zexus
const Y =;  // Syntax Error: Expected expression
```

## Differences from `let`

| Feature | `let` | `const` |
|---------|-------|--------|
| Reassignable | ✅ Yes | ❌ No |
| Type | Mutable variable | Immutable variable |
| Performance | Standard | Can be optimized |
| Use Case | Changing values | Fixed configuration |

## Performance Notes

- Constants are optimized by the compiler (no reassignment checks needed)
- No runtime overhead compared to regular variables
- Can enable inline optimization opportunities

## Related Commands

- **`let`** - Mutable variable declaration
- **`seal`** - Make existing objects immutable
- **`if`** - Conditional with constants
- **`while`** - Loops with constant limits

## Best Practices

1. **Use for configuration**: Configuration values that never change
2. **Use for constants**: Mathematical constants, API endpoints
3. **Name convention**: UPPERCASE_WITH_UNDERSCORES for semantic clarity
4. **Scope appropriately**: Define constants in appropriate block scope
5. **Combine with seal**: Use `const seal obj` for complete immutability

## Implementation Details

### Files Modified
- `src/zexus/zexus_token.py` - Added CONST token
- `src/zexus/lexer.py` - Added "const" keyword recognition
- `src/zexus/zexus_ast.py` - Added ConstStatement AST node
- `src/zexus/parser.py` - Added parse_const_statement()
- `src/zexus/evaluator/statements.py` - Added eval_const_statement()
- `src/zexus/evaluator/core.py` - Added ConstStatement dispatch
- `src/zexus/object.py` - Added set_const() and const tracking to Environment

### Testing Checklist
- ✅ Basic constant declaration
- ✅ Const with primitives (int, float, string, bool)
- ✅ Const with complex types (object, array)
- ✅ Const with functions
- ✅ Reassignment prevention
- ✅ Scope isolation
- ✅ Combined with seal
- ✅ Error messages

## Version History

### v1.1.0 (December 2025)
- Initial implementation
- Full support for all value types
- Immutability enforcement at runtime
- Integration with seal for complete object protection

## See Also

- [ENHANCEMENT_PACKAGE/00_START_HERE.md](../../ENHANCEMENT_PACKAGE/00_START_HERE.md) - Overview of all features
- [ENHANCEMENT_PACKAGE/CODE_EXAMPLES.md](../../ENHANCEMENT_PACKAGE/CODE_EXAMPLES.md#2-const---immutable-variables-) - Additional examples
- [ENHANCEMENT_PACKAGE/IMPLEMENTATION_GUIDE.md](../../ENHANCEMENT_PACKAGE/IMPLEMENTATION_GUIDE.md#feature-const---immutable-variables) - Implementation details
