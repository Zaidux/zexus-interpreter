# elif - Else-If Conditional Branching

## Overview

The `elif` keyword provides else-if conditional branching, allowing multiple mutually-exclusive conditions to be tested sequentially. This creates cleaner, more readable code compared to nested if-else statements.

**Status**: ✅ Implemented in v1.1.0

## Syntax

```zexus
if condition1 {
  // Block 1
} elif condition2 {
  // Block 2
} elif condition3 {
  // Block 3
} else {
  // Default block
}
```

## Components

- **if condition1**: First condition to test
- **elif condition2, condition3, ...**: Additional conditions to test (zero or more)
- **else**: Optional final block if no conditions match
- **Blocks**: Statements to execute if corresponding condition is true

## Behavior

- Conditions are evaluated left-to-right (top-to-bottom)
- Only the first true condition's block executes
- Remaining conditions are not evaluated once a match is found
- The `else` block executes only if no prior conditions were true
- Each condition must be a valid boolean expression

## Examples

### Example 1: Grade Classification

```zexus
let score = 85;
if score >= 90 {
  print("Grade: A");
} elif score >= 80 {
  print("Grade: B");
} elif score >= 70 {
  print("Grade: C");
} elif score >= 60 {
  print("Grade: D");
} else {
  print("Grade: F");
}
// Output: Grade: B
```

### Example 2: HTTP Status Codes

```zexus
let status = 404;
if status == 200 {
  print("Success");
} elif status == 301 || status == 302 {
  print("Redirect");
} elif status == 400 || status == 401 || status == 403 {
  print("Client error");
} elif status == 404 {
  print("Not found");
} elif status >= 500 {
  print("Server error");
} else {
  print("Unknown status");
}
// Output: Not found
```

### Example 3: Time of Day

```zexus
let hour = 14;
if hour < 6 {
  print("Night");
} elif hour < 12 {
  print("Morning");
} elif hour < 18 {
  print("Afternoon");
} elif hour < 21 {
  print("Evening");
} else {
  print("Late night");
}
// Output: Afternoon
```

### Example 4: User Role Access Control

```zexus
let user_role = "moderator";
if user_role == "admin" {
  print("Full access");
} elif user_role == "moderator" {
  print("Moderation access");
} elif user_role == "user" {
  print("Basic access");
} elif user_role == "guest" {
  print("View-only access");
} else {
  print("No access");
}
// Output: Moderation access
```

### Example 5: Type Checking

```zexus
let value = 42;
if type(value) == "string" {
  print("String value");
} elif type(value) == "i64" {
  print("Integer value");
} elif type(value) == "bool" {
  print("Boolean value");
} elif type(value) == "object" {
  print("Object value");
} else {
  print("Unknown type");
}
// Output: Integer value
```

### Example 6: Complex Conditions with elif

```zexus
let age = 25;
let employed = true;

if age < 18 {
  print("Minor");
} elif age < 25 && employed {
  print("Young professional");
} elif age < 25 {
  print("Young adult (student)");
} elif age < 65 {
  print("Adult");
} else {
  print("Senior");
}
// Output: Young professional
```

### Example 7: Nested elif

```zexus
let x = 10;
let y = 20;

if x > y {
  print("x is larger");
} elif x == y {
  print("Equal");
} elif x < y {
  if y > 30 {
    print("x smaller, y is much larger");
  } elif y > 15 {
    print("x smaller, y is moderately larger");
  } else {
    print("x smaller, y is slightly larger");
  }
}
// Output: x smaller, y is moderately larger
```

### Example 8: No else clause

```zexus
let code = 200;
if code == 200 {
  print("OK");
} elif code == 404 {
  print("Not Found");
} elif code == 500 {
  print("Server Error");
}
// Output: OK
// (if no condition matches, nothing prints)
```

## Compared to Nested If-Else

### Without elif (nested if-else)

```zexus
if x > 10 {
  print("large");
} else {
  if x > 5 {
    print("medium");
  } else {
    if x > 0 {
      print("small");
    } else {
      print("negative");
    }
  }
}
```

### With elif (cleaner)

```zexus
if x > 10 {
  print("large");
} elif x > 5 {
  print("medium");
} elif x > 0 {
  print("small");
} else {
  print("negative");
}
```

## Best Practices

1. **Use for multiple alternatives**: 3+ branches suggest elif is cleaner than nested if
2. **Order by likelihood**: Put most likely conditions first for performance
3. **Order by specificity**: Check specific conditions before general ones
4. **Keep it readable**: Long elif chains can be hard to read (use pattern matching for complex cases)
5. **Avoid duplication**: Don't repeat conditions in multiple branches

### Good Example

```zexus
let value = 42;
if value < 0 {
  print("negative");
} elif value == 0 {
  print("zero");
} elif value <= 100 {
  print("small");
} else {
  print("large");
}
```

### Problematic Example (Too Many Branches)

```zexus
// Consider pattern matching instead
if x == 1 {
  // ...
} elif x == 2 {
  // ...
} elif x == 3 {
  // ...
} elif x == 4 {
  // ...
} elif x == 5 {
  // ...
}
// Better: Use pattern matching (when available)
```

## Error Handling

### Invalid Condition

```zexus
if true {
  print("1");
} elif {  // Error: Expected condition
  print("2");
}
```

### Missing Block

```zexus
if true
  print("1");
elif true  // Error: Expected block start
  print("2");
```

## Performance Considerations

- Short-circuit evaluation: Subsequent conditions not tested after first match
- No performance overhead vs nested if-else
- Compiler can optimize based on condition patterns

## Differences from Nested If-Else

| Aspect | elif | nested if-else |
|--------|------|-----------------|
| Readability | Excellent | Poor (with many branches) |
| Performance | Same | Same |
| Indentation | Flat | Deep |
| Maintainability | Easy | Harder |
| Short-circuit | Yes | Yes |

## Related Commands

- **`if`** - Conditional statement
- **`else`** - Default branch
- **`pattern`** - Pattern matching (for complex conditions)
- **`&&`, `\|\|`** - Logical operators for complex conditions

## Implementation Details

### Files Modified
- `src/zexus/zexus_token.py` - Added ELIF token
- `src/zexus/lexer.py` - Added "elif" keyword recognition
- `src/zexus/zexus_ast.py` - Extended IfStatement and IfExpression with elif_parts
- `src/zexus/parser.py` - Updated parse_if_statement() for elif parsing
- `src/zexus/evaluator/statements.py` - Updated eval_if_statement() for elif evaluation

### AST Structure

```python
class IfStatement(Statement):
    def __init__(self, condition, consequence, elif_parts=None, alternative=None):
        self.condition = condition
        self.consequence = consequence
        self.elif_parts = elif_parts or []  # List of (condition, consequence) tuples
        self.alternative = alternative
```

### Evaluation Logic

1. Evaluate main condition
2. If true, execute consequence and stop
3. For each elif_part:
   - Evaluate elif condition
   - If true, execute elif consequence and stop
4. If no conditions matched and alternative exists, execute it

### Testing Checklist
- ✅ Single elif
- ✅ Multiple elif chains
- ✅ elif without else
- ✅ elif with complex conditions
- ✅ Nested elif
- ✅ elif with boolean operators
- ✅ Short-circuit evaluation

## Version History

### v1.1.0 (December 2025)
- Initial implementation
- Full elif chain support
- Works with all condition types
- Proper short-circuit evaluation

## Common Questions

### Q: How many elif clauses can I have?
**A**: Theoretically unlimited, but readability suggests keeping to 5 or fewer.

### Q: Can I use elif without else?
**A**: Yes, elif without else is perfectly valid.

### Q: Are conditions evaluated left-to-right?
**A**: Yes, conditions are evaluated in order until one is true.

### Q: Can I use assignments in elif conditions?
**A**: No, only boolean expressions allowed (assignments are not expressions in Zexus).

## See Also

- [ENHANCEMENT_PACKAGE/00_START_HERE.md](../../ENHANCEMENT_PACKAGE/00_START_HERE.md) - Overview of all features
- [ENHANCEMENT_PACKAGE/CODE_EXAMPLES.md](../../ENHANCEMENT_PACKAGE/CODE_EXAMPLES.md#10-elif---else-if-conditionals-) - More examples
- [ENHANCEMENT_PACKAGE/IMPLEMENTATION_GUIDE.md](../../ENHANCEMENT_PACKAGE/IMPLEMENTATION_GUIDE.md#feature-elif---else-if-conditionals) - Implementation details
