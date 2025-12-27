# Print Statement Multi-Argument Support Fix

## Problem
Print statements with multiple comma-separated arguments were only displaying the first argument. For example:
```zexus
let x = 42
print("Value:", x)  // Only displayed "Value:" instead of "Value: 42"
```

## Root Cause
The Zexus parser had **three different print statement parsers**, all with the same issue:
1. They only parsed a single expression after the `print` keyword
2. The comma-separated arguments inside `print(...)` parentheses were not being split

The structural analyzer correctly collected all tokens including commas, but the parsers treated them as a single complex expression instead of multiple separate arguments.

## Solution
Updated all three print parsers to:
1. Strip the outer `print(...)` parentheses
2. Split arguments by commas at depth 0 (not inside nested parentheses)
3. Parse each argument as a separate expression
4. Store results in a `values` list instead of single `value`

### Files Modified

#### 1. `/src/zexus/zexus_ast.py` (Lines 178-183)
**Changed:** PrintStatement to support multiple values
```python
class PrintStatement(Statement):
    def __init__(self, value=None, values=None): 
        self.value = value
        self.values = values if values is not None else ([value] if value is not None else [])
```

#### 2. `/src/zexus/parser/strategy_context.py`
**Updated THREE different print parsers:**

- **`_parse_print_statement`** (Line 3140): Main parser for top-level print statements
- **`_parse_print_statement_block`** (Line 884): Parser for print in specific blocks  
- **`_parse_block_statements` PRINT handler** (Line 1663): Parser for print inside action/function bodies

All three now:
1. Strip outer parentheses if present
2. Split by commas at depth 0
3. Parse each argument separately
4. Return `PrintStatement(values=[...])`

#### 3. `/src/zexus/parser/parser.py` (Line 1181)
**Updated:** `parse_print_statement` to handle multiple arguments
- Parses first expression
- Loops while `peek_token_is(COMMA)` to collect additional arguments
- Maintains backward compatibility with `.value` field

#### 4. `/src/zexus/evaluator/statements.py` (Line 2297)
**Updated:** `eval_print_statement` to handle values list
- Checks for `node.values` (new) or `node.value` (legacy)
- Evaluates all expressions
- Joins output with spaces

#### 5. `/src/zexus/parser/strategy_structural.py` (Line 555)
**Fixed:** Removed premature breaking on RPAREN for PRINT statements
- Only `RETURN` and `CONTINUE` break after closing parens
- `PRINT` can now collect full argument list including commas

## Testing

### Before Fix
```zexus
print("a =", a)  → Output: "a ="
print("x =", x, "y =", y)  → Output: "x ="
```

### After Fix
```zexus
print("a =", 42)  → Output: "a = 42"
print("x =", 10, "y =", 20)  → Output: "x = 10 y = 20"
print("Mixed:", "string", 100, "test")  → Output: "Mixed: string 100 test"
```

### Test Cases Verified
✅ Single argument: `print("hello")`
✅ Two arguments: `print("Value:", 42)`
✅ Multiple arguments: `print("a", "b", "c")`
✅ Inside actions: `action test(x) { print("x =", x) }`
✅ Inside functions: Works correctly
✅ Mixed types: Strings, numbers, variables, expressions

## Impact
- **Fixes**: Entity method parameter debugging (what appeared to be a parameter passing bug was actually print not displaying values)
- **Improves**: Developer experience with multi-argument print statements
- **Maintains**: Backward compatibility with single-argument prints
- **Performance**: Minimal impact - only evaluates expressions that are provided

## Related Issues
This fix revealed that what initially appeared to be an "entity method parameters not being passed" bug was actually the print statement not displaying multiple arguments correctly.

## Status
✅ **COMPLETE** - All three print parsers updated, tested, and working correctly
