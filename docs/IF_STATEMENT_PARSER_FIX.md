# If Statement Parser Fix

## Issue
The context parser (`strategy_context.py`) had a bug in parsing if statements with colon syntax (`:`) where:
1. It only supported brace-style blocks (`{ }`) but not colon-style blocks
2. The condition token collection logic was breaking early when encountering function calls

## Symptoms
- If statements with colon syntax like `if x > 3: print("yes")` would hang/fail
- If statements with function calls in conditions like `if len(list) > 0 { ... }` wouldn't execute
- The condition tokens were being truncated at the closing parenthesis of function calls

## Root Cause
1. **Missing Colon Support**: The parser at line 1475 only checked for `LBRACE` (`{`) tokens, not `COLON` (`:`) tokens
2. **Premature Token Collection Break**: When collecting condition tokens, the parser would break early when encountering a closing paren, assuming it was wrapping the whole condition

## Solution
Updated `/workspaces/zexus-interpreter/src/zexus/parser/strategy_context.py`:

### 1. Added Colon-Style Block Support
Modified the condition token collection loop to check for both `LBRACE` and `COLON`:
```python
while j < len(tokens) and tokens[j].type not in [LBRACE, COLON]:
```

Added colon-style block parsing for if, elif, and else blocks:
```python
elif j < len(tokens) and tokens[j].type == COLON:
    # Colon-style block - collect until next statement keyword
    j += 1
    inner_tokens = []
    while j < len(tokens):
        if tokens[j].type in [IF, ELIF, ELSE, WHILE, FOR, ...]:
            break
        inner_tokens.append(tokens[j])
        j += 1
```

### 2. Fixed Parenthesis Handling Logic
Improved the parenthesis tracking to only break when the closing paren actually ends the condition:
```python
# Only break if we're closing the tentatively skipped outer paren
# AND there are more tokens after it (not end of condition)
if paren_depth == 0 and skipped_outer_paren and len(cond_tokens) > 0:
    j += 1
    # Check if next token is { or : (end of condition)
    if j < len(tokens) and tokens[j].type in [LBRACE, COLON]:
        break
    # Otherwise continue collecting
    skipped_outer_paren = False
    continue
```

## Results
✅ Colon-style if statements now parse correctly
✅ Function calls in conditions work properly
✅ Both brace and colon syntax supported
✅ elif and else blocks also support both syntaxes

## Test Cases Fixed
- `if x > 3: print("yes")` - Simple colon-style
- `if len(__ARGS__) > 0 { ... }` - Function call in condition
- `if argc == 0 { ... } elif argc == 1 { ... } else { ... }` - Full elif/else chains

## Files Modified
- `src/zexus/parser/strategy_context.py` (Lines ~1467-1650)
  - Updated if statement parsing (lines 1467-1535)
  - Updated elif/else parsing (lines 1537-1650)

## Impact
This fix is critical for the main entry point enhancements (__ARGS__, __MODULE__, etc.) as they rely on conditional logic to check argument counts and module states.
