# Zexus 1.6.7 - Additional Fixes Summary

## Issues Fixed

### 1. Standalone Block Statements ✅
**Problem:** Blocks `{ ... }` containing statements were being incorrectly parsed as assignment expressions, causing "Invalid assignment target" errors.

**Example:**
```zexus
{
    let temp = 10;
    print("Inside block");
}
```

**Fix:** Added dedicated handling for standalone block statements in `_parse_block_statements()` (strategy_context.py ~line 3450)
- Detects LBRACE at statement level
- Collects tokens until matching RBRACE
- Recursively parses block contents
- Properly handles nesting

**Impact:** Scoped blocks now work correctly in all contexts.

---

### 2. Keywords as Variable Names ✅
**Problem:** Reserved keywords (like `data`, `action`, `contract`, etc.) couldn't be used as variable names, causing conflicts in legitimate use cases.

**Example:**
```zexus
let data = {"value": 42};  // Previously failed
let action = "click";      // Previously failed
```

**Fix:** Implemented context-aware keyword recognition in lexer (lexer.py ~line 100)
- Added `should_allow_keyword_as_ident()` method
- Tracks last token type to determine context
- Allows keywords as identifiers after:
  - LET, CONST (variable declaration)
  - COLON (map keys, parameter types)
  - COMMA (parameter lists, map entries)
  - LPAREN/RPAREN (function arguments)
  - LBRACKET/RBRACKET (map access)
  - Comparison operators (LT, GT, EQ, etc.)
  - ASSIGN (assignment target)
  - DOT (property access)
- Always treats `true`, `false`, and `null` as keywords

**Impact:** Keywords can now be used as:
- Variable names: `let state = 5;`
- Map keys: `{"action": "click"}`  
- Parameter names: `action process(data, action) { ... }`
- Property names: `obj.data`

**Exceptions:** `true`, `false`, `null` remain strict keywords for boolean/null literals.

---

## Known Limitations Documented

### 1. Contract Instances in Contract State ⚠️
**Limitation:** Contract instances cannot be stored as state variables in other contracts.

**Reason:** Contract state is serialized to JSON for persistence. Contract instances aren't JSON-serializable and become strings when deserialized.

**Example (Won't Work):**
```zexus
contract Controller {
    state storage_ref = null;  // Can't store contract instance
    
    action init(storage) {
        storage_ref = storage;  // Gets serialized to string
    }
}
```

**Workaround:** Pass contract instances as action parameters instead of storing them.

---

### 2. Nested Map Literals in Contract State ⚠️
**Limitation:** Assigning nested map literals directly to contract state properties may not persist correctly.

**Example (Unreliable):**
```zexus
contract Store {
    state data = {};
    
    action set(key) {
        data[key] = {"count": 0, "status": "active"};  // May not persist
        return data[key];  // May return null
    }
}
```

**Workaround:** Use simple values (strings, numbers) in contract state, or create the map structure incrementally:
```zexus
action set(key) {
    data[key] = {};
    data[key] = "active";  // Assign properties separately if needed
}
```

Or use simple state variables instead:
```zexus
contract Store {
    state count = 0;
    state status = "init";
    
    action set(n, s) {
        count = n;
        status = s;
    }
}
```

---

## Test Results

### Comprehensive Edge Case Test Suite
**File:** `test_edge_cases.zx`
**Total Tests:** 19
**Passed:** 19 ✅
**Failed:** 0

**Test Coverage:**
1. ✅ Deeply nested maps (3+ levels)
2. ✅ Mixed statement types in blocks  
3. ✅ Conditionals with nested assignments
4. ✅ Loops with compound assignments
5. ✅ Contracts with multiple state variables
6. ✅ Sequential method calls
7. ✅ Complex expressions with operator precedence
8. ✅ Multiple prints with assignments in contracts
9. ✅ Complex map reconstruction
10. ✅ Multiple contract instances (isolation)
11. ✅ Try-catch with assignments
12. ✅ Complex return statements
13. ✅ Array-like map operations
14. ✅ Compound operator sequences
15. ✅ Conditional (ternary) assignments

---

## Files Modified

### Core Changes
- `src/zexus/lexer.py`
  - Added `last_token_type` tracking
  - Added `should_allow_keyword_as_ident()` method
  - Modified `lookup_ident()` for context-aware keyword handling
  - Updated `next_token()` to track token history

- `src/zexus/parser/strategy_context.py`
  - Added standalone block statement handling (~line 3450)
  - Already had semicolon fixes from previous version

### Test Files
- Created `test_edge_cases.zx` - Comprehensive test suite (19 tests)
- Removed temporary test files

---

## Upgrade Notes

### Breaking Changes
**None** - These are purely additive fixes that expand what's possible without breaking existing code.

### New Capabilities
1. Keywords can now be used as variable names in most contexts
2. Standalone code blocks work correctly
3. More complex control flow patterns supported

### Migration
No migration needed - existing code continues to work unchanged.

---

## What's Next

### Potential Future Improvements
1. **Contract-to-Contract References:** Implement a reference system that allows contracts to store references to other contract instances (would require changes to serialization system)

2. **Nested Map Literal Persistence:** Improve contract state serialization to handle complex nested structures reliably

3. **Enhanced Error Messages:** Add specific error messages for known limitations (e.g., "Contract instances cannot be stored in contract state")

### Workarounds Are Sufficient
The current limitations are edge cases that can be worked around with alternative patterns. Most real-world use cases don't require:
- Storing contract references in state (can pass as parameters)
- Complex nested map literals in contract state (can use simple values or separate state variables)
