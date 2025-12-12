# Zexus Test Suite Issues Report

**Date:** December 12, 2025
**Scope:** All test files in `src/tests/`
**Total Tests Run:** 8 test files

---

## Summary

| Test File | Status | Issue Count |
|-----------|--------|-------------|
| test_phase1_modifiers.zx | ✅ PASSED | 0 |
| test_phase2_plugins.zx | ✅ PASSED | 0 |
| test_phase3_security.zx | ✅ PASSED | 0 |
| test_phase4_vfs.zx | ✅ PASSED | 0 |
| test_phase5_types.zx | ✅ PASSED | 0 |
| test_phase6_metaprogramming.zx | ❌ FAILED | 1 |
| test_phase7_optimization.zx | ❌ FAILED | 1 |
| test_phase9_advanced_types.zx | ❌ FAILED | 1 |
| test_phase10_ecosystem.zx | ✅ PASSED | 0 |
| test_all_phases.zx | ❌ FAILED | 1 |
| test_complexity_features.zx | ❌ FAILED | 2 |

**Pass Rate:** 5/11 (45%)
**Total Issues:** 6 unique issues (1 Fixed, 1 Improved, 4 Remaining)

---

## Detailed Issues

### Issue 1: Variable Scope in Security Contexts
**File:** `test_phase3_security.zx`
**Status:** ✅ FIXED
**Resolution:** Fixed variable scoping in restricted contexts. `permissions` map is now correctly accessible.

---

### Issue 2: Function Return Not Callable (Anonymous Functions)
**File:** `test_phase6_metaprogramming.zx`
**Status:** ❌ CRITICAL
**Error:** `Not a function: <zexus.object.Null object>`

**Details:**
- Generated getter function returns Null object instead of function.
- The returned function is an anonymous function (`function() { ... }`), which parses as `ActionLiteral`.
- `eval_return_statement` likely handles `FunctionLiteral` but misses `ActionLiteral`.

**Context:**
```zexus
function generateGetter(propertyName) {
    return function() {  // Returns ActionLiteral
        return value_of_ + propertyName;
    };
}
let getter = generateGetter("username");
let value = getter();  // ❌ Error: Not a function
```

**Impact:** Code generation and metaprogramming patterns broken.

---

### Issue 3: Type Coercion in Arithmetic
**File:** `test_phase7_optimization.zx`
**Status:** ❌ HIGH
**Error:** `Type mismatch: STRING * INTEGER`

**Details:**
- Comparison (`>`) fixed, but arithmetic (`*`) still fails.
- String value being multiplied by integer.
- Expectation: `"10" * 2` should probably be `20` (if auto-convert) or `"1010"` (if python-like). Zexus seems to want numeric operation here based on context.

**Context:**
```
❌ Runtime Error: Type mismatch: STRING * INTEGER
```

**Impact:** Optimization features fail on type-sensitive operations.

---

### Issue 4: Advanced Types Runtime Error
**File:** `test_phase9_advanced_types.zx`
**Status:** ⚠️ IMPROVED (Hang Fixed)
**Error:** `Not a function: <zexus.object.Null object>`

**Details:**
- Infinite loop in map parsing is **FIXED**.
- Now failing with runtime error when accessing map methods.
- `container["get"]()` fails because `container["get"]` evaluates to Null.
- Likely related to Issue 2 (Anonymous functions in Maps/Returns).

**Impact:** Advanced type features unusable.

---

### Issue 5: Variable Scope in Nested Blocks
**File:** `test_all_phases.zx`
**Status:** ❌ HIGH
**Error:** `Identifier 'r1' not found`

**Details:**
- Variable `r1` created in function but not accessible.
- Occurs in Phase 1 inline function test.
- Environment keys show parent scope variables but not current scope.

**Stack Trace:**
```
[DEBUG] Identifier not found: r1;
env_keys=['totalTests', 'passedTests', 'failedTests', 'runTest', 'add', 'multiply']
```

**Impact:** Comprehensive integration tests fail early.

---

### Issue 6: Variable Scope in Modules/Packages
**File:** `test_complexity_features.zx`
**Status:** ❌ HIGH
**Error:** `Identifier 'connection' not found` / `Identifier 'email' not found`

**Details:**
- **Using Statement (RAII) is FIXED** (Test 6 passed).
- **Module/Package Scope is BROKEN**.
- Variables defined inside `module` or `package` blocks are not accessible as expected.

**Context:**
```zexus
package database {
    module connection { ... }
}
// Accessing database.connection fails or internal variables fail
```

**Impact:** Modular programming features broken.

---

### Issue 7: Type Alias Re-registration
**File:** `test_complexity_features.zx`
**Status:** ❌ MEDIUM
**Error:** `Type alias 'UserId' already registered`

**Details:**
- Type alias defined in multiple tests in same file.
- Registry doesn't allow re-registration.

**Stack Trace:**
```python
ValueError: Type alias 'UserId' already registered
```

**Impact:** Multiple tests with same type aliases fail.

---

## Recommendations

### Immediate Actions
1. **Fix ActionLiteral Return** (Issue 2 & 4): Ensure `eval_return_statement` wraps `ActionLiteral` (anonymous functions) into `Function` objects. This should fix `test_phase6` and `test_phase9`.
2. **Fix Arithmetic Coercion** (Issue 3): Extend `eval_infix_expression` to handle `STRING * INTEGER` (and others).
3. **Fix Module Scope** (Issue 6): Investigate `eval_module_statement` and `eval_package_statement` environment handling.

### Short-term Actions
1. **Fix Nested Scope** (Issue 5): Debug `eval_block_statement` environment chaining.
2. **Fix Registry** (Issue 7): Add `clear()` to registry or allow overwrites.

---

## Test Success Details

### ✅ test_phase1_modifiers.zx
All modifier tests passed.

### ✅ test_phase2_plugins.zx
All plugin system tests passed.

### ✅ test_phase3_security.zx
All capability security tests passed.

### ✅ test_phase4_vfs.zx
All VFS tests passed.

### ✅ test_phase5_types.zx
Type system tests passed.

### ✅ test_phase10_ecosystem.zx
All ecosystem tests passed.
