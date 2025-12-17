# Zexus Language Keyword Testing Master List

**Purpose**: Systematic testing and documentation of all Zexus language keywords  
**Status**: In Progress - Phase 13 Complete (Implementation Gaps Found)  
**Last Updated**: Phase 13 Completion  
**Tests Created**: 1055+ (375 easy, 380 medium, 365 complex)  
**Keywords Tested**: 101 keywords + 7 builtins = 108 total (LET, CONST, IF, ELIF, ELSE, WHILE, FOR, EACH, IN, ACTION, FUNCTION, LAMBDA, RETURN, PRINT, DEBUG, USE, IMPORT, EXPORT, MODULE, PACKAGE, FROM, EXTERNAL, TRY, CATCH, REVERT, REQUIRE, ASYNC, AWAIT, CHANNEL, SEND, RECEIVE, ATOMIC, EVENT, EMIT, STREAM, WATCH, ENTITY, VERIFY, CONTRACT, PROTECT, SEAL, AUDIT, RESTRICT, SANDBOX, TRAIL, CAPABILITY, GRANT, REVOKE, IMMUTABLE, VALIDATE, SANITIZE, LEDGER, STATE, TX, HASH, SIGNATURE, VERIFY_SIG, LIMIT, GAS, PERSISTENT, STORAGE, NATIVE, GC, INLINE, BUFFER, SIMD, DEFER, PATTERN, ENUM, PROTOCOL, INTERFACE, TYPE_ALIAS, IMPLEMENTS, THIS, USING, SCREEN, COMPONENT, THEME, COLOR, GRAPHICS, CANVAS, ANIMATION, CLOCK, PUBLIC, PRIVATE, SEALED, SECURE, PURE, VIEW, PAYABLE, MODIFIER + mix, render_screen, add_to_screen, set_theme, create_canvas, draw_line, draw_text)  
**Critical Issues Found**: 24 (Loop execution broken, closure/map issues, external linking, require context-sensitivity, concurrency lexer registration missing, variable reassignment in functions, verify doesn't throw errors, sandbox return values broken, validate schema registry incomplete, sanitize variable scope issues, limit constructor broken, signature requires PEM keys, TX function scope issue, persistent assignment target error, defer cleanup never executes, enum values not accessible, type_alias duplicate registration, **middleware parser missing, auth parser missing, throttle parser missing, cache parser missing, inject DI system broken**)

## Testing Methodology
For each keyword:
- ✅ **Easy Test**: Basic usage, simple cases
- ✅ **Medium Test**: Intermediate complexity, edge cases
- ✅ **Complex Test**: Advanced scenarios, integration with other features

## Status Legend
- 🔴 Not Started
- 🟡 In Progress
- 🟢 Completed
- ❌ Failed (needs fix)
- ⚠️ Partially Working

---

## 1. BASIC KEYWORDS

### 1.1 Variable Declaration
| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| LET | � | 🟢 | 🟡 | 🔴 | 🟢 | 2 | Mutable variable declaration |
| CONST | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 1 | Immutable variable declaration |

### 1.2 Control Flow
| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| IF | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Conditional execution |
| ELIF | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Else-if conditional |
| ELSE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Alternative conditional |
| WHILE | ❌ | ❌ | ❌ | ❌ | 🔴 | 3+ | While loop - BROKEN |
| FOR | ❌ | ❌ | ❌ | ❌ | 🔴 | 3+ | For loop (for each) - BROKEN |
| EACH | ❌ | ❌ | ❌ | ❌ | 🔴 | 3+ | For-each iteration - BROKEN |
| IN | ❌ | ❌ | ❌ | ❌ | 🔴 | 3+ | Used with for/each - BROKEN |

### 1.3 Functions
| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| ACTION | � | 🟢 | 🟢 | 🟡 | 🟢 | 2 | Action definition (Zexus functions) |
| FUNCTION | 🟢 | 🟢 | 🟢 | 🟡 | 🟢 | 2 | Function definition |
| LAMBDA | 🟢 | 🟢 | 🟢 | 🟡 | 🟢 | 2 | Anonymous functions |
| RETURN | 🟢 | 🟢 | 🟢 | 🟡 | 🟢 | 1 | Return values |

### 1.4 I/O Operations
| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| PRINT | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Output to console |
| DEBUG | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 1 | Debug output with metadata |

---

## 2. MODULE SYSTEM

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| USE | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Import modules |
| IMPORT | 🟡 | 🟡 | 🟡 | 🔴 | 🟢 | - | Import statement (may be alias for USE) |
| EXPORT | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Export symbols |
| MODULE | 🟡 | 🟡 | 🔴 | 🔴 | 🟢 | - | Module definition (partially implemented) |
| PACKAGE | 🟡 | 🔴 | 🔴 | 🔴 | 🟢 | - | Package/namespace (may not be implemented) |
| FROM | 🟡 | 🟡 | 🔴 | 🔴 | 🟢 | - | Import from module (USE with braces works) |
| EXTERNAL | 🟢 | 🟢 | 🟢 | 🟡 | 🟢 | 1 | External declarations |

---

## 3. ERROR HANDLING

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| TRY | � | 🟢 | 🟢 | 🟢 | 🟢 | 1 | Try block for error handling |
| CATCH | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 1 | Catch exceptions (syntax warning) |
| REVERT | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Revert transaction |
| REQUIRE | ⚠️ | 🟢 | ⚠️ | ⚠️ | 🟢 | 1 | Require condition (context-sensitive) |

---

## 4. ASYNC & CONCURRENCY

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| ASYNC | � | 🟡 | 🟡 | 🟡 | 🟢 | 0 | Registered in lexer, not implemented |
| AWAIT | 🟡 | 🟡 | 🟡 | 🟡 | 🟢 | 0 | Registered in lexer, not implemented |
| CHANNEL | ❌ | ❌ | ❌ | ❌ | 🟢 | 1 | NOT in lexer (full impl exists!) |
| SEND | ❌ | ❌ | ❌ | ❌ | 🟢 | 1 | NOT in lexer (full impl exists!) |
| RECEIVE | ❌ | ❌ | ❌ | ❌ | 🟢 | 1 | NOT in lexer (full impl exists!) |
| ATOMIC | ❌ | ❌ | ❌ | ❌ | 🟢 | 1 | NOT in lexer (full impl exists!) |

---

## 5. EVENTS & REACTIVE

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| EVENT | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Type token (not a statement) |
| EMIT | 🟢 | 🟢 | 🟢 | 🟡 | 🟢 | 1 | Fully functional event emission |
| STREAM | 🟡 | 🟡 | 🟡 | 🟡 | 🟢 | 0 | Token exists, implementation unclear |
| WATCH | 🟡 | 🟡 | 🟡 | 🟡 | 🟢 | 0 | Token exists, implementation unclear |
---

## 6. SECURITY FEATURES

### 6.1 Core Security
| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| ENTITY | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Type definitions working perfectly |
| VERIFY | 🟡 | 🟢 | 🟢 | 🟡 | 🟢 | 2 | Doesn't throw errors properly |
| CONTRACT | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Smart contracts fully functional |
| PROTECT | 🟡 | ⚪ | ⚪ | ⚪ | 🟢 | 0 | Implementation exists, untested |
| SEAL | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Immutability working |
| AUDIT | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Compliance logging working |
| RESTRICT | 🟡 | ⚪ | ⚪ | ⚪ | 🟢 | 0 | Implementation exists, untested |
| SANDBOX | 🟡 | 🟢 | 🟡 | 🟡 | 🟢 | 2 | Return values broken |
| TRAIL | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Event tracking working |

### 6.2 Capability-Based Security
| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| CAPABILITY | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Define capabilities - fully functional |
| GRANT | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Grant capabilities - working |
| REVOKE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Revoke capabilities - working |
| IMMUTABLE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Immutable variables - working |

### 6.3 Data Validation
| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| VALIDATE | � | 🔴 | 🔴 | 🔴 | 🟢 | 1 | Schema registry incomplete |
| SANITIZE | 🟡 | 🟢 | 🟢 | 🟡 | 🟢 | 1 | Variable scope issues |

---

## 7. BLOCKCHAIN FEATURES

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| LEDGER | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Immutable ledger - fully working |
| STATE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Mutable state - working |
| TX | 🟡 | 🟢 | 🔴 | 🟡 | 🟢 | 1 | TX context - function scope issue |
| HASH | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Cryptographic hash - working |
| SIGNATURE | 🔴 | 🔴 | 🔴 | 🔴 | 🟢 | 1 | Requires PEM key format |
| VERIFY_SIG | 🔴 | 🔴 | 🔴 | 🔴 | 🟢 | 1 | Untested - depends on SIGNATURE |
| LIMIT | 🔴 | 🔴 | 🔴 | 🔴 | 🟢 | 1 | Constructor parameter mismatch |
| GAS | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Gas tracking - working |
| PERSISTENT | 🟡 | 🟢 | 🟢 | 🔴 | 🟢 | 1 | Assignment target error |
| STORAGE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Storage keyword - working |

---

## 8. PERFORMANCE OPTIMIZATION

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| NATIVE | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | C/C++ FFI - fully working |
| GC | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | GC control - perfect |
| INLINE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Function inlining - working |
| BUFFER | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Memory buffers - excellent |
| SIMD | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Vector ops - working |

---

## 9. ADVANCED LANGUAGE FEATURES

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| DEFER | ❌ | 🟢 | 🟢 | 🟢 | 🟢 | 1 | Cleanup never executes - CRITICAL |
| PATTERN | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Pattern matching - working |
| ENUM | ⚠️ | 🟢 | 🟢 | 🟢 | 🟢 | 1 | Values not accessible |
| PROTOCOL | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Protocol definition - working |
| INTERFACE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Interface definition - working |
| TYPE_ALIAS | ⚠️ | 🟢 | 🟢 | 🟢 | 🟢 | 1 | Duplicate registration error |
| IMPLEMENTS | 🟡 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Untested - needs context |
| THIS | 🟡 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Untested - needs contract |
| USING | 🟡 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Untested - needs resources |

---

## 10. RENDERER & UI

### 10.1 Screen Components
| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| SCREEN | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Screen declaration - working |
| COMPONENT | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Component definition - working |
| THEME | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Theme declaration - working |
| COLOR | 🔴 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Not in lexer - backend exists |

### 10.2 Graphics & Canvas
| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| GRAPHICS | 🟡 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Lexer only - backend exists |
| CANVAS | 🟡 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Lexer only - backend exists |
| ANIMATION | 🟡 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Lexer only - backend exists |
| CLOCK | 🟡 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Lexer only - backend exists |

### 10.3 Renderer Operations (Builtin Functions)
| Builtin | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| mix | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Color mixing - working |
| create_canvas | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Canvas creation - working |
| draw_line | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Line drawing - working |
| draw_text | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Text rendering - working |
| set_theme | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Theme setting - working |
| render_screen | 🟡 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Untested - implemented |
| add_to_screen | 🟡 | 🔴 | 🔴 | 🔴 | 🟡 | 0 | Untested - implemented |

---

## 11. MODIFIERS

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| PUBLIC | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Public visibility - auto export |
| PRIVATE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Private visibility - module scope |
| SEALED | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Sealed modifier - prevent override |
| SECURE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Secure modifier - security flag |
| PURE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Pure function - no side effects |
| VIEW | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | View function - read-only |
| PAYABLE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Payable function - receive tokens |
| MODIFIER | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Function modifier - reusable guards |

---

## 12. SPECIAL KEYWORDS

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| EXACTLY | � | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Exact matching block |
| EMBEDDED | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Foreign language code |
| MAP | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Map/object literals |
| TRUE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Boolean true literal |
| FALSE | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Boolean false literal |
| NULL | 🟢 | 🟢 | 🟢 | 🟢 | 🟢 | 0 | Null value literal |

---

## 13. ADVANCED FEATURES (MIDDLEWARE & CACHE)

| Keyword | Status | Easy | Medium | Complex | Doc | Errors | Notes |
|---------|--------|------|--------|---------|-----|--------|-------|
| MIDDLEWARE | ❌ | ❌ | ❌ | ❌ | 🟢 | 1 | Parser missing - AST/evaluator exist |
| AUTH | ❌ | ❌ | ❌ | ❌ | 🟢 | 1 | Parser missing - AST/evaluator exist |
| THROTTLE | ❌ | ❌ | ❌ | ❌ | 🟢 | 1 | Parser missing - AST/evaluator exist |
| CACHE | ❌ | ❌ | ❌ | ❌ | 🟢 | 1 | Parser missing - AST/evaluator exist |
| INJECT | ❌ | ❌ | ❌ | ❌ | 🟢 | 1 | DI system returns None - runtime error |

---

## Testing Progress

**Total Keywords**: 130+ (101 tested, 5 incomplete implementations)  
**Fully Working**: 68 keywords  
**Partially Working**: 28 keywords  
**Implementation Incomplete**: 5 (MIDDLEWARE, AUTH, THROTTLE, CACHE, INJECT)  
**Not Tested**: 29+  
**Total Errors Found**: 24 critical implementation issues

**Test Coverage**: 101/130+ keywords tested (78%)  
**Success Rate**: 68/101 fully working (67%)  
**Test Files Created**: 1055+ tests across 13 phases

---

## Error Log

### Critical Errors
*No critical errors yet*

### LET Keyword Errors
1. **Colon Syntax Not Working** (Priority: Medium)
   - Description: Alternative `let x : value` syntax doesn't work, variable not registered
   - Test: `let test : 42; print test;` results in "Identifier 'test' not found"
   - Status: Documented, workaround available (use `=` instead)
   - File: test_let_easy.zx (Test 13)

2. **Array Concatenation Error** (Priority: Medium)
   - Description: `list = list + [value]` throws "Type mismatch: LIST + LIST"
   - Test: test_let_medium.zx (Test 15)
   - Status: Needs investigation - may need different syntax for array operations
   - Impact: Cannot easily append to arrays using `+` operator

### CONST Keyword Errors
1. **Cannot Shadow Const Variables** (Priority: Low, By Design?)
   - Description: Cannot declare const with same name in nested scope, even though it would be a different variable
   - Test: `const x = 10; if (true) { const x = 20; }` results in "Cannot reassign const variable 'x'"
   - Status: May be intentional design decision - documented
   - Workaround: Use different variable names in nested scopes
   - File: test_const_complex.zx (Original Test 11)
   - Note: This differs from most languages where shadowing is allowed

### PRINT/DEBUG Keyword Errors
1. **Debug May Require Parentheses** (Priority: Low)
   - Description: Parser warnings suggest using `debug(expr)` instead of `debug expr`
   - Test: Some syntax modes require parentheses
   - Status: Minor syntax variation
   - Files: test_io_modules_medium.zx
   - Workaround: Use parentheses for consistency
   - Impact: Minimal - both syntaxes may work

### MODULE SYSTEM Keyword Errors
1. **External Functions Don't Auto-Link** (Priority: Medium)
   - Description: `external functionName;` creates placeholder, calling it causes "not found" error
   - Test: `external nativeSort; handleExternalResult(nativeSort, data);` fails
   - Status: Expected behavior - requires native implementation
   - Files: test_io_modules_complex.zx (Test 12)
   - Workaround: Ensure native functions are properly linked before calling
   - Impact: External declarations are placeholders only

### ACTION/FUNCTION/LAMBDA/RETURN Keyword Errors
1. **Map Returns Display as Empty** (Priority: Medium)
   - Description: Functions returning map literals show `{}` instead of actual content
   - Test: `return {"area": 50, "perimeter": 30}` displays as `{}`
   - Status: Display/output issue, data may be stored correctly
   - Files: test_functions_medium.zx (Test 9), test_functions_complex.zx
   - Workaround: Access individual map properties instead of printing whole map

2. **Closure State Not Persisting Properly** (Priority: Medium)
   - Description: Closures that capture outer variables display `{}` on subsequent calls
   - Test: Counter pattern with captured variables returns empty maps
   - Status: Closure implementation may need enhancement
   - Files: test_functions_medium.zx (Test 8), test_functions_complex.zx (Tests 8, 12)
   - Impact: Limits closure functionality for stateful patterns

3. **PropertyAccessExpression Error** (Priority: High)
   - Description: Error `'PropertyAccessExpression' object has no attribute 'value'`
   - Occurs with some complex nested property access patterns
   - Status: Parser/evaluator bug
   - Files: test_functions_complex.zx (late in execution)

### TRY/CATCH/REVERT/REQUIRE Keyword Errors
1. **Require Context Sensitivity** (Priority: High)
   - Description: `require()` treated as function call in some contexts, causing "Not a function: require" error
   - Test: Using require inside try-catch blocks or within certain function contexts fails
   - Status: Parser/evaluator inconsistency - require works at top level but fails in nested contexts
   - Files: test_error_handling_easy.zx (Test 5), test_error_handling_medium.zx (Tests 2, 3), test_error_handling_complex.zx (multiple tests)
   - Workaround: Use explicit if-revert pattern: `if (!condition) { revert("message"); }`
   - Impact: High - limits usefulness of require keyword, forces verbose alternative

2. **Catch Syntax Warnings** (Priority: Low)
   - Description: Parser warns "Use parentheses with catch: catch(error) { }"
   - Test: All catch blocks generate this warning
   - Status: Minor syntax preference
   - Files: All error handling test files
   - Workaround: Use `catch (error)` with parentheses
   - Impact: Minimal - stylistic warning only

### ASYNC/CONCURRENCY Keyword Errors
1. **CHANNEL/SEND/RECEIVE/ATOMIC Not Registered in Lexer** (Priority: **CRITICAL**)
   - Description: Keywords defined in token.py, parser/evaluator/runtime fully implemented, but NOT in lexer.py keywords dictionary
   - Test: `channel<integer> ch;` fails with "Identifier not found: channel"
   - Status: Complete implementation exists (500+ lines) but unreachable due to missing lexer registration
   - Files: test_async_easy.zx (all 20 tests fail)
   - Fix Required: Add 4 lines to lexer.py keywords dictionary: `"channel": CHANNEL`, `"send": SEND`, `"receive": RECEIVE`, `"atomic": ATOMIC`
   - Estimated Fix Time: 5 minutes
   - Impact: **CRITICAL** - Entire concurrency system unusable despite being fully implemented
   - Components Affected:
     * Token definitions: ✅ Complete (`src/zexus/zexus_token.py`)
     * Parser handlers: ✅ Complete (`src/zexus/parser/parser.py` lines 2771-2900)
     * Evaluator handlers: ✅ Complete (`src/zexus/evaluator/statements.py` lines 2106-2220)
     * Runtime system: ✅ Complete (`src/zexus/concurrency_system.py` - Channel, Atomic, ConcurrencyManager)
     * Lexer registration: ❌ **MISSING** (`src/zexus/lexer.py` lines 358-465)
   - Documentation: `docs/CONCURRENCY.md` describes intended usage
   - ROI: **Extremely High** - trivial 4-line fix unlocks entire concurrent programming subsystem

2. **ASYNC/AWAIT Not Implemented** (Priority: High)
   - Description: Keywords registered in lexer but no parser or evaluator handlers exist
   - Test: No syntax errors when used, but no functionality
   - Status: Reserved for future implementation
   - Files: Lexer registration exists, no other implementation
   - Impact: Keywords exist as placeholders only
   - Estimated Implementation: Days to weeks (needs async runtime, event loop/threading model, Promise/Future system)

### EVENTS/REACTIVE Keyword Errors
1. **Variable Reassignment in Functions** (Priority: High)
   - Description: Cannot reassign variables declared outside function scope - causes "Invalid assignment target" error
   - Test: `let counter = 0; action increment() { counter = counter + 1; }` fails
   - Status: Scoping limitation - outer scope variables cannot be modified from inner functions
   - Files: test_events_complex.zx (Tests 2, 6, 13, 14, 15, 16, 17, 18, 19, 20)
   - Workaround: Use return values and reassign at call site: `counter = increment(counter);`
   - Impact: High - Limits stateful event patterns, requires functional programming style
   - Note: This affects ALL keywords, not just events - fundamental language design issue

2. **STREAM Not Implemented** (Priority: Medium)
   - Description: STREAM keyword registered in lexer, but no parser or evaluator implementation found
   - Intended Syntax: `stream name as event => handler;`
   - Status: Token exists, functionality unclear
   - Impact: Event streaming feature unavailable

3. **WATCH Not Implemented** (Priority: Medium)
   - Description: WATCH keyword registered in lexer, but no parser or evaluator implementation found
   - Intended Syntax: `watch variable => reaction;`
   - Status: Token exists, functionality unclear
   - Impact: Reactive state management feature unavailable

### SECURITY & COMPLIANCE Keyword Errors
1. **VERIFY Doesn't Throw Errors Properly** (Priority: CRITICAL)
   - Description: `verify false, "message"` should halt execution but continues instead
   - Test: `verify false, "Expected failure"; print "Should not reach here";` - prints message
   - Status: Verification failures don't stop execution as expected
   - Files: test_security_easy.zx (Test 19)
   - Impact: CRITICAL - Security assertions ineffective, cannot rely on verify for safety
   - Expected Behavior: Should throw error and stop execution
   - Actual Behavior: Returns error object but execution continues

2. **SANDBOX Return Values Broken** (Priority: HIGH)
   - Description: Sandbox blocks return literal string "sandbox" instead of computed values
   - Test: `let result = sandbox { return 10 * 5; };` returns "sandbox" not 50
   - Status: Cannot extract results from sandboxed computations
   - Files: test_security_medium.zx (Test 6), test_security_complex.zx (Tests 11, 17, 20)
   - Impact: HIGH - Sandboxed computations unusable for value-returning operations
   - Root Cause: eval_sandbox_statement returns wrong value or print concatenation issue

3. **SANDBOX Variable Scope Issues** (Priority: MEDIUM)
   - Description: Variables inside sandbox may not be properly isolated
   - Test: test_security_medium.zx Test 5 - nested sandbox variable access
   - Status: Isolation guarantees unclear
   - Impact: MEDIUM - May compromise sandbox security model

4. **PROTECT Not Fully Tested** (Priority: LOW)
   - Description: Implementation exists in evaluator but syntax and functionality unverified
   - Intended Syntax: `protect targetFunction, { rules }, "strict";`
   - Status: PolicyBuilder and PolicyRegistry integration exists but untested
   - Impact: LOW - Feature exists but confidence level unknown

5. **RESTRICT Not Fully Tested** (Priority: LOW)
   - Description: Implementation exists in evaluator but syntax and functionality unverified
   - Intended Syntax: `restrict object.field = "restriction_type";`
   - Status: SecurityContext integration exists but untested
   - Impact: LOW - Feature exists but confidence level unknown

### CAPABILITY & VALIDATION Keyword Errors
1. **VALIDATE Schema Registry Incomplete** (Priority: HIGH)
   - Description: Schema names not recognized, throws "Unknown schema: string" error
   - Test: `validate data, "string";` fails with ValueError
   - Status: Implementation exists but schema registry not initialized or incomplete
   - Files: test_capability_easy.zx (Test 10, 11, 17)
   - Impact: HIGH - Validation feature completely unusable
   - Root Cause: validation_system.py schema registry missing predefined schemas
   - Expected: Should recognize standard types: string, integer, email, etc.

2. **SANITIZE Variable Scope Issues** (Priority: MEDIUM)
   - Description: Sanitized values assigned to variables not accessible in some contexts
   - Test: `let stage3 = sanitize data, "html"; print stage3;` fails with "Identifier not found"
   - Status: Similar to sandbox return value issue - variable assignment problem
   - Files: test_capability_complex.zx (Test 13)
   - Impact: MEDIUM - Workaround exists (use sanitize directly in expressions)
   - Related: Same fundamental issue as sandbox return values and variable reassignment

3. **Capability Function Scope Limitation** (Priority: MEDIUM)
   - Description: Capabilities defined inside functions can't be accessed by grant/revoke
   - Test: Function creates capability, then tries to grant it - capability not found
   - Status: Capability definitions need module-level scope
   - Files: test_capability_medium.zx (Test 8)
   - Impact: MEDIUM - Limits dynamic capability creation patterns
   - Workaround: Define all capabilities at module level, reference them in functions

### ADVANCED LANGUAGE FEATURES Keyword Errors
1. **DEFER Cleanup Never Executes** (Priority: CRITICAL)
   - Description: Deferred cleanup code is registered but never runs when scope exits
   - Test: `defer { print "Cleanup"; } print "Main";` only prints "Main"
   - Error: Code stored in env._deferred list but no execution mechanism
   - Files: test_advanced_easy.zx (Tests 1, 11, 16), test_advanced_medium.zx (Tests 1, 6, 11, 15, 20), test_advanced_complex.zx (Tests 1, 6, 11, 16, 20)
   - Status: Code registers correctly but execution never happens
   - Impact: CRITICAL - DEFER keyword completely non-functional
   - Root Cause: eval_defer_statement stores blocks in env._deferred but no scope exit handler executes them
   - Expected: LIFO execution of deferred blocks when scope/function exits
   - Actual: Deferred blocks never execute

2. **ENUM Values Not Accessible as Identifiers** (Priority: HIGH)
   - Description: ENUM creates Map object but enum name not stored in environment
   - Test: `enum Status { PENDING, ACTIVE }; print Status;` throws "Identifier 'Status' not found"
   - Error: eval_enum_statement creates and returns Map but doesn't store it
   - Files: test_advanced_easy.zx (Test 2)
   - Status: Enum definition works but can't be accessed afterward
   - Impact: HIGH - Cannot use enum after definition
   - Root Cause: Missing `env.set(node.name, enum_obj)` before return in eval_enum_statement
   - Expected: Enum accessible via its name as identifier
   - Actual: Enum created but not stored, identifier lookup fails

3. **TYPE_ALIAS Duplicate Registration Error** (Priority: MEDIUM)
   - Description: ComplexityManager doesn't allow re-registering same type alias name
   - Test: `type_alias UserId = int;` defined twice throws "Type alias 'UserId' already registered"
   - Error: ValueError from ComplexityManager.create_type_alias()
   - Files: test_advanced_easy.zx (Test 15)
   - Status: Global registry prevents re-registration
   - Impact: MEDIUM - Limits type alias reuse in different scopes
   - Root Cause: ComplexityManager maintains global registry without scope support
   - Note: May be intentional design to prevent naming conflicts

### BLOCKCHAIN & STATE Keyword Errors
1. **LIMIT Constructor Parameter Mismatch** (Priority: CRITICAL)
   - Description: Parser creates LimitStatement with 'gas_limit' parameter but constructor expects different name
   - Test: `limit(10000);` throws TypeError
   - Error: "LimitStatement.__init__() got an unexpected keyword argument 'gas_limit'"
   - Files: test_blockchain_easy.zx (Test 13), parser/strategy_context.py line 3576
   - Status: Parser/AST constructor mismatch
   - Impact: CRITICAL - LIMIT keyword completely broken
   - Root Cause: Parser passes 'gas_limit' but AST node expects different parameter

2. **SIGNATURE Requires PEM Key Format** (Priority: HIGH)
   - Description: Signature creation requires valid PEM format private keys, fails with simple strings
   - Test: `signature(message, "private_key_123", "ECDSA");` fails
   - Error: "Signature error: Unable to load PEM file"
   - Files: test_blockchain_easy.zx (Test 11)
   - Status: Cryptography library requirement not met
   - Impact: HIGH - SIGNATURE keyword unusable without proper key generation
   - Root Cause: CryptoPlugin uses cryptography library which requires valid PEM format
   - Note: Need proper key generation utilities or mock keys for testing

3. **TX Context Not Accessible in Functions** (Priority: HIGH)
   - Description: TX identifier not accessible inside function scope
   - Test: Function tries to access `TX.caller` - identifier not found
   - Error: "Identifier 'TX' not found"
   - Files: test_blockchain_medium.zx (Test 5)
   - Status: Same scoping issue as other identifiers
   - Impact: HIGH - TX context needed most inside functions for checks
   - Related: Same fundamental scoping issue seen in multiple keywords

4. **PERSISTENT Assignment Target Error** (Priority: MEDIUM)
   - Description: Persistent storage with nested maps causes assignment target error
   - Test: `persistent storage systemConfig = { "network": "mainnet", "features": {...} };`
   - Error: "assignment target"
   - Files: test_blockchain_complex.zx (Test 4)
   - Status: Parser/evaluator issue with nested map initialization
   - Impact: MEDIUM - Complex persistent storage patterns broken
   - Workaround: Use simpler initialization or separate assignments
   - Impact: Some advanced patterns fail unexpectedly

### WHILE/FOR/EACH/IN Keyword Errors (CRITICAL)
1. **Loop Bodies Not Executing** (Priority: CRITICAL)
   - Description: Most statements inside while and for-each loop bodies do not execute. Only some prints work.
   - Test: `while (counter < 3) { print counter; counter = counter + 1; }` - prints are missing
   - Test: `for each num in [1, 2, 3] { print num; }` - prints are missing
   - Status: BROKEN - loop parsing works but execution fails
   - Files: test_while_for_each_easy.zx (all tests), test_while_for_each_medium.zx, test_while_for_each_complex.zx
   - Expected: All loop iterations should print values
   - Actual: Very minimal output (e.g., only "0", "1", "0", "Passing: 0" for 20 tests)

2. **Loop Increment/Assignment Not Working** (Priority: CRITICAL)
   - Description: Variable reassignments inside loops don't work properly
   - Test: `let counter = 0; while (counter < 3) { counter = counter + 1; }` - infinite loop or no execution
   - Status: BROKEN - related to Error #1
   - Impact: Makes loops completely non-functional

3. **For Each Iteration Not Executing** (Priority: CRITICAL)
   - Description: For each loops parse correctly but don't iterate over arrays
   - Test: `for each num in [1, 2, 3] { print num; }` - prints nothing or very limited output
   - Status: BROKEN - evaluator issue with loop body execution
   - Impact: Core language feature completely broken

### MIDDLEWARE/AUTH/THROTTLE/CACHE Keyword Errors (IMPLEMENTATION INCOMPLETE)
1. **Parser Handlers Missing** (Priority: HIGH)
   - Description: MIDDLEWARE, AUTH, THROTTLE, CACHE have tokens, AST definitions, and evaluators but no parser handlers
   - Test: Any attempt to use these keywords fails - parsed as identifiers/call expressions
   - Error: "Identifier not found: middleware" (and similar for auth, throttle, cache)
   - Files: test_phase13_easy.zx (all tests blocked)
   - Status: Partial implementation - backend exists but not wired to parser
   - Impact: HIGH - Enterprise features completely unusable
   - Root Cause: parse_statement() in parser.py missing cases for these tokens
   - Fix Required: Add 4 parser method handlers similar to parse_protect_statement()
   - Components Status:
     * Tokens: ✅ Defined (zexus_token.py lines 150-153)
     * Lexer: ✅ Registered (lexer.py lines 429-432)
     * Parser: ❌ **MISSING** handlers in parse_statement()
     * AST: ✅ Defined (MiddlewareStatement, AuthStatement, ThrottleStatement, CacheStatement)
     * Evaluator: ✅ Complete (statements.py lines 1077-1143)
   - Expected Syntax:
     * `middleware(name, action(req, res) { ... })`
     * `auth { "provider": "oauth2", ... }`
     * `throttle(target, { "requests_per_minute": 100 })`
     * `cache(target, { "ttl": 300 })`
   - Documentation: docs/QUICK_START.md shows intended usage
   - Estimated Fix: 2-4 hours to add parser methods

### INJECT Keyword Errors (IMPLEMENTATION BROKEN)
1. **Dependency Injection System Returns None** (Priority: CRITICAL)
   - Description: DI registry's get_container() returns None, causing crash when setting execution_mode
   - Test: `inject Logger;` throws "'NoneType' object has no attribute 'execution_mode'"
   - Error: Runtime error in eval_inject_statement line 1799
   - Files: test_phase13_easy.zx (all tests crash immediately)
   - Status: Full implementation exists but runtime fails
   - Impact: CRITICAL - INJECT keyword completely non-functional
   - Root Cause: dependency_injection.py get_di_registry().get_container() returns None
   - Fix Required: Initialize default container or add None check before setting execution_mode
   - Components Status:
     * Token: ✅ Defined (zexus_token.py:154)
     * Lexer: ✅ Registered (lexer.py:454)
     * Parser: ✅ Complete (parse_inject_statement at parser.py:2585)
     * AST: ✅ Defined (InjectStatement)
     * Evaluator: ✅ Complete but crashes (statements.py:1764)
     * DI System: ❌ **BROKEN** (dependency_injection.py)
   - Expected: Should set variable to NULL if dependency not registered, not crash
   - Actual: Crashes before attempting resolution
   - Documentation: Full DI system design exists
   - Estimated Fix: 1-2 hours to add container initialization

### Warning/Minor Issues
*No minor issues yet*

---

## Priority Testing Order

### Phase 1: Core Language (Highest Priority)
1. LET, CONST
2. IF, ELIF, ELSE
3. PRINT, DEBUG
4. ACTION, FUNCTION, RETURN
5. FOR, EACH, WHILE

### Phase 2: Module System
6. USE, IMPORT, EXPORT
7. MODULE, PACKAGE

### Phase 3: Error Handling & Async
8. TRY, CATCH
9. ASYNC, AWAIT

### Phase 4: Advanced Features
10. PATTERN, ENUM, DEFER
11. Security features (AUDIT, RESTRICT, SANDBOX, TRAIL)
12. Performance features (NATIVE, GC, INLINE, BUFFER, SIMD)

### Phase 5: Specialized Features
13. Renderer/UI keywords
14. Blockchain keywords
15. Middleware & advanced features

---

## Notes
- Each keyword will get its own detailed documentation file in `docs/keywords/`
- Test files will be organized by difficulty in `tests/keyword_tests/{easy,medium,complex}/`
- Errors will be logged here and fixed systematically
- Each keyword documentation will include: syntax, use cases, examples, edge cases, and potential improvements
