# Zexus Interpreter - Known Issues & Runtime Errors

## Overview
This document tracks runtime errors, bugs, and issues encountered during development and testing of the Zexus interpreter. Issues are categorized by severity and component.

---

## 🔴 CRITICAL ISSUES

### Issue #1: Undefined Interface Identifier
**Severity:** HIGH  
**Component:** Evaluator - Complexity System  
**Status:** FIXED (2025-12-09)  
**Found In:** test_complexity_features.zx, test_security_features.zx  

**Error:**
```
❌ Runtime Error: Identifier 'interface' not found
[DEBUG] Identifier not found: interface; env_keys=[]
```

**Description:**
When using `interface Shape { ... };` statements in print strings or referencing interface definitions later, the evaluator doesn't properly store the interface identifier in the environment. The statement parses correctly, but the AST node isn't being evaluated to register the interface in the environment.

**Root Cause:**
The `eval_interface_statement` handler in `statements.py` needs to properly register the interface and return it so it's accessible as an identifier.

**Location:**
- File: `src/zexus/evaluator/statements.py` (lines ~1551+)
- Method: `eval_interface_statement`

**Fix Strategy:**
1. Update handler to store interface in environment after creation
2. Ensure the handler returns the interface object
3. Test with: `python3 zx-run src/tests/test_verification_simple.zx`

**Resolution:**
- Implemented: the lexer now recognizes the `interface` keyword and emits the `INTERFACE` token so the structural/context parsers create a proper `InterfaceStatement` node. The evaluator's `eval_interface_statement` registers the interface and stores it in the environment. 
- Verified: reproduced the scenario locally; interface identifier is now present in the environment after evaluation.
- Commit: `2e395f8` ("Fix: recognize 'interface' keyword in lexer (resolves Undefined Interface Identifier) and remove debug prints")

---

### Issue #2: Undefined Capability Identifier
**Severity:** HIGH  
**Component:** Evaluator - Security System  
**Status:** FIXED (2025-12-10)  
**Found In:** test_security_features.zx  

**Error:**
```
❌ Runtime Error: Identifier 'capability' not found
[DEBUG] Identifier not found: capability; env_keys=[]
```

**Description:**
When defining `capability admin_access;` statements and then trying to reference them (even in string context), the identifier is not found. The capability statement parses but doesn't register the capability name in the environment.

**Root Cause:**
The `eval_capability_statement` handler in `statements.py` creates the capability but was not storing it properly as an identifier in the environment.

**Location:**
- File: `src/zexus/evaluator/statements.py` (lines ~1320-1353)
- Method: `eval_capability_statement`

**Fix Strategy:**
1. Store capability in environment as identifier using env.set(cap_name, cap)
2. Add capability keyword to both lexers (interpreter and compiler)
3. Ensure handler returns the capability object for proper referencing

**Resolution:**
- Added "capability", "grant", "revoke" keywords to lexer keyword mappings in both src/zexus/lexer.py and src/zexus/compiler/lexer.py
- Modified eval_capability_statement to call env.set(cap_name, cap) so the capability object is stored as an identifier
- Modified return value to return the cap object instead of a message string
- Verified: capability identifiers are now accessible after definition and can be referenced in expressions
- Commit: TBD

---

### Issue #3: Print Statement Concatenation with Undefined Variables
**Severity:** MEDIUM  
**Component:** Evaluator - Expression Evaluation  
**Status:** OPEN  
**Found In:** test_complexity_features.zx (line 53)  

**Error:**
```
"Module function result: 5 + 3 =  + result"
```

**Description:**
String concatenation fails silently when variables in expressions are undefined. The expression evaluator doesn't properly handle `+` operator when one operand evaluates to an error or undefined value. Instead of throwing an error, it produces malformed output.

**Root Cause:**
The binary expression evaluator for `+` operator doesn't validate operand types or properly handle error cases. Missing operands are silently skipped rather than reported.

**Location:**
- File: `src/zexus/evaluator/expressions.py`
- Method: `eval_infix_expression` (likely for PLUS operator)

**Fix Strategy:**
1. Add validation in concatenation operator to check for errors
2. Return proper error instead of malformed output
3. Update error handling in expression evaluation chain

---

## 🟡 HIGH PRIORITY ISSUES

### Issue #4: Module Member Access Not Working
**Severity:** HIGH  
**Component:** Evaluator - Complexity System  
**Status:** FIXED (2025-12-10)  
**Found In:** test_complexity_features.zx (lines 44-48)  

**Code:**
```zexus
module math_operations {
    function add(a, b) { return a + b; };
};
let result = math_operations.add(5, 3);
```

**Problem:**
Module definitions parse but function definitions inside modules are not being stored as module members. The module is created, module keyword is recognized, but the member collection fails.

**Root Cause:**
The module body is being parsed as an empty MapLiteral instead of properly parsing the function statements inside. The structural/traditional parser treats the braces as a map literal rather than executing statements within the module block.

**Location:**
- File: `src/zexus/evaluator/statements.py` - `eval_module_statement` (lines ~1584-1620)
- File: `src/zexus/evaluator/core.py` - PropertyAccessExpression handler (lines ~369+)
- File: `src/zexus/evaluator/functions.py` - eval_method_call_expression (lines ~151+)

**Resolution:**
1. ✅ Parser adjusted so `module { ... }` bodies are parsed as `BlockStatement` (not `MapLiteral`).
2. ✅ `eval_module_statement` now evaluates the module body in a nested `Environment` and registers declared members as `ModuleMember` objects, respecting modifiers (`public`/`private`/`protected`) when present.
3. ✅ `eval_method_call_expression` routes module method calls to the stored member value and calls functions via `apply_function`.
4. ✅ Verified via `debug_module.py` that `module math_operations { function add(a,b) { return a + b; } }` results in module `math_operations` containing `add` as a function member and that `math_operations.add(5,3)` is callable.

**Notes:**
- All members without explicit modifiers are currently marked `public` by default. Modifier-aware visibility is supported during registration.
- Commit: Work staged in local commits; see recent changes in the working tree.

---

### Issue #5: Type Alias Not Resolving in Type Annotations
**Severity:** HIGH  
**Component:** Parser/Evaluator - Complexity System  
**Status:** OPEN  
**Found In:** test_complexity_features.zx (lines 74-75)  

**Code:**
```zexus
type_alias UserId = integer;
let user_id: UserId = 42;
```

**Problem:**
Type aliases parse, but type annotations using the alias (`:UserId`) don't resolve to the actual type. The parser accepts the syntax but the evaluator doesn't recognize `UserId` as a type.

**Root Cause:**
The evaluator doesn't have a phase to resolve type aliases during variable binding. Type information is stored in the TypeAliasStatement but not integrated with the variable assignment process.

**Location:**
- File: `src/zexus/evaluator/statements.py`
- Methods: `eval_type_alias_statement`, `eval_let_statement`
- Also: `src/zexus/complexity_system.py`

**Fix Strategy:**
1. Add type alias resolution to variable binding
2. Store type alias mappings in a registry accessible during assignment
3. Implement type checking using resolved aliases

---

## 🟠 MEDIUM PRIORITY ISSUES

### Issue #6: Using Statement Resource Cleanup Not Triggered
**Severity:** MEDIUM  
**Component:** Evaluator - Complexity System (RAII)  
**Status:** OPEN  
**Found In:** test_complexity_features.zx (lines 58-62)  

**Code:**
```zexus
using(file = "test.txt") {
    print "Inside using block - file: test.txt";
};
```

**Problem:**
The `using` statement executes the body but cleanup methods (`close()` or `cleanup()`) are not being called on the resource after the block completes.

**Root Cause:**
The `eval_using_statement` handler has a try-finally block that attempts cleanup, but:
1. String resources don't have `close()` or `cleanup()` methods
2. The cleanup logic may not be executing properly
3. No validation that cleanup actually occurred

**Location:**
- File: `src/zexus/evaluator/statements.py`
- Method: `eval_using_statement` (lines ~1610+)

**Fix Strategy:**
1. Implement cleanup protocol for built-in types (String, File-like objects)
2. Add debug logging to verify cleanup execution
3. Create File-like wrapper objects that support cleanup
4. Test with explicit resource tracking

---

### Issue #7: Package Hierarchies Not Properly Nested
**Severity:** MEDIUM  
**Component:** Evaluator - Complexity System  
**Status:** OPEN  
**Found In:** test_complexity_features.zx (lines 154-167)  

**Code:**
```zexus
package app.api.v1.endpoints {
    module users { ... };
    module posts { ... };
};
```

**Problem:**
Packages with dotted names parse, but the nesting structure isn't created properly. Accessing `app.api.v1.endpoints.users` would fail.

**Root Cause:**
The `eval_package_statement` handler treats the dotted name as a single identifier rather than creating a hierarchical structure. Package nesting isn't implemented.

**Location:**
- File: `src/zexus/evaluator/statements.py`
- Method: `eval_package_statement`
- Also: `src/zexus/complexity_system.py` - Package class

**Fix Strategy:**
1. Parse dotted package names into hierarchy
2. Create nested Package objects
3. Implement hierarchical lookup in environment
4. Test nested access patterns

---

## 🔵 LOW PRIORITY ISSUES

### Issue #8: Debug Output Too Verbose
**Severity:** LOW  
**Component:** Parser - All  
**Status:** OPEN  

**Problem:**
When running test files, the output includes excessive debug logging from structural analyzer and parser, making actual test output hard to read.

**Example Output:**
```
[STRUCT_BLOCK] id=0 type=statement subtype=PRINT ...
🔍 [Generic] Parsing generic block with tokens: [...]
✅ Parsed: PrintStatement at line 5
```

**Root Cause:**
Debug logging is enabled by default in configuration. This is useful during development but clutters test output.

**Location:**
- File: `src/zexus/config.py` or similar config
- File: `src/zexus/parser/strategy_structural.py` (debug print statements)

**Fix Strategy:**
1. Add configuration option to disable debug output
2. Or use proper logging levels instead of print statements
3. Make debug output optional via command-line flag
4. Test with: `python3 zx-run --quiet src/tests/test_verification_simple.zx`

---

## 📋 ISSUE TRACKING TEMPLATE

When adding new issues, use this format:

```markdown
### Issue #X: [Brief Title]
**Severity:** [CRITICAL/HIGH/MEDIUM/LOW]
**Component:** [Component Name]
**Status:** [OPEN/IN_PROGRESS/RESOLVED]
**Found In:** [File or Test]

**Error:**
[Exact error message]

**Description:**
[What's broken and why it matters]

**Root Cause:**
[Technical explanation]

**Location:**
- File: [path]
- Method: [method name]

**Fix Strategy:**
1. [Step 1]
2. [Step 2]
```

---

## 📊 SUMMARY

| Severity | Count | Status |
|----------|-------|--------|
| 🔴 CRITICAL | 3 | Open |
| 🟡 HIGH | 2 | Open |
| 🟠 MEDIUM | 2 | Open |
| 🔵 LOW | 1 | Open |
| **TOTAL** | **8** | **8 Open** |

---

## 🔧 Resolution Progress

- [ ] Fix undefined identifier issues (interface, capability)
- [ ] Fix string concatenation with missing variables
- [ ] Implement module member access
- [ ] Implement type alias resolution
- [ ] Implement resource cleanup (RAII)
- [ ] Implement package hierarchies
- [ ] Reduce debug output verbosity

---

## 📝 Notes

- Issues are tracked in order of discovery
- Severity is based on impact to functionality
- Status is updated as fixes are implemented
- Each issue includes specific locations and fix strategies
- Tests should be updated as issues are resolved

---

## New Issues Discovered (after adding Concurrency features)

### Issue #9: Evaluator builtins not exported for test injection
**Severity:** MEDIUM
**Component:** Evaluator package
**Status:** RESOLVED (patched)

**Error / Symptom:** Tests attempted `from zexus.evaluator import evaluate, builtins as evaluator_builtins` and failed with ImportError because `builtins` was not exported from the evaluator package.

**Root Cause:** The evaluator package only exported `Evaluator` and `evaluate` previously; tests expect a module-level `builtins` dict to be available for test injection.

**Fix Strategy Implemented:** Added a module-level `builtins = {}` to `src/zexus/evaluator/__init__.py` and updated `evaluate()` in `src/zexus/evaluator/core.py` to merge any injected entries into each `Evaluator` instance at runtime.

**Files Changed:**
- `src/zexus/evaluator/__init__.py` (export `builtins`)
- `src/zexus/evaluator/core.py` (merge module-level builtins into evaluator instance)

### Issue #10: Compiler pipeline errors after parser/context changes
**Severity:** HIGH
**Component:** Compiler / Parser integration
**Status:** OPEN

**Errors observed (from `tests/test_integration.py` run):**
- `Compiler errors: ["Line 3: Unexpected token ')'", "Line 3: Expected ')', got '{'", 'Line 4: Object key must be string or identifier', "Line 5: Unexpected token '}'"]`
- `Compiler errors: ['Line 3: Object key must be string or identifier', "Line 5: Unexpected token '}'"]`
- `Semantic analyzer internal error: maximum recursion depth exceeded`

**Description:** After the recent parser/context parser updates (including concurrency handlers), compiler tests began failing in the compilation phase. The structural/context parsing prints show the interpreter side parsed statements fine, but the compiler's parser/semantic analyzer reports syntax/semantic errors. This likely indicates the compiler pipeline's parser or downstream semantic analysis expects slightly different token shapes or relies on behaviors changed by context parsing.

**Root Cause (suspected):** Changes to parsing strategies (context parser and structural analyzer) modified how tokens are grouped or how certain constructs (e.g., function/action declarations, async/await, call expressions) are represented. The compiler's front-end parser/semantic analyzer appears more strict and fails on sequences that the interpreter's contextual parser accepts.

**Immediate Mitigation / Next Steps:**
1. Re-run the compiler with debug/sanitized token output for failing test case to compare interpreter vs compiler parse trees.
2. Add unit tests comparing parse outputs for simple action/function declarations (including `action async test_async() { ... }`) to find the divergence.
3. Inspect the compiler's parser and semantic analyzer for assumptions about token ordering (e.g., expecting specific parentheses/brace placements) and update either the interpreter parser to preserve older shapes or the compiler to accept the tolerant forms.
4. Reduce debug output while testing by toggling config (to make logs easier to read).

**Files to Inspect:** `src/zexus/parser/parser.py`, `src/zexus/parser/strategy_context.py`, `src/zexus/compiler/*` (parser and semantic analyzer files)

---

I'll continue investigating Issue #10 next (compare parse trees and trace where the compiler disagrees).
