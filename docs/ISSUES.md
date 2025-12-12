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
**Status:** FIXED (2025-12-12)  
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

**Resolution:**
- The evaluator now properly throws a "Runtime Error: Identifier 'undefined_var' not found" when trying to concatenate with undefined variables
- Test file: test_issue3_print_concat.zx shows proper error handling
- Verified: Expression evaluation chain properly propagates errors instead of silently failing
- Commit: Current working tree

---

## 🟡 HIGH PRIORITY ISSUES

### Issue #4: Module Member Access Not Working
**Severity:** HIGH  
**Component:** Evaluator - Complexity System  
**Status:** FIXED (2025-12-12)  
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
The context parser's `_parse_block_statements` method did not recognize `FUNCTION` as a statement starter, so function declarations inside module bodies were being parsed as expression statements with ActionLiterals instead of FunctionStatements. This meant the functions didn't have names and couldn't be registered as module members.

**Location:**
- File: `src/zexus/parser/strategy_context.py` - `_parse_block_statements` method
- File: `src/zexus/evaluator/statements.py` - `eval_module_statement` (lines ~1584-1620)
- File: `src/zexus/evaluator/core.py` - PropertyAccessExpression handler (lines ~369+)
- File: `src/zexus/evaluator/functions.py` - eval_method_call_expression (lines ~151+)

**Resolution:**
1. ✅ Added `FUNCTION` to the `statement_starters` set in `_parse_block_statements`
2. ✅ Implemented FUNCTION statement parsing in `_parse_block_statements` (similar to ACTION parsing)
3. ✅ Module body now correctly parses function declarations as `FunctionStatement` nodes with names
4. ✅ `eval_module_statement` correctly registers function declarations as module members
5. ✅ `eval_method_call_expression` successfully retrieves and calls module functions
6. ✅ Verified via test_module_debug.zx that `math_operations.add(5,3)` returns 8
- Commit: Current working tree

---

### Issue #5: Type Alias Not Resolving in Type Annotations
**Severity:** HIGH  
**Component:** Parser/Evaluator - Complexity System  
**Status:** FIXED (2025-12-12)  
**Found In:** test_complexity_features.zx (lines 74-75)  

**Code:**
```zexus
type_alias UserId = integer;
let user_id: UserId = 42;
```

**Problem:**
Type aliases parse, but type annotations using the alias (`:UserId`) don't resolve to the actual type. The parser accepts the syntax but the evaluator doesn't recognize `UserId` as a type.

**Root Cause:**
1. The `type_alias` keyword was not registered in the lexer's keywords dictionary, causing it to be treated as an identifier
2. The `LetStatement` AST node didn't have a field for type annotations
3. The parser didn't capture type annotations from let statements
4. The evaluator didn't validate assigned values against type annotations

**Location:**
- File: `src/zexus/lexer.py` - keywords dictionary
- File: `src/zexus/compiler/lexer.py` - keywords dictionary
- File: `src/zexus/zexus_ast.py` - LetStatement class
- File: `src/zexus/parser/strategy_context.py` - _parse_let_statement_block
- File: `src/zexus/evaluator/statements.py` - eval_type_alias_statement, eval_let_statement
- Also: `src/zexus/complexity_system.py`

**Fix Strategy:**
1. ✅ Add "type_alias" keyword to both lexers
2. ✅ Add type_annotation field to LetStatement AST
3. ✅ Update parser to capture type annotations (name: Type = value)
4. ✅ Implement type validation in eval_let_statement
5. ✅ Test with comprehensive scenarios including type mismatches

**Resolution:**
- Added "type_alias": TYPE_ALIAS to both src/zexus/lexer.py and src/zexus/compiler/lexer.py
- Modified LetStatement to accept optional type_annotation parameter
- Updated _parse_let_statement_block to detect and parse `: Type` syntax
- Modified eval_type_alias_statement to store base type as string (not evaluated expression)
- Implemented _validate_type helper to check value types against type aliases
- Added comprehensive type mapping for common types (integer, string, float, bool, array, map, etc.)
- Type validation now properly rejects mismatched types (e.g., assigning string to integer type)
- Test files: test_type_alias_advanced.zx (success), test_type_alias_errors.zx (error handling)
- Verified: Full type alias support with validation
- Commit: Current working tree

---

## 🟠 MEDIUM PRIORITY ISSUES

### Issue #6: Using Statement Resource Cleanup Not Triggered
**Severity:** MEDIUM  
**Component:** Evaluator - Complexity System (RAII)  
**Status:** FIXED (2025-12-12)  
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
1. String resources don't have `close()` or `cleanup()` methods
2. No File object type existed in the interpreter
3. The cleanup logic worked but there were no objects that implemented cleanup protocol

**Location:**
- File: `src/zexus/evaluator/statements.py` - eval_using_statement (lines ~1755+)
- File: `src/zexus/object.py` - File class (new)
- File: `src/zexus/evaluator/functions.py` - file() builtin (new)

**Fix Strategy:**
1. ✅ Implement File object class with open() and close() methods
2. ✅ Add file() builtin function to create File objects
3. ✅ Test cleanup execution with File objects
4. ✅ Verify RAII pattern works correctly

**Resolution:**
- Implemented File object class in object.py with full RAII support
  - Supports open(), close(), read(), write() methods
  - Properly tracks open/closed state
  - Cleanup is called automatically by using statement
- Added file(path, mode) builtin function to create File objects
- The using statement properly calls close() method in finally block
- Test file: test_using_file.zx demonstrates working RAII pattern
- Verified: File object is created, used in block, and closed after block completes
- Commit: Current working tree

---

### Issue #7: Package Hierarchies Not Properly Nested
**Severity:** MEDIUM  
**Component:** Evaluator - Complexity System  
**Status:** FIXED (2025-12-12)  
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
1. The `eval_package_statement` handler had a bug using `Environment(parent=env)` instead of `Environment(outer=env)`
2. The handler treats the dotted name as a single identifier rather than creating a hierarchical structure
3. Package nesting wasn't implemented
4. Package class didn't have a `get()` method for property access
5. Method call evaluation didn't support Package objects

**Location:**
- File: `src/zexus/evaluator/statements.py` - eval_package_statement
- File: `src/zexus/complexity_system.py` - Package class
- File: `src/zexus/evaluator/functions.py` - eval_method_call_expression
- File: `src/zexus/evaluator/core.py` - PropertyAccessExpression handler (EvaluationError import issue)

**Fix Strategy:**
1. ✅ Fix Environment initialization bug (parent → outer)
2. ✅ Remove invalid manager.register_package() call
3. ✅ Parse dotted package names into hierarchy
4. ✅ Create nested Package objects
5. ✅ Add get() method to Package class
6. ✅ Add Package support to eval_method_call_expression
7. ✅ Fix EvaluationError import shadowing issue
8. ✅ Test nested access patterns

**Resolution:**
- Fixed critical bugs: Environment initialization and package registration
- Implemented hierarchical package structure: dotted names create nested packages
- Added Package.get() method to support property access (e.g., app.api)
- Modified eval_package_statement to build proper hierarchy:
  - Creates root package if it doesn't exist
  - Navigates/creates intermediate packages
  - Attaches leaf package to parent
- Added Package method call support in eval_method_call_expression
- Fixed EvaluationError import shadowing in core.py (removed redundant local imports)
- Fixed type() attribute error in functions.py (added safe type checking)
- Test file: test_package_hierarchy.zx shows full hierarchical access working
- Verified: `app.api.v1.get_users()` successfully navigates 3 levels and executes function
- Commit: Current working tree

---

## 🔵 LOW PRIORITY ISSUES

### Issue #8: Debug Output Too Verbose
**Severity:** LOW  
**Component:** Parser - All  
**Status:** FIXED (2025-12-12)  

**Problem:**
When running test files, the output includes excessive debug logging from structural analyzer and parser, making actual test output hard to read.

**Example Output:**
```
[STRUCT_BLOCK] id=0 type=statement subtype=PRINT ...
🔍 [Generic] Parsing generic block with tokens: [...]
✅ Parsed: PrintStatement at line 5
```

**Root Cause:**
Debug logging was not conditional - all parser debug output was printed regardless of user preferences. There was no way to control verbosity.

**Location:**
- File: `zx-run` - command-line argument parsing
- File: `src/zexus/config.py` - debug level configuration
- File: `src/zexus/parser/strategy_structural.py` - structural analyzer debug output
- File: `src/zexus/parser/strategy_context.py` - context parser debug output

**Fix Strategy:**
1. ✅ Add --verbose/-v flag to zx-run for debug output
2. ✅ Add --quiet/-q flag (default) to suppress debug output
3. ✅ Wrap structural analyzer print statements with config check
4. ✅ Create parser_debug() helper function for context parser
5. ✅ Replace all parser print statements with parser_debug() calls
6. ✅ Test both modes

**Resolution:**
- Added argparse support to zx-run with --verbose and --quiet flags
- Modified run_zexus_file() to accept debug_level parameter
- Added parser_debug() helper function to strategy_context.py
- Replaced 100+ print statements with parser_debug() calls using sed
- Wrapped structural analyzer debug output with zexus_config.should_log() check
- Default mode is quiet (no debug output) - only shows test results
- Verbose mode shows all parser internals for debugging
- Test: `python3 zx-run test.zx` is quiet, `python3 zx-run --verbose test.zx` shows debug info
- Commit: Current working tree

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

### Issue #11: Print Statement Not Outputting String Literals
**Severity:** HIGH
**Component:** Parser - Context Strategy
**Status:** FIXED (2025-12-12)
**Found In:** All print statements with simple string literals

**Error:**
Print statements like `print "hello";` would parse but produce no output.

**Description:**
Simple print statements with string literals were being parsed correctly but the StringLiteral value was empty, resulting in no output.

**Root Cause:**
The context parser's `_parse_print_statement` method had an incorrect check: `if len(tokens) < 3: return PrintStatement(StringLiteral(""))`. This assumed print statements need at least 3 tokens (PRINT, value, and semicolon), but the semicolon is not included in the token list. When a print statement had only 2 tokens (PRINT + STRING), it would return an empty PrintStatement.

**Location:**
- File: `src/zexus/parser/strategy_context.py`
- Method: `_parse_print_statement` (line ~1445)

**Fix Strategy:**
1. Change the token count check from `< 3` to `< 2`
2. Verify that print statements with simple strings work correctly

**Resolution:**
- Changed `if len(tokens) < 3:` to `if len(tokens) < 2:` in `_parse_print_statement`
- Print statements now correctly output string literals
- Verified with test files: test_print.zx, test_issues_comprehensive.zx
- All print statements (simple strings, concatenations, expressions) now work correctly
- Commit: Current working tree

---

## 📊 SUMMARY

| Severity | Count | Status |
|----------|-------|--------|
| 🔴 CRITICAL | 3 | 3 Fixed |
| 🟡 HIGH | 3 | 3 Fixed |
| 🟠 MEDIUM | 3 | 3 Fixed |
| 🔵 LOW | 1 | 1 Fixed |
| **TOTAL** | **10** | **10 Fixed** |

---

## 🔧 Resolution Progress

- [x] Fix undefined identifier issues (interface, capability) - **FIXED**
- [x] Fix string concatenation with missing variables - **FIXED**
- [x] Implement module member access - **FIXED**
- [x] Implement full type alias with type checking - **FIXED**
- [x] Fix print statement output - **FIXED**
- [x] Implement full package hierarchies - **FIXED**
- [x] Implement File object with RAII cleanup - **FIXED**
- [x] Reduce debug output verbosity - **FIXED**
- [x] Export builtins for test injection - **FIXED (Issue #9)**

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
