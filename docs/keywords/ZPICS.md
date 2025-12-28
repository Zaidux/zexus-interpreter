# ZPICS - Zexus Parser Invariant Checking System

## Overview

ZPICS (Zexus Parser Invariant Checking System) is a comprehensive regression prevention system that ensures parser changes don't break existing functionality. It works by maintaining parse tree fingerprints and validating statement boundaries against golden test cases.

## Problem Statement

During parser development, we encountered a critical issue: changes made to fix one parsing scenario broke previously working code. Specifically:

- **Initial Fix**: Enhanced LET statement token collection to properly handle function calls after property assignments
- **Regression**: The fix broke type annotation parsing (`let x : string = "value"`)
- **Root Cause**: The parser detected `string =` as a new assignment statement instead of part of the type annotation

This highlighted the need for automated regression detection when modifying the parser.

## Solution: ZPICS

ZPICS provides a multi-layered approach to prevent parser AND runtime regressions:

### 1. Parse Tree Fingerprinting

For each test case, ZPICS captures:
- **Source code hash**: Ensures the test hasn't changed
- **Statement count**: Number of top-level statements parsed
- **Token boundaries**: Start/end positions for each statement
- **Variable declarations**: All declared variables
- **Statement types**: Sequence of statement types (LET, ACTION, PRINT, etc.)
- **AST structure**: Hierarchical structure of the parse tree

These elements are combined into a unique fingerprint that represents how the code parses.

### 2. Runtime Behavior Tracking

For each test case, ZPICS also captures runtime behavior:
- **Stdout output**: All printed output
- **Stderr output**: Error messages
- **Exit code**: Whether program succeeded or failed
- **Final variables**: Values of all variables after execution
- **Execution time**: Performance baseline
- **Execution fingerprint**: Unique hash of runtime behavior

### 3. Golden Test Suite

Critical parsing scenarios are preserved as golden tests in `tests/golden/`:

#### type_annotations.zx
Tests that type annotations don't break statement parsing:
```zexus
let simple_typed : string = "hello"
let number_typed : integer = 42
```

#### persistence_after_assignment.zx
Tests that function calls execute after property assignments:
```zexus
let order = Order(1, "pending")
order.status = "completed"
persist_set("order_1", order)  # Must not be consumed by previous line
```

#### property_assignments.zx
Tests that property assignments are properly separated:
```zexus
user.name = "Bob"
user.email = "bob@example.com"  # Each assignment is separate
```

#### function_calls_after_let.zx
Tests that LET statements don't consume following function calls:
```zexus
let value = 42
let result = process(value)  # process() call must execute
```

#### mixed_statement_boundaries.zx
Tests complex scenarios with multiple statement types mixed together.

#### nested_expressions.zx
Tests that nested parentheses/brackets don't interfere with statement parsing.

### 3. Baseline Snapshots

Before making parser OR evaluator changes:
```bash
python3 zpics baseline
```

This creates JSON snapshots of:
1. How each golden test parses (parse tree fingerprint)
2. How each golden test executes (runtime behavior fingerprint)

Snapshots are stored in:
- `.zpics_snapshots/` - Parser snapshots
- `.zpics_runtime/` - Runtime snapshots

### 4. Validation

After making parser OR evaluator changes:
```bash
python3 zpics validate
```

This validates BOTH:
1. **Parser invariants** - Compares current parse against baseline
2. **Runtime behavior** - Compares current execution against baseline

And reports:

**Critical Violations**:
- Statement count changed
- Token boundaries shifted
- Variables missing or extra
- Statement type sequence changed
- Parse tree fingerprint mismatch
- **Output changed** (runtime)
- **Exit code changed** (runtime)
- **Variable values changed** (runtime)
- **Execution fingerprint mismatch** (runtime)

**Warnings**:
- Extra variables declared (might be intentional)
- **Performance regression** (runtime - execution time doubled)

**Info**:
- Source code hash changed (test was modified)

## Usage

### Creating Baseline Snapshots

```bash
# Create snapshots for all golden tests
python3 zpics baseline

# Create snapshots for custom directory
python3 zpics baseline --golden-dir custom/tests
```

### Validating Parser Changes

```bash
# Validate against baseline
python3 zpics validate

# Validate with custom directories
python3 zpics validate --golden-dir custom/tests
```

### Listing Snapshots

```bash
# List all snapshots
python3 zpics list

# List snapshots in custom directory
python3 zpics list --snapshot-dir custom/snapshots
```

## Workflow Integration

### Before Parser Changes

1. Ensure all tests pass
2. Create baseline snapshots: `python3 zpics baseline`
3. Commit snapshots to git

### During Development

1. Make parser changes
2. Run validation: `python3 zpics validate`
3. If violations detected:
   - Review each violation carefully
   - Fix regressions OR update golden tests if changes are intentional
   - Re-validate until clean

### After Parser Changes

1. Validate passes without critical violations
2. Run full test suite
3. Update baseline if golden tests changed: `python3 zpics baseline`
4. Commit updated snapshots

## Example Output

### Successful Validation

```
======================================================================
VALIDATING PARSER INVARIANTS
======================================================================
✅ All parser invariants validated successfully!
   Tested 6 golden test cases

======================================================================
VALIDATING RUNTIME BEHAVIOR
======================================================================
✅ All runtime invariants validated successfully!
   Tested 6 golden test cases
```

### Validation with Violations

**Parser Violation:**
```
======================================================================
ZPICS VALIDATION REPORT
======================================================================

❌ CRITICAL VIOLATIONS: 2
----------------------------------------------------------------------
  Test: type_annotations
  Type: statement_count_mismatch
  Statement count changed from 5 to 4
  Expected: 5
  Actual: 4

  Test: type_annotations
  Type: missing_variables
  Variables missing: {'simple_typed'}
  Expected: ['simple_typed', 'number_typed', 'complex_typed']
  Actual: ['string', 'number_typed', 'complex_typed']

======================================================================
Total violations: 2
Critical: 2 | Warnings: 0 | Info: 0
======================================================================
```

**Runtime Violation:**
```
======================================================================
ZPICS RUNTIME VALIDATION REPORT
======================================================================

❌ CRITICAL VIOLATIONS: 1
----------------------------------------------------------------------
  Test: type_annotations
  Type: stdout_changed
  Program output has changed
  Expected output:
    Simple: hello
    Number: 42
  Actual output:
    Simple: goodbye
    Number: 42

======================================================================
Total violations: 1
Critical: 1 | Warnings: 0 | Info: 0
======================================================================
```

## Architecture

### ParseSnapshot

Dataclass representing a single parse result:
- Serializable to/from JSON
- Generates deterministic fingerprints
- Tracks all critical parse metadata

### ZPICSValidator

Main validation engine:
- Creates snapshots from source code
- Saves/loads snapshots to disk
- Compares current vs baseline
- Generates violation reports

### InvariantViolation

Represents a detected issue:
- Test name and violation type
- Expected vs actual values
- Severity level
- Human-readable description

## Benefits

1. **Early Detection**: Catches regressions immediately during development
2. **Confidence**: Make parser changes knowing breaking changes will be caught
3. **Documentation**: Golden tests serve as parser behavior specification
4. **Debugging**: Violations pinpoint exactly what changed in the parse tree
5. **Automation**: Can be integrated into CI/CD pipelines

## Future Enhancements

- **CI Integration**: Automatic validation on pull requests
- **Performance Tracking**: Monitor parse time regressions
- **Diff Visualization**: Visual comparison of AST changes
- **Auto-fix Suggestions**: Suggest parser fixes for common violations
- **Coverage Analysis**: Track which parser paths are tested

## Technical Details

### Snapshot Storage

Snapshots are stored as JSON in `.zpics_snapshots/`:
```json
{
  "source_code": "let x : string = \"hello\"",
  "source_hash": "a1b2c3d4...",
  "statements_count": 1,
  "token_boundaries": [[0, 25]],
  "variable_declarations": ["x"],
  "statement_types": ["LetStatement"],
  "ast_structure": {...},
  "parse_metadata": {...}
}
```

### Fingerprint Generation

Fingerprints are SHA-256 hashes of:
- Source hash
- Statement count
- Token boundaries (sorted)
- Variable declarations (sorted)
- Statement types sequence

This ensures deterministic comparison across runs.

## Contributing

When adding new parser features:

1. Create golden test demonstrating the feature
2. Generate baseline snapshot
3. Make parser changes
4. Validate against baseline
5. Add documentation for the feature
6. Commit golden test and baseline together

## Related Documentation

- [Parser Architecture](../ARCHITECTURE.md)
- [Parser Optimization Guide](../PARSER_OPTIMIZATION_GUIDE.md)
- [Testing Guide](../guides/testing.md)

---

**Created**: December 2025  
**Author**: Zexus Team  
**Status**: Active
