# Golden Tests for ZPICS

This directory contains critical parsing scenarios that must not regress when making parser changes.

## Test Files

### type_annotations.zx
Tests type annotation parsing in LET statements. Ensures that `let x : type = value` doesn't break at the type annotation's `=` sign.

### persistence_after_assignment.zx
Tests that persistence operations (`persist_set`, `persist_get`) execute correctly after property assignments. Ensures function calls aren't consumed by previous statements.

### property_assignments.zx
Tests that property assignments (`obj.prop = value`) are properly separated into individual statements, including consecutive property assignments on the same object.

### function_calls_after_let.zx
Tests that function calls following LET statements are correctly identified as separate statements and execute properly.

### mixed_statement_boundaries.zx
Complex scenario mixing type annotations, property assignments, function calls, multi-parameter typed functions, and entity declarations to ensure all statement boundaries are correctly identified.

### nested_expressions.zx
Tests that deeply nested expressions with parentheses, brackets, and braces don't interfere with statement boundary detection.

## Usage

These tests are automatically used by ZPICS. To validate parser changes:

```bash
# Create baseline (first time or after intentional changes)
python3 zpics baseline

# Validate after parser modifications
python3 zpics validate
```

## Adding New Golden Tests

1. Create a `.zx` file in this directory demonstrating the parsing scenario
2. Run `python3 zpics baseline` to create the baseline snapshot
3. Commit both the test file and snapshot together
4. Document the test purpose in this README

## Important

- **Never modify these files without regenerating baselines**
- **Each file should test a specific parsing scenario**
- **Keep tests small and focused** (easier to debug violations)
- **Add comments explaining what's being tested**
