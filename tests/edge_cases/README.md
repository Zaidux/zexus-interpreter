# Edge Case Tests

This directory contains comprehensive edge case tests for the Zexus interpreter.

## Purpose

These tests verify that the Zexus interpreter handles edge cases gracefully and doesn't crash on:
- Invalid inputs
- Boundary conditions
- Null/undefined values
- Arithmetic errors
- Empty collections
- And other corner cases

## Test Files

### `test_comprehensive_edge_cases.py`
Main test suite covering all major edge cases (18 tests):
- Arithmetic operations (division by zero, modulo by zero, large numbers, etc.)
- Null and empty values
- Collections and indexing
- Boolean operations
- Comparison operators
- Control flow (if/else, while loops)
- Function definitions and calls
- String handling

**Run with:**
```bash
python tests/edge_cases/test_comprehensive_edge_cases.py
```

### `test_arithmetic_edge_cases.py`
Focused tests for arithmetic edge cases:
- Division by zero
- Modulo by zero
- Float division by zero
- Very large numbers
- Negative numbers
- Float precision

**Run with:**
```bash
python tests/edge_cases/test_arithmetic_edge_cases.py
```

## Test Results

```
TOTAL: 18 passed, 0 failed (100%)
```

All edge cases are properly handled.

## What These Tests Verify

### Safety
- ✅ No crashes on division by zero
- ✅ No crashes on null values
- ✅ No crashes on empty collections
- ✅ No crashes on undefined variables

### Correctness
- ✅ Arithmetic operations return correct results
- ✅ Boolean logic works as expected
- ✅ Comparisons work correctly
- ✅ Control flow executes properly
- ✅ Functions handle parameters correctly

### Error Handling
- ✅ Division by zero returns error with helpful message
- ✅ Modulo by zero returns error with helpful message
- ✅ Invalid operations caught gracefully

## Adding New Tests

To add new edge case tests:

1. Add a new test function to `test_comprehensive_edge_cases.py`:
```python
def test_your_edge_case():
    """Description of what you're testing."""
    env, result = run_code("""
    # Your Zexus code here
    """)
    # Your assertions here
    print("✅ Your test: works correctly")
```

2. Add it to the appropriate category in the `tests` list at the bottom.

3. Run the test suite to verify it passes.

## Integration with CI/CD

These tests can be integrated into CI/CD pipelines:

```bash
# Run all edge case tests
python -m pytest tests/edge_cases/ -v

# Or run individually
python tests/edge_cases/test_comprehensive_edge_cases.py
```

## Related Documentation

- [Edge Case Testing Report](../../docs/EDGE_CASE_TESTING_REPORT.md) - Full report of all fixes
- [Test Results](../TEST_RESULTS.md) - Overall test results
- [Main README](../../README.md) - Project overview

## Issues Fixed

These tests helped identify and fix:
1. Missing `Environment.assign()` method (while loops failed)
2. Various edge cases in arithmetic operations
3. Null handling in various contexts

See [EDGE_CASE_TESTING_REPORT.md](../../docs/EDGE_CASE_TESTING_REPORT.md) for details.
