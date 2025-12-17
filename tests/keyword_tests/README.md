# Keyword Tests

This directory contains comprehensive tests for all Zexus language keywords.

## Directory Structure

```
keyword_tests/
├── README.md                   # This file
├── run_keyword_test.sh        # Test runner script
├── easy/                      # Easy difficulty tests
│   └── test_<keyword>_easy.zx
├── medium/                    # Medium difficulty tests
│   └── test_<keyword>_medium.zx
└── complex/                   # Complex difficulty tests
    └── test_<keyword>_complex.zx
```

## Running Tests

### Using the Test Runner

```bash
# From the project root
cd /workspaces/zexus-interpreter

# Run a specific test level
./tests/keyword_tests/run_keyword_test.sh <keyword> <level>

# Examples
./tests/keyword_tests/run_keyword_test.sh let easy
./tests/keyword_tests/run_keyword_test.sh let medium
./tests/keyword_tests/run_keyword_test.sh let complex
./tests/keyword_tests/run_keyword_test.sh let all
```

### Direct Execution

```bash
# Run a test file directly
./zx tests/keyword_tests/easy/test_let_easy.zx
./zx tests/keyword_tests/medium/test_let_medium.zx
./zx tests/keyword_tests/complex/test_let_complex.zx
```

## Test Levels

### Easy Tests
- **Purpose**: Verify basic functionality
- **Complexity**: Simple, single-feature tests
- **Count**: 15-20 tests per keyword
- **Expected**: Should all pass

### Medium Tests
- **Purpose**: Test integration and edge cases
- **Complexity**: Intermediate, multiple features
- **Count**: 15-20 tests per keyword
- **Expected**: Most should pass

### Complex Tests
- **Purpose**: Stress test and advanced scenarios
- **Complexity**: Advanced, multi-feature integration
- **Count**: 10-15 tests per keyword
- **Expected**: May reveal edge cases

## Test Format

Each test file follows this structure:

```zexus
# Test N: Description
<test code>
# Expected output: <expected result>
```

### Example

```zexus
# Test 1: Basic integer variable
let x = 10;
print x;
# Expected output: 10

# Test 2: Variable reassignment
let counter = 0;
counter = counter + 1;
print counter;
# Expected output: 1
```

## Keywords Tested

See `../../docs/KEYWORD_TESTING_MASTER_LIST.md` for the complete list of keywords and their test status.

### Currently Available

- [x] LET (easy ✅, medium ⚠️, complex ⚠️)
- [ ] CONST
- [ ] IF, ELIF, ELSE
- [ ] PRINT, DEBUG
- [ ] And more...

## Error Reporting

If a test fails:

1. **Note the error message**
2. **Check the master list** at `docs/KEYWORD_TESTING_MASTER_LIST.md`
3. **Add to error log** if it's a new issue
4. **Document workarounds** in test comments

### Example Error Documentation

```zexus
# Test 13: Alternative syntax with colon
# DISABLED - BUG FOUND: Colon syntax not working correctly
# let value : 100;
# print value;
# Using = instead for now
let value = 100;
print value;
```

## Contributing New Tests

To add tests for a new keyword:

1. **Create test files** in each directory:
   ```bash
   touch easy/test_<keyword>_easy.zx
   touch medium/test_<keyword>_medium.zx
   touch complex/test_<keyword>_complex.zx
   ```

2. **Write tests** following the format above

3. **Run tests** using the test runner

4. **Document** in the master list

5. **Create documentation** in `docs/keywords/<KEYWORD>.md`

## Test Writing Guidelines

### DO
- ✅ Use descriptive test names
- ✅ Include expected output
- ✅ Test edge cases
- ✅ Document workarounds for bugs
- ✅ Keep tests focused and simple
- ✅ Add comments for clarity

### DON'T
- ❌ Create overly complex tests in the "easy" category
- ❌ Mix multiple unrelated features
- ❌ Skip error documentation
- ❌ Leave tests without expected output
- ❌ Use deprecated syntax without noting it

## Test Coverage Goals

For each keyword, aim to test:

1. **Basic Usage** (Easy)
   - Simple declarations
   - Basic operations
   - Type compatibility

2. **Integration** (Medium)
   - With other keywords
   - With data structures
   - With functions

3. **Advanced Scenarios** (Complex)
   - Nested usage
   - State management
   - Performance implications

## Automation

The test runner (`run_keyword_test.sh`) provides:

- ✅ Single command test execution
- ✅ Multiple test level support
- ✅ Pass/fail status reporting
- ✅ Output formatting

Future enhancements planned:
- [ ] JSON test results
- [ ] HTML report generation
- [ ] CI/CD integration
- [ ] Coverage metrics

## Resources

- **Master Tracking**: `/docs/KEYWORD_TESTING_MASTER_LIST.md`
- **Project Summary**: `/docs/KEYWORD_TESTING_PROJECT_SUMMARY.md`
- **Keyword Docs**: `/docs/keywords/`
- **Examples**: `/examples/`

## Status

**Last Updated**: December 16, 2025  
**Total Tests**: 50+  
**Keywords Covered**: 1/130+  
**Errors Found**: 2
