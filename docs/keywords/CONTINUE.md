# CONTINUE Keyword Documentation

## Overview

The `CONTINUE` keyword enables **error recovery mode** in Zexus, allowing programs to continue execution even when errors are encountered. This is particularly useful for:

- **Debugging and Development**: Continue running tests even when some fail
- **Data Processing**: Process all records even if some are invalid
- **Logging and Monitoring**: Collect all error information before halting
- **Graceful Degradation**: Continue with fallback behavior when primary operations fail
- **Batch Operations**: Complete as many operations as possible

## Syntax

```zexus
continue;
```

The `CONTINUE` statement is simple and takes no arguments. It can be placed anywhere in your code, typically at the beginning of a block or after initialization.

## Behavior

### Without CONTINUE (Default Behavior)

```zexus
print "Step 1";
revert("Error occurred");
print "Step 2";  // This will NOT execute
```

**Output:**
```
Step 1
❌ Error: Transaction reverted: Error occurred
```

### With CONTINUE (Error Recovery Mode)

```zexus
print "Step 1";
continue;
revert("Error occurred");
print "Step 2";  // This WILL execute
```

**Output:**
```
Step 1
[ERROR] Transaction reverted: Error occurred
Step 2
```

## Key Features

### 1. Error Logging
When `CONTINUE` is active, all errors are:
- **Logged** with `[ERROR]` prefix
- **Printed** to console with stack trace
- **Stored** in the evaluator's error log for later review

### 2. Execution Continues
After an error:
- The error is logged
- Execution continues with the next statement
- Variables and state remain accessible
- Functions continue to execute

### 3. Global Scope
Once `CONTINUE` is called:
- It affects the **entire program** execution
- Cannot be disabled once enabled
- Applies to all subsequent errors

## Usage Examples

### Example 1: Basic Error Recovery

```zexus
print "Starting data processing...";
continue;

let data = [100, -50, 200, -30, 300];
let total = 0;

for each item in data {
    if (item < 0) {
        revert("Invalid item: " + item);
    }
    total = total + item;
}

print "Total: " + total;
print "Processing complete despite errors!";
```

### Example 2: Validation Pipeline

```zexus
continue;  // Enable error recovery

action validateEmail(email) {
    if (email == "invalid") {
        revert("Invalid email format");
        return false;
    }
    return true;
}

action validateAge(age) {
    if (age < 0) {
        revert("Invalid age");
        return false;
    }
    return true;
}

// Process multiple records
validateEmail("user@example.com");  // OK
validateEmail("invalid");           // Error logged, continues
validateAge(25);                    // OK
validateAge(-5);                    // Error logged, continues

print "All validations attempted!";
```

### Example 3: Batch Processing

```zexus
print "=== Batch Processing ===";
continue;

action processRecord(id) {
    if (id % 2 == 0) {
        revert("Processing failed for ID: " + id);
        return null;
    }
    return "Processed: " + id;
}

let ids = [1, 2, 3, 4, 5];
let successCount = 0;

for each id in ids {
    let result = processRecord(id);
    if (result != null) {
        successCount = successCount + 1;
    }
}

print "Successfully processed: " + successCount + " records";
```

### Example 4: Resource Cleanup

```zexus
continue;

action processWithCleanup() {
    print "Acquiring resources...";
    
    // Simulate error during processing
    revert("Processing error");
    
    // Cleanup still happens with CONTINUE
    print "Releasing resources...";
    
    return "completed";
}

let status = processWithCleanup();
print "Status: " + status;
```

## Rules and Best Practices

### ✅ DO:

1. **Place CONTINUE early** - Typically at the start of your script or function
   ```zexus
   continue;
   // rest of your code
   ```

2. **Use for batch operations** - Process all items even if some fail
   ```zexus
   continue;
   for each item in items {
       processItem(item);  // Errors won't stop the loop
   }
   ```

3. **Log errors for review** - Errors are automatically logged
   ```zexus
   continue;
   // All errors will be in the error log
   ```

4. **Combine with try-catch** - Use both for fine-grained control
   ```zexus
   continue;
   try {
       riskyOperation();
   } catch (e) {
       print "Caught: " + e;
   }
   ```

### ❌ DON'T:

1. **Don't use in production without understanding** - Know that errors will be logged but not halt
   ```zexus
   // ⚠️ Production systems should handle errors explicitly
   continue;
   criticalDatabaseOperation();  // Error won't stop execution!
   ```

2. **Don't rely on CONTINUE for critical errors** - Some errors require immediate halt
   ```zexus
   // ❌ Bad: Critical errors should stop execution
   continue;
   connectToDatabase();  // If this fails, continuing is dangerous
   ```

3. **Don't use multiple CONTINUE statements** - Only the first one is needed
   ```zexus
   // ⚠️ Unnecessary
   continue;
   continue;  // This does nothing
   ```

4. **Don't expect to disable CONTINUE** - It's permanent for the session
   ```zexus
   continue;
   // No way to turn off error recovery mode
   ```

## Advanced Use Cases

### 1. Testing Framework

```zexus
continue;  // Run all tests even if some fail

action test_addition() {
    let result = 2 + 2;
    require(result == 4, "Addition failed");
    print "✓ Addition test passed";
}

action test_subtraction() {
    let result = 5 - 3;
    require(result == 99, "Subtraction failed");  // Will fail
    print "✓ Subtraction test passed";
}

action test_multiplication() {
    let result = 3 * 4;
    require(result == 12, "Multiplication failed");
    print "✓ Multiplication test passed";
}

test_addition();
test_subtraction();     // Fails but continues
test_multiplication();

print "All tests executed!";
```

### 2. ETL Pipeline

```zexus
continue;

action extractData(source) {
    if (source == "broken") {
        revert("Source unavailable: " + source);
        return null;
    }
    return "data from " + source;
}

action transformData(data) {
    if (data == null) {
        return null;
    }
    return "transformed: " + data;
}

action loadData(data) {
    if (data == null) {
        print "Skipping null data";
        return;
    }
    print "Loaded: " + data;
}

let sources = ["db1", "broken", "db2", "broken", "db3"];

for each source in sources {
    let extracted = extractData(source);
    let transformed = transformData(extracted);
    loadData(transformed);
}

print "ETL pipeline completed!";
```

### 3. Circuit Breaker Pattern

```zexus
continue;

let failures = 0;
let threshold = 3;
let circuitOpen = false;

action performOperation(shouldFail) {
    if (circuitOpen) {
        print "Circuit breaker is open - operation skipped";
        return "skipped";
    }
    
    if (shouldFail) {
        failures = failures + 1;
        revert("Operation failed (attempt " + failures + ")");
        
        if (failures >= threshold) {
            circuitOpen = true;
            print "⚠️ Circuit breaker opened after " + threshold + " failures";
        }
        return "failed";
    }
    
    failures = 0;  // Reset on success
    return "success";
}

// Simulate failing operations
performOperation(true);   // Failure 1
performOperation(true);   // Failure 2
performOperation(true);   // Failure 3 - Circuit opens
performOperation(false);  // Skipped - circuit is open

print "Circuit breaker pattern demonstrated";
```

## Error Handling Flow

```
┌─────────────────┐
│  Execute Code   │
└────────┬────────┘
         │
         ▼
   ┌─────────┐
   │ Error?  │──No──▶ Continue normally
   └────┬────┘
        │ Yes
        ▼
   ┌──────────────┐
   │ CONTINUE on? │──No──▶ Halt execution, return error
   └────┬─────────┘
        │ Yes
        ▼
   ┌──────────────────┐
   │ Log error        │
   │ Print to console │
   │ Store in log     │
   └────┬─────────────┘
        │
        ▼
   ┌──────────────────┐
   │ Continue with    │
   │ next statement   │
   └──────────────────┘
```

## Comparison with Try-Catch

| Feature | CONTINUE | TRY-CATCH |
|---------|----------|-----------|
| Scope | Global (entire program) | Local (specific block) |
| Error Handling | Log and continue | Execute catch block |
| Control | Automatic | Manual (requires catch) |
| Use Case | Batch operations, testing | Fine-grained error handling |
| Combination | Can be used together | Can be used together |

### Using Both Together

```zexus
continue;  // Global error recovery

try {
    riskyOperation();
} catch (e) {
    print "Specific error handling: " + e;
}

// CONTINUE ensures execution continues even if catch fails
```

## Performance Considerations

- **Minimal Overhead**: Error logging adds negligible performance impact
- **Memory**: Error messages are stored in evaluator's error log
- **Best Practice**: Use for development/testing, consider disabling in production

## Compatibility

- **Works with**: All error-producing statements (revert, require, etc.)
- **Blocks**: Works within all block types (if, while, for, functions)
- **Scopes**: Affects all scopes once enabled
- **Version**: Available in Zexus v1.5.0+

## Summary

The `CONTINUE` keyword is a powerful tool for:
- ✅ **Development and Testing**: Run all tests even if some fail
- ✅ **Data Processing**: Handle all records despite invalid data
- ✅ **Error Collection**: Gather all errors before stopping
- ✅ **Graceful Degradation**: Fall back to safe defaults
- ⚠️ **Use Carefully**: Understand that errors won't halt execution

**Critical Safety Warning**: Do NOT use CONTINUE when:
- Database connections fail (could corrupt data)
- Security validations fail (could create vulnerabilities)
- File system operations fail (could lose data)
- Authentication/authorization errors occur (security risk)
- Critical system resources are unavailable

**Remember**: With great power comes great responsibility. Use `CONTINUE` when you want to collect all errors and process all data, but ensure your application can handle partial failures gracefully. It's primarily designed for development, testing, and non-critical batch operations.
