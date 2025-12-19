# Error Handling Keywords: TRY, CATCH, REVERT, REQUIRE

## Overview

Zexus provides a comprehensive error handling system inspired by both traditional try-catch mechanisms and smart contract patterns. The error handling keywords enable robust, safe code that can gracefully handle failures, validate conditions, and revert transactions when needed.

### Keywords Covered
- **TRY**: Defines a block of code that may throw errors
- **CATCH**: Handles errors thrown in the try block
- **REVERT**: Explicitly triggers a transaction reversal with an optional message
- **REQUIRE**: Validates conditions and reverts if they fail (guard clause pattern)

---

## TRY Keyword

### Syntax
```zexus
try {
    // Code that may throw errors
} catch (errorVariable) {
    // Error handling code
}
```

### Purpose
The `try` keyword marks a block of code for error handling. If any error occurs within the try block, execution jumps to the corresponding catch block.

### Basic Usage

#### Simple Try-Catch
```zexus
try {
    let x = 10;
    print x;
} catch (error) {
    print "Error occurred: " + error;
}
```

#### Try-Catch with Operations
```zexus
try {
    let result = 100 / 5;
    print "Result: " + result;
} catch (e) {
    print "Calculation failed: " + e;
}
```

### Advanced Patterns

#### Nested Try-Catch
```zexus
try {
    print "Outer try";
    try {
        print "Inner try";
        revert("Inner error");
    } catch (innerError) {
        print "Caught at inner level";
    }
    print "After inner try-catch";
} catch (outerError) {
    print "Caught at outer level";
}
```

#### Try-Catch in Functions
```zexus
action safeDivide(a, b) {
    try {
        if (b == 0) {
            revert("Division by zero");
        }
        return a / b;
    } catch (e) {
        return 0;  // Safe default
    }
}

print safeDivide(10, 2);   // 5
print safeDivide(10, 0);   // 0
```

---

## CATCH Keyword

### Syntax
```zexus
catch (errorVariable) {
    // Handle error
}

// Or with default variable name
catch (error) {
    // Handle error
}
```

### Purpose
The `catch` keyword defines how to handle errors thrown in the corresponding try block. The error variable contains information about the error.

### Basic Usage

#### Basic Error Catching
```zexus
try {
    revert("Something went wrong");
} catch (e) {
    print "Caught error";
}
```

#### Custom Error Variable Name
```zexus
try {
    require(false, "Validation failed");
} catch (validationError) {
    print "Validation error occurred";
}
```

### Advanced Patterns

#### Error Recovery
```zexus
action getConfigValue(config, key, defaultValue) {
    try {
        require(config != null, "Config is null");
        // Try to get value
        return config[key];
    } catch (e) {
        return defaultValue;  // Fallback
    }
}
```

#### Multiple Error Handling Levels
```zexus
try {
    // Level 1 operations
    try {
        // Level 2 operations
        try {
            // Level 3 operations
            revert("Deep error");
        } catch (e3) {
            print "Handled at level 3";
            revert("Propagating up");
        }
    } catch (e2) {
        print "Handled at level 2";
    }
} catch (e1) {
    print "Handled at level 1";
}
```

### Known Issues

⚠️ **Catch Syntax Warning**: Parser may warn about parentheses:
```
❌ Line X: Use parentheses with catch: catch(error) { }
```
**Workaround**: Use `catch (error)` with parentheses for consistency.

---

## REVERT Keyword

### Syntax
```zexus
revert();                           // Basic revert
revert("Error message");            // Revert with message
revert("Error: " + dynamicValue);   // Revert with expression
```

### Purpose
The `revert` keyword explicitly triggers a transaction reversal. It's commonly used in smart contract patterns to rollback state changes when conditions aren't met.

### Basic Usage

#### Simple Revert
```zexus
try {
    revert();
    print "This will not execute";
} catch (error) {
    print "Transaction reverted";
}
```

#### Revert with Message
```zexus
try {
    revert("Unauthorized access");
} catch (e) {
    print "Caught revert";
}
```

### Advanced Patterns

#### Conditional Revert
```zexus
action checkBalance(account, required) {
    try {
        if (account < required) {
            let shortfall = required - account;
            revert("Insufficient funds");
        }
        return "Sufficient balance";
    } catch (e) {
        return "Error: Transaction failed";
    }
}
```

#### Dynamic Error Messages
```zexus
action validateRange(value, min, max) {
    try {
        if (value < min) {
            revert("Value below minimum: " + min);
        }
        if (value > max) {
            revert("Value above maximum: " + max);
        }
        return "Valid";
    } catch (e) {
        return "Validation error";
    }
}
```

#### Transaction Rollback Pattern
```zexus
action executeTransaction(from, to, amount) {
    let originalState = saveState();  // Backup
    
    try {
        // Perform operations
        if (amount <= 0) {
            revert("Invalid amount");
        }
        if (from == to) {
            revert("Cannot send to self");
        }
        
        // Commit changes
        return "Success";
    } catch (error) {
        restoreState(originalState);  // Rollback
        return "Transaction reverted";
    }
}
```

### Known Issues

⚠️ **Revert Outside Try-Catch**: Using `revert()` outside a try-catch block may not be caught properly and could cause program termination.

**Workaround**: Always use `revert()` within a try-catch block for proper error handling.

---

## REQUIRE Keyword

### Syntax
```zexus
require(condition);                              // Basic assertion
require(condition, "Error message");             // With message
require(complex && expression, "Description");   // Complex condition

// Enhanced: Tolerance blocks for conditional bypasses
require condition {
    // Tolerance logic: if this returns truthy, bypass the requirement
}
```

### Purpose
The `require` keyword is a guard clause that validates conditions. If the condition is false, it triggers a revert. This is a common pattern in smart contracts for input validation and state checking.

**Enhanced Feature**: Tolerance blocks allow you to define conditional bypass logic for VIP users, loyalty programs, emergency overrides, and multi-tier requirements.

### Basic Usage

#### Simple Require
```zexus
let balance = 100;
require(balance > 0);
print "Balance check passed";
```

#### Require with Message
```zexus
let amount = 50;
require(amount <= 100, "Amount too large");
print "Amount validated";
```

### Advanced Patterns

#### Multiple Sequential Requires
```zexus
action validateTransfer(sender, recipient, amount) {
    require(amount > 0, "Amount must be positive");
    require(sender != recipient, "Cannot send to self");
    require(amount <= 1000, "Amount exceeds limit");
    return "Transfer valid";
}
```

#### Require with Complex Conditions
```zexus
action validateUser(age, balance, isActive) {
    require(age >= 18 && balance > 0 && isActive, "User validation failed");
    return "Valid user";
}
```

#### Require with Logical Operators
```zexus
// AND condition - both must be true
let x = 10;
let y = 20;
require(x > 5 && y > 15, "Conditions not met");

// OR condition - at least one must be true
let a = 5;
let b = 3;
require(a > 10 || b < 5, "Neither condition met");

// NOT condition
let isPaused = false;
require(!isPaused, "System is paused");
```

#### Require with Comparisons
```zexus
// Equality
require(actual == expected, "Values don't match");

// Inequality
require(value != 0, "Value cannot be zero");

// Range checks
require(value >= min, "Below minimum");
require(value <= max, "Above maximum");
```

### Tolerance Block Feature

#### VIP Fee Waiver
```zexus
action processPayment(balance, isVIP) {
    // Standard users need 0.1 BNB, but VIP users bypass this
    require balance >= 0.1 {
        if (isVIP) return true;
    }
    print "Payment processed";
}
```

#### Loyalty Discount Bypass
```zexus
action processOrder(amount, loyaltyPoints) {
    // Minimum purchase $100, but 500+ loyalty points bypass this
    require amount >= 100 {
        if (loyaltyPoints >= 500) return true;
    }
    print "Order placed";
}
```

#### Emergency Override
```zexus
action systemAccess(isAdmin, emergency, maintenanceMode) {
    // Maintenance blocks access, but admin in emergency can bypass
    require !maintenanceMode {
        if (isAdmin && emergency) return true;
    }
    print "Access granted";
}
```

#### Multi-Tier Requirements
```zexus
action accessPremiumFeature(balance, tier) {
    // Need 1.0 ETH, but gold/platinum tiers bypass
    require balance >= 1.0 {
        if (tier == "gold" || tier == "platinum") return true;
    }
    print "Premium feature unlocked";
}
```

### Smart Contract Patterns

#### Token Transfer Validation
```zexus
action tokenTransfer(from, to, amount, balances) {
    require(from != null, "Sender required");
    require(to != null, "Recipient required");
    require(amount > 0, "Amount must be positive");
    require(from != to, "Cannot transfer to self");
    require(balances[from] >= amount, "Insufficient balance");
    
    // Execute transfer
    balances[from] = balances[from] - amount;
    balances[to] = balances[to] + amount;
    return "Transfer successful";
}
```

#### Authorization Check
```zexus
action restrictedAction(caller, owner) {
    require(caller == owner, "Only owner can perform this action");
    // Execute restricted operation
    return "Action completed";
}
```

#### State Validation
```zexus
let contractState = "active";

action stateCheck() {
    require(contractState == "active", "Contract is not active");
    require(contractState != "paused", "Contract is paused");
    return "State is valid";
}
```

### Known Issues

⚠️ **Require as Function Call**: In some contexts, `require` may be incorrectly treated as a function call rather than a statement, causing "Not a function: require" errors.

**Status**: This appears to be a context-dependent parsing issue. Requires work in standalone statements but may fail inside certain function contexts.

**Workaround**: 
1. Use require at the top level of functions when possible
2. Consider using explicit if-revert patterns as an alternative:
```zexus
if (!condition) {
    revert("Error message");
}
```

---

## Error Handling Patterns

### 1. Guard Clause Pattern
```zexus
action processPayment(amount, balance) {
    require(amount > 0, "Invalid amount");
    require(balance >= amount, "Insufficient balance");
    
    // Main logic only executes if all guards pass
    print "Processing payment";
    return "Success";
}
```

### 2. Try-Catch-Finally Pattern (Cleanup)
```zexus
action withResource(shouldFail) {
    let resource = "acquired";
    print "Resource acquired";
    
    try {
        if (shouldFail) {
            revert("Operation failed");
        }
        print "Operation succeeded";
    } catch (e) {
        print "Error: " + e;
    }
    
    // Cleanup always happens
    print "Resource released";
    return "Done";
}
```

### 3. Error Recovery with Fallbacks
```zexus
action safeOperation(value, fallback) {
    try {
        require(value > 0, "Invalid value");
        return value * 2;
    } catch (e) {
        return fallback;  // Graceful degradation
    }
}

print safeOperation(10, 0);   // 20
print safeOperation(-5, 99);  // 99
```

### 4. Validation Pipeline
```zexus
action validateInput(data) {
    require(data != null, "Data is null");
    return true;
}

action validateFormat(data) {
    require(data != "", "Data is empty");
    return true;
}

action validateBusiness(data) {
    require(data != "invalid", "Business rule violation");
    return true;
}

action processWithPipeline(input) {
    try {
        validateInput(input);
        validateFormat(input);
        validateBusiness(input);
        return "Processed: " + input;
    } catch (e) {
        return "Validation failed";
    }
}
```

### 5. Retry Pattern
```zexus
let attemptCount = 0;

action unreliableOperation(shouldSucceed) {
    attemptCount = attemptCount + 1;
    try {
        print "Attempt " + attemptCount;
        if (!shouldSucceed && attemptCount < 3) {
            revert("Operation failed");
        }
        return "Success on attempt " + attemptCount;
    } catch (e) {
        return "Failed attempt " + attemptCount;
    }
}

// Call multiple times to retry
unreliableOperation(false);  // Attempt 1
unreliableOperation(false);  // Attempt 2
unreliableOperation(true);   // Attempt 3 - success
```

### 6. Circuit Breaker Pattern
```zexus
let failureCount = 0;
let circuitOpen = false;

action protectedOperation(shouldFail) {
    try {
        if (circuitOpen) {
            revert("Circuit breaker open");
        }
        
        if (shouldFail) {
            failureCount = failureCount + 1;
            if (failureCount >= 3) {
                circuitOpen = true;
            }
            revert("Operation failed");
        }
        
        failureCount = 0;  // Reset on success
        return "Success";
    } catch (e) {
        return "Blocked or failed";
    }
}
```

### 7. Error Aggregation
```zexus
action validateAll(value) {
    let errors = [];
    
    try {
        require(value > 0, "Must be positive");
    } catch (e) {
        print "Error: Must be positive";
    }
    
    try {
        require(value < 100, "Must be less than 100");
    } catch (e) {
        print "Error: Must be less than 100";
    }
    
    return "Validation complete";
}
```

### 8. State Machine with Error Handling
```zexus
let state = "idle";

action transition(newState) {
    try {
        if (state == "idle" && newState == "running") {
            state = newState;
            return "Transitioned to running";
        } elif (state == "running" && newState == "paused") {
            state = newState;
            return "Transitioned to paused";
        } else {
            revert("Invalid state transition");
        }
    } catch (e) {
        return "Transition failed";
    }
}
```

---

## Best Practices

### 1. Use Require for Preconditions
```zexus
// ✅ Good: Clear validation at function start
action transfer(from, to, amount) {
    require(from != null, "Sender required");
    require(to != null, "Recipient required");
    require(amount > 0, "Amount must be positive");
    // ... main logic
}

// ❌ Avoid: Scattered validation
action transfer(from, to, amount) {
    if (from == null) {
        return "Error";
    }
    // ... some code ...
    if (to == null) {
        return "Error";
    }
}
```

### 2. Provide Descriptive Error Messages
```zexus
// ✅ Good: Specific, actionable message
require(balance >= amount, "Insufficient balance: need " + amount + ", have " + balance);

// ❌ Avoid: Vague message
require(balance >= amount, "Error");
```

### 3. Handle Errors at the Appropriate Level
```zexus
// ✅ Good: Handle at the level where you can meaningfully respond
action processData(data) {
    try {
        validateData(data);
        transformData(data);
        saveData(data);
        return "Success";
    } catch (e) {
        logError(e);
        return "Processing failed";
    }
}

// ❌ Avoid: Catching and ignoring errors
try {
    criticalOperation();
} catch (e) {
    // Silent failure
}
```

### 4. Use Try-Catch for External/Unreliable Operations
```zexus
// ✅ Good: Protect against failures in external calls
action fetchUserData(userId) {
    try {
        let data = externalAPI(userId);
        return data;
    } catch (e) {
        return defaultUserData;
    }
}
```

### 5. Clean Up Resources
```zexus
// ✅ Good: Always release resources
action withFile(filename) {
    let file = openFile(filename);
    try {
        processFile(file);
    } catch (e) {
        print "Error processing file";
    }
    closeFile(file);  // Always executes
}
```

### 6. Don't Use Require for Control Flow
```zexus
// ❌ Avoid: Using require for normal logic
require(value > 10, "Continue processing");  // Bad

// ✅ Good: Use if for control flow
if (value > 10) {
    // Continue processing
}
```

### 7. Fail Fast with Require
```zexus
// ✅ Good: Validate early, fail fast
action complexOperation(a, b, c) {
    require(a > 0, "Invalid a");
    require(b > 0, "Invalid b");
    require(c > 0, "Invalid c");
    
    // Now safely proceed with complex logic
    return a * b * c;
}
```

---

## Edge Cases and Limitations

### Issue 1: Require Context Sensitivity
**Problem**: `require()` may be treated as a function call in some contexts.

```zexus
// Sometimes fails with "Not a function: require"
action test() {
    try {
        require(condition);  // May fail
    } catch (e) {
        print e;
    }
}
```

**Workaround**: Use explicit if-revert:
```zexus
action test() {
    try {
        if (!condition) {
            revert("Condition failed");
        }
    } catch (e) {
        print e;
    }
}
```

### Issue 2: Catch Syntax Warnings
**Problem**: Parser warns about catch syntax.

```
❌ Line X: Use parentheses with catch: catch(error) { }
```

**Solution**: Always use parentheses:
```zexus
// ✅ Recommended
catch (error) {
    // Handle
}
```

### Issue 3: Error Propagation in Nested Contexts
**Problem**: Errors in deeply nested try-catch blocks may not propagate as expected.

```zexus
try {
    try {
        try {
            revert("Deep error");
        } catch (e3) {
            print "Level 3";
            revert("Propagating");
        }
    } catch (e2) {
        print "Level 2";  // May catch propagated error
    }
} catch (e1) {
    print "Level 1";  // May not reach here
}
```

**Current Behavior**: Errors are caught at the innermost catch that can handle them, but re-throwing may not work as expected.

### Issue 4: Lambda/Closure Error Handling
**Problem**: Error handling in lambdas may not work correctly.

```zexus
let safeLambda = lambda(x) {
    try {
        return x * 2;
    } catch (e) {
        return 0;
    }
};

print safeLambda(10);  // May display as {} instead of result
```

**Related To**: Known lambda/closure display issues.

---

## Real-World Examples

### Example 1: Safe Token Transfer
```zexus
let balances = {"Alice": 1000, "Bob": 500};

action safeTransfer(from, to, amount) {
    try {
        // Validate inputs
        require(from != null, "Sender required");
        require(to != null, "Recipient required");
        require(amount > 0, "Amount must be positive");
        require(from != to, "Cannot transfer to self");
        
        // Check balance
        require(balances[from] >= amount, "Insufficient balance");
        
        // Execute transfer
        balances[from] = balances[from] - amount;
        balances[to] = balances[to] + amount;
        
        print "Transfer successful: " + amount;
        return "Success";
    } catch (error) {
        print "Transfer failed";
        return "Failed";
    }
}

safeTransfer("Alice", "Bob", 100);
```

### Example 2: User Authentication and Authorization
```zexus
action authenticateAndExecute(userId, action, permissions) {
    try {
        // Authentication
        require(userId != null, "User ID required");
        require(userId != "", "Invalid user ID");
        
        // Authorization
        let userLevel = permissions[userId];
        require(userLevel >= 2, "Insufficient permissions");
        
        // Action validation
        require(action == "read" || action == "write", "Unknown action");
        
        // Execute
        print "Executing " + action + " for " + userId;
        return "Success";
    } catch (e) {
        print "Authentication or authorization failed";
        return "Denied";
    }
}
```

### Example 3: Data Validation Pipeline
```zexus
action processUserInput(input) {
    try {
        // Stage 1: Null check
        require(input != null, "Input cannot be null");
        
        // Stage 2: Format validation
        require(input != "", "Input cannot be empty");
        
        // Stage 3: Business rules
        require(input != "admin", "Reserved keyword");
        require(input != "system", "Reserved keyword");
        
        // Processing
        let processed = "Processed: " + input;
        
        // Postcondition
        require(processed != null, "Processing failed");
        
        return processed;
    } catch (validationError) {
        return "Input validation failed";
    }
}
```

### Example 4: Complete Workflow with Multiple Error Points
```zexus
action completeWorkflow(userId, operation, data) {
    try {
        // Step 1: Authentication
        require(userId != null, "Authentication failed");
        print "✓ User authenticated";
        
        // Step 2: Authorization
        require(operation != "delete", "Operation not authorized");
        print "✓ User authorized";
        
        // Step 3: Input validation
        require(data != null, "Invalid data");
        require(data != "", "Empty data");
        print "✓ Data validated";
        
        // Step 4: Business logic
        require(operation == "create" || operation == "update", "Unknown operation");
        print "✓ Executing " + operation;
        
        // Step 5: Success
        print "✓ Workflow completed";
        return "Success: " + operation;
    } catch (error) {
        print "✗ Workflow failed";
        return "Failed: Workflow aborted";
    }
}

// Usage
completeWorkflow("user123", "create", "payload data");
completeWorkflow("user123", "delete", "payload");  // Will fail authorization
```

---

## Performance Considerations

### Try-Catch Overhead
Try-catch blocks have minimal overhead in the normal (non-error) path. The performance cost only matters when errors are actually thrown.

**Recommendation**: Use try-catch liberally for robustness without worrying about performance in most cases.

### Require vs If-Revert
`require()` and explicit `if-revert` patterns have similar performance characteristics.

```zexus
// Similar performance
require(condition, "Error");

if (!condition) {
    revert("Error");
}
```

**Recommendation**: Use `require()` for cleaner, more readable code. Use explicit `if-revert` as a workaround when encountering parsing issues.

---

## Testing Error Handling

### Test Structure
```zexus
// Test normal path
try {
    let result = operation();
    print "Normal path: " + result;
} catch (e) {
    print "Should not catch: " + e;
}

// Test error path
try {
    errorOperation();
    print "Should not reach here";
} catch (e) {
    print "Error path: Caught as expected";
}
```

### Testing Patterns

#### Test Multiple Error Conditions
```zexus
action testValidation(value) {
    let testResults = [];
    
    // Test 1: Negative value
    try {
        require(value > 0, "Must be positive");
        testResults = testResults + ["positive: pass"];
    } catch (e) {
        testResults = testResults + ["positive: fail"];
    }
    
    // Test 2: Range check
    try {
        require(value < 100, "Must be < 100");
        testResults = testResults + ["range: pass"];
    } catch (e) {
        testResults = testResults + ["range: fail"];
    }
    
    return testResults;
}
```

---

## Comparison with Other Languages

### JavaScript/TypeScript
```javascript
// JavaScript
try {
    if (!condition) throw new Error("Failed");
    return value;
} catch (error) {
    return defaultValue;
}
```

```zexus
// Zexus equivalent
try {
    require(condition, "Failed");
    return value;
} catch (error) {
    return defaultValue;
}
```

### Solidity (Smart Contracts)
```solidity
// Solidity
function transfer(address to, uint amount) public {
    require(amount > 0, "Amount must be positive");
    require(balances[msg.sender] >= amount, "Insufficient balance");
    
    balances[msg.sender] -= amount;
    balances[to] += amount;
}
```

```zexus
// Zexus equivalent
action transfer(to, amount, sender) {
    require(amount > 0, "Amount must be positive");
    require(balances[sender] >= amount, "Insufficient balance");
    
    balances[sender] = balances[sender] - amount;
    balances[to] = balances[to] + amount;
}
```

### Rust
```rust
// Rust
fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        return Err("Division by zero".to_string());
    }
    Ok(a / b)
}
```

```zexus
// Zexus equivalent
action divide(a, b) {
    try {
        require(b != 0, "Division by zero");
        return a / b;
    } catch (e) {
        return null;
    }
}
```

---

## Summary

### TRY-CATCH
- ✅ **Syntax**: `try { } catch (error) { }`
- ✅ **Basic Functionality**: Works correctly
- ✅ **Nesting**: Supported
- ⚠️ **Warning**: Parser suggests parentheses in catch

### REVERT
- ✅ **Syntax**: `revert()` or `revert("message")`
- ✅ **Basic Functionality**: Works correctly
- ✅ **Dynamic Messages**: Supported
- ⚠️ **Limitation**: Should be used inside try-catch

### REQUIRE
- ✅ **Syntax**: `require(condition)` or `require(condition, "message")`
- ✅ **Basic Functionality**: Works at top level
- ⚠️ **Issue**: May fail as "Not a function" in some contexts
- 🔄 **Workaround**: Use explicit if-revert pattern

### Status
- **TRY/CATCH**: Production-ready with minor syntax warnings
- **REVERT**: Production-ready
- **REQUIRE**: Partially working - context-dependent issues

---

## Future Enhancements

### Potential Improvements
1. **Finally Block**: Add support for cleanup code that always executes
2. **Multiple Catch Blocks**: Handle different error types differently
3. **Error Types**: Structured error objects with type information
4. **Stack Traces**: Better error messages with call stack information
5. **Assert Keyword**: Non-recoverable assertions for development
6. **Custom Error Types**: User-defined error classes

### Proposed Syntax
```zexus
// Future: Finally block
try {
    openResource();
} catch (e) {
    handleError(e);
} finally {
    closeResource();
}

// Future: Typed errors
try {
    riskyOperation();
} catch (ValidationError e) {
    handleValidation(e);
} catch (NetworkError e) {
    handleNetwork(e);
}

// Future: Custom errors
error ValidationError(message, field);
throw ValidationError("Invalid email", "email");
```

---

## Related Keywords
- **IF/ELIF/ELSE**: Conditional logic (alternative to require for control flow)
- **RETURN**: Early returns (alternative to error handling in some cases)
- **DEBUG**: Debugging output (useful for error investigation)

---

*Last Updated: December 16, 2025*
*Tested with Zexus Interpreter*
