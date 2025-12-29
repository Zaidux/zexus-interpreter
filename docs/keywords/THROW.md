# THROW Keyword Documentation

## Overview

The `THROW` keyword provides **explicit error throwing** in Zexus, allowing you to raise errors intentionally with custom messages. This is essential for:

- **Error Signaling**: Explicitly indicate error conditions in your code
- **Validation Failures**: Reject invalid inputs with descriptive messages
- **Early Returns**: Exit functions with error states
- **API Error Handling**: Provide meaningful error messages to callers
- **Debugging**: Create intentional errors during development

## Syntax

```zexus
throw expression;
```

The `THROW` statement takes an expression (typically a string) that becomes the error message.

## Behavior

### Basic Error Throwing

```zexus
throw "Something went wrong";
```

This creates an `EvaluationError` with the message "Something went wrong" and propagates it up the call stack.

### With Variables

```zexus
let errorMsg = "Invalid operation"
throw errorMsg;
```

### With Expressions

```zexus
let code = 404
throw "Error code: " + string(code);
// Throws: "Error code: 404"
```

## Key Features

### 1. Immediate Error Propagation
When `THROW` is executed:
- An **EvaluationError** is created with the provided message
- Execution **stops immediately** in the current context
- The error **propagates up** the call stack
- Can be **caught by try-catch** blocks

### 2. Integration with Try-Catch

```zexus
try {
    let value = -1
    if value < 0 {
        throw "Value cannot be negative"
    }
    print("Value is valid: " + string(value))
} catch (error) {
    print("Caught error: " + error)
}
```

**Output:**
```
Caught error: Value cannot be negative
```

### 3. Use in Functions

```zexus
action validateAge(age) {
    if age < 0 {
        throw "Age cannot be negative"
    }
    if age > 150 {
        throw "Age is unrealistic"
    }
    return age
}

try {
    let userAge = validateAge(-5)
    print("Age: " + string(userAge))
} catch (e) {
    print("Validation error: " + e)
}
```

**Output:**
```
Validation error: Age cannot be negative
```

## Usage Examples

### Example 1: Input Validation

```zexus
action divide(a, b) {
    if b == 0 {
        throw "Division by zero is not allowed"
    }
    return a / b
}

try {
    let result = divide(10, 0)
    print("Result: " + string(result))
} catch (error) {
    print("Error: " + error)
}
```

**Output:**
```
Error: Division by zero is not allowed
```

### Example 2: API Error Handling

```zexus
action fetchUser(userId) {
    if userId < 1 {
        throw "Invalid user ID: must be positive"
    }
    
    let user = database.getUser(userId)
    
    if user == null {
        throw "User not found: ID " + string(userId)
    }
    
    return user
}

try {
    let user = fetchUser(0)
    print("User: " + user.name)
} catch (e) {
    print("Failed to fetch user: " + e)
}
```

### Example 3: Nested Error Handling

```zexus
action processData(data) {
    if data == null {
        throw "Data is null"
    }
    
    try {
        if data.value < 0 {
            throw "Data value is negative"
        }
        return data.value * 2
    } catch (innerError) {
        print("Inner error caught: " + innerError)
        throw "Failed to process data"
    }
}

try {
    let result = processData({value: -10})
    print("Result: " + string(result))
} catch (outerError) {
    print("Outer error: " + outerError)
}
```

**Output:**
```
Inner error caught: Data value is negative
Outer error: Failed to process data
```

### Example 4: Guard Clauses Pattern

```zexus
action createUser(name, email, age) {
    // Guard clauses using throw
    if name == null || name == "" {
        throw "Name is required"
    }
    
    if email == null || !is_email(email) {
        throw "Valid email is required"
    }
    
    if age < 18 {
        throw "User must be at least 18 years old"
    }
    
    // All validation passed
    return {
        name: name,
        email: email,
        age: age,
        created_at: timestamp()
    }
}

try {
    let user = createUser("", "invalid", 15)
    print("User created: " + user.name)
} catch (validationError) {
    print("Validation failed: " + validationError)
}
```

**Output:**
```
Validation failed: Name is required
```

### Example 5: Smart Contract Usage

```zexus
contract TokenContract {
    state balances: Map<string, integer>
    
    action transfer(from, to, amount) {
        // Validate sender has sufficient balance
        let senderBalance = balances.get(from, 0)
        
        if senderBalance < amount {
            throw "Insufficient balance: have " + string(senderBalance) + ", need " + string(amount)
        }
        
        if amount <= 0 {
            throw "Transfer amount must be positive"
        }
        
        if to == null || to == "" {
            throw "Invalid recipient address"
        }
        
        // Perform transfer
        balances[from] = senderBalance - amount
        balances[to] = balances.get(to, 0) + amount
        
        emit Transfer(from, to, amount)
    }
}

try {
    let contract = TokenContract()
    contract.transfer("alice", "bob", 100)
} catch (error) {
    print("Transfer failed: " + error)
}
```

### Example 6: Error Recovery with Continue

```zexus
# Enable error recovery mode
continue;

action processItem(item) {
    if item < 0 {
        throw "Negative item: " + string(item)
    }
    return item * 2
}

let items = [1, -2, 3, -4, 5]

for each item in items {
    try {
        let result = processItem(item)
        print("Processed: " + string(result))
    } catch (e) {
        print("Error (continuing): " + e)
    }
}

print("All items processed despite errors!")
```

**Output:**
```
Processed: 2
Error (continuing): Negative item: -2
Processed: 6
Error (continuing): Negative item: -4
Processed: 10
All items processed despite errors!
```

## Rules and Best Practices

### ✅ DO:

1. **Provide descriptive error messages**
   ```zexus
   // ✅ Good
   throw "Invalid email format: " + email
   
   // ❌ Bad
   throw "Error"
   ```

2. **Use throw for exceptional conditions**
   ```zexus
   // ✅ Good - exceptional condition
   if fileNotFound {
       throw "File not found: " + filename
   }
   
   // ❌ Bad - normal control flow
   if userLoggedIn {
       throw "User is logged in"  // Use return instead
   }
   ```

3. **Combine with try-catch for recovery**
   ```zexus
   try {
       riskyOperation()
   } catch (e) {
       print("Error: " + e)
       // Fallback logic here
   }
   ```

4. **Include context in error messages**
   ```zexus
   throw "Failed to parse user ID: " + rawInput + " (expected integer)"
   ```

### ❌ DON'T:

1. **Don't use throw for normal control flow**
   ```zexus
   // ❌ Bad
   action findItem(items, target) {
       for each item in items {
           if item == target {
               throw "Found"  // Use return instead
           }
       }
   }
   ```

2. **Don't throw without context**
   ```zexus
   // ❌ Bad
   throw "Error"
   
   // ✅ Good
   throw "Database connection failed: timeout after 30s"
   ```

3. **Don't use throw when validate/verify is better**
   ```zexus
   // ❌ Less ideal
   if !condition {
       throw "Condition failed"
   }
   
   // ✅ Better for validation
   verify condition, "Condition failed"
   ```

## Comparison with Other Error Keywords

### THROW vs REVERT

| Feature | THROW | REVERT |
|---------|-------|--------|
| Purpose | General error throwing | Transaction/state reversal |
| Context | Any code | Smart contracts |
| Side Effects | None | Reverts state changes |
| Use Case | Input validation, errors | Blockchain transactions |

```zexus
# THROW - general errors
action processInput(data) {
    if data == null {
        throw "Data is null"  # Just throws error
    }
}

# REVERT - contract state reversal
contract Token {
    action transfer(to, amount) {
        if balance < amount {
            revert("Insufficient balance")  # Reverts state
        }
    }
}
```

### THROW vs REQUIRE

| Feature | THROW | REQUIRE |
|---------|-------|---------|
| Style | Explicit if-check + throw | Implicit condition check |
| Guard Clauses | Manual if statements | Built-in assertion |
| Tolerance | No built-in tolerance | Supports tolerance blocks |

```zexus
# THROW - explicit
action validateAge(age) {
    if age < 18 {
        throw "Must be 18 or older"
    }
}

# REQUIRE - implicit
action validateAgeRequire(age) {
    require age >= 18, "Must be 18 or older"
}
```

### THROW vs Error Return Values

| Feature | THROW | Return Error |
|---------|-------|--------------|
| Propagation | Automatic | Manual checking |
| Stack Unwinding | Yes | No |
| Try-Catch | Works with try-catch | Custom error handling |

```zexus
# THROW - automatic propagation
action divide(a, b) {
    if b == 0 {
        throw "Division by zero"
    }
    return a / b
}

# Error return - manual checking
action divideWithErrorReturn(a, b) {
    if b == 0 {
        return {error: "Division by zero"}
    }
    return {result: a / b}
}
```

## Error Handling Patterns

### Pattern 1: Fail-Fast Validation

```zexus
action processOrder(order) {
    # Multiple guard clauses
    if order == null {
        throw "Order is null"
    }
    if order.items.length == 0 {
        throw "Order has no items"
    }
    if order.total <= 0 {
        throw "Order total must be positive"
    }
    
    # Process order
    return placeOrder(order)
}
```

### Pattern 2: Specific Error Types

```zexus
action connectDatabase(config) {
    if config == null {
        throw "CONFIG_NULL"
    }
    if !config.host {
        throw "CONFIG_MISSING_HOST"
    }
    if !config.port {
        throw "CONFIG_MISSING_PORT"
    }
    
    # Connection logic
}

try {
    connectDatabase(null)
} catch (error) {
    if error == "CONFIG_NULL" {
        print("Please provide configuration")
    } elif error == "CONFIG_MISSING_HOST" {
        print("Please specify database host")
    } else {
        print("Connection error: " + error)
    }
}
```

### Pattern 3: Error Wrapping

```zexus
action loadUserData(userId) {
    try {
        let rawData = fetchFromDatabase(userId)
        return parseUserData(rawData)
    } catch (dbError) {
        throw "Failed to load user " + string(userId) + ": " + dbError
    }
}
```

## Edge Cases & Gotchas

### 1. Throwing Non-String Values

```zexus
throw 404;  # Will be converted to string: "404"

throw {code: 500, message: "Server error"};  # Converted to string representation
```

### 2. Throw in Loops with Error Recovery

```zexus
continue;  # Enable error recovery

let i = 0
while i < 5 {
    if i == 2 {
        throw "Error at iteration " + string(i)  # Logged but continues
    }
    print("Iteration: " + string(i))
    i = i + 1
}
print("Loop completed despite error")
```

### 3. Throw vs Break

```zexus
# THROW - error propagation
for each item in items {
    if item.isCorrupt {
        throw "Corrupt data found"  # Error propagates
    }
}

# BREAK - loop exit
for each item in items {
    if item.isTarget {
        break  # Just exits loop
    }
}
```

## Implementation Details

### Files Modified
- `src/zexus/zexus_token.py` - Added THROW token
- `src/zexus/lexer.py` - Added "throw" keyword recognition
- `src/zexus/zexus_ast.py` - Added ThrowStatement AST node
- `src/zexus/parser/parser.py` - Added parse_throw_statement()
- `src/zexus/parser/strategy_structural.py` - Added THROW to statement_starters
- `src/zexus/parser/strategy_context.py` - Added THROW handler in _parse_block_statements
- `src/zexus/evaluator/core.py` - Added ThrowStatement dispatch
- `src/zexus/evaluator/statements.py` - Added eval_throw_statement()

### How It Works Internally

1. **Parsing**: `throw` keyword creates a `ThrowStatement` AST node with message expression
2. **Evaluation**: `eval_throw_statement()` evaluates the message expression
3. **Error Creation**: Creates an `EvaluationError` with the message and stack trace
4. **Propagation**: Returns the error object which propagates up the call stack
5. **Try-Catch**: Can be caught by try-catch blocks for error recovery

## Compatibility

- **Version:** Zexus v1.6.0+
- **Breaking Changes:** None
- **Backward Compatible:** Yes (new keyword)
- **Platform:** All supported platforms

## Summary

The `THROW` keyword is essential for explicit error handling in Zexus:

✅ **Explicit error signaling** - Clearly indicate error conditions
✅ **Descriptive messages** - Provide context with error messages
✅ **Try-catch integration** - Works seamlessly with exception handling
✅ **Guard clauses** - Perfect for input validation patterns
✅ **Stack trace support** - Includes call stack information
✅ **Expression-based** - Can throw any expression as a message

Use `THROW` for:
- Input validation failures
- Exceptional error conditions  
- Early exits on invalid state
- Guard clauses in functions
- Smart contract error handling

**Related Keywords**: TRY, CATCH, REVERT, REQUIRE, VERIFY  
**Category**: Error Handling  
**Status**: ✅ Fully Implemented  
**Documentation**: Complete  
**Last Updated**: December 29, 2025
