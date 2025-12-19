# PRINT and DEBUG Keywords - Complete Guide

## Overview
These two keywords provide output and debugging capabilities in Zexus:
- **PRINT**: Output values to console/stdout
- **DEBUG**: Output debug information with metadata

## PRINT Keyword

### Syntax
```zexus
print expression;
```

### Basic Usage

#### Simple Output
```zexus
print "Hello, World!";
// Output: Hello, World!

print 42;
// Output: 42

print true;
// Output: true
```

#### Print Variables
```zexus
let message = "Welcome";
print message;
// Output: Welcome

let count = 10;
print count;
// Output: 10
```

#### Print Expressions
```zexus
print 5 + 3;
// Output: 8

print "Hello, " + "World!";
// Output: Hello, World!

print 10 * (5 + 2);
// Output: 70
```

### Advanced Print Usage

#### 1. Print Data Structures

**Arrays**:
```zexus
let numbers = [1, 2, 3, 4, 5];
print numbers;
// Output: [1, 2, 3, 4, 5]
```

**Maps/Objects**:
```zexus
let person = {"name": "Alice", "age": 30};
print person;
// Output: {name: Alice, age: 30}
```

**Nested Structures**:
```zexus
let data = {
    "users": [1, 2, 3],
    "count": 3,
    "meta": {"version": "1.0"}
};
print data;
// Output: {users: [1, 2, 3], count: 3, meta: {version: 1.0}}
```

#### 2. Print Function Results
```zexus
action calculate(x, y) {
    return x * y;
}

print calculate(5, 7);
// Output: 35
```

#### 3. Print in Loops
```zexus
for each item in [1, 2, 3] {
    print "Item: " + item;
}
// Output: Item: 1, Item: 2, Item: 3
```

#### 4. Print with Conditionals
```zexus
let score = 85;

if (score >= 80) {
    print "Grade: A";
} else {
    print "Grade: B";
}
// Output: Grade: A
```

#### 5. Print Formatted Strings
```zexus
let name = "Bob";
let age = 25;
print "Name: " + name + ", Age: " + age;
// Output: Name: Bob, Age: 25
```

### Print Behavior

#### Type Handling
| Type | Output Format |
|------|---------------|
| String | Plain text |
| Integer | Number |
| Boolean | true/false |
| Null | null |
| Array | [elem1, elem2, ...] |
| Map | {key1: val1, key2: val2, ...} |
| Function | function: name |

#### Multiple Prints
Each `print` statement outputs on its own line:
```zexus
print "Line 1";
print "Line 2";
print "Line 3";
// Output:
// Line 1
// Line 2
// Line 3
```

---

## DEBUG Keyword

### Syntax (DUAL-MODE)

DEBUG supports two modes:

**Function Mode** (returns value):
```zexus
let x = debug(expression);  // Returns the value for use in expressions
print debug(42);            // Can be used anywhere a value is expected
```

**Statement Mode** (logs with metadata):
```zexus
debug expression;  // Logs value with metadata (line numbers, context)
```

### Basic Usage

#### Function Mode (Returns Value)
```zexus
// Use in variable assignment
let x = debug(42);
// Output: [DEBUG] 42
// x now equals 42

// Use in expressions
print debug(10 + 20);
// Output: [DEBUG] 30
//        30

// Chain with other operations
let result = debug(5 * 5) + 10;
// Output: [DEBUG] 25
// result = 35
```

#### Statement Mode (Logs with Metadata)
```zexus
// Simple debug
let x = 42;
debug x;
// Output: 🔍 DEBUG: 42

// Debug expressions
debug 10 + 20;
// Output: 🔍 DEBUG: 30

// Debug strings
debug "Debug message";
// Output: 🔍 DEBUG: Debug message
```

### Advanced Debug Usage

#### 1. Debug in Functions
```zexus
action processData(data) {
    debug "Processing: " + data;
    let result = data * 2;
    debug "Result: " + result;
    return result;
}

processData(10);
// Output: [DEBUG] Processing: 10
//         [DEBUG] Result: 20
```

#### 2. Debug Complex Data
```zexus
let userData = {
    "id": 1,
    "name": "Alice",
    "roles": ["admin", "user"]
};

debug userData;
// Output: [DEBUG] {id: 1, name: Alice, roles: [admin, user]}
```

#### 3. Debug in Loops
```zexus
let i = 0;
while (i < 3) {
    debug "Iteration: " + i;
    i = i + 1;
}
// Output: [DEBUG] Iteration: 0
//         [DEBUG] Iteration: 1
//         [DEBUG] Iteration: 2
```

#### 4. Debug vs Print
```zexus
print "Normal output";
debug "Debug output";
print "Back to normal";

// Output:
// Normal output
// [DEBUG] Debug output
// Back to normal
```

### Debug vs Print Comparison

| Feature | PRINT | DEBUG (Function) | DEBUG (Statement) |
|---------|-------|------------------|------------------|
| **Purpose** | User-facing output | Return value + log | Developer diagnostics |
| **Output Format** | Plain value | [DEBUG] prefix | 🔍 DEBUG: prefix |
| **Returns Value** | No (statement) | Yes | No (statement) |
| **Use in Expressions** | No | Yes | No |
| **When to Use** | Final program output | Debug + use value | Development only |
| **Production** | Keep for user messages | Remove or disable | Remove or disable |
| **Metadata** | None | Basic logging | Enhanced (future) |

---

## Common Patterns

### Pattern 1: Progress Tracking
```zexus
action processItems(items) {
    print "Processing " + 3 + " items...";
    
    for each item in items {
        debug "Processing item: " + item;
        // Process item
    }
    
    print "Complete!";
}
```

### Pattern 2: Error Messages
```zexus
action divide(a, b) {
    if (b == 0) {
        print "Error: Division by zero";
        return null;
    }
    
    let result = a / b;
    debug "Division result: " + result;
    return result;
}
```

### Pattern 3: State Logging
```zexus
let state = "idle";

action changeState(newState) {
    debug "State transition: " + state + " -> " + newState;
    state = newState;
    print "Current state: " + state;
}
```

### Pattern 4: Data Inspection
```zexus
action inspectData(data) {
    debug "Data type check";
    debug data;
    
    print "Data processed";
    return data;
}
```

### Pattern 5: Verbose Mode Simulation
```zexus
let verbose = true;

action log(message) {
    if (verbose) {
        debug message;
    }
}

action process() {
    log("Starting process...");
    // Do work
    log("Process complete");
    print "Done";
}
```

### Pattern 6: Dual-Mode DEBUG Usage
```zexus
// Function mode - use return value
action calculate(x) {
    let doubled = debug(x * 2);  // Logs and returns value
    return doubled;
}

// Statement mode - just log
action validate(data) {
    debug data;  // Logs with metadata
    if (data > 0) {
        return true;
    }
    return false;
}

// Mixed usage
action process(input) {
    debug input;                    // Statement: logs input
    let normalized = debug(input * 1.5);  // Function: logs and returns
    debug normalized;               // Statement: logs result
    return normalized;
}
```

---

## Best Practices

### ✅ DO

1. **Use print for user-facing output**
```zexus
// Good
print "Welcome to the application";
print "Result: " + result;
```

2. **Use debug for development**
```zexus
// Good - helps during development
debug "Function called with: " + param;
debug "Internal state: " + internalValue;
```

3. **Format print output clearly**
```zexus
// Good
print "Name: " + name + ", Age: " + age;
print "Total: " + total;

// Less clear
print name;
print age;
```

4. **Debug intermediate values**
```zexus
// Good
let step1 = calculate1(x);
debug "Step 1: " + step1;

let step2 = calculate2(step1);
debug "Step 2: " + step2;

return step3;
```

### ❌ DON'T

1. **Don't print sensitive data**
```zexus
// ❌ Bad
print "Password: " + password;
print userCreditCard;
```

2. **Don't spam print in loops**
```zexus
// ❌ Bad - too much output
let i = 0;
while (i < 1000) {
    print i;  // Prints 1000 times!
    i = i + 1;
}
```

3. **Don't mix debug in production**
```zexus
// ❌ Bad for production
action processPayment(amount) {
    debug "Processing $" + amount;  // Should remove before deploy
    // Process payment
}
```

4. **Don't print without context**
```zexus
// ❌ Bad - unclear output
print x;
print y;
print z;

// ✅ Good - clear context
print "X coordinate: " + x;
print "Y coordinate: " + y;
print "Z coordinate: " + z;
```

---

## Edge Cases & Gotchas

### 1. Printing Null
```zexus
let value = null;
print value;
// Output: null
```

### 2. Printing Empty Structures
```zexus
print [];
print {};
// Output: [], {}
```

### 3. Printing Functions
```zexus
action test() {
    return 42;
}

print test;
// Output: function: test (or similar)
```

### 4. Print Order in Async (if applicable)
```zexus
// Prints may appear in unexpected order with async
print "Start";
// async operation
print "End";
```

### 5. Debugging Complex Expressions
```zexus
// This prints the result, not the expression
debug 10 + 20 * 3;
// Output: [DEBUG] 70 (not "10 + 20 * 3")
```

---

## Integration with Other Features

### With Functions
```zexus
action greet(name) {
    let message = "Hello, " + name;
    print message;
    return message;
}

greet("Alice");
```

### With Conditionals
```zexus
if (condition) {
    print "Condition is true";
    debug condition;
} else {
    print "Condition is false";
}
```

### With Loops
```zexus
for each item in items {
    debug "Processing: " + item;
    print "Done with: " + item;
}
```

### With Error Handling
```zexus
try {
    // Operation
    print "Success";
} catch (error) {
    print "Error occurred";
    debug error;
}
```

---

## Performance Considerations

1. **Print has minimal overhead** - suitable for production
2. **Debug may have additional overhead** - consider disabling in production
3. **Printing large structures** can be slow - consider formatting
4. **Loop printing** can create massive output - use sparingly

---

## Debugging Tips

1. **Use debug to trace execution**
```zexus
debug "Entering function";
// Function code
debug "Exiting function";
```

2. **Print intermediate values**
```zexus
let step1 = calc1();
print "Step 1: " + step1;

let step2 = calc2(step1);
print "Step 2: " + step2;
```

3. **Debug conditionals**
```zexus
debug "Checking condition: " + (x > 5);
if (x > 5) {
    print "Condition true";
}
```

4. **Trace data flow**
```zexus
action process(input) {
    debug "Input: " + input;
    let output = transform(input);
    debug "Output: " + output;
    return output;
}
```

---

## Known Issues ⚠️

### Issues Found (December 2025)

1. **~~Debug May Require Parentheses~~** ✅ **RESOLVED** (December 18, 2025)
   - **Status**: DUAL-MODE implementation complete
   - **Solution**: Both `debug(expr)` and `debug expr;` now work
   - **Function Mode**: `debug(42)` returns value, usable in expressions
   - **Statement Mode**: `debug x;` logs with metadata
   - **Implementation**: Parser detects parentheses and routes accordingly

2. **Print in Loops Sometimes Skipped** (Priority: High)
   - Related to loop execution issues (WHILE/FOR bugs)
   - Print statements inside loops may not execute
   - Status: Loop execution problem, not print itself
   - Impact: Debugging loops is difficult

---

## Future Enhancements

### Potential Features

1. **Print Formatting**
```zexus
// Future syntax
print("Value: {}", x);
print("Name: {}, Age: {}", name, age);
```

2. **Debug Levels**
```zexus
// Future syntax
debug.info("Info message");
debug.warn("Warning message");
debug.error("Error message");
```

3. **Conditional Debug**
```zexus
// Future syntax
debug.when(condition, "Message");
```

4. **Pretty Print**
```zexus
// Future syntax
print.pretty(complexObject);
```

5. **Print to File**
```zexus
// Future syntax
print.to("output.txt", message);
```

---

## Summary

### When to Use Each

**PRINT**:
- User-facing output
- Program results
- Status messages
- Final output
- Production code

**DEBUG**:
- Development diagnostics
- Tracing execution
- Inspecting values
- Troubleshooting
- Remove before production

### Key Takeaways
1. `print` is for users, `debug` is for developers
2. Both accept any expression or value
3. Print outputs one value per line
4. Debug adds [DEBUG] prefix for visibility
5. Use sparingly in loops to avoid output spam
6. Remove or disable debug statements in production
7. Format output with string concatenation for clarity

---

**Related Keywords**: STRING, CONCAT (for formatting)  
**Category**: I/O Operations  
**Status**: ✅ Fully Working (print fully functional, debug dual-mode complete)  
**Tests Created**: 20 easy, 20 medium, 20 complex  
**Documentation**: Complete  
**Last Updated**: December 18, 2025

### DEBUG Implementation Details

**Dual-Mode System**:
- **Function Mode**: `debug(x)` - Parser returns Identifier, creates CallExpression
- **Statement Mode**: `debug x;` - Parser creates DebugStatement
- **Detection**: Parser checks if DEBUG token followed by LPAREN
- **Files Modified**:
  * Lexer: Restored DEBUG keyword (lexer.py:380)
  * Parser: Dual-mode logic (parser.py:813-831)
  * Context Strategy: Call expression support (strategy_context.py:2267)
  * Structural Analyzer: Assignment RHS support (strategy_structural.py:416)
  * Evaluator: Function returns value, statement logs with metadata
- **Test Files**: test_debug_minimal.zx, test_debug_statement.zx, test_debug_dual.zx
