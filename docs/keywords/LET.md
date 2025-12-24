# LET Keyword - Complete Guide

## Overview
The `let` keyword in Zexus is used to declare **mutable variables**. Variables declared with `let` can be reassigned to different values after their initial declaration.

## Syntax

Zexus supports **three syntax variations** for variable declaration:

```zexus
let variable_name = value;              // Standard assignment
let variable_name : value;              // Colon syntax (alternative)
let variable_name : Type = value;       // Type annotation with assignment
```

**All three syntaxes are fully supported as of December 18, 2025!**

**Note**: The semicolon at the end is optional in Zexus.

## Basic Usage

### Simple Variable Declaration
```zexus
let x = 10;
let name = "Alice";
let isActive = true;
```

### Variable Reassignment
Since `let` creates mutable variables, you can change their values:

```zexus
let counter = 0;
counter = counter + 1;    // counter is now 1
counter = 100;             // counter is now 100
```

## Type Annotations (Optional)

Zexus supports optional type annotations for variables:

```zexus
let age: integer = 25;
let username: string = "john_doe";
let isLoggedIn: bool = false;
```

### Supported Types
- `integer` or `int` - Whole numbers
- `string` or `str` - Text values
- `bool` or `boolean` - True/False values
- `float` - Decimal numbers
- Custom types and type aliases

## Advanced Features

### 1. Expression Assignment
You can assign the result of any expression to a variable:

```zexus
let sum = 10 + 20 + 30;
let greeting = "Hello, " + "World!";
let result = (5 * 3) + (10 / 2);
```

### 2. Function Return Values
```zexus
action calculate(a, b) {
    return a + b;
}

let total = calculate(5, 10);    // total = 15
```

### 3. Complex Data Structures
```zexus
let numbers = [1, 2, 3, 4, 5];
let person = map("name": "Bob", "age": 30);
let matrix = [[1, 2], [3, 4], [5, 6]];
```

### 4. Conditional Assignment
```zexus
let status = age >= 18 ? "adult" : "minor";
let value = userInput ?? "default_value";    // Nullish coalescing
```

### 5. Lambda Functions
```zexus
let multiply = lambda(a, b) => a * b;
let result = multiply(4, 5);    // result = 20
```

## Comparison with CONST

| Feature | LET | CONST |
|---------|-----|-------|
| Mutability | ✅ Can be reassigned | ❌ Cannot be reassigned |
| When to use | Values that change | Fixed values |
| Performance | Standard | May optimize better |

**Example:**
```zexus
let mutableVar = 10;
mutableVar = 20;    // ✅ Allowed

const immutableVar = 10;
immutableVar = 20;  // ❌ Error: Cannot reassign const variable
```

## Scope Rules ⚠️ IMPORTANT

**Zexus uses FUNCTION-LEVEL SCOPING, not block-level scoping!**

This is a key design decision that makes Zexus different from many modern languages:

### Function Scope ✅
Functions create new scopes:

```zexus
action example() {
    let localVar = "I'm local";
    print localVar;    // ✅ Works
}

example();
print localVar;    // ❌ Error: localVar is not defined
```

### Block Scope ❌
**Blocks do NOT create new scopes:**

```zexus
let x = 10;
print x;  // 10

if (true) {
    let x = 20;  // ⚠️ This REASSIGNS the outer x, doesn't shadow it!
    print x;     // 20
}

print x;  // 20 (the value changed!)
```

**To shadow a variable, use a function:**

```zexus
let x = 10;
print x;  // 10

action test() {
    let x = 20;  // ✅ This creates a NEW variable in function scope
    print x;     // 20
}

test();
print x;  // 10 (outer x unchanged)
```

### Loop Scope ❌
**Loops also do NOT create new scopes:**

```zexus
let i = 100;

for each i in [1, 2, 3] {
    print i;  // 1, 2, 3 (overwrites outer i)
}

print i;  // 3 (i was modified by the loop!)
```

### Why This Matters

**DO:**
- Use functions to create isolated scopes
- Use unique variable names in the same scope
- Be aware that blocks/loops share the outer scope

**DON'T:**
- Expect block-level scoping like in JavaScript, Python, or Rust
- Reuse variable names in nested blocks expecting shadowing
- Assume loop variables are local to the loop

## Common Patterns

### 1. Counter Pattern
```zexus
let counter = 0;

action increment() {
    counter = counter + 1;
}

increment();
increment();
print counter;    // Output: 2
```

### 2. Accumulator Pattern
```zexus
let total = 0;

for each num in [1, 2, 3, 4, 5] {
    total = total + num;
}

print total;    // Output: 15
```

### 3. State Management
```zexus
let currentState = "idle";

action updateState(newState) {
    currentState = newState;
    print "State changed to: " + currentState;
}

updateState("loading");
updateState("ready");
```

### 4. Temporary Variables
```zexus
action swap(a, b) {
    let temp = a;
    a = b;
    b = temp;
    return [a, b];
}
```

## Edge Cases & Gotchas

### 1. Uninitialized Variables
```zexus
// ❌ Error: Must initialize variable
let x;    // Not allowed in Zexus

// ✅ Correct: Always provide a value
let x = null;
let x = 0;
```

### 2. Shadowing ⚠️
**Shadowing only works in FUNCTIONS, not blocks!**

```zexus
// ❌ DOES NOT WORK - blocks don't create scopes
let x = 10;
if (true) {
    let x = 20;      // This REASSIGNS outer x!
    print x;         // Output: 20
}
print x;            // Output: 20 (x was changed!)

// ✅ WORKS - functions create scopes
let x = 10;
action test() {
    let x = 20;      // This creates a new variable
    print x;         // Output: 20
}
test();
print x;            // Output: 10 (outer x unchanged)
```

### 3. Type Mismatch (with type annotations)
```zexus
let age: integer = 25;
age = "twenty-five";    // ❌ Error: Type mismatch
```

### 4. Redeclaration in Same Scope
```zexus
let x = 10;
let x = 20;    // ⚠️ May cause issues - avoid redeclaring in same scope
```

## Best Practices

### ✅ DO
- Use descriptive variable names: `let userAge = 25;`
- Initialize with meaningful values
- Use `let` when the value will change
- Group related variable declarations together
- Use type annotations for clarity in complex code

### ❌ DON'T
- Use single-letter names (except in loops): `let x = 10;`
- Leave variables uninitialized
- Use `let` for values that never change (use `const` instead)
- Redeclare variables in the same scope
- Mix assignment operators (stick to `=`, not `:`)

## Performance Considerations

1. **Memory**: Each `let` variable allocates memory
2. **Scope**: Variables are freed when they go out of scope
3. **Reassignment**: Changing values is fast (no new allocation for primitives)
4. **Type Checking**: Type annotations add minimal overhead

## Integration with Other Features

### With Async/Await
```zexus
async action fetchData() {
    let data = await getFromAPI();
    return data;
}
```

### With Pattern Matching
```zexus
let value = 42;

pattern value {
    case 42 => print "The answer!";
    case _ => print "Something else";
}
```

### With Error Handling
```zexus
try {
    let result = riskyOperation();
    print result;
} catch (error) {
    let errorMsg = "Failed: " + error;
    print errorMsg;
}
```

## Debugging Tips

1. **Use DEBUG statement**:
```zexus
let x = 10;
debug x;    // Shows variable info
```

2. **Track changes**:
```zexus
let counter = 0;
print "Initial: " + counter;
counter = counter + 1;
print "After increment: " + counter;
```

3. **Check type**:
```zexus
let value = 42;
print typeof(value);    // Check the type
```

## Future Enhancements

### Potential Upgrades
1. **Destructuring**: `let [a, b, c] = [1, 2, 3];`
2. **Multiple declarations**: `let x = 1, y = 2, z = 3;`
3. **Default values**: `let x = y || 10;`
4. **Const assertion**: `let x = 10 as const;`

### Patches Needed
- Better error messages for type mismatches
- Warning for unused variables
- Automatic type inference improvements
- Performance optimization for large arrays/maps

## Examples from Real Use Cases

### 1. Game Score Tracking
```zexus
let playerScore = 0;
let highScore = 100;

action addPoints(points) {
    playerScore = playerScore + points;
    
    if (playerScore > highScore) {
        highScore = playerScore;
        print "New high score!";
    }
}
```

### 2. API Response Handling
```zexus
action processResponse(response) {
    let status = response.status;
    let data = response.data;
    let errorMsg = response.error ?? "No error";
    
    if (status == 200) {
        return data;
    } else {
        print "Error: " + errorMsg;
        return null;
    }
}
```

### 3. Data Transformation
```zexus
let numbers = [1, 2, 3, 4, 5];
let doubled = [];

for each num in numbers {
    let result = num * 2;
    doubled = doubled + [result];
}

print doubled;    // Output: [2, 4, 6, 8, 10]
```

## File Reading with << Operator

**New in v3.0**: The `<<` operator enables reading file contents directly into variables!

### Syntax

```zexus
let variable << "filepath";       // Read file contents as string
let code << "script.zx";          // Works with any file extension
let data << "data.json";          // JSON, text, code, etc.
```

### Basic Usage

```zexus
// Read a text file
let content << "README.txt";
print(content);

// Read JSON data
let jsonData << "config.json";
print(jsonData);

// Read code from another file
let helperCode << "helpers.zx";
print(helperCode);
```

### Features

- **Any File Type**: Works with .txt, .json, .zx, .py, .cpp, .js, etc.
- **Automatic Reading**: File is read and stored as a string
- **Mutable Storage**: Since it uses `let`, the variable can be reassigned
- **Path Support**: Relative and absolute paths supported

### Comparison with LOG <<

| Feature | `let code << file` | `log << file` |
|---------|-------------------|---------------|
| **Purpose** | Read file as string | Execute code from file |
| **File Types** | Any extension | .zx files only |
| **Result** | String variable | Code execution |
| **Use Case** | Data loading | Hidden code layers |

### Example: Template Loading

```zexus
// Load email template
let template << "email_template.txt";

// Replace placeholders
let email = template;
email = email.replace("{{NAME}}", "Alice");
email = email.replace("{{DATE}}", "2025-12-24");

print(email);
```

### Example: Configuration Loading

```zexus
// Load configuration
let configJson << "config.json";
let config = JSON.parse(configJson);

print("Theme: " + config.theme);
print("Version: " + config.version);
```

### Error Handling

If the file doesn't exist or can't be read, an error is raised:

```zexus
let data << "missing.txt";  // ❌ Error: File not found
```

**Best Practice**: Always ensure the file exists before reading, or use try-catch blocks.

## Summary

The `let` keyword is fundamental to Zexus programming. It creates mutable variables that can be reassigned, making it perfect for:
- Counters and accumulators
- State management
- Temporary calculations
- Values that change over time

Remember: Use `let` for mutable data and `const` for immutable data. This makes your code more predictable and easier to maintain.

---

**Related Keywords**: CONST, ASSIGN, IDENT, TYPE_ALIAS, LOG  
**Category**: Variable Declaration  
**Status**: ✅ Fully Implemented  
**Last Updated**: December 24, 2025

### Recent Updates
- ✅ **Dec 24, 2025**: Added `let << file` operator for file reading
- ✅ **Dec 18, 2025**: Added colon syntax support: `let x : 42;`
- ✅ **Dec 18, 2025**: Documented function-level scoping behavior
- ✅ **Dec 18, 2025**: Clarified shadowing limitations (functions only)
- ✅ **Dec 18, 2025**: Updated all syntax variations with type annotations
