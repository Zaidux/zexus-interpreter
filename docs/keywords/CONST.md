# CONST Keyword - Complete Guide

## Overview
The `const` keyword in Zexus is used to declare **immutable (constant) variables**. Once a value is assigned to a `const` variable, it **cannot be reassigned**. This makes your code more predictable and helps prevent accidental modifications.

## Syntax

Zexus supports **three syntax variations** for constant declaration:

```zexus
const variable_name = value;              // Standard assignment
const variable_name : value;              // Colon syntax (alternative)
const variable_name : Type = value;       // Type annotation with assignment
```

**All three syntaxes are fully supported as of December 18, 2025!**

**Note**: The semicolon at the end is optional in Zexus.

## Basic Usage

### Simple Constant Declaration
```zexus
const PI = 3.14159;
const APP_NAME = "MyApp";
const MAX_USERS = 1000;
const IS_PRODUCTION = true;
```

### Immutability Enforcement
```zexus
const x = 10;
x = 20;    // ❌ Error: Cannot reassign const variable 'x'
```

This is the key difference from `let` - **const variables cannot be reassigned**.

## When to Use CONST

### ✅ Use CONST for:
- **Configuration values** that shouldn't change
- **Mathematical constants** (PI, E, etc.)
- **Fixed limits** and thresholds
- **Function results** that won't be modified
- **References** that should remain stable
- **API keys** and other secrets (in combination with security features)

### ❌ Don't Use CONST for:
- Values that need to change (use `let` instead)
- Loop counters and accumulators
- State that updates over time
- Temporary calculation variables that get reassigned

## Type Annotations (Optional)

Like `let`, `const` supports optional type annotations:

```zexus
const age: integer = 25;
const username: string = "admin";
const isEnabled: bool = true;
```

### Supported Types
- `integer` or `int` - Whole numbers
- `string` or `str` - Text values
- `bool` or `boolean` - True/False values
- `float` - Decimal numbers
- Custom types and type aliases

## Advanced Features

### 1. Expression Assignment
Constants can be assigned from any expression:

```zexus
const total = 10 + 20 + 30;
const message = "Hello, " + "World!";
const result = (5 * 3) + (10 / 2);
```

### 2. Function Return Values
```zexus
action calculate(a, b) {
    return a * b;
}

const product = calculate(5, 10);    // product = 50 (immutable)
```

### 3. Complex Data Structures
```zexus
const numbers = [1, 2, 3, 4, 5];
const config = {"theme": "dark", "size": 16};
const matrix = [[1, 2], [3, 4], [5, 6]];
```

**Important Note**: While the variable binding is immutable, the contents of arrays and objects may still be modifiable depending on the implementation. The constant refers to the *reference*, not necessarily deep immutability of nested structures.

### 4. Conditional Assignment
```zexus
const status = age >= 18 ? "adult" : "minor";
const value = userInput ?? "default_value";    // Nullish coalescing
```

### 5. Lambda Functions
```zexus
const multiply = lambda(a, b) => a * b;
const result = multiply(4, 5);    // result = 20
```

## Comparison with LET

| Feature | CONST | LET |
|---------|-------|-----|
| Mutability | ❌ Cannot be reassigned | ✅ Can be reassigned |
| When to use | Fixed values | Values that change |
| Safety | Higher (prevents accidents) | Standard |
| Performance | May optimize better | Standard |
| Best for | Configuration, constants | Counters, state |

**Example:**
```zexus
const immutableVar = 10;
immutableVar = 20;  // ❌ Error: Cannot reassign const variable

let mutableVar = 10;
mutableVar = 20;    // ✅ Allowed
```

## Scope Rules ⚠️ IMPORTANT

**Zexus uses FUNCTION-LEVEL SCOPING, not block-level scoping!**

Constants follow the same scoping rules as `let` variables:

### Function Scope ✅
Functions create new scopes:

```zexus
action example() {
    const localConst = "I'm local";
    print localConst;    // ✅ Works
}

example();
print localConst;    // ❌ Error: localConst is not defined
```

### Block Scope ❌
**Blocks do NOT create new scopes - attempting to declare const with same name causes error:**

```zexus
const x = 10;

if (true) {
    const x = 20;  // ❌ Error: "Cannot reassign const variable 'x'"
}
```

**This is because blocks share the same scope, so redeclaring const x is seen as reassignment!**

**Workaround - Use different names:**

```zexus
const outerValue = 10;

if (true) {
    const innerValue = 20;  // ✅ Works - different name
    print innerValue;
}
```

**Or use a function:**

```zexus
const x = 10;

action test() {
    const x = 20;  // ✅ Works - function creates new scope
    print x;       // 20
}

test();
print x;  // 10 (unchanged)
```

### Loop Scope ❌
**Loops do NOT create new scopes:**

```zexus
for each i in [1, 2, 3] {
    const loopConst = i * 2;  // ❌ Error on second iteration!
    // Trying to redeclare loopConst in same scope
}
```

## Shadowing Behavior ⚠️ BY DESIGN

**Const variables cannot be shadowed in blocks/loops because Zexus uses function-level scoping!**

### Why This Happens

Blocks and IF/WHILE/FOR statements **do NOT create new scopes** in Zexus. They all share the same environment as their parent scope. Therefore:

```zexus
const x = 10;

if (true) {
    const x = 20;    // ❌ Error: "Cannot reassign const variable 'x'"
    // This is seen as trying to redeclare x in the SAME scope!
}
```

### Shadowing ONLY Works in Functions

```zexus
const x = 10;

action test() {
    const x = 20;  // ✅ Works - function creates new scope
    print x;       // 20
}

test();
print x;  // 10 (unchanged)
```

### Workarounds

**Option 1: Use different variable names:**

```zexus
const outerValue = 10;

if (true) {
    const innerValue = 20;    // ✅ Works
}
```

**Option 2: Wrap in a function:**

```zexus
const x = 10;

action process() {
    const x = 20;  // ✅ Creates new scope
    // Do work with local x
}

process();
```

**This behavior is consistent across LET and CONST - it's a core Zexus design decision.**

## Common Patterns

### 1. Configuration Constants
```zexus
const CONFIG = {
    "api_url": "https://api.example.com",
    "timeout": 5000,
    "retry_attempts": 3,
    "debug_mode": false
};

const API_URL = CONFIG["api_url"];
const TIMEOUT = CONFIG["timeout"];
```

### 2. Mathematical Constants
```zexus
const PI = 3.14159265;
const E = 2.71828;
const GOLDEN_RATIO = 1.618;

action calculateCircleArea(radius) {
    return PI * radius * radius;
}
```

### 3. Enum-like Values
```zexus
const STATUS_PENDING = "pending";
const STATUS_APPROVED = "approved";
const STATUS_REJECTED = "rejected";

const currentStatus = STATUS_PENDING;
```

### 4. Function Factories
```zexus
const makeMultiplier = action(factor) {
    return lambda(x) => x * factor;
};

const double = makeMultiplier(2);
const triple = makeMultiplier(3);

print double(10);    // Output: 20
print triple(10);    // Output: 30
```

### 5. Immutable Results
```zexus
const userData = fetchUserFromAPI();
const processedData = transformData(userData);
const finalResult = validateAndFormat(processedData);
```

## Edge Cases & Gotchas

### 1. Must Initialize
```zexus
// ❌ Error: Must initialize const
const x;

// ✅ Correct: Always provide a value
const x = 0;
const x = null;
```

### 2. No Reassignment
```zexus
const count = 0;
count = count + 1;    // ❌ Error: Cannot reassign const
```

If you need to update a value, use `let` instead:

```zexus
let count = 0;
count = count + 1;    // ✅ Works
```

### 3. Cannot Shadow with Same Name
```zexus
const x = 10;

if (true) {
    const x = 20;    // ❌ Error in Zexus
}
```

### 4. Reference vs Deep Immutability
```zexus
const arr = [1, 2, 3];
// The binding 'arr' cannot change, but...
// Array/object content mutability depends on implementation
```

## Best Practices

### ✅ DO
- Use UPPERCASE for true constants: `const MAX_SIZE = 100;`
- Use `const` by default, `let` when you need mutability
- Group related constants together
- Use descriptive names: `const USER_TIMEOUT_MS = 5000;`
- Use `const` for configuration objects
- Use `const` for function references

### ❌ DON'T
- Use `const` for values that need to change
- Try to reassign const variables
- Shadow const variables with the same name
- Use generic names: `const x = 10;`
- Mix `const` and `let` unnecessarily

## Performance Considerations

1. **Optimization**: Compilers can better optimize `const` since the value won't change
2. **Memory**: Same as `let` - no additional overhead
3. **Safety**: Prevents accidental bugs from reassignment
4. **Readability**: Makes intent clear - this value is fixed

## Integration with Other Features

### With Functions
```zexus
const greet = action(name) {
    return "Hello, " + name + "!";
};

const greeting = greet("Alice");
print greeting;
```

### With Async/Await
```zexus
async action fetchData() {
    const data = await getFromAPI();
    return data;
}

const result = await fetchData();
```

### With Pattern Matching
```zexus
const value = 42;

pattern value {
    case 42 => print "The answer!";
    case _ => print "Something else";
}
```

### With Error Handling
```zexus
try {
    const result = riskyOperation();
    print result;
} catch (error) {
    const errorMsg = "Failed: " + error;
    print errorMsg;
}
```

## Debugging Tips

1. **Use DEBUG statement**:
```zexus
const x = 10;
debug x;    // Shows variable info and const status
```

2. **Check immutability**:
```zexus
const value = 42;
// Try to reassign and catch the error
try {
    value = 50;
} catch (e) {
    print "Correctly immutable!";
}
```

3. **Verify scope**:
```zexus
const global = "global";

action test() {
    const local = "local";
    print global;    // ✅ Can access
    print local;     // ✅ Can access
}

test();
print global;    // ✅ Can access
print local;     // ❌ Error: not defined
```

## Real-World Examples

### 1. API Configuration
```zexus
const API_CONFIG = {
    "base_url": "https://api.myapp.com/v1",
    "api_key": "abc123",
    "endpoints": {
        "users": "/users",
        "posts": "/posts",
        "auth": "/auth"
    }
};

action fetchUsers() {
    const url = API_CONFIG["base_url"] + API_CONFIG["endpoints"]["users"];
    return fetch(url);
}
```

### 2. Game Constants
```zexus
const PLAYER_MAX_HEALTH = 100;
const PLAYER_SPEED = 5;
const GRAVITY = 9.8;

const playerConfig = {
    "health": PLAYER_MAX_HEALTH,
    "speed": PLAYER_SPEED,
    "jumpForce": 10
};
```

### 3. Calculation Helpers
```zexus
const calculateTax = lambda(amount, rate) => amount * rate;
const formatCurrency = lambda(amount) => "$" + amount;

const price = 100;
const taxRate = 0.08;
const tax = calculateTax(price, taxRate);
const total = price + tax;

print formatCurrency(total);    // Output: $108
```

### 4. Data Processing Pipeline
```zexus
const rawData = fetchFromDatabase();
const filteredData = filterActive(rawData);
const mappedData = mapToDTO(filteredData);
const sortedData = sortByDate(mappedData);
const finalData = formatForDisplay(sortedData);
```

## Comparison Table: CONST vs LET

| Aspect | CONST | LET |
|--------|-------|-----|
| **Syntax** | `const x = 10;` | `let x = 10;` |
| **Reassignment** | ❌ Not allowed | ✅ Allowed |
| **Shadowing** | ❌ Not allowed (same name) | ✅ Allowed |
| **Scope** | Block/Lexical | Block/Lexical |
| **Initialization** | Required | Required |
| **Use Case** | Fixed values | Changing values |
| **Safety** | High | Medium |
| **Performance** | May optimize | Standard |
| **Best For** | Config, constants | Counters, state |

## Future Enhancements

### Potential Upgrades
1. **Deep Freeze**: Make nested structures truly immutable
2. **Compile-Time Constants**: Evaluate at compile time for better performance
3. **Const Assertions**: `const x = [1, 2, 3] as readonly;`
4. **Destructuring**: `const [a, b, c] = [1, 2, 3];`

### Patches Needed
- Allow shadowing in nested scopes (or document why it's disallowed)
- Better error messages for reassignment attempts
- Warnings for unused constants
- Deep immutability for arrays and objects

## File Reading with << Operator

**New in v3.0**: The `<<` operator enables reading file contents directly into **immutable** constants!

### Syntax

```zexus
const variable << "filepath";       // Read file as immutable string
const config << "config.json";      // Configuration files
const template << "template.txt";   // Templates and data
```

### Basic Usage

```zexus
// Read a configuration file (immutable)
const appConfig << "config.json";
print(appConfig);

// appConfig = "something else";  // ❌ Error: Cannot reassign const

// Read a template (immutable)
const emailTemplate << "email.txt";
print(emailTemplate);
```

### Why Use const << Instead of let <<?

**Use `const <<`** when the file content should never change:
- ✅ Configuration files that shouldn't be modified
- ✅ Templates that remain constant
- ✅ Reference data that's read-only
- ✅ Secrets and API keys (immutable)

**Use `let <<`** when you might modify the content:
- Data that gets transformed
- Templates with placeholder replacement
- Content that's processed and updated

### Features

- **Immutable**: Cannot reassign after reading
- **Any File Type**: Works with .txt, .json, .zx, .py, .cpp, .js, etc.
- **Path Support**: Relative and absolute paths
- **Type Safety**: Content is always a string

### Comparison Table

| Feature | `const code << file` | `let code << file` | `log << file` |
|---------|---------------------|-------------------|---------------|
| **Mutability** | ❌ Immutable | ✅ Mutable | N/A (execution) |
| **Purpose** | Read as constant | Read as variable | Execute code |
| **File Types** | Any extension | Any extension | .zx only |
| **Reassignment** | Not allowed | Allowed | N/A |
| **Use Case** | Config, templates | Data processing | Hidden layers |

### Example: Immutable Configuration

```zexus
// Load configuration as immutable
const dbConfig << "database.json";
const apiKeys << "secrets.json";

// Parse and use
const db = JSON.parse(dbConfig);
const keys = JSON.parse(apiKeys);

// These cannot be reassigned
// dbConfig = "new value";  // ❌ Error: Cannot reassign const

print("Database: " + db.host);
print("API Key loaded: " + (keys.api_key ? "yes" : "no"));
```

### Example: Template Constants

```zexus
// Load email templates as constants
const welcomeEmail << "templates/welcome.txt";
const resetEmail << "templates/password_reset.txt";

// Use templates (no modification to const itself)
action sendWelcome(name) {
    let email = welcomeEmail;  // Copy to mutable variable
    email = email.replace("{{NAME}}", name);
    sendEmail(email);
}

sendWelcome("Alice");
```

### Error Handling

File not found errors:

```zexus
const data << "missing.txt";  // ❌ Error: File not found
```

Reassignment errors:

```zexus
const content << "file.txt";
content = "new content";  // ❌ Error: Cannot reassign const variable 'content'
```

### Security Benefits

Using `const <<` for sensitive data provides extra safety:

```zexus
// API keys loaded as constants (cannot be accidentally changed)
const apiKeys << "secrets.json";
const credentials << "auth.json";

// These are now protected from reassignment
// apiKeys = "{}";  // ❌ Blocked by const!
```

## Summary

The `const` keyword is essential for writing safe, maintainable Zexus code. It declares variables that cannot be reassigned, making your intentions clear and preventing accidental modifications.

### Key Takeaways:
- ✅ Use `const` for values that won't change
- ✅ Use `let` for values that need to be updated
- ❌ Cannot reassign const variables
- ❌ Cannot shadow const with same name in nested scope
- 🎯 Const makes code more predictable and safer

**Rule of Thumb**: Start with `const` by default, switch to `let` only when you need mutability.

---

**Related Keywords**: LET, IMMUTABLE, ASSIGN, IDENT, LOG  
**Category**: Variable Declaration  
**Status**: ✅ Fully Implemented  
**Last Updated**: December 24, 2025

### Recent Updates
- ✅ **Dec 24, 2025**: Added `const << file` operator for immutable file reading
- ✅ **Dec 18, 2025**: Added colon syntax support: `const x : 42;`
- ✅ **Dec 18, 2025**: Documented function-level scoping behavior
- ✅ **Dec 18, 2025**: Clarified why block-level shadowing doesn't work
- ✅ **Dec 18, 2025**: Added workarounds and best practices
- ✅ **Dec 18, 2025**: Updated all syntax variations with type annotations
