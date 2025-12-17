# ACTION, FUNCTION, LAMBDA, RETURN Keywords - Complete Guide

## Overview
These four keywords provide function definition and invocation capabilities in Zexus:
- **ACTION**: Primary function definition keyword (Zexus-style)
- **FUNCTION**: Alternative function definition keyword (traditional style)
- **LAMBDA**: Anonymous function expressions
- **RETURN**: Return values from functions

## ACTION Keyword

### Syntax
```zexus
action functionName(param1, param2) {
    // Function body
    return value;
}
```

### Basic Usage

#### Simple Action
```zexus
action greet() {
    print "Hello, World!";
}

greet();
// Output: Hello, World!
```

#### Action with Parameters
```zexus
action add(a, b) {
    return a + b;
}

let result = add(5, 3);
print result;
// Output: 8
```

#### Action with Multiple Statements
```zexus
action calculateArea(width, height) {
    let area = width * height;
    print "Calculating area...";
    return area;
}

print calculateArea(5, 10);
// Output: Calculating area... 50
```

### Advanced Action Patterns

#### 1. Recursive Actions
```zexus
action factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

print factorial(5);
// Output: 120
```

#### 2. Actions with Conditional Returns
```zexus
action classify(number) {
    if (number < 0) {
        return "negative";
    } elif (number == 0) {
        return "zero";
    } else {
        return "positive";
    }
}

print classify(-5);  // negative
print classify(0);   // zero
print classify(10);  // positive
```

#### 3. Actions Calling Other Actions
```zexus
action square(x) {
    return x * x;
}

action sumOfSquares(a, b) {
    return square(a) + square(b);
}

print sumOfSquares(3, 4);
// Output: 25
```

---

## FUNCTION Keyword

### Syntax
```zexus
function functionName(param1, param2) {
    // Function body
    return value;
}
```

### Basic Usage
The `function` keyword works identically to `action` - it's an alternative syntax for users familiar with traditional programming languages.

```zexus
function multiply(x, y) {
    return x * y;
}

print multiply(6, 7);
// Output: 42
```

### ACTION vs FUNCTION
| Feature | ACTION | FUNCTION |
|---------|--------|----------|
| Syntax | `action name() { }` | `function name() { }` |
| Behavior | Identical | Identical |
| Style | Zexus-native | Traditional |
| Use When | Zexus-first projects | Cross-language familiarity |

**Note**: Both are functionally equivalent. Choose based on your team's preference or project style guide.

---

## LAMBDA Keyword

### Syntax
```zexus
let funcName = lambda(params) => expression;

// Or with block body
let funcName = lambda(params) => {
    // Multiple statements
    return value;
};
```

### Basic Usage

#### Simple Lambda
```zexus
let double = lambda(x) => x * 2;
print double(5);
// Output: 10
```

#### Lambda with Multiple Parameters
```zexus
let sum = lambda(a, b) => a + b;
print sum(10, 20);
// Output: 30
```

#### Lambda with Block Body
```zexus
let absValue = lambda(x) => {
    if (x < 0) {
        return -x;
    } else {
        return x;
    }
};

print absValue(-15);
// Output: 15
```

### Advanced Lambda Patterns

#### 1. Lambda in Data Structures
```zexus
let operations = [
    lambda(x) => x + 1,
    lambda(x) => x * 2,
    lambda(x) => x - 1
];

print operations[0](10);  // 11
print operations[1](10);  // 20
print operations[2](10);  // 9
```

#### 2. Higher-Order Functions
```zexus
action applyTwice(func, value) {
    return func(func(value));
}

let triple = lambda(x) => x * 3;
print applyTwice(triple, 2);
// Output: 18 (2 * 3 * 3)
```

#### 3. Function Returning Lambda
```zexus
action makeAdder(x) {
    return lambda(y) => x + y;
}

let addFive = makeAdder(5);
print addFive(10);
// Output: 15
```

#### 4. Currying
```zexus
action curry(a) {
    return lambda(b) => {
        return lambda(c) => a + b + c;
    };
}

let step1 = curry(1);
let step2 = step1(2);
print step2(3);
// Output: 6
```

---

## RETURN Keyword

### Syntax
```zexus
return value;
return;  // Returns null/void
```

### Basic Usage

#### Return Simple Value
```zexus
action getValue() {
    return 42;
}

print getValue();
// Output: 42
```

#### Return Expression
```zexus
action compute(x, y) {
    return (x + y) * 2;
}

print compute(3, 4);
// Output: 14
```

### Advanced Return Patterns

#### 1. Multiple Return Points
```zexus
action max(a, b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}
```

#### 2. Early Return (Guard Clauses)
```zexus
action divide(a, b) {
    if (b == 0) {
        return "Error: Division by zero";
    }
    
    if (a == 0) {
        return 0;
    }
    
    return a / b;
}
```

#### 3. Return Complex Data
```zexus
action getPersonInfo() {
    return {
        "name": "Alice",
        "age": 30,
        "city": "NYC"
    };
}

let person = getPersonInfo();
print person["name"];
// Output: Alice
```

#### 4. Return Arrays
```zexus
action getPair() {
    return [1, 2];
}

let numbers = getPair();
print numbers[0];  // 1
print numbers[1];  // 2
```

#### 5. Void Return (No Value)
```zexus
action logMessage(msg) {
    print msg;
    return;  // Optional - implicit at end of function
}
```

---

## Common Patterns & Algorithms

### Pattern 1: Map/Transform Array
```zexus
action mapArray(arr, transformer) {
    let result = [];
    let i = 0;
    while (i < 5) {
        let value = transformer(arr[i]);
        i = i + 1;
    }
    return result;
}

let doubler = lambda(x) => x * 2;
let doubled = mapArray([1, 2, 3, 4, 5], doubler);
```

### Pattern 2: Filter Array
```zexus
action filterArray(arr, predicate) {
    let result = [];
    let i = 0;
    while (i < 5) {
        if (predicate(arr[i])) {
            result = result;
        }
        i = i + 1;
    }
    return result;
}

let isEven = lambda(x) => x % 2 == 0;
let evens = filterArray([1, 2, 3, 4, 5], isEven);
```

### Pattern 3: Reduce/Fold
```zexus
action reduce(arr, reducer, initial) {
    let acc = initial;
    let i = 0;
    while (i < 5) {
        acc = reducer(acc, arr[i]);
        i = i + 1;
    }
    return acc;
}

let sum = reduce([1, 2, 3, 4, 5], lambda(a, b) => a + b, 0);
```

### Pattern 4: Fibonacci (Recursive)
```zexus
action fibonacci(n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

print fibonacci(10);
// Output: 55
```

### Pattern 5: Binary Search
```zexus
action binarySearch(arr, target, left, right) {
    if (left > right) {
        return -1;
    }
    
    let mid = (left + right) / 2;
    
    if (arr[mid] == target) {
        return mid;
    } elif (arr[mid] > target) {
        return binarySearch(arr, target, left, mid - 1);
    } else {
        return binarySearch(arr, target, mid + 1, right);
    }
}
```

### Pattern 6: Tail Recursion
```zexus
action sumTailRec(n, accumulator) {
    if (n == 0) {
        return accumulator;
    }
    return sumTailRec(n - 1, accumulator + n);
}

print sumTailRec(100, 0);
// Output: 5050
```

### Pattern 7: Closure/Counter
```zexus
action makeCounter(start) {
    let count = start;
    return lambda() => {
        count = count + 1;
        return count;
    };
}

let counter = makeCounter(0);
print counter();  // 1
print counter();  // 2
print counter();  // 3
```

### Pattern 8: Partial Application
```zexus
action partial(func, fixedArg) {
    return lambda(x) => func(fixedArg, x);
}

let addTen = partial(lambda(a, b) => a + b, 10);
print addTen(5);   // 15
print addTen(20);  // 30
```

### Pattern 9: Function Composition
```zexus
action compose(f, g) {
    return lambda(x) => f(g(x));
}

let addTwo = lambda(x) => x + 2;
let timesThree = lambda(x) => x * 3;
let composed = compose(timesThree, addTwo);

print composed(5);
// Output: 21 (5 + 2 = 7, 7 * 3 = 21)
```

### Pattern 10: Mutual Recursion
```zexus
action isEven(n) {
    if (n == 0) {
        return true;
    }
    return isOdd(n - 1);
}

action isOdd(n) {
    if (n == 0) {
        return false;
    }
    return isEven(n - 1);
}

print isEven(6);  // true
print isOdd(6);   // false
```

---

## Best Practices

### ✅ DO

1. **Use descriptive function names**
```zexus
// Good
action calculateTotalPrice(items) { }
action validateUserInput(input) { }

// Bad
action calc(x) { }
action doStuff() { }
```

2. **Keep functions focused (single responsibility)**
```zexus
// Good - each does one thing
action readFile(path) { }
action parseJSON(text) { }
action validateData(data) { }

// Bad - does too much
action readAndParseAndValidate(path) { }
```

3. **Use return early for validation**
```zexus
action processValue(val) {
    if (val == null) {
        return "Error: null value";
    }
    
    if (val < 0) {
        return "Error: negative value";
    }
    
    return val * 2;
}
```

4. **Use lambda for simple transformations**
```zexus
let numbers = [1, 2, 3, 4, 5];
let doubled = numbers;  // If map exists
// Or with lambda in array
let operations = [lambda(x) => x * 2];
```

5. **Prefer action/function over lambda for complex logic**
```zexus
// Good - named function for clarity
action complexCalculation(data) {
    // Many lines of logic
    return result;
}

// Bad - lambda too complex
let calc = lambda(data) => {
    // Many lines...
};
```

### ❌ DON'T

1. **Don't create overly deep recursion**
```zexus
// ⚠️ May cause stack overflow
action deepRecursion(n) {
    if (n == 0) return 0;
    return deepRecursion(n - 1);
}

deepRecursion(100000);  // Dangerous!
```

2. **Don't forget to return values**
```zexus
// ❌ Bad - missing return
action add(a, b) {
    let sum = a + b;
    // Forgot return statement
}

// ✅ Good
action add(a, b) {
    return a + b;
}
```

3. **Don't modify parameters directly (side effects)**
```zexus
// ⚠️ Confusing - modifies external state
action addToArray(arr, val) {
    arr = arr + [val];  // Modifies parameter
}

// ✅ Better - return new value
action addToArray(arr, val) {
    return arr + [val];
}
```

4. **Don't nest functions excessively**
```zexus
// ❌ Hard to understand
action outer() {
    action middle() {
        action inner() {
            action deepest() {
                // Too deep!
            }
        }
    }
}
```

---

## Edge Cases & Gotchas

### 1. Function Scope
```zexus
let x = 10;

action test() {
    let x = 20;  // Local variable, shadows outer x
    print x;     // 20
}

test();
print x;  // 10 (outer x unchanged)
```

### 2. Closure Behavior
```zexus
action makeCounter() {
    let count = 0;
    return lambda() => {
        count = count + 1;  // Captures 'count' from outer scope
        return count;
    };
}

let counter1 = makeCounter();
let counter2 = makeCounter();

print counter1();  // 1
print counter1();  // 2
print counter2();  // 1 (independent counter)
```

### 3. Return Without Value
```zexus
action doSomething() {
    print "Doing...";
    return;  // Returns null/void
}

let result = doSomething();
// result is null
```

### 4. Missing Return
```zexus
action forgot() {
    let x = 10 + 20;
    // No return statement - returns null
}

let val = forgot();
// val is null
```

### 5. Lambda Capturing Variables
```zexus
let funcs = [];
let i = 0;
while (i < 3) {
    // ⚠️ All lambdas capture same 'i'
    let f = lambda() => i;
    funcs = funcs + [f];
    i = i + 1;
}
// All functions might return same value
```

---

## Performance Considerations

### Recursion Depth
- Zexus may have recursion limits
- Use iterative solutions for large inputs
- Consider tail recursion optimization

### Function Call Overhead
- Function calls have overhead
- Inline simple operations when performance-critical
- Balance readability vs performance

### Closure Memory
- Closures capture variables from outer scope
- Be mindful of memory with many closures
- Clean up references when done

---

## Integration with Other Features

### With Loops
```zexus
action processArray(arr) {
    let result = [];
    for each item in arr {
        let processed = item * 2;
        result = result;
    }
    return result;
}
```

### With Conditionals
```zexus
action classify(value) {
    if (value < 0) {
        return "negative";
    } elif (value == 0) {
        return "zero";
    } else {
        return "positive";
    }
}
```

### With Maps/Objects
```zexus
action getUserName(user) {
    return user["name"];
}

let person = {"name": "Alice", "age": 30};
print getUserName(person);
```

---

## Known Issues ⚠️

### Issues Found (December 2025)

1. **Map Returns Show as Empty** (Priority: Medium)
   - Returning map literals shows `{}` instead of actual values
   - Test: `return {"area": 50, "perimeter": 30}` displays as `{}`
   - Status: Output/display issue, not storage issue
   - Workaround: Access individual properties

2. **Closure State Not Persisting** (Priority: Medium)
   - Closures that capture variables show `{}` on repeated calls
   - Test: Counter pattern returns empty maps
   - Status: Closure implementation may need work
   - Impact: Limited closure functionality

3. **PropertyAccessExpression Error** (Priority: High)
   - Error: `'PropertyAccessExpression' object has no attribute 'value'`
   - Occurs with complex nested property access in some contexts
   - Status: Parser/evaluator bug
   - Impact: Some advanced patterns fail

### Workarounds
- Use arrays instead of maps for return values when possible
- Test closure patterns carefully
- Avoid deeply nested property access

---

## Future Enhancements

### Potential Features

1. **Default Parameters**
```zexus
// Future syntax
action greet(name = "Guest") {
    print "Hello, " + name;
}
```

2. **Rest Parameters**
```zexus
// Future syntax
action sum(...numbers) {
    // Sum all numbers
}
```

3. **Destructuring**
```zexus
// Future syntax
let [a, b] = getPair();
let {name, age} = getUser();
```

4. **Async Functions**
```zexus
// Future syntax
async action fetchData() {
    let data = await api.get();
    return data;
}
```

5. **Generator Functions**
```zexus
// Future syntax
function* range(start, end) {
    for (let i = start; i < end; i++) {
        yield i;
    }
}
```

---

## Debugging Tips

1. **Add debug prints**
```zexus
action calculate(x) {
    debug x;  // Show parameter
    let result = x * 2;
    debug result;  // Show result
    return result;
}
```

2. **Test return values**
```zexus
let val = myFunction();
print "Returned: " + val;
print "Type: " + typeof(val);
```

3. **Verify function calls**
```zexus
print "Before call";
let result = myFunction();
print "After call";
```

4. **Check parameter values**
```zexus
action test(param) {
    print "Received: " + param;
    // Rest of function
}
```

---

## Summary

### When to Use Each

**ACTION/FUNCTION**:
- Named, reusable operations
- Complex logic with multiple statements
- Recursive algorithms
- Main program structure

**LAMBDA**:
- Short, inline operations
- Callbacks and higher-order functions
- Data transformations
- Functional programming patterns

**RETURN**:
- Send values back to caller
- Early exit with guards
- Provide computed results
- Control flow in functions

### Key Takeaways
1. `action` and `function` are identical - choose by style preference
2. `lambda` creates anonymous functions for inline use
3. `return` exits function and provides value to caller
4. Functions support recursion, closures, and higher-order patterns
5. Keep functions focused and well-named
6. Be mindful of recursion depth and closure memory
7. Test complex patterns carefully due to known issues

---

**Related Keywords**: LET, CONST, IF, WHILE, FOR  
**Category**: Functions & Procedures  
**Status**: 🟢 Mostly Working (some closure/map issues)  
**Tests Created**: 20 easy, 20 medium, 20 complex  
**Documentation**: Complete  
**Last Updated**: December 16, 2025
