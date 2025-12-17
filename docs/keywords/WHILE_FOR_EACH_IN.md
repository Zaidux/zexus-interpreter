# WHILE, FOR, EACH, IN Keywords - Complete Guide

> **✅ STATUS UPDATE (December 17, 2025)**: These keywords are now **FULLY FUNCTIONAL**! Previous critical parsing bugs have been fixed. All loop types (while, for-each) now work correctly with proper iteration, variable reassignment, and nested loops.

## Overview
These four keywords work together to provide iteration and looping capabilities in Zexus:
- **WHILE**: Executes a block repeatedly while a condition is true ✅
- **FOR**: Keyword used with EACH to create for-each loops ✅
- **EACH**: Specifies iteration over a collection ✅
- **IN**: Connects the iterator variable to the iterable collection ✅

Together they form two main loop constructs: `while` loops and `for each...in` loops.

---

## WHILE Keyword

### Syntax
```zexus
while (condition) {
    // Loop body
}
```

### Basic Usage

#### Simple Counter Loop
```zexus
let counter = 0;
while (counter < 5) {
    print counter;
    counter = counter + 1;
}
// Output: 0, 1, 2, 3, 4
```

#### Countdown Loop
```zexus
let countdown = 10;
while (countdown > 0) {
    print countdown;
    countdown = countdown - 1;
}
// Output: 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
```

### Advanced While Patterns

#### 1. While with Complex Conditions
```zexus
let x = 0;
let y = 10;
while (x < 5 && y > 5) {
    print "x: " + x + ", y: " + y;
    x = x + 1;
    y = y - 1;
}
```

#### 2. While with Early Termination
```zexus
let found = false;
let index = 0;
let items = [1, 2, 3, 4, 5];

while (index < 5 && !found) {
    if (items[index] == 3) {
        print "Found at index: " + index;
        found = true;
    }
    index = index + 1;
}
```

#### 3. Infinite Loop with Break Logic
```zexus
let running = true;
let count = 0;

while (running) {
    print count;
    count = count + 1;
    
    if (count >= 10) {
        running = false;
    }
}
```

#### 4. While with State Machine
```zexus
let state = "start";
let iterations = 0;

while (iterations < 5) {
    if (state == "start") {
        print "Starting...";
        state = "running";
    } elif (state == "running") {
        print "Processing...";
        state = "done";
    } else {
        print "Complete";
    }
    iterations = iterations + 1;
}
```

### While Use Cases

1. **Counting**: Traditional counter-based loops
2. **Search**: Find elements with early termination
3. **Accumulation**: Build sums, products, or aggregations
4. **State Machines**: Manage complex state transitions
5. **Polling**: Check conditions repeatedly
6. **Game Loops**: Run until game over condition

---

## FOR, EACH, IN Keywords

### Syntax
```zexus
for each variable in iterable {
    // Loop body
}
```

**Note**: These three keywords must be used together. You cannot use FOR without EACH and IN.

### Basic Usage

#### Iterate Over Array
```zexus
for each num in [1, 2, 3, 4, 5] {
    print num;
}
// Output: 1, 2, 3, 4, 5
```

#### Iterate Over String Array
```zexus
for each name in ["Alice", "Bob", "Charlie"] {
    print "Hello, " + name;
}
// Output: Hello, Alice; Hello, Bob; Hello, Charlie
```

#### Process Each Element
```zexus
for each value in [10, 20, 30] {
    let doubled = value * 2;
    print doubled;
}
// Output: 20, 40, 60
```

### Advanced For Each Patterns

#### 1. Accumulation Pattern
```zexus
let total = 0;
for each num in [1, 2, 3, 4, 5] {
    total = total + num;
}
print "Sum: " + total;
// Output: Sum: 15
```

#### 2. Filtering Pattern
```zexus
let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
for each num in numbers {
    if (num % 2 == 0) {
        print num + " is even";
    }
}
// Output: 2 is even, 4 is even, 6 is even, 8 is even, 10 is even
```

#### 3. Transformation Pattern
```zexus
let prices = [10, 20, 30];
for each price in prices {
    let withTax = price * 1.08;
    print "Price with tax: " + withTax;
}
```

#### 4. Nested Iteration (Matrix)
```zexus
let matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];

for each row in matrix {
    for each cell in row {
        print cell;
    }
}
// Output: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

#### 5. Multiple Level Nesting
```zexus
let layer1 = [1, 2];
let layer2 = [10, 20];
let layer3 = [100, 200];

for each a in layer1 {
    for each b in layer2 {
        for each c in layer3 {
            print a + b + c;
        }
    }
}
// Output: 111, 211, 121, 221, 112, 212, 122, 222
```

### For Each Use Cases

1. **Data Processing**: Transform every element in a collection
2. **Aggregation**: Sum, average, min, max calculations
3. **Filtering**: Select elements matching criteria
4. **Validation**: Check all elements meet requirements
5. **Batch Operations**: Perform same action on multiple items
6. **Matrix Operations**: Work with 2D/3D data structures

---

## Comparison: WHILE vs FOR EACH

| Feature | WHILE | FOR EACH |
|---------|-------|----------|
| **Use Case** | Condition-based repetition | Iterate over collection |
| **Syntax** | `while (condition) { }` | `for each item in array { }` |
| **Index Control** | Manual (you manage counter) | Automatic (handled by loop) |
| **Iteration Count** | Unknown beforehand | Fixed by collection size |
| **Early Exit** | Check condition | Iterate all elements* |
| **Best For** | Dynamic conditions, state machines | Array/list processing |

*Note: Early exit with for-each typically requires a flag variable

---

## Common Patterns & Algorithms

### Pattern 1: FizzBuzz
```zexus
let num = 1;
while (num <= 15) {
    if (num % 15 == 0) {
        print "FizzBuzz";
    } elif (num % 3 == 0) {
        print "Fizz";
    } elif (num % 5 == 0) {
        print "Buzz";
    } else {
        print num;
    }
    num = num + 1;
}
```

### Pattern 2: Find Maximum
```zexus
let numbers = [42, 17, 93, 26, 88, 11];
let max = numbers[0];

for each num in numbers {
    if (num > max) {
        max = num;
    }
}

print "Maximum: " + max;
```

### Pattern 3: Count Occurrences
```zexus
let items = [1, 2, 3, 2, 1, 2, 4];
let target = 2;
let count = 0;

for each item in items {
    if (item == target) {
        count = count + 1;
    }
}

print "Found " + count + " occurrences";
```

### Pattern 4: Factorial Calculation
```zexus
let n = 5;
let factorial = 1;
let counter = 1;

while (counter <= n) {
    factorial = factorial * counter;
    counter = counter + 1;
}

print factorial;  // Output: 120
```

### Pattern 5: Prime Number Check
```zexus
let testNum = 17;
let isPrime = true;
let divisor = 2;

while (divisor < testNum) {
    if (testNum % divisor == 0) {
        isPrime = false;
    }
    divisor = divisor + 1;
}

print isPrime;  // Output: true
```

### Pattern 6: Fibonacci Sequence
```zexus
let fib1 = 0;
let fib2 = 1;
let count = 0;

print fib1;
print fib2;

while (count < 8) {
    let next = fib1 + fib2;
    print next;
    fib1 = fib2;
    fib2 = next;
    count = count + 1;
}
// Output: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
```

### Pattern 7: Bubble Sort
```zexus
let arr = [5, 2, 8, 1, 9];
let len = 5;
let i = 0;

while (i < len) {
    let j = 0;
    while (j < len - 1) {
        if (arr[j] > arr[j + 1]) {
            let temp = arr[j];
            arr[j] = arr[j + 1];
            arr[j + 1] = temp;
        }
        j = j + 1;
    }
    i = i + 1;
}

for each element in arr {
    print element;
}
// Output: 1, 2, 5, 8, 9
```

### Pattern 8: Running Average
```zexus
let data = [10, 20, 30, 40, 50];
let sum = 0;
let count = 0;

for each value in data {
    sum = sum + value;
    count = count + 1;
    let average = sum / count;
    print "Average: " + average;
}
```

---

## Edge Cases & Gotchas

### 1. Empty Collections
```zexus
// For each on empty array - no iterations
for each item in [] {
    print item;
}
// Nothing prints - this is correct behavior
```

### 2. Infinite While Loops
```zexus
// ⚠️ DANGER: Infinite loop!
let x = 0;
while (x < 10) {
    print x;
    // Forgot to increment x!
}

// ✅ FIXED: Always update loop variable
let x = 0;
while (x < 10) {
    print x;
    x = x + 1;
}
```

### 3. Immediate False Condition
```zexus
let x = 10;
while (x < 5) {
    print "This never executes";
}
print "Loop skipped";  // This prints
```

### 4. Off-by-One Errors
```zexus
// Wrong: Misses 5
let i = 0;
while (i < 5) {
    print i;
    i = i + 1;
}
// Output: 0, 1, 2, 3, 4 (correct for < 5)

// Right: Includes 5
let i = 1;
while (i <= 5) {
    print i;
    i = i + 1;
}
// Output: 1, 2, 3, 4, 5
```

### 5. Variable Scope in Loops
```zexus
for each num in [1, 2, 3] {
    let doubled = num * 2;
    print doubled;
}
// doubled is NOT accessible here (loop scope)
print doubled;  // Error: doubled not defined
```

### 6. Modifying Loop Variable in For Each
```zexus
// The iterator variable shouldn't be reassigned
for each num in [1, 2, 3] {
    num = num * 2;  // ⚠️ Doesn't affect array
    print num;       // Prints 2, 4, 6
}
// Original array unchanged
```

---

## Best Practices

### ✅ DO

1. **Use meaningful iterator names**
```zexus
// Good
for each student in students { }
for each price in prices { }

// Bad
for each x in list { }
```

2. **Initialize counters before while loops**
```zexus
let counter = 0;
while (counter < 10) {
    // ...
    counter = counter + 1;
}
```

3. **Use for-each for simple iteration**
```zexus
// Preferred
for each item in items {
    print item;
}

// Less clear
let i = 0;
while (i < 4) {
    print items[i];
    i = i + 1;
}
```

4. **Keep loop bodies focused**
```zexus
// Each loop should do one thing well
for each num in numbers {
    let sum = sum + num;
}
```

### ❌ DON'T

1. **Don't forget to update loop variables**
```zexus
// ❌ Infinite loop
let x = 0;
while (x < 10) {
    print x;  // x never changes!
}
```

2. **Don't nest loops excessively (>3 levels)**
```zexus
// ❌ Too deep - hard to understand
for each a in list1 {
    for each b in list2 {
        for each c in list3 {
            for each d in list4 {  // Too much!
```

3. **Don't use while when for-each is clearer**
```zexus
// ❌ Unclear
let i = 0;
while (i < arr.length) {
    process(arr[i]);
    i = i + 1;
}

// ✅ Clear
for each item in arr {
    process(item);
}
```

4. **Don't modify collection during iteration**
```zexus
// ⚠️ Potentially dangerous
let list = [1, 2, 3];
for each item in list {
    list = list + [item * 2];  // Modifying what we're iterating!
}
```

---

## Performance Considerations

### Loop Overhead
- **While loops**: Minimal overhead, condition check per iteration
- **For each loops**: Automatic iteration, no manual index management
- **Nested loops**: O(n²) or worse - use with caution

### Optimization Tips

1. **Hoist invariant calculations**
```zexus
// ❌ Bad: Recalculates every iteration
while (i < count) {
    let limit = calculateLimit();  // Same every time!
    if (i < limit) { }
}

// ✅ Good: Calculate once
let limit = calculateLimit();
while (i < count) {
    if (i < limit) { }
}
```

2. **Minimize work in condition**
```zexus
// ❌ Expensive condition
while (complexCalculation() < limit) { }

// ✅ Cache result
let result = complexCalculation();
while (result < limit) {
    // Update result as needed
}
```

3. **Use early exit when possible**
```zexus
let found = false;
let i = 0;
while (i < 1000 && !found) {
    if (items[i] == target) {
        found = true;  // Stop searching
    }
    i = i + 1;
}
```

---

## Integration with Other Features

### With Functions
```zexus
action processArray(arr) {
    let sum = 0;
    for each num in arr {
        sum = sum + num;
    }
    return sum;
}

let total = processArray([1, 2, 3, 4, 5]);
print total;  // Output: 15
```

### With Conditionals
```zexus
for each num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] {
    if (num % 2 == 0) {
        print num + " is even";
    } elif (num % 3 == 0) {
        print num + " is divisible by 3";
    } else {
        print num;
    }
}
```

### With Maps/Objects
```zexus
let userData = {"name": "Alice", "age": 30, "city": "NYC"};
let keys = ["name", "age", "city"];

for each key in keys {
    print key + ": " + userData[key];
}
```

---

## Known Issues ⚠️

### ✅ Fixed Issues (December 17, 2025)

1. **~~Loop Bodies Not Executing Properly~~** ✅ **FIXED**
   - **Issue**: Statements inside loop bodies failed to execute properly
   - **Root Cause**: Missing WHILE and FOR handlers in `_parse_block_statements()` function
   - **Fix**: Added explicit parsing handlers in `src/zexus/parser/strategy_context.py` (lines 1614-1755)
   - **Status**: ✅ FULLY RESOLVED - All loop types now work correctly
   - **Verification**: 
     * Print statements execute correctly ✅
     * Variable reassignments work properly ✅
     * All iterations complete as expected ✅
     * Conditions without parentheses parse correctly (e.g., `while i < 10`) ✅
     * Conditions with parentheses parse correctly (e.g., `while (i < 10)`) ✅

2. **~~Incomplete Iteration~~** ✅ **FIXED**
   - **Status**: RESOLVED - Was a symptom of the parsing issue
   - **Verification**: For-each loops iterate over all elements correctly

3. **~~Silent Failures~~** ✅ **FIXED**
   - **Status**: RESOLVED - Loops now parse and execute without silent failures
   - **Verification**: Debug logging shows proper parsing and execution flow

### Current Status: ✅ FULLY FUNCTIONAL
- WHILE loops work correctly with all condition types
- FOR EACH loops properly iterate over arrays and collections
- Variable reassignment in loops works as expected
- Nested loops function properly
- Complex conditions and expressions are handled correctly
- No known issues remain for these keywords

---

## Future Enhancements

### Potential Features

1. **Break and Continue**
```zexus
// Future syntax
for each item in items {
    if (item < 0) continue;
    if (item > 100) break;
    process(item);
}
```

2. **Loop Labels**
```zexus
// Future syntax
outer: while (x < 10) {
    inner: for each item in items {
        break outer;  // Break outer loop
    }
}
```

3. **Range Syntax**
```zexus
// Future syntax
for each i in 0..10 {
    print i;
}
```

4. **Loop with Index**
```zexus
// Future syntax
for each (item, index) in items {
    print index + ": " + item;
}
```

5. **Parallel Iteration**
```zexus
// Future syntax
for each item in items parallel {
    process(item);  // Process in parallel
}
```

---

## Debugging Tips

1. **Add debug prints**
```zexus
while (counter < 10) {
    debug counter;  // Shows counter value and type
    counter = counter + 1;
}
```

2. **Verify loop entry**
```zexus
print "Before loop";
while (condition) {
    print "Inside loop";
}
print "After loop";
```

3. **Check iteration count**
```zexus
let iterations = 0;
for each item in items {
    iterations = iterations + 1;
}
print "Total iterations: " + iterations;
```

4. **Validate conditions**
```zexus
print "Condition result: " + (x < 10);
while (x < 10) {
    // ...
}
```

---

## Summary

### When to Use WHILE
- Condition-based repetition
- Unknown number of iterations
- State machines
- Search with early exit
- Dynamic termination conditions

### When to Use FOR EACH
- Iterate over known collection
- Process every element
- Simple array/list traversal
- No need for index manipulation
- Cleaner, more readable code

### Key Takeaways
1. `while` loops are condition-driven, `for each` loops are collection-driven
2. Always update loop variables to avoid infinite loops
3. Use `for each` for simple iterations, `while` for complex conditions
4. Be mindful of scope - loop variables are local to the loop body
5. Test edge cases: empty collections, immediate false conditions
6. Avoid excessive nesting (max 2-3 levels)

---

**Related Keywords**: BREAK (future), CONTINUE (future), IF, ELIF, ELSE  
**Category**: Control Flow - Loops  
**Status**: ❌ BROKEN - Critical execution issues (December 2025)  
**Priority**: CRITICAL - Core language feature  
**Tests Created**: 20 easy, 20 medium, 15 complex  
**Documentation**: Complete  
**Last Updated**: December 16, 2025
