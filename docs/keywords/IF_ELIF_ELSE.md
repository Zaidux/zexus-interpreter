# IF / ELIF / ELSE Keywords - Complete Guide

## Overview
The `if`, `elif`, and `else` keywords provide **conditional execution** in Zexus. They allow your program to make decisions and execute different code paths based on conditions.

## Syntax

### Basic IF
```zexus
if (condition) {
    // code to execute if condition is true
}

// Alternative: parentheses optional
if condition {
    // code to execute
}
```

### IF-ELSE
```zexus
if (condition) {
    // code if true
} else {
    // code if false
}
```

### IF-ELIF-ELSE
```zexus
if (condition1) {
    // code if condition1 is true
} elif (condition2) {
    // code if condition2 is true
} elif (condition3) {
    // code if condition3 is true
} else {
    // code if all conditions are false
}
```

## Basic Usage

### Simple IF Statement
```zexus
let age = 25;

if (age >= 18) {
    print "Adult";
}
```

### IF-ELSE
```zexus
let score = 75;

if (score >= 60) {
    print "Pass";
} else {
    print "Fail";
}
```

### IF-ELIF-ELSE
```zexus
let grade = 85;

if (grade >= 90) {
    print "A";
} elif (grade >= 80) {
    print "B";
} elif (grade >= 70) {
    print "C";
} else {
    print "F";
}
```

## Conditions

### Comparison Operators
```zexus
if (x == y)   // Equal to
if (x != y)   // Not equal to
if (x < y)    // Less than
if (x > y)    // Greater than
if (x <= y)   // Less than or equal
if (x >= y)   // Greater than or equal
```

### Logical Operators
```zexus
if (a && b)   // AND: both must be true
if (a || b)   // OR: at least one must be true
if (!a)       // NOT: inverts the condition
```

### Combined Conditions
```zexus
if (age >= 18 && hasLicense) {
    print "Can drive";
}

if (isWeekend || isHoliday) {
    print "Day off";
}

if ((x < y) && (y < z) || (x == 10)) {
    print "Complex condition";
}
```

## Advanced Features

### 1. Nested IF Statements
```zexus
if (age > 18) {
    if (hasLicense) {
        print "Can drive";
    } else {
        print "Need license";
    }
} else {
    print "Too young";
}
```

### 2. Multiple ELIF Clauses
You can have as many `elif` clauses as needed:

```zexus
let dayNum = 3;

if (dayNum == 1) {
    print "Monday";
} elif (dayNum == 2) {
    print "Tuesday";
} elif (dayNum == 3) {
    print "Wednesday";
} elif (dayNum == 4) {
    print "Thursday";
} elif (dayNum == 5) {
    print "Friday";
} elif (dayNum == 6) {
    print "Saturday";
} elif (dayNum == 7) {
    print "Sunday";
} else {
    print "Invalid day";
}
```

### 3. Conditions Without Parentheses
Parentheses are optional in Zexus:

```zexus
if true {
    print "Works without parentheses";
}

if x > 10 {
    print "Also works";
}
```

### 4. Functions with Conditional Returns
```zexus
action checkSign(num) {
    if (num > 0) {
        return "positive";
    } elif (num < 0) {
        return "negative";
    } else {
        return "zero";
    }
}
```

### 5. Guard Clauses Pattern
```zexus
action validateUser(user) {
    if (user == null) {
        return "Error: Null user";
    }
    
    if (user["age"] < 18) {
        return "Error: Too young";
    }
    
    if (!user["active"]) {
        return "Error: Inactive";
    }
    
    return "Valid user";
}
```

## Truthiness in Zexus

Values are evaluated as follows:

### Truthy Values
- `true` - Boolean true
- Non-zero numbers - `1`, `-5`, `3.14`
- Non-empty strings - `"hello"`
- Non-empty arrays - `[1, 2, 3]`
- Non-empty maps - `{"key": "value"}`

### Falsy Values
- `false` - Boolean false
- `0` - Zero
- `null` - Null value
- Empty strings - `""`  (may vary)
- Empty arrays - `[]` (may vary)

**Example:**
```zexus
if (0) {
    print "Won't print";
}

if (1) {
    print "Will print";
}

if (null) {
    print "Won't print";
}
```

## Common Patterns

### 1. Range Checking
```zexus
let temperature = 75;

if (temperature < 32) {
    print "Freezing";
} elif (temperature < 60) {
    print "Cold";
} elif (temperature < 80) {
    print "Comfortable";
} else {
    print "Hot";
}
```

### 2. FizzBuzz Pattern
```zexus
for each num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
    if (num % 15 == 0) {
        print "FizzBuzz";
    } elif (num % 3 == 0) {
        print "Fizz";
    } elif (num % 5 == 0) {
        print "Buzz";
    } else {
        print num;
    }
}
```

### 3. Min/Max Functions
```zexus
action max(a, b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

action min(a, b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}
```

### 4. State Machine
```zexus
let state = "idle";

if (state == "idle") {
    print "Starting";
    state = "running";
} elif (state == "running") {
    print "Processing";
    state = "done";
} elif (state == "done") {
    print "Finished";
}
```

### 5. Input Validation
```zexus
action validateInput(value) {
    if (value == null) {
        return false;
    }
    
    if (value < 0) {
        return false;
    }
    
    if (value > 100) {
        return false;
    }
    
    return true;
}
```

## Nested Conditions

### Deep Nesting Example
```zexus
let level1 = true;
let level2 = true;
let level3 = true;

if (level1) {
    if (level2) {
        if (level3) {
            print "All three levels true";
        }
    }
}
```

**Alternative: Flatten with AND**
```zexus
if (level1 && level2 && level3) {
    print "All three levels true";
}
```

### Decision Tree
```zexus
action categorize(age, income) {
    if (age < 25) {
        if (income < 30000) {
            return "Young, Low Income";
        } else {
            return "Young, High Income";
        }
    } elif (age < 50) {
        if (income < 50000) {
            return "Mid Age, Low Income";
        } else {
            return "Mid Age, High Income";
        }
    } else {
        return "Senior";
    }
}
```

## Integration with Other Features

### With Loops
```zexus
for each i in [1, 2, 3, 4, 5] {
    if (i % 2 == 0) {
        print i + " is even";
    } else {
        print i + " is odd";
    }
}
```

### With Functions
```zexus
action isEven(n) {
    if (n % 2 == 0) {
        return true;
    } else {
        return false;
    }
}

if (isEven(10)) {
    print "10 is even";
}
```

### With Arrays/Maps
```zexus
let numbers = [10, 20, 30];

if (numbers[0] < numbers[2]) {
    print "First is less than last";
}

let config = {"debug": true, "verbose": false};

if (config["debug"]) {
    print "Debug mode enabled";
}
```

### Recursive Functions
```zexus
action factorial(n) {
    if (n <= 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

print factorial(5);  // Output: 120
```

## Best Practices

### ✅ DO
- Use parentheses for complex conditions for clarity
- Keep conditions simple and readable
- Use early returns (guard clauses) to reduce nesting
- Prefer flat conditions over deeply nested ones
- Use meaningful variable names in conditions
- Add comments for complex logic

### ❌ DON'T
- Nest conditions more than 3-4 levels deep
- Write overly complex conditions on one line
- Compare booleans explicitly: `if (isReady == true)` → `if (isReady)`
- Use magic numbers in conditions
- Create unreachable code after returns

## Example: Good vs Bad

### ❌ Bad (Deep Nesting)
```zexus
if (user != null) {
    if (user["age"] >= 18) {
        if (user["active"]) {
            if (user["verified"]) {
                return "Valid";
            }
        }
    }
}
return "Invalid";
```

### ✅ Good (Guard Clauses)
```zexus
if (user == null) {
    return "Invalid";
}

if (user["age"] < 18) {
    return "Invalid";
}

if (!user["active"]) {
    return "Invalid";
}

if (!user["verified"]) {
    return "Invalid";
}

return "Valid";
```

### ❌ Bad (Complex Condition)
```zexus
if ((x > 0 && y > 0 && z > 0) || (x < 0 && y < 0 && z < 0) || (x == 0 && y == 0)) {
    // ...
}
```

### ✅ Good (Extracted Logic)
```zexus
let allPositive = (x > 0 && y > 0 && z > 0);
let allNegative = (x < 0 && y < 0 && z < 0);
let bothZero = (x == 0 && y == 0);

if (allPositive || allNegative || bothZero) {
    // ...
}
```

## Edge Cases & Gotchas

### 1. Operator Precedence
```zexus
// Be careful with operator precedence
if (x > 5 && y > 10 || z > 15) {
    // Evaluates as: (x > 5 && y > 10) || (z > 15)
}

// Use parentheses for clarity
if ((x > 5 && y > 10) || z > 15) {
    // Clearer intent
}
```

### 2. Short-Circuit Evaluation
```zexus
// AND (&&) short-circuits if first is false
if (false && expensiveFunction()) {
    // expensiveFunction() is never called
}

// OR (||) short-circuits if first is true
if (true || expensiveFunction()) {
    // expensiveFunction() is never called
}
```

### 3. Empty Blocks
```zexus
// Empty if blocks are valid but not recommended
if (condition) {
    // Empty - why have this?
}

// Consider removing or adding a comment
if (condition) {
    // TODO: Implement this logic
}
```

### 4. ELIF Order Matters
```zexus
let score = 95;

// First match wins
if (score >= 60) {
    print "Pass";  // This executes
} elif (score >= 90) {
    print "Excellent";  // Never reached!
}

// Correct order: specific to general
if (score >= 90) {
    print "Excellent";
} elif (score >= 60) {
    print "Pass";
}
```

## Performance Considerations

1. **Condition Ordering**: Place most likely conditions first for faster evaluation
2. **Short-Circuit**: Use `&&` and `||` to avoid unnecessary evaluations
3. **Avoid Complex Expressions**: Pre-compute complex conditions
4. **Early Returns**: Exit functions early to skip unnecessary code

## Debugging Tips

### 1. Print Conditions
```zexus
let condition = (x > 5 && y < 10);
print "Condition result: " + condition;

if (condition) {
    print "Executed";
}
```

### 2. Test Each Branch
```zexus
// Ensure all branches are tested
if (value < 0) {
    print "Negative";  // Test with -5
} elif (value == 0) {
    print "Zero";       // Test with 0
} else {
    print "Positive";   // Test with 5
}
```

### 3. Use DEBUG
```zexus
debug condition;  // Shows the condition value
if (condition) {
    debug "Inside if block";
}
```

## Real-World Examples

### 1. User Authentication
```zexus
action authenticate(username, password) {
    if (username == null || password == null) {
        return {"success": false, "message": "Missing credentials"};
    }
    
    if (username == "admin" && password == "secret123") {
        return {"success": true, "user": "admin"};
    }
    
    return {"success": false, "message": "Invalid credentials"};
}
```

### 2. Grade Calculator
```zexus
action calculateGrade(score, attendance) {
    if (score >= 90 && attendance >= 90) {
        return "A+";
    } elif (score >= 90) {
        return "A";
    } elif (score >= 80) {
        return "B";
    } elif (score >= 70) {
        return "C";
    } elif (score >= 60) {
        return "D";
    } else {
        return "F";
    }
}
```

### 3. Shopping Cart Logic
```zexus
action calculateDiscount(total, isMember, itemCount) {
    let discount = 0;
    
    if (isMember) {
        discount = discount + 10;  // 10% member discount
    }
    
    if (total > 100) {
        discount = discount + 5;   // 5% bulk discount
    }
    
    if (itemCount > 10) {
        discount = discount + 5;   // 5% quantity discount
    }
    
    return discount;
}
```

### 4. Date Validation
```zexus
action isValidDate(day, month, year) {
    if (month < 1 || month > 12) {
        return false;
    }
    
    if (day < 1 || day > 31) {
        return false;
    }
    
    if (month == 2) {
        if (day > 29) {
            return false;
        }
    } elif (month == 4 || month == 6 || month == 9 || month == 11) {
        if (day > 30) {
            return false;
        }
    }
    
    return true;
}
```

## Summary

The `if`, `elif`, and `else` keywords are fundamental to control flow in Zexus:

### Key Points
- ✅ `if` executes code when a condition is true
- ✅ `elif` checks additional conditions if previous ones were false
- ✅ `else` executes when all conditions are false
- ✅ Parentheses around conditions are optional
- ✅ Multiple `elif` clauses are allowed
- ✅ Nesting is supported but should be kept shallow
- ✅ Conditions short-circuit with `&&` and `||`

### When to Use Each
- **IF alone**: Single condition check
- **IF-ELSE**: Binary choice (true/false)
- **IF-ELIF-ELSE**: Multiple mutually exclusive conditions
- **Nested IF**: Complex decision trees (use sparingly)

---

**Related Keywords**: TRUE, FALSE, NULL, AND, OR, NOT  
**Category**: Control Flow  
**Status**: ✅ Fully Implemented  
**Last Updated**: December 16, 2025
