# Zexus Language Guide

A practical, test-verified guide to the Zexus programming language.
Every example in this guide was executed and verified against Zexus v1.8.3.

> **How to run:** Save code to a `.zx` file and run with `zx run file.zx`

---

## Table of Contents

- [Basics](#basics)
  - [Hello World](#hello-world)
  - [Variables](#variables)
  - [Data Types](#data-types)
  - [Arithmetic](#arithmetic)
  - [Comparison & Logical Operators](#comparison--logical-operators)
  - [Comments](#comments)
  - [String Concatenation](#string-concatenation)
  - [String Interpolation](#string-interpolation)
  - [Print Output](#print-output)
- [Intermediate](#intermediate)
  - [If / Else](#if--else)
  - [While Loops](#while-loops)
  - [For Each Loops](#for-each-loops)
  - [Break and Continue](#break-and-continue)
  - [Functions](#functions)
  - [Lists](#lists)
  - [Maps (Dictionaries)](#maps-dictionaries)
  - [Nested Data Structures](#nested-data-structures)
  - [Built-in Functions](#built-in-functions)
  - [Error Handling (Try/Catch/Throw)](#error-handling-trycatchthrow)
- [Advanced](#advanced)
  - [Contracts (Stateful Objects)](#contracts-stateful-objects)
  - [Closures](#closures)
  - [Higher-Order Functions](#higher-order-functions)
  - [Recursion](#recursion)
  - [Protect / Verify (Policy Rules)](#protect--verify-policy-rules)
  - [Emit (Events)](#emit-events)
  - [Indexed Iteration](#indexed-iteration)
  - [Map Iteration](#map-iteration)
- [Known Quirks & Gotchas](#known-quirks--gotchas)

---

## Basics

### Hello World

```zexus
print("Hello, World!")
```

**Output:** `Hello, World!`

---

### Variables

Declare variables with `let`. Reassign with `=`.

```zexus
let name = "Alice"
let age = 25
print(name)
print(age)
```

**Output:**
```
Alice
25
```

Reassignment uses plain `=`:

```zexus
let x = 10
x = x + 5
print(x)
```

**Output:** `15`

---

### Data Types

Zexus has five core types: **string**, **integer**, **float**, **boolean**, and **null**.

```zexus
let text = "Hello"       // string
let num = 42             // integer
let decimal = 3.14       // float
let flag = true          // boolean
let nothing = null       // null
print(text)
print(num)
print(decimal)
print(flag)
print(nothing)
```

**Output:**
```
Hello
42
3.14
true
null
```

---

### Arithmetic

```zexus
let a = 10
let b = 3
print(a + b)     // 13    (addition)
print(a - b)     // 7     (subtraction)
print(a * b)     // 30    (multiplication)
print(a / b)     // 3.33  (division)
print(a % b)     // 1     (modulo)
print(a ** 2)    // 100   (exponentiation)
```

**Output:**
```
13
7
30
3.3333333333333335
1
100
```

---

### Comparison & Logical Operators

```zexus
print(10 == 10)     // true
print(10 != 5)      // true
print(10 > 5)       // true
print(10 >= 10)     // true
print(10 < 20)      // true
print(10 <= 9)      // false

print(true and false)   // false
print(true or false)    // true
print(5 > 3 and 2 < 4) // true
```

---

### Comments

```zexus
// This is a single-line comment

/* This is a
   block comment */

let x = 42 // inline comment
print(x)
```

**Output:** `42`

---

### String Concatenation

Join strings with `+`. Use `str()` to convert other types.

```zexus
let greeting = "Hello" + " " + "World"
print(greeting)
print("Value: " + str(42))
```

**Output:**
```
Hello World
Value: 42
```

---

### String Interpolation

Use `${expression}` inside double-quoted strings.

```zexus
let name = "Zexus"
let msg = "Hello, ${name}!"
print(msg)
```

**Output:** `Hello, Zexus!`

---

### Print Output

`print()` is the primary output function. It prints any value.

```zexus
print("text")
print(42)
print(true)
print([1, 2, 3])
print({key: "value"})
```

---

## Intermediate

### If / Else

Use `{}` braces for blocks.

```zexus
let x = 10
if x > 5 {
  print("x is greater than 5")
} else {
  print("x is 5 or less")
}
```

**Output:** `x is greater than 5`

Chained conditions:

```zexus
let x = 10
if x > 20 {
  print("big")
} else if x > 5 {
  print("medium")
} else {
  print("small")
}
```

**Output:** `medium`

---

### While Loops

```zexus
let i = 0
while i < 5 {
  print(i)
  i = i + 1
}
```

**Output:**
```
0
1
2
3
4
```

---

### For Each Loops

Iterate over lists with `for each`:

```zexus
let items = [1, 2, 3, 4, 5]
for each item in items {
  print(item)
}
```

**Output:**
```
1
2
3
4
5
```

---

### Break and Continue

**Break** exits a loop early:

```zexus
let items = [1, 2, 3, 4, 5]
let i = 0
while i < length(items) {
  if items[i] == 3 {
    print("Found 3, breaking!")
    break
  }
  print(items[i])
  i = i + 1
}
```

**Output:**
```
1
2
Found 3, breaking!
```

**Continue** skips to the next iteration:

```zexus
let i = 0
while i < 5 {
  i = i + 1
  if i == 3 {
    continue
  }
  print(i)
}
```

**Output:**
```
1
2
4
5
```

---

### Functions

Declare with `function`, return values with `return`.

```zexus
function greet(name) {
  return "Hello, " + name
}
let msg = greet("World")
print(msg)
```

**Output:** `Hello, World`

Functions with multiple parameters:

```zexus
function add(a, b) {
  return a + b
}
print(add(3, 7))
```

**Output:** `10`

---

### Lists

Create with `[]`, access by index (0-based), modify with `.push()` and index assignment.

```zexus
let nums = [10, 20, 30]
print(nums)           // [10, 20, 30]
print(nums[0])        // 10
print(nums[1])        // 20
print(length(nums))   // 3

nums.push(40)
print(nums)           // [10, 20, 30, 40]

nums[0] = 99
print(nums)           // [99, 20, 30, 40]
```

---

### Maps (Dictionaries)

Create with `{}`, access by key using `["key"]`.

```zexus
let person = {name: "Alice", age: 30}
print(person)
print(person["name"])    // Alice
print(person["age"])     // 30
```

---

### Nested Data Structures

Lists and maps can be nested:

```zexus
let matrix = [[1, 2], [3, 4], [5, 6]]
print(matrix[0])       // [1, 2]
print(matrix[1][1])    // 4
```

---

### Built-in Functions

| Function     | Description                          | Example              | Result    |
|-------------|--------------------------------------|----------------------|-----------|
| `print(x)`  | Print a value                        | `print("hi")`       | `hi`      |
| `length(x)` | Length of string or list             | `length("hello")`   | `5`       |
| `str(x)`    | Convert to string                    | `str(42)`            | `"42"`    |
| `typeof(x)` | Get type name                        | `typeof("hi")`      | `string`  |
| `abs(x)`    | Absolute value                       | `abs(-10)`           | `10`      |

---

### Error Handling (Try/Catch/Throw)

Catch runtime errors with `try/catch`:

```zexus
try {
  let x = 10 / 0
} catch (e) {
  print("Caught: " + str(e))
}
```

**Output:** `Caught: Division by zero`

Throw custom errors:

```zexus
try {
  throw "Something went wrong"
} catch (e) {
  print("Caught: " + str(e))
}
```

**Output:** `Caught: Something went wrong`

---

## Advanced

### Contracts (Stateful Objects)

Contracts are Zexus's primary way to create stateful, encapsulated objects. They use `state` blocks for data and `action` for methods.

```zexus
contract Counter {
  state {
    count: 0
  }

  action increment() {
    this.count = this.count + 1
  }

  action get_count() {
    return this.count
  }
}

let c = Counter()
c.increment()
c.increment()
c.increment()
print(c.get_count())
```

**Output:** `3`

A more complete example:

```zexus
contract Wallet {
  state {
    balance: 100
  }

  action deposit(amount) {
    this.balance = this.balance + amount
  }

  action withdraw(amount) {
    if this.balance >= amount {
      this.balance = this.balance - amount
      return true
    }
    return false
  }

  action get_balance() {
    return this.balance
  }
}

let w = Wallet()
print(w.get_balance())    // 100
w.deposit(50)
print(w.get_balance())    // 150
w.withdraw(30)
print(w.get_balance())    // 120
```

---

### Closures

Inner functions can access variables from outer scopes:

```zexus
function outer() {
  let x = 10
  function inner() {
    return x + 5
  }
  return inner()
}
print(outer())
```

**Output:** `15`

---

### Higher-Order Functions

Functions can be passed as arguments:

```zexus
function apply(fn, val) {
  return fn(val)
}

function double(x) {
  return x * 2
}

print(apply(double, 5))
```

**Output:** `10`

Nested function calls:

```zexus
function add(a, b) {
  return a + b
}

function multiply(a, b) {
  return a * b
}

let result = add(multiply(3, 4), 5)
print(result)
```

**Output:** `17`

---

### Recursion

Functions can call themselves:

```zexus
function factorial(n) {
  if n <= 1 {
    return 1
  }
  return n * factorial(n - 1)
}
print(factorial(5))
print(factorial(10))
```

**Output:**
```
120
3628800
```

---

### Protect / Verify (Policy Rules)

Define declarative security rules with `protect` and `verify`:

```zexus
protect rule no_negative_balance {
  verify balance >= 0
  message "Balance cannot be negative"
}

let balance = 100
balance = balance - 50
print(balance)
```

**Output:** `50`

---

### Emit (Events)

Emit named events with data payloads:

```zexus
emit event user_login {
  username: "alice"
}
print("Event emitted")
```

**Output:** `Event emitted`

---

### Indexed Iteration

Use two variables in `for each` to get index and value:

```zexus
let items = [3, 1, 4, 1, 5]
for each i, item in items {
  print(str(i) + ": " + str(item))
}
```

**Output:**
```
0: 3
1: 1
2: 4
3: 1
4: 5
```

---

### Map Iteration

Iterate over map keys and values:

```zexus
let data = {x: 1, y: 2, z: 3}
for each key, val in data {
  print(key + ": " + str(val))
}
```

**Output:**
```
x: 1
y: 2
z: 3
```

---

## Known Quirks & Gotchas

These are verified behaviors as of v1.8.4.

### 1. Compound assignment works (fixed in v1.8.4)

`+=`, `-=`, `*=`, `/=`, `%=`, `**=` all work correctly.

```zexus
let x = 10
x += 5
print(x)   // prints 15 ✅
```

### 2. Map indexed assignment works (confirmed in v1.8.4)

Assigning values to map keys via `map["key"] = value` works correctly.

```zexus
let data = {count: 0}
data["count"] = 42
print(data["count"])   // prints 42 ✅
```

### 3. Entity constructors with arguments (fixed in v1.8.4)

Entities accept positional arguments matching their declared fields.

```zexus
entity Dog {
  name
  breed
}
let d = Dog("Rex", "Husky")
print(d.name)    // "Rex" ✅
print(d.breed)   // "Husky" ✅
```

You can also use typed fields:

```zexus
entity User {
  name: string
  age: int = 0
}
let u = User("Alice", 25)
```

### 4. `not` operator works (fixed in v1.8.4)

The `not` keyword is now properly supported as a prefix operator.

```zexus
print(not true)    // false ✅
print(not false)   // true ✅
if not done {
  print("still going")
}
```

### 5. `for i in range(n)` works (fixed in v1.8.4)

Both `for each` and `for...in` syntax work correctly.

```zexus
// ✅ Shorthand syntax
for i in range(5) {
  print(i)   // 0, 1, 2, 3, 4
}

// ✅ Traditional syntax
for each item in [10, 20, 30] {
  print(item)
}
```

### 6. Use `str()` when concatenating non-strings

Concatenating a string with a number without `str()` may cause an error or unexpected result.

```zexus
// ✅ Always convert explicitly
print("Value: " + str(42))
```

### 7. Division by zero is caught but verbose

Dividing by zero in a `try/catch` works but may print extra error messages before your catch handler runs. This is normal behavior.

### 8. String interpolation falls back to interpreter

Using `"${expr}"` triggers a VM fallback to interpreter mode. It works correctly but you'll see a warning message in the console output.

---

## Quick Reference

| Feature | Syntax | Works? |
|---------|--------|--------|
| Variable declaration | `let x = 10` | ✅ |
| Reassignment | `x = x + 1` | ✅ |
| Compound assignment | `x += 1` | ✅ |
| If/else | `if cond { } else { }` | ✅ |
| While loop | `while cond { }` | ✅ |
| For each | `for each item in list { }` | ✅ |
| For in range | `for i in range(n) { }` | ✅ |
| Functions | `function name(args) { }` | ✅ |
| Return | `return value` | ✅ |
| Lists | `[1, 2, 3]` | ✅ |
| List push | `list.push(val)` | ✅ |
| List index | `list[0]` | ✅ |
| Maps | `{key: value}` | ✅ |
| Map access | `map["key"]` | ✅ |
| Map mutation | `map["key"] = val` | ✅ |
| Contracts | `contract Name { state { } action fn() { } }` | ✅ |
| Entities | `entity Name { field }` | ✅ |
| Entity constructor | `Entity("arg1", "arg2")` | ✅ |
| Not operator | `not true` / `!false` | ✅ |
| Try/catch | `try { } catch (e) { }` | ✅ |
| Throw | `throw "message"` | ✅ |
| Break | `break` | ✅ |
| Continue | `continue` | ✅ |
| String interpolation | `"${expr}"` | ✅ |
| Comments | `//` and `/* */` | ✅ |
| Closures | nested functions | ✅ |
| Recursion | self-calling functions | ✅ |
| Protect/Verify | `protect rule { verify ... }` | ✅ |
| Emit events | `emit event name { }` | ✅ |
| Exponentiation | `a ** b` | ✅ |

**Legend:** ✅ = works reliably
