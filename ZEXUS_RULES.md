# Zexus Rules — Avoid Mistakes & Assumptions

A concise list of rules, verified against Zexus v1.8.3, to prevent common mistakes.

---

## Rule 1: Always use `let` to declare variables

Every variable must be declared with `let` before use.

```zexus
// ✅ Correct
let x = 10

// ❌ Wrong — undeclared variable
x = 10
```

---

## Rule 2: Compound assignment operators work

Compound operators (`+=`, `-=`, `*=`, `/=`, `%=`, `**=`) work correctly as of v1.8.4.

```zexus
// ✅ All compound operators work
let x = 10
x += 5
print(x)   // 15

x -= 3
print(x)   // 12

x *= 2
print(x)   // 24
```

---

## Rule 3: Both `for each` and `for...in range()` work

Both loop syntaxes work correctly as of v1.8.4.

```zexus
// ✅ Range-based for loop
for i in range(5) {
  print(i)
}

// ✅ for-each loop
for each item in [10, 20, 30] {
  print(item)
}

// ✅ While loop (also works)
let i = 0
while i < 5 {
  print(i)
  i = i + 1
}
```

---

## Rule 4: Entities support constructors

Entities accept positional constructor arguments matching their declared fields (fixed in v1.8.4). Contracts are still recommended for objects with methods and mutable state.

```zexus
// ✅ Entity with constructor args (v1.8.4+)
entity Dog {
  name
  breed
}
let d = Dog("Rex", "Husky")
print(d.name)    // "Rex"

// ✅ Contract — fully functional with state and methods
contract Dog {
  state { name: "default" }
  action set_name(n) { this.name = n }
  action get_name() { return this.name }
}
```

---

## Rule 5: Always use `str()` when concatenating with non-strings

Mixing types in string concatenation without `str()` may cause errors.

```zexus
// ❌ May error
print("Count: " + 42)

// ✅ Safe
print("Count: " + str(42))
```

---

## Rule 6: Use `{}` braces for all blocks

Zexus uses curly braces for if, while, for each, function, contract, and try/catch blocks. Do not omit them.

```zexus
// ✅ Always use braces
if x > 5 {
  print("yes")
}

// ❌ No braceless blocks
if x > 5
  print("yes")
```

---

## Rule 7: Create maps with all values upfront

Map indexed assignment (`map["key"] = value`) may not persist. Define all needed key-value pairs at creation time.

```zexus
// ⚠️ May not persist
let m = {}
m["key"] = "value"

// ✅ Define upfront
let m = {key: "value", other: 123}
```

---

## Rule 8: Use `this` inside contracts, not `self`

Inside contract actions, use `this` to refer to the contract's state.

```zexus
contract Counter {
  state { count: 0 }
  action increment() {
    this.count = this.count + 1
  }
}
```

---

## Rule 9: Wrap risky operations in `try/catch`

Division by zero, missing keys, and type errors can crash your program. Use `try/catch` for safety.

```zexus
try {
  let result = 10 / 0
} catch (e) {
  print("Error: " + str(e))
}
```

---

## Rule 10: Use `//` or `/* */` for comments

Both single-line and block comments are supported.

```zexus
// single-line comment
/* multi-line
   block comment */
```

---

## Rule 11: Lists are 0-indexed

The first element is at index `0`.

```zexus
let items = ["a", "b", "c"]
print(items[0])   // "a"
print(items[2])   // "c"
```

---

## Rule 12: Use `length()` not `.length`

Length is a function call, not a property.

```zexus
// ✅ Correct
print(length([1, 2, 3]))

// ❌ Wrong
print([1, 2, 3].length)
```

---

## Rule 13: String interpolation uses `${}` in double quotes

Only double-quoted strings support interpolation.

```zexus
let name = "Zexus"
print("Hello, ${name}!")   // ✅ Works
```

---

## Rule 14: `break` and `continue` work in loops

Both work in `while` loops and `for each` loops.

```zexus
let i = 0
while i < 10 {
  i = i + 1
  if i == 5 { break }
  if i == 3 { continue }
  print(i)
}
```

---

## Rule 15: Functions must be declared before they are called

Define functions above the code that calls them.

```zexus
// ✅ Declare first, then call
function greet(name) {
  return "Hi, " + name
}
print(greet("Alice"))
```

---

## Summary Table

| Do This | Not This | Why |
|---------|----------|-----|
| `x += 1` or `x = x + 1` | — | both work correctly (v1.8.4) |
| `for i in range(n)` | — | works correctly (v1.8.4) |
| `Entity("arg1", "arg2")` | — | entity constructors work (v1.8.4) |
| `not condition` | — | `not` operator works (v1.8.4) |
| `str(val)` in concatenation | raw number in `+` | type mismatch error |
| `length(list)` | `list.length` | length is a function |
| `map["key"] = val` | — | map mutation works correctly |
| `{}` braces on all blocks | braceless blocks | parser requires braces |
