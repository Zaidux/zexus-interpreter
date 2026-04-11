# Zexus Keywords Reference

Complete list of all keywords in the Zexus language (v1.8.4).

---

## Core Language

| Keyword | Description | Example |
|---------|-------------|---------|
| `let` | Declare a mutable variable | `let x = 10` |
| `const` | Declare an immutable constant | `const PI = 3.14` |
| `print` | Print a value to stdout | `print("hello")` |
| `return` | Return a value from a function | `return x + 1` |
| `this` | Reference the current contract/entity instance | `this.balance` |

## Data Types

| Keyword | Description | Example |
|---------|-------------|---------|
| `true` | Boolean true literal | `let active = true` |
| `false` | Boolean false literal | `let done = false` |
| `null` | Null/empty value | `let empty = null` |
| `data` | Define a data structure | `data Point { x: int, y: int }` |

## Control Flow

| Keyword | Description | Example |
|---------|-------------|---------|
| `if` | Conditional branch | `if x > 5 { ... }` |
| `elif` | Else-if branch | `elif x > 3 { ... }` |
| `else` | Default branch | `else { ... }` |
| `while` | Loop while condition is true | `while i < 10 { ... }` |
| `for` | Begin a loop | `for i in range(10) { ... }` |
| `each` | Used with `for` for iteration | `for each item in list { ... }` |
| `in` | Membership or iteration target | `for x in items { ... }` |
| `break` | Exit a loop early | `break` |
| `continue` | Skip to next loop iteration | `continue` |
| `match` | Pattern matching | `match value { case 1: ... }` |
| `case` | A match branch | `case "a": print("A")` |
| `default` | Default match branch | `default: print("other")` |

## Functions

| Keyword | Description | Example |
|---------|-------------|---------|
| `function` | Declare a named function | `function add(a, b) { return a + b }` |
| `action` | Declare an action (method) in contract/entity | `action deposit(amount) { ... }` |
| `lambda` | Anonymous function expression | `let double = lambda(x) { x * 2 }` |

## Error Handling

| Keyword | Description | Example |
|---------|-------------|---------|
| `try` | Begin error handling block | `try { ... }` |
| `catch` | Handle caught errors | `catch (e) { print(e) }` |
| `finally` | Always-executed cleanup block | `finally { cleanup() }` |
| `throw` | Throw an error/exception | `throw "error message"` |

## Modules & Imports

| Keyword | Description | Example |
|---------|-------------|---------|
| `use` | Import a module or file | `use "utils.zx"` |
| `export` | Export names from a module | `export { add, subtract }` |
| `import` | Alternative import syntax | `import math` |
| `from` | Specify import source | `use { add } from "math.zx"` |
| `module` | Declare a module | `module math { ... }` |
| `package` | Declare a package | `package utils` |
| `using` | Bring names into scope | `using math` |

## Logical Operators

| Keyword | Description | Example |
|---------|-------------|---------|
| `and` | Logical AND | `if a and b { ... }` |
| `or` | Logical OR | `if a or b { ... }` |
| `not` | Logical NOT (prefix) | `if not done { ... }` |

## Object-Oriented

| Keyword | Description | Example |
|---------|-------------|---------|
| `entity` | Define an entity (class-like) | `entity User { name; age }` |
| `contract` | Define a stateful contract | `contract Wallet { state { ... } }` |
| `state` | Declare contract state block | `state { balance: 0 }` |
| `implements` | Interface implementation | `entity Dog implements Animal { ... }` |
| `interface` | Declare an interface | `interface Printable { ... }` |
| `enum` | Define an enumeration | `enum Color { RED, GREEN, BLUE }` |
| `type_alias` | Create a type alias | `type_alias Name = string` |

## Security & Policy

| Keyword | Description | Example |
|---------|-------------|---------|
| `protect` | Define a protection rule | `protect rule name { ... }` |
| `verify` | Verify a condition in a protect block | `verify balance >= 0` |
| `seal` | Seal an entity to prevent changes | `seal User` |
| `audit` | Log an action for auditing | `audit "user logged in"` |
| `restrict` | Restrict operations | `restrict { no_network }` |
| `sandbox` | Run code in a sandbox | `sandbox { untrusted_code() }` |
| `trail` | Audit trail logging | `trail "event occurred"` |
| `validate` | Validate data | `validate input` |
| `sanitize` | Sanitize user input | `sanitize user_input` |
| `immutable` | Mark data as immutable | `immutable data` |
| `capability` | Define a capability | `capability FileAccess { ... }` |
| `grant` | Grant a capability | `grant FileAccess to user` |
| `revoke` | Revoke a capability | `revoke FileAccess from user` |

## Async & Concurrency

| Keyword | Description | Example |
|---------|-------------|---------|
| `async` | Mark a function as async | `async function fetch() { ... }` |
| `await` | Wait for async result | `let data = await fetch()` |
| `channel` | Create a communication channel | `channel ch` |
| `send` | Send data to a channel | `send(ch, "data")` |
| `receive` | Receive data from a channel | `let msg = receive(ch)` |
| `atomic` | Atomic operation block | `atomic { counter += 1 }` |

## Events

| Keyword | Description | Example |
|---------|-------------|---------|
| `event` | Define an event type | `event Transfer { from, to, amount }` |
| `emit` | Emit an event | `emit event Transfer { from: a, to: b }` |

## Access Modifiers

| Keyword | Description | Example |
|---------|-------------|---------|
| `public` | Public access modifier | `public action get() { ... }` |
| `private` | Private access modifier | `private let secret = "key"` |
| `sealed` | Sealed (no inheritance) | `sealed entity Final { ... }` |
| `external` | External function declaration | `external function clib()` |

## Blockchain / Smart Contracts

| Keyword | Description | Example |
|---------|-------------|---------|
| `ledger` | Access blockchain ledger | `ledger.record(tx)` |
| `require` | Assert a condition (revert if false) | `require(amount > 0)` |
| `revert` | Revert a transaction | `revert "insufficient funds"` |
| `limit` | Gas/resource limit | `limit { gas: 1000 }` |
| `persistent` | Persistent storage | `persistent storage data { ... }` |
| `storage` | Storage declaration | `storage { balance: 0 }` |

## Performance & Low-Level

| Keyword | Description | Example |
|---------|-------------|---------|
| `native` | Native function binding | `native function fast_hash()` |
| `gc` | Garbage collection hint | `gc collect` |
| `inline` | Inline function hint | `inline function square(x) { x * x }` |
| `buffer` | Buffer allocation | `buffer data = alloc(1024)` |
| `simd` | SIMD operations | `simd add(vec_a, vec_b)` |

## Other

| Keyword | Description | Example |
|---------|-------------|---------|
| `defer` | Defer execution until scope exit | `defer { file.close() }` |
| `pattern` | Pattern definition | `pattern EmailFormat { ... }` |
| `stream` | Stream processing | `stream data { ... }` |
| `watch` | Watch for changes (reactive) | `watch variable { ... }` |
| `debug` | Debug output | `debug "value is: " + str(x)` |
| `log` | Log a message | `log "event occurred"` |
| `as` | Type/name alias in import | `use "math.zx" as m` |
| `inject` | Dependency injection | `inject logger: Logger` |
| `secure` | Security modifier | `secure action transfer() { ... }` |
| `pure` | Pure function modifier | `pure function add(a, b) { ... }` |
| `view` | View-only (no state changes) | `view action getBalance() { ... }` |
| `payable` | Accepts value transfers | `payable action deposit() { ... }` |
| `middleware` | Middleware declaration | `middleware auth { ... }` |
| `auth` | Authentication middleware | `auth required` |
| `throttle` | Rate limiting | `throttle 100/minute` |
| `cache` | Caching directive | `cache ttl=300` |

---

## Operators (Keyword-Style)

| Operator | Description | Example |
|----------|-------------|---------|
| `and` | Logical AND | `a and b` |
| `or` | Logical OR | `a or b` |
| `not` | Logical NOT | `not condition` |
| `in` | Membership test | `"x" in list` |

## Symbol Operators

| Symbol | Description |
|--------|-------------|
| `+` `-` `*` `/` `%` `**` | Arithmetic |
| `+=` `-=` `*=` `/=` `%=` `**=` | Compound assignment |
| `==` `!=` `<` `>` `<=` `>=` | Comparison |
| `=` | Assignment |
| `.` | Property access |
| `[]` | Index access |
| `()` | Call / grouping |
| `{}` | Block / map literal |
| `//` `/* */` | Comments |
