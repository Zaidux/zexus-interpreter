# Zexus Language Guide

**Verified against zexus 2.0.** Every snippet in this guide was executed
before publication — the verification script lives in the repo history of
this file's regeneration. If a snippet fails for you, it's a bug: open an
issue.

The canonical language contract is [GRAMMAR.md](GRAMMAR.md). This guide is
the tutorial; GRAMMAR.md is the law.

---

## 1. Hello, world

```zexus
print("Hello, Zexus!")
```

Run: `zx run hello.zx`

## 2. Variables and types

```zexus
let count = 10              // integer (checked arithmetic: overflow traps)
const name = "Zexus"        // immutable binding
let price = 19.99           // float
let active = true           // boolean
let nothing = null          // null
let mask = 0xFF             // hex literal → 255
```

Integer arithmetic is **checked by default** — overflow aborts with an
error instead of wrapping. `wrapping_add(a, b)` is the explicit opt-out.

## 3. Functions

`fn` is the canonical keyword (`function` is accepted through the 2.x
warn phase):

```zexus
fn fib(n) {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}
print(fib(10))              // 55
```

Closures:

```zexus
fn make_adder(n) { return fn(x) { return x + n } }
let add5 = make_adder(5)
print(add5(3))              // 8
```

Module-level state mutated from functions lands where you read it:

```zexus
let chain = []
fn add_block() { chain.push("b") }
add_block()
print(chain.len())          // 1
```

## 4. Control flow

```zexus
// while
let n = 0
while n < 3 { n = n + 1 }

// for-each (canonical)
for each x in [1, 2, 3] { print(x) }

// for-in (also canonical)
for x in [1, 2, 3] { print(x) }

// ranges: exclusive end, Python semantics
let total = 0
for i in 0..4 { total = total + i }
print(total)                // 6

// if / elif / else
if total > 5 { print("big") } elif total > 2 { print("mid") } else { print("small") }

// ternary expression
let label = total > 5 ? "big" : "small"
```

## 5. Pattern matching

`match` with `pattern => body` arms and `_` wildcard — in statement or
value position:

```zexus
let desc = match 5 % 2 { 0 => "even" _ => "odd" }
print(desc)                             // odd

match "b" {
    "a" => { print("was a") }
    "b" => { print("was b") }
    _   => { print("other") }          // was b
}
```

## 6. Strings

Methods are the canonical form (see GRAMMAR.md section 4):

```zexus
print("  hi  ".trim())              // "hi"
print("hello".len())                // 5
print("hello".contains("ell"))      // true
print("hello".upper())              // HELLO
print("abcdef".slice(1, 3))         // "bc"
print("a-b-c".split("-").len())     // 3
print("a-b".split("-").join("_"))   // a_b
print("hello".replace("l", "L"))    // heLLo
print("42".to_int() + 1)            // 43
print("hi".to_hex())                // 6869
print("6869".from_hex())            // hi
```

Interpolation:

```zexus
let name = "Zexus"
print("Hello ${name}!")              // Hello Zexus!
```

Escapes: `\n` `\t` `\r` `\\` `\"` `\xNN` (byte) `\uNNNN` (unicode).

## 7. Bytes (binary payloads)

Raw byte sequences for protocol and crypto work. `\xNN` is the **byte**,
not the unicode codepoint:

```zexus
let probe = b"\x00\x01\xff"
print(probe.len())                  // 3
print(probe.to_hex())               // 0001ff
print(b"hi" + b"!")                 // concat
print(b"\x41".to_string())          // "A"
print(bytes_from_hex("deadbeef").to_hex())   // deadbeef
print(b"abc".at(1))                 // 98 (byte value)
print(b"payload".contains(b"load")) // true
```

## 8. Collections

```zexus
// lists
let l = [1, 2]
l.push(3)
print(l.len())                      // 3

// maps
let m = {"a": 1}
m["b"] = 2
print(m["a"] + m["b"])              // 3
```

## 9. Contracts

The crypto category's core — state declared in one `state { }` block,
actions with `this.` access:

```zexus
contract Counter {
    state { count: 0 }
    action increment() { this.count = this.count + 1 }
    action get() { return this.count }
}

let c = Counter()
c.increment()
c.increment()
print(c.get())                      // 2 — identical on VM and tree-walk
```

Guards and events inside actions:

```zexus
action transfer(to, amount) {
    require(amount > 0, "zero transfer")
    // ... state changes ...
    emit Transfer(msg.sender, to, amount)
}
```

Field-level access control (defense):

```zexus
let user = {"password": "secret", "name": "bob"}
restrict user.password = "deny"     // registers a security restriction
print(user.name)                    // bob
```

## 10. Entities and enums

```zexus
data Point { x: integer, y: integer }
let p = Point{x: 1, y: 2}
print(p.x + p.y)                    // 3

enum Color { Red Green Blue }
print(Color.Red)                    // 0
```

Note: fields not set during construction read as `null` (declare
defaults or set all fields).

## 11. Errors

```zexus
try {
    throw "boom"
} catch e {
    print("caught:", e)
}
```

Errors are values on the tree-walk evaluator and abort on the VM (one
documented divergence — see ROADMAP.md Phase G).

## 12. The safety model

**Safe by default, dangerous on demand.**

```zexus
// file_read_text, http_get, sockets: DENIED until granted
grant self io_full
print(file_write_text("note.txt", "hello"))
revoke self io_full                 // denied again
```

```zexus
grant self network                  // enables http_get and sockets
```

Capability sets expand: `network` grants `network.tcp` + `network.http`
+ more. `revoke` withdraws exactly what the matching grant issued.

**Sandbox** — stricter than ungranted: blocks builtins entirely inside
the block:

```zexus
sandbox {
    print("sandboxed print ok")     // print is fine
    // file_read_text("x")          // ❌ not allowed inside sandbox
}
```

## 13. Modules

```zexus
use "crypto"
print(hash_sha256("abc"))

use "json"
let s = stringify({"ok": true})

use "netsec"                        // security_headers, dns_lookup, ...
use "pentest"                       // fingerprint_web, discover_subdomains, ...
```

## 14. Concurrency

```zexus
async action fetch_val() { return 42 }
print(await fetch_val())            // 42

defer { print("runs after the main body") }
```

## 15. Concurrency, queries, and policy (Tier 1-3 wiring)

All of the following are wired, enforced, and regression-tested
(`tests/grammar/test_phase_wiring.py`):

### Contract invariants (enforced)

```zexus
contract Bank {
    state { balance: 100 }
    invariant no_overdraft { this.balance >= 0 }
    action withdraw(amount) { this.balance = this.balance - amount }
    action check() { return this.balance }
}
let b = Bank()
b.withdraw(30)                 // ok: 70
try { b.withdraw(200) } catch e { }
print(b.check())               // 70 — the violating action ROLLED BACK
```

### Channels

```zexus
channel<integer> numbers = 10;
send(numbers, 42);
let a = receive(numbers);      // 42
```

### find ... where (collection query)

```zexus
let users = [{"name": "alice", "age": 30}, {"name": "bob", "age": 16}]
let adult = find u in users where (u.age >= 18)   // alice's map
let none = find u in users where (u.age > 100)    // null
```

### seal (immutable binding)

```zexus
let config = {"k": "v"}
seal config
// config.k = "x"            // ❌ rejected: sealed
```

### throttle + middleware + dispatch

```zexus
throttle api, { rate: 3, burst: 3 }
if throttle_check("api") { /* within limits */ }

fn guard(req) {
    if req.token == "secret" { return true }
    return false
}
middleware guard
fn route(req) { return "data" }
register_route("r", route)
dispatch("r", {"token": "wrong"})   // "middleware 'guard' rejected the request"
dispatch("r", {"token": "secret"})  // "data"
```

## 15b. What is NOT wired (as of 2.0)

- anonymous `action(a, b) { ... }` as expression value (V-004/R-029)
- `trail audit/print/debug` — parses and runs; the event-following
  output is not yet visible

When in doubt, the executable differential corpus
(`tests/grammar/test_differential.py`) is the source of truth.

## 16. Next steps

- [GRAMMAR.md](GRAMMAR.md) — the contract (canonical forms, migration table)
- [QUICK_START.md](QUICK_START.md) — verified commands
- [examples/](examples/) — recon tool, API server, crypto, contracts
