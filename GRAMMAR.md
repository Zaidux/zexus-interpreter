# Zexus Unified Grammar — v2 Specification

**Status:** canonical spec (staged rollout — see Migration)
**Supersedes:** the four coexisting contract syntaxes, the emit/protect variants,
and every construct marked "legacy" below.

## 1. Design principles

1. **One construct, one form.** No synonyms, no alternative spellings, no
   "also accepted". If two syntaxes exist today, exactly one is canonical and
   the others are legacy (migration table, §9).
2. **~35 keywords.** Domain power lives in the standard library
   (`use "pentest"`, `use "crypto"`, `use "sast"`), not in the grammar.
   A keyword must justify itself for all three categories or it is library.
3. **Declarations use blocks; expressions stay expressiony.**
4. **Optional types everywhere with one syntax:** `name: Type`.
5. **Fail loudly.** Any parse error is fatal (v1.9+). Statements are never
   silently dropped — the v1.x lenient parser made the docs, samples, and
   sibling repos describe languages that don't exist.
6. **Safety defaults:** checked integer arithmetic (overflow traps unless
   `wrapping()` is explicit); taint-tracked strings; sandbox mode available.

## 2. Category mapping

| Category | Grammar contribution | Library surface |
|---|---|---|
| Exploit / DAST / bug bounty | `bytes`, `0x`/`b"\xNN"` literals, `match`, string methods | `pentest`, `payloads`, `netsec`, `http`, `sockets`, `fuzz`, `db` |
| Defense / SAST / audit | `invariant`, `protect`, taint functions, sandbox | `sast` (AST-based rules), `audit`, `contract_audit`, SARIF export |
| Crypto / contracts / chain | `contract`, `state`, `action`, `require`, `emit`, `persist`, `address`, checked ints | `crypto` (secp256k1/Keccak/AES), `blockchain` (devnet), rust_core FFI |

Shared by all three: `fn`, `let`/`const`, control flow, `try/catch`, modules.

## 3. Core language

```
program        := toplevel*
toplevel       := declaration | statement

declaration    := letdecl | constdecl | funcdecl | contractdecl | eventdecl
                | structdecl | enumdecl | protectdecl | usedecl | exportdecl

letdecl        := 'let' IDENT [':' type] '=' expr
constdecl      := 'const' IDENT [':' type] '=' expr
funcdecl       := 'fn' IDENT '(' [params] ')' ['->' type] block
params         := IDENT [':' type] (',' IDENT [':' type])*

usedecl        := 'use' STRING ['as' IDENT]
exportdecl     := 'export' (funcdecl | contractdecl | eventdecl
                | constdecl | structdecl)

statement      := declaration | assignment | exprstmt
                | ifstmt | whilestmt | forstmt | matchstmt | trystmt
                | 'require' '(' expr [',' STRING] ')'
                | 'emit' IDENT '(' [args] ')'
                | 'return' [expr] | 'break' | 'continue'

ifstmt         := 'if' expr block ('elif' expr block)* ['else' block]
whilestmt      := 'while' expr block
forstmt        := 'for' IDENT 'in' expr block
                | 'for' IDENT ',' IDENT 'in' expr block      // index, value
matchstmt      := 'match' expr '{' matcharm* '}'
matcharm       := pattern '=>' (block | expr)
trystmt        := 'try' block 'catch' [IDENT] block
```

**Canonical decisions (legacy → here in §9):** `fn` (not `function`/`action`
outside contracts); `for x in xs` (not `for each x in`); one `match` arm
syntax (`=>`); ternary `?:` (the `if … then … else` expression form is
legacy); one comment token `//` (both accepted at the lexer for now; `#`
is legacy).

## 4. Types

```
type := 'int' | 'float' | 'string' | 'bytes' | 'bool'
      | 'address'                      // distinct from string — never
                                        // interchangeable with user input
      | 'list' '<' type '>' | 'map' '<' type ',' type '>'
      | IDENT                           // struct / enum / contract names
```

- **`bytes`** — first-class: `b"\xde\xad\xbe\xef"`, concatenation with `+`,
  `len()`, slicing, `to_hex()`/`from_hex()`. Hex numeric literals `0xFF`
  and `\xNN`/`\uNNNN` escapes are part of the lexer.
- **Checked arithmetic**: `+ - *` on `int` trap on overflow.
  `wrapping_add(x, y)` etc. are the explicit opt-out (protocol parsers
  that rely on wraparound must say so — an audit signal in itself).
- **String methods**: `.len() .slice() .contains() .split() .trim() …`
  on the object model (the zero-method String of v1.x is legacy).

## 5. Contracts (crypto + defense)

```
contractdecl   := ['export'] 'contract' IDENT ['(' params ')']
                  '{' (stateblock | actiondecl | invariantdecl)* '}'

stateblock     := 'state' '{' statefield* '}'
statefield     := ['persist'] IDENT ':' type '=' expr

actiondecl     := 'action' IDENT '(' [params] ')' ['->' type] block

invariantdecl  := 'invariant' IDENT block        // defense: verified after
                                                 // EVERY action; violation
                                                 // aborts the action
eventdecl      := 'event' IDENT '{' (IDENT ':' type ',')* '}'
```

Canonical example:

```zexus
export contract Token(state owner: Address) {
    state {
        balances: Map<Address, int> = {}
        supply: int = 1_000_000
    }

    invariant supply_ok { sum(balances.values) <= supply }

    action transfer(to: Address, amount: int) {
        require(amount > 0, "zero transfer")
        require(balances[msg.sender] >= amount, "insufficient balance")
        balances[msg.sender] -= amount          // checked; traps on overflow
        balances[to] += amount
        emit Transfer(msg.sender, to, amount)
    }
}

event Transfer(from: Address, to: Address, amount: int)
```

Rules:
- Contract state is declared **only** in the `state { }` block (one form).
- Contract methods are `action`; free functions are `fn`. Inside actions,
  state is accessed as `this.balances` (bare access is an undefined-local
  error, not a silent fallback). `msg.sender` is available in actions.
- `require(cond, "msg")` is the guard form — call syntax, one form.
- `emit Name(args)` is the emit form — call syntax, one form.
- `persist` marks durable storage (replaces `persistent storage`).

## 6. Defense surface

```
protectdecl    := 'protect' IDENT '{' ('verify' expr)*
                   ['on' 'violation' block] '}'
```

```zexus
protect no_negative_balance {
    verify balance >= 0
    on violation { audit("balance went negative", severity: "critical") }
}
```

- `invariant` blocks in contracts: checked after every action.
- Taint is functional, not grammatical: `sanitize(x, "sql")`,
  `mark_sanitized(x)`, `is_trusted(x)` — regular calls, composable.
- Capability grants are the explicit opt-in for dangerous operations:
  `grant self network` / `grant self io_full` / `revoke self io_full`.
  Named capability sets expand to their constituents (`network` →
  network.tcp + network.http); grants land in the runtime context the
  gated builtins check. Gated by default: `http_get`, `file_read_text`,
  `file_write_text`, sockets, process spawn.
- Sandboxing is a runtime mode (`zx run --sandbox untrusted.zx`),
  not syntax.

## 7. Exploit surface

No grammar additions beyond §4 — the power is library:

```zexus
use "pentest" as pt
use "payloads" as pw

let probe = b"\x00\x01\xff" + from_hex("deadbeef")
match pt.http_probe(target, timeout: 3.0) {
    {status: 200, headers: _} => pt.report("alive", severity: "info")
    {status: 500}             => pw.fuzz_param(target, "q")
    _                         => pt.next_target()
}
```

## 8. Reserved for v2 (not yet implemented — do NOT document as working)

`async fn` / `await` (stdlib-level parallelism covers current needs),
`property` blocks for fuzzing (`property p { for any x: invariant }`),
`extern "rust"` FFI declarations, `enum` with payload types.

## 9. Migration table (legacy → canonical)

| Legacy form (v1.x) | Canonical (this spec) | Phase |
|---|---|---|
| `state balances = {}` (assignment form) | `state { balances = {} }` | warn in 1.9, error in 2.0 |
| `data balances = {}` | `state { balances = {} }` | warn in 1.9, error in 2.0 |
| `persistent storage x: T` | `state { x: T = default }` + `persist` | **error now** (was parsed-and-discarded) |
| `function f() {}` | `fn f() {}` | warn in 1.9, error in 2.0 |
| `action` outside a contract | `fn` | error in 2.0 |
| bare contract fields (no `this.`) inside actions | `this.field` | warn in 1.9 |
| `emit event Name { k: v }` | `emit Name(args)` | **error now** (was silently dropped) |
| `emit Name { k: v }` (object form) | `emit Name(args)` | warn in 1.9 |
| `protect rule name { verify e }` | `protect(target, rules)` call; block form per §6 in 2.0 | **error now** (was silently dropped) |
| `for each x in xs` | `for x in xs` | warn in 1.9 |
| `if c then a else b` (expression) | `c ? a : b` | warn in 1.9 |
| `sanitize data as sql` (keyword form) | `sanitize(data, "sql")` | warn in 1.9 |
| `entity` blocks | `struct` | warn in 1.9 |
| `contract` in expression position (`contract Token(...)`) | `deploy Token(...)` | error now |

## 10. Non-goals

- No category-specific grammars. If a feature only serves one category and
  can be a function, it is a function.
- No lenient parsing. Ever. The v1.x behavior of dropping statements whose
  parse failed (unless the error contained the word "critical") corrupted
  every artifact around this project; it does not come back.
