# Potentially Redundant Keywords in Zexus

Analysis of keywords that could potentially be removed safely. These either duplicate
existing functionality, are unimplemented stubs, or are extremely unlikely to be used.

---

## 🔴 High Confidence — Safe to Remove

These keywords are parsed but have no meaningful runtime implementation.

| Keyword | Reason for Removal |
|---------|-------------------|
| `component` | Parsed into AST but only stores a string — no actual UI component system exists |
| `theme` | Stub implementation only (prints a message) — no real theming system |
| `storage` | Completely unused in evaluator — functionality covered by `persistent` and `ledger` |
| `then` | Pure syntax sugar in `if` expressions — never actually executed as its own statement |
| `secure` | Parsed as an access modifier but never enforced at runtime |
| `pure` | Parsed as a function modifier but never enforced (no purity checking) |
| `view` | Parsed as a modifier but never enforced (no state-change prevention) |
| `payable` | Parsed as a modifier but never enforced (no value transfer checking) |

**Estimated code reduction:** ~300–500 lines of stub implementations and parser rules.

---

## 🟡 Medium Confidence — Consider Removing

These have minimal functionality or overlap with other features.

| Keyword | Reason |
|---------|--------|
| `event` | Used only as a label/decorator in `emit event ...` — no event system dispatch |
| `screen` | Only prints a placeholder message — no real graphics rendering |
| `color` | Minimal stub — no actual color rendering pipeline |
| `canvas` | Stub — no actual canvas drawing support |
| `graphics` | Stub — no real graphics engine |
| `animation` | Stub — no animation framework |
| `clock` | Stub — no timing/animation loop |
| `modifier` | No Solidity-like modifier system implemented |
| `middleware` | Parsed but no HTTP middleware pipeline exists |
| `auth` | Parsed but no authentication middleware exists |
| `throttle` | Builtin function exists but keyword form is unused |
| `cache` | Builtin function exists but keyword form is unused |

---

## 🟢 Keep — Actively Used

These keywords have meaningful implementations and should be retained:

**Core Language:** `let`, `const`, `print`, `return`, `if`, `elif`, `else`, `while`, `for`, `each`, `in`,
`function`, `action`, `lambda`, `break`, `continue`, `true`, `false`, `null`

**Modules:** `use`, `export`, `import`, `from`, `as`

**Error Handling:** `try`, `catch`, `finally`, `throw`

**OOP:** `entity`, `contract`, `state`, `enum`, `interface`, `implements`, `this`

**Security:** `protect`, `verify`, `audit`, `restrict`, `sandbox`, `trail`,
`capability`, `grant`, `revoke`, `validate`, `sanitize`, `seal`, `immutable`

**Async:** `async`, `await`, `channel`, `send`, `receive`, `atomic`

**Events:** `emit`

**Logic:** `and`, `or`, `not`

**Performance:** `native`, `gc`, `inline`, `buffer`, `simd`

**Other:** `match`, `case`, `default`, `defer`, `pattern`, `stream`, `watch`,
`debug`, `log`, `data`, `external`, `inject`, `module`, `package`, `using`,
`type_alias`, `protocol`, `public`, `private`, `sealed`

**Blockchain:** `ledger`, `persistent`, `require`, `revert`, `limit`

---

## Removal Process

If you decide to remove any keywords:

1. Remove the token constant from `src/zexus/zexus_token.py`
2. Remove the keyword mapping from `src/zexus/lexer.py` (`_KEYWORDS` dict)
3. Remove from `_STRICT_KEYWORDS` if present
4. Remove from `statement_starters` set in `src/zexus/parser/strategy_structural.py`
5. Remove parser handlers in `strategy_context.py` and `parser.py`
6. Remove evaluator handlers in `evaluator/statements.py`
7. Remove any VM compiler handlers in `vm/compiler.py`
8. Run tests to ensure nothing breaks: `python -m pytest tests/ -x -q`
