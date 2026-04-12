# How Zexus Parses and Evaluates Programs

This document explains the complete execution pipeline of the Zexus programming language — from raw source code to final output. It covers every phase in detail, similar to how Python's documentation explains its execution model.

## Overview

When you run `zx run program.zx`, Zexus processes your code through **6 phases**:

```
Source Code → Lexer → Parser → AST → Evaluator/VM → Output
      ↑          ↑        ↑       ↑         ↑
   1. Read   2. Tokenize 3. Parse 4. Tree  5. Execute
```

---

## Phase 1: Source Reading & Validation

**File:** `src/zexus/cli/main.py`

The CLI reads the `.zx` file and performs structural validation:

1. **File flags** (`@zexus` comments) are extracted for execution config
2. **Brace balance check** — ensures every `{`, `[`, `(` has a matching closer. If not, Zexus stops immediately with a clear error pointing to the unmatched brace (like Python stops on unmatched parentheses)
3. **Syntax validation** — the `SyntaxValidator` checks for common mistakes and can auto-fix some issues

```
// This will be caught immediately:
action greet(name) {
    print("Hello " + name)
// ← Missing closing brace!
```

**Result:** `BraceMismatchError: Unclosed '{' — expected closing '}' before end of file`

---

## Phase 2: Lexical Analysis (Tokenization)

**File:** `src/zexus/lexer.py`

The **Lexer** (also called tokenizer or scanner) reads the source code character by character and produces a stream of **tokens**. Each token has a type and a literal value.

```zexus
let name = "Alice"
```

Becomes the token stream:

```
LET("let") → IDENT("name") → ASSIGN("=") → STRING("Alice")
```

Key behaviors:
- **Keywords** like `let`, `const`, `if`, `action`, `entity`, `for`, `while`, `match`, `contract` are recognized as distinct token types
- **String literals** support `"double"`, `'single'`, and `` `template` `` quotes
- **Numbers** are distinguished as `INT` or `FLOAT`
- **Operators** like `+=`, `-=`, `**`, `<=`, `!=`, `??` are single tokens
- **Comments** (`//` single-line, `/* */` multi-line) are skipped

The lexer **does not** validate grammar — it only produces tokens. Invalid tokens (like `@#$`) become `ILLEGAL` tokens that the parser catches.

---

## Phase 3: Parsing (AST Construction)

**File:** `src/zexus/parser/parser.py`

The **UltimateParser** takes the token stream and builds an **Abstract Syntax Tree (AST)** — a tree structure that represents the program's logical structure.

### Multi-Strategy Architecture

Zexus uses a unique **multi-strategy parsing** approach with 3 strategies:

1. **Structural Analyzer** (`strategy_structural.py`) — First pass. Scans tokens to identify top-level blocks (functions, loops, if-statements). Creates a "block map" of the program structure.

2. **Context Stack Parser** (`strategy_context.py`) — Uses the block map to parse each block with full context. Handles complex nested structures, compound assignments, and advanced patterns.

3. **Traditional Pratt Parser** (fallback) — A classic recursive-descent parser with operator precedence (Pratt parsing). Used as a fallback and for expression parsing.

### How Parsing Works

The parser processes tokens using a **statement dispatch table** — an O(1) lookup that maps token types to parsing functions:

```python
LET    → parse_let_statement()
IF     → parse_if_statement()
FOR    → parse_for_each_statement()
ACTION → parse_action_statement()
ENTITY → parse_entity_statement()
# ... 100+ statement types
```

For expressions, it uses **Pratt parsing** with precedence levels:

```
LOWEST < TERNARY < ASSIGN < NULLISH < LOGICAL < EQUALS < COMPARISON < SUM < PRODUCT < PREFIX < CALL
```

This means `2 + 3 * 4` correctly parses as `2 + (3 * 4)` because `PRODUCT > SUM`.

### Example AST

```zexus
let x = 2 + 3 * 4
```

Produces:

```
Program
└── LetStatement
    ├── name: "x"
    └── value: InfixExpression
        ├── operator: "+"
        ├── left: IntegerLiteral(2)
        └── right: InfixExpression
            ├── operator: "*"
            ├── left: IntegerLiteral(3)
            └── right: IntegerLiteral(4)
```

### Error Recovery

If the parser encounters an error, it uses the **ErrorRecoveryEngine** to:
1. Skip to the next recognizable statement boundary
2. Continue parsing the rest of the file
3. Report all errors at once (not just the first one)

This is similar to how modern compilers report multiple errors in a single pass.

---

## Phase 4: Static Analysis (Optional)

**File:** `src/zexus/type_checker.py`

Before execution, Zexus optionally runs a **static type checker** that walks the AST and identifies:
- Type mismatches
- Unused variables
- Unreachable code

This phase produces warnings but does not block execution (similar to TypeScript's approach).

---

## Phase 5: Evaluation / Execution

Zexus supports **two execution engines**:

### 5a: Tree-Walk Evaluator (Default)

**File:** `src/zexus/evaluator/core.py`

The evaluator recursively walks the AST, evaluating each node:

```python
# Simplified logic:
def eval_node(node, env):
    if isinstance(node, LetStatement):
        value = eval_node(node.value, env)
        env.set(node.name, value)
    elif isinstance(node, IfExpression):
        condition = eval_node(node.condition, env)
        if is_truthy(condition):
            return eval_node(node.consequence, env)
        elif node.alternative:
            return eval_node(node.alternative, env)
    elif isinstance(node, CallExpression):
        fn = eval_node(node.function, env)
        args = [eval_node(a, env) for a in node.arguments]
        return apply_function(fn, args)
    ...
```

Key behaviors:

- **Environment chain** — Variables are stored in nested `Environment` objects. Each scope (function, block) gets its own environment linked to its parent, creating a scope chain. When looking up a variable, Zexus walks up the chain until it finds it or reaches the global scope.

- **Eager evaluation** — Expressions are evaluated immediately when encountered (unlike lazy languages like Haskell).

- **Sequential execution** — Statements in a block execute top-to-bottom. The program runs until all statements are executed or an error/return is encountered.

- **Error propagation** — When an error occurs (like dividing by zero or accessing an undefined variable), it creates an `EvaluationError` that propagates up through the call stack. This is similar to Python's exception propagation — the first unhandled error stops execution of the current scope.

- **Return value unwrapping** — When a `return` statement is hit inside a function, the value is wrapped in a `ReturnValue` object that bubbles up through all nested blocks until the function boundary unwraps it. This is how `return` works inside nested if/for blocks.

### 5b: Bytecode VM (High-Performance)

**File:** `src/zexus/vm/vm.py`, `src/zexus/vm/compiler.py`

For performance-critical code, Zexus can compile the AST to **bytecode** and execute it on a stack-based **Virtual Machine**:

```
AST → Bytecode Compiler → Instructions → VM Stack Machine
```

The VM supports multiple modes:
- **AUTO** — Automatically decide per-expression whether to use VM or tree-walk
- **FULL** — Compile everything to bytecode
- **HYBRID** — Use VM for hot paths, tree-walk for complex features

The bytecode compiler produces instructions like:
```
LOAD_CONST 2
LOAD_CONST 3
LOAD_CONST 4
MULTIPLY
ADD
STORE_NAME "x"
```

---

## Phase 6: Output & Error Reporting

**File:** `src/zexus/error_reporter.py`

When an error occurs at any phase, Zexus produces rich, context-aware error messages:

```
ERROR: NameError[NAME]
  → program.zx:5:10

  5 | print(naem)
              ^

  Identifier 'naem' not found

  💡 Suggestion: Did you mean 'name'?
```

Error features:
- **Source context** — Shows the exact line and points to the error column
- **"Did you mean?"** — Uses Levenshtein distance to suggest similar variable/function names
- **Category tagging** — Distinguishes user code errors from internal interpreter bugs
- **Suggestion system** — Provides actionable fix suggestions for common mistakes

---

## Execution Flow Diagram

```
┌─────────────────┐
│  Source (.zx)    │
└────────┬────────┘
         │
    ┌────▼────┐
    │  Lexer  │  Characters → Tokens
    └────┬────┘
         │
    ┌────▼────────────────┐
    │  Structural Analyzer │  Tokens → Block Map
    └────┬────────────────┘
         │
    ┌────▼────────────────┐
    │  Context Stack Parser│  Block Map + Tokens → AST
    └────┬────────────────┘
         │
    ┌────▼──────────────┐
    │  Type Checker      │  AST → Diagnostics (optional)
    └────┬──────────────┘
         │
    ┌────▼───────────────────────────────┐
    │        Execution Engine             │
    │  ┌──────────┐   ┌───────────────┐  │
    │  │ Tree-Walk │   │ Bytecode VM   │  │
    │  │ Evaluator │   │ (fast path)   │  │
    │  └──────────┘   └───────────────┘  │
    └────┬───────────────────────────────┘
         │
    ┌────▼────┐
    │  Output │  print(), return values, errors
    └─────────┘
```

---

## Key Differences from Python

| Aspect | Python | Zexus |
|--------|--------|-------|
| Blocks | Indentation-based | Curly braces `{ }` |
| Parsing | Single-pass PEG parser | Multi-strategy (structural + context + Pratt) |
| Errors stop execution? | First unhandled exception stops | Same — first unhandled error propagates up |
| Variable declaration | Just assign: `x = 5` | Must declare: `let x = 5` |
| Functions | `def name():` | `action name() { }` or `function name() { }` |
| Classes | `class Name:` | `entity Name { }` |
| Type checking | Optional (mypy) | Built-in optional static checker |
| VM | CPython bytecode VM | Optional bytecode VM with auto/hybrid modes |

---

## Common Error Scenarios

### 1. Undefined Variable
```zexus
print(naem)  // 'naem' is not declared
```
**Error:** `Identifier 'naem' not found` → **Suggestion:** `Did you mean 'name'?`

### 2. Unmatched Braces
```zexus
action greet() {
    print("hello")
```
**Error:** `Unclosed '{' — expected closing '}' before end of file`

### 3. Unknown Function
```zexus
let x = squrt(16)  // Typo: should be sqrt
```
**Error:** `Identifier 'squrt' not found` → **Suggestion:** `Did you mean the built-in function 'sqrt'?`

### 4. Type Error
```zexus
let x = "hello" + 5  // Can't add string and number
```
**Error:** `Type mismatch: cannot apply '+' to STRING and INTEGER`

### 5. Division by Zero
```zexus
let x = 10 / 0
```
**Error:** `Division by zero`

### 6. Not a Function
```zexus
let x = 42
x(10)  // 42 is not callable
```
**Error:** `Not a function: INTEGER` → **Suggestion:** `'42' is a INTEGER, not a function.`
