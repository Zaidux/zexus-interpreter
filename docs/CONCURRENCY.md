# Concurrency & Performance — Zexus Interpreter

Overview

This document describes the newly added concurrency features in Zexus:

- Tokens: `CHANNEL`, `SEND`, `RECEIVE`, `ATOMIC`
- AST nodes: `ChannelStatement`, `SendStatement`, `ReceiveStatement`, `AtomicStatement`
- Parser support: structural and context-aware parsing recognize and produce AST nodes for concurrency constructs
- Evaluator support: dispatch routes were added; evaluator handlers live in `src/zexus/evaluator/statements.py`
- Runtime system: `src/zexus/concurrency_system.py` contains `Channel`, `Atomic`, and `ConcurrencyManager` implementations.

Syntax Summary

- Channel declaration:
  - `channel<integer> numbers;` — unbuffered typed channel
  - `channel<string>[10] messages;` — buffered channel with capacity 10
  - `channel inbox;` — untyped channel (defaults)

- Sending to a channel:
  - `send(numbers, 42);`

- Receiving from a channel:
  - `value = receive(numbers);`
  - `receive(numbers);` (statement form)

- Atomic operations:
  - Block form:
    ```zexus
    atomic {
      counter = counter + 1;
      total = total + 1;
    };
    ```
  - Expression form:
    ```zexus
    atomic(counter = counter + 1);
    ```

Notes and current limitations

- Parser supports both structural and context-aware parsing for the above forms. Parser is tolerant but will emit parse errors (logged) when syntax is malformed.
- The concurrency runtime (`concurrency_system.py`) is a Python implementation based on `threading` primitives. It's designed to be used by the interpreter runtime and may require integration adjustments when running programs in-process.
- Tests for concurrency features have not yet been created — they will be added next.

Examples

1) Unbuffered channel:

```zexus
channel<integer> numbers;
send(numbers, 1);
let value = receive(numbers);
print value;
```

2) Buffered channel:

```zexus
channel<string>[5] messages;
send(messages, "hello");
let m = receive(messages);
print m;
```

3) Atomic block:

```zexus
atomic {
  x = x + 1;
}
```

Where to look

- Parser: `src/zexus/parser/parser.py` and `src/zexus/parser/strategy_context.py`
- AST nodes: `src/zexus/zexus_ast.py`
- Evaluator: `src/zexus/evaluator/statements.py` and `src/zexus/evaluator/core.py`
- Runtime: `src/zexus/concurrency_system.py`

If you'd like, I can now create a dedicated `src/tests/test_concurrency_features.zx` and start writing targeted tests.
