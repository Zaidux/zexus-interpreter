# Async & Concurrency — Zexus Language Reference

## Overview

Zexus supports modern concurrent programming with channels, async/await, atomic operations, and structured concurrency primitives.

### Keywords
| Keyword | Purpose |
|---------|---------|
| `async` | Mark functions/actions as asynchronous |
| `await` | Wait for async operations to complete |
| `channel` | Create message-passing channels |
| `send` | Send messages to channels |
| `receive` | Receive messages from channels |
| `atomic` | Execute operations atomically (indivisibly) |

---

## Channels

### Creating Channels
```zexus
channel<integer> numbers          // Unbuffered
channel<string> messages = 10     // Buffered (capacity 10)
```

### Send & Receive
```zexus
send(numbers, 42)
let value = receive(numbers)     // Blocks until message available

// Recommended pattern in async contexts:
let _ = send(channel, value)     // Explicit assignment avoids race conditions
```

### Close
```zexus
close_channel(numbers)
```

---

## Atomic Operations

```zexus
let counter = 0

// Single expression
atomic(counter = counter + 1)

// Block form
atomic {
    x = x + 1
    y = y + 1
}
```

---

## Async / Await

```zexus
async action fetchData(url) {
    let response = await httpGet(url)
    ret response
}

let data = await fetchData("/api/data")
```

---

## Structured Concurrency (v1.8.4+)

### select — Multiplex Across Channels
```zexus
// Python API
from concurrency_system import select, Channel

ch1 = Channel("ch1", capacity=5)
ch2 = Channel("ch2", capacity=5)
ch2.send("hello")
idx, val = select(ch1, ch2, timeout=1.0)  # Returns (1, "hello")
```

### TaskGroup — Structured Task Groups
```zexus
with TaskGroup(name="workers") as g:
    g.spawn(worker, 1)
    g.spawn(worker, 2)
    g.spawn(worker, 3)
# All tasks complete before exiting the block

print(g.results)   // Collected return values
print(g.errors)    // Any exceptions
```

### BackpressuredChannel — Overflow Policies
```zexus
ch = BackpressuredChannel("queue", capacity=100, policy=BackpressurePolicy.DROP_OLDEST)
// Policies: DROP_NEWEST, DROP_OLDEST, BLOCK, RAISE
print(ch.dropped_count)
```

### TaskContext — Cross-Task Tracing
```zexus
ctx = TaskContext(trace_id="req-123", deadline=datetime.now() + timedelta(seconds=30))
ctx.run(my_task)  // Sets context as thread-local current

// Inside my_task:
current = TaskContext.current()
print(current.trace_id)        // "req-123"
print(current.is_expired())    // False
```

---

## Patterns

### Producer-Consumer
```zexus
channel<integer> jobs = 5

async action producer() {
    let _ = send(jobs, 1)
    let _ = send(jobs, 2)
    let _ = send(jobs, 3)
}

async action consumer() {
    let job = receive(jobs)
    print("Processing: " ++ str(job))
}
```

### Pipeline
```zexus
channel<integer> stage1
channel<integer> stage2

send(stage1, 10)
let val = receive(stage1)
send(stage2, val * 2)
let result = receive(stage2)   // 20
```

---

## Runtime Implementation

- **Channel**: Thread-safe `queue.Queue` with close semantics and timeout support
- **Atomic**: Mutex-protected code regions via `threading.Lock`
- **ConcurrencyManager**: Lifecycle management for channels and atomic regions
- **select()**: Polling with exponential backoff across multiple channels
- **TaskGroup**: Thread-based structured concurrency with cancellation
- **BackpressuredChannel**: Configurable overflow policies
- **TaskContext**: `threading.local()` for context propagation

Source: `src/zexus/concurrency_system.py`
