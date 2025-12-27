# WaitGroup and Barrier Synchronization

## Overview

Zexus now includes Go-style synchronization primitives for coordinating concurrent operations.

## WaitGroup

A WaitGroup allows waiting for a collection of async operations to complete.

### API

```zexus
// Create a wait group
let wg = wait_group()

// Add delta to the counter
wg_add(wg, 2)  // Expecting 2 tasks

// Decrement counter by 1
wg_done(wg)

// Wait until counter reaches zero
wg_wait(wg)           // Block indefinitely
wg_wait(wg, 5.0)      // Wait max 5 seconds
```

### Example

```zexus
let wg = wait_group()
wg_add(wg, 3)

async action worker(id) {
    print("Worker", id, "starting")
    sleep(random() * 2)
    print("Worker", id, "done")
    wg_done(wg)
}

async worker(1)
async worker(2)
async worker(3)

print("Waiting for workers...")
wg_wait(wg)
print("All workers completed!")
```

## Barrier

A Barrier allows N tasks to synchronize at a common point.

### API

```zexus
// Create a barrier for N parties
let barrier = barrier(3)

// Wait for all parties to arrive
barrier_wait(barrier)         // Block until all arrive
barrier_wait(barrier, 10.0)   // Wait max 10 seconds

// Reset barrier for reuse
barrier_reset(barrier)
```

### Example

```zexus
let barrier = barrier(3)

async action phase_worker(id) {
    print("Worker", id, "- Phase 1")
    sleep(random())
    
    barrier_wait(barrier)  // Sync point
    
    print("Worker", id, "- Phase 2")
    sleep(random())
    
    barrier_wait(barrier)  // Another sync point
    
    print("Worker", id, "- Complete")
}

async phase_worker(1)
async phase_worker(2)
async phase_worker(3)
```

## Implementation Details

- **Thread-safe**: All operations use mutex locks
- **Timeout support**: Both wait operations support optional timeout
- **Error handling**: Returns EvaluationError on misuse
- **Reentrant**: WaitGroup supports negative deltas (for cleanup)

## Type Compatibility

The builtin functions accept both:
- Zexus Integer/Float objects (`Integer(value=2)`)
- Python int/float primitives (`2`)

This ensures compatibility with both compiled and interpreted contexts.

## Known Limitations

1. **Async scope**: Currently, async tasks may have limited access to outer-scope variables. This is a known issue being addressed.

2. **Manual management**: Unlike Go's defer-based cleanup, Zexus requires explicit wg_done() calls. Use `defer { wg_done(wg) }` for automatic cleanup.

## Future Enhancements

- [ ] Method syntax: `wg.add(1)` instead of `wg_add(wg, 1)`
- [ ] Automatic scope capture for async tasks
- [ ] WaitGroup.wait() returning error on timeout
- [ ] Barrier.parties() to query barrier size
