# DEFER Statement

**Purpose**: Register cleanup code execution for guaranteed execution when scope exits (LIFO order).

**Why Use DEFER**:
- Automatic resource cleanup (files, locks, connections)
- Guaranteed execution even if errors occur
- LIFO ordering for proper cleanup sequence
- Cleaner code than try-finally patterns
- Prevents resource leaks

## Syntax

```
defer <code_block>;
defer <expression>;
defer { multiple(); statements(); }
```

## Examples

### File Cleanup

```zexus
let file = open("data.txt");
defer file.close();
// Use file...
// file.close() guaranteed to execute
```

### Multiple Cleanups (LIFO Order)

```zexus
let lock = acquire_lock();
defer lock.release();

let file = open("data.txt");
defer file.close();

// Released in order: file.close(), then lock.release()
```

### Database Transactions

```zexus
let conn = db.connect();
defer conn.close();

let tx = conn.begin_transaction();
defer tx.rollback();

tx.execute("INSERT INTO users VALUES (...)");
tx.commit();
// Cleanup: tx.rollback() (no-op after commit), then conn.close()
```

### Memory Cleanup

```zexus
let buffer = allocate(10000);
defer free(buffer);

process_data(buffer);
// buffer guaranteed to be freed
```

### Block Syntax

```zexus
defer {
  cleanup1();
  cleanup2();
  cleanup3();
}
```

## Execution Semantics

- **LIFO**: Deferred code executes in Last-In-First-Out order
- **Guaranteed**: Executes even if current scope exits via return, error, or break
- **Scope-based**: Attached to current function/block scope
- **No arguments**: Deferred code doesn't receive arguments

## Advanced Examples

### Resource Pool with Defer

```zexus
action process_with_resource(data) {
  let resource = pool.acquire();
  defer pool.release(resource);
  
  let result = resource.process(data);
  if error(result) {
    return error("Processing failed");
    // pool.release(resource) still executes!
  }
  
  return result;
}
```

### Nested Defers

```zexus
action setup_full_stack() {
  let db = database.connect();
  defer db.close();
  
  {
    let cache = cache_layer.init();
    defer cache.shutdown();
    
    let queue = message_queue.start();
    defer queue.stop();
    
    // Work with all three...
  }
  // queue.stop(), cache.shutdown() in that order, then db.close()
}
```

### Logging with Defer

```zexus
action critical_operation(input) {
  log("Starting critical operation");
  defer log("Critical operation completed");
  
  let result = perform_risky_work(input);
  return result;
  // "Critical operation completed" logged regardless of result
}
```

## Performance Characteristics

- **Overhead**: Minimal (O(1) per defer, O(n) at scope exit)
- **Storage**: Stack-based, constant memory per defer
- **Execution**: Fast (direct call, no wrapper)

## Best Practices

1. **Pair with Acquisition**: Every acquire has a corresponding defer
2. **LIFO Awareness**: Remember reverse execution order
3. **Simple Cleanups**: Keep deferred code short and simple
4. **Error Safe**: Avoid errors in cleanup code
5. **Document Intent**: Comment why cleanup is needed

```zexus
action safe_operation(path) {
  let file = open(path);
  // DEFER: Ensure file handle is closed even on error
  defer file.close();
  
  // Safe to use file here
  return process_file(file);
}
```

## Combining with Other Features

```zexus
// With SANDBOX: Cleanup in isolated context
defer {
  sandbox("cleanup") {
    cleanup_resources();
  }
}

// With RESTRICT: Restrict cleanup access
restrict cleanup_resource = "admin-only";
defer cleanup_resource();

// With TRAIL: Log cleanup execution
trail *, "cleanup";
defer cleanup();  // Traced
```

## Common Patterns

### Mutex/Lock Protection

```zexus
action thread_safe_operation() {
  let lock = acquire_mutex();
  defer lock.unlock();
  
  // Critical section
  shared_resource.modify();
}
```

### Temporary File

```zexus
action process_file(original) {
  let temp = create_temp_file();
  defer delete_file(temp);
  
  copy_file(original, temp);
  process_in_place(temp);
  upload_result(temp);
}
```

### State Restoration

```zexus
action modify_with_restore(new_state) {
  let old_state = get_state();
  defer set_state(old_state);
  
  set_state(new_state);
  try_operation();
  // State restored if operation fails
}
```

## Limitations

- ✗ Cannot pass arguments to deferred code
- ✗ Cannot capture return value from deferred code
- ✗ Execution order is scope-based, not global
- ✗ Errors in deferred code cannot be caught here

## See Also

- TRY/CATCH: Error handling
- SANDBOX: Isolated cleanup
- RESTRICT: Controlled cleanup access
