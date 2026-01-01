# Resource Limits - Security Guide

## Overview

Zexus provides automatic resource limits to prevent resource exhaustion attacks and accidental infinite loops. These limits are enforced at runtime and protect against:

- Infinite loops (via iteration limits)
- Stack overflow (via call depth limits)
- Denial-of-service attacks (via timeout protection)
- Memory exhaustion (optional, via memory limits)

**Security Fix #7: Resource Limits**

---

## Default Limits

| Resource | Default Limit | Purpose |
|----------|--------------|---------|
| **Loop Iterations** | 1,000,000 | Prevents infinite loops |
| **Call Depth** | 100 nested calls | Prevents stack overflow |
| **Execution Timeout** | 30 seconds | Prevents DoS (disabled by default) |
| **Memory Usage** | 500 MB | Prevents memory exhaustion (disabled by default) |

---

## Loop Iteration Limits

### Automatic Protection

Every loop (while, foreach) is automatically tracked:

```zexus
# Safe: Normal loop
let i = 0
while (i < 1000) {
    i = i + 1
}  # ✓ Works fine

# Dangerous: Exceeds limit
let j = 0
while (j < 2000000) {
    j = j + 1  
}  # ❌ ERROR: Iteration limit exceeded: 1,000,000 iterations
```

### Error Message

```
Iteration limit exceeded: 1,000,000 iterations
This prevents infinite loops and resource exhaustion.

Suggestion: Review your loop conditions or increase the limit with:
  zx-run --max-iterations 10000000 script.zx
```

### Best Practices

1. **Use require() to validate bounds**
   ```zexus
   action process_batch(items) {
       require(len(items) <= 100000, "Batch too large")
       
       foreach item in items {
           process(item)
       }
   }
   ```

2. **Break large operations into chunks**
   ```zexus
   action process_large_dataset(data) {
       let chunk_size = 10000
       let i = 0
       
       while (i < len(data)) {
           let chunk = slice(data, i, i + chunk_size)
           process_chunk(chunk)
           i = i + chunk_size
       }
   }
   ```

3. **Add progress tracking**
   ```zexus
   let total = 500000
   let i = 0
   
   while (i < total) {
       if (i % 50000 == 0) {
           print "Progress: " + string(i) + " / " + string(total)
       }
       i = i + 1
   }
   ```

---

## Call Depth Limits

### Automatic Protection

Every function/action call is tracked:

```zexus
# Safe: Normal recursion
action factorial(n) {
    if (n <= 1) { return 1 }
    return n * factorial(n - 1)
}

let result = factorial(10)  # ✓ Works fine (10 nested calls)

# Dangerous: Exceeds limit
action deep_recursion(n) {
    if (n <= 0) { return 0 }
    return deep_recursion(n - 1) + 1
}

let bad = deep_recursion(200)  # ❌ ERROR: Call depth limit exceeded: 100 nested calls
```

### Error Message

```
Call depth limit exceeded: 100 nested calls (infinite_recursion)
This prevents stack overflow from excessive recursion.

Suggestion: Review your recursion or increase limit with:
  zx-run --max-call-depth 5000 script.zx
```

### Best Practices

1. **Use iteration instead of recursion for large datasets**
   ```zexus
   # Instead of recursive sum:
   action recursive_sum(arr, index) {
       if (index >= len(arr)) { return 0 }
       return arr[index] + recursive_sum(arr, index + 1)
   }
   
   # Use iterative sum:
   action iterative_sum(arr) {
       let total = 0
       foreach item in arr {
           total = total + item
       }
       return total
   }
   ```

2. **Add base case validation**
   ```zexus
   action safe_factorial(n) {
       require(n >= 0 && n <= 50, "Factorial input out of safe range")
       
       if (n <= 1) { return 1 }
       return n * safe_factorial(n - 1)
   }
   ```

3. **Use tail recursion when possible**
   ```zexus
   action tail_factorial(n, acc) {
       if (n <= 1) { return acc }
       return tail_factorial(n - 1, n * acc)
   }
   
   action factorial(n) {
       return tail_factorial(n, 1)
   }
   ```

---

## Execution Timeout (Optional)

Timeout protection can be enabled to limit script execution time:

```bash
# Enable 10-second timeout
zx-run --timeout 10 script.zx
```

This is useful for:
- Web servers processing untrusted scripts
- CI/CD pipelines with time limits
- Testing infrastructure

**Note:** Timeout is disabled by default and only works on Linux/Unix systems (uses SIGALRM).

---

## Memory Limits (Optional)

Memory limits can be enabled to prevent memory exhaustion:

```bash
# Enable 100MB memory limit
zx-run --max-memory 100 script.zx
```

**Note:** Requires `psutil` package:
```bash
pip install psutil
```

---

## Configuring Limits

### Command-Line Options

```bash
# Increase iteration limit
zx-run --max-iterations 10000000 script.zx

# Increase call depth
zx-run --max-call-depth 500 script.zx

# Enable and configure timeout
zx-run --timeout 60 script.zx

# Enable and configure memory limit
zx-run --max-memory 1000 script.zx

# Combine multiple limits
zx-run --max-iterations 5000000 --max-call-depth 200 --timeout 30 script.zx
```

### Programmatic Configuration

```python
from zexus.evaluator import Evaluator
from zexus.evaluator.resource_limiter import ResourceLimiter

# Create custom resource limiter
limiter = ResourceLimiter(
    max_iterations=5_000_000,
    max_call_depth=200,
    timeout_seconds=60,
    enable_timeout=True
)

# Create evaluator with custom limits
evaluator = Evaluator(resource_limiter=limiter)
```

---

## Real-World Examples

### Example 1: Web Server Processing User Scripts

```zexus
# Server enforces strict limits for untrusted code
action process_user_script(code) {
    # Limits:
    # - 100,000 iterations (0.1M)
    # - 50 call depth
    # - 5 second timeout
    
    let result = eval_user_code(code)
    return result
}
```

### Example 2: Data Processing Pipeline

```zexus
# Process large dataset in chunks
action process_dataset(filename) {
    let data = load_file(filename)
    require(len(data) <= 10000000, "Dataset too large")
    
    let results = []
    let chunk_size = 50000
    let i = 0
    
    while (i < len(data)) {
        let chunk = slice(data, i, i + chunk_size)
        let processed = process_chunk(chunk)
        results = concat(results, processed)
        
        i = i + chunk_size
        
        # Progress tracking
        if (i % 500000 == 0) {
            print "Processed: " + string(i) + " records"
        }
    }
    
    return results
}
```

### Example 3: Smart Contract with Gas-Like Limits

```zexus
contract TokenTransfer {
    action batch_transfer(recipients, amounts) {
        # Limit batch size to prevent resource exhaustion
        require(len(recipients) <= 100, "Batch too large")
        require(len(recipients) == len(amounts), "Mismatched arrays")
        
        let i = 0
        while (i < len(recipients)) {
            transfer(recipients[i], amounts[i])
            i = i + 1
        }
    }
}
```

---

## Security Implications

### Prevents Denial-of-Service

```zexus
# BEFORE (vulnerable):
action evil_loop() {
    while (true) {
        # Infinite loop - DoS attack!
    }
}

# AFTER (protected):
action evil_loop() {
    while (true) {
        # ✓ Automatically stopped at 1,000,000 iterations
    }
}
```

### Prevents Stack Overflow

```zexus
# BEFORE (vulnerable):
action stack_overflow(n) {
    return stack_overflow(n + 1)  # Crashes interpreter!
}

# AFTER (protected):
action stack_overflow(n) {
    return stack_overflow(n + 1)  # ✓ Stops at 100 calls
}
```

### Resource Usage Statistics

```python
from zexus.evaluator import Evaluator

evaluator = Evaluator()
# ... run code ...

stats = evaluator.resource_limiter.get_stats()
print(f"Iterations used: {stats['iterations']:,} / {stats['max_iterations']:,}")
print(f"Call depth: {stats['call_depth']} / {stats['max_call_depth']}")
print(f"Iteration usage: {stats['iteration_percent']:.1f}%")
```

---

## Common Patterns

### Pattern 1: Batch Processing with Progress

```zexus
action process_with_progress(items) {
    let total = len(items)
    let processed = 0
    
    foreach item in items {
        process_item(item)
        processed = processed + 1
        
        if (processed % 1000 == 0) {
            print "Progress: " + string(processed) + " / " + string(total)
        }
    }
}
```

### Pattern 2: Safe Recursive Function

```zexus
action safe_fibonacci(n, depth) {
    require(depth < 50, "Recursion too deep")
    
    if (n <= 1) { return n }
    return safe_fibonacci(n - 1, depth + 1) + safe_fibonacci(n - 2, depth + 1)
}

action fibonacci(n) {
    return safe_fibonacci(n, 0)
}
```

### Pattern 3: Chunked Data Processing

```zexus
action process_in_chunks(data, chunk_size) {
    let results = []
    let i = 0
    
    while (i < len(data)) {
        let end = min(i + chunk_size, len(data))
        let chunk = slice(data, i, end)
        
        let chunk_result = process_chunk(chunk)
        results = concat(results, chunk_result)
        
        i = end
    }
    
    return results
}
```

---

## Performance Considerations

Resource limits add minimal overhead:

| Operation | Overhead |
|-----------|----------|
| Loop iteration check | ~0.01% per loop |
| Call depth check | ~0.05% per function call |
| Timeout check | Negligible (signal-based) |
| Memory check | ~0.1% (if enabled) |

**Total impact:** < 1% for typical programs

---

## Comparison with Other Languages

| Language | Loop Protection | Stack Protection | Timeout | Memory Limits |
|----------|----------------|------------------|---------|---------------|
| **Zexus** | ✅ Automatic (1M) | ✅ Automatic (100) | ✅ Optional | ✅ Optional |
| Python | ❌ Manual | ⚠️ ~1000 (system) | ❌ Manual | ❌ Manual |
| JavaScript | ❌ Manual | ⚠️ Varies | ❌ Manual | ❌ Manual |
| Lua | ❌ Manual | ❌ Manual | ❌ Manual | ❌ Manual |
| Solidity | ✅ Gas limits | ✅ Gas limits | ✅ Gas limits | ✅ Gas limits |

Zexus provides **automatic protection** similar to Solidity's gas system, but simpler and more developer-friendly.

---

## Troubleshooting

### Q: My legitimate loop is hitting the limit

A: Increase the iteration limit:
```bash
zx-run --max-iterations 10000000 script.zx
```

Or restructure your code to use chunking.

### Q: My recursion is hitting the call depth limit

A: Either:
1. Increase the limit: `zx-run --max-call-depth 200 script.zx`
2. Convert to iteration
3. Add explicit depth tracking with `require()`

### Q: Can I disable resource limits entirely?

A: No. Resource limits are a core security feature and cannot be disabled. You can increase them to very high values if needed for trusted code.

### Q: Why is the call depth limit so low (100)?

A: Each Zexus function call creates multiple Python stack frames. A limit of 100 Zexus calls ≈ 500-800 Python frames, staying well below Python's ~1000 limit.

---

## Summary

Zexus's automatic resource limits:

✅ **Prevent** infinite loops and excessive recursion  
✅ **Detect** resource exhaustion attempts automatically  
✅ **Provide** clear error messages with suggestions  
✅ **Protect** production systems from DoS attacks  
✅ **Require** minimal code changes  
✅ **Add** negligible performance overhead  

Use `require()` to validate bounds and consider chunking for large datasets.

---

**Related Documentation:**
- [Security Features Guide](SECURITY_FEATURES.md)
- [Contract require() Function](CONTRACT_REQUIRE.md)
- [Quick Reference](QUICK_REFERENCE.md)
