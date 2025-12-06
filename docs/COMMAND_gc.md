# GC Statement

**Purpose**: Control garbage collection behavior for performance tuning and memory optimization.

**Why Use GC**:
- Optimize memory usage patterns
- Reduce unpredictable GC pauses
- Force collection at strategic points
- Monitor GC activity for debugging

## Syntax

```
gc "<action>";
```

## Actions

### gc "collect"
Force immediate garbage collection and return collected object count.

```zexus
let collected = gc "collect";
print "Collected " + collected + " objects";
```

### gc "pause"
Pause automatic garbage collection.

```zexus
gc "pause";
// Performance-critical section without GC interruptions
for i in range(1000000) {
  // ... heavy computation ...
}
gc "resume";
```

### gc "resume"
Resume automatic garbage collection after pausing.

```zexus
gc "pause";
// ... work without GC ...
gc "resume";
```

### gc "enable_debug"
Enable garbage collection debug output for monitoring.

```zexus
gc "enable_debug";
// GC statistics will be printed to stdout
```

### gc "disable_debug"
Disable garbage collection debug output.

```zexus
gc "disable_debug";
```

## Examples

### Explicit Collection Before Critical Section

```zexus
gc "collect";  // Clean up before memory-sensitive operation
let large_data = allocate(1000000);
process_data(large_data);
gc "collect";  // Clean up after
```

### Performance Testing with GC Control

```zexus
gc "pause";
let start = now();
// Run performance-critical code
for i in range(100000) {
  let result = complex_calculation(i);
}
let elapsed = now() - start;
gc "resume";
print "Elapsed: " + elapsed + "ms (GC disabled)";
```

### Monitoring GC Activity

```zexus
gc "enable_debug";
// ... application code ...
// GC stats printed to console
gc "disable_debug";
```

## Return Values

| Action | Returns |
|--------|---------|
| `"collect"` | Integer - number of objects collected |
| `"pause"` | String - "GC paused" |
| `"resume"` | String - "GC resumed" |
| `"enable_debug"` | String - "GC debug enabled" |
| `"disable_debug"` | String - "GC debug disabled" |

## Performance Impact

| Action | Overhead | Use Case |
|--------|----------|----------|
| `collect` | Medium (triggers GC) | End of initialization, before critical section |
| `pause` | Minimal | Performance-critical sections |
| `resume` | Minimal | After critical section |
| `enable_debug` | Low | Development/debugging |

## Best Practices

1. **Collect Before Critical Sections**: Force collection to clean up before latency-sensitive code
2. **Pause During Performance Tests**: Disable GC to get accurate performance measurements
3. **Use Debug Sparingly**: Enable debug only during development/debugging
4. **Monitor Collections**: Track GC frequency to identify memory leaks

## Advanced Example: Batch Processing with GC Control

```zexus
action process_batch(items) {
  gc "pause";
  gc "enable_debug";
  
  for item in items {
    let result = process(item);
    store_result(result);
  }
  
  gc "disable_debug";
  gc "resume";
  gc "collect";  // Final cleanup
}

process_batch(large_dataset);
```

## Notes

- GC actions return immediately; collection happens asynchronously for non-blocking behavior
- `pause` and `resume` must be balanced (each `pause` should have a corresponding `resume`)
- GC pause/resume nesting is supported for hierarchical control
- Debug output goes to stderr to avoid interfering with program output
