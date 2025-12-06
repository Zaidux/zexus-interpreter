# Zexus Performance Features Guide

This guide covers the 5 performance optimization features in Zexus: **NATIVE**, **GC**, **INLINE**, **BUFFER**, and **SIMD**.

## Overview

| Feature | Purpose | Use Case | Overhead |
|---------|---------|----------|----------|
| **NATIVE** | Call C/C++ directly | Compute-heavy, crypto, system APIs | FFI marshalling (~1-10µs) |
| **GC** | Control garbage collection | Memory-intensive workloads | Minimal (control only) |
| **INLINE** | Mark functions for inlining | Hot-path optimization | Negligible if effective |
| **BUFFER** | Direct memory access | Low-level data, binary I/O | Direct memory access (~0.1µs) |
| **SIMD** | Vectorized operations | Batch processing, math | 0-90% faster than scalar |

## Quick Start

### Calling Native C Code
```zexus
native "libmath.so", "pow"(2, 8) as result;
print result;  // 256
```

### Controlling Garbage Collection
```zexus
gc "pause";
// ... critical section without GC interruptions ...
gc "resume";
```

### Marking Functions for Inlining
```zexus
action fast_add(a, b) { return a + b; }
inline fast_add;
```

### Direct Memory Access
```zexus
buffer data = allocate(1024);
buffer data.write(0, [1, 2, 3]);
let bytes = buffer data.read(0, 3);
```

### SIMD Vector Operations
```zexus
let a = [1, 2, 3, 4];
let b = [5, 6, 7, 8];
simd a + b;  // Vectorized addition
```

## Feature Comparison

### NATIVE vs Performance

**Use NATIVE when:**
- Leveraging existing C/C++ libraries
- Need compute performance beyond interpreter capabilities
- Interfacing with system APIs
- Processing large datasets with specialized algorithms

**Example:**
```zexus
// Fast SHA256 hashing
native "libcrypto.so", "SHA256"(data) as hash;
```

### GC vs Latency

**Use GC when:**
- Minimizing pause times is critical
- Performance-testing code paths
- Managing memory in batch operations
- Debugging memory issues

**Example:**
```zexus
gc "pause";
for i in range(1000000) {
  // No GC pauses in this loop
  process(data[i]);
}
gc "resume";
```

### INLINE vs Function Call Overhead

**Use INLINE when:**
- Function appears in hot-path (called 1000s of times)
- Function is very simple (< 10 lines)
- Call overhead matters (< 1000µs total time)

**Example:**
```zexus
action norm(v) {
  let sum = 0;
  for i in range(len(v)) {
    sum = sum + v[i] * v[i];
  }
  return sum;
}
inline norm;  // Hot-path optimization
```

### BUFFER vs Memory Overhead

**Use BUFFER when:**
- Working with binary data
- Need zero-copy data passing
- Implementing custom data structures
- Memory layout matters

**Example:**
```zexus
// Efficient binary processing
buffer file = allocate(65536);
load_file("data.bin", file);
process_binary(file);
buffer file.free();
```

### SIMD vs Throughput

**Use SIMD when:**
- Processing arrays/matrices
- Data parallelism exists
- Dataset is large (1000+ elements)
- Operations are vectorizable

**Example:**
```zexus
// Matrix multiply with SIMD
simd C = matrix_multiply(A, B);
```

## Performance Tuning Methodology

### 1. Profile First
```zexus
gc "enable_debug";  // Monitor GC activity
let start = now();
// ... operation ...
let elapsed = now() - start;
print "Elapsed: " + elapsed + "ms";
gc "disable_debug";
```

### 2. Identify Bottlenecks
- Is it CPU-bound? → Use NATIVE or SIMD
- Is it memory-bound? → Use BUFFER or GC control
- Are functions called frequently? → Use INLINE

### 3. Optimize Incrementally
```zexus
// Start with GC control
gc "pause";
result = compute_intensive(data);
gc "resume";

// Then add SIMD if available
gc "pause";
simd result = compute_intensive_simd(data);
gc "resume";

// Then consider NATIVE if still needed
gc "pause";
native "libopt.so", "compute_intensive"(data) as result;
gc "resume";
```

### 4. Measure Impact
```zexus
// Baseline
let t0 = now();
action_baseline();
let baseline = now() - t0;

// With optimization
let t1 = now();
action_optimized();
let optimized = now() - t1;

print "Speedup: " + (baseline / optimized) + "x";
```

## Real-World Scenarios

### Scenario 1: Image Processing Pipeline

```zexus
action process_images(files) {
  gc "pause";
  
  for file in files {
    buffer img = allocate(3 * 1024 * 1024);  // 3MP
    load_file(file, img);
    
    // SIMD for filters
    simd apply_blur_filter(img);
    simd apply_sharpen(img);
    
    // NATIVE for advanced processing
    native "libfilter.so", "edge_detect"(img) as edges;
    
    save_file("output/" + file, img);
    buffer img.free();
  }
  
  gc "resume";
  gc "collect";
}
```

**Performance**: ~10x faster than pure Zexus

### Scenario 2: Financial Calculations

```zexus
action calculate_portfolio_returns(prices, weights) {
  gc "pause";
  gc "enable_debug";
  
  // Vectorized operations
  simd returns = price_changes(prices);
  simd weighted_returns = returns * weights;
  simd total_return = sum(weighted_returns);
  
  // Risk calculation
  simd variance = simd_variance(returns);
  simd volatility = simd_sqrt(variance);
  
  gc "disable_debug";
  gc "resume";
  
  return { return: total_return, risk: volatility };
}
```

**Performance**: ~100x faster than loops

### Scenario 3: Cryptographic Operations

```zexus
action secure_message_digest(message, algorithm) {
  // NATIVE for speed + security
  native "libcrypto.so", "digest"(message, algorithm) as hash;
  
  // SANDBOX for isolation
  sandbox("crypto") {
    native "libcrypto.so", "verify"(hash, signature) as valid;
  }
  
  return valid;
}
```

**Security + Performance**: Leverages system crypto + sandboxing

### Scenario 4: Machine Learning Inference

```zexus
action infer_batch(images, model) {
  buffer batch_data = allocate(batch_size * image_size);
  
  // Load batch into contiguous buffer
  for i in range(batch_size) {
    buffer batch_data.write(i * image_size, images[i]);
  }
  
  // SIMD matrix multiply for forward pass
  gc "pause";
  simd activations = matrix_multiply_simd(batch_data, model.weights);
  simd add_bias_simd(activations, model.bias);
  simd apply_relu(activations);
  gc "resume";
  
  buffer batch_data.free();
  return activations;
}
```

**Performance**: ~50x faster than per-image inference

## Integration Patterns

### Pattern 1: Performance + Security
```zexus
// Fast computation in isolated environment
sandbox("math-only") {
  gc "pause";
  simd result = matrix_multiply(A, B);
  gc "resume";
}
```

### Pattern 2: Optimization + Monitoring
```zexus
// INLINE function and monitor calls
inline compute;
trail *, "compute_call";

// Trace execution
gc "enable_debug";
compute(data);
gc "disable_debug";
```

### Pattern 3: Memory + Performance
```zexus
// Buffer for data layout, SIMD for operations, NATIVE for edge cases
buffer data = allocate(size);
simd process_vectorizable(data);
native "libspec.so", "process_special_case"(data) as result;
buffer data.free();
```

### Pattern 4: Layered Optimization
```zexus
// 1. GC control - eliminate pause times
gc "pause";

// 2. SIMD - vectorize operations
simd batch_transform(data);

// 3. INLINE - reduce function call overhead
inline helper_func;

// 4. BUFFER - memory layout optimization
buffer optimized = allocate(size);

// 5. NATIVE - last-resort optimization
native "libopt.so", "final_pass"(optimized) as result;

gc "resume";
```

## Anti-Patterns (What NOT to Do)

❌ **Don't use NATIVE for trivial operations**
```zexus
// BAD: Overhead exceeds benefit
native "libmath.so", "add"(1, 2) as result;
```

❌ **Don't pause GC indefinitely**
```zexus
// BAD: Memory will build up
gc "pause";
while true {
  // ... infinite loop ...
}
```

❌ **Don't inline everything**
```zexus
// BAD: May increase code size without benefit
inline simple_func;
inline complex_func;
inline rarely_called_func;
```

❌ **Don't allocate huge buffers unnecessarily**
```zexus
// BAD: Waste of memory
buffer huge = allocate(1000000000);  // 1GB!
```

❌ **Don't use SIMD for scalar data**
```zexus
// BAD: Overhead not worth it
simd x + y;  // Just use + operator for scalars
```

## Troubleshooting

### Issue: NATIVE Library Not Found
```
Error: Failed to load native library 'libmath.so'
```

**Solution:**
```bash
# Add library to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Or use full path
native "/usr/local/lib/libmath.so", "func"() as result;
```

### Issue: Buffer Crash / Segmentation Fault
```
// Uncontrolled buffer access
buffer buf = allocate(10);
buffer buf.write(1000, [1, 2, 3]);  // Out of bounds!
```

**Solution:**
```zexus
action safe_write(buf, offset, size, data) {
  if offset + len(data) > size {
    return error("Out of bounds");
  }
  buffer buf.write(offset, data);
}
```

### Issue: GC Pause Not Taking Effect
```zexus
gc "pause";
// ... GC still happening ...
gc "resume";
```

**Solution:** Ensure `pause` and `resume` are balanced. Enable debug to verify:
```zexus
gc "enable_debug";
gc "pause";
// ... code ...
gc "resume";
gc "disable_debug";
```

### Issue: SIMD Not Faster Than Scalar
**Causes:**
- Data too small (< 1000 elements)
- Operation not vectorizable
- Memory bandwidth limiting
- Cache misses from large arrays

**Solution:** Profile and verify vectorization effectiveness

## References

- [COMMAND_native.md](./COMMAND_native.md): Detailed NATIVE documentation
- [COMMAND_gc.md](./COMMAND_gc.md): Detailed GC documentation
- [COMMAND_inline.md](./COMMAND_inline.md): Detailed INLINE documentation
- [COMMAND_buffer.md](./COMMAND_buffer.md): Detailed BUFFER documentation
- [COMMAND_simd.md](./COMMAND_simd.md): Detailed SIMD documentation

## Performance Checklists

### Pre-Optimization
- [ ] Measured baseline performance
- [ ] Identified specific bottleneck
- [ ] Selected appropriate feature(s)
- [ ] Reviewed relevant documentation

### Implementation
- [ ] Added performance feature
- [ ] Verified correctness (tests pass)
- [ ] Profiled new implementation
- [ ] Documented rationale in code

### Validation
- [ ] Measured improvement
- [ ] Calculated speedup
- [ ] Verified security (if applicable)
- [ ] Documented results

## Advanced Topics

See individual documentation files for:
- FFI type marshalling in NATIVE
- GC memory layout and tuning
- Inline expansion heuristics
- Buffer alignment and layout
- SIMD instruction selection

## Summary

Zexus performance features provide multiple levers for optimization:

1. **NATIVE** - When pure Zexus is too slow
2. **GC** - When GC pauses matter
3. **INLINE** - When function call overhead matters
4. **BUFFER** - When memory layout/layout matters
5. **SIMD** - When data parallelism is available

Use profiling and measurement to guide which features to use. Start with one feature, measure impact, then add more as needed.

**Remember: First make it correct, then make it fast.** Profile first, optimize second.
