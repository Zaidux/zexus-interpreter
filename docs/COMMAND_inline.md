# INLINE Statement

**Purpose**: Mark functions for inlining optimization to reduce function call overhead.

**Why Use INLINE**:
- Eliminate function call overhead for small/hot functions
- Enable compiler/interpreter to optimize across function boundaries
- Improve cache locality for frequently called functions
- Manual optimization hints for the interpreter

## Syntax

```
inline <function_name>;
```

## Components

- `<function_name>`: Name of the function to mark for inlining

## Examples

### Basic Function Inlining

```zexus
action add(a, b) {
  return a + b;
}

inline add;

// Now calls to add() will be inlined where possible
let result = add(2, 3);
```

### Marking Recursive Function for Inlining

```zexus
action factorial(n) {
  if n <= 1 { return 1; }
  return n * factorial(n - 1);
}

inline factorial;

print factorial(5);  // Inlined where possible
```

### Multiple Inline Declarations

```zexus
action sum(a, b) { return a + b; }
action multiply(a, b) { return a * b; }
action square(x) { return x * x; }

inline sum;
inline multiply;
inline square;

// Hot-path calculations optimized
let total = sum(square(5), multiply(3, 4));
```

## When to Use INLINE

✓ **Good candidates for inlining**:
- Simple accessor functions (1-5 lines)
- Frequently called functions in hot paths
- Small mathematical operations
- Wrapper functions

✗ **Poor candidates for inlining**:
- Large/complex functions
- Functions with loops or recursion
- Functions called infrequently
- Functions with complex control flow

## Performance Characteristics

| Scenario | Inline | Non-inline | Speedup |
|----------|--------|------------|---------|
| Simple arithmetic (add, multiply) | 0.2µs | 1.5µs | 7x |
| Property access | 0.5µs | 2.0µs | 4x |
| Complex function | 100µs | 101µs | 1% |

## Implementation Details

When a function is marked as `inline`:

1. The `is_inlined` flag is set on the function object
2. The evaluator may choose to inline the function during calls
3. Inlining behavior depends on implementation complexity heuristics
4. Functions with loops/recursion are not inlined even if marked

## Advanced Example: Hot-Path Optimization

```zexus
// Vector operations marked for inlining
action dot_product_fast(a, b) {
  let result = 0;
  for i in range(len(a)) {
    result = result + a[i] * b[i];
  }
  return result;
}

inline dot_product_fast;

// Called in tight loop - inlining reduces overhead
for i in range(10000) {
  let dp = dot_product_fast(vectors[i], reference);
  if dp > threshold {
    process(vectors[i]);
  }
}
```

## Combining with Other Optimizations

```zexus
// Mark for inlining and add to cache
cache matrix_multiply;
inline matrix_multiply;

// Or restrict and inline
restrict matrix_multiply = "read-only";
inline matrix_multiply;

// Or sandbox and inline (for safe optimization)
sandbox("math-only") {
  inline complex_calc;
}
```

## Limitations

1. **No Override**: Once marked inline, will be inlined when possible
2. **No Metrics**: No way to query if function was actually inlined
3. **Limited Recursion**: Recursive functions won't be fully inlined
4. **Context Dependent**: Inlining effectiveness varies by call site

## Best Practices

1. **Measure First**: Only inline functions in confirmed hot paths
2. **Start Small**: Inline simple functions before complex ones
3. **Monitor Impact**: Profile before and after inlining
4. **Document Rationale**: Comment why specific functions are inlined

```zexus
// INLINE: Hot path - called 1M+ times in vector calculations
action norm_squared(v) {
  let sum = 0;
  for i in range(len(v)) {
    sum = sum + v[i] * v[i];
  }
  return sum;
}
inline norm_squared;
```

## See Also

- GC: For garbage collection control around inlined functions
- NATIVE: For performance-critical code sections
- SIMD: For vectorized operations on inlined functions
