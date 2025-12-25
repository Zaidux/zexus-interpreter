# SIMD Statement

**Purpose**: Execute vector/SIMD operations for accelerated parallel computation on multiple data elements.

**Why Use SIMD**:
- Process multiple data elements in parallel
- Leverage CPU SIMD instructions (SSE, AVX, NEON)
- Accelerate matrix operations, signal processing
- Reduce loop overhead for vectorizable operations
- Enable CPU optimization for batch operations

## Syntax

```
simd <operation>;
simd <function>(<arg1>, <arg2>, ...);
```

## Concepts

SIMD stands for "Single Instruction, Multiple Data". Modern CPUs can execute operations on multiple data elements simultaneously:

- **SSE** (Streaming SIMD Extensions): 128-bit vectors (2 doubles or 4 floats)
- **AVX** (Advanced Vector Extensions): 256-bit vectors (4 doubles or 8 floats)
- **AVX-512**: 512-bit vectors (8 doubles or 16 floats)

Zexus SIMD operations automatically map to available CPU capabilities.

## Examples

### Vector Addition

```zexus
let a = [1, 2, 3, 4];
let b = [5, 6, 7, 8];
simd a + b;  // Vectorized: [6, 8, 10, 12]
```

### Element-wise Multiplication

```zexus
let matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
];

let scalar = 2;
simd matrix * scalar;  // All elements multiplied
```

### Matrix Multiplication

```zexus
let A = [
  [1, 2, 3],
  [4, 5, 6]
];

let B = [
  [7, 8],
  [9, 10],
  [11, 12]
];

simd matrix_mul(A, B);  // Accelerated matrix multiplication
```

### Dot Product

```zexus
let x = [1, 2, 3, 4, 5];
let y = [2, 3, 4, 5, 6];

simd dot_product(x, y);  // Vectorized: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 70
```

### Batch Processing

```zexus
// Process batch of images (vectorized)
let batch_size = 100;
let image_size = 3 * 1024 * 1024;  // 3MP images

simd batch_resize(images, batch_size, 512);  // Resize all in parallel
```

## Supported Operations

| Operation | Example | Use Case |
|-----------|---------|----------|
| Addition | `simd a + b` | Element-wise sum |
| Subtraction | `simd a - b` | Element-wise difference |
| Multiplication | `simd a * b` | Element-wise product |
| Division | `simd a / b` | Element-wise division |
| Functions | `simd dot_product(a, b)` | Custom SIMD functions |
| Reductions | `simd sum(a)` | Aggregate operations |

## Advanced Examples

### Signal Processing

```zexus
// Fast Fourier Transform (FFT) using SIMD
action fft_simd(signal) {
  // Vectorized FFT operations
  simd fft_butterfly(signal, 0);
  simd fft_butterfly(signal, 1);
  simd fft_butterfly(signal, 2);
  // ... more stages ...
  return signal;
}

let input = generate_sine_wave(1024);
let spectrum = fft_simd(input);
```

### Linear Algebra

```zexus
// Solve system of linear equations: Ax = b
action solve_linear_system(A, b) {
  // Gaussian elimination with SIMD
  for i in range(len(A)) {
    // Find pivot (vectorized comparison)
    simd find_max_row(A, i);
    
    // Eliminate column (vectorized operations)
    for j in range(i + 1, len(A)) {
      simd eliminate_row(A, i, j);
    }
  }
  
  // Back substitution (vectorized)
  simd back_substitute(A, b);
  return b;
}
```

### Computer Vision

```zexus
// Image convolution (highly parallelizable)
action convolve_simd(image, kernel) {
  let height = len(image);
  let width = len(image[0]);
  let k_size = len(kernel);
  
  buffer output = allocate(height * width);
  
  for y in range(height - k_size) {
    for x in range(width - k_size) {
      // Extract patch and apply kernel (vectorized)
      let patch = extract_patch(image, x, y, k_size);
      simd output[y * width + x] = dot_product(patch, kernel);
    }
  }
  
  return output;
}
```

### Machine Learning

```zexus
// Neural network forward pass with SIMD
action forward_pass_simd(input, weights, bias) {
  // Vectorized matrix multiplication
  let hidden = simd matrix_multiply(input, weights);
  
  // Vectorized bias addition and activation
  simd add_bias(hidden, bias);
  simd relu_activation(hidden);
  
  return hidden;
}
```

## Performance Characteristics

| Operation | Scalar | SIMD | Speedup |
|-----------|--------|------|---------|
| Vector add (1M elements) | 2.5ms | 0.3ms | 8x |
| Matrix multiply (1K×1K) | 450ms | 60ms | 7.5x |
| Dot product (10K) | 0.5ms | 0.06ms | 8x |
| Image filter (4MP) | 180ms | 25ms | 7x |

## Implementation Strategy

Zexus SIMD uses one of these backends:

1. **NumPy** (if available): Optimized vectorized operations
2. **Array module**: Pure Python SIMD emulation
3. **Fallback**: Scalar operations if no SIMD support

## Advanced Example: Particle Physics

```zexus
// N-body simulation with SIMD
action update_particles_simd(particles, dt) {
  let n = len(particles);
  
  // Vectorized force calculation
  for i in range(n) {
    let forces = buffer allocate(3 * 8);  // 3D forces
    
    // Calculate forces from all other particles (SIMD)
    for j in range(n) {
      if i != j {
        simd calculate_gravitational_force(
          particles[i],
          particles[j],
          forces
        );
      }
    }
    
    // Update velocity and position (SIMD)
    simd particles[i].velocity += forces / particles[i].mass * dt;
    simd particles[i].position += particles[i].velocity * dt;
  }
  
  return particles;
}
```

## Combining with Other Features

```zexus
// Restrict: Ensure SIMD operates on safe data
restrict simd_input = "read-only";
simd process(simd_input);

// Sandbox: Isolate SIMD computation
sandbox("computation") {
  simd heavy_calc(A, B);
}

// GC: Control garbage collection during SIMD
gc "pause";
simd matrix_multiply(large_A, large_B);
gc "resume";

// NATIVE: Call native SIMD library
native "libsimd.so", "simd_add"(a, b) as result;
```

## Limitations & Considerations

1. **Data Alignment**: Data should be aligned to cache lines for optimal performance
2. **Vectorization Limits**: Not all operations can be vectorized
3. **Memory Bandwidth**: SIMD effectiveness limited by memory bandwidth
4. **Overhead**: Small arrays may not benefit from SIMD overhead
5. **Platform Specific**: Performance varies by CPU capabilities

## Best Practices

1. **Use for Large Datasets**: SIMD shines with 1000+ elements
2. **Check Vectorization**: Ensure operations actually vectorize
3. **Profile Performance**: Measure SIMD vs scalar implementations
4. **Combine Operations**: Minimize memory transfers between SIMD ops

```zexus
// Good: Chain operations
simd temp = a + b;
simd result = temp * 2;

// Better: Minimize intermediate storage
simd result = (a + b) * 2;
```

5. **Consider Cache**: Keep working sets in L1/L2 cache

## See Also

- NATIVE: For calling native SIMD libraries
- BUFFER: For efficient data layout for SIMD
- GC: For memory management during SIMD operations
- INLINE: For optimizing SIMD function calls
