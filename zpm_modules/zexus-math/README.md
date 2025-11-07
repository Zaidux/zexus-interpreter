# Zexus Math Library 🧮

A comprehensive mathematical standard library for the Zexus programming language, featuring advanced numerical computing, linear algebra, signal processing, machine learning, cryptography, and more.

## 🚀 Features

### **Core Mathematics**
- **Complex Numbers**: Full complex arithmetic with polar coordinates
- **Linear Algebra**: Matrices, determinants, inverses, eigenvalues, SVD
- **Calculus**: Derivatives, integrals, root finding, differential equations
- **Special Functions**: Gamma, Bessel, Error functions

### **Advanced Computing**
- **Numerical Optimization**: Gradient descent, linear programming
- **Signal Processing**: FFT, digital filters, convolution, windowing
- **Machine Learning**: Neural networks, clustering, regression
- **Cryptography**: RSA, elliptic curves, modular arithmetic
- **Geometric Algebra**: Multivector calculus, rotations

### **Expert Features**
- **Numerical Analysis**: Condition numbers, pseudo-inverses, stability
- **Symbolic Mathematics**: Computer algebra system, symbolic differentiation
- **Mathematical Physics**: Quantum mechanics, general relativity
- **High-Performance**: GPU acceleration, cache optimization, sparse matrices

### **Zexus-Specific Features**
- **Async Math**: Parallel computation with async/await
- **Event-Driven**: Progress tracking with event system
- **Protocol-Based**: Mathematical abstractions using protocols
- **Renderer Integration**: Function plotting and visualization

## 📦 Installation

```zexus
use "zexus-math" as math
```

🎯 Quick Start

Basic Arithmetic & Algebra

```zexus
use "zexus-math" as math

// Complex numbers
let z1 = math.complex(3, 4)
let z2 = math.complex(1, -2)
let sum = z1.add(z2)  // 4 + 2i

// Matrix operations
let A = math.matrix(2, 2, [1, 2, 3, 4])
let det = A.determinant()  // -2.0
let inv = A.inverse()      // [[-2, 1], [1.5, -0.5]]
```

Calculus & Analysis

```zexus
// Define a function
let f = math.Polynomial{coefficients: [1, -2, 1]}  // x² - 2x + 1

// Find roots
let root = math.newton_raphson(f, 0.5)  // 1.0

// Calculate derivatives
let slope = math.derivative(f, 2.0)     // 2.0

// Definite integrals
let area = math.integrate(f, 0, 1)      // ~0.333
```

Advanced Applications

```zexus
// Signal processing with windowing
let signal = [sin(2*math.PI*5*t/64) for t in range(0, 64)]
let spectrum = math.fft_with_window(signal, "hann")

// Machine learning clustering
let clusters = math.kmeans(data_points, 3)

// Cryptography - RSA keys
let keys = math.generate_rsa_keys(2048)

// Symbolic mathematics
let x = math.SymbolicVariable{name: "x"}
let expr = math.SymbolicAdd{left: x, right: math.SymbolicConstant{value: 1}}
let derivative = math.symbolic_derivative(expr, "x")  // 1
```

Async Mathematical Computing

```zexus
action async real_time_analysis() {
    // Parallel integration
    let result = await math.parallel_integrate(f, 0, 1, 8)
    
    // Monte Carlo methods
    let mc_result = await math.monte_carlo_integrate(f, 0, 1, 100000)
    
    // Real-time signal processing
    while true {
        let spectrum = math.fft(live_signal)
        await math.sleep(0.1)
    }
}
```

📚 Complete Module Structure

```
zexus-math/
├── index.zx                      # Main entry point
├── README.md                     # Documentation
├── math/
│   ├── core.zx                   # Constants, basic math, utilities
│   ├── complex.zx                # Complex number system
│   ├── linalg.zx                 # Linear algebra fundamentals
│   ├── calculus.zx               # Differentiation, integration
│   ├── optimization.zx           # Numerical optimization
│   ├── signal.zx                 # Signal processing, FFT, filters
│   ├── diffeq.zx                 # Differential equations
│   ├── ml.zx                     # Machine learning primitives
│   ├── crypto.zx                 # Cryptographic mathematics
│   ├── geometric.zx              # Geometric algebra
│   ├── async.zx                  # Async math operations
│   ├── stats.zx                  # Statistics and probability
│   ├── special.zx                # Special functions
│   ├── numerical.zx              # Numerical stability, error analysis
│   ├── advanced_linalg.zx        # SVD, eigenvalues, sparse matrices
│   ├── interpolation.zx          # Interpolation, approximation
│   ├── physics.zx                # Mathematical physics
│   ├── symbolic.zx               # Computer algebra system
│   ├── performance.zx            # High-performance math, GPU
│   └── validation.zx             # Testing & validation suite
└── examples/
    ├── basic_usage.zx            # Basic examples
    ├── advanced_calculator.zx    # Advanced usage
    ├── expert_calculator.zx      # Expert-level features
    └── unified_usage.zx          # Complete integration examples
```

🎨 Visualization

```zexus
// Function plotting
await math.plot_function(f, -10, 10)

// 3D surface plots
await math.plot_surface(math.lorenz_attractor)
```

🔧 Advanced Usage

Custom Mathematical Functions

```zexus
contract MyFunction implements math.MathFunction {
    action evaluate(x: float) -> float {
        return sin(x) * exp(-x*x)
    }
    
    action derivative() -> MathFunction {
        // Implement analytical derivative
        return MyFunctionDerivative{}
    }
}
```

Event-Driven Computation

```zexus
// Track convergence progress
math.on_math_event("progress", action(event) {
    print("Operation: " + event.operation + " - " + string(event.progress * 100) + "%")
})

let solution = await math.iterative_solver(f, initial_guess)
```

High-Performance Computing

```zexus
// Sparse matrix operations for large datasets
let A = math.sparse_matrix(10000, 10000)
A.set(0, 0, 1.0)
A.set(9999, 9999, 1.0)

// Cache-optimized matrix multiplication
let C = math.cache_optimized_matmul(A, B, 64)

// GPU acceleration (when available)
math.set_math_config("enable_gpu", true)
let result = math.gpu_matrix_multiply(large_A, large_B)
```

📈 Performance Features

· Optimized algorithms for numerical stability
· Async parallel computation for heavy workloads
· Memory-efficient matrix operations with sparse support
· Real-time capable signal processing
· GPU acceleration for large-scale computations
· Cache-aware algorithms for optimal performance

🤝 Contributing

1. Follow Zexus protocol-based design patterns
2. Include async versions of computationally intensive functions
3. Add event emission for long-running operations
4. Provide comprehensive mathematical documentation
5. Include usage examples
6. Add validation tests for numerical stability

📄 License

MIT License - Feel free to use in your Zexus projects!

---

Zexus Math Library v2.0.0 - Enterprise-grade mathematical computing for Zexus! 🚀

Now with advanced numerical analysis, symbolic mathematics, and high-performance computing capabilities.

```

## **Key Points About Zexus Async/Sync:**

1. **Synchronous**: Default, blocking, predictable
2. **Asynchronous**: Opt-in with `async/await`, non-blocking, concurrent  
3. **Seamless Integration**: Can mix sync and async code
4. **Real Concurrency**: Not just I/O - true parallel execution
5. **Events + Async**: Reactive programming patterns