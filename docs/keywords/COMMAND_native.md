# NATIVE Statement

**Purpose**: Call C/C++ code directly from Zexus for performance-critical operations.

**Why Use NATIVE**:
- Leverage existing C/C++ libraries without rewrites
- Bypass interpreter overhead for compute-heavy tasks
- Access system-level APIs and libraries
- Maintain native code performance while using Zexus for orchestration

## Syntax

```
native "<library_path>", "<function_name>"(arg1, arg2, ...) [as result_var];
```

## Components

- `<library_path>`: Path to shared library (.so on Linux, .dll on Windows, .dylib on macOS)
- `<function_name>`: Name of the C/C++ function to call
- `arg1, arg2, ...`: Arguments passed to the native function
- `result_var`: Optional variable name to store the return value

## Examples

### Basic C Function Call

```zexus
// Call pow function from libmath.so
native "libmath.so", "pow"(2, 3) as result;
print result;  // Output: 8
```

### SHA256 Hashing

```zexus
// Using libcrypto for cryptographic operations
native "libcrypto.so", "SHA256"(data) as hash;
print hash;
```

### Matrix Operations

```zexus
// Using a specialized math library
let matrix_a = [1, 2, 3, 4];
let matrix_b = [5, 6, 7, 8];
native "libmatrix.so", "multiply"(matrix_a, matrix_b) as result;
print result;
```

## Type Conversion

Zexus automatically converts between native types and Zexus objects:

| Zexus Type | C Type |
|-----------|---------|
| Integer | int32_t, int64_t |
| String | const char* |
| List | array pointer |
| Map | struct pointer |
| Boolean | bool |

## Error Handling

If library loading or function calling fails, a descriptive error is returned:

```zexus
native "invalid_lib.so", "func"() as result;
// Error: Failed to load native library 'invalid_lib.so': ...
```

## Performance Characteristics

- **Overhead**: Minimal (just FFI marshalling)
- **Latency**: Same as direct C/C++ call
- **Use for**: Compute-intensive operations, crypto, large data processing

## Library Linking Tips

1. Ensure library is in LD_LIBRARY_PATH (Linux) or system library path
2. Compile C/C++ code as shared library: `gcc -shared -o lib.so file.c`
3. Use extern "C" for C++ functions to avoid name mangling

## Security Considerations

⚠️ **WARNING**: NATIVE statements call external code directly. This bypasses Zexus's security model:
- No sandboxing applied to native code
- No type checking beyond marshalling
- Can access any system resource

Use NATIVE only with trusted libraries. Combine with SANDBOX for additional isolation:

```zexus
sandbox("restricted") {
  native "libcrypto.so", "sign"(data) as signature;
}
```
