# BUFFER Statement

**Purpose**: Direct memory access and manipulation for low-level data structure operations.

**Why Use BUFFER**:
- Work with raw binary data efficiently
- Avoid intermediate object allocation overhead
- Implement custom data structures (tree nodes, graphs, etc.)
- Zero-copy data passing to native code
- Memory-mapped file operations

## Syntax

```
buffer <name> = allocate(<size>);
buffer <name>.read(<offset>, <length>);
buffer <name>.write(<offset>, <data>);
buffer <name>.free();
```

## Operations

### allocate(size)
Allocate a buffer of specified size (in bytes).

```zexus
buffer my_buffer = allocate(1024);
print "Allocated 1KB buffer";
```

### read(offset, length)
Read `length` bytes from buffer starting at `offset`.

```zexus
buffer data = allocate(100);
let bytes = buffer data.read(0, 10);  // Read first 10 bytes
```

### write(offset, data)
Write data to buffer at specified offset.

```zexus
buffer buf = allocate(100);
buffer buf.write(0, [65, 66, 67, 68]);  // Write ABCD
```

### free()
Deallocate the buffer.

```zexus
buffer buf = allocate(1000);
// ... use buffer ...
buffer buf.free();
```

## Examples

### Binary Data Processing

```zexus
// Read binary file into buffer
buffer file_data = allocate(4096);
let header = buffer file_data.read(0, 4);
let version = header[0];
let size = header[1] | (header[2] << 8) | (header[3] << 16);
```

### Custom Data Structure

```zexus
// Implement linked list node in buffer
action allocate_node(value, next_ptr) {
  buffer node = allocate(16);  // 8 bytes value + 8 bytes pointer
  buffer node.write(0, [value]);
  buffer node.write(8, [next_ptr]);
  return node;
}

let node1 = allocate_node(42, 0);
let value = buffer node1.read(0, 1);
print value;  // [42]
```

### Stream Processing

```zexus
buffer chunk = allocate(65536);  // 64KB chunks

for file_block in file_stream {
  buffer chunk.write(0, file_block);
  let processed = process_chunk(buffer chunk.read(0, buffer chunk.size));
  output(processed);
}

buffer chunk.free();
```

### Memory Reuse Pattern

```zexus
// Reuse buffer for multiple operations
buffer work_buffer = allocate(10000);

// First operation
buffer work_buffer.write(0, batch1);
process(buffer work_buffer.read(0, len(batch1)));

// Second operation - reuse same memory
buffer work_buffer.write(0, batch2);
process(buffer work_buffer.read(0, len(batch2)));

buffer work_buffer.free();
```

## Type Conversions

Data is converted between Zexus and binary representations:

```zexus
// Array of bytes
buffer buf.write(0, [1, 2, 3, 4]);

// String conversion
buffer buf.write(0, "hello");

// Integer conversion
buffer buf.write(0, 42);  // Converted to bytes
```

## Performance Characteristics

| Operation | Time Complexity | Memory |
|-----------|-----------------|--------|
| allocate | O(1) | n bytes |
| read | O(length) | n bytes copy |
| write | O(length) | 0 (in-place) |
| free | O(1) | reclaimed |

## Advanced Example: Image Processing

```zexus
action process_image(filename, width, height) {
  let size = width * height * 4;  // RGBA
  buffer img = allocate(size);
  
  // Load image into buffer (simulated)
  load_file(filename, img);
  
  // Process pixels
  for y in range(height) {
    for x in range(width) {
      let pixel_offset = (y * width + x) * 4;
      let pixel = buffer img.read(pixel_offset, 4);
      let processed = apply_filter(pixel);
      buffer img.write(pixel_offset, processed);
    }
  }
  
  // Save result
  save_file("output.raw", img);
  buffer img.free();
}

process_image("input.raw", 1920, 1080);
```

## Integration with NATIVE

```zexus
// Allocate buffer and pass to C function
buffer raw_data = allocate(8192);
buffer raw_data.write(0, input_data);

native "libcompress.so", "deflate"(raw_data, 8192) as compressed;

buffer raw_data.free();
```

## Memory Safety Notes

⚠️ **Important**:
- Buffer bounds are NOT automatically checked
- Writing past allocated size causes undefined behavior
- Use SANDBOX to restrict buffer operations if needed
- Always call `free()` to prevent memory leaks

Example of unsafe code:
```zexus
buffer buf = allocate(10);
buffer buf.write(100, [1, 2, 3]);  // ❌ Out of bounds!
```

## Best Practices

1. **Track Buffer Size**: Keep original allocation size for bounds checking
2. **Always Free**: Pair `allocate()` with `free()`
3. **Document Format**: Comment on buffer layout/structure
4. **Use Actions**: Wrap buffer operations in actions for reusability

```zexus
action safe_buffer_write(buf, offset, size, data) {
  if offset + len(data) > size {
    return error("Write would exceed buffer bounds");
  }
  buffer buf.write(offset, data);
  return true;
}
```

5. **Validate Input**: Check external data before writing to buffer

## See Also

- NATIVE: For efficient data passing to C/C++ functions
- SIMD: For vectorized operations on buffered data
- SANDBOX: For restricting buffer access in untrusted code
