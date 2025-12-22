# Zexus VM Profiler - Usage Guide

**Author:** Zexus Development Team  
**Date:** December 22, 2025  
**Status:** Phase 8.1 Complete

---

## Overview

The Zexus VM Profiler provides detailed instruction-level profiling for the Zexus Virtual Machine, enabling developers to identify performance bottlenecks, hot loops, and optimization opportunities.

---

## Quick Start

### Basic Usage

```python
from src.zexus.vm.vm import VM
from src.zexus.vm.bytecode import BytecodeBuilder

# Create VM with profiling enabled
vm = VM(
    enable_profiling=True,
    profiling_level="DETAILED",
    use_jit=False  # Disable JIT for clearer profiling
)

# Create and execute bytecode
builder = BytecodeBuilder()
builder.emit_load_const(10)
builder.emit_load_const(20)
builder.emit_add()
builder.emit_return()

bytecode = builder.build()
result = vm.execute(bytecode)

# Get profiling report
report = vm.get_profiling_report(format='text', top_n=20)
print(report)
```

---

## Profiling Levels

### `NONE` - No Profiling
- Profiling completely disabled
- Zero overhead
- Use for production

### `BASIC` - Count Only
- Instruction execution counts
- Opcode frequency distribution
- ~100% overhead (interpreted Python)
- Minimal memory usage

### `DETAILED` - Count + Timing (Recommended)
- Everything in BASIC
- Execution timing per instruction
- Statistical analysis (min/max/avg/p50/p95/p99)
- ~100-200% overhead (interpreted Python)
- Moderate memory usage

### `FULL` - Complete Profiling
- Everything in DETAILED
- Memory operation tracking
- Branch prediction analysis
- Hot loop detection
- ~100-200% overhead (interpreted Python)
- Higher memory usage

---

## API Reference

### Creating a Profiled VM

```python
vm = VM(
    enable_profiling=True,      # Enable profiling
    profiling_level="DETAILED",  # NONE|BASIC|DETAILED|FULL
    use_jit=False               # Recommended: disable JIT for clearer profiling
)
```

### Profiler Control

```python
# Start profiling
vm.start_profiling()

# Execute code
vm.execute(bytecode)

# Stop profiling
vm.stop_profiling()

# Reset profiler statistics
vm.reset_profiler()
```

### Getting Results

```python
# Get summary statistics
summary = vm.get_profiling_summary()
print(f"Total instructions: {summary['total_instructions']}")
print(f"Instructions/sec: {summary['instructions_per_sec']:.0f}")
print(f"Profiling overhead: {summary['overhead_percentage']:.1f}%")

# Get detailed report (text format)
report_text = vm.get_profiling_report(format='text', top_n=20)
print(report_text)

# Get report in JSON format
import json
report_json = vm.get_profiling_report(format='json', top_n=20)
data = json.loads(report_json)

# Get report in HTML format
report_html = vm.get_profiling_report(format='html', top_n=20)
with open('profile_report.html', 'w') as f:
    f.write(report_html)
```

---

## Understanding Report Output

### Text Report Example

```
================================================================================
ZEXUS VM INSTRUCTION PROFILING REPORT
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Profiling Level:       DETAILED
Total Instructions:    1,234
Unique Instructions:   45
Total Time:            0.0125 seconds
Instructions/Second:   98,720
Profiling Overhead:    2.5% (0.31ms)
Hot Loops Detected:    2

MOST COMMON OPCODES
--------------------------------------------------------------------------------
  LOAD_CONST            450 (36.5%)
  ADD                   200 (16.2%)
  LOAD_NAME             150 (12.2%)
  STORE_NAME            100 ( 8.1%)
  ...

TOP 20 HOTTEST INSTRUCTIONS (by count)
--------------------------------------------------------------------------------
    IP Opcode                    Count     Avg (μs)   Total (ms)
--------------------------------------------------------------------------------
    10 LOAD_CONST                  450        12.50         5.63
    15 ADD                          200        15.20         3.04
    ...

HOT LOOPS (>1000 iterations)
--------------------------------------------------------------------------------
 Start    End   Iterations   Total (ms)   Avg/iter (μs)
--------------------------------------------------------------------------------
    20     35         2500        125.00           50.00
    ...
```

### JSON Report Structure

```json
{
  "summary": {
    "profiling_level": "DETAILED",
    "total_instructions": 1234,
    "unique_instructions": 45,
    "total_time_sec": 0.0125,
    "instructions_per_sec": 98720,
    "profiling_overhead_ms": 0.31,
    "overhead_percentage": 2.5,
    "hot_loops_detected": 2,
    "most_common_opcodes": {
      "LOAD_CONST": 450,
      "ADD": 200,
      ...
    }
  },
  "hottest_instructions": [
    {
      "opcode": "LOAD_CONST",
      "ip": 10,
      "count": 450,
      "avg_time_us": 12.50,
      "total_time_ms": 5.63,
      "p50_us": 11.20,
      "p95_us": 18.50,
      "p99_us": 22.30
    },
    ...
  ],
  "hot_loops": [
    {
      "start_ip": 20,
      "end_ip": 35,
      "iterations": 2500,
      "total_time_ms": 125.00,
      "avg_iteration_us": 50.00
    }
  ]
}
```

---

## Use Cases

### 1. Finding Performance Bottlenecks

```python
vm = VM(enable_profiling=True, profiling_level="DETAILED")

# Run your code
vm.execute(bytecode)

# Get slowest instructions
summary = vm.get_profiling_summary()
report = vm.get_profiling_report(format='text', top_n=10)

# Focus optimization on slowest instructions
print(report)
```

### 2. Detecting Hot Loops

```python
vm = VM(enable_profiling=True, profiling_level="FULL")

# Run code with loops
vm.execute(loop_bytecode)

# Get hot loop report
import json
report_data = json.loads(vm.get_profiling_report(format='json'))

for loop in report_data['hot_loops']:
    print(f"Hot loop: IP {loop['start_ip']}-{loop['end_ip']}")
    print(f"  Iterations: {loop['iterations']}")
    print(f"  Avg time/iter: {loop['avg_iteration_us']:.2f}μs")
```

### 3. Comparing Optimizations

```python
# Before optimization
vm1 = VM(enable_profiling=True, profiling_level="DETAILED")
vm1.execute(bytecode_v1)
stats1 = vm1.get_profiling_summary()

# After optimization
vm2 = VM(enable_profiling=True, profiling_level="DETAILED")
vm2.execute(bytecode_v2)
stats2 = vm2.get_profiling_summary()

# Compare
speedup = stats1['total_time_sec'] / stats2['total_time_sec']
print(f"Speedup: {speedup:.2f}x")
print(f"Instruction reduction: {stats1['total_instructions'] - stats2['total_instructions']}")
```

### 4. Profiling Zexus Programs

```python
from src.zexus.evaluator.core import Evaluator

code = """
let sum = 0
let i = 0
while (i < 1000) {
    sum = sum + i
    i = i + 1
}
sum
"""

evaluator = Evaluator()

# Enable VM with profiling
evaluator.use_vm = True
evaluator.vm_instance = VM(
    enable_profiling=True,
    profiling_level="FULL"
)

result = evaluator.eval(code)

# Get profiling report
report = evaluator.vm_instance.get_profiling_report(format='text')
print(report)
```

---

## Performance Considerations

### Overhead by Level

| Level | Overhead (Interpreted) | Overhead (JIT) | Use Case |
|-------|------------------------|----------------|----------|
| NONE | 0% | 0% | Production |
| BASIC | ~100% | <1% | Quick profiling |
| DETAILED | ~100-200% | <5% | Performance analysis |
| FULL | ~100-200% | <10% | Detailed analysis |

**Note:** Overhead percentages are for interpreted Python. In JIT-compiled code, overhead would be significantly lower.

### Best Practices

1. **Disable JIT during profiling** - JIT can mask instruction-level behavior
2. **Use DETAILED for most cases** - Best balance of detail and overhead
3. **Profile representative workloads** - Use real-world data/scenarios
4. **Profile multiple runs** - Average results for statistical significance
5. **Focus on hot paths** - Optimize the top 20% that takes 80% of time

### When to Profile

✅ **Profile:**
- During development to understand performance
- Before optimizing (measure first!)
- To verify optimization improvements
- To identify unexpected bottlenecks

❌ **Don't Profile:**
- In production (unless necessary)
- With JIT enabled (masks instruction behavior)
- On trivial/fast code (overhead dominates)

---

## Interpreting Results

### Hottest Instructions
Instructions executed most frequently. Optimizing these has the biggest impact.

### Slowest Instructions
Instructions taking the most total time. May indicate:
- Complex operations (division, modulo)
- External calls (I/O, system calls)
- GC triggers
- Cache misses

### Hot Loops
Loops executed >1000 times. Prime candidates for:
- Loop unrolling
- Vectorization (SIMD)
- JIT compilation
- Algorithm optimization

### Branch Prediction
- High `branch_taken` rate (>90%) - predictable, good for optimization
- Low rate (<50%) - unpredictable, harder to optimize

---

## Example: Real-World Profiling Session

```python
from src.zexus.vm.vm import VM
from src.zexus.vm.bytecode import BytecodeBuilder

# Create profiled VM
vm = VM(
    enable_profiling=True,
    profiling_level="DETAILED",
    use_jit=False
)

# Build fibonacci bytecode
builder = BytecodeBuilder()

# n = 20
builder.emit_load_const(20)
builder.emit_store_name("n")

# a = 0, b = 1
builder.emit_load_const(0)
builder.emit_store_name("a")
builder.emit_load_const(1)
builder.emit_store_name("b")

# i = 0
builder.emit_load_const(0)
builder.emit_store_name("i")

# while i < n:
builder.emit_label("loop")
builder.emit_load_name("i")
builder.emit_load_name("n")
builder.emit_lt()
builder.emit_jump_if_false("end")

#   temp = a + b
builder.emit_load_name("a")
builder.emit_load_name("b")
builder.emit_add()
builder.emit_store_name("temp")

#   a = b
builder.emit_load_name("b")
builder.emit_store_name("a")

#   b = temp
builder.emit_load_name("temp")
builder.emit_store_name("b")

#   i = i + 1
builder.emit_load_name("i")
builder.emit_load_const(1)
builder.emit_add()
builder.emit_store_name("i")

builder.emit_jump("loop")

# return a
builder.emit_label("end")
builder.emit_load_name("a")
builder.emit_return()

bytecode = builder.build()

# Execute
result = vm.execute(bytecode)
print(f"Fibonacci(20) = {result}")

# Get detailed profiling report
print("\n" + vm.get_profiling_report(format='text', top_n=15))

# Get summary
summary = vm.get_profiling_summary()
print(f"\n📊 Performance Summary:")
print(f"   Total Instructions: {summary['total_instructions']:,}")
print(f"   Execution Time: {summary['total_time_sec']*1000:.2f}ms")
print(f"   Instructions/sec: {summary['instructions_per_sec']:,.0f}")
```

---

## Advanced Features

### Direct Profiler Access

```python
from src.zexus.vm.profiler import InstructionProfiler, ProfilingLevel

# Create profiler directly
profiler = InstructionProfiler(level=ProfilingLevel.FULL)

# Use in custom scenarios
profiler.start()

# ... your code ...
profiler.record_instruction(0, "LOAD_CONST", 42)
profiler.measure_instruction(0, 0.001)  # 1ms

profiler.stop()

# Get results
summary = profiler.get_summary()
report = profiler.generate_report(format='json')
```

### Hot Loop Analysis

```python
profiler = vm.profiler
hot_loops = profiler.get_hot_loops(min_iterations=1000)

for loop in hot_loops:
    print(f"\nHot Loop: IP {loop.start_ip} -> {loop.end_ip}")
    print(f"  Iterations: {loop.iterations:,}")
    print(f"  Total Time: {loop.total_time*1000:.2f}ms")
    print(f"  Avg/Iteration: {(loop.total_time/loop.iterations)*1_000_000:.2f}μs")
    print(f"  Instructions: {', '.join(loop.instructions[:5])}")
```

---

## Troubleshooting

### High Overhead
- **Problem:** Profiling slows code >5x
- **Solution:** Use BASIC level or disable profiling for fast code

### Missing Data
- **Problem:** No profiling data collected
- **Solution:** Ensure `enable_profiling=True` and `start_profiling()` was called

### Inaccurate Timing
- **Problem:** Timing seems wrong
- **Solution:** Disable JIT, run multiple times, use larger workloads

---

## Future Enhancements

- [ ] Flamegraph visualization
- [ ] Call graph analysis
- [ ] Memory allocation profiling
- [ ] Cache miss simulation
- [ ] Integration with external profilers (cProfile, Py-Spy)

---

## References

- [VM_OPTIMIZATION_PHASE_8_MASTER_LIST.md](VM_OPTIMIZATION_PHASE_8_MASTER_LIST.md) - Implementation details
- [VM_INTEGRATION_SUMMARY.md](VM_INTEGRATION_SUMMARY.md) - VM architecture
- [src/zexus/vm/profiler.py](../../../src/zexus/vm/profiler.py) - Profiler source code
- [tests/vm/test_profiler.py](../../../tests/vm/test_profiler.py) - Profiler tests

---

**Last Updated:** December 22, 2025  
**Version:** 1.0.0  
**Status:** Production Ready ✅
