# Zexus Performance Profiler

The Zexus profiler helps you identify performance bottlenecks in your code.

## Features

- **Execution Time Profiling**: Measure function execution time
- **Memory Profiling**: Track memory allocation and peak usage
- **Call Graph Analysis**: Understand function call relationships
- **Hotspot Detection**: Identify performance bottlenecks
- **Detailed Reports**: Comprehensive profiling reports

## Usage

### Command Line

Profile a Zexus file:

```bash
zx profile myfile.zx
```

With custom options:

```bash
zx profile --memory --top 30 myfile.zx
```

### VS Code Extension

1. Open a Zexus file
2. Command Palette → "Zexus: Profile Performance"
3. View results in terminal

### Programmatic API

```zexus
use {start_profiling, stop_profiling, print_profile} from "profiler"

# Start profiling
start_profiling(enable_memory: true)

# Your code here
action process_data(items) {
    for each item in items {
        calculate(item)
    }
}

process_data(get_data())

# Stop and print report
print_profile()
```

## Profile Report

Example output:

```
================================================================================
ZEXUS PERFORMANCE PROFILE REPORT
================================================================================

Total Time: 1.2345 seconds
Total Calls: 1523
Peak Memory: 12.34 MB

Function                                      Calls      Total Time       Self Time        Avg Time
----------------------------------------------------------------------------------------------------
process_data                                     100        0.8234s        0.3456s      0.008234s
calculate_result                                 500        0.3456s        0.3456s      0.000691s
helper_function                                  923        0.0888s        0.0888s      0.000096s

================================================================================
TOP HOTSPOTS (by total time)
================================================================================
1. process_data: 0.8234s (66.7%)
2. calculate_result: 0.3456s (28.0%)
3. helper_function: 0.0888s (7.2%)
================================================================================
```

## Metrics Explained

### Total Time
Total wall-clock time from start to finish.

### Total Calls
Total number of function calls made.

### Per-Function Metrics

- **Calls**: Number of times function was called
- **Total Time**: Total time spent in function including nested calls
- **Self Time**: Time spent in function excluding nested calls
- **Avg Time**: Average time per call (Total Time / Calls)

### Memory Metrics

- **Peak Memory**: Maximum memory used during execution
- **Memory Allocated**: Memory allocated by function
- **Memory Peak**: Peak memory when function was active

### Hotspots

Functions taking the most time, sorted by percentage of total time.

## Best Practices

1. **Profile Production-Like Data**: Use realistic data sizes
2. **Profile Multiple Runs**: Results can vary between runs
3. **Focus on Hotspots**: Optimize the top 20% of time-consuming functions
4. **Consider Trade-offs**: Memory vs speed, code complexity
5. **Profile After Changes**: Verify optimizations work

## Optimization Tips

### Based on Profile Results

**High Total Time, Low Self Time**
- Function calls many slow child functions
- Optimize child functions first

**High Self Time**
- Function itself is slow
- Look for:
  - Unnecessary loops
  - Inefficient algorithms
  - Repeated calculations

**High Call Count**
- Function called too often
- Consider:
  - Caching results
  - Batch processing
  - Reducing recursion

**High Memory Usage**
- Memory leaks or excessive allocation
- Check for:
  - Unclosed resources
  - Large temporary objects
  - Unnecessary copies

## Advanced Usage

### Selective Profiling

Profile specific sections:

```zexus
use {Profiler} from "profiler"

let profiler = Profiler()

# Profile only critical section
profiler.start()
critical_operation()
let report = profiler.stop()

profiler.print_report(report)
```

### Custom Profiling

```zexus
use {profile_function} from "profiler"

# Decorator-style profiling
action @profile_function calculate(x, y) {
    return x * y + complex_math(x, y)
}
```

### Export Profile Data

```zexus
use {stop_profiling} from "profiler"

let report = stop_profiling()
let data = report.to_dict()

# Save to file
file_write_json("profile.json", data)
```

## Troubleshooting

### Profiling Overhead

Profiling adds some overhead. For accurate results:
- Run multiple times
- Compare with unprofiled runs
- Focus on relative times, not absolute

### Memory Profiling Issues

If memory profiling doesn't work:
1. Check Python version (3.4+ required for tracemalloc)
2. Disable memory profiling: `start_profiling(enable_memory: false)`

## Integration with CI/CD

Track performance over time:

```bash
# In CI pipeline
zx profile --json myfile.zx > profile.json

# Compare with baseline
python compare_profiles.py profile.json baseline.json
```

## API Reference

See [Profiler API](../api/PROFILER_API.md) for detailed API documentation.
