# LOG Keyword - Output Redirection & Code Generation

## Overview

The `LOG` keyword redirects subsequent print output to a file. Output redirection is **scope-aware**, meaning it automatically restores to the previous output destination when the current block exits.

**New in v2:** Support for `>>` append operator and any file extension, enabling cross-block code generation and data sharing.

## Syntax

```zexus
log > filepath;   // Write mode (scope-aware append)
log >> filepath;  // Explicit append mode (recommended for cross-block use)
```

## Parameters

- **filepath**: String literal or expression evaluating to a file path
  - Can be a relative path (saved relative to current working directory)
  - Can be an absolute path
  - **Supports any file extension**: .txt, .py, .zx, .cpp, .rs, .js, .json, etc.
  - File is opened in append mode (won't overwrite existing content)

## Basic Usage

### Simple File Logging

```zexus
action calculate {
    a = 10;
    b = 20;
    
    log > "calculation.txt";
    print("Result: " + (a + b));
    print("Addition complete");
}

calculate();
// Output is written to calculation.txt
```

### Absolute Path

```zexus
log > "/var/log/app.log";
print("Application started");
```

### Variable Path

```zexus
let logfile = "output.txt";
log > logfile;
print("Logging to variable path");
```

## Scope-Aware Behavior

LOG redirection is automatically restored when the block exits:

```zexus
print("Before action - console");

action calculate {
    a = 10;
    b = 20;
    
    log > "test_output.txt";
    print("Inside action: " + (a + b));
    // Output goes to test_output.txt
}

calculate();

print("After action - console");
// Output restored to console automatically
```

### Nested Scopes

```zexus
action outer {
    print("Outer - console");
    
    log > "outer.log";
    print("Outer - logged");
    
    action inner {
        log > "inner.log";
        print("Inner - logged");
    }
    
    inner();
    print("Back to outer - logged");
}

outer();
print("Back to console");
```

**Output:**
- Console: "Outer - console", "Back to console"
- outer.log: "Outer - logged", "Back to outer - logged"
- inner.log: "Inner - logged"

## Advanced Usage

### Cross-Block Code Generation

Generate code in one block, use in another:

```zexus
// Block A: Generate Python code
action generatePython {
    log >> "script.py";
    print("def calculate(x, y):");
    print("    return x + y");
    print("");
    print("result = calculate(10, 20)");
}

generatePython();

// Block B: Read and execute
let code = read_file("script.py");
eval_file("script.py", "python");
```

### Hidden Code Layers

Generate code in any language for later execution:

```zexus
// Generate C++ code
action generateCppModule {
    log >> "math.cpp";
    print("#include <iostream>");
    print("int multiply(int a, int b) {");
    print("    return a * b;");
    print("}");
}

// Generate Rust code
action generateRustModule {
    log >> "utils.rs";
    print("pub fn add(a: i32, b: i32) -> i32 {");
    print("    a + b");
    print("}");
}

// Generate Zexus code
action generateZexusModule {
    log >> "helpers.zx";
    print("action divide(a, b) {");
    print("    return a / b;");
    print("}");
}

generateCppModule();
generateRustModule();
generateZexusModule();

// Later: Execute the generated Zexus code
eval_file("helpers.zx");
```

### Multi-Block Data Sharing

Multiple blocks appending to the same file:

```zexus
action collectData1 {
    log >> "data.txt";
    print("Data from block 1: [1, 2, 3]");
}

action collectData2 {
    log >> "data.txt";
    print("Data from block 2: [4, 5, 6]");
}

action collectData3 {
    log >> "data.txt";
    print("Data from block 3: [7, 8, 9]");
}

collectData1();
collectData2();
collectData3();

// Read all collected data
let all_data = read_file("data.txt");
print(all_data);
```

### JSON Data Generation

```zexus
action generateConfig {
    log >> "config.json";
    print("{");
    print("  \"server\": {");
    print("    \"host\": \"localhost\",");
    print("    \"port\": 8080");
    print("  },");
    print("  \"debug\": true");
    print("}");
}

generateConfig();

// Read and parse JSON
let config = read_file("config.json");
print("Config generated:");
print(config);
```

### Conditional Logging

```zexus
action processData(debug) {
    if (debug) {
        log > "debug.log";
    }
    
    print("Processing data...");
    // Logs to file if debug=true, console if debug=false
}

processData(true);   // Logs to debug.log
processData(false);  // Logs to console
```

### Multiple Log Files

```zexus
action generateReports {
    log > "summary.txt";
    print("Summary Report");
    print("=============");
    
    log > "details.txt";
    print("Detailed Report");
    print("===============");
}

generateReports();
// summary.txt gets first two prints
// details.txt gets last two prints
```

### Error Logging

```zexus
action processWithErrorLog {
    try {
        log > "error.log";
        // Processing code
        if (error_condition) {
            print("ERROR: Something went wrong");
        }
    } catch (e) {
        log > "error.log";
        print("EXCEPTION: " + e);
    }
}
```

## File Behavior

### Append Mode

LOG always opens files in **append mode**. Multiple runs will add to the file:

```zexus
// First run
log > "output.txt";
print("Line 1");

// Second run
log > "output.txt";
print("Line 2");

// output.txt contains:
// Line 1
// Line 2
```

### File Creation

If the file doesn't exist, LOG creates it automatically:

```zexus
log > "new_file.txt";
print("This creates the file");
```

### Path Normalization

Relative paths are resolved relative to the current working directory:

```zexus
// If CWD is /home/user/project
log > "output.txt";
// Creates /home/user/project/output.txt

log > "logs/output.txt";
// Creates /home/user/project/logs/output.txt (if logs/ exists)
```

## Error Handling

### Invalid File Path

```zexus
log > "/invalid/path/file.txt";
// Error: Cannot open log file '/invalid/path/file.txt': [Errno 2] No such file or directory
```

### Permission Denied

```zexus
log > "/root/protected.txt";
// Error: Cannot open log file '/root/protected.txt': [Errno 13] Permission denied
```

## Best Practices

### 1. Use Explicit Paths

```zexus
// Good: Clear, explicit path
log > "logs/application.log";

// Avoid: Ambiguous relative path
log > "../../output.txt";
```

### 2. Close Logs with Scope

Let the scope handle cleanup automatically:

```zexus
// Good: Automatic cleanup
action processData {
    log > "data.log";
    print("Processing...");
    // Log file closed automatically when action exits
}

// No need to manually close
```

### 3. Separate Concerns

Use different log files for different purposes:

```zexus
action runApp {
    log > "app.log";
    print("Application started");
    
    if (debug_mode) {
        log > "debug.log";
        print("Debug info");
    }
    
    log > "audit.log";
    print("User action recorded");
}
```

### 4. Combine with Error Handling

```zexus
action safeProcess {
    try {
        log > "process.log";
        print("Starting process");
        // Processing code
        print("Process complete");
    } catch (error) {
        log > "error.log";
        print("ERROR: " + error);
    }
}
```

## Implementation Details

- **Token**: `LOG`
- **AST Node**: `LogStatement(filepath)`
- **Evaluation**: Opens file in append mode, redirects `sys.stdout`
- **Cleanup**: Automatic restoration via `_restore_stdout()` in block's `finally` clause
- **Stack-Based**: Uses `env._stdout_stack` to track redirection levels

## Platform Compatibility

### Discard Output (Unix/Linux)

```zexus
log > "/dev/null";
print("This goes nowhere");
```

### Discard Output (Windows)

```zexus
log > "NUL";
print("This goes nowhere");
```

## Comparison with Other Languages

### Python

```python
# Python
import sys
sys.stdout = open('output.txt', 'a')
print("Logged")
sys.stdout = sys.__stdout__  # Manual restore
```

### Zexus

```zexus
// Zexus - automatic restoration
log > "output.txt";
print("Logged");
// Automatically restored
```

## Related Features

- **print**: Output text (affected by LOG)
- **debug**: Debug output (affected by LOG)
- **action**: Defines scopes for LOG restoration
- **try/catch**: Error handling with LOG

## Common Patterns

### Application Logging

```zexus
action logEvent(event_type, message) {
    log > "events.log";
    let timestamp = time();
    print("[" + timestamp + "] " + event_type + ": " + message);
}

logEvent("INFO", "Application started");
logEvent("ERROR", "Connection failed");
```

### Debug Mode

```zexus
let DEBUG = true;

action process(data) {
    if (DEBUG) {
        log > "debug.log";
        print("DEBUG: Processing " + data);
    }
    
    // Normal processing
    let result = transform(data);
    
    if (DEBUG) {
        print("DEBUG: Result = " + result);
    }
    
    return result;
}
```

### Audit Trail

```zexus
action recordAudit(user, action, details) {
    log > "audit.log";
    let timestamp = time();
    print("[" + timestamp + "] User: " + user);
    print("Action: " + action);
    print("Details: " + details);
    print("---");
}

recordAudit("admin", "login", "Successful authentication");
recordAudit("user123", "file_access", "Read file: data.txt");
```

## Built-in Functions for File Operations

### read_file(path)

Read the entire contents of a file as a string.

**Syntax:**
```zexus
let content = read_file("filename.txt");
```

**Parameters:**
- `path` (string): Relative or absolute file path

**Returns:** String containing file contents

**Errors:**
- File not found
- Permission denied
- Read error

**Example:**
```zexus
action generateData {
    log >> "data.txt";
    print("Line 1");
    print("Line 2");
}

generateData();

let data = read_file("data.txt");
print("Read from file:");
print(data);
```

### eval_file(path, [language])

Execute code from a file, optionally specifying the language.

**Syntax:**
```zexus
eval_file("script.zx");              // Auto-detect from extension
eval_file("script.py", "python");    // Explicit language
```

**Parameters:**
- `path` (string): Relative or absolute file path
- `language` (optional string): Language override ("zx", "python", "js", etc.)

**Supported Languages:**
- **zx/zexus**: Execute Zexus code (`.zx` files)
- **py/python**: Execute Python code (`.py` files)
- **js/javascript**: Execute JavaScript via Node.js (`.js` files)
- **cpp/c++/c**: Planned - compilation support
- **rs/rust**: Planned - compilation support

**Returns:** Result of execution (language-dependent)

**Example:**
```zexus
// Generate and execute Zexus code
action generateHelper {
    log >> "helper.zx";
    print("action add(a, b) {");
    print("    return a + b;");
    print("}");
}

generateHelper();
eval_file("helper.zx");

// Now we can use the generated function
let result = add(5, 10);
print("Result: " + result);  // 15
```

**Python Interop Example:**
```zexus
action generatePython {
    log >> "calculate.py";
    print("x = 10");
    print("y = 20");
    print("result = x * y");
    print("print(f'Python: {result}')");
}

generatePython();
eval_file("calculate.py", "python");
// Output: Python: 200
```

**Error Handling:**
```zexus
try {
    eval_file("missing.zx");
} catch (err) {
    print("Error: " + err);
}
```

## Use Cases

### 1. Dynamic Module Generation

```zexus
action generateMathModule {
    log >> "math_extended.zx";
    print("action square(x) { return x * x; }");
    print("action cube(x) { return x * x * x; }");
    print("action pow4(x) { return x * x * x * x; }");
}

generateMathModule();
eval_file("math_extended.zx");

// Use generated functions
print(square(5));   // 25
print(cube(3));     // 27
print(pow4(2));     // 16
```

### 2. Configuration File Generation

```zexus
action generateConfig {
    log >> "app.config.zx";
    print("let config = {");
    print("    api_url: \"https://api.example.com\",");
    print("    timeout: 5000,");
    print("    debug: true");
    print("};");
}

generateConfig();
eval_file("app.config.zx");
print(config.api_url);
```

### 3. Template Engine

```zexus
action generateTemplate(name, age) {
    log >> "user_" + name + ".html";
    print("<!DOCTYPE html>");
    print("<html>");
    print("<body>");
    print("  <h1>Welcome, " + name + "</h1>");
    print("  <p>Age: " + age + "</p>");
    print("</body>");
    print("</html>");
}

generateTemplate("Alice", 30);
generateTemplate("Bob", 25);

let alice_html = read_file("user_Alice.html");
print(alice_html);
```

### 4. Build System Integration

```zexus
action generateMakefile {
    log >> "Makefile";
    print("CC = gcc");
    print("CFLAGS = -Wall -O2");
    print("");
    print("all: program");
    print("");
    print("program: main.o utils.o");
    print("\t$(CC) $(CFLAGS) -o program main.o utils.o");
}

generateMakefile();
```

### 5. Test Data Generation

```zexus
action generateTestData {
    log >> "test_data.json";
    print("[");
    
    let i = 0;
    while (i < 100) {
        log >> "test_data.json";
        print("  {\"id\": " + i + ", \"value\": " + (i * 10) + "},");
        i = i + 1;
    }
    
    log >> "test_data.json";
    print("  {\"id\": 100, \"value\": 1000}");
    print("]");
}

generateTestData();
```

## See Also

- **read_file()**: Read file contents
- **eval_file()**: Execute code from files
- [PRINT](PRINT.md) - Output to console
- [DEBUG](DEBUG.md) - Debug output
- [ACTION](ACTION_FUNCTION_LAMBDA_RETURN.md) - Function scopes
- [TRY/CATCH](TRY_CATCH.md) - Error handling
