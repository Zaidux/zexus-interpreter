# LOG Keyword - Output Redirection & Code Generation

## Overview

The `LOG` keyword and `<<` operator provide powerful file-based code generation, execution, and import capabilities:

1. **Output Redirection** (`log >`, `log >>`): Redirect print output to files
2. **Code Execution** (`log <<`): Import and execute Zexus code inline (Hidden Layers)
3. **File Reading** (`let var <<`): Read file contents into variables

Output redirection is **scope-aware**, meaning it automatically restores to the previous output destination when the current block exits.

**New in v2:** Support for `>>` append operator and any file extension, enabling cross-block code generation and data sharing.

**New in v3:** Support for `<<` operator:
- `log << file` - Execute Zexus code from file (Hidden Layers)
- `let var << file` - Read file contents as string (any extension)

## Syntax

```zexus
log > filepath;      // Write mode (scope-aware append)
log >> filepath;     // Explicit append mode (recommended for cross-block use)
log << filepath;     // Import/execute Zexus code from file (Hidden Layer)
let var << filepath; // Read file contents into variable (any file type)
```

## Parameters

- **filepath**: String literal or expression evaluating to a file path
  - Can be a relative path (saved relative to current working directory)
  - Can be an absolute path
  - **Supports any file extension**: .txt, .py, .zx, .cpp, .rs, .js, .json, etc.
  - File is opened in append mode (won't overwrite existing content) for `>` and `>>`
  - File is read and executed for `log <<` (supports .zx files only)
  - File is read as string for `let <<` (supports all file types)

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

### Hidden Code Layers (`<<` Import Operator)

**New in v3:** The `<<` operator allows you to import and execute generated code directly into the current scope without explicitly calling `eval_file()`. This creates a "hidden layer" where code generation and execution are seamlessly integrated.

#### Inline Code Import

```zexus
// Step 1: Generate helper functions
action generateHelpers {
    log >> "helpers.zx";
    print("action add(a, b) {");
    print("    return a + b;");
    print("}");
    print("");
    print("action multiply(a, b) {");
    print("    return a * b;");
    print("}");
}

generateHelpers();

// Step 2: Import and use immediately (Hidden Layer)
log << "helpers.zx";

// Step 3: Use the imported functions directly
let sum = add(10, 20);        // 30
let product = multiply(5, 6);  // 30
print("Sum: " + sum);
print("Product: " + product);
```

#### Comparison: `<<` vs `eval_file()`

```zexus
// Traditional approach with eval_file()
action generateCode {
    log >> "module.zx";
    print("action square(x) { return x * x; }");
}
generateCode();
eval_file("module.zx");  // Explicit call
let result = square(5);

// New approach with << (Hidden Layer)
action generateCode {
    log >> "module.zx";
    print("action square(x) { return x * x; }");
}
generateCode();
log << "module.zx";  // Inline import
let result = square(5);
```

**Benefits of `<<`:**
- **Cleaner syntax**: No need for explicit `eval_file()` calls
- **Intent clarity**: Makes it obvious you're importing generated code
- **Scope integration**: Imported code merges directly into current scope
- **Hidden layers**: Code generation + execution in one seamless flow

#### Dynamic Module System

Build a dynamic module system with code generation:

```zexus
// Generate multiple modules
action generateMathModule {
    log >> "math_ops.zx";
    print("action square(x) { return x * x; }");
    print("action cube(x) { return x * x * x; }");
}

action generateStringModule {
    log >> "string_ops.zx";
    print("action repeat(s, n) {");
    print("    let result = \"\";");
    print("    let i = 0;");
    print("    while (i < n) {");
    print("        result = result + s;");
    print("        i = i + 1;");
    print("    }");
    print("    return result;");
    print("}");
}

// Generate the modules
generateMathModule();
generateStringModule();

// Import them all at once (Hidden Layers)
log << "math_ops.zx";
log << "string_ops.zx";

// Use imported functions
print(square(7));           // 49
print(cube(3));             // 27
print(repeat("Hi", 3));     // HiHiHi
```

#### Conditional Code Generation + Import

```zexus
let feature_enabled = true;

if (feature_enabled) {
    // Generate advanced features
    action generateAdvanced {
        log >> "advanced.zx";
        print("action fibonacci(n) {");
        print("    if (n <= 1) { return n; }");
        print("    return fibonacci(n - 1) + fibonacci(n - 2);");
        print("}");
    }
    
    generateAdvanced();
    log << "advanced.zx";  // Import if feature enabled
    
    print("Fib(10) = " + fibonacci(10));  // 55
}
```

#### Build-Time Code Generation

Generate optimized code at build time, import at runtime:

```zexus
// Build script: Generate lookup tables
action buildLookupTable {
    log >> "lookup.zx";
    print("let LOOKUP = {");
    
    let i = 0;
    while (i < 100) {
        log >> "lookup.zx";
        print("    " + i + ": " + (i * i) + ",");
        i = i + 1;
    }
    
    log >> "lookup.zx";
    print("};");
}

buildLookupTable();

// Runtime: Import the pre-computed table
log << "lookup.zx";

// Use the lookup table
print("Square of 25: " + LOOKUP[25]);  // 625
```

### Hidden Code Layers (Multi-Language)

Generate code in multiple languages (note: currently only .zx files are supported for `<<`):

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

// Import Zexus code using << (Hidden Layer)
log << "helpers.zx";
let result = divide(100, 5);  // 20

// For other languages, still use eval_file()
eval_file("math.cpp", "cpp");    // When C++ support is added
eval_file("utils.rs", "rust");   // When Rust support is added
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

## File Import with << Operator

### `let variable << "filename.ext"` - Read File into Variable

The `<<` operator can also be used with `let` statements to read file contents into a variable as a string. This is distinct from `log <<` which executes code.

**Syntax:**
```zexus
let variable_name << "filepath";
```

**Behavior:**
- Reads the entire file content as a string
- **Supports any file extension** (.txt, .json, .zx, .py, .cpp, .md, etc.)
- Returns raw file content without execution or processing
- Files are read with UTF-8 encoding by default
- Falls back to binary mode for non-text files

**Use Cases:**
- Read generated code as text for inspection or manipulation
- Load configuration files
- Import templates or data files
- Read documentation or text content
- Work with multi-file code generation pipelines

### Basic File Import

```zexus
// Read a text file
let content << "readme.txt";
print("File contains: " + content);

// Read generated Zexus code (as string, not executed)
action generateCode {
    log >> "utils.zx";
    print("action add(a, b) { return a + b; }");
}
generateCode();

let code_string << "utils.zx";
print("Generated code:");
print(code_string);
```

### JSON Data Import

```zexus
// Generate JSON data
action generateConfig {
    log >> "config.json";
    print("{");
    print("  \"host\": \"localhost\",");
    print("  \"port\": 8080,");
    print("  \"debug\": true");
    print("}");
}
generateConfig();

// Import JSON as string
let json_text << "config.json";
print("JSON config:");
print(json_text);

// Can then parse with read_json() builtin
let config = read_json("config.json");
```

### Template Loading

```zexus
// Generate HTML template
action generateTemplate {
    log >> "page.html";
    print("<!DOCTYPE html>");
    print("<html>");
    print("<head><title>{{title}}</title></head>");
    print("<body>");
    print("  <h1>{{heading}}</h1>");
    print("  <p>{{content}}</p>");
    print("</body>");
    print("</html>");
}
generateTemplate();

// Load template as string for processing
let template << "page.html";
// Now can do string replacement/processing
print("Template loaded, length: " + len(template));
```

### Multi-Language Code Reading

```zexus
// Generate Python code
action generatePython {
    log >> "script.py";
    print("def calculate(x, y):");
    print("    return x + y");
}

// Generate C++ code
action generateCpp {
    log >> "math.cpp";
    print("#include <iostream>");
    print("int add(int a, int b) {");
    print("    return a + b;");
    print("}");
}

generatePython();
generateCpp();

// Read both as strings
let python_code << "script.py";
let cpp_code << "math.cpp";

print("Python code:");
print(python_code);
print("\nC++ code:");
print(cpp_code);
```

### Comparison: `log <<` vs `let <<`

```zexus
// Setup: Generate a Zexus code file
action setup {
    log >> "demo.zx";
    print("let value = 42;");
    print("action double(x) { return x * 2; }");
}
setup();

// Method 1: log << (Execute code)
log << "demo.zx";
print(value);         // 42
print(double(10));    // 20

// Method 2: let << (Read as string)
let code << "demo.zx";
print("Code string:");
print(code);          // Prints: let value = 42;\naction double(x) { return x * 2; }
// Note: code is NOT executed, just read as text
```

**Key Differences:**

| Feature | `log << file` | `let var << file` |
|---------|---------------|-------------------|
| **Purpose** | Execute code | Read content |
| **Returns** | NULL (side effects) | String object |
| **File Types** | .zx only | Any extension |
| **Scope Impact** | Defines functions/variables | Assigns to one variable |
| **Use Case** | Dynamic module loading | Template/config loading |

### Code Inspection Pipeline

```zexus
// Generate code
action generateModule {
    log >> "module.zx";
    print("action process(data) {");
    print("    return data * 2;");
    print("}");
}

generateModule();

// Read code for inspection
let source << "module.zx";
print("Generated source:");
print(source);
print("\nSource length: " + len(source));

// Decide whether to execute based on inspection
if (len(source) > 0) {
    print("\nExecuting module...");
    log << "module.zx";
    print("Result: " + process(5));  // 10
}
```

### Documentation Generation

```zexus
// Generate documentation
action generateDocs {
    log >> "API.md";
    print("# API Documentation");
    print("");
    print("## add(a, b)");
    print("Adds two numbers.");
    print("");
    print("## multiply(a, b)");
    print("Multiplies two numbers.");
}

generateDocs();

// Read documentation
let docs << "API.md";
print("Generated documentation:");
print(docs);
```

### File Extension Support

The `let << file` operator supports **any file extension**:

```zexus
let text_file << "notes.txt";       // Plain text
let markdown << "README.md";         // Markdown
let python << "script.py";           // Python code
let json << "data.json";             // JSON data
let cpp << "program.cpp";            // C++ code
let rust << "lib.rs";                // Rust code
let javascript << "app.js";          // JavaScript
let yaml << "config.yaml";           // YAML
let xml << "data.xml";               // XML
let csv << "data.csv";               // CSV
```

### Error Handling

```zexus
// File not found
try {
    let content << "nonexistent.txt";
    print(content);
} catch (err) {
    print("Error: " + err);  // "Cannot import file '...': File not found"
}

// Permission denied (if applicable)
try {
    let protected << "/root/secret.txt";
    print(protected);
} catch (err) {
    print("Error: " + err);
}
```

### Best Practices

1. **Use `let <<` for data, `log <<` for code execution**
   ```zexus
   let config << "config.json";     // Read config as string
   log << "helpers.zx";              // Execute helper functions
   ```

2. **Validate file content before use**
   ```zexus
   let code << "generated.zx";
   if (len(code) > 0) {
       // Process or execute
   }
   ```

3. **Combine with string operations**
   ```zexus
   let template << "email.txt";
   // Replace placeholders
   let email = replace(template, "{{name}}", user_name);
   ```

4. **Use for build pipelines**
   ```zexus
   // Generate
   action build {
       log >> "output.js";
       print("// Generated code");
   }
   build();
   
   // Verify
   let output << "output.js";
   if (contains(output, "Generated code")) {
       print("Build successful!");
   }
   ```

## See Also

- **read_file()**: Read file contents
- **eval_file()**: Execute code from files
- [PRINT](PRINT.md) - Output to console
- [DEBUG](DEBUG.md) - Debug output
- [ACTION](ACTION_FUNCTION_LAMBDA_RETURN.md) - Function scopes
- [TRY/CATCH](TRY_CATCH.md) - Error handling
