# LOG Keyword - Output Redirection

## Overview

The `LOG` keyword redirects subsequent print output to a file. Output redirection is **scope-aware**, meaning it automatically restores to the previous output destination when the current block exits.

## Syntax

```zexus
log > filepath;
```

## Parameters

- **filepath**: String literal or expression evaluating to a file path
  - Can be a relative path (saved relative to current working directory)
  - Can be an absolute path
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

## See Also

- [PRINT](PRINT.md) - Output to console
- [DEBUG](DEBUG.md) - Debug output
- [ACTION](ACTION_FUNCTION_LAMBDA_RETURN.md) - Function scopes
- [TRY/CATCH](TRY_CATCH.md) - Error handling
