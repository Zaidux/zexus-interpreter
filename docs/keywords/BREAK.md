# BREAK Keyword Documentation

## Overview

The `BREAK` keyword provides **loop control flow** in Zexus, allowing you to exit a loop prematurely when a specific condition is met. This is essential for:

- **Early Loop Termination**: Exit when you've found what you're looking for
- **Conditional Processing**: Stop processing when a threshold is reached
- **Performance Optimization**: Avoid unnecessary iterations
- **Search Operations**: Exit immediately when target is found
- **Error Conditions**: Stop loop execution on critical errors

## Syntax

```zexus
break;
```

The `BREAK` statement is simple and takes no arguments. It can only be used inside loops (`while`, `for each`).

## Behavior

### In While Loops

```zexus
let i = 0
while true {
    i = i + 1
    if i == 5 {
        break
    }
    print("i = " + string(i))
}
print("Exited at i = " + string(i))
```

**Output:**
```
i = 1
i = 2
i = 3
i = 4
Exited at i = 5
```

### In For Each Loops

```zexus
let items = [10, 20, 30, 40, 50]
for each item in items {
    if item > 30 {
        print("Found item > 30: " + string(item))
        break
    }
    print("Processing: " + string(item))
}
```

**Output:**
```
Processing: 10
Processing: 20
Processing: 30
Found item > 30: 40
```

## Key Features

### 1. Immediate Exit
When `BREAK` is executed:
- The loop **terminates immediately**
- No further iterations are performed
- Execution continues **after the loop**
- Loop variables retain their final values

### 2. Inner Loop Only
In nested loops, `BREAK` only exits the **innermost loop**:

```zexus
let outer = 0
while outer < 3 {
    outer = outer + 1
    print("Outer: " + string(outer))
    
    let inner = 0
    while inner < 5 {
        inner = inner + 1
        if inner > 2 {
            break  # Only exits inner loop
        }
        print("  Inner: " + string(inner))
    }
}
```

**Output:**
```
Outer: 1
  Inner: 1
  Inner: 2
Outer: 2
  Inner: 1
  Inner: 2
Outer: 3
  Inner: 1
  Inner: 2
```

### 3. Scope Preservation
After `BREAK`:
- Variables remain in scope
- Values are preserved
- Execution continues normally

## Usage Examples

### Example 1: Search Operation

```zexus
print("Searching for target value...")

let numbers = [5, 12, 8, 23, 15, 42, 7, 19]
let target = 42
let found = false
let index = 0

for each num in numbers {
    if num == target {
        found = true
        print("Found " + string(target) + " at index " + string(index))
        break
    }
    index = index + 1
}

if !found {
    print("Target not found")
}
```

**Output:**
```
Searching for target value...
Found 42 at index 5
```

### Example 2: Validation with Limits

```zexus
print("Validating user inputs (max 100 items)...")

let inputs = getUserInputs()  # Large array
let validCount = 0
let errorCount = 0

for each input in inputs {
    if validCount >= 100 {
        print("Reached validation limit")
        break
    }
    
    if isValid(input) {
        validCount = validCount + 1
    } else {
        errorCount = errorCount + 1
    }
}

print("Validated: " + string(validCount))
print("Errors: " + string(errorCount))
```

### Example 3: Channel Communication

```zexus
# Reading from channel until null (closed)
print("Consuming messages from channel...")

channel<integer> data_channel

async action producer() {
    let i = 1
    while i <= 5 {
        send(data_channel, i * 10)
        i = i + 1
    }
    close_channel(data_channel)
}

async action consumer() {
    let total = 0
    while true {
        let value = receive(data_channel)
        if value == null {
            break  # Channel closed, exit loop
        }
        total = total + value
        print("Received: " + string(value))
    }
    print("Total: " + string(total))
}

async producer()
async consumer()
sleep(1)  # Wait for completion
```

**Output:**
```
Consuming messages from channel...
Received: 10
Received: 20
Received: 30
Received: 40
Received: 50
Total: 150
```

### Example 4: Infinite Loop with Exit Condition

```zexus
print("Monitoring system (Ctrl+C to stop)...")

let counter = 0
let threshold = 1000

while true {
    counter = counter + 1
    
    let status = checkSystemStatus()
    
    if status == "CRITICAL" {
        print("Critical error detected at iteration " + string(counter))
        break
    }
    
    if counter >= threshold {
        print("Reached maximum iterations")
        break
    }
    
    if counter % 100 == 0 {
        print("Status check #" + string(counter) + ": OK")
    }
}

print("Monitoring stopped")
```

### Example 5: Nested Loop Search

```zexus
print("Searching 2D matrix...")

let matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

let target = 7
let found = false
let rowIndex = 0

for each row in matrix {
    let colIndex = 0
    for each value in row {
        if value == target {
            print("Found " + string(target) + " at [" + string(rowIndex) + "][" + string(colIndex) + "]")
            found = true
            break  # Exit inner loop
        }
        colIndex = colIndex + 1
    }
    
    if found {
        break  # Exit outer loop
    }
    rowIndex = rowIndex + 1
}
```

**Output:**
```
Searching 2D matrix...
Found 7 at [1][2]
```

## Rules and Best Practices

### ✅ DO:

1. **Use for early termination**
   ```zexus
   for each item in largeList {
       if isTargetFound(item) {
           break  # Stop searching
       }
   }
   ```

2. **Combine with conditional checks**
   ```zexus
   while true {
       let data = fetchData()
       if data == null || data == "END" {
           break
       }
       process(data)
   }
   ```

3. **Use in channel operations**
   ```zexus
   while true {
       let msg = receive(channel)
       if msg == null {
           break  # Channel closed
       }
       handleMessage(msg)
   }
   ```

4. **Document why you're breaking**
   ```zexus
   if count > MAX_ALLOWED {
       print("Exceeded maximum: " + string(MAX_ALLOWED))
       break  # Prevent overflow
   }
   ```

### ❌ DON'T:

1. **Don't use outside loops**
   ```zexus
   # ❌ ERROR: break outside loop
   if condition {
       break  # This will cause an error
   }
   ```

2. **Don't use as a substitute for proper conditions**
   ```zexus
   # ❌ Poor practice
   while true {
       if i >= 10 {
           break
       }
       i = i + 1
   }
   
   # ✅ Better
   while i < 10 {
       i = i + 1
   }
   ```

3. **Don't overuse in simple iterations**
   ```zexus
   # ❌ Unnecessary complexity
   for each item in items {
       process(item)
       if isLastItem(item) {
           break
       }
   }
   
   # ✅ Simpler
   for each item in items {
       process(item)
   }
   ```

## Comparison with Other Keywords

### BREAK vs CONTINUE (error recovery)

| Feature | BREAK | CONTINUE |
|---------|-------|----------|
| Purpose | Exit loop | Enable error recovery |
| Scope | Current loop only | Entire program |
| Execution | Stops loop immediately | Continues execution after errors |
| Use Case | Loop control flow | Error handling |

**Note:** Zexus has two different uses of "continue":
- `continue` (error recovery keyword) - Enables error recovery mode
- Loop continue (not yet implemented) - Skip to next iteration

### BREAK vs RETURN

| Feature | BREAK | RETURN |
|---------|-------|--------|
| Exits | Current loop | Entire function |
| Returns Value | No | Yes (optional) |
| Scope | Loop body | Function body |

```zexus
action findValue(items, target) {
    for each item in items {
        if item == target {
            return item  # Exit function, return value
        }
    }
    return null
}

action processUntilCondition(items) {
    for each item in items {
        if shouldStop(item) {
            break  # Exit loop, continue function
        }
        process(item)
    }
    return "Processing complete"
}
```

## Error Handling

### Invalid Usage

```zexus
# This will cause a runtime error or unexpected behavior:
if condition {
    break  # ❌ Not in a loop
}
```

**Error message:**
```
BreakException not caught - break used outside of loop context
```

### Safe Patterns

```zexus
# ✅ Always use break inside loops
while true {
    let shouldExit = checkCondition()
    if shouldExit {
        break  # Safe - inside while loop
    }
}

for each item in items {
    if isTarget(item) {
        break  # Safe - inside for each loop
    }
}
```

## Performance Considerations

### Optimization Benefits

Using `BREAK` can significantly improve performance:

```zexus
# Without break: O(n) - always checks all items
let found = false
for each item in millionItems {
    if item == target {
        found = true
        # Continues checking remaining items ❌
    }
}

# With break: O(1) to O(n) - stops when found
let found = false
for each item in millionItems {
    if item == target {
        found = true
        break  # Stops immediately ✅
    }
}
```

### Best Practices for Performance

1. **Early exit in search operations**
2. **Limit iterations in validation**
3. **Stop on critical conditions**
4. **Prevent infinite loops with timeouts**

## Implementation Details

### Files Modified
- `src/zexus/zexus_token.py` - Added BREAK token
- `src/zexus/lexer.py` - Added "break" keyword recognition
- `src/zexus/zexus_ast.py` - Added BreakStatement AST node
- `src/zexus/parser/parser.py` - Added parse_break_statement()
- `src/zexus/parser/strategy_structural.py` - Added BREAK to statement_starters
- `src/zexus/parser/strategy_context.py` - Added BREAK handler in _parse_block_statements
- `src/zexus/evaluator/core.py` - Added BreakStatement dispatch
- `src/zexus/evaluator/statements.py` - Added BreakException class and handlers

### How It Works Internally

1. **Parsing**: `break` keyword creates a `BreakStatement` AST node
2. **Evaluation**: `eval_break_statement()` returns a `BreakException` object
3. **Loop Handling**: Loops catch `BreakException` and exit cleanly
4. **Propagation**: Block statements propagate `BreakException` to outer loops
5. **Cleanup**: Loop returns `NULL` when break is encountered

## Compatibility

- **Version:** Zexus v1.6.0+
- **Breaking Changes:** None
- **Backward Compatible:** Yes (new keyword)
- **Platform:** All supported platforms

## Summary

The `BREAK` keyword is essential for controlling loop execution in Zexus:

✅ **Simple syntax:** Just `break;`
✅ **Immediate exit:** Stops the current loop instantly
✅ **Nested loop support:** Only affects innermost loop
✅ **Works with all loop types:** `while` and `for each`
✅ **Performance benefits:** Early termination saves iterations
✅ **Clean implementation:** Uses exception-based control flow

Use `BREAK` whenever you need to exit a loop early based on conditions, search results, or error states. It's a fundamental control flow tool that makes your code more efficient and readable.
