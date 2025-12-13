# WATCH Feature - Reactive State Management

## Overview

The WATCH feature provides reactive state management in Zexus, allowing automatic re-execution of code blocks when dependencies change. This implements a reactive programming model similar to computed properties in Vue.js or reactive statements in Svelte.

## Implementation Date
December 13, 2025

## Syntax

### Implicit Watch
Automatically captures all variable dependencies within the block:

```zexus
let x = 10;
let y = 0;

watch {
    y = x * 2;
    print("x changed, y is now: " + y);
}
// Executes immediately, then re-executes whenever x changes
```

### Explicit Watch
Watches a specific expression:

```zexus
let name = "Alice";

watch name => {
    print("Name changed to: " + name);
}
// Only executes the block when 'name' changes
```

## Architecture

### 1. AST Node (`src/zexus/zexus_ast.py`)

```python
class WatchStatement(Statement):
    def __init__(self, reaction, watched_expr=None):
        self.reaction = reaction  # BlockStatement: code to execute
        self.watched_expr = watched_expr  # Optional Expression: explicit dependency
```

- **reaction**: The code block to execute when dependencies change
- **watched_expr**: Optional expression for explicit dependency tracking

### 2. Parser Support

#### Strategy Context Parser (`src/zexus/parser/strategy_context.py`)
- Handles both implicit (`watch { }`) and explicit (`watch expr => { }`) syntax
- Integrated into the advanced/tolerant parsing pipeline
- Added WATCH to statement_starters set in structural analyzer

#### Traditional Parser (`src/zexus/parser/parser.py`)
- Updated `parse_watch_statement` to correctly order constructor parameters
- Fixed parameter order bug: `WatchStatement(reaction, watched_expr)` not `WatchStatement(watched_expr, reaction)`

### 3. Runtime Objects (`src/zexus/object.py`)

#### Global Dependency Tracking
```python
_dependency_collector_stack = []

def start_collecting_dependencies():
    _dependency_collector_stack.append(set())

def stop_collecting_dependencies():
    return _dependency_collector_stack.pop() if _dependency_collector_stack else set()

def record_dependency(env, name):
    if _dependency_collector_stack:
        _dependency_collector_stack[-1].add((env, name))
```

#### Environment Class Enhancements
```python
class Environment:
    def __init__(self, outer=None):
        # ... existing code ...
        self.watchers = {}  # name -> list of callbacks
    
    def get(self, name):
        val = self.store.get(name)
        if val is not None:
            record_dependency(self, name)  # Track reads
            return val
        # ... rest of get logic ...
    
    def set(self, name, val):
        self.store[name] = val
        self.notify_watchers(name, val)  # Trigger watchers
        return val
    
    def assign(self, name, val):
        # ... existing assignment logic ...
        self.notify_watchers(name, val)  # Trigger watchers on all paths
        return val
    
    def add_watcher(self, name, callback):
        if name not in self.watchers:
            self.watchers[name] = []
        self.watchers[name].append(callback)
    
    def notify_watchers(self, name, new_val):
        if name in self.watchers:
            callbacks = self.watchers[name][:]
            for cb in callbacks:
                try:
                    cb(new_val)
                except Exception as e:
                    print(f"Error in watcher for {name}: {e}")
```

### 4. Evaluator (`src/zexus/evaluator/statements.py`)

```python
def eval_watch_statement(self, node, env, stack_trace):
    # 1. Start collecting dependencies
    start_collecting_dependencies()
    
    # 2. Evaluate watched expression or block
    if node.watched_expr:
        # Explicit: just evaluate the expression to capture deps
        res = self.eval_node(node.watched_expr, env, stack_trace)
    else:
        # Implicit: execute block and capture deps
        res = self.eval_node(node.reaction, env, stack_trace)
    
    # 3. Stop collecting and get dependencies
    deps = stop_collecting_dependencies()
    
    # 4. Define reaction callback with infinite loop guard
    executing = [False]
    def reaction_callback(new_val):
        if executing[0]:
            return  # Prevent infinite recursion
        executing[0] = True
        try:
            self.eval_node(node.reaction, env, [])
        finally:
            executing[0] = False
    
    # 5. Register callback for each dependency
    for dep_env, name in deps:
        dep_env.add_watcher(name, reaction_callback)
    
    return NULL
```

## Key Design Decisions

### 1. Dependency Collection
- Uses a **global stack** to track nested watch statements
- Records `(environment, variable_name)` tuples during evaluation
- Automatically captures all `Environment.get()` calls within the watch block

### 2. Infinite Loop Prevention
- Each reaction callback has a **guard flag** (`executing`)
- Prevents recursive triggering when the reaction modifies its own dependencies
- Uses a mutable list `[False]` to capture state in closure

### 3. Static Dependencies
- Dependencies are captured **once** during initial evaluation
- Callbacks do NOT re-collect dependencies during execution
- This prevents dynamic dependency changes from causing unexpected behavior

### 4. Notification Timing
- Watchers are triggered **synchronously** when variables change
- All registered callbacks execute immediately in order
- No batching or deferred execution

## Usage Examples

### Example 1: Computed Property
```zexus
let firstName = "John";
let lastName = "Doe";
let fullName = "";

watch {
    fullName = firstName + " " + lastName;
}

print(fullName);  // "John Doe"
firstName = "Jane";
print(fullName);  // "Jane Doe" (automatically updated)
```

### Example 2: Validation
```zexus
let age = 15;
let isAdult = false;

watch age => {
    isAdult = age >= 18;
    if (isAdult) {
        print("User is an adult");
    }
}

age = 20;  // Triggers: "User is an adult"
```

### Example 3: Chained Reactions
```zexus
let celsius = 0;
let fahrenheit = 0;
let kelvin = 0;

watch celsius => {
    fahrenheit = (celsius * 9 / 5) + 32;
}

watch fahrenheit => {
    kelvin = (fahrenheit - 32) * 5 / 9 + 273.15;
}

celsius = 100;
// fahrenheit becomes 212
// kelvin becomes 373.15
```

### Example 4: Side Effects
```zexus
let items = [];
let itemCount = 0;

watch items => {
    itemCount = len(items);
    print("Item count updated: " + itemCount);
}

items = [1, 2, 3];  // Triggers: "Item count updated: 3"
```

## Implementation Challenges & Solutions

### Challenge 1: Parameter Order Bug
**Problem**: Traditional parser was passing parameters to `WatchStatement` constructor in wrong order  
**Solution**: Fixed `parse_watch_statement` in parser.py to use `WatchStatement(reaction, watched_expr)`

### Challenge 2: Infinite Loops
**Problem**: Reactions that modify their own dependencies cause infinite recursion  
**Solution**: Added `executing` flag guard in reaction callback

### Challenge 3: Environment Scope
**Problem**: Watchers registered on wrong environment wouldn't trigger  
**Solution**: Record exact environment reference with each dependency during collection

### Challenge 4: Implicit vs Explicit Semantics
**Problem**: Implicit watch should execute immediately, explicit should not  
**Solution**: 
- Implicit: Execute block during setup to capture deps AND produce side effects
- Explicit: Only evaluate expression to capture deps, don't execute reaction

## Testing

Comprehensive tests in `test_watch_feature.zx`:

1. **Test 1: Implicit Watch** ✅
   - Captures multiple dependencies (x, y)
   - Executes block immediately
   - Re-executes on dependency changes
   - Prevents infinite loops

2. **Test 2: Explicit Watch** ✅
   - Watches single variable (a)
   - Does NOT execute initially
   - Executes reaction only when watched variable changes

3. **Test 3: Chained Watch** ✅
   - Multiple watches with cascading dependencies
   - Demonstrates reactive chains: p → q → r

## Performance Considerations

### Pros
- ✅ Minimal overhead when watchers not used
- ✅ O(1) dependency lookup
- ✅ No global polling or dirty checking

### Cons
- ❌ Memory overhead: Each watched variable stores callback list
- ❌ Static dependencies only (captured at definition time)
- ❌ No automatic cleanup (watchers persist for environment lifetime)

## Future Enhancements

### Potential Improvements
1. **Watcher Cleanup API**
   ```zexus
   let handle = watch x => { ... };
   handle.dispose();  // Remove watcher
   ```

2. **Batch Updates**
   ```zexus
   batch {
       x = 10;
       y = 20;
       // Watchers fire once after batch
   }
   ```

3. **Computed Properties**
   ```zexus
   let computed = watch(x, y) => x + y;
   print(computed.value);  // Auto-updates
   ```

4. **Async Reactions**
   ```zexus
   watch data => async {
       await saveToDatabase(data);
   }
   ```

5. **Conditional Watching**
   ```zexus
   watch x if (enabled) => {
       // Only active when enabled is true
   }
   ```

## Integration with Other Features

### Compatible With
- ✅ Functions and closures
- ✅ Scope chains (outer environments)
- ✅ Try-catch error handling
- ✅ Const variables (watch but can't modify)
- ✅ Module system

### Special Cases
- **Sealed variables**: Can watch but reaction can't modify sealed values
- **Module exports**: Watchers don't cross module boundaries
- **Const dependencies**: Can watch const variables (they trigger on initial set)

## Conclusion

The WATCH feature successfully implements reactive state management in Zexus with:
- Clean, intuitive syntax
- Automatic dependency tracking
- Infinite loop protection
- Support for both implicit and explicit dependency models
- Minimal runtime overhead

This foundation enables reactive programming patterns and paves the way for UI frameworks and reactive data pipelines in Zexus.
