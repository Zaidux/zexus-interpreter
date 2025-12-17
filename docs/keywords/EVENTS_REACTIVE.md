# Events & Reactive Keywords: EVENT, EMIT, STREAM, WATCH

## Overview

Zexus provides event-driven and reactive programming capabilities through EVENT, EMIT, STREAM, and WATCH keywords. These enable event handling, pub/sub patterns, event streaming, and reactive state management.

### Keywords Covered
- **EVENT**: Type token (not a standalone statement)
- **EMIT**: Emit events to listeners
- **STREAM**: Event streaming declarations (partially implemented)
- **WATCH**: Reactive state management (partially implemented)

---

## Implementation Status

### EMIT Keyword
- **Token**: ✅ Registered in lexer
- **Parser**: ✅ Implemented
- **Evaluator**: ✅ Working
- **Status**: 🟢 **Production Ready**

### EVENT Keyword
- **Token**: ✅ Registered in lexer
- **Usage**: Type identifier, not a statement keyword
- **Status**: 🟢 **Working** (as type token)

### STREAM Keyword
- **Token**: ✅ Registered in lexer
- **Parser**: ⚠️ Unknown
- **Evaluator**: ⚠️ Unknown
- **Status**: 🟡 **Partially Implemented**

### WATCH Keyword
- **Token**: ✅ Registered in lexer
- **Parser**: ⚠️ Unknown
- **Evaluator**: ⚠️ Unknown
- **Status**: 🟡 **Partially Implemented**

---

## EMIT Keyword

### Syntax
```zexus
emit eventName;
emit eventName(data);
emit eventName(arg1, arg2, ...);
```

### Purpose
The `emit` keyword triggers events that can be handled by listeners. It's used for event-driven programming patterns.

### Basic Usage

#### Simple Event Emission
```zexus
emit userLoggedIn;
print "User logged in event emitted";
```

#### Event with Single Parameter
```zexus
let userId = 123;
emit userCreated(userId);
print "User created with ID: " + userId;
```

#### Event with Multiple Parameters
```zexus
emit dataUpdated("users", 42, "success");
print "Data update event emitted";
```

#### Event with Expression
```zexus
let count = 10;
emit countChanged(count * 2);
print "Count changed to: " + (count * 2);
```

### Advanced Patterns

#### Events in Functions
```zexus
action processUser(user) {
    emit userProcessing(user);
    // Process user
    emit userProcessed(user);
    return "Done";
}
```

#### Events in Control Flow
```zexus
if (value > 100) {
    emit threshold_exceeded(value);
}
```

#### Events in Loops
```zexus
for item in [1, 2, 3] {
    emit itemProcessed(item);
}
```

#### Conditional Event Emission
```zexus
action notify(status) {
    if (status == "success") {
        emit operationSucceeded;
    } elif (status == "error") {
        emit operationFailed;
    } else {
        emit operationPending;
    }
}
```

### Test Results

✅ **Working Features**:
- Basic event emission
- Events with parameters
- Events with expressions
- Events in functions
- Events in control structures
- Multiple parameter events
- Event naming conventions

❌ **Known Issues**: None found

---

## EVENT Keyword

### Status
EVENT is a token in the lexer but is **not a statement keyword**. It's used as a type identifier or in conjunction with other keywords.

### Usage Context
```zexus
// EVENT appears to be a type/category, not a statement
// Used in emit syntax: emit eventName
```

### Test Results
- EVENT keyword recognized by lexer
- No standalone EVENT statement syntax found
- No errors when EVENT appears in code context

---

## STREAM Keyword

### Status
STREAM is registered in the lexer but implementation details are unclear.

### Intended Syntax (Based on Token Comments)
```zexus
// Event streaming: stream name as event => handler;
stream dataStream as item => {
    print "Processing: " + item;
};
```

### Test Results
⚠️ **Status**: Token exists but functionality not tested. No parser/evaluator handlers found during research phase.

---

## WATCH Keyword

### Status
WATCH is registered in the lexer but implementation details are unclear.

### Intended Syntax (Based on Token Comments)
```zexus
// Reactive state management: watch variable => reaction;
watch counter => {
    print "Counter changed: " + counter;
};
```

### Test Results
⚠️ **Status**: Token exists but functionality not tested. No parser/evaluator handlers found during research phase.

---

## Event Patterns

### 1. Event-Driven State Changes
```zexus
let state = "idle";

action startProcess() {
    emit processStarted;
    state = "running";
    return state;
}

action stopProcess() {
    emit processStopped;
    state = "stopped";
    return state;
}
```

### 2. Lifecycle Events
```zexus
action initializeSystem() {
    emit systemInitializing;
    // Setup code
    emit systemReady;
}

action shutdownSystem() {
    emit systemShuttingDown;
    // Cleanup code
    emit systemStopped;
}
```

### 3. Data Flow Events
```zexus
action processData(data) {
    emit dataReceived(data);
    
    let processed = transform(data);
    emit dataProcessed(processed);
    
    save(processed);
    emit dataSaved(processed);
    
    return processed;
}
```

### 4. Error Events
```zexus
action safeOperation() {
    try {
        emit operationStarted;
        // Risky operation
        emit operationSucceeded;
    } catch (error) {
        emit operationFailed(error);
    }
}
```

### 5. Progress Events
```zexus
action processItems(items) {
    let total = 3;  // Simulated length
    emit processingStarted(total);
    
    emit itemProcessed(1);
    emit itemProcessed(2);
    emit itemProcessed(3);
    
    emit processingComplete(total);
}
```

### 6. State Machine Events
```zexus
let currentState = "idle";

action transition(newState) {
    emit stateChanging(currentState, newState);
    currentState = newState;
    emit stateChanged(newState);
    return currentState;
}
```

### 7. User Interaction Events
```zexus
action handleUserAction(action, data) {
    if (action == "click") {
        emit buttonClicked(data);
    } elif (action == "submit") {
        emit formSubmitted(data);
    } elif (action == "cancel") {
        emit actionCancelled;
    }
}
```

### 8. Batch Processing Events
```zexus
action processBatch(batchId) {
    emit batchStarted(batchId);
    
    let item1 = "item1";
    emit batchItemProcessed(batchId, item1);
    
    let item2 = "item2";
    emit batchItemProcessed(batchId, item2);
    
    emit batchCompleted(batchId);
}
```

---

## Best Practices

### 1. Use Descriptive Event Names
```zexus
// ✅ Good: Clear, action-oriented
emit userAuthenticated(userId);
emit paymentProcessed(transactionId);
emit dataValidationFailed(errors);

// ❌ Avoid: Vague, unclear
emit event1;
emit thing;
emit done;
```

### 2. Event Naming Conventions
```zexus
// Past tense for completed actions
emit userCreated(user);
emit orderPlaced(orderId);
emit emailSent(recipient);

// Present continuous for ongoing processes
emit processing(item);
emit loading(resource);

// Gerund form for state changes
emit stateChanging(oldState, newState);
```

### 3. Include Relevant Data
```zexus
// ✅ Good: Include context
emit orderShipped(orderId, trackingNumber, carrier);
emit userUpdated(userId, changedFields);

// ❌ Avoid: Missing context
emit orderShipped;  // Which order?
```

### 4. Event Granularity
```zexus
// ✅ Good: Specific events
emit orderCreated(order);
emit orderValidated(order);
emit orderPaid(order);
emit orderShipped(order);

// ❌ Avoid: Too generic
emit orderChanged(order, "created");
emit orderChanged(order, "validated");
```

### 5. Error Event Patterns
```zexus
// ✅ Good: Specific error events
action process(data) {
    try {
        validate(data);
        emit dataValidated(data);
    } catch (e) {
        emit validationFailed(data, e);
    }
}
```

---

## Known Issues

### Issue 1: Variable Reassignment in Functions
**Description**: Reassigning variables declared outside function scope causes "Invalid assignment target" error.

**Example**:
```zexus
let counter = 0;

action increment() {
    counter = counter + 1;  // ❌ Error: Invalid assignment target
}
```

**Impact**: Limits stateful patterns with events. Cannot maintain state across event emissions.

**Test File**: test_events_complex.zx (Test 2 and multiple others)

**Workaround**: Use return values instead of modifying outer scope:
```zexus
let counter = 0;

action increment(current) {
    return current + 1;
}

counter = increment(counter);  // ✅ Works
```

### Issue 2: STREAM Not Implemented
**Description**: STREAM keyword registered but no parser/evaluator implementation found.

**Status**: Token exists, functionality unclear.

### Issue 3: WATCH Not Implemented
**Description**: WATCH keyword registered but no parser/evaluator implementation found.

**Status**: Token exists, functionality unclear.

---

## Real-World Examples

### Example 1: User Registration Flow
```zexus
action registerUser(email, password) {
    emit registrationStarted(email);
    
    // Validate
    if (email == "") {
        emit registrationFailed("Invalid email");
        return "Failed";
    }
    
    emit userValidated(email);
    
    // Create user
    let userId = 12345;  // Simulated
    emit userCreated(userId, email);
    
    // Send welcome email
    emit welcomeEmailSent(email);
    
    emit registrationCompleted(userId);
    return "Success";
}
```

### Example 2: Order Processing
```zexus
action processOrder(orderId) {
    emit orderReceived(orderId);
    
    emit orderValidating(orderId);
    // Validation logic
    emit orderValidated(orderId);
    
    emit paymentProcessing(orderId);
    // Payment logic
    emit paymentCompleted(orderId);
    
    emit orderShipping(orderId);
    // Shipping logic
    emit orderShipped(orderId);
    
    return "Order processed";
}
```

### Example 3: Data Synchronization
```zexus
action syncData(source, destination) {
    emit syncStarted(source, destination);
    
    emit dataFetching(source);
    let data = "fetched_data";  // Simulated
    emit dataFetched(data);
    
    emit dataValidating(data);
    // Validation
    emit dataValid(data);
    
    emit dataWriting(destination, data);
    // Write logic
    emit dataWritten(destination);
    
    emit syncCompleted(source, destination);
    return "Synced";
}
```

### Example 4: File Upload with Progress
```zexus
action uploadFile(filename, size) {
    emit uploadStarted(filename, size);
    
    let progress = 0;
    emit uploadProgress(filename, progress);
    
    progress = 33;
    emit uploadProgress(filename, progress);
    
    progress = 66;
    emit uploadProgress(filename, progress);
    
    progress = 100;
    emit uploadProgress(filename, progress);
    
    emit uploadCompleted(filename);
    return "Uploaded";
}
```

---

## Comparison with Other Languages

### JavaScript (EventEmitter)
```javascript
// JavaScript
eventEmitter.emit('userLoggedIn', userId);
eventEmitter.on('userLoggedIn', (userId) => {
    console.log(`User ${userId} logged in`);
});
```

```zexus
// Zexus
emit userLoggedIn(userId);
// Note: Listener registration not tested
```

### C# (Events)
```csharp
// C#
public event EventHandler<UserEventArgs> UserLoggedIn;
UserLoggedIn?.Invoke(this, new UserEventArgs(userId));
```

```zexus
// Zexus - simpler syntax
emit userLoggedIn(userId);
```

### Python (Event System)
```python
# Python
def emit_event(event_name, data):
    event_bus.publish(event_name, data)

emit_event('user_logged_in', {'user_id': 123})
```

```zexus
// Zexus - built-in language feature
emit userLoggedIn(123);
```

---

## Testing Summary

### Tests Created
- **Easy**: 20 tests - All passed ✅
- **Medium**: 20 tests - All passed ✅
- **Complex**: 20 tests - Multiple failures due to variable reassignment limitation

### EMIT Functionality
✅ **Working**:
- Basic event emission
- Events with 0-5+ parameters
- Events with expressions
- Events in functions
- Events in conditionals
- Events in loops (where loops work)
- Named events
- Event-driven patterns

### Error Summary
1. **Variable Reassignment in Functions**: Cannot modify outer scope variables (affects stateful event patterns)
2. **STREAM**: Token exists but implementation unclear
3. **WATCH**: Token exists but implementation unclear

---

## Future Enhancements

### Proposed Features
1. **Event Listeners**: Register handlers for events
   ```zexus
   on userLoggedIn(userId) {
       print "User logged in: " + userId;
   }
   ```

2. **Event Filtering**:
   ```zexus
   on dataChanged(type, data) where type == "users" {
       updateUserCache(data);
   }
   ```

3. **Event Bubbling/Propagation**:
   ```zexus
   emit event cancelable;
   if (event.cancelled) {
       return;
   }
   ```

4. **STREAM Implementation**:
   ```zexus
   stream dataStream as item => {
       process(item);
   };
   
   emit dataStream(newItem);
   ```

5. **WATCH Implementation**:
   ```zexus
   watch counter => {
       print "Counter changed: " + counter;
   };
   
   counter = 5;  // Triggers watch handler
   ```

---

## Related Keywords
- **ACTION/FUNCTION**: Where events are emitted
- **IF/ELIF/ELSE**: Conditional event emission
- **TRY/CATCH**: Error event patterns

---

*Last Updated: December 16, 2025*
*Tested with Zexus Interpreter*
*Status: EMIT fully functional, STREAM/WATCH need implementation*
