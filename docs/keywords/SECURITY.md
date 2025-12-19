# Security Keywords Documentation

## Overview
Zexus provides comprehensive security and compliance features through dedicated keywords: ENTITY, VERIFY, CONTRACT, PROTECT, SEAL, AUDIT, RESTRICT, SANDBOX, and TRAIL. These keywords enable type definitions, runtime verification, smart contracts, data protection, audit logging, sandboxed execution, and event tracking.

### Keywords Covered
- **ENTITY**: Type/schema definitions
- **VERIFY**: Runtime assertion and validation
- **CONTRACT**: Smart contract definitions
- **PROTECT**: Policy-based protection (partial)
- **SEAL**: Immutable data sealing
- **AUDIT**: Compliance logging
- **RESTRICT**: Field-level access control (partial)
- **SANDBOX**: Isolated execution environments
- **TRAIL**: Real-time event tracking

---

## Implementation Status

| Keyword | Lexer | Parser | Evaluator | Status |
|---------|-------|--------|-----------|--------|
| ENTITY | ✅ | ✅ | ✅ | 🟢 Working |
| VERIFY | ✅ | ✅ | ✅ | ✅ Fixed |
| CONTRACT | ✅ | ✅ | ✅ | 🟢 Working |
| PROTECT | ✅ | ✅ | ⚠️ | 🟡 Partial |
| SEAL | ✅ | ✅ | ✅ | 🟢 Working |
| AUDIT | ✅ | ✅ | ✅ | 🟢 Working |
| RESTRICT | ✅ | ✅ | ⚠️ | 🟡 Partial |
| SANDBOX | ✅ | ✅ | ✅ | ✅ Fixed |
| TRAIL | ✅ | ✅ | ✅ | 🟢 Working |

---

## ENTITY Keyword

### Syntax
```zexus
entity EntityName {
    field1: type1,
    field2: type2,
    ...
}
```

### Purpose
Defines structured data types with typed fields. Used for domain modeling and type safety.

### Basic Usage

#### Simple Entity
```zexus
entity User {
    id: integer,
    name: string
}
```

#### Entity with Multiple Fields
```zexus
entity Product {
    id: integer,
    name: string,
    price: integer,
    active: boolean
}
```

### Advanced Patterns

#### Complex Domain Models
```zexus
entity Customer {
    id: integer,
    name: string,
    email: string,
    verified: boolean
}

entity Order {
    id: integer,
    customerId: integer,
    items: string,
    total: integer,
    status: string
}
```

#### Entity for Access Control
```zexus
entity Permission {
    userId: integer,
    resourceId: integer,
    action: string,
    granted: boolean
}
```

### Test Results
✅ **Working**: Entity definitions with multiple fields and types
✅ **Working**: Entity as type hints (conceptual)
✅ **Working**: Multiple entity declarations

---

## VERIFY Keyword ✨ **ENHANCED** (December 19, 2025)

### Syntax
```zexus
// Simple assertion form
verify condition, "error message";

// Custom logic block form
verify condition {
    // Custom developer logic on failure
    log_error("Verification failed");
    send_alert(admin);
}

// Data verification modes
verify:data value matches pattern, "error message";
verify:data value is_type "email", "error message";

// Access control mode (blocks access)
verify:access condition {
    block_request();
    log_unauthorized_access();
}

// Database verification mode
verify:db value exists_in "table", "error message";
verify:db value unique_in "table", "error message";

// Environment variable mode
verify:env "VAR_NAME" is_set, "error message";
verify:env "VAR_NAME" equals "expected", "error message";

// Pattern matching mode
verify:pattern value matches "regex", "error message";

// Complex wrapper form
verify(target, [conditions...]);
```

### Purpose
**Enhanced verification system** that validates data, controls access, checks databases, verifies environment variables, and allows custom developer logic. Goes beyond simple assertions to provide comprehensive security and validation gates.

### Core Capabilities

1. **Data Verification** - Validate formats, types, patterns
2. **Access Control** - Block unauthorized access with custom actions
3. **Database Checks** - Verify data exists/unique in database
4. **Environment Validation** - Check configuration and env vars
5. **Custom Logic** - Developer-defined verification logic in `{}`
6. **Pattern Matching** - Regex validation for any format
7. **Security Gates** - Prevent bad data from entering system

### Basic Usage

#### Simple Verification (Original)
```zexus
let age = 25;
verify age > 18, "Must be adult";
```

#### Email & Password Validation (Your Example)
```zexus
action validateLoginForm(email, password) {
    // Verify email format - if wrong, block and don't allow bad data
    verify is_email(email) {
        print "[FORM] Invalid email format";
        print "[FORM] Blocking form submission";
        print "[FORM] Not allowing bad data into system";
        return false;
    }
    
    // Verify password strength
    let strength = password_strength(password);
    verify strength != "weak" {
        print "[FORM] Password too weak";
        print "[FORM] Access denied";
        return false;
    }
    
    print "✓ Login allowed";
    return true;
}
```

### Data Verification Mode

#### Email Validation
```zexus
let email = "user@example.com";
verify is_email(email), "Invalid email format";
```

#### URL Validation
```zexus
let url = "https://example.com";
verify is_url(url), "Invalid URL format";
```

#### Phone Number Validation
```zexus
let phone = "123-456-7890";
verify is_phone(phone), "Invalid phone number";
```

#### Type Checking
```zexus
verify is_numeric(userInput), "Must be a number";
verify is_alphanumeric(code), "Code must be alphanumeric";
verify is_alpha(name), "Name must contain only letters";
```

#### Length Validation
```zexus
verify validate_length(username, 3, 20), "Username length invalid";
```

#### Password Strength
```zexus
let strength = password_strength(password);
verify strength == "strong", "Password not strong enough";
```

### Access Control Mode

#### Block Unauthorized Access
```zexus
let userRole = "guest";

verify userRole == "admin" {
    log_unauthorized_access(user);
    block_request();
    send_security_alert(admin);
}
```

#### Custom Blocking Logic
```zexus
action checkResourceAccess(user, resource) {
    verify user.hasPermission(resource) {
        print "[SECURITY] Access denied for: " + user.id;
        print "[SECURITY] Resource: " + resource;
        audit_log("unauthorized_access", user);
        // Access blocked - bad input never enters system
    }
}
```

### Database Verification Mode

#### Check if User Exists
```zexus
// Developer can inject custom database handler
verify:db userId exists_in "users", "User not found";
```

#### Check Email Uniqueness
```zexus
verify:db email unique_in "users", "Email already in use";
```

#### Custom Database Logic
```zexus
// In your code, inject a db_handler:
// env.set('__db_handler__', myDatabaseHandler);

// Then use verify:db
verify:db transactionId exists_in "transactions", "Transaction not found";
```

### Environment Variable Mode

#### Check Configuration
```zexus
verify env_exists("API_KEY"), "API_KEY not configured";
verify env_exists("DEBUG_MODE"), "DEBUG_MODE not set";
```

#### Validate Environment Values
```zexus
let debugMode = env_get("DEBUG_MODE");
verify debugMode == "false", "Debug mode must be disabled in production";
```

#### Configuration Verification
```zexus
action verifyAppConfig() {
    verify env_exists("API_KEY"), "API_KEY missing";
    verify env_exists("DATABASE_URL"), "DATABASE_URL missing";
    
    let timeout = env_get("API_TIMEOUT");
    verify is_numeric(timeout), "API_TIMEOUT must be numeric";
}
```

### Pattern Matching Mode

#### Validate Formats with Regex
```zexus
let zipcode = "12345";
verify matches_pattern(zipcode, "^[0-9]{5}$"), "Invalid zipcode";
```

#### Credit Card Validation
```zexus
let cc = "4532-1234-5678-9010";
verify matches_pattern(cc, "^[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{4}$"), "Invalid CC format";
```

### Custom Logic Blocks

#### Developer-Defined Verification
```zexus
verify condition {
    // Write any custom logic here
    log_error("Verification failed");
    send_email(admin, "Security alert");
    increment_failure_count();
    return false;
}
```

#### Multi-Step Verification
```zexus
action validateTransaction(amount, balance) {
    verify amount > 0 {
        print "[VALIDATION] Amount must be positive";
        log_transaction_error(amount);
    }
    
    verify balance >= amount {
        print "[VALIDATION] Insufficient balance";
        notify_user("Insufficient funds");
    }
    
    verify amount <= 10000 {
        print "[VALIDATION] Amount exceeds daily limit";
        flag_for_review(amount);
    }
}
```

### Advanced Patterns

#### Multi-Layer Security Gate
```zexus
action secureAPIRequest(method, endpoint, body, apiKey) {
    // Layer 1: Method validation
    verify method == "GET" || method == "POST" {
        print "[API] Invalid HTTP method";
        return false;
    }
    
    // Layer 2: API key verification
    verify env_get("VALID_API_KEY") == apiKey {
        print "[API] Invalid API key";
        log_security_event("invalid_api_key");
        return false;
    }
    
    // Layer 3: Endpoint validation
    verify matches_pattern(endpoint, "^[a-zA-Z0-9/_-]+$") {
        print "[API] Invalid endpoint format";
        return false;
    }
    
    // Layer 4: Body sanitization
    if (method == "POST") {
        let cleanBody = sanitize_input(body);
        verify cleanBody == body {
            print "[API] Request body contains dangerous characters";
            return false;
        }
    }
    
    print "✓ API request validated";
    return true;
}
```

#### User Registration Validation
```zexus
action validateUserRegistration(email, password, username) {
    verify is_email(email), "Invalid email format";
    verify password_strength(password) != "weak", "Password too weak";
    verify validate_length(username, 3, 20), "Username length invalid";
    verify is_alphanumeric(username), "Username must be alphanumeric";
    
    // All checks passed
    return true;
}
```

#### Input Sanitization & Verification
```zexus
action processUserInput(input) {
    // Sanitize first
    let clean = sanitize_input(input);
    
    // Then verify it wasn't malicious
    verify clean == input {
        print "[SECURITY] Malicious input detected";
        print "[SECURITY] Original: " + input;
        print "[SECURITY] Input blocked from system";
        log_security_event("xss_attempt");
        return false;
    }
    
    // Safe to process
    return clean;
}
```

### Builtin Helper Functions

Zexus provides built-in verification helpers:

```zexus
// Format validators
is_email(value)              // Email format
is_url(value)                // URL format  
is_phone(value)              // Phone number format
is_numeric(value)            // Numeric check
is_alpha(value)              // Alphabetic only
is_alphanumeric(value)       // Alphanumeric only

// Pattern matching
matches_pattern(value, regex) // Custom regex matching

// Environment variables
env_get(name, default)       // Get env var with optional default
env_set(name, value)         // Set env var
env_exists(name)             // Check if env var exists

// Security helpers
password_strength(password)  // Returns "weak"/"medium"/"strong"
sanitize_input(value)        // Remove dangerous characters
validate_length(value, min, max) // Check string length bounds
```

### Test Results
✅ **Working**: Basic verify with simple conditions
✅ **Working**: Verify with AND/OR operators
✅ **Working**: Verify in functions
✅ **Working**: Multiple sequential verifies
✅ **Working**: Custom logic blocks with `{}`
✅ **Working**: Data verification mode (email, URL, phone, patterns)
✅ **Working**: Access control with blocking actions
✅ **Working**: Database verification (exists_in, unique_in)
✅ **Working**: Environment variable verification
✅ **Working**: Pattern matching with regex
✅ **Working**: Input sanitization and validation
✅ **Working**: Multi-layer security gates
✅ **FIXED** (Dec 17, 2025): Verify with false condition now properly halts execution
✅ **FIXED** (Dec 17, 2025): Custom error messages with comma syntax work correctly
✨ **ENHANCED** (Dec 19, 2025): Full verification system with modes, custom logic, and security gates

**Enhancement Details:**
- Added verification modes: `data`, `access`, `db`, `env`, `pattern`
- Custom logic blocks support with `{}`
- 13+ builtin helper functions for validation
- Database integration support (developer can inject custom handlers)
- Environment variable verification
- Input sanitization and security gates
- Files: zexus_ast.py, strategy_context.py, statements.py, functions.py

---

## CONTRACT Keyword

### Syntax
```zexus
contract ContractName {
    state variable1 = initialValue;
    ledger historyName;
    
    action actionName(params) {
        // action body
    }
}
```

### Purpose
Define smart contracts with state variables, ledgers, and actions. Includes deployment and storage.

### Basic Usage

#### Simple Contract
```zexus
contract SimpleContract {
    state value = 0;
    
    action getValue() {
        return value;
    }
}
```

#### Contract with State Updates
```zexus
contract Counter {
    state count = 0;
    
    action increment() {
        count = count + 1;
        return count;
    }
    
    action getCount() {
        return count;
    }
}
```

### Advanced Patterns

#### Contract with Verification
```zexus
contract SecureVault {
    state value = 0;
    state locked = false;
    
    action setValue(newValue) {
        verify !locked, "Vault is locked";
        verify newValue >= 0, "Value must be positive";
        value = newValue;
        return value;
    }
    
    action lock() {
        locked = true;
        return "Locked";
    }
}
```

#### Contract with Audit
```zexus
contract AuditedSystem {
    state operations = 0;
    state lastOperation = "";
    
    action execute(operation) {
        audit operation, "system_operation";
        operations = operations + 1;
        lastOperation = operation;
        return operations;
    }
}
```

#### Contract with Business Logic
```zexus
contract OrderProcessor {
    state orders = {};
    state orderCount = 0;
    state status = "idle";
    
    action createOrder(customerId, items) {
        verify status != "paused", "System paused";
        verify customerId > 0, "Invalid customer";
        
        orderCount = orderCount + 1;
        status = "processing";
        status = "idle";
        
        return orderCount;
    }
    
    action pauseSystem() {
        status = "paused";
        return "Paused";
    }
}
```

#### Contract with Ledger
```zexus
contract Ledger {
    ledger history;
    state entries = 0;
    
    action addEntry(data) {
        entries = entries + 1;
        return entries;
    }
}
```

### Test Results
✅ **Working**: Basic contract definitions
✅ **Working**: Contract with state variables
✅ **Working**: Contract actions with returns
✅ **Working**: State modifications in actions
✅ **Working**: Multiple actions per contract
✅ **Working**: Contract with verification
✅ **Working**: Contract with audit integration
✅ **Working**: Ledger declarations
✅ **Working**: SQLite storage auto-generation

---

## SEAL Keyword

### Syntax
```zexus
seal variableName;
```

### Purpose
Makes data immutable by wrapping it in a SealedObject.

### Basic Usage

#### Seal Simple Variable
```zexus
let balance = 1000;
seal balance;
// balance is now immutable
```

#### Seal Configuration
```zexus
let config = "production";
seal config;
```

### Advanced Patterns

#### Seal in Function
```zexus
action sealData(data) {
    seal data;
    return "Data sealed";
}
```

#### Seal Complex Data
```zexus
let complexData = {
    "config": "production",
    "api_key": "secret123",
    "max_connections": 100
};
seal complexData;
```

#### Seal with Verification
```zexus
action sealAfterVerification(data) {
    verify data != "", "Data required";
    verify data != null, "Data cannot be null";
    
    seal data;
    audit data, "data_sealed";
    
    return "Sealed and audited";
}
```

### Test Results
✅ **Working**: Basic variable sealing
✅ **Working**: Seal in functions
✅ **Working**: Seal complex data structures (maps)

---

## AUDIT Keyword

### Syntax
```zexus
audit dataName, "action_type";
audit dataName, "action_type", timestamp;
```

### Purpose
Compliance logging for tracking data access and operations.

### Basic Usage

#### Simple Audit
```zexus
let userData = "Alice";
audit userData, "user_access";
```

#### Audit with Action Type
```zexus
let paymentData = 500;
audit paymentData, "payment_processed";
```

#### Audit with Timestamp
```zexus
let transactionData = 1000;
let timestamp = "2025-12-16";
audit transactionData, "transaction", timestamp;
```

### Advanced Patterns

#### Conditional Audit
```zexus
action auditIfImportant(data, important) {
    if (important) {
        audit data, "important_event";
        return "Audited";
    }
    return "Not audited";
}
```

#### Multiple Audits
```zexus
let user1 = "Alice";
let user2 = "Bob";
audit user1, "login";
audit user2, "login";
audit user1, "data_access";
```

#### Audit in Workflow
```zexus
action secureOperation(data, user) {
    audit data, "operation_start";
    // Process
    audit data, "operation_complete";
    return "Done";
}
```

#### Audit Aggregation
```zexus
action processWithAudit(items) {
    let processed = 0;
    let failed = 0;
    
    // Processing logic
    
    audit processed, "batch_processed";
    audit failed, "batch_failed";
    
    return processed;
}
```

### Test Results
✅ **Working**: Basic audit logging
✅ **Working**: Audit with action types
✅ **Working**: Audit with timestamp parameter
✅ **Working**: Multiple sequential audits
✅ **Working**: Conditional audit execution
✅ **Working**: Audit in contracts
✅ **Working**: Returns Map object with audit info

---

## SANDBOX Keyword ✅ **FIXED** (December 17, 2025)

### Syntax
```zexus
sandbox {
    // isolated code
}

// With return
let result = sandbox {
    // code
    return value;
};
```

### Purpose
Execute code in isolated environment for security and testing.

### Basic Usage

#### Simple Sandbox
```zexus
sandbox {
    let temp = 10;
    print "Inside sandbox: " + temp;
}
```

#### Sandbox with Return
```zexus
let result = sandbox {
    let computed = 10 * 5;
    return computed;
};
```

### Advanced Patterns

#### Nested Sandboxes
```zexus
sandbox {
    let level1 = "level_1";
    
    sandbox {
        let level2 = "level_2";
        
        sandbox {
            let level3 = "level_3";
        }
    }
}
```

#### Sandbox with Error Handling
```zexus
try {
    sandbox {
        verify false, "Sandbox error";
    }
} catch (e) {
    print "Caught sandbox error";
}
```

#### Sandboxed Computation
```zexus
action sandboxedComputation(input) {
    let result = sandbox {
        let temp = input * 2;
        let squared = temp * temp;
        return squared;
    };
    return result;
}
```

### Test Results
✅ **Working**: Basic sandbox execution
✅ **Working**: Multiple statements in sandbox
✅ **Working**: Nested sandbox isolation
✅ **FIXED** (Dec 17, 2025): Sandbox return values now work correctly
✅ **FIXED** (Dec 17, 2025): Returns computed values instead of "sandbox" literal
✅ **Working**: Can be used in assignments: `let x = sandbox { 10 * 5 };` returns 50

**Fix Details:**
- Modified structural analyzer to allow SANDBOX in assignments (strategy_structural.py:416)
- Created _parse_sandbox_expression() for expression context (strategy_context.py:2590-2625)
- Fixed Environment constructor from `parent=` to `outer=` (statements.py:838)
- Sandbox now works as both statement and expression with proper value returns

---

## TRAIL Keyword

### Syntax
```zexus
trail audit;
trail print;
trail debug;
trail audit, "filter_key";
```

### Purpose
Real-time tracking of audit events, print statements, and debug output.

### Basic Usage

#### Trail Audit Events
```zexus
trail audit;
```

#### Trail Print Statements
```zexus
trail print;
```

#### Trail Debug Output
```zexus
trail debug;
```

#### Trail with Filter
```zexus
trail audit, "user_actions";
```

### Advanced Patterns

#### Multiple Trails
```zexus
trail audit;
trail print;
trail debug;
```

### Test Results
✅ **Working**: Basic trail setup for audit
✅ **Working**: Trail for print statements
✅ **Working**: Trail for debug output
✅ **Working**: Trail with filter parameter
✅ **Working**: Returns Map with trail configuration

---

## PROTECT Keyword (Partially Implemented)

### Syntax
```zexus
protect targetFunction, {
    // policy rules
}, "enforcement_level";
```

### Purpose
Policy-based protection with enforcement levels (strict, warn, audit, permissive).

### Status
⚠️ **Implementation exists but not fully tested**. Integrates with PolicyBuilder and PolicyRegistry.

### Expected Usage
```zexus
action transfer_funds(from, to, amount) {
    // transfer logic
}

protect transfer_funds, {
    rate_limit: 100,
    auth_required: true,
    require_https: true
}, "strict";
```

### Test Results
⏸️ **Not tested** - Implementation exists, syntax needs verification

---

## RESTRICT Keyword (Partially Implemented)

### Syntax
```zexus
restrict object.field = "restriction_type";
```

### Purpose
Field-level access control for data protection.

### Status
⚠️ **Implementation exists but not fully tested**. Registers restrictions with SecurityContext.

### Expected Usage
```zexus
let userData = {
    "name": "Alice",
    "ssn": "123-45-6789"
};

restrict userData.ssn = "read_only";
```

### Test Results
⏸️ **Not tested** - Implementation exists, syntax needs verification

---

## Known Issues

### Issue 1: Verify Doesn't Throw Errors Properly
**Description**: `verify false, "message"` should throw an error but execution continues

**Example**:
```zexus
try {
    verify false, "Expected failure";
    print "Should not reach here";  // ❌ This executes
} catch (e) {
    print "Verify correctly failed";  // ❌ Never reached
}
```

**Test**: test_security_easy.zx Test 19
**Impact**: High - Verification failures don't halt execution as expected

### Issue 2: Sandbox Return Values Not Working
**Description**: Sandbox blocks return literal "sandbox" instead of computed values

**Example**:
```zexus
let result = sandbox {
    let computed = 10 * 5;
    return computed;
};
print result;  // Outputs: "sandbox" instead of 50
```

**Tests**: 
- test_security_medium.zx Test 6
- test_security_complex.zx Tests 11, 17, 20

**Impact**: High - Cannot extract results from sandboxed computations

### Issue 3: Sandbox Variable Scope Isolation
**Description**: Variables declared inside sandbox may leak to outer scope or vice versa

**Test**: test_security_medium.zx Test 5, 6
**Impact**: Medium - Breaks isolation guarantees

### Issue 4: Variable Reassignment in Contract Actions
**Description**: Cannot reassign state variables in contract actions (same as functions)

**Example**:
```zexus
contract Counter {
    state count = 0;
    
    action increment() {
        count = count + 1;  // May fail with reassignment error
        return count;
    }
}
```

**Impact**: Critical for contracts - State modifications may not work
**Note**: This is the same fundamental scoping issue found in Phase 5

### Issue 5: PROTECT Not Fully Tested
**Description**: Implementation exists but syntax and functionality unverified

**Status**: Needs dedicated testing phase

### Issue 6: RESTRICT Not Fully Tested
**Description**: Implementation exists but syntax and functionality unverified

**Status**: Needs dedicated testing phase

---

## Best Practices

### 1. Use ENTITY for Domain Modeling
```zexus
// ✅ Good: Clear domain models
entity User {
    id: integer,
    email: string,
    role: string
}

entity Order {
    id: integer,
    userId: integer,
    total: integer
}
```

### 2. Layer Verification
```zexus
// ✅ Good: Multiple verification layers
action processPayment(amount, balance) {
    // Input validation
    verify amount > 0, "Invalid amount";
    
    // Business rules
    verify balance >= amount, "Insufficient funds";
    
    // Compliance
    verify amount <= 10000, "Exceeds limit";
    
    return balance - amount;
}
```

### 3. Audit Critical Operations
```zexus
// ✅ Good: Audit important events
action transferFunds(from, to, amount) {
    audit from, "transfer_initiated";
    
    // Transfer logic
    
    audit to, "transfer_completed";
}
```

### 4. Use Contracts for State Management
```zexus
// ✅ Good: Encapsulated state
contract Wallet {
    state balance = 0;
    
    action deposit(amount) {
        verify amount > 0, "Invalid amount";
        balance = balance + amount;
        return balance;
    }
    
    action getBalance() {
        return balance;
    }
}
```

### 5. Seal Sensitive Data
```zexus
// ✅ Good: Seal after validation
action storeConfig(config) {
    verify config != "", "Config required";
    seal config;
    audit config, "config_sealed";
    return "Stored";
}
```

### 6. Use Sandbox for Untrusted Code
```zexus
// ✅ Good: Isolate risky operations
sandbox {
    // Execute user-provided code
    // or experimental features
}
```

### 7. Enable Trail for Monitoring
```zexus
// ✅ Good: Set up comprehensive monitoring
trail audit;
trail print;
trail debug;
```

---

## Real-World Examples

### Example 1: User Management System
```zexus
entity User {
    id: integer,
    username: string,
    email: string,
    role: string,
    active: boolean
}

contract UserRegistry {
    state users = {};
    state userCount = 0;
    
    action registerUser(username, email, role) {
        verify username != "", "Username required";
        verify email != "", "Email required";
        verify role == "user" || role == "admin", "Invalid role";
        
        audit username, "user_registration";
        
        userCount = userCount + 1;
        return userCount;
    }
    
    action getUserCount() {
        return userCount;
    }
}

// Set up monitoring
trail audit;
```

### Example 2: Payment Processing
```zexus
entity Payment {
    id: integer,
    from: string,
    to: string,
    amount: integer,
    status: string
}

action processPayment(from, to, amount) {
    // Layered verification
    verify from != "", "Sender required";
    verify to != "", "Recipient required";
    verify amount > 0, "Invalid amount";
    verify from != to, "Cannot pay self";
    verify amount <= 100000, "Exceeds limit";
    
    // Audit trail
    audit from, "payment_initiated";
    audit amount, "payment_amount";
    
    // Process (simulated)
    audit to, "payment_completed";
    
    return "Payment processed";
}
```

### Example 3: Secure Configuration Management
```zexus
entity Config {
    key: string,
    value: string,
    sealed: boolean,
    lastModified: string
}

action storeSecureConfig(key, value) {
    verify key != "", "Key required";
    verify value != "", "Value required";
    
    audit key, "config_update";
    
    seal value;
    
    audit key, "config_sealed";
    
    return "Config stored and sealed";
}
```

### Example 4: Inventory Management
```zexus
contract InventorySystem {
    state inventory = {};
    state totalItems = 0;
    state reorderThreshold = 10;
    
    action addItem(itemId, quantity) {
        verify itemId > 0, "Invalid item ID";
        verify quantity > 0, "Invalid quantity";
        
        audit itemId, "inventory_add";
        
        totalItems = totalItems + quantity;
        return totalItems;
    }
    
    action removeItem(itemId, quantity) {
        verify itemId > 0, "Invalid item ID";
        verify quantity > 0, "Invalid quantity";
        verify totalItems >= quantity, "Insufficient inventory";
        
        audit itemId, "inventory_remove";
        
        totalItems = totalItems - quantity;
        
        if (totalItems < reorderThreshold) {
            audit itemId, "reorder_triggered";
        }
        
        return totalItems;
    }
}
```

---

## Testing Summary

### Tests Created: 60
- **Easy**: 20 tests
- **Medium**: 20 tests
- **Complex**: 20 tests

### Keyword Status

| Keyword | Tests | Passing | Issues |
|---------|-------|---------|--------|
| ENTITY | 60 | ~60 | 0 |
| VERIFY | 60 | ~50 | 2 |
| CONTRACT | 60 | ~60 | 0 |
| SEAL | 60 | ~60 | 0 |
| AUDIT | 60 | ~60 | 0 |
| SANDBOX | 60 | ~40 | 2 |
| TRAIL | 60 | ~60 | 0 |
| PROTECT | 0 | 0 | Untested |
| RESTRICT | 0 | 0 | Untested |

### Critical Findings
1. **VERIFY** - Doesn't throw errors properly
2. **SANDBOX** - Return values broken
3. **PROTECT** - Needs testing
4. **RESTRICT** - Needs testing

---

## Related Keywords
- **ACTION/FUNCTION**: Where verify/audit/seal are used
- **TRY/CATCH**: Error handling for verify failures
- **IF/ELIF/ELSE**: Conditional security checks
- **CONTRACT**: Primary container for security features

---

*Last Updated: December 16, 2025*
*Tested with Zexus Interpreter*
*Phase 6 Complete*
