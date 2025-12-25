# Zexus Security Features Implementation

## Overview

This document describes the newly implemented core security features for the Zexus interpreter, enhancing it with robust access control, data validation, and functional purity enforcement.

## Features Implemented

### 1. Capability-Based Security (CAPABILITY / GRANT / REVOKE)

**Purpose:** Implement a formal permission system that defines what entities (users, modules, functions) can do, shifting from Access Control Lists (ACLs) to Capability-Based Security.

**Key Components:**

- **`CapabilityManager`** - Central authority for capability grants/checks
  - Manages capabilities and their levels (DENY, RESTRICTED, ALLOWED, UNRESTRICTED)
  - Tracks granted capabilities per entity
  - Maintains audit log of all capability checks
  - Supports multiple capability policies (AllowAll, DenyAll, Selective)

- **Base Capabilities** - Always available operations:
  - `core.language` - Core language features
  - `core.control` - Control flow (if, while, for)
  - `core.math` - Math operations
  - `core.strings` - String operations
  - `core.arrays` - Array operations
  - `core.objects` - Object operations

- **Privileged Capabilities** - Require explicit grant:
  - `io.read` / `io.write` / `io.delete` - File operations
  - `network.tcp` / `network.http` - Network access
  - `crypto.keygen` / `crypto.sign` - Cryptographic operations
  - `exec.shell` / `exec.spawn` - Process execution
  - `sys.env` / `sys.time` / `sys.exit` - System access

**Language Syntax:**

```zexus
// Define a capability
capability read_file = {
    description: "Read file system",
    scope: "io",
    level: "restricted"
};

// Grant capabilities to an entity
grant user1 {
    read_file,
    read_network,
    crypto.sign
};

// Revoke capabilities from an entity
revoke untrusted_plugin {
    write_file,
    exec.shell
};
```

**Use Cases:**

- Sandbox untrusted plugins with minimal capabilities
- Enforce role-based access control
- Audit security-sensitive operations
- Multi-tenant isolation

---

### 2. Pure Function Enforcement (PURE / IMMUTABLE)

**Purpose:** Enforce referential transparency and side-effect detection to enable:
- Easier reasoning about code
- Better optimizations (caching, parallelization)
- Security analysis and distributed execution

**Key Components:**

- **`PurityAnalyzer`** - Analyzes functions for purity
  - Detects global variable access
  - Identifies I/O operations
  - Flags non-deterministic operations (random, time)
  - Tracks external function calls
  - Detects parameter mutations

- **`PurityEnforcer`** - Enforces purity constraints
  - Validates pure functions don't call impure functions
  - Prevents side effects in pure execution
  - Maintains call trace for verification

- **`Immutability`** - Tracks and enforces immutable data
  - Prevents modification after creation
  - Supports deep immutability
  - Works with recursive structures

**Language Syntax:**

```zexus
// Pure function declaration
pure function add(a, b) {
    return a + b;
}

// Immutable variable
immutable const PI = 3.14159;
immutable let config = { /* ... */ };

// Function with immutable parameters
function process(immutable data) {
    // data cannot be modified
}
```

**Purity Levels:**

- **PURE** - No side effects, fully deterministic
- **RESTRICTED** - Side effects only on local state
- **IMPURE** - Has external side effects
- **UNKNOWN** - Purity not yet determined

---

### 3. Data Validation & Sanitization (VALIDATE / SANITIZE)

**Purpose:** Provide high-level primitives for:
- Type and format validation
- Schema compliance checking
- Input sanitization for injection prevention

**Key Components:**

- **`ValidationSchema`** - Define validation rules
  - Type checking
  - Custom validators
  - Composite validators (AND logic)
  - Field-level validation

- **`StandardValidators`** - Built-in common validators:
  - `EMAIL` - RFC-compliant email format
  - `URL` - HTTP/HTTPS URLs
  - `IPV4` / `IPV6` - IP addresses
  - `PHONE` - Phone numbers
  - `UUID` - UUID format
  - `ALPHANUMERIC` - Alphanumeric characters only
  - `POSITIVE_INT` - Non-negative integers
  - `NON_EMPTY_STRING` - Non-empty strings

- **`Sanitizer`** - Clean untrusted input for different contexts:
  - **HTML** - Entity encoding for web display
  - **URL** - Percent encoding for query parameters
  - **SQL** - Quote escaping for database queries
  - **JAVASCRIPT** - Character escaping for JSON/JS
  - **CSV** - Field quoting and escaping

- **`ValidationManager`** - Central validation orchestration
  - Register custom validators
  - Register schemas
  - Track validation/sanitization history
  - Audit trail

**Language Syntax:**

```zexus
// Validate single value
validate user_input, email;

// Validate against schema
validate user_input, {
    name: string,
    email: email,
    age: number(18, 120),
    phone: phone
};

// Sanitize for specific context
let html_safe = sanitize(user_input, {
    encoding: "html",
    rules: ["remove_scripts"]
});

// Sanitize for SQL
let sql_safe = sanitize(user_input, {
    encoding: "sql"
});
```

---

## Implementation Details

### File Structure

```
src/zexus/
├── zexus_token.py              # New tokens: CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE
├── zexus_ast.py                # New AST nodes for security statements
├── capability_system.py        # Capability manager and policies (extended)
├── purity_system.py            # Pure function enforcement (new)
├── validation_system.py        # Validation & sanitization (new)
├── parser/
│   └── strategy_context.py    # Parser handlers for new statements
└── evaluator/
    ├── core.py                # Dispatch for new statement types
    └── statements.py          # Evaluation handlers for new statements
```

### New Tokens

```python
CAPABILITY = "CAPABILITY"      # Define capabilities
GRANT = "GRANT"                # Grant capabilities
REVOKE = "REVOKE"              # Revoke capabilities
IMMUTABLE = "IMMUTABLE"        # Enforce immutability
VALIDATE = "VALIDATE"          # Validate data
SANITIZE = "SANITIZE"          # Sanitize input
```

### New AST Statements

```python
class CapabilityStatement(Statement)     # capability name = definition;
class GrantStatement(Statement)          # grant entity { capabilities };
class RevokeStatement(Statement)         # revoke entity { capabilities };
class ValidateStatement(Statement)       # validate value, schema;
class SanitizeStatement(Statement)       # sanitize value, rules;
class ImmutableStatement(Statement)      # immutable variable = value;
```

### Evaluator Handlers

All new statements have evaluator handlers in `src/zexus/evaluator/statements.py`:

- `eval_capability_statement()` - Define and store capabilities
- `eval_grant_statement()` - Grant capabilities to entities
- `eval_revoke_statement()` - Revoke capabilities
- `eval_validate_statement()` - Validate data against schema
- `eval_sanitize_statement()` - Sanitize untrusted input
- `eval_immutable_statement()` - Mark variables as immutable

---

## Usage Examples

### Example 1: Sandboxed Plugin Execution

```zexus
// Define security boundaries
capability io.read = { description: "Read files" };
capability io.write = { description: "Write files" };

// Load and isolate untrusted plugin
action run_untrusted_plugin(plugin_path) {
    // Validate path
    validate plugin_path, alphanumeric;
    
    // Load plugin
    let plugin = load_module(plugin_path);
    
    // Grant only read access
    grant plugin {
        core.language,
        core.math,
        io.read
    };
    
    // Deny dangerous operations
    revoke plugin {
        io.write,
        network.tcp,
        exec.shell
    };
    
    // Execute in sandbox
    sandbox {
        plugin.main();
    }
}
```

### Example 2: Secure Form Processing

```zexus
action handle_user_registration() {
    let form_data = get_request_body();
    
    // Step 1: Validate structure
    validate form_data, {
        username: alphanumeric,
        email: email,
        password: string(8, 128),
        age: number(18, 120)
    };
    
    // Step 2: Sanitize for database
    let safe_data = {
        username: sanitize(form_data.username, encoding: "sql"),
        email: sanitize(form_data.email, encoding: "sql"),
        age: form_data.age
    };
    
    // Step 3: Store user
    store_user(safe_data);
    
    // Step 4: Sanitize for HTML response
    let response = {
        username: sanitize(form_data.username, encoding: "html"),
        email: sanitize(form_data.email, encoding: "html")
    };
    
    return response;
}
```

### Example 3: Pure Computation Pipeline

```zexus
// Pure mathematical function
pure function calculate_tax(amount, rate) {
    return amount * rate;
}

// Pure data transformation
pure function process_orders(orders) {
    // Validate input
    validate orders, array;
    
    // Transform each order
    return [
        {
            id: order.id,
            subtotal: order.price * order.quantity,
            tax: calculate_tax(order.price * order.quantity, 0.08),
            total: order.price * order.quantity * 1.08
        }
        for order in orders
    ];
}

// Immutable configuration
immutable let TAX_RATE = 0.08;
immutable let DISCOUNT_RULES = {
    BULK: 0.10,
    LOYALTY: 0.05,
    SEASONAL: 0.15
};
```

---

## Testing

Comprehensive test suite in `test_security_features.py`:

### Test Coverage

- **Capability System**
  - Grant/check capabilities
  - Selective and deny-all policies
  - Audit logging
  - Base capability availability
  - Required capability validation

- **Purity System**
  - Pure function detection
  - Impure operation detection (I/O, globals, exceptions)
  - Immutability marking
  - Nested structure handling

- **Validation System**
  - Standard validators (email, URL, IPv4, phone, UUID)
  - Range validation
  - Schema validation
  - Missing field detection
  - Custom validator creation

- **Sanitization System**
  - HTML sanitization
  - URL encoding
  - SQL escaping
  - JavaScript escaping
  - CSV escaping
  - Dictionary and nested structure sanitization

- **Integration Tests**
  - Capability with validation
  - Purity with immutability
  - Sanitize-then-validate workflow

### Running Tests

```bash
pytest test_security_features.py -v
```

---

## Security Considerations

### Defense in Depth

The security features should be used in combination:

1. **Capabilities** - Limit what code can access
2. **Validation** - Check input structure before processing
3. **Sanitization** - Clean untrusted data for output context
4. **Pure Functions** - Simplify security analysis

### Best Practices

1. **Default Deny** - Start with DenyAllPolicy, explicitly grant needed capabilities
2. **Validate Early** - Validate external input immediately upon receipt
3. **Sanitize Late** - Sanitize just before output in the relevant context
4. **Audit Everything** - Enable and monitor capability and validation logs
5. **Immutability** - Use immutable for configuration and sensitive data
6. **Composability** - Combine pure functions for safety and performance

### Threat Models Addressed

- **Privilege Escalation** - Capabilities prevent unauthorized access
- **Injection Attacks** - Validation and sanitization prevent injection
- **Data Integrity** - Immutability prevents unauthorized modification
- **Side Effects** - Pure function enforcement enables safe parallelization
- **Unauthorized Modification** - Capabilities + audit logs track changes

---

## Migration Path

Existing Zexus code can be incrementally secured:

### Phase 1: Add Capabilities
- Define capabilities for sensitive operations
- Audit current capability usage
- Identify untrusted code paths

### Phase 2: Add Validation
- Identify external input sources
- Add validation schemas
- Log validation failures

### Phase 3: Add Sanitization
- Map output contexts (HTML, SQL, URL, JS)
- Add appropriate sanitization
- Test with security scanners

### Phase 4: Enforce Purity
- Mark mathematical functions as pure
- Use immutability for config
- Enable purity enforcement

---

## Future Enhancements

### Potential Additions

1. **Capability Delegation** - Allow capabilities to be temporarily delegated
2. **Time-Limited Capabilities** - Capabilities that expire after duration
3. **Rate Limiting** - Built-in rate limiting for sensitive operations
4. **Cryptographic Verification** - Verify capability tokens cryptographically
5. **Behavioral Analysis** - Detect anomalous capability usage
6. **Machine Learning** - Anomaly detection for security events
7. **Formal Verification** - Prove properties about pure functions
8. **Information Flow** - Track information flow through program

### Integration Points

- Interact with container/sandbox technologies
- Integrate with OS-level capabilities
- Connect to security monitoring systems
- Support for RBAC frameworks
- Integration with cryptographic libraries

---

## Performance Impact

- **Capability Checks** - O(1) hash table lookup
- **Validation** - Depends on schema complexity, typically O(n)
- **Sanitization** - Linear in input size
- **Purity Analysis** - One-time at function definition
- **Immutability Tracking** - O(1) memory overhead per object

### Optimization Tips

- Cache validation results for repeated patterns
- Batch capability checks
- Use selective policies instead of allow-all
- Profile sanitization bottlenecks

---

## References

### Capability-Based Security
- Object Capabilities (wikipedia)
- CapTP Protocol
- E Programming Language

### Data Validation
- OWASP Input Validation Cheat Sheet
- CWE-20: Improper Input Validation

### Injection Prevention
- OWASP SQL Injection Prevention
- OWASP XSS Prevention
- CWE-94: Command Injection

### Functional Purity
- Referential Transparency
- Pure Functions in Haskell
- Functional Programming principles

---

## Support & Documentation

- See `SECURITY_FEATURES_GUIDE.zx` for detailed examples
- See `test_security_features.py` for test patterns
- See source code comments for implementation details

## Summary

The new security features provide Zexus with:
- ✅ Fine-grained access control via capabilities
- ✅ Referential transparency via pure functions
- ✅ Input validation against schemas
- ✅ Context-aware output sanitization
- ✅ Comprehensive audit trails
- ✅ Defense-in-depth security model

These features enable building secure, analyzable, and maintainable Zexus applications.
