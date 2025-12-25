# audit - Log All Data Access for Compliance

## Overview

The `audit` command logs all data access and modifications for compliance tracking and security auditing. This is essential for systems that must track and report data access patterns, modifications, and security-sensitive operations.

**Status**: ✅ Implemented in v1.1.0

## Syntax

```zexus
audit data_name, "action_type";
audit data_name, "action_type", timestamp;
```

## Parameters

- **data_name**: Identifier for the variable/data to audit (must exist in current scope)
- **action_type**: String literal describing the action ("access", "modification", "deletion", "view", etc.)
- **timestamp**: Optional timestamp or datetime expression (uses current time if omitted)

## Return Value

Returns an audit log entry as a `Map` object containing:
- `data_name`: String - the name of the audited data
- `action`: String - the action type performed
- `timestamp`: String - ISO format timestamp of the audit
- `data_type`: String - the type of data being audited

## Behavior

- Creates a compliance log entry without modifying the audited data
- Can be used with any variable type (primitives, objects, arrays, functions)
- Works transparently with the environment's audit trail
- Returns a map object that can be stored, printed, or analyzed
- Timestamps are ISO 8601 formatted (automatic if not provided)

## Examples

### Example 1: Basic Data Access Audit

```zexus
let user_data = {
  id: 123,
  name: "Alice",
  email: "alice@example.com"
};

// Log the access to user_data
audit user_data, "access";
// Returns: {data_name: "user_data", action: "access", timestamp: "2025-01-15T10:30:45.123456", data_type: "MAP"}

print(audit user_data, "access");
// Output: {data_name: "user_data", action: "access", timestamp: "2025-01-15T10:30:45.123456", data_type: "MAP"}
```

### Example 2: Audit with Custom Timestamp

```zexus
let CONFIG = {
  api_key: "sk_live_abc123",
  environment: "production",
  debug: false
};

let audit_time = "2025-01-15T14:22:30Z";

audit CONFIG, "modification", audit_time;
// Returns audit entry with the specified timestamp
```

### Example 3: Audit Sensitive Data Access

```zexus
let password_hash = "bcrypt$2y$12$abcdef123456";
let api_token = "ghp_xyz789";
let ssn = "123-45-6789";

audit password_hash, "sensitive_access";
audit api_token, "sensitive_access";
audit ssn, "deletion";

print("All sensitive data accesses logged for compliance.");
```

### Example 4: Audit Function and Array Operations

```zexus
let calculate_total = fn(items) {
  let sum = 0;
  for each item in items {
    sum = sum + item;
  }
  return sum;
};

audit calculate_total, "function_called";
// Returns: {data_name: "calculate_total", action: "function_called", ..., data_type: "FUNCTION"}

let transaction_list = [100, 250, 75, 500];

audit transaction_list, "array_access";
// Returns: {data_name: "transaction_list", action: "array_access", ..., data_type: "ARRAY"}
```

### Example 5: Building an Audit Trail

```zexus
let audit_trail = [];
let sensitive_config = {
  db_password: "secret123",
  api_url: "https://api.secure.com"
};

// Simulate multiple accesses
audit sensitive_config, "view";
audit sensitive_config, "modification";
audit sensitive_config, "deletion";

// In a real system, each audit entry would be appended to audit_trail
// for later analysis and compliance reporting
```

### Example 6: Conditional Auditing

```zexus
let user_role = "admin";
let critical_resource = {value: 999, owner: "system"};

if user_role == "admin" {
  let audit_log = audit critical_resource, "admin_access";
  print("Admin access logged: ", audit_log);
} else {
  let audit_log = audit critical_resource, "unauthorized_attempt";
  print("Unauthorized access attempt logged.");
}
```

### Example 7: Audit in Try-Catch (Error Tracking)

```zexus
let protected_data = {secret: "classified"};

try {
  audit protected_data, "access_attempt";
  // Process data...
} catch error {
  audit protected_data, "access_failed";
  print("Error during data access: ", error);
}
```

## Use Cases

### 1. Compliance Reporting (GDPR, HIPAA, SOC2)

Track all accesses to personally identifiable information (PII) and protected health information (PHI) for regulatory compliance.

```zexus
let patient_record = {ssn: "123-45-6789", medical_history: [...]};
audit patient_record, "medical_record_access";  // HIPAA compliance
```

### 2. Security Incident Response

Log suspicious data access patterns to identify potential breaches or unauthorized access attempts.

```zexus
let database_credentials = {host: "db.secure.com", user: "admin", pass: "***"};
audit database_credentials, "suspicious_access";
```

### 3. Data Lifecycle Management

Track when data is created, modified, and deleted for data governance.

```zexus
let customer_data = {...};
audit customer_data, "created";
audit customer_data, "modified";
audit customer_data, "deleted";
```

### 4. API Request Auditing

Log API calls and sensitive parameter usage.

```zexus
let api_request = {endpoint: "/api/users", method: "POST", body: {...}};
audit api_request, "api_call";
```

### 5. Financial Transaction Tracking

Monitor all financial operations for fraud detection and reconciliation.

```zexus
let transaction = {amount: 5000, from_account: "ACC_001", to_account: "ACC_002"};
audit transaction, "fund_transfer";
```

## Integration with Other Features

### With const (Immutable Audit Records)

```zexus
const AUDIT_ACTIONS = ["access", "modification", "deletion"];
let data = {value: 42};

audit data, AUDIT_ACTIONS[0];  // Audit using const action list
```

### With Try-Catch Error Handling

```zexus
let sensitive_resource = {...};

try {
  audit sensitive_resource, "access";
  // Process resource
} catch err {
  audit sensitive_resource, "access_error";
}
```

### With Maps and Structured Data

```zexus
let audit_log_entry = audit user_credentials, "authentication_attempt";

print(audit_log_entry.data_name);   // "user_credentials"
print(audit_log_entry.action);      // "authentication_attempt"
print(audit_log_entry.timestamp);   // ISO 8601 timestamp
```

## Performance Considerations

- **No-Copy Logging**: Audit entries reference the original data, not copies
- **Minimal Overhead**: Audit logging adds negligible performance impact
- **Memory Efficient**: Audit entries are lightweight map objects
- **Timestamp Resolution**: Automatic timestamps are millisecond-precise

## Best Practices

1. **Consistent Action Names**: Use standardized action types across your codebase
   ```zexus
   // ✅ Good
   audit data, "access";
   audit data, "modification";
   
   // ❌ Avoid
   audit data, "read";
   audit data, "update";
   ```

2. **Audit at Critical Points**: Log access at entry and exit points of sensitive functions
   ```zexus
   fn process_payment(payment_data) {
     audit payment_data, "payment_processing_start";
     // ... process payment ...
     audit payment_data, "payment_processing_complete";
     return result;
   }
   ```

3. **Include Context Timestamps**: Use custom timestamps for distributed systems
   ```zexus
   audit transaction, "initiated", request_timestamp;
   ```

4. **Store Audit Entries**: Build persistent audit trails
   ```zexus
   let audit_entries = [];
   audit_entries = [audit_entries, audit data, "access"];
   ```

5. **Don't Log in Loops**: Audit before/after loop iterations, not inside
   ```zexus
   // ✅ Efficient
   audit data, "batch_access";
   for each item in items {
     // Process
   }
   
   // ❌ Inefficient (creates too many entries)
   for each item in items {
     audit item, "access";
   }
   ```

## Error Handling

- **Non-existent Data**: Returns error if identifier not found
  ```zexus
  audit undefined_variable, "access";  // Error: identifier 'undefined_variable' not found
  ```

- **Invalid Action Type**: Must be a string literal or expression evaluating to string
  ```zexus
  audit data, 123;  // Error: expected string for action_type
  ```

## Compatibility

| Feature | Compatible |
|---------|-----------|
| `const` variables | ✅ Yes |
| `let` variables | ✅ Yes |
| Functions | ✅ Yes |
| Objects/Maps | ✅ Yes |
| Arrays/Lists | ✅ Yes |
| Sealed objects | ✅ Yes |
| Try-catch blocks | ✅ Yes |
| Custom timestamps | ✅ Yes |

## Advanced Examples

### Multi-Level Data Audit

```zexus
let user = {
  id: 1,
  credentials: {
    password: "hashed_pwd",
    2fa_enabled: true
  },
  sessions: [...]
};

audit user, "user_loaded";
audit user.credentials, "credentials_accessed";
audit user.sessions, "sessions_accessed";
```

### Audit Middleware Pattern

```zexus
fn audit_and_process(data, action) {
  let audit_entry = audit data, action;
  // Do something with audit_entry
  return audit_entry;
}

let result = audit_and_process(my_data, "transform");
```

## Related Commands

- `seal` - Mark data as immutable for additional security
- `const` - Declare immutable variables for configuration
- `try-catch` - Handle errors in audit processing
- `protect` - Apply protection rules alongside auditing

## Changelog

### v1.1.0
- Initial implementation of `audit` command
- Support for custom timestamps
- Automatic ISO 8601 formatting
- Integration with all data types

## Migration Guide

If you have existing logging code, migrate to `audit`:

```zexus
// Old way
print("Accessed user: ", user_id);

// New way with audit (compliance-compliant)
audit user_record, "access";
print("User access logged");
```

