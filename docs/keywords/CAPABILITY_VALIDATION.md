# Capability & Validation Keywords Documentation

## Overview
Zexus provides capability-based security and data validation features through CAPABILITY, GRANT, REVOKE, IMMUTABLE, VALIDATE, and SANITIZE keywords. These enable fine-grained access control, immutability guarantees, and input sanitization.

### Keywords Covered
- **CAPABILITY**: Define security capabilities
- **GRANT**: Grant capabilities to entities
- **REVOKE**: Revoke capabilities from entities
- **IMMUTABLE**: Create immutable variables
- **VALIDATE**: Data validation against schemas
- **SANITIZE**: Sanitize untrusted input

---

## Implementation Status

| Keyword | Lexer | Parser | Evaluator | Status |
|---------|-------|--------|-----------|--------|
| CAPABILITY | ✅ | ✅ | ✅ | 🟢 Working |
| GRANT | ✅ | ✅ | ✅ | 🟢 Working |
| REVOKE | ✅ | ✅ | ✅ | 🟢 Working |
| IMMUTABLE | ✅ | ✅ | ✅ | 🟢 Working |
| VALIDATE | ✅ | ✅ | ✅ | 🟢 Working - FIXED ✅ |
| SANITIZE | ✅ | ✅ | ✅ | 🟢 Working - FIXED ✅ |

---

## CAPABILITY Keyword

### Syntax
```zexus
capability capabilityName {
    "description": "Description text",
    "scope": "scope_name"
};
```

### Purpose
Define security capabilities for capability-based access control systems.

### Basic Usage

#### Simple Capability
```zexus
capability read_data {
    "description": "Read data access",
    "scope": "data"
};
```

#### Multiple Capabilities
```zexus
capability read_files {
    "description": "Read file access",
    "scope": "files"
};
capability write_files {
    "description": "Write file access",
    "scope": "files"
};
capability execute_files {
    "description": "Execute file access",
    "scope": "files"
};
```

### Advanced Patterns

#### Role-Based Capabilities
```zexus
capability viewer_access {
    "description": "Viewer role",
    "scope": "roles"
};
capability editor_access {
    "description": "Editor role",
    "scope": "roles"
};
capability owner_access {
    "description": "Owner role",
    "scope": "roles"
};
```

#### Resource-Specific Capabilities
```zexus
capability read_users {
    "description": "Read users",
    "scope": "users"
};
capability read_products {
    "description": "Read products",
    "scope": "products"
};
capability read_orders {
    "description": "Read orders",
    "scope": "orders"
};
```

#### Hierarchical Capabilities
```zexus
capability tier1_read {
    "description": "Tier 1 read access",
    "scope": "tier1"
};
capability tier2_write {
    "description": "Tier 2 write access",
    "scope": "tier2"
};
capability tier3_admin {
    "description": "Tier 3 admin access",
    "scope": "tier3"
};
```

### Test Results
✅ **Working**: Basic capability definitions
✅ **Working**: Multiple capabilities
✅ **Working**: Nested scope definitions
✅ **Working**: Capability objects created and stored

---

## GRANT Keyword

### Syntax
```zexus
grant "entityName" capabilityName;
grant "entityName" cap1, cap2, cap3;
```

### Purpose
Grant one or more capabilities to an entity (user, role, or resource).

### Basic Usage

#### Grant Single Capability
```zexus
capability read_data {
    "description": "Read data",
    "scope": "data"
};
grant "user1" read_data;
```

#### Grant Multiple Capabilities
```zexus
grant "admin" read_data, write_data, delete_data;
```

### Advanced Patterns

#### Role-Based Grants
```zexus
grant "viewer" viewer_access;
grant "editor" viewer_access, editor_access;
grant "owner" viewer_access, editor_access, owner_access;
```

#### Hierarchical Access Control
```zexus
grant "basic_user" tier1_read;
grant "power_user" tier1_read, tier2_read, tier2_write;
grant "administrator" tier1_read, tier2_read, tier2_write, tier3_admin;
```

#### Dynamic Grant Management
```zexus
action grantUserAccess(userId, accessLevel) {
    if (accessLevel == "basic") {
        grant userId read_access;
    } elif (accessLevel == "advanced") {
        grant userId read_access, write_access;
    }
}
```

### Test Results
✅ **Working**: Single capability grants
✅ **Working**: Multiple capability grants
✅ **Working**: Dynamic grant operations
✅ **Working**: Returns confirmation message

---

## REVOKE Keyword

### Syntax
```zexus
revoke "entityName" capabilityName;
revoke "entityName" cap1, cap2, cap3;
```

### Purpose
Revoke one or more capabilities from an entity.

### Basic Usage

#### Revoke Single Capability
```zexus
revoke "user1" read_data;
```

#### Revoke Multiple Capabilities
```zexus
revoke "user2" read_data, write_data;
```

### Advanced Patterns

#### Session Management
```zexus
capability session_access {
    "description": "Session access",
    "scope": "session"
};
// Login
grant "user_session_123" session_access;
// Logout
revoke "user_session_123" session_access;
```

#### Temporary Access
```zexus
capability temp_access {
    "description": "Temporary access",
    "scope": "temporary"
};
grant "guest" temp_access;
// After timeout
revoke "guest" temp_access;
```

#### Bulk Revocation
```zexus
revoke "bulk_user" cap1, cap2, cap3, cap4, cap5;
```

### Test Results
✅ **Working**: Single capability revocation
✅ **Working**: Multiple capability revocation
✅ **Working**: Dynamic revocation
✅ **Working**: Returns confirmation message

---

## IMMUTABLE Keyword

### Syntax
```zexus
// Create immutable with value
immutable variableName = value;

// Mark existing variable immutable
immutable existingVariable;
```

### Purpose
Create immutable variables that cannot be modified after creation.

### Basic Usage

#### Immutable Primitives
```zexus
immutable config = "production";
immutable maxUsers = 100;
immutable isProduction = true;
```

#### Immutable Collections
```zexus
immutable allowedHosts = ["localhost", "127.0.0.1", "example.com"];
```

#### Immutable Maps
```zexus
immutable appConfig = {
    "name": "MyApp",
    "version": "1.0.0",
    "env": "production"
};
```

### Advanced Patterns

#### Configuration Management
```zexus
immutable dbHost = "localhost";
immutable dbPort = 5432;
immutable dbName = "production_db";
immutable maxConnections = 100;
```

#### Security Policies
```zexus
immutable securityPolicy = {
    "minPasswordLength": 12,
    "requireSpecialChars": true,
    "maxLoginAttempts": 3,
    "sessionTimeout": 1800,
    "enableMFA": true
};
```

#### Environment Configuration
```zexus
immutable environment = "production";
immutable debugMode = false;
immutable logLevel = "ERROR";
immutable apiEndpoints = {
    "auth": "https://auth.example.com/api",
    "data": "https://data.example.com/api"
};
```

#### Cryptographic Configuration
```zexus
immutable encryptionAlgorithm = "AES-256-GCM";
immutable hashAlgorithm = "SHA-256";
immutable keyDerivationFunction = "PBKDF2";
immutable iterations = 100000;
immutable saltLength = 32;
```

#### Nested Immutable Structures
```zexus
immutable applicationConfig = {
    "app": {
        "name": "SecureApp",
        "version": "2.0.0"
    },
    "database": {
        "primary": {
            "host": "db-primary.internal",
            "port": 5432,
            "ssl": true
        },
        "replica": {
            "host": "db-replica.internal",
            "port": 5432
        }
    }
};
```

### Test Results
✅ **Working**: Immutable primitive types
✅ **Working**: Immutable collections
✅ **Working**: Immutable maps
✅ **Working**: Nested immutable structures
✅ **Working**: Mark existing variable immutable

---

## VALIDATE Keyword

### Syntax
```zexus
validate data, "schemaName";
validate data, schemaObject;
```

### Purpose
Validate data against predefined schemas or validation rules.

### Basic Usage

#### String Validation
```zexus
let data = "test_data";
validate data, "string";
```

#### Email Validation
```zexus
let email = "user@example.com";
validate data, "email";
```

### Advanced Patterns

#### Schema Validation (Intended)
```zexus
let userData = {
    "name": "Alice",
    "age": 25,
    "email": "alice@example.com"
};
validate userData, {
    "name": "string",
    "age": "integer",
    "email": "email"
};
```

### Test Results
✅ **Status**: Fixed (Dec 17, 2025)
- `validate "hello", "string"` - Works correctly ✅
- `validate 42, "integer"` - Works correctly ✅
- `validate "user@example.com", "email"` - Works correctly ✅
- Built-in schemas registered: string, integer, number, boolean, email, url, phone, uuid, ipv4, ipv6
- Schema registry properly initialized on startup

---

## SANITIZE Keyword

### Syntax
```zexus
let sanitized = sanitize data;
let sanitized = sanitize data, "encoding";
```

### Purpose
Sanitize untrusted input to prevent injection attacks.

### Basic Usage

#### HTML Sanitization
```zexus
let userInput = "<script>alert('xss')</script>";
let sanitized = sanitize userInput;
```

#### Encoding-Specific Sanitization
```zexus
let htmlInput = "<div>Hello</div>";
let sanitizedHtml = sanitize htmlInput, "html";
```

### Advanced Patterns

#### SQL Injection Prevention
```zexus
let sqlInput = "'; DROP TABLE users; --";
let sanitizedSql = sanitize sqlInput, "sql";
```

#### XSS Prevention
```zexus
let xssPayload = "<svg onload=alert('xss')>";
let sanitizedXss = sanitize xssPayload, "html";
```

#### Path Traversal Prevention
```zexus
let pathTraversal = "../../../etc/passwd";
let sanitizedPath = sanitize pathTraversal, "path";
```

#### Command Injection Prevention
```zexus
let commandInjection = "; rm -rf /";
let sanitizedCmd = sanitize commandInjection, "shell";
```

#### Multi-Layer Sanitization
```zexus
let maliciousInput = "<script>alert(1)</script><iframe>";
let stage1 = sanitize maliciousInput, "html";
let stage2 = sanitize stage1, "html";
let stage3 = sanitize stage2, "html";
```

#### Context-Aware Sanitization
```zexus
let htmlContext = "<div>Content</div>";
let jsContext = "var x = 'input';";
let sqlContext = "SELECT * FROM table";
let urlContext = "http://example.com?param=value";

let sanitizedHtml = sanitize htmlContext, "html";
let sanitizedJs = sanitize jsContext, "js";
let sanitizedSql = sanitize sqlContext, "sql";
let sanitizedUrl = sanitize urlContext, "url";
```

### Test Results
✅ **Working**: Basic sanitization
✅ **Working**: HTML encoding
✅ **Working**: Multiple encoding types
✅ **FIXED** (December 17, 2025): Variable scope issue resolved
  - SANITIZE now works in assignment expressions
  - `let clean = sanitize data, "html"` works correctly
  - Multi-layer sanitization works: `let stage3 = sanitize stage2, "html"`
  - HTML properly escaped: `<script>` → `&lt;script&gt;`

---

## Known Issues

### Issue 1: VALIDATE Unknown Schema Error
**Description**: Schema names not recognized, throws "Unknown schema: string"

**Example**:
```zexus
let data = "test";
validate data, "string";  // ❌ Error: Unknown schema: string
```

**Test**: test_capability_easy.zx Test 10
**Impact**: High - Validation feature unusable
**Root Cause**: Schema registry incomplete or not initialized

### Issue 2: SANITIZE Variable Scope
**Description**: Sanitized values not accessible when assigned to variables

**Example**:
```zexus
let stage3 = sanitize data, "html";
print stage3;  // ❌ Identifier 'stage3' not found
```

**Tests**: test_capability_complex.zx Test 13
**Impact**: Medium - Similar to sandbox return value issue

### Issue 3: Capability Scope in Functions
**Description**: Capabilities defined inside functions can't be accessed by grant

**Example**:
```zexus
action setupAccess() {
    capability admin_full { "description": "Admin", "scope": "admin" };
    grant "user" admin_full;  // ❌ 'admin_full' not found
}
```

**Test**: test_capability_medium.zx Test 8
**Impact**: Medium - Limits dynamic capability creation

---

## Best Practices

### 1. Define Capabilities at Module Level
```zexus
// ✅ Good: Define capabilities at top level
capability read_access {
    "description": "Read access",
    "scope": "data"
};

action grantAccess(user) {
    grant user read_access;
}
```

### 2. Use Descriptive Capability Names
```zexus
// ✅ Good: Clear, descriptive names
capability can_create_users {
    "description": "Create users",
    "scope": "users"
};

// ❌ Avoid: Vague names
capability cap1 {
    "description": "Something",
    "scope": "stuff"
};
```

### 3. Implement Hierarchical Access
```zexus
// ✅ Good: Clear hierarchy
grant "viewer" read_access;
grant "editor" read_access, write_access;
grant "admin" read_access, write_access, delete_access;
```

### 4. Use Immutable for Security-Critical Config
```zexus
// ✅ Good: Lock down critical settings
immutable encryptionKey = "...";
immutable allowedOrigins = ["https://app.example.com"];
immutable maxLoginAttempts = 3;
```

### 5. Sanitize All User Input
```zexus
// ✅ Good: Always sanitize untrusted input
let userComment = getUserInput();
let sanitized = sanitize userComment, "html";
storeComment(sanitized);
```

### 6. Revoke Capabilities on Logout
```zexus
// ✅ Good: Clean up on session end
action logout(sessionId) {
    revoke sessionId session_access;
    revoke sessionId user_capabilities;
}
```

---

## Real-World Examples

### Example 1: Multi-Tier Access Control
```zexus
// Define capability tiers
capability tier1_read {
    "description": "Basic read access",
    "scope": "tier1"
};
capability tier2_write {
    "description": "Intermediate write access",
    "scope": "tier2"
};
capability tier3_admin {
    "description": "Full admin access",
    "scope": "tier3"
};

// Grant based on user level
grant "basic_user" tier1_read;
grant "power_user" tier1_read, tier2_write;
grant "administrator" tier1_read, tier2_write, tier3_admin;
```

### Example 2: Immutable Application Configuration
```zexus
immutable appConfig = {
    "name": "SecureApp",
    "version": "2.0.0",
    "environment": "production",
    "database": {
        "host": "db.internal",
        "port": 5432,
        "ssl": true
    },
    "security": {
        "mfa": true,
        "sessionTimeout": 3600
    }
};
```

### Example 3: Input Sanitization Pipeline
```zexus
action processUserContent(rawContent) {
    // Layer 1: Remove scripts
    let step1 = sanitize rawContent, "html";
    
    // Layer 2: Clean attributes
    let step2 = sanitize step1, "html";
    
    // Layer 3: Final validation
    let clean = sanitize step2, "html";
    
    return clean;
}

let userComment = "<script>bad</script><p>Hello</p>";
let safeComment = processUserContent(userComment);
```

### Example 4: Session Management
```zexus
capability session_access {
    "description": "Active session access",
    "scope": "session"
};

action login(userId) {
    let sessionId = "session_" + userId;
    grant sessionId session_access;
    return sessionId;
}

action logout(sessionId) {
    revoke sessionId session_access;
    return "Logged out";
}
```

### Example 5: Rate Limiting Configuration
```zexus
immutable rateLimits = {
    "api": {
        "requestsPerMinute": 60,
        "requestsPerHour": 1000
    },
    "auth": {
        "attemptsPerMinute": 5,
        "lockoutDuration": 300
    },
    "upload": {
        "maxFileSize": 10485760,
        "maxFilesPerHour": 100
    }
};
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
| CAPABILITY | 60 | ~60 | 0 |
| GRANT | 60 | ~60 | 0 |
| REVOKE | 60 | ~60 | 0 |
| IMMUTABLE | 60 | ~60 | 0 |
| VALIDATE | 60 | ~10 | 1 |
| SANITIZE | 60 | ~55 | 1 |

### Critical Findings
1. **VALIDATE** - Schema registry incomplete
2. **SANITIZE** - Variable scope issues
3. **CAPABILITY** - Function scope limitations

---

## Related Keywords
- **ENTITY**: Type definitions for capability subjects
- **VERIFY**: Runtime verification (different from validate)
- **SEAL**: Similar to immutable for objects
- **AUDIT**: Track capability grants/revokes

---

*Last Updated: December 16, 2025*
*Tested with Zexus Interpreter*
*Phase 6 Complete - All 15 Security Keywords*
