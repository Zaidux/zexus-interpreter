# VERIFY Keyword - Complete Enhancement Guide

**Date**: December 19, 2025  
**Status**: ✨ Fully Enhanced & Production Ready

## Overview

The VERIFY keyword has been significantly enhanced to provide a comprehensive verification system that goes far beyond simple assertions. It now supports:

- **Data/Format Verification** - Email, URL, phone, type checking
- **Access Control** - Block unauthorized access with custom actions
- **Database Integration** - Verify data exists/unique in databases  
- **Environment Validation** - Check configuration and env vars
- **Pattern Matching** - Regex validation for any format
- **Custom Logic Blocks** - Developer-defined verification logic
- **Security Gates** - Prevent bad data from entering the system

## Why This Enhancement Matters

Traditional verification just throws errors. **Zexus VERIFY now does more:**

1. **Blocks Access** - Stops unauthorized users from accessing resources
2. **Prevents Bad Data** - Keeps malicious input out of your system
3. **Validates Formats** - Ensures emails, URLs, phones are valid
4. **Checks Databases** - Verifies data exists or is unique
5. **Validates Config** - Ensures environment variables are set correctly
6. **Custom Actions** - Execute developer-defined logic on failure

## Syntax Summary

```zexus
// Basic assertion (original)
verify condition, "error message";

// With custom logic block
verify condition {
    // Your custom logic here
    log_error("Failed");
    send_alert(admin);
}

// Data verification
verify is_email(input), "Invalid email";
verify is_url(url), "Invalid URL";
verify is_phone(phone), "Invalid phone";

// Access control (blocks access)
verify userRole == "admin" {
    log_unauthorized_access(user);
    block_request();
}

// Database verification
verify:db userId exists_in "users", "User not found";
verify:db email unique_in "users", "Email taken";

// Environment variables
verify env_exists("API_KEY"), "API_KEY not set";
verify env_get("DEBUG") == "false", "Debug must be off";

// Pattern matching
verify matches_pattern(zip, "^[0-9]{5}$"), "Invalid zipcode";
```

## Real-World Use Case: Email/Password Form

**The Problem**: Users input bad data (invalid emails, weak passwords) that get into your database and cause issues.

**The Solution**: Use VERIFY to block bad data at the gate:

```zexus
action validateLoginForm(email, password) {
    // Verify email format - block if invalid
    verify is_email(email) {
        print "[FORM] Invalid email format";
        print "[FORM] Blocking form submission";
        print "[FORM] Bad data will NOT enter system";
        return false;
    }
    
    // Verify password strength - block if weak
    let strength = password_strength(password);
    verify strength != "weak" {
        print "[FORM] Password too weak";
        print "[FORM] Access denied";
        return false;
    }
    
    // All checks passed - allow login
    print "✓ Login allowed - data is valid";
    return true;
}

// Valid attempt
validateLoginForm("user@example.com", "StrongP@ss123");  // ✓ Passes

// Invalid attempts - BLOCKED
validateLoginForm("not-an-email", "StrongP@ss123");      // ✗ Blocked
validateLoginForm("user@example.com", "123");            // ✗ Blocked
```

## New Builtin Helper Functions

Zexus now includes 13+ verification helpers:

```zexus
// Format validators
is_email(value)              // Email: user@example.com
is_url(value)                // URL: https://example.com
is_phone(value)              // Phone: 123-456-7890
is_numeric(value)            // Checks if numeric
is_alpha(value)              // Only letters
is_alphanumeric(value)       // Letters and numbers only

// Pattern matching
matches_pattern(value, regex) // Custom regex validation

// Environment variables
env_get(name, default)       // Get env var with optional default
env_set(name, value)         // Set env var
env_exists(name)             // Check if env var exists

// Security helpers
password_strength(password)  // Returns "weak"/"medium"/"strong"
sanitize_input(value)        // Remove dangerous characters
validate_length(value, min, max) // Check string length
```

## Advanced Examples

### Multi-Layer Security Validation

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
            print "[API] Request contains dangerous characters";
            return false;
        }
    }
    
    print "✓ API request validated";
    return true;
}
```

### User Registration with Complete Validation

```zexus
action registerUser(email, password, username) {
    // Email validation
    verify is_email(email) {
        print "Invalid email format";
        return false;
    }
    
    // Password strength check
    let strength = password_strength(password);
    verify strength == "strong" {
        print "Password must be strong";
        print "Requirements: 12+ chars, uppercase, lowercase, numbers, symbols";
        return false;
    }
    
    // Username validation
    verify validate_length(username, 3, 20) {
        print "Username must be 3-20 characters";
        return false;
    }
    
    verify is_alphanumeric(username) {
        print "Username must be alphanumeric only";
        return false;
    }
    
    // Database uniqueness check (if db_handler injected)
    // verify:db email unique_in "users", "Email already registered";
    // verify:db username unique_in "users", "Username taken";
    
    print "✓ User registration validated";
    return true;
}
```

### Configuration Validation

```zexus
action verifyAppConfiguration() {
    print "Checking application configuration...";
    
    // Required environment variables
    verify env_exists("API_KEY") {
        print "[CONFIG] API_KEY not set";
        print "[CONFIG] Set via: export API_KEY=your_key";
        return false;
    }
    
    verify env_exists("DATABASE_URL") {
        print "[CONFIG] DATABASE_URL not set";
        return false;
    }
    
    verify env_exists("API_TIMEOUT") {
        print "[CONFIG] API_TIMEOUT not set";
        return false;
    }
    
    // Validate timeout is numeric
    let timeout = env_get("API_TIMEOUT");
    verify is_numeric(timeout) {
        print "[CONFIG] API_TIMEOUT must be a number";
        return false;
    }
    
    // Ensure debug is off in production
    let debugMode = env_get("DEBUG_MODE", "false");
    verify debugMode == "false" {
        print "[CONFIG] DEBUG_MODE must be 'false' in production";
        return false;
    }
    
    print "✓ All configuration valid";
    return true;
}
```

### Input Sanitization & XSS Prevention

```zexus
action processUserInput(input) {
    print "Processing user input...";
    
    // Sanitize input first
    let clean = sanitize_input(input);
    
    // Verify it wasn't malicious
    verify clean == input {
        print "[SECURITY] Malicious input detected";
        print "[SECURITY] Original: " + input;
        print "[SECURITY] Sanitized: " + clean;
        print "[SECURITY] Input blocked from entering system";
        log_security_event("xss_attempt", input);
        return false;
    }
    
    // Safe to process
    print "✓ Input safe to process";
    return clean;
}

// Examples
processUserInput("Hello World");                    // ✓ Safe
processUserInput("<script>alert('xss')</script>");  // ✗ Blocked
processUserInput("DROP TABLE users");               // ✗ Blocked
```

## Database Integration

Developers can inject custom database handlers to enable database verification:

```zexus
// In your Zexus application initialization:
// Python side:
// env.set('__db_handler__', my_database_handler)

// Then in Zexus code:
verify:db userId exists_in "users", "User not found";
verify:db email unique_in "users", "Email already in use";
verify:db transactionId exists_in "transactions", "Transaction not found";
```

**Database Handler Interface** (Python):
```python
class DatabaseHandler:
    def exists_in(self, table, value):
        # Check if value exists in table
        return self.db.query(f"SELECT * FROM {table} WHERE id = ?", value) is not None
    
    def unique_in(self, table, value):
        # Check if value is unique (doesn't exist)
        return self.db.query(f"SELECT * FROM {table} WHERE id = ?", value) is None
```

## Migration Guide

### Old Code
```zexus
// Before enhancement
if (!is_valid_email(email)) {
    throw "Invalid email";
}
```

### New Code
```zexus
// After enhancement - more concise, more powerful
verify is_email(email) {
    log_error("Invalid email attempt");
    block_submission();
}
```

## Testing

Two comprehensive test suites are provided:

1. **test_verify_enhanced.zx** - Tests all builtin helpers and basic functionality
2. **test_verify_modes.zx** - Tests verification modes and real-world use cases

Run tests:
```bash
./zx test_verify_enhanced.zx
./zx test_verify_modes.zx
```

## Implementation Files

All enhancements are implemented across these files:

1. **zexus_ast.py** - Extended `VerifyStatement` with new properties
2. **strategy_context.py** - Parser support for all modes and syntax
3. **statements.py** - Evaluator with mode handlers
4. **functions.py** - 13+ builtin verification helpers

## Performance

Verification operations are optimized:

- Email/URL/Phone validation: O(1) regex matching
- Pattern matching: O(n) where n = pattern length
- Environment variables: O(1) dict lookup
- Database queries: O(1) with proper indexing

## Security Considerations

1. **Input Sanitization** - Always sanitize user input before verification
2. **Env Vars** - Never store secrets in code, use environment variables
3. **Database** - Use parameterized queries in your database handler
4. **Regex** - Be careful with complex regex patterns (ReDoS attacks)
5. **Error Messages** - Don't leak sensitive information in error messages

## Best Practices

1. **Layer Verification** - Use multiple verification checks for critical operations
2. **Fail Early** - Verify inputs at the entry point, before processing
3. **Custom Logic** - Use `{}` blocks for logging, alerting, and blocking
4. **Clear Messages** - Provide helpful error messages for users
5. **Test Thoroughly** - Verify both valid and invalid inputs

## Future Enhancements

Potential future additions:

- ✅ Database mode with custom queries
- ✅ Environment variable mode
- ✅ Pattern matching mode
- ⏳ OAuth/JWT token verification
- ⏳ File format verification (PDF, images)
- ⏳ Credit card Luhn algorithm validation
- ⏳ International phone number validation

## Conclusion

The enhanced VERIFY keyword transforms Zexus into a powerful validation framework. It's no longer just about assertions—it's about:

- **Blocking bad actors**
- **Protecting your data**
- **Validating everything**
- **Customizing security**

Use VERIFY to build secure, robust applications that reject invalid data at every gate.

---

**Questions?** See [SECURITY.md](SECURITY.md) for more security features.  
**Examples?** Check [test_verify_enhanced.zx](../test_verify_enhanced.zx) and [test_verify_modes.zx](../test_verify_modes.zx)  
**Issues?** Report to the Zexus team.

**Last Updated**: December 19, 2025  
**Version**: 2.0 (Enhanced)
