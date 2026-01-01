# Cryptographic Functions - Quick Reference

**Status:** ✅ IMPLEMENTED (Security Fix #5)  
**Version:** Zexus v1.6.3  
**Date:** 2026-01-01

---

## Overview

Zexus provides enterprise-grade cryptographic functions for secure password storage, token generation, and timing-attack resistant comparisons.

**Never store passwords in plain text!** Always use `hash_password()`.

---

## Functions

### `hash_password(password)` 

Securely hash a password using bcrypt with automatic salting.

**Parameters:**
- `password` (string) - The password to hash

**Returns:** String - bcrypt hash ($2b$12$...)

**Example:**
```zexus
let password = "UserPassword123!"
let hashed = hash_password(password)
# hashed = "$2b$12$Jl6NUkNjrxemZFC9mo9gVOlPUDArpOKSCdmXA6IG.Lh49QXKsd0sK"

# Store hashed in database - NEVER store plain password!
```

**Features:**
- Industry-standard bcrypt algorithm
- Automatic unique salt for each hash
- Computationally expensive (resistant to brute force)
- Same password produces different hashes each time (due to salt)

---

### `verify_password(password, hash)`

Verify a password against a bcrypt hash using constant-time comparison.

**Parameters:**
- `password` (string) - The password to verify
- `hash` (string) - The bcrypt hash to compare against

**Returns:** Boolean - true if password matches, false otherwise

**Example:**
```zexus
# Registration
let user_password = "MyPassword123"
let stored_hash = hash_password(user_password)

# Login - later
let login_attempt = "MyPassword123"
let is_valid = verify_password(login_attempt, stored_hash)

if is_valid {
    print "Login successful!"
} else {
    print "Invalid password"
}
```

**Security:**
- Constant-time comparison (prevents timing attacks)
- Works with bcrypt hashes only
- Safe against brute-force when hash is properly generated

---

### `crypto_random(length?)`

Generate cryptographically secure random hex string.

**Parameters:**
- `length` (integer, optional) - Number of random bytes (default: 32)
  - Returns length * 2 hex characters (each byte = 2 hex digits)

**Returns:** String - Random hex string

**Example:**
```zexus
# Session token (32 bytes = 64 hex chars)
let session_token = crypto_random()
# "8d61ddb1bd49ee6bd3d459cde7b0de3685aec42e92b60fe786ba64cb50c637a5"

# API key (16 bytes = 32 hex chars)
let api_key = crypto_random(16)
# "ee55a1c20c07f4e6ba0d964ad9b9c81c"

# Short code (4 bytes = 8 hex chars)
let reset_code = crypto_random(4)
# "a3b4c5d6"
```

**Use Cases:**
- Session tokens
- API keys
- Password reset tokens
- CSRF tokens
- Nonces for cryptographic operations

**Security:**
- Uses Python's `secrets` module (CSPRNG)
- Suitable for security-sensitive applications
- **NOT** for gambling/games (use `random()` instead)

---

### `constant_time_compare(a, b)`

Compare two strings in constant time (timing-attack resistant).

**Parameters:**
- `a` (string) - First string
- `b` (string) - Second string

**Returns:** Boolean - true if strings are identical, false otherwise

**Example:**
```zexus
# Comparing secret tokens
let stored_token = "abc123def456"
let user_token = "abc123def456"

let is_match = constant_time_compare(stored_token, user_token)
if is_match {
    print "Tokens match!"
}
```

**Why Use This:**

❌ **NEVER DO THIS:**
```zexus
# Vulnerable to timing attacks!
if api_key == stored_api_key {
    grant_access()
}
```

The `==` operator returns immediately when it finds a mismatch. An attacker can measure response time to guess the correct key character by character!

✅ **DO THIS INSTEAD:**
```zexus
# Timing-attack resistant
if constant_time_compare(api_key, stored_api_key) {
    grant_access()
}
```

**Use Cases:**
- Comparing API keys
- Comparing session tokens
- Comparing HMAC signatures
- Comparing password hashes (though bcrypt does this internally)
- Any security-sensitive string comparison

---

## Complete Authentication Example

```zexus
# ============================================================================
# USER REGISTRATION
# ============================================================================

action register_user(username, password) {
    # Validate password strength
    require(len(password) >= 8, "Password must be at least 8 characters")
    
    # Hash password with bcrypt
    let password_hash = hash_password(password)
    
    # Store in database (pseudo-code)
    # db.insert("users", {username: username, password_hash: password_hash})
    
    print "User registered successfully!"
    return true
}

# ============================================================================
# USER LOGIN
# ============================================================================

action login_user(username, password) {
    # Retrieve stored hash from database (pseudo-code)
    # let stored_hash = db.query("SELECT password_hash FROM users WHERE username = ?", username)
    let stored_hash = "$2b$12$YKel3/aOlo6mDmY8adeYzOYBQF3kl4tcDEWfGmuJ8KYboFB5YmaXW"
    
    # Verify password
    let is_valid = verify_password(password, stored_hash)
    
    if is_valid {
        # Generate session token
        let session_token = crypto_random(32)
        
        # Store session (pseudo-code)
        # db.insert("sessions", {username: username, token: session_token})
        
        print "Login successful! Session: " + session_token
        return session_token
    } else {
        print "Invalid credentials"
        return null
    }
}

# ============================================================================
# SESSION VALIDATION
# ============================================================================

action validate_session(username, session_token) {
    # Retrieve stored token from database (pseudo-code)
    # let stored_token = db.query("SELECT token FROM sessions WHERE username = ?", username)
    let stored_token = "a3afef45b15cbecc502f6c378a88f49c57a56e25056b69a3ef00cc961667ec32"
    
    # Use constant-time comparison for security
    let is_valid = constant_time_compare(session_token, stored_token)
    
    if is_valid {
        print "Session valid"
        return true
    } else {
        print "Invalid session"
        return false
    }
}

# ============================================================================
# USAGE
# ============================================================================

# Register
register_user("alice", "SecurePassword123!")

# Login
let token = login_user("alice", "SecurePassword123!")

# Validate session
validate_session("alice", token)
```

---

## Security Best Practices

### ✅ DO:

1. **Always hash passwords before storage**
   ```zexus
   let hashed = hash_password(user_password)
   # Store hashed, never plain password
   ```

2. **Use crypto_random() for security-sensitive tokens**
   ```zexus
   let session_id = crypto_random(32)
   let api_key = crypto_random(24)
   ```

3. **Use constant_time_compare() for secret comparisons**
   ```zexus
   if constant_time_compare(token, stored_token) {
       # grant access
   }
   ```

4. **Enforce password requirements**
   ```zexus
   require(len(password) >= 12, "Password too short")
   require(contains_uppercase(password), "Needs uppercase")
   ```

### ❌ DON'T:

1. **Never store plain text passwords**
   ```zexus
   # ❌ VULNERABLE!
   db.insert("users", {username: user, password: plain_password})
   ```

2. **Never use regular random() for security**
   ```zexus
   # ❌ PREDICTABLE!
   let token = string(random(1000000, 9999999))  # Not cryptographically secure
   
   # ✅ SECURE!
   let token = crypto_random(16)
   ```

3. **Never use == for secret comparison**
   ```zexus
   # ❌ TIMING ATTACK VULNERABLE!
   if api_key == stored_key {
       grant_access()
   }
   
   # ✅ TIMING-ATTACK RESISTANT!
   if constant_time_compare(api_key, stored_key) {
       grant_access()
   }
   ```

4. **Never hash passwords with MD5/SHA1**
   ```zexus
   # Zexus doesn't provide MD5/SHA1 for password hashing
   # Use hash_password() only - it uses bcrypt
   ```

---

## Requirements

**Python Dependencies:**
- `bcrypt` >= 4.0.0 (for password hashing)
- `secrets` (Python standard library - no install needed)

**Installation:**
```bash
pip install bcrypt
```

---

## Performance Notes

- **hash_password()**: Deliberately slow (~100-300ms) to resist brute force
- **verify_password()**: Same performance as hash_password()
- **crypto_random()**: Fast (< 1ms for typical token sizes)
- **constant_time_compare()**: Constant time (timing independent of content)

The slow performance of bcrypt is **intentional** - it makes brute-force attacks computationally infeasible.

---

## Further Reading

- [OWASP Password Storage Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)
- [bcrypt Documentation](https://github.com/pyca/bcrypt/)
- [Timing Attacks Explained](https://codahale.com/a-lesson-in-timing-attacks/)
- [Cryptographically Secure RNG](https://docs.python.org/3/library/secrets.html)
