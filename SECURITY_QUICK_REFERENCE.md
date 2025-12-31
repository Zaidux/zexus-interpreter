# Zexus Security Quick Reference

**For Developers:** Essential security practices when writing Zexus code  
**Last Updated:** 2025-12-31 (Added contract `require()` validation)

---

## 🔴 Critical - Always Do This

### ✅ Use require() for Contract Validation (NEW - Fix #3)

```zexus
# ✅ SECURE - Validate all contract inputs
action transfer(from, to, amount) {
    require(amount > 0, "Amount must be positive");
    require(balance[from] >= amount, "Insufficient balance");
    require(from != to, "Cannot transfer to self");
    
    // Safe to proceed
    balance[from] = balance[from] - amount;
    balance[to] = balance[to] + amount;
}
```

### ✅ Sanitize All External Input

```zexus
# ❌ VULNERABLE
user_input = get_input();
query = "SELECT * FROM users WHERE name = '" + user_input + "'";

# ✅ SECURE
user_input = get_input();
sanitize user_input as sql;
query = "SELECT * FROM users WHERE name = '" + user_input + "'";
```

### ✅ Use Capabilities for Access Control

```zexus
# Define capability
capability admin_access;

# Grant only when needed
grant admin_user admin_access;

# Check before sensitive operations
action delete_user(user_id: integer) {
    # Capability check should happen here
    # Implementation needed: require(has_capability(sender, admin_access))
}
```

### ✅ Validate Business Logic

```zexus
# ❌ VULNERABLE - Negative amounts
action transfer(amount: integer) {
    balance = balance - amount;
}

# ✅ SECURE - Validation
action transfer(amount: integer) {
    require(amount > 0, "Amount must be positive");
    require(balance >= amount, "Insufficient balance");
    balance = balance - amount;
}
```

### ✅ Protect Contract State

```zexus
# ❌ VULNERABLE - No access control
contract Wallet {
    persistent storage owner: string
    
    action change_owner(new_owner: string) {
        owner = new_owner;  # Anyone can change!
    }
}

# ✅ SECURE - Access control
contract Wallet {
    persistent storage owner: string
    
    action change_owner(new_owner: string) {
        require(sender == owner, "Only owner");
        owner = new_owner;
    }
}
```

---

## 🟠 High Priority

### Prevent Reentrancy

```zexus
# ❌ VULNERABLE - External call before state update
contract Bank {
    action withdraw(amount: integer) {
        transfer(sender, amount);  # External call
        balances[sender] = balances[sender] - amount;  # State after
    }
}

# ✅ SECURE - State update before external call
contract Bank {
    action withdraw(amount: integer) {
        balances[sender] = balances[sender] - amount;  # State first
        transfer(sender, amount);  # External call after
    }
}
```

### Prevent Integer Overflow/Underflow

```zexus
# ❌ VULNERABLE
action buy(quantity: integer, price: integer) {
    total = quantity * price;  # Can overflow!
}

# ✅ SECURE - Check for overflow
action buy(quantity: integer, price: integer) {
    require(quantity > 0, "Invalid quantity");
    require(price > 0, "Invalid price");
    require(quantity <= 1000000, "Quantity too large");
    require(price <= 1000000, "Price too large");
    total = quantity * price;
}
```

### Validate File Paths

```zexus
# ❌ VULNERABLE (Before Fix)
action read_user_file(filename: string) {
    return read_file(filename);  # Path traversal was possible!
}

# ✅ SECURE - Now automatically protected!
action read_user_file(filename: string) {
    # Path traversal protection is now built-in!
    # This will automatically block ../../../etc/passwd
    return read_file(filename);
}

# ✅ EXTRA SECURE - Configure allowed directories
# In Python setup code:
import FileSystemModule;
FileSystemModule.configure_security(
    allowed_dirs=["/var/app/data", "/var/app/uploads"],
    strict=True
);
```

**What's Protected (As of 2025-12-31):**
- ✅ `../../../etc/passwd` → Blocked
- ✅ `/etc/shadow` → Blocked  
- ✅ `C:\Windows\System32\*` → Blocked
- ✅ URL-encoded paths → Blocked
- ✅ All file operations (read, write, remove, etc.)

---

## 🟡 Medium Priority

### Never Store Passwords in Plain Text

```zexus
# ❌ VULNERABLE
user_password = "secret123";
store(user_id, user_password);

# ✅ SECURE - Hash passwords
# Note: Need crypto library
import crypto;
password_hash = crypto.hash_password(user_password, "bcrypt");
store(user_id, password_hash);
```

### Use Secure Random for Security

```zexus
# ❌ VULNERABLE - Predictable
session_id = random();

# ✅ SECURE - Cryptographically secure
# Note: Need crypto library
import crypto;
session_id = crypto.random_string(64);
```

### Implement Rate Limiting

```zexus
# Use protect keyword
protect(login_action, {
    rate_limit: 10,  # 10 attempts
    window: 60  # per minute
});
```

---

## Sanitization Reference

### SQL Sanitization

```zexus
user_input = "'; DROP TABLE users; --";
sanitize user_input as sql;
# Now safe to use in SQL queries
```

### HTML/XSS Sanitization

```zexus
user_comment = "<script>alert('XSS')</script>";
sanitize user_comment as html;
# Now safe to display in HTML
```

### URL Sanitization

```zexus
user_url = "javascript:alert('XSS')";
sanitize user_url as url;
# Now safe to use as URL
```

### CSV Sanitization

```zexus
user_data = "=cmd|'/c calc'!A1";
sanitize user_data as csv;
# Now safe for CSV export
```

---

## Entity Security Patterns

### Immutable Fields

```zexus
# ❌ VULNERABLE - All fields mutable
entity User {
    id: integer,
    role: string = "user"
}

user = User { id: 1, role: "user" };
user.role = "admin";  # Can escalate!

# ✅ SECURE - Use validation in actions
entity User {
    id: integer,
    private _role: string = "user"
    
    action get_role() -> string {
        return _role;
    }
    
    action promote_to_admin(admin_key: string) {
        require(admin_key == ADMIN_SECRET, "Unauthorized");
        _role = "admin";
    }
}
```

---

## Smart Contract Security Checklist

- [ ] **Access Control:** All state-changing functions check sender
- [ ] **Input Validation:** All parameters validated (positive, non-zero, etc.)
- [ ] **Checks-Effects-Interactions:** State updates before external calls
- [ ] **Reentrancy Guard:** Use nonreentrant pattern if available
- [ ] **Integer Safety:** Check for overflow/underflow
- [ ] **Event Logging:** Emit events for all important actions
- [ ] **Emergency Stop:** Implement pause/unpause mechanism
- [ ] **Upgrade Path:** Plan for contract upgrades

---

## Common Vulnerability Patterns to Avoid

### 1. String Concatenation in Queries

```zexus
# ❌ NEVER DO THIS
query = "SELECT * FROM users WHERE id = " + user_input;

# ✅ ALWAYS SANITIZE
sanitize user_input as sql;
query = "SELECT * FROM users WHERE id = " + user_input;
```

### 2. Missing Null Checks

```zexus
# ❌ VULNERABLE
value = user.profile.email;  # Crashes if profile is null

# ✅ SECURE
if user.profile != null {
    value = user.profile.email;
}
```

### 3. Trusting User Input

```zexus
# ❌ VULNERABLE
action set_admin(is_admin: boolean) {
    user.admin = is_admin;  # User controls this!
}

# ✅ SECURE
action set_admin(is_admin: boolean, admin_password: string) {
    require(admin_password == ADMIN_SECRET, "Unauthorized");
    user.admin = is_admin;
}
```

### 4. Missing Rate Limits

```zexus
# ❌ VULNERABLE
action login(username: string, password: string) {
    # No rate limiting - brute force possible
}

# ✅ SECURE
protect(login, {
    rate_limit: 5,
    window: 300  # 5 attempts per 5 minutes
});
action login(username: string, password: string) {
    # Protected by rate limiting
}
```

---

## Security Testing

### Test for Common Vulnerabilities

```zexus
# Create test file: security_test.zx

# Test SQL injection
malicious = "'; DROP TABLE users; --";
sanitize malicious as sql;
assert(malicious != "'; DROP TABLE users; --", "Sanitization failed");

# Test XSS
xss = "<script>alert('xss')</script>";
sanitize xss as html;
assert(!xss.contains("<script>"), "XSS not prevented");

# Test negative amounts
try {
    transfer(-100);
    assert(false, "Negative transfer should fail");
} catch error {
    assert(true, "Correctly rejected negative amount");
}
```

---

## Security Resources

### Documentation
- Full vulnerability report: `VULNERABILITY_FINDINGS.md`
- Test results: `VULNERABILITY_TEST_RESULTS.md`
- Security features: `docs/SECURITY_FEATURES.md`

### Test Suite
- Run tests: `python3 -m zexus run tests/vulnerability_tests.zx`
- Location: `tests/vulnerability_tests.zx`

### Getting Help
- Report security issues: Create private security advisory on GitHub
- Questions: See documentation or create issue

---

## Quick Win Checklist

Use this checklist for every Zexus project:

- [x] All user input is sanitized
- [x] File paths are validated (no `..`) ✅ FIXED 2025-12-31
- [ ] Capabilities are used for access control
- [ ] Contract functions have access control
- [ ] All amounts/quantities validated as positive
- [ ] No plain text passwords in code
- [ ] Rate limiting on authentication
- [ ] Error messages don't leak sensitive data
- [x] Persistent storage has size limits ✅ FIXED 2025-12-31
- [ ] Functions have timeout protection

---

**Remember:** Security is not optional. These patterns prevent real attacks that cause real damage.

**When in doubt:** Sanitize, validate, and require authorization.

---

Last Updated: 2025-12-31
