# Zexus Security Features - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Define Entities (Type-Safe Structs)

```zexus
entity User {
    username: string,
    email: string,
    role: string = "user",
    is_active: boolean = true
}

let john = User.create({
    username: "john",
    email: "john@example.com",
    role: "admin"
})

print(john.username)  # Output: "john"
```

### 2. Protect Functions with Verification

```zexus
action transfer_money(to_user: User, amount: integer) -> boolean {
    print("Transferring " + amount + " to " + to_user.username)
    return true
}

# Add runtime verification checks
verify(transfer_money, [
    action() { return balance >= amount }      # Check balance
    action() { return to_user != null }        # User exists
    action() { return amount > 0 }             # Valid amount
])
```

### 3. Create Smart Contracts (Persistent State)

```zexus
contract BankAccount {
    persistent storage owner: User
    persistent storage balance: integer = 1000
    
    action withdraw(amount: integer) -> boolean {
        if (balance >= amount) {
            balance = balance - amount
            return true
        }
        return false
    }
    
    action deposit(amount: integer) -> boolean {
        balance = balance + amount
        return true
    }
}

# Deploy the contract
let my_account = BankAccount.deploy({
    owner: john,
    balance: 5000
})

# Use it
my_account.withdraw(100)
print(my_account.get_state().balance)  # Output: 4900
```

### 4. Enforce Security Rules with Protect

```zexus
protect(transfer_money, {
    rate_limit: 10,           # Max 10 calls per minute
    auth_required: true,      # Must be authenticated
    require_https: true,      # Only over HTTPS
    min_password_strength: "strong",
    session_timeout: 3600,    # 1 hour
    allowed_ips: ["10.0.0.0/8"],
    blocked_ips: []
})
```

### 5. Configure Global Authentication

```zexus
auth {
    provider: "oauth2",
    scopes: ["read", "write", "delete"],
    token_expiry: 3600,
    mfa_required: true,
    session_timeout: 1800
}
```

### 6. Add Request Middleware

```zexus
middleware(log_requests, action(request, response) {
    print("Incoming request: " + request.method + " " + request.path)
    return true  # Continue to next handler
})

middleware(check_token, action(request, response) {
    let token = request.headers["Authorization"]
    if (token == null) {
        response.status = 401
        return false  # Stop chain
    }
    return true  # Continue
})
```

### 7. Rate Limiting (Throttle)

```zexus
throttle(api_endpoint, {
    requests_per_minute: 100,
    burst_size: 10,
    per_user: true
})
```

### 8. Performance Caching

```zexus
cache(expensive_database_query, {
    ttl: 300,                    # Cache for 5 minutes
    invalidate_on: ["data_changed"]
})
```

---

## 📋 Complete Example: Secure Banking API

```zexus
# 1. Define entity for users
entity BankUser {
    id: string,
    name: string,
    email: string,
    balance: integer = 0,
    account_type: string = "standard"
}

# 2. Create a smart contract for the bank
contract Bank {
    persistent storage accounts: Map<string, BankUser>
    persistent storage total_deposits: integer = 0
    
    action create_account(user: BankUser) -> boolean {
        accounts[user.id] = user
        return true
    }
    
    action deposit(user_id: string, amount: integer) -> boolean {
        let user = accounts[user_id]
        if (user == null || amount <= 0) {
            return false
        }
        user.balance = user.balance + amount
        total_deposits = total_deposits + amount
        return true
    }
    
    action withdraw(user_id: string, amount: integer) -> boolean {
        let user = accounts[user_id]
        if (user == null || amount <= 0 || user.balance < amount) {
            return false
        }
        user.balance = user.balance - amount
        return true
    }
}

# 3. Export for access control (file-level)
export Bank to "bank_service.zx" with "read_write"

# 4. Configure authentication
auth {
    provider: "oauth2",
    scopes: ["bank:read", "bank:write"],
    token_expiry: 3600,
    mfa_required: true,
    session_timeout: 1800
}

# 5. Add verification checks (runtime)
verify(Bank.deposit, [
    action() { return user_id != null }
    action() { return amount > 0 }
    action() { return amount <= 10000 }  # Max single deposit
])

verify(Bank.withdraw, [
    action() { return user_id != null }
    action() { return amount > 0 }
    action() { return accounts[user_id].balance >= amount }
])

# 6. Enforce security rules
protect(Bank.deposit, {
    rate_limit: 100,
    auth_required: true,
    require_https: true,
    min_password_strength: "strong",
    session_timeout: 3600,
    allowed_ips: ["10.0.0.0/8"]
})

protect(Bank.withdraw, {
    rate_limit: 50,  # More restrictive for withdrawals
    auth_required: true,
    require_https: true,
    min_password_strength: "strong",
    session_timeout: 1800
})

# 7. Add middleware for request logging
middleware(audit_log, action(request, response) {
    print("[AUDIT] " + request.method + " " + request.path)
    return true
})

middleware(check_maintenance, action(request, response) {
    if (is_under_maintenance && request.path != "/status") {
        response.status = 503
        return false
    }
    return true
})

# 8. Rate limiting
throttle(Bank.deposit, {
    requests_per_minute: 100,
    burst_size: 5,
    per_user: true
})

# 9. Caching for read operations
cache(Bank.get_balance, {
    ttl: 60,  # Cache balance for 1 minute
    invalidate_on: ["balance_changed"]
})

# Usage
let bank = Bank.deploy({
    accounts: {},
    total_deposits: 0
})

let alice = BankUser.create({
    id: "alice123",
    name: "Alice",
    email: "alice@bank.com",
    balance: 0,
    account_type: "premium"
})

bank.create_account(alice)
bank.deposit("alice123", 1000)
bank.withdraw("alice123", 500)

print(alice.balance)  # Output: 500
```

---

## 🔑 Key Points

| Feature | Purpose | Use Case |
|---------|---------|----------|
| **entity** | Type-safe data structures | Define user, account, transaction objects |
| **export** | File-level access control | Restrict which files can access functions |
| **verify** | Runtime condition checks | Ensure user is authenticated, balance sufficient |
| **protect** | Security guardrails | Rate limit, require HTTPS, enforce password strength |
| **contract** | Persistent state & transactions | Smart contracts, blockchain-like state |
| **auth** | Authentication configuration | OAuth2, JWT, MFA setup |
| **middleware** | Request processing pipeline | Log requests, check maintenance, validate headers |
| **throttle** | Rate limiting | Prevent abuse, handle spikes |
| **cache** | Performance optimization | Cache expensive operations |

---

## 🛡️ Security Layers Summary

```
Layer 1 (File):     export        WHO can access
Layer 2 (Runtime):  verify        IF they should be allowed
Layer 3 (Rules):    protect       Enforce security rules
Layer 4 (Process):  middleware    Process & validate requests
Layer 5 (Auth):     auth          Handle authentication
Layer 6 (Throttle): throttle      Prevent abuse
Layer 7 (Cache):    cache         Optimize performance
Layer 8 (State):    contract      Persistent storage
Layer 9 (Types):    entity        Type safety
```

---

## ✅ Checklist Before Deploying

- [ ] Entity definitions for all data types
- [ ] Export restrictions for sensitive functions
- [ ] Verify checks for all conditions
- [ ] Protect rules with appropriate rate limits
- [ ] Auth configured with MFA enabled
- [ ] Middleware added for audit logging
- [ ] Throttle limits set to prevent abuse
- [ ] Cache policies for expensive operations
- [ ] Contract storage for persistent state
- [ ] HTTPS enforced in protect rules
- [ ] IP whitelist/blacklist configured
- [ ] Session timeouts set
- [ ] Password strength requirements defined

---

## 🐛 Debugging Tips

1. **Check if verify checks are failing:**
   ```zexus
   verify(my_func, [check1, check2], error_handler)
   ```

2. **Check contract state:**
   ```zexus
   print(contract.get_state())
   print(contract.get_balance())
   ```

3. **Check rate limiter status:**
   ```zexus
   print(throttle.get_requests_today())
   ```

4. **Check cache hit rate:**
   ```zexus
   print(cache.get_stats())
   ```

5. **Enable audit logging:**
   ```zexus
   middleware(audit, action(req, res) {
       print("[AUDIT] " + JSON.stringify(req))
       return true
   })
   ```

---

## 📚 Learn More

- **Full documentation:** See `src/README.md` for comprehensive guide
- **Advanced patterns:** See `SECURITY_FEATURES.md` for deep dive
- **Examples:** Check `examples/` directory for more samples

Happy securing! 🎉
