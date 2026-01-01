# Integer Overflow Protection

## Overview

Zexus provides automatic integer overflow protection to prevent arithmetic operations from producing incorrect results due to integer overflow or underflow. This is a critical security feature that protects against:

- Resource exhaustion attacks via huge integers
- Financial calculation errors in smart contracts
- Logic errors from wrapped integer values
- Potential security vulnerabilities in arithmetic-heavy code

## Safe Integer Range

Zexus uses 64-bit signed integers with the following safe range:

```
MAX_SAFE_INT = 9,223,372,036,854,775,807  (2^63 - 1)
MIN_SAFE_INT = -9,223,372,036,854,775,808 (-2^63)
```

Any arithmetic operation that produces a result outside this range will trigger an error.

## Protected Operations

All integer arithmetic operations are automatically protected:

### Addition (`+`)

```zexus
let max = 9223372036854775807
let result = max + 1000000  # ERROR: Integer overflow in addition
```

### Subtraction (`-`)

```zexus
let min = -9223372036854775808
let result = min - 1000000  # ERROR: Integer overflow in subtraction
```

### Multiplication (`*`)

```zexus
let big = 10000000000000000
let result = big * big  # ERROR: Integer overflow in multiplication
```

### Division (`/`)

Division is also protected against overflow (though rare) and division by zero:

```zexus
let result = 100 / 0  # ERROR: Division by zero
```

## Error Messages

When overflow is detected, Zexus provides helpful error messages:

```
Integer overflow in addition: result 9223372036854776807 exceeds safe integer range
MAX_SAFE_INT = 9,223,372,036,854,775,807
MIN_SAFE_INT = -9,223,372,036,854,775,808

Suggestion: Use require() to validate input bounds before arithmetic, or break large calculations into smaller steps with explicit bounds checking.
```

## Best Practices

### 1. Validate Inputs Before Arithmetic

Use `require()` to validate that inputs won't cause overflow:

```zexus
action safe_multiply(a, b) {
    let max_safe = 9223372036854775807
    
    # Validate that multiplication won't overflow
    require(a <= max_safe / 2, "First number too large for safe multiplication")
    require(b <= max_safe / 2, "Second number too large for safe multiplication")
    
    return a * b
}
```

### 2. Break Large Calculations Into Steps

Instead of:
```zexus
# Risky: may overflow
let result = huge_number1 * huge_number2 + huge_number3
```

Do:
```zexus
# Safer: validate at each step
let temp = safe_multiply(huge_number1, huge_number2)
require(temp <= 9000000000000000000, "Intermediate result too large")

let result = safe_add(temp, huge_number3)
```

### 3. Use Explicit Bounds in Financial Calculations

```zexus
action calculate_compound_interest(principal, rate_percent, years) {
    # Validate reasonable bounds for financial calculations
    require(principal > 0 && principal <= 1000000000000, "Invalid principal")
    require(rate_percent > 0 && rate_percent <= 100, "Invalid rate")
    require(years > 0 && years <= 100, "Invalid years")
    
    let amount = principal
    let i = 0
    
    while (i < years) {
        let interest = amount * rate_percent / 100
        amount = amount + interest
        
        # Ensure we haven't exceeded reasonable bounds
        require(amount <= 10000000000000000, "Amount exceeded safe bounds")
        i = i + 1
    }
    
    return amount
}
```

### 4. Set Maximum Values for User Inputs

```zexus
action process_payment(amount) {
    # Enforce business logic limits
    let MAX_PAYMENT = 1000000000000  # $1 trillion limit
    require(amount > 0 && amount <= MAX_PAYMENT, "Payment amount out of range")
    
    # Safe to proceed with arithmetic
    let fee = amount * 3 / 100
    let total = amount + fee
    
    return total
}
```

## Real-World Example: Token Contract

```zexus
contract Token {
    state {
        balances: {},
        total_supply: 0
    }
    
    action mint(to, amount) {
        # Validate amount is reasonable
        require(amount > 0, "Amount must be positive")
        require(amount <= 1000000000000, "Mint amount too large")
        
        # Check that adding to total supply won't overflow
        let max_supply = 1000000000000000000  # 1 quintillion
        require(total_supply + amount <= max_supply, "Total supply would exceed maximum")
        
        # Safe to mint
        state.total_supply = state.total_supply + amount
        
        let current_balance = get(state.balances, to, 0)
        put(state.balances, to, current_balance + amount)
        
        return true
    }
    
    action transfer(from, to, amount) {
        require(amount > 0, "Amount must be positive")
        
        # Get current balances
        let from_balance = get(state.balances, from, 0)
        let to_balance = get(state.balances, to, 0)
        
        # Validate sufficient balance
        require(from_balance >= amount, "Insufficient balance")
        
        # Check that adding to recipient's balance won't overflow
        # (Very unlikely but good to check in critical financial code)
        let max_balance = 10000000000000000
        require(to_balance + amount <= max_balance, "Recipient balance would overflow")
        
        # Safe to transfer
        put(state.balances, from, from_balance - amount)
        put(state.balances, to, to_balance + amount)
        
        return true
    }
}
```

## Performance Considerations

The overflow protection adds minimal overhead:

- **Addition/Subtraction:** Single comparison after operation
- **Multiplication:** Single comparison after operation  
- **Division:** Same as before (division by zero already checked)

Typical overhead: **< 1%** for arithmetic-heavy code

## Integration with Other Security Features

### Works With require()

```zexus
action validate_and_multiply(a, b) {
    let result = a * b  # May error if overflow
    require(result > 0, "Result must be positive")  # Additional validation
    return result
}
```

### Compatible With Sanitization

```zexus
action process_user_number(input) {
    let num = parse_int(sanitize(input, "number"))
    
    # Even with sanitized input, validate bounds
    require(num >= 0 && num <= 1000000, "Number out of range")
    
    # Safe arithmetic
    let result = num * 1000
    return result
}
```

## Common Patterns

### Pattern 1: Safe Range Check

```zexus
action in_safe_range(value) {
    let max = 9223372036854775807
    let min = -9223372036854775808
    return value >= min && value <= max
}
```

### Pattern 2: Safe Addition Helper

```zexus
action safe_add(a, b) {
    let max = 9223372036854775807
    let min = -9223372036854775808
    
    # Check if addition would overflow
    if (a > 0 && b > max - a) {
        return error("Addition would overflow")
    }
    if (a < 0 && b < min - a) {
        return error("Addition would underflow")
    }
    
    return a + b
}
```

### Pattern 3: Safe Multiplication Helper

```zexus
action safe_multiply(a, b) {
    let max = 9223372036854775807
    
    # Handle edge cases
    if (a == 0 || b == 0) { return 0 }
    
    # Check if multiplication would overflow
    if (a > 0 && b > 0 && a > max / b) {
        return error("Multiplication would overflow")
    }
    if (a < 0 && b < 0 && a < max / b) {
        return error("Multiplication would overflow")
    }
    
    return a * b
}
```

## Security Implications

### Prevents Integer Overflow Attacks

```zexus
# BEFORE (vulnerable):
action allocate_buffer(size) {
    let buffer_size = size * 8  # Could overflow to small number!
    # Allocate buffer_size bytes... (DANGEROUS)
}

# AFTER (protected):
action allocate_buffer(size) {
    require(size > 0 && size <= 1000000, "Invalid buffer size")
    let buffer_size = size * 8  # Overflow detected automatically
    # Safe allocation
}
```

### Protects Financial Calculations

```zexus
# BEFORE (vulnerable):
action calculate_total(price, quantity) {
    return price * quantity  # Could overflow!
}

# AFTER (protected):
action calculate_total(price, quantity) {
    require(price > 0 && price <= 1000000000, "Invalid price")
    require(quantity > 0 && quantity <= 1000000, "Invalid quantity")
    
    let total = price * quantity  # Protected against overflow
    require(total <= 100000000000000, "Total exceeds maximum")
    return total
}
```

## Comparison with Other Languages

| Language | Default Overflow Behavior | Protection |
|----------|--------------------------|------------|
| **Zexus** | ✅ Automatic error on overflow | Built-in |
| C/C++ | ⚠️ Wraps silently | Manual checks needed |
| Java | ⚠️ Wraps silently | Manual checks or `Math.addExact()` |
| Python | ✅ Arbitrary precision | Automatic (different approach) |
| Rust | ⚠️ Panics in debug, wraps in release | Opt-in with `checked_*` methods |
| JavaScript | ⚠️ Wraps or becomes Infinity | Manual checks needed |

Zexus takes the safest approach: **all overflows are errors by default**.

## FAQ

### Q: Can I disable overflow protection?

A: No. Overflow protection is a core security feature and cannot be disabled. Use explicit `require()` checks if you need custom bounds.

### Q: What if I need larger numbers?

A: For most use cases, 64-bit integers (±9 quintillion) are sufficient. If you need larger numbers:
- Break calculations into smaller steps
- Use fixed-point arithmetic (multiply by 1000, etc.)
- Future: We may add BigInt support

### Q: Does this affect performance?

A: Minimal impact (< 1%). The overflow checks are simple comparisons after each operation.

### Q: What about other operations like bitwise operators?

A: Currently only arithmetic operators (+, -, *, /) are protected. Bitwise operators may be added in future versions.

## Summary

Zexus's automatic integer overflow protection:

✅ **Prevents** silent integer overflow bugs  
✅ **Detects** arithmetic errors automatically  
✅ **Provides** clear error messages with suggestions  
✅ **Protects** financial and security-critical calculations  
✅ **Requires** minimal code changes  
✅ **Adds** negligible performance overhead  

Use `require()` to validate bounds before arithmetic for best results.

---

**Related Documentation:**
- [Contract require() Function](CONTRACT_REQUIRE.md)
- [Security Features Guide](SECURITY_FEATURES.md)
- [Quick Reference](QUICK_REFERENCE.md)
