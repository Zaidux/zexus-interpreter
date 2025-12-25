# Blockchain Keywords Reference

This document describes the blockchain-specific keywords implemented in Zexus for smart contract development.

## Table of Contents
- [Contract Keywords](#contract-keywords)
- [Modifiers](#modifiers)
- [Contract References](#contract-references)
- [Events](#events)
- [Function Modifiers](#function-modifiers)

---

## Contract Keywords

### `implements`

Specifies that a contract implements a protocol (interface).

**Syntax:**
```zexus
contract ContractName implements ProtocolName {
    // Contract body
}
```

**Example:**
```zexus
protocol Transferable {
    action transfer(to, amount)
    action balance() -> int
}

contract Token implements Transferable {
    state balances = {}
    
    action transfer(to, amount) {
        // Implementation
    }
    
    action balance() -> int {
        return balances[TX.caller]
    }
}
```

**Use Cases:**
- Ensure contracts adhere to standard interfaces (ERC20, ERC721, etc.)
- Enable polymorphism and interface-based programming
- Enforce implementation of required methods

---

## Modifiers

Modifiers are keywords that change the behavior of actions (functions) within contracts.

### `pure`

Marks an action as read-only with no state modification. Pure functions can only read state and compute values.

**Syntax:**
```zexus
action pure functionName(parameters) {
    // Read-only code
}
```

**Example:**
```zexus
contract Calculator {
    state result = 0
    
    action pure add(a, b) {
        return a + b  // No state modification
    }
    
    action pure getResult() {
        return result  // Only reads state
    }
}
```

**Restrictions:**
- Cannot modify state variables
- Cannot call non-pure actions
- Cannot emit events
- Cannot revert transactions

**Benefits:**
- Lower gas costs (in blockchain contexts)
- Predictable behavior
- Safe to call without transaction fees

---

### `view`

Alias for `pure`. Marks an action as read-only. Common in Solidity/Ethereum contexts.

**Syntax:**
```zexus
action view functionName(parameters) {
    // Read-only code
}
```

**Example:**
```zexus
contract Token {
    state balances = {}
    
    action view balanceOf(account) {
        return balances[account] ?? 0
    }
}
```

**Note:** `view` and `pure` are functionally identical in Zexus.

---

### `payable`

Marks an action as capable of receiving token/cryptocurrency transfers.

**Syntax:**
```zexus
action payable functionName(parameters) {
    // Can receive value
}
```

**Example:**
```zexus
contract Wallet {
    state balance = 0
    
    action payable deposit() {
        balance = balance + TX.value
        emit Deposit(TX.caller, TX.value)
    }
    
    action payable send(recipient, amount) {
        require(balance >= amount, "Insufficient balance")
        balance = balance - amount
        // Transfer logic
    }
}
```

**Use Cases:**
- Receiving deposits
- Processing payments
- Handling token transfers

---

## Contract References

### `this`

Reference to the current contract instance. Used to access contract properties and methods.

**Syntax:**
```zexus
this.propertyName
this.methodName(arguments)
```

**Example:**
```zexus
contract Token {
    state balances = {}
    state owner
    
    action init() {
        this.owner = TX.caller
        this.balances[this.owner] = 1000000
    }
    
    action getOwner() {
        return this.owner
    }
    
    action transfer(to, amount) {
        let from = TX.caller
        require(this.balances[from] >= amount, "Insufficient balance")
        
        this.balances[from] = this.balances[from] - amount
        this.balances[to] = this.balances[to] + amount
    }
}
```

**Use Cases:**
- Explicit contract property access
- Disambiguating between local and contract variables
- Self-referential operations
- Calling other contract methods

---

## Events

### `emit`

Emits an event for logging and external monitoring. Events are important for tracking contract activity.

**Syntax:**
```zexus
emit EventName(argument1, argument2, ...)
```

**Example:**
```zexus
contract Token {
    state balances = {}
    
    action transfer(recipient, amount) {
        let sender = TX.caller
        
        require(balances[sender] >= amount, "Insufficient balance")
        
        balances[sender] = balances[sender] - amount
        balances[recipient] = balances[recipient] + amount
        
        // Emit transfer event
        emit Transfer(sender, recipient, amount)
    }
    
    action mint(account, amount) {
        require(TX.caller == owner, "Only owner")
        balances[account] = balances[account] + amount
        
        // Emit mint event
        emit Mint(account, amount, TX.timestamp)
    }
}
```

**Use Cases:**
- Transaction logging
- State change notifications
- External system integration
- Audit trails
- UI updates in dApps

**Event Output:**
Events are typically logged to the console during execution:
```
🔔 Event: Transfer(0x123..., 0x456..., 100)
🔔 Event: Mint(0x789..., 1000, 1734278400)
```

---

## Function Modifiers

### `modifier`

Declares a reusable function modifier that can be applied to actions. Modifiers wrap action execution with pre/post conditions.

**Syntax:**
```zexus
modifier modifierName {
    // Pre-condition code
    // Action execution happens here implicitly
    // Post-condition code
}

action functionName() modifier modifierName {
    // Action body
}
```

**Example:**
```zexus
contract SecureVault {
    state owner
    state paused = false
    
    // Define modifiers
    modifier onlyOwner {
        require(TX.caller == owner, "Not authorized")
    }
    
    modifier whenNotPaused {
        require(!paused, "Contract is paused")
    }
    
    // Apply modifiers to actions
    action withdraw(amount) modifier onlyOwner modifier whenNotPaused {
        // Withdrawal logic
        emit Withdrawal(TX.caller, amount)
    }
    
    action pause() modifier onlyOwner {
        paused = true
        emit Paused()
    }
    
    action unpause() modifier onlyOwner {
        paused = false
        emit Unpaused()
    }
}
```

**Multiple Modifiers:**
You can apply multiple modifiers to a single action:
```zexus
action criticalOperation() modifier onlyOwner modifier whenNotPaused modifier validAmount {
    // Operation code
}
```

**Common Modifier Patterns:**

1. **Access Control:**
```zexus
modifier onlyOwner {
    require(TX.caller == owner, "Not owner")
}

modifier onlyAdmin {
    require(admins[TX.caller] == true, "Not admin")
}
```

2. **State Validation:**
```zexus
modifier whenNotPaused {
    require(!paused, "Contract paused")
}

modifier whenActive {
    require(active, "Contract not active")
}
```

3. **Value Validation:**
```zexus
modifier validAmount(amount) {
    require(amount > 0, "Amount must be positive")
}

modifier sufficientBalance(amount) {
    require(balances[TX.caller] >= amount, "Insufficient balance")
}
```

4. **Time Constraints:**
```zexus
modifier afterDeadline {
    require(TX.timestamp > deadline, "Before deadline")
}

modifier beforeDeadline {
    require(TX.timestamp < deadline, "After deadline")
}
```

---

## Complete Example

Here's a comprehensive example using all the new blockchain keywords:

```zexus
// Define protocol
protocol ERC20 {
    action transfer(to, amount) -> boolean
    action balanceOf(account) -> int
    action approve(spender, amount) -> boolean
}

// Implement protocol
contract MyToken implements ERC20 {
    state balances = {}
    state allowances = {}
    state owner
    state paused = false
    
    // Modifier declarations
    modifier onlyOwner {
        require(TX.caller == this.owner, "Not owner")
    }
    
    modifier whenNotPaused {
        require(!this.paused, "Contract paused")
    }
    
    // Constructor
    action init(initialSupply) {
        this.owner = TX.caller
        this.balances[this.owner] = initialSupply
        emit TokenCreated(this.owner, initialSupply)
    }
    
    // Pure/View functions
    action pure balanceOf(account) {
        return this.balances[account] ?? 0
    }
    
    action view totalSupply() {
        return 1000000
    }
    
    // Payable function
    action payable deposit() {
        this.balances[TX.caller] = this.balances[TX.caller] + TX.value
        emit Deposit(TX.caller, TX.value)
    }
    
    // Transfer with modifier
    action transfer(to, amount) modifier whenNotPaused {
        let from = TX.caller
        require(this.balances[from] >= amount, "Insufficient balance")
        
        this.balances[from] = this.balances[from] - amount
        this.balances[to] = this.balances[to] + amount
        
        emit Transfer(from, to, amount)
        return true
    }
    
    // Approve with modifier
    action approve(spender, amount) modifier whenNotPaused {
        let key = TX.caller + ":" + spender
        this.allowances[key] = amount
        emit Approval(TX.caller, spender, amount)
        return true
    }
    
    // Admin function with modifier
    action pause() modifier onlyOwner {
        this.paused = true
        emit Paused()
    }
    
    action unpause() modifier onlyOwner {
        this.paused = false
        emit Unpaused()
    }
}
```

---

## Integration with Existing Features

The new blockchain keywords integrate seamlessly with existing Zexus features:

### With State Management
```zexus
contract Ledger {
    state entries = []
    ledger transaction_log
    
    action recordTransaction(tx) {
        this.entries = push(this.entries, tx)
        emit TransactionRecorded(tx.id, TX.timestamp)
    }
}
```

### With Transaction Context (TX)
```zexus
contract Auction {
    action pure getCurrentBidder() {
        return TX.caller
    }
    
    action pure getTimestamp() {
        return TX.timestamp
    }
    
    action pure getGasRemaining() {
        return TX.gas_remaining
    }
}
```

### With Require Statements
```zexus
contract Validator {
    modifier validInput(value) {
        require(value > 0, "Value must be positive")
        require(value < 1000000, "Value too large")
    }
    
    action process(value) modifier validInput(value) {
        // Processing logic
    }
}
```

### With Tolerance Blocks (Enhanced REQUIRE)
```zexus
contract VIPValidator {
    modifier valueCheck(value, caller) {
        // Standard users need value >= 100, VIP bypass
        require value >= 100 {
            if (isVIP(caller)) return true;
        }
    }
    
    action premiumProcess(value) modifier valueCheck(value, TX.caller) {
        // Premium processing for VIP or high-value users
    }
}
```

---

## Best Practices

1. **Use `pure`/`view` for read-only functions** to save gas and improve clarity
2. **Mark payment-receiving functions as `payable`** explicitly
3. **Use `this` for clarity** when accessing contract properties
4. **Emit events for all state changes** to enable tracking and monitoring
5. **Create reusable modifiers** for common validation patterns
6. **Implement protocols** for standardization and interoperability
7. **Document event parameters** for external consumers
8. **Chain modifiers logically** (e.g., authentication before validation)

---

## Gas Optimization Tips

- Pure functions are cheaper than regular functions
- Emit events sparingly (each event costs gas)
- Modifiers add overhead - use judiciously
- Cache `this.property` access in local variables when used multiple times

---

## See Also

- [BLOCKCHAIN_FEATURES.md](BLOCKCHAIN_FEATURES.md) - Overview of blockchain capabilities
- [SECURITY_FEATURES.md](SECURITY_FEATURES.md) - Security features and patterns
- [MODULE_SYSTEM.md](MODULE_SYSTEM.md) - Module and import system
- [QUICK_START.md](QUICK_START.md) - Getting started with Zexus

---

**Version:** 1.0.4  
**Last Updated:** December 15, 2025  
**Status:** Production Ready ✅
