# Zexus Blockchain & Smart Contract Features

**Production-ready blockchain support for Zexus**

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [State & Immutability](#state--immutability)
4. [Transaction Context](#transaction-context)
5. [Cryptographic Primitives](#cryptographic-primitives)
6. [Resource Management](#resource-management)
7. [Smart Contract Examples](#smart-contract-examples)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)

---

## Overview

Zexus provides **native blockchain and smart contract support** with:

✅ **Immutable Ledger** - Versioned, append-only state storage  
✅ **Transaction Context** - Built-in TX object with caller, timestamp, gas tracking  
✅ **Gas Limits** - Prevent DoS attacks with execution cost limits  
✅ **Cryptographic Primitives** - Hashing, signatures, verification  
✅ **Security Enforcement** - Require statements, access control, revert handling  

**Why Use Keywords Instead of Plugins?**

- **Compile-time Enforcement**: Immutability and gas limits checked at compile time
- **Performance**: Keywords are faster than function calls
- **Safety**: Type system integration prevents errors
- **Clarity**: Intent is explicit (`ledger` vs `let`)

---

## Architecture

### Keywords vs. Built-in Functions

**Keywords** (Compile-time enforcement):
- `ledger` - Immutable state ledger
- `state` - Mutable contract state
- `contract` - Smart contract definition
- `limit` - Gas/resource limit
- `revert` - Transaction rollback
- `require` - Conditional revert

**Built-in Functions** (Runtime operations):
- `hash()` - Cryptographic hashing
- `sign()` - Create digital signatures
- `verify_sig()` - Verify signatures
- `TX` object - Transaction context

---

## State & Immutability

### The `ledger` Keyword

**Ledger** creates an **immutable, versioned data store**. Every write creates a new version instead of modifying the old value.

#### Syntax

```zexus
ledger variable_name;
ledger variable_name = initial_value;
```

#### Features

- **Immutability**: Old values never change
- **Versioning**: Complete history of all changes
- **Cryptographic Integrity**: Each entry is hashed and linked
- **Audit Trail**: Full transaction history

#### Example

```zexus
contract Bank {
    ledger transaction_history;
    
    action recordDeposit(account, amount) {
        // Creates new ledger entry, old entries remain unchanged
        transaction_history = {
            account: account,
            amount: amount,
            type: "deposit",
            timestamp: TX.timestamp,
            tx_hash: hash(account + amount, "SHA256")
        };
        
        print("Deposit recorded in immutable ledger");
    }
}
```

### The `state` Keyword

**State** declares mutable variables within a contract (still subject to transaction rollback).

#### Syntax

```zexus
state variable_name;
state variable_name = initial_value;
```

#### Example

```zexus
contract Counter {
    state count = 0;
    state owner;
    
    action init() {
        owner = TX.caller;
    }
    
    action increment() {
        require(TX.caller == owner, "Only owner can increment");
        count = count + 1;
    }
}
```

### `ledger` vs `state`

| Feature | `ledger` | `state` |
|---------|----------|---------|
| Mutability | Immutable (append-only) | Mutable |
| History | Full version history | Current value only |
| Use Case | Transaction logs, audit trails | Balances, counters, flags |
| Gas Cost | Higher (versioning overhead) | Lower |

---

## Transaction Context

### The `TX` Object

The **TX object** provides immutable information about the current transaction execution.

#### Properties

```zexus
TX.caller         // Address/ID of the entity executing the code
TX.timestamp      // Canonical, un-tamperable time of execution
TX.block_hash     // Cryptographic reference to the preceding state
TX.gas_limit      // Maximum gas allowed for this transaction
TX.gas_used       // Gas consumed so far
TX.gas_remaining  // Remaining gas for execution
```

#### Example: Access Control

```zexus
contract Ownable {
    state owner;
    
    action init() {
        owner = TX.caller;
        print("Contract initialized by: " + owner);
    }
    
    action restrictedFunction() {
        require(TX.caller == owner, "Only owner can call this");
        print("Owner-only function executed");
    }
}
```

#### Example: Time-based Logic

```zexus
contract TimeLock {
    state unlock_time;
    state locked_value;
    
    action lock(value, duration) {
        locked_value = value;
        unlock_time = TX.timestamp + duration;
        print("Value locked until: " + unlock_time);
    }
    
    action unlock() {
        require(TX.timestamp >= unlock_time, "Still locked");
        print("Value unlocked: " + locked_value);
        return locked_value;
    }
}
```

---

## Cryptographic Primitives

### Hashing

**`hash(data, algorithm)`** - Cryptographic hashing

#### Supported Algorithms

- `SHA256` - Standard SHA-256
- `SHA512` - SHA-512  
- `SHA3-256` - SHA-3 (256-bit)
- `SHA3-512` - SHA-3 (512-bit)
- `KECCAK256` - Ethereum-style Keccak-256
- `BLAKE2B` - BLAKE2b
- `BLAKE2S` - BLAKE2s

#### Examples

```zexus
// SHA256 hash
let hash1 = hash("Hello, Blockchain!", "SHA256");
print("SHA256: " + hash1);

// Keccak256 (Ethereum style)
let hash2 = keccak256("transaction_data");
print("Keccak256: " + hash2);  // Returns with '0x' prefix

// Hash transaction data
let tx_data = sender + recipient + amount;
let tx_hash = hash(tx_data, "SHA256");
```

### Digital Signatures

#### Generate Keypair

```zexus
let keypair = generateKeypair("ECDSA");
// Returns: { private_key: "...", public_key: "..." }

print("Private key: " + keypair.private_key);
print("Public key: " + keypair.public_key);
```

#### Sign Data

```zexus
let message = "Transfer 100 tokens to Alice";
let signature = sign(message, keypair.private_key, "ECDSA");
print("Signature: " + signature);
```

#### Verify Signature

```zexus
let is_valid = verify_sig(message, signature, keypair.public_key, "ECDSA");

if (is_valid) {
    print("✓ Signature is valid!");
} else {
    print("❌ Signature is invalid!");
}
```

#### Complete Example: Signed Transactions

```zexus
contract SignedTransfer {
    state balances = {};
    state nonces = {};
    
    action transfer(recipient, amount, signature, signer_pubkey) {
        // Construct message
        let nonce = nonces[TX.caller];
        let message = TX.caller + recipient + amount + nonce;
        
        // Verify signature
        require(verify_sig(message, signature, signer_pubkey), "Invalid signature");
        
        // Verify signer is the caller
        let signer_address = deriveAddress(signer_pubkey);
        require(signer_address == TX.caller, "Signature doesn't match caller");
        
        // Execute transfer
        require(balances[TX.caller] >= amount, "Insufficient balance");
        balances[TX.caller] = balances[TX.caller] - amount;
        balances[recipient] = balances[recipient] + amount;
        
        // Increment nonce (prevent replay attacks)
        nonces[TX.caller] = nonce + 1;
        
        print("Signed transfer successful");
    }
}
```

### Address Derivation

```zexus
let keypair = generateKeypair("ECDSA");
let address = deriveAddress(keypair.public_key);
print("Ethereum-style address: " + address);  // 0x...
```

### Random Number Generation

```zexus
let random = randomBytes(32);
print("Random bytes (hex): " + random);
```

---

## Resource Management

### The `limit` Keyword

**`limit`** sets a gas/resource limit for an action to prevent DoS attacks.

#### Syntax

```zexus
action functionName(params) limit gas_amount {
    // function body
}
```

#### Gas Cost Model

| Operation | Gas Cost |
|-----------|----------|
| Base transaction | 21,000 |
| Storage write | 20,000 |
| Storage read | 200 |
| Addition/Subtraction | 3 |
| Multiplication/Division | 5 |
| Hash (SHA256) | 60 |
| Signature creation | 3,000 |
| Signature verification | 3,000 |
| Function call | 100 |

#### Examples

```zexus
// Simple transfer - low gas
action transfer(recipient, amount) limit 50000 {
    require(balances[TX.caller] >= amount, "Insufficient balance");
    balances[TX.caller] = balances[TX.caller] - amount;
    balances[recipient] = balances[recipient] + amount;
}

// Complex operation - higher gas
action swap(token_a, token_b, amount) limit 150000 {
    // Multi-step token swap
    // Requires more gas due to multiple storage operations
}

// Very expensive operation
action massUpdate(accounts) limit 1000000 {
    // Updates many accounts
    // High gas limit to accommodate loops
}
```

### The `require` Statement

**`require`** checks a condition and reverts the transaction if false.

#### Syntax

```zexus
require(condition);
require(condition, "Error message");

// Enhanced: Tolerance blocks for conditional bypasses
require condition {
    // Tolerance logic: return truthy to bypass requirement
}
```

#### Examples

```zexus
// Check balance
require(balance >= amount, "Insufficient funds");

// Check ownership
require(TX.caller == owner, "Only owner allowed");

// Check positive amount
require(amount > 0, "Amount must be positive");

// Check time
require(TX.timestamp >= unlock_time, "Still locked");

// Check address
require(recipient != TX.caller, "Cannot send to self");

// VIP bypass for minimum balance
require balance >= 0.5 {
    if (isVIP(TX.caller)) return true;
}

// Loyalty points waive minimum purchase
require amount >= 100 {
    if (loyaltyPoints[TX.caller] >= 1000) return true;
}
```

### The `revert` Statement

**`revert`** manually triggers a transaction rollback.

#### Syntax

```zexus
revert();
revert("Reason for revert");
```

#### Example

```zexus
action conditionalTransfer(recipient, amount) {
    if (balances[TX.caller] < amount) {
        revert("Insufficient balance for transfer");
    }
    
    if (amount > 1000000) {
        revert("Amount exceeds maximum transfer limit");
    }
    
    // Proceed with transfer
    balances[TX.caller] = balances[TX.caller] - amount;
    balances[recipient] = balances[recipient] + amount;
}
```

---

## Smart Contract Examples

### ERC20 Token Contract

See [`examples/token_contract.zx`](../examples/token_contract.zx) for a complete implementation with:

- ✅ Balance tracking
- ✅ Transfer functionality
- ✅ Approve/TransferFrom pattern
- ✅ Mint/Burn capabilities
- ✅ Pause/Unpause mechanism
- ✅ Ownership transfer
- ✅ Full gas optimization
- ✅ Immutable ledger for audit trail

### Multi-Signature Wallet

```zexus
contract MultiSigWallet {
    state owners = [];
    state required_confirmations;
    state transactions = {};
    state confirmations = {};
    state transaction_count = 0;
    
    action init(owner_list, required) {
        require(owner_list.length > 0, "Must have at least one owner");
        require(required > 0 && required <= owner_list.length, "Invalid required confirmations");
        
        owners = owner_list;
        required_confirmations = required;
    }
    
    action submitTransaction(recipient, amount) limit 100000 {
        require(isOwner(TX.caller), "Only owners can submit");
        
        let tx_id = transaction_count;
        transactions[tx_id] = {
            recipient: recipient,
            amount: amount,
            executed: false
        };
        
        transaction_count = transaction_count + 1;
        confirmations[tx_id] = [TX.caller];
        
        print("Transaction " + tx_id + " submitted");
        return tx_id;
    }
    
    action confirmTransaction(tx_id) limit 80000 {
        require(isOwner(TX.caller), "Only owners can confirm");
        require(transactions[tx_id], "Transaction doesn't exist");
        require(!transactions[tx_id].executed, "Already executed");
        require(!hasConfirmed(tx_id, TX.caller), "Already confirmed");
        
        confirmations[tx_id].push(TX.caller);
        
        if (confirmations[tx_id].length >= required_confirmations) {
            executeTransaction(tx_id);
        }
    }
    
    action executeTransaction(tx_id) limit 150000 {
        let tx = transactions[tx_id];
        require(!tx.executed, "Already executed");
        require(confirmations[tx_id].length >= required_confirmations, "Not enough confirmations");
        
        // Execute transfer
        balances[tx.recipient] = balances[tx.recipient] + tx.amount;
        tx.executed = true;
        
        print("Transaction " + tx_id + " executed");
    }
    
    action pure isOwner(address) {
        for (let i = 0; i < owners.length; i = i + 1) {
            if (owners[i] == address) {
                return true;
            }
        }
        return false;
    }
    
    action pure hasConfirmed(tx_id, address) {
        let confs = confirmations[tx_id];
        for (let i = 0; i < confs.length; i = i + 1) {
            if (confs[i] == address) {
                return true;
            }
        }
        return false;
    }
}
```

### Staking Contract

```zexus
contract Staking {
    state stakes = {};
    state total_staked = 0;
    state reward_rate = 5;  // 5% annual
    state minimum_stake = 100;
    
    action stake(amount) limit 60000 {
        require(amount >= minimum_stake, "Below minimum stake");
        require(balances[TX.caller] >= amount, "Insufficient balance");
        
        // Transfer tokens to contract
        balances[TX.caller] = balances[TX.caller] - amount;
        
        // Record stake
        stakes[TX.caller] = {
            amount: amount,
            start_time: TX.timestamp,
            rewards_claimed: 0
        };
        
        total_staked = total_staked + amount;
        print("Staked " + amount + " tokens");
    }
    
    action unstake() limit 80000 {
        require(stakes[TX.caller], "No active stake");
        
        let stake = stakes[TX.caller];
        let rewards = calculateRewards(TX.caller);
        let total = stake.amount + rewards;
        
        // Return stake + rewards
        balances[TX.caller] = balances[TX.caller] + total;
        total_staked = total_staked - stake.amount;
        
        // Remove stake
        stakes[TX.caller] = null;
        
        print("Unstaked " + stake.amount + " + " + rewards + " rewards");
    }
    
    action pure calculateRewards(staker) {
        let stake = stakes[staker];
        if (!stake) return 0;
        
        let duration = TX.timestamp - stake.start_time;
        let seconds_per_year = 31536000;
        let rewards = (stake.amount * reward_rate * duration) / (100 * seconds_per_year);
        
        return rewards - stake.rewards_claimed;
    }
}
```

---

## Best Practices

### 1. **Use `require` for All Validations**

```zexus
// ❌ Bad
if (amount <= 0) {
    return false;
}

// ✅ Good
require(amount > 0, "Amount must be positive");
```

### 2. **Check Effects Interactions Pattern**

```zexus
action transfer(recipient, amount) {
    // 1. Checks
    require(balances[TX.caller] >= amount, "Insufficient balance");
    require(amount > 0, "Amount must be positive");
    
    // 2. Effects
    balances[TX.caller] = balances[TX.caller] - amount;
    balances[recipient] = balances[recipient] + amount;
    
    // 3. Interactions (external calls would go here)
}
```

### 3. **Use Immutable Ledger for Critical Data**

```zexus
// ❌ Bad - can lose history
state transaction_log;

// ✅ Good - immutable audit trail
ledger transaction_history;
```

### 4. **Set Appropriate Gas Limits**

```zexus
// Simple operations
action transfer() limit 50000 { }

// Complex operations
action multiSwap() limit 200000 { }

// Very complex operations
action batchProcess() limit 1000000 { }
```

### 5. **Implement Access Control**

```zexus
state owner;
state admins = {};

function secure onlyOwner() {
    require(TX.caller == owner, "Only owner");
}

function secure onlyAdmin() {
    require(admins[TX.caller], "Only admin");
}
```

### 6. **Prevent Reentrancy**

```zexus
state locked = false;

action nonReentrant transfer(recipient, amount) {
    require(!locked, "Reentrancy detected");
    locked = true;
    
    // Transfer logic here
    
    locked = false;
}
```

### 7. **Use Events for External Monitoring**

```zexus
action transfer(recipient, amount) {
    // ... transfer logic ...
    
    // Emit event for off-chain indexers
    ledger transfer_event = {
        from: TX.caller,
        to: recipient,
        amount: amount,
        timestamp: TX.timestamp
    };
}
```

---

## API Reference

### Keywords

#### `ledger`
```zexus
ledger variable_name;
ledger variable_name = initial_value;
```
Declares an immutable, versioned state variable.

#### `state`
```zexus
state variable_name;
state variable_name = initial_value;
```
Declares a mutable contract state variable.

#### `contract`
```zexus
contract ContractName {
    // state variables
    // actions
}
```
Defines a smart contract.

#### `limit`
```zexus
action functionName() limit gas_amount {
    // body
}
```
Sets gas limit for an action.

#### `require`
```zexus
require(condition);
require(condition, "Error message");
```
Asserts condition, reverts if false.

#### `revert`
```zexus
revert();
revert("Reason");
```
Manually reverts transaction.

### Built-in Functions

#### `hash(data, algorithm)`
```zexus
let hash = hash("data", "SHA256");
```
Cryptographic hashing. Algorithms: SHA256, SHA512, KECCAK256, etc.

#### `keccak256(data)`
```zexus
let hash = keccak256("data");  // Returns 0x...
```
Ethereum-style Keccak-256 hash with '0x' prefix.

#### `generateKeypair(algorithm)`
```zexus
let keypair = generateKeypair("ECDSA");
// Returns: {private_key: "...", public_key: "..."}
```
Generates cryptographic keypair.

#### `sign(data, private_key, algorithm)`
```zexus
let signature = sign(message, private_key, "ECDSA");
```
Creates digital signature.

#### `verify_sig(data, signature, public_key, algorithm)`
```zexus
let is_valid = verify_sig(message, sig, public_key, "ECDSA");
```
Verifies digital signature.

#### `deriveAddress(public_key)`
```zexus
let address = deriveAddress(public_key);  // Returns 0x...
```
Derives Ethereum-style address from public key.

#### `randomBytes(length)`
```zexus
let random = randomBytes(32);
```
Generates cryptographically secure random bytes.

### TX Object Properties

```zexus
TX.caller         // string: Transaction caller address
TX.timestamp      // number: Transaction timestamp
TX.block_hash     // string: Block hash reference
TX.gas_limit      // number: Maximum gas for transaction
TX.gas_used       // number: Gas consumed so far
TX.gas_remaining  // number: Remaining gas
```

---

## Migration Guide

### From Traditional to Blockchain Zexus

#### Before (Traditional)
```zexus
let balances = {};
let owner = "admin";

action transfer(recipient, amount) {
    if (balances[sender] < amount) {
        print("Insufficient balance");
        return false;
    }
    balances[sender] = balances[sender] - amount;
    balances[recipient] = balances[recipient] + amount;
    return true;
}
```

#### After (Blockchain)
```zexus
contract Token {
    state balances = {};
    ledger transaction_history;
    state owner;
    
    action init() {
        owner = TX.caller;
    }
    
    action transfer(recipient, amount) limit 50000 {
        require(balances[TX.caller] >= amount, "Insufficient balance");
        
        balances[TX.caller] = balances[TX.caller] - amount;
        balances[recipient] = balances[recipient] + amount;
        
        transaction_history = {
            from: TX.caller,
            to: recipient,
            amount: amount,
            timestamp: TX.timestamp
        };
    }
}
```

**Key Changes:**
- ✅ `state` instead of `let` for contract state
- ✅ `ledger` for immutable audit trail
- ✅ `require` for validation (instead of if/return)
- ✅ `limit` for gas tracking
- ✅ `TX.caller` for access control
- ✅ `contract` wrapper for organization

---

## Conclusion

Zexus blockchain features provide **production-ready smart contract development** with:

- ✅ **Security**: Built-in require, revert, gas limits
- ✅ **Immutability**: Ledger system prevents data tampering
- ✅ **Auditability**: Complete transaction history
- ✅ **Cryptography**: Industry-standard hashing and signatures
- ✅ **Performance**: Gas tracking prevents DoS
- ✅ **Simplicity**: Clean syntax, familiar patterns

**Next Steps:**
1. Review the [Token Contract Example](../examples/token_contract.zx)
2. Run the [Blockchain Test Suite](../src/tests/test_blockchain_features.zx)
3. Build your own smart contracts!

For questions or contributions, see the main [README.md](../README.md).
