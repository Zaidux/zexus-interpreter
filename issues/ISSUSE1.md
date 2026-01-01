# Zexus Blockchain Testing - Issues Report

**Date**: January 1, 2026  
**Zexus Version**: 1.6.3  
**Project**: Ziver Chain Blockchain  
**Tester**: GitHub Copilot AI Assistant

---

## 🎯 Executive Summary

This document details all issues encountered while testing Zexus blockchain functionality. These issues prevent proper blockchain implementation and need to be addressed in the Zexus interpreter.

**Total Issues Found**: 6 major issues across 4 test files

---

## 📋 Test Files & Issues

### 1. File: `test_blockchain.zx`

**Status**: ❌ Failed  
**Location**: `/workspaces/Ziver-Chain/test_blockchain.zx`

#### Issue 1.1: Entity Property Access Returns NULL

**Code:**
```zexus
entity Block {
    index: integer
    timestamp: integer
    previous_hash: string
    hash: string
    transactions: list
    validator: string
}

action create_simple_genesis() -> Block {
    let genesis = Block{
        index: 0,
        timestamp: 1735747200,
        previous_hash: "0",
        hash: "genesis_hash_000",
        transactions: [],
        validator: "system"
    }
    return genesis
}

let genesis_block = create_simple_genesis()
print("Genesis Block Index: " + string(genesis_block.index))
```

**Error:**
```
Genesis Block Index: null

ERROR: ZexusError
  → <runtime>

  Type mismatch: cannot add STRING and NULL
  Use explicit conversion: string(value) to convert to string before concatenation
```

**Problem**: When accessing entity properties using dot notation (e.g., `genesis_block.index`), the value returns `null` instead of the actual value (0).

**Expected Behavior**: `genesis_block.index` should return `0`, not `null`.

**Workaround Attempted**: Switching from `entity` to `data` type, but same issue persists.

---

#### Issue 1.2: Contract State Assignment Fails

**Code:**
```zexus
contract SimpleToken {
    state total_supply = 1000000
    state balances = {}
    
    action init() {
        balances["genesis"] = total_supply
        print("Token initialized with supply: " + string(total_supply))
    }
}

let token = SimpleToken()
token.init()
```

**Error:**
```
📄 SmartContract.instantiate() called for: SimpleToken
🔗 Contract Address: 1bb0f537-ac72-4e
Available actions: ['init', 'transfer']

✅ Result: ❌ Error: Assignment to property failed
```

**Problem**: Cannot assign values to contract state variables from within action methods.

**Expected Behavior**: `balances["genesis"] = total_supply` should successfully assign the value to the state map.

---

### 2. File: `test_simple_blockchain.zx`

**Status**: ❌ Failed  
**Location**: `/workspaces/Ziver-Chain/test_simple_blockchain.zx`

#### Issue 2.1: len() Not Supported for MAP Type

**Code:**
```zexus
let validators = {}
validators["v1"] = 1000
validators["v2"] = 2000

print("Validators count: " + string(len(validators)))
```

**Error:**
```
ERROR: ZexusError
  → <runtime>

  len() not supported for MAP
```

**Problem**: The `len()` function doesn't work with MAP/dictionary types, only with lists.

**Expected Behavior**: `len()` should return the number of keys in a map, similar to Python's `len(dict)`.

**Impact**: Cannot get the count of validators, balances, or any other map-based structures without manually counting.

---

#### Issue 2.2: Contract Persistent State Assignment

**Code:**
```zexus
contract TokenContract {
    persistent state owner = null
    persistent state total_tokens = 0
    persistent state token_balances = {}
    
    action initialize(initial_supply, owner_address) {
        this.owner = owner_address
        this.total_tokens = initial_supply
        this.token_balances = {}
        this.token_balances[owner_address] = initial_supply
        return true
    }
}
```

**Error:**
```
✅ Result: ❌ Error: Assignment to property failed
```

**Problem**: Cannot assign to contract state variables using `this.property = value` syntax.

**Expected Behavior**: State variables should be assignable from within contract actions.

**Note**: Attempted both `this.property` and direct `property` assignment - both fail.

---

### 3. File: `test_full_blockchain.zx`

**Status**: ❌ Failed  
**Location**: `/workspaces/Ziver-Chain/test_full_blockchain.zx`

#### Issue 3.1: Contract State Variable Comparison Type Error

**Code:**
```zexus
contract Consensus {
    state validators = {}
    state total_stake = 0
    state min_stake = 1000
    
    action register_validator(address, stake) {
        require(stake >= min_stake, "Stake too low")
        // ...
    }
}
```

**Error:**
```
✅ Result: ❌ Error: Type error: cannot compare INTEGER >= STRING
Use explicit conversion if needed: int(value) or float(value)
```

**Problem**: State variable `min_stake` initialized as integer (1000) is somehow being treated as STRING when used in comparisons.

**Expected Behavior**: Integer state variables should maintain their type and be comparable to integer parameters.

**Workaround Used**: Changed to hardcoded literal `require(stake >= 1000, ...)` instead of using state variable.

---

#### Issue 3.2: Cannot Assign Map of Maps in Contract State

**Code:**
```zexus
contract Consensus {
    state validators = {}
    
    action register_validator(address, stake) {
        validators[address] = {
            "stake": stake,
            "active": true,
            "blocks_validated": 0
        }
    }
}
```

**Error:**
```
✅ Result: ❌ Error: Assignment to property failed
```

**Problem**: Cannot assign a map (object) as a value in a contract state map. Nested structures fail.

**Expected Behavior**: Should be able to store complex objects in state maps.

**Workaround Used**: Simplified to `validators[address] = stake` (just storing the integer).

---

### 4. File: `test_working_blockchain.zx`

**Status**: ⚠️ Partially Working  
**Location**: `/workspaces/Ziver-Chain/test_working_blockchain.zx`

#### Issue 4.1: Map State Not Persisting Across Function Calls

**Code:**
```zexus
let balances = {"alice": 1000000}

action transfer(from_addr, to_addr, amount) {
    balances[from_addr] = balances[from_addr] - amount
    
    let recipient_balance = 0
    if balances[to_addr] != null {
        recipient_balance = balances[to_addr]
    }
    balances[to_addr] = recipient_balance + amount
    
    print("  📤 Transferred: " + string(amount) + " tokens")
    return true
}

action get_balance(addr) {
    if balances[addr] != null {
        return balances[addr]
    }
    return 0
}

# Initial: alice = 1000000
transfer("alice", "bob", 50000)  # Should: alice = 950000, bob = 50000
print("Alice: " + string(get_balance("alice")))  # Actual: 1000000 (unchanged!)
print("Bob: " + string(get_balance("bob")))      # Actual: 0 (unchanged!)
```

**Output:**
```
💰 Initial balance (Alice): 1000000
📤 Transferred: 50000 tokens
   From: alice → To: bob
📤 Transferred: 25000 tokens
   From: alice → To: charlie
💰 Alice balance: 1000000
💰 Bob balance: 0
💰 Charlie balance: 0
```

**Problem**: 
1. The `transfer()` function executes without errors
2. The print statements inside `transfer()` work
3. BUT the changes to the `balances` map are NOT persisted
4. When `get_balance()` is called afterward, it returns the original values

**Expected Behavior**: Map modifications in one function should persist and be visible to subsequent function calls.

**Severity**: 🔴 CRITICAL - This breaks all blockchain functionality. Tokens can't transfer, balances can't update, state can't change.

**Similar Issues in Production Files**:
This same issue likely affects:
- `/workspaces/Ziver-Chain/src/core/block.zx` - Block storage
- `/workspaces/Ziver-Chain/src/core/consensus.zx` - Validator tracking
- `/workspaces/Ziver-Chain/src/core/state.zx` - Network parameters

---

## 📊 Issue Categories & Severity

### 🔴 Critical Issues (Blockers)
1. **Map state persistence failure** - Core blockchain functionality broken
2. **Contract state assignment failure** - Smart contracts unusable
3. **Entity property access returns null** - Data structures broken

### 🟡 High Priority Issues
4. **len() not supported for maps** - Missing basic functionality
5. **State variable type conversion** - Unexpected type behavior
6. **Nested map assignment in contracts** - Limits data complexity

---

## 🔧 Recommended Fixes

### For Zexus Interpreter Developers:

1. **Map Mutation Persistence** (Issue 4.1)
   - Ensure map mutations inside functions persist to the outer scope
   - Maps should be passed by reference, not copied
   - Test with nested function calls

2. **Contract State Management** (Issues 1.2, 2.2, 3.2)
   - Implement proper state variable assignment in contract actions
   - Support both `this.property` and direct `property` access
   - Allow complex nested structures in state

3. **Entity/Data Property Access** (Issue 1.1)
   - Fix dot notation property access for entity/data types
   - Ensure properties return their actual values, not null
   - Test with various data types (integer, string, list, map)

4. **Map Built-in Functions** (Issue 2.1)
   - Implement `len()` for MAP types
   - Consider adding: `keys()`, `values()`, `items()` for maps
   - Document map iteration patterns

5. **Type Inference for State Variables** (Issue 3.1)
   - Preserve type information for contract state variables
   - Don't convert integers to strings implicitly
   - Maintain type consistency across function calls

---

## 🧪 Reproduction Steps

All test files are available in `/workspaces/Ziver-Chain/`:
- `test_blockchain.zx` - Entity/data issues
- `test_simple_blockchain.zx` - Map and contract issues
- `test_full_blockchain.zx` - Contract state issues
- `test_working_blockchain.zx` - Persistence issues

**To reproduce any issue**:
```bash
cd /workspaces/Ziver-Chain
zx run <test_file_name>.zx
```

---

## 📝 Additional Notes

### What DOES Work:
✅ Basic variables and arithmetic  
✅ String concatenation  
✅ Lists (arrays) - access, push, len()  
✅ If/else conditions  
✅ For loops  
✅ Action definitions  
✅ Print statements  
✅ Require statements (validation)  
✅ Audit function (logging)  

### What DOESN'T Work:
❌ Contract state mutations  
❌ Entity/Data property access  
❌ Map state persistence across functions  
❌ len() on maps  
❌ Complex nested structures in contracts  
❌ State variable type preservation  

---

## 💡 Workarounds (Temporary)

Until these issues are fixed in Zexus:

1. **For balances/state**: Use global lists instead of maps where possible
2. **For counting maps**: Maintain a separate counter variable
3. **For contracts**: Avoid using state variables; pass everything as parameters
4. **For entities**: Use map literals `{}` instead of entity types
5. **For persistence**: Use `persist_set()` and `persist_get()` built-in functions

---

## 🎯 Impact on Ziver Chain Project

These issues currently prevent:
- ✗ Token transfers
- ✗ Validator registration
- ✗ Block storage
- ✗ Smart contract execution
- ✗ Account balance tracking
- ✗ Any stateful blockchain operations

**Estimated Fix Priority**: These should be P0 bugs in the Zexus interpreter.

---

## 📞 Contact

If you need more details or reproduction steps for any issue:
- Repository: https://github.com/ZiverLabs/Ziver-Chain
- Test files: `/workspaces/Ziver-Chain/test_*.zx`
- Core files: `/workspaces/Ziver-Chain/src/core/*.zx`

---

**Report Generated**: January 1, 2026  
**Zexus Version Tested**: 1.6.3  
**Python Version**: 3.12  
**Installation Method**: `pip install zexus`
