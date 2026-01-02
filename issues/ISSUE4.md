# Zexus 1.6.6 - Known Limitations & Workarounds

**Version:** Zexus 1.6.6  
**Date:** January 2, 2026  
**Status:** 🚨 CRITICAL - Affects Production Code

---

## Table of Contents

1. [Module System Limitations](#module-system-limitations)
2. [Nested Map Update Limitations](#nested-map-update-limitations)
3. [Reserved Keywords](#reserved-keywords)
4. [Impact on Production Files](#impact-on-production-files)
5. [Workarounds](#workarounds)

---

## 1. Module System Limitations

### 🚨 Issue: Named Imports Not Supported

**Status:** ❌ BROKEN  
**Severity:** Medium  
**Test File:** [test_core_imports_real.zx](../test_core_imports_real.zx)

### What Doesn't Work:

```zexus
// ❌ FAILS - Named imports not supported
use { Block, Transaction } from "./src/core/block.zx"
use { QuantumCrypto } from "./src/core/crypto.zx"
use { PoSConsensus } from "./src/core/consensus.zx"
```

**Error Message:**
```
ERROR: ZexusError
'Block' is not exported from ./src/core/block.zx
```

**Even though the file contains:**
```zexus
export {
    Block,
    Transaction,
    create_genesis_block,
    // ... other exports
}
```

### ✅ Workaround:

Use direct file inclusion instead:

```zexus
// ✅ WORKS - Direct file inclusion
use "./src/core/block.zx"
use "./src/core/crypto.zx"
use "./src/core/consensus.zx"
```

### Impact:

- **Low** - Workaround is simple and works perfectly
- All exported items become available in the importing file's scope
- No functional impact, just less granular imports

### Test Evidence:

From [../test_core_imports_real.zx](../test_core_imports_real.zx) lines 9-20:

```zexus
try {
    use { Block, Transaction, BlockHeader, create_genesis_block } from "./src/core/block.zx"
    print("  ✅ block.zx imported successfully")
} catch error {
    print("  ❌ block.zx import failed: " + string(error))
}
// Result: ❌ 'Block' is not exported from ./src/core/block.zx
```

From [../test_direct_load.zx](../test_direct_load.zx):

```zexus
use "./src/core/block.zx"
print("✅ block.zx loaded successfully!")
// Result: ✅ Works perfectly
```

---

## 2. Nested Map Update Limitations

### 🚨 Issue: Multiple Patterns for Updating Nested Map Values Are Broken

**Status:** ❌ CRITICAL  
**Severity:** HIGH - Breaks production code  
**Test Files:** 
- [../test_comprehensive_patterns.zx](../test_comprehensive_patterns.zx)
- [../test_definitive.zx](../test_definitive.zx)
- [../test_inline_reconstruction.zx](../test_inline_reconstruction.zx)
- [../test_reconstruction_module.zx](../test_reconstruction_module.zx)

---

### ❌ Pattern 1: Direct Property Assignment (Dot Notation)

**Status:** FAILS with parser error

```zexus
let validators = {}
validators["val_001"] = {"stake": 10000, "performance": 1.0}

// ❌ FAILS
validators["val_001"].performance = validators["val_001"].performance + 0.1
```

**Error:**
```
Invalid assignment target
```

**Production Code Affected:**

From [src/core/consensus.zx](src/core/consensus.zx) line 269:
```zexus
this.validators[address].last_active = datetime.now().timestamp()
```

From [src/core/consensus.zx](src/core/consensus.zx) lines 272-274:
```zexus
this.validators[address].performance = math.min(
    1.0, 
    this.validators[address].performance + performance_boost
)
```

**Test Evidence:** [test_comprehensive_patterns.zx](test_comprehensive_patterns.zx) lines 17-27

---

### ❌ Pattern 2: Direct Property Assignment (Bracket Notation)

**Status:** FAILS with parser error

```zexus
let validators = {}
validators["val_001"] = {"stake": 10000, "performance": 1.0}

// ❌ FAILS
validators["val_001"]["performance"] = validators["val_001"]["performance"] + 0.1
```

**Error:**
```
Invalid assignment target
```

**Test Evidence:** [test_comprehensive_patterns.zx](test_comprehensive_patterns.zx) lines 29-39

---

### ❌ Pattern 3: Temp Variable + Modify + Reassign

**Status:** FAILS - temp variable not created

```zexus
let validators = {}
validators["val_001"] = {"stake": 10000, "performance": 1.0}

// ❌ FAILS - 'temp' is never created
let temp = validators["val_001"]
temp["performance"] = temp["performance"] + 0.1
validators["val_001"] = temp
```

**Error:**
```
Identifier 'temp' not found
```

**Why This Fails:**
The statement `let temp = validators["val_001"]` does not create the variable `temp`. The Zexus parser/evaluator fails to assign nested map values to local variables.

**Test Evidence:** 
- [test_comprehensive_patterns.zx](test_comprehensive_patterns.zx) lines 41-53
- [test_reconstruction_module.zx](test_reconstruction_module.zx) lines 22-30

From [test_reconstruction_module.zx](test_reconstruction_module.zx):
```zexus
let validators = {}
validators["val_001"] = {"stake": 10000, "performance": 1.0, "rewards": 0}

let old_val = validators["val_001"]  // This line fails silently
validators["val_001"] = {
    "stake": old_val["stake"],  // Error: Identifier 'old_val' not found
    "performance": old_val["performance"] + 0.1,
    "rewards": old_val["rewards"] + 100
}
```

**Result:** `Identifier 'old_val' not found`

---

### ❌ Pattern 4: Compound Assignment (+=, -=, etc.)

**Status:** PARTIALLY WORKS but returns NULL values

```zexus
let validators = {}
validators["val_001"] = {"stake": 10000, "rewards": 0}

// ⚠️ Executes but returns NULL
validators["val_001"]["rewards"] += 100
// validators["val_001"]["rewards"] is now NULL, not 100
```

**Production Code Affected:**

From [src/core/consensus.zx](src/core/consensus.zx) line 412:
```zexus
this.validators[validator].total_rewards += final_reward
```

**Impact:** This will execute without error but `total_rewards` will become `null`.

---

### ⚠️ Pattern 5: Inline Reconstruction (Module-Level)

**Status:** ✅ WORKS at module level  
**Status:** ❌ BROKEN in contract state

#### ✅ Module Level - WORKS:

```zexus
// At module level (outside contracts)
let validators = {}
validators["val_001"] = {"stake": 10000, "rewards": 0, "count": 0}

// ✅ WORKS - Multiple sequential updates
validators["val_001"] = {
    "stake": validators["val_001"]["stake"],
    "rewards": validators["val_001"]["rewards"] + 100,
    "count": validators["val_001"]["count"] + 1
}

validators["val_001"] = {
    "stake": validators["val_001"]["stake"],
    "rewards": validators["val_001"]["rewards"] + 200,
    "count": validators["val_001"]["count"] + 1
}

validators["val_001"] = {
    "stake": validators["val_001"]["stake"],
    "rewards": validators["val_001"]["rewards"] + 300,
    "count": validators["val_001"]["count"] + 1
}

// Result: rewards = 600, count = 3 ✅ CORRECT
```

**Test Evidence:** [test_definitive.zx](test_definitive.zx) lines 9-42

**Output:**
```
Initial: stake=10000, rewards=0, count=0
Update 1: rewards=100, count=1
Update 2: rewards=300, count=2
Update 3: rewards=600, count=3
✅ Module-level multiple updates WORK!
```

#### ❌ Contract State - COMPLETELY BROKEN:

```zexus
contract ValidatorRegistry {
    state validators = {}
    
    action register(address, stake) {
        validators[address] = {"stake": stake, "rewards": 0, "count": 0}
    }
    
    // ❌ BROKEN - Returns NULL values
    action add_reward(address, amount) {
        validators[address] = {
            "stake": validators[address]["stake"],      // Returns NULL
            "rewards": validators[address]["rewards"] + amount,  // Returns NULL
            "count": validators[address]["count"] + 1   // Returns NULL
        }
    }
}

let registry = ValidatorRegistry()
registry.register("val_002", 15000)
registry.add_reward("val_002", 100)

let result = registry.get_validator("val_002")
// Result: stake=NULL, rewards=NULL, count=NULL ❌
```

**Test Evidence:** [test_definitive.zx](test_definitive.zx) lines 44-98

**Output:**
```
Initial: stake=null, rewards=null
After reward 1: rewards=null, count=null
After reward 2: rewards=null, count=null
After reward 3: rewards=null, count=null
❌ FAILED: Expected rewards=600, count=3, got rewards=null, count=null
```

**From [test_inline_reconstruction.zx](test_inline_reconstruction.zx) lines 54-87:**
```
✅ Test 1: Inline - access map multiple times in same statement
  Initial rewards: 0
  After update: 100
  ✅ Inline reconstruction WORKS!

✅ Test 2: Multiple inline updates
  Count: null, Total: null
  ❌ Failed - count=null, total=null

✅ Test 3: Contract using inline reconstruction
  Final value: null, count: null
  ❌ Failed - value=null, count=null
```

---

## 3. Reserved Keywords

### 🚨 Issue: Certain Keywords Cannot Be Used as Variable/Field Names

**Status:** ❌ BROKEN  
**Severity:** Medium

### Reserved Keywords:

1. **`storage`** - Cannot be used as variable or field name
2. **`data`** - Cannot be used as variable or field name

### ❌ What Doesn't Work:

```zexus
// ❌ FAILS
entity HealthComponents {
    consensus: ComponentHealth
    storage: ComponentHealth  // 'storage' is reserved keyword
}

// ❌ FAILS
let storage = {}  // Cannot use 'storage' as variable name
let data = {}     // Cannot use 'data' as variable name
```

### ✅ Workaround:

Use alternative names:

```zexus
// ✅ WORKS
entity HealthComponents {
    consensus: ComponentHealth
    data_storage: ComponentHealth  // Use compound name
}

let mystore = {}  // Use different name
let mydata = {}   // Use different name
```

### Production Code Affected:

From [src/core/state.zx](src/core/state.zx) line 113 (FIXED):

```diff
entity HealthComponents {
    consensus: ComponentHealth
    networking: ComponentHealth
-   storage: ComponentHealth
+   data_storage: ComponentHealth  // FIXED
    computation: ComponentHealth
    security: ComponentHealth
}
```

**Status:** ✅ Already fixed in production code

### Note on `persistent storage`:

The keyword pair `persistent storage` is **NOT** affected - it's a valid syntax for contract state:

```zexus
// ✅ WORKS - This is correct syntax
contract MyContract {
    persistent storage validators: map
    persistent storage total_stake: integer
}
```

The issue is only with using `storage` as a standalone variable/field name.

---

## 4. Impact on Production Files

### 🚨 Critical: [src/core/consensus.zx](src/core/consensus.zx)

**Lines Affected:** 50+ instances of nested map updates in contract state

#### Broken Functions:

1. **`update_validator_activity(address)`** - Lines 267-275
   ```zexus
   action update_validator_activity(address: string) {
       if this.validators[address] != null {
           // ❌ BROKEN - Direct property assignment
           this.validators[address].last_active = datetime.now().timestamp()
           
           // ❌ BROKEN - Nested calculation and assignment
           this.validators[address].performance = math.min(
               1.0, 
               this.validators[address].performance + performance_boost
           )
       }
   }
   ```
   **Impact:** Will fail with "Invalid assignment target"

2. **`process_block_reward(validator, block)`** - Line 412
   ```zexus
   // ❌ BROKEN - Compound assignment in contract state
   this.validators[validator].total_rewards += final_reward
   ```
   **Impact:** Will execute but set `total_rewards` to NULL

3. **Similar patterns in:**
   - `slash_validator()`
   - `update_validator_performance()`
   - `delegate_stake()`
   - `undelegate_stake()`
   - Any function that modifies nested validator properties

### Impact Summary:

| File | Contract | Broken Patterns | Functions Affected | Severity |
|------|----------|----------------|-------------------|----------|
| [consensus.zx](src/core/consensus.zx) | `PoSConsensus` | Direct assignment, compound ops | 10+ actions | 🚨 CRITICAL |
| [state.zx](src/core/state.zx) | `SelfEvolvingState` | Field name only | 1 entity | ✅ FIXED |
| [crypto.zx](src/core/crypto.zx) | `QuantumCrypto` | None detected | N/A | ✅ OK |
| [zvm.zx](src/core/zvm.zx) | `ZiverVirtualMachine` | None detected | N/A | ✅ OK |
| [block.zx](src/core/block.zx) | N/A | None detected | N/A | ✅ OK |

---

## 5. Workarounds

### For Module-Level Code:

#### ✅ Use Inline Reconstruction:

```zexus
// Module-level maps (outside contracts)
let validators = {}
validators["val_001"] = {"stake": 10000, "rewards": 0}

// ✅ WORKS - Inline reconstruction
validators["val_001"] = {
    "stake": validators["val_001"]["stake"],
    "rewards": validators["val_001"]["rewards"] + 100
}
```

**Test:** [test_definitive.zx](test_definitive.zx) lines 9-42 - ✅ VERIFIED WORKING

---

### For Contract State:

#### 🚨 NO WORKING WORKAROUND CURRENTLY

The inline reconstruction pattern that works at module level **FAILS in contract state** and returns NULL values.

**Test:** [test_definitive.zx](test_definitive.zx) lines 44-98 - ❌ VERIFIED BROKEN

#### ⚠️ Potential Workarounds (UNTESTED):

1. **Use Separate State Variables:**
   ```zexus
   contract ValidatorRegistry {
       state validator_stakes = {}
       state validator_rewards = {}
       state validator_counts = {}
       
       action add_reward(address, amount) {
           validator_rewards[address] = validator_rewards[address] + amount
           validator_counts[address] = validator_counts[address] + 1
       }
   }
   ```
   **Pros:** Avoids nested map updates  
   **Cons:** More complex, harder to maintain, loses data cohesion

2. **Store As String, Parse on Read:**
   ```zexus
   contract ValidatorRegistry {
       state validators = {}  // Store JSON strings
       
       action add_reward(address, amount) {
           // Would need JSON parsing/stringifying
           // Not tested - may not work
       }
   }
   ```
   **Pros:** Might avoid nested update issue  
   **Cons:** Performance overhead, complex implementation

3. **Wait for Zexus 1.6.7+ Fix:**
   - Report issue to Zexus maintainers
   - Wait for patch that fixes contract state map updates
   - This is the recommended approach

---

## Summary of All Limitations

### ❌ Completely Broken:

1. Named imports: `use { X } from "file"`
2. Direct property assignment in maps: `map[key].field = value`
3. Bracket property assignment in maps: `map[key]["field"] = value`
4. Temp variable from map: `let temp = map[key]` (variable not created)
5. **Contract state nested map updates** (returns NULL)

### ⚠️ Partially Working:

1. Compound assignment in maps: `map[key]["field"] += value` (executes but returns NULL)
2. Inline reconstruction at module level: Works ✅
3. Inline reconstruction in contracts: Broken ❌ (returns NULL)

### ✅ Working Workarounds:

1. Direct file inclusion: `use "./file.zx"` ✅
2. Avoid reserved keywords: Use `data_storage` instead of `storage` ✅
3. Module-level inline reconstruction ✅
4. Separate state variables in contracts (untested workaround)

---

## Recommendations

### Immediate Actions:

1. ✅ **Use direct file inclusion** for all imports
2. ✅ **Avoid reserved keywords** `storage` and `data` as variable names
3. ❌ **DO NOT use nested map updates in contract state** - they return NULL
4. ⚠️ **Refactor [consensus.zx](src/core/consensus.zx)** to avoid nested updates

### For Production Deployment:

1. **BLOCK deployment** until nested map updates in contracts are fixed OR
2. **Refactor affected contracts** to use separate state variables OR
3. **Report to Zexus team** and request urgent fix for version 1.6.7

### Testing:

All limitations documented here are **verified with test files**:

- ✅ [test_core_imports_real.zx](test_core_imports_real.zx) - Module import tests
- ✅ [test_comprehensive_patterns.zx](test_comprehensive_patterns.zx) - All pattern tests
- ✅ [test_definitive.zx](test_definitive.zx) - Definitive module vs contract tests
- ✅ [test_inline_reconstruction.zx](test_inline_reconstruction.zx) - Inline pattern tests
- ✅ [test_reconstruction_module.zx](test_reconstruction_module.zx) - Temp variable tests

---

## Conclusion

**Zexus 1.6.6 has critical limitations that prevent production deployment of the Ziver-Chain PoS consensus contract.**

The most critical issue is that **nested map updates in contract state return NULL values**, which breaks core validator management functionality in [src/core/consensus.zx](src/core/consensus.zx).

**Status:** 🚨 **DEPLOYMENT BLOCKED** until fix or refactoring completed.

---

*Document Version: 1.0*  
*Last Updated: January 2, 2026*  
*Zexus Version: 1.6.6*  
*Test Suite: Comprehensive (5 test files)*
