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

**Status:** ✅ FIXED (January 2, 2026)  
**Severity:** Medium  
**Test File:** [test_core_imports_real.zx](../test_core_imports_real.zx)

**Fix Summary:**
- Added EXPORT handler to context parser (`strategy_context.py`)
- Created `_parse_export_statement_block` method to properly parse export statements
- Export statements now correctly populate the module's exports dictionary
- Named imports now work as expected

### What Now Works:

```zexus
// ✅ WORKS - Named imports now supported
use { Block, Transaction } from "./src/core/block.zx"
use { QuantumCrypto } from "./src/core/crypto.zx"
use { PoSConsensus } from "./src/core/consensus.zx"
```

**Module exports:**
```zexus
export {
    Block,
    Transaction,
    create_genesis_block
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

From [test_core_imports_real.zx](../test_core_imports_real.zx) lines 9-20:

```zexus
try {
    use { Block, Transaction, BlockHeader, create_genesis_block } from "./src/core/block.zx"
    print("  ✅ block.zx imported successfully")
} catch error {
    print("  ❌ block.zx import failed: " + string(error))
}
// Result: ❌ 'Block' is not exported from ./src/core/block.zx
```

From [test_direct_load.zx](../test_direct_load.zx):

```zexus
use "./src/core/block.zx"
print("✅ block.zx loaded successfully!")
// Result: ✅ Works perfectly
```

---

## 2. Nested Map Update Limitations

### 🚨 Issue: Multiple Patterns for Updating Nested Map Values Are Broken

**Status:** ✅ ALL FIXED (January 2, 2026)  
**Severity:** ~~HIGH~~ → **RESOLVED**  
**Test Files:** 
- [test_comprehensive_patterns.zx](../test_comprehensive_patterns.zx)
- [test_definitive.zx](../test_definitive.zx)
- [test_inline_reconstruction.zx](../test_inline_reconstruction.zx)
- [test_reconstruction_module.zx](../test_reconstruction_module.zx)

**Fix Summary:**
- ✅ **Pattern 1 FIXED**: Direct property assignment (dot notation) now works
- ✅ **Pattern 2 FIXED**: Direct property assignment (bracket notation) now works
- ✅ **Pattern 3 FIXED**: Temp variable from map now works correctly
- ✅ **Pattern 4 FIXED**: Compound assignment (+=, -=, etc.) now works correctly
- ✅ **Pattern 5 FIXED**: Inline reconstruction works for both module and contract state
- ✅ **Pattern 6 FIXED**: Contract state map operations fully working

**Status Update (January 2, 2026):**
All 6 patterns have been fixed! The Zexus interpreter now fully supports nested map operations, compound assignments, and contract state maps.

---

### ✅ Pattern 1: Direct Property Assignment (Dot Notation)

**Status:** ✅ FIXED

```zexus
let validators = {}
validators["val_001"] = {"stake": 10000, "performance": 1.0}

// ✅ NOW WORKS
validators["val_001"].performance = validators["val_001"].performance + 0.1
```

---

### ✅ Pattern 2: Direct Property Assignment (Bracket Notation)

**Status:** ✅ FIXED

```zexus
let validators = {}
validators["val_001"] = {"stake": 10000, "performance": 1.0}

// ✅ NOW WORKS
validators["val_001"]["performance"] = validators["val_001"]["performance"] + 0.1
```

---

### ✅ Pattern 3: Temp Variable + Modify + Reassign

**Status:** ✅ FIXED

```zexus
let validators = {}
validators["val_001"] = {"stake": 10000, "performance": 1.0}

// ✅ NOW WORKS - temp variable is created correctly
let temp = validators["val_001"]
temp["performance"] = temp["performance"] + 0.1
validators["val_001"] = temp
```

---

### ✅ Pattern 4: Compound Assignment (+=, -=, etc.)

**Status:** ✅ FIXED (January 2, 2026) - Compound operators now work correctly

```zexus
let validators = {}
validators["val_001"] = {"stake": 10000, "rewards": 0}

// ✅ NOW WORKS - Adds correctly
validators["val_001"]["rewards"] += 100  // rewards = 100
validators["val_001"]["rewards"] += 200  // rewards = 300
validators["val_001"]["rewards"] += 300  // rewards = 600
// Expected: 600, Actual: 600 ✅ CORRECT
```

**Root Cause:**
The lexer tokenizes `+=` as two separate tokens: `[PLUS, ASSIGN]`. The parser was not recognizing this pattern as a compound assignment and was treating it as an invalid statement.

**Fix:**
Modified `_parse_assignment_statement` in `strategy_context.py` to:
1. Detect when an operator (`+`, `-`, `*`, `/`, `%`) appears immediately before `=`
2. Transform compound assignment into regular assignment with binary expression
3. Example: `x += 50` → `x = x + 50`

**Files Modified:**
- `src/zexus/parser/strategy_context.py`: Added compound operator detection and transformation

**Verification:**
```zexus
let x = 100
x += 50  // x = 150
x += 30  // x = 180

let map = {"rewards": 0}
map["rewards"] += 100  // 100
map["rewards"] += 200  // 300
map["rewards"] += 300  // 600 ✅ CORRECT
```

**Impact:**
- ✅ All compound assignment operators now work correctly
- ✅ Works with simple variables, nested maps, and property access
- ✅ [src/core/consensus.zx](../src/core/consensus.zx) - Validator reward accumulation now functional

---

### ✅ Pattern 5: Inline Reconstruction (Module-Level)

**Status:** ✅ WORKS at module level

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

// Can repeat multiple times successfully
validators["val_001"] = {
    "stake": validators["val_001"]["stake"],
    "rewards": validators["val_001"]["rewards"] + 200,
    "count": validators["val_001"]["count"] + 1
}
```

---

### ✅ Pattern 6: Contract State Map Operations - FIXED

**Status:** ✅ FIXED - Contract state map operations now working correctly

```zexus
contract ValidatorRegistry {
    state validators = {}  // Map state variable
    
    action register(address, stake) {
        // ✅ Now works correctly
        validators[address] = {"stake": stake}
    }
    
    action get_validator(address) {
        // ✅ Returns the map value correctly
        return validators[address]
    }
}
```

**Test Results:**
- ✅ Simple state variables work: `state x = 0; x = 42;` ✅ WORKS
- ✅ Map state assignment works: `state data = {}; data = {"x": 1};` ✅ FIXED
- ✅ Map state access works: `return data;` returns the map ✅ FIXED

**Root Causes Found & Fixed:**

1. **Parser Issue with DATA Keyword:**
   - `state data = {}` was not parsing the `data` variable because DATA is a keyword
   - Fixed in `strategy_context.py`: Added check to allow keywords as state variable names
   - State variables can now use reserved keywords like `data`, `verify`, etc.

2. **Assignment Statement Parsing:**
   - Action body parser was not recognizing assignment statements with keyword identifiers
   - Fixed in `_parse_block_statements`: DATA tokens followed by `=` now fall through to assignment parsing
   - Added check: `len(run_tokens) > 0` before breaking on statement starters

3. **Return Statement Parsing:**
   - `return data` was breaking because DATA is in statement_starters
   - Fixed: Don't break on statement starters when collecting the FIRST token (the return value)
   - Now correctly collects keyword identifiers as return values

4. **Expression Parsing:**
   - Keywords used as identifiers were being parsed as StringLiterals
   - Fixed in `_parse_single_token_expression`: Keywords with alphabetic literals now create Identifiers
   - Allows keywords to be used as variable names in expression contexts

5. **ReturnValue Unwrapping:**
   - Contract method calls were returning wrapped ReturnValue objects
   - Fixed in `functions.py`: Added unwrapping for contract `call_method` results
   - Return values are now properly extracted before being returned to caller

**Files Modified:**
- `src/zexus/parser/strategy_context.py`: Parser fixes for STATE, DATA, RETURN, and expression parsing
- `src/zexus/evaluator/functions.py`: Added ReturnValue unwrapping for contract calls

**Verification:**
```zexus
contract Test {
    state data = {}
    
    action test_reassign() {
        print("Before: " + string(data))  // Before: {}
        data = {"x": 1, "y": 2}
        print("After: " + string(data))   // After: {x: 1, y: 2}
    }
    
    action get() {
        return data  // Returns {x: 1, y: 2}
    }
}

let c = Test()
c.test_reassign()
print("Retrieved: " + string(c.get()))  // Retrieved: {x: 1, y: 2}
```

**Impact:**
- ✅ ALL contracts using map-based state now work correctly
- ✅ [src/core/consensus.zx](../src/core/consensus.zx) - Validator management now functional
- ✅ Production contracts using `state map_name = {}` pattern fully supported

---

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

### ✅ Pattern 5: Inline Reconstruction

**Status:** ✅ FIXED (January 2, 2026) - Works at both module and contract levels

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

#### ✅ Contract State - NOW FIXED:

```zexus
contract ValidatorRegistry {
    state validators = {}
    
    action register(address, stake) {
        validators[address] = {"stake": stake, "rewards": 0, "count": 0}
    }
    
    // ✅ NOW WORKS - Returns correct values
    action add_reward(address, amount) {
        validators[address] = {
            "stake": validators[address]["stake"],
            "rewards": validators[address]["rewards"] + amount,
            "count": validators[address]["count"] + 1
        }
    }
}

let registry = ValidatorRegistry()
registry.register("val_002", 15000)
registry.add_reward("val_002", 100)

let result = registry.get_validator("val_002")
// Result: stake=15000, rewards=100, count=1 ✅ CORRECT
```

**Fix:** Same parser fixes that resolved Pattern 6 (Contract State Map Operations) also fixed this pattern.
The ability to properly parse, evaluate, and persist contract state maps enables inline reconstruction to work correctly.

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

### ✅ FIXED (January 2, 2026):

1. ✅ Named imports: `use { X } from "file"` - NOW WORKS
2. ✅ Direct property assignment in maps: `map[key].field = value` - NOW WORKS  
3. ✅ Bracket property assignment in maps: `map[key]["field"] = value` - NOW WORKS
4. ✅ Temp variable from map: `let temp = map[key]` - NOW WORKS
5. ✅ Inline reconstruction at module level - WORKS PERFECTLY

### ❌ Still Broken:

1. ❌ **CRITICAL**: Contract state map operations (assignment fails silently)
2. ⚠️ Compound assignment returns wrong value (separate issue, not map-specific)

### ✅ Working Workarounds:

1. ✅ **Use direct file inclusion** for imports: `use "./file.zx"` 
2. ✅ **Avoid reserved keywords** `storage` and `data` as variable names
3. ✅ **Use module-level maps** instead of contract state maps
4. ✅ **Use simple state variables** in contracts (not maps)

---

## Recommendations

### Immediate Actions:

1. ✅ **Named imports are now supported** - can use selective imports
2. ✅ **Nested map updates work at module level** - use for non-contract code
3. ✅ **Avoid reserved keywords** `storage` and `data` as variable names
4. ❌ **DO NOT use map state variables in contracts** - they are completely broken

### For Production Deployment:

1. **BLOCK deployment** of any contract using `state map_name = {}` pattern
2. **Refactor affected contracts** to use simple state variables OR
3. **Move validator logic to module-level** (outside contracts) OR  
4. **Report critical bug** to Zexus team for urgent fix

### Testing:

All limitations documented here are **verified with custom tests**:

- ✅ Named imports - Verified working
- ✅ Direct property assignment - Verified working
- ✅ Bracket notation assignment - Verified working
- ✅ Temp variable from map - Verified working
- ✅ Module-level inline reconstruction - Verified working
- ❌ Contract state maps - Verified broken (silent failure)

---

## Conclusion

**Zexus 1.6.6 has made significant progress but still has ONE critical limitation:**

✅ **GOOD NEWS:** Most nested map operations now work correctly at module level!
- Named imports ✅
- Direct property assignment ✅
- Temp variables from maps ✅
- Multiple inline reconstructions ✅

❌ **CRITICAL ISSUE:** Contract state map operations are completely broken.
- Map assignments in contract state fail silently
- This blocks ALL production contracts using map-based state

**Status:** 🚨 **DEPLOYMENT BLOCKED** for contracts using map state variables.

Module-level code and contracts with simple state variables can be deployed.

---

*Document Version: 2.0*  
*Last Updated: January 2, 2026 (Fixes applied)*  
*Zexus Version: 1.6.6*  
*Test Suite: Comprehensive (custom tests + original 5 test files)*
*Fixes: Named imports, nested map updates (module-level)*
*Remaining Issue: Contract state maps (critical)*
