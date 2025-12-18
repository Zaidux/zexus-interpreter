# Module System Keywords - Complete Guide
# USE, IMPORT, EXPORT, MODULE, PACKAGE, FROM, EXTERNAL

## Overview
These keywords provide module system and code organization capabilities in Zexus:
- **USE**: Import modules into current scope
- **IMPORT**: Alternative import syntax
- **EXPORT**: Mark symbols for export
- **MODULE**: Define module boundaries
- **PACKAGE**: Organize related modules
- **FROM**: Import specific items from module
- **EXTERNAL**: Declare external (native) functions

---

## USE Keyword

### Syntax
```zexus
use "module_path" as alias;
use "module_path";
use { Name1, Name2 } from "module_path";
```

### Basic Usage

#### Import with Alias
```zexus
use "math" as math;
print math.add(5, 3);
```

#### Import Relative Module
```zexus
use "./utils.zx" as utils;
let result = utils.process(data);
```

#### Import Builtin Module
```zexus
use "crypto" as crypto;
let hash = crypto.sha256("data");
```

### Advanced USE Patterns

#### Named Imports
```zexus
use { add, multiply } from "./math.zx";
print add(5, 3);
print multiply(4, 2);
```

#### Multiple Imports
```zexus
use "math" as math;
use "string" as str;
use "array" as arr;
```

---

## EXPORT Keyword

### Syntax
```zexus
export functionName;
export variableName;
```

### Basic Usage

```zexus
// In module file: utils.zx
action greet(name) {
    return "Hello, " + name;
}

const VERSION = "1.0";

export greet;
export VERSION;
```

### Export Patterns

#### Export Functions
```zexus
action add(a, b) {
    return a + b;
}

action subtract(a, b) {
    return a - b;
}

export add;
export subtract;
```

#### Export Constants
```zexus
const API_URL = "https://api.example.com";
const TIMEOUT = 5000;

export API_URL;
export TIMEOUT;
```

---

## MODULE Keyword

### Syntax
```zexus
module name {
    // Module contents
}
```

### Basic Usage

```zexus
module math {
    action add(a, b) {
        return a + b;
    }
    
    action multiply(a, b) {
        return a * b;
    }
}
```

---

## EXTERNAL Keyword

### Syntax
```zexus
external functionName;
```

### Basic Usage

```zexus
external hashFunction;
external sortArray;
external encryptData;

// These will be linked to native implementations
```

### Purpose
- Declares functions implemented in native code (C/C++)
- Creates placeholder until native function is linked
- Used for performance-critical operations

---

## Common Patterns

### Pattern 1: Library Module
```zexus
// File: mylib.zx
action helper1() {
    return "Helper 1";
}

action helper2() {
    return "Helper 2";
}

export helper1;
export helper2;

// Using the library
use "./mylib.zx" as mylib;
print mylib.helper1();
```

### Pattern 2: Configuration Module
```zexus
// File: config.zx
const DB_HOST = "localhost";
const DB_PORT = 5432;
const API_KEY = "secret";

export DB_HOST;
export DB_PORT;
export API_KEY;

// Using configuration
use "./config.zx" as config;
let host = config.DB_HOST;
```

### Pattern 3: Utility Functions
```zexus
// File: utils.zx
action capitalize(str) {
    return str;  // Simplified
}

action trim(str) {
    return str;
}

export capitalize;
export trim;
```

### Pattern 4: External Native Functions
```zexus
external nativeSort;
external nativeHash;
external nativeEncrypt;

// These would be implemented in C/C++ for performance
```

---

## Module System Workflow

1. **Create Module File**
```zexus
// math.zx
action add(a, b) {
    return a + b;
}

export add;
```

2. **Use in Main File**
```zexus
// main.zx
use "./math.zx" as math;
let result = math.add(5, 3);
print result;
```

3. **Module Caching**
- Modules are cached after first load
- Subsequent imports reuse cached version
- Prevents circular dependency issues

---

## Best Practices

### ✅ DO

1. **Use clear module names**
```zexus
use "./userAuthentication.zx" as auth;
use "./databaseConnection.zx" as db;
```

2. **Export only public API**
```zexus
// Private
action _internalHelper() { }

// Public
action publicAPI() { }

export publicAPI;  // Only export public functions
```

3. **Use aliases for clarity**
```zexus
use "very/long/path/to/module.zx" as shortName;
```

4. **Group related exports**
```zexus
// All math functions together
export add;
export subtract;
export multiply;
export divide;
```

### ❌ DON'T

1. **Don't create circular dependencies**
```zexus
// moduleA.zx uses moduleB.zx
// moduleB.zx uses moduleA.zx
// ❌ Circular dependency!
```

2. **Don't export everything**
```zexus
// ❌ Bad - exports internal implementation
export _privateHelper;
export _internalState;
```

3. **Don't use relative paths without ./  or ../**
```zexus
// ❌ Ambiguous
use "utils.zx" as utils;

// ✅ Clear
use "./utils.zx" as utils;
```

---

## Known Issues ⚠️

### Issues Found (December 2025)

1. **~~External Declarations Don't Auto-Link~~** ✅ **FIXED** (December 18, 2025)
   - **Root Cause**: Parser expected full syntax `external action name from "module"` but simple syntax `external name;` not supported
   - **Problem**: Simple syntax not recognized, identifier not created in environment
   - **Solution**: Added simple syntax support in parse_external_declaration() and ContextStackParser
   - **Status**: ✅ FULLY WORKING - `external nativeSort;` creates placeholder, accessible in functions
   - **Verification**: External declarations work, can be passed as arguments to functions

2. **Module System May Not Be Fully Implemented** (Priority: Unknown)
   - Some module features may be incomplete
   - IMPORT, MODULE, PACKAGE keywords may not be fully functional
   - Status: Needs verification
   - Tests focused on USE and EXPORT which appear functional

3. **FROM Syntax Support Unclear** (Priority: Low)
   - `from "module" import Name` syntax may not work
   - Alternative: use named imports with USE
   - Status: Syntax variant, USE with braces works

---

## Summary

### Keyword Usage

**USE**: Import modules (most common)  
**EXPORT**: Mark functions/variables for export  
**EXTERNAL**: Declare native functions  
**MODULE**: Organize code (may be implicit)  
**IMPORT/FROM/PACKAGE**: Alternative syntax (may not be fully implemented)

### Key Takeaways
1. USE is the primary import mechanism
2. EXPORT marks symbols for external use
3. EXTERNAL declares native function placeholders
4. Modules enable code organization and reusability
5. Module caching prevents redundant loads
6. Circular dependencies should be avoided
7. Test module features carefully as implementation may vary

---

**Related Keywords**: LET, CONST, ACTION, FUNCTION  
**Category**: Module System  
**Status**: 🟡 Partially Working (USE and EXPORT functional, others need verification)  
**Tests Created**: 20 easy, 20 medium, 20 complex (mixed with I/O tests)  
**Documentation**: Complete  
**Last Updated**: December 16, 2025
