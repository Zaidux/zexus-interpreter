# Keywords After Dot - Context-Sensitive Property Access

## Overview

Zexus supports **context-sensitive keyword parsing**, allowing reserved keywords like `verify`, `data`, `hash`, and others to be used as property and method names when they appear after a dot (`.`) operator. This feature eliminates naming conflicts and provides a natural, intuitive syntax for accessing object properties and methods.

**Key Benefits:**
- Use any keyword as a property or method name after a dot
- Natural syntax without escape sequences or workarounds
- Maintains keyword functionality in statement contexts
- Consistent with modern language design (Python, JavaScript)

## Syntax

```zexus
object.keyword         // Property access
object.keyword()       // Method call
object.keyword.nested  // Chained access
```

## Supported Contexts

### ✅ Keywords as Property Names

```zexus
data Config {
    data: string,      // 'data' is a keyword but valid as field name
    verify: bool,      // 'verify' is a keyword
    hash: string       // 'hash' is a keyword
}

let config = Config("sample", true, "abc123");
print(config.data);    // ✅ "sample"
print(config.verify);  // ✅ true
print(config.hash);    // ✅ "abc123"
```

### ✅ Keywords as Method Names

```zexus
data verified Transaction {
    amount: number require amount > 0
}

let tx = Transaction(100);
print(tx.verify());    // ✅ Returns: true
print(tx.hash());      // ✅ Returns: cryptographic hash
```

### ✅ Chained Property Access

```zexus
object.data.verify.hash  // ✅ All keywords work in chains
map.get("key").data      // ✅ Works with method returns
user.profile.data.name   // ✅ Nested access
```

### ❌ Keywords at Statement Start (Reserved)

```zexus
verify condition;              // ✅ Statement keyword
data TypeName { ... }         // ✅ Statement keyword
hash value;                   // ✅ Statement keyword

let verify = "test";          // ❌ Error: 'verify' is reserved
const data = {};              // ❌ Error: 'data' is reserved
```

## How It Works

### Lexer Behavior

The lexer tokenizes keywords normally, producing keyword tokens like `VERIFY`, `DATA`, `HASH`:

```zexus
t.verify()
// Tokens: [IDENT('t'), DOT, VERIFY('verify'), LPAREN, RPAREN]
```

### Parser Context-Awareness

The parser checks the context:
- **After DOT:** Accept any token with a literal value (keywords become identifiers)
- **At statement start:** Enforce keyword restrictions

```python
# In strategy_context.py line 2816
if not name_token.literal:
    break  # Only reject tokens without literals
# This allows keywords (which have literals) to be property names
```

### Evaluator Behavior

The evaluator treats keyword-named properties identically to regular properties:

```zexus
t.verify   // Looks up 'verify' property on object t
t.verify() // Calls 'verify' method on object t
```

## Use Cases

### 1. Dataclass Methods

Auto-generated dataclass methods use keyword names:

```zexus
data verified User {
    name: string,
    email: string
}

let user = User("Alice", "alice@example.com");

// All these use keywords as method names
user.verify()   // ✅ Validates constraints
user.hash()     // ✅ Cryptographic hash
user.data       // ✅ Could be a field name
```

### 2. API Responses

```zexus
data APIResponse {
    data: any,          // Common in REST APIs
    status: number,
    error: string = ""
}

let response = APIResponse({ users: [] }, 200);
print(response.data);   // ✅ Natural access
```

### 3. Blockchain Objects

```zexus
data Block {
    hash: string,
    data: string,
    verify: function
}

let block = Block("0x123...", "payload", validateBlock);
print(block.hash);      // ✅ Access hash value
print(block.verify());  // ✅ Call verify function
```

### 4. Metadata Objects

```zexus
data FileMetadata {
    data: blob,
    hash: string,
    verify: bool
}

let file = FileMetadata(readFile("doc.pdf"), "abc123", true);
if (file.verify) {
    print("Hash: " + file.hash);
}
```

## Examples

### Example 1: Verified Dataclass

```zexus
data verified Transaction {
    from: string,
    to: string,
    amount: number require amount > 0
}

let tx = Transaction("Alice", "Bob", 100);

// Use keyword methods naturally
if (tx.verify()) {
    print("Transaction is valid");
    print("Hash: " + tx.hash());
} else {
    print("Transaction failed verification");
}
```

### Example 2: Config with Reserved Names

```zexus
data Config {
    data: string,        // Keyword as field
    verify: bool,        // Keyword as field
    log: string,         // Keyword as field
    const: number        // Keyword as field
}

let config = Config("payload", true, "debug", 42);

// Access all fields naturally
print("Data: " + config.data);
print("Verify: " + config.verify);
print("Log: " + config.log);
print("Const: " + config.const);
```

### Example 3: Method Chaining

```zexus
data Builder {
    data: map,
    
    set: function(key, val) {
        this.data[key] = val;
        return this;
    }
}

let builder = Builder({});
builder.set("verify", true)
       .set("hash", "abc")
       .set("data", "payload");

print(builder.data);  // ✅ Accesses data field
```

## Implementation Details

### Parser Strategy

The parser uses a two-phase approach:

1. **Token Recognition:** Lexer produces keyword tokens based on reserved words
2. **Context Analysis:** Parser checks if keyword follows a DOT operator
3. **Token Conversion:** Keywords after DOT are treated as identifiers

### File: `strategy_context.py`

The key implementation is in the property access parser:

```python
def _parse_expression(self, ...):
    # ... previous code ...
    
    if self.cur_token.type == DOT:
        self.next_token()
        name_token = self.cur_token
        
        # Accept any token with a literal (keywords or identifiers)
        if not name_token.literal:
            break  # Only reject tokens without literals
        
        # Create identifier from token literal
        property_name = Identifier(name_token.literal)
        # ... continue parsing ...
```

This allows tokens like `VERIFY('verify')`, `DATA('data')`, `HASH('hash')` to be used as property names.

### Backward Compatibility

This feature is **fully backward compatible**:

- Existing code using keywords in property names continues to work
- Keywords at statement level remain reserved
- No breaking changes to lexer or keyword table

## Design Philosophy

### Consistency with Modern Languages

Many modern languages support context-sensitive keywords:

**Python:**
```python
obj.class    # ✅ Works despite 'class' being a keyword
obj.def      # ✅ Works despite 'def' being a keyword
```

**JavaScript:**
```javascript
obj.new      // ✅ Works despite 'new' being a keyword
obj.class    // ✅ Works despite 'class' being a keyword
```

**Zexus:**
```zexus
obj.verify   // ✅ Works despite 'verify' being a keyword
obj.data     // ✅ Works despite 'data' being a keyword
```

### Natural API Design

This feature enables natural, idiomatic APIs:

```zexus
// ✅ Natural - reads like English
if (tx.verify()) {
    db.save(tx.hash());
}

// ❌ Awkward - requires workarounds
if (tx.verifyTransaction()) {  // Verbose
    db.save(tx.getHash());      // Unnatural
}
```

## Best Practices

### 1. Use Keywords as Properties When Natural

```zexus
// ✅ Good - 'data' is the natural name
data Response {
    data: any,
    status: number
}

// ❌ Bad - avoiding 'data' makes API awkward
data Response {
    payload: any,    // Less intuitive
    status: number
}
```

### 2. Leverage Auto-Generated Methods

```zexus
// ✅ Good - use standard method names
data verified User {
    name: string
}
user.verify()   // Standard verify method
user.hash()     // Standard hash method

// ❌ Bad - custom names reduce clarity
user.validateUser()
user.computeHash()
```

### 3. Document When Using Keywords

```zexus
/**
 * Configuration object
 * @property {string} data - Main configuration payload
 * @property {bool} verify - Whether to verify on load
 */
data Config {
    data: string,
    verify: bool
}
```

### 4. Avoid Overuse in Statements

```zexus
// ✅ Good - keywords only as properties
let transaction = Transaction(...);
if (transaction.verify()) { ... }

// ❌ Bad - trying to use keywords as variables
// let verify = true;  // Error: 'verify' is reserved
```

## Limitations

### Cannot Use Keywords as Variable Names

```zexus
let verify = true;        // ❌ Error: reserved keyword
const data = "test";      // ❌ Error: reserved keyword
```

**Solution:** Use keywords only after dots:

```zexus
let obj = { verify: true };   // ✅ Works
obj.verify                    // ✅ Access keyword property
```

### Cannot Use Keywords in Function Parameters

```zexus
function test(verify, data) {  // ❌ Error: reserved keywords
    print(verify);
}
```

**Solution:** Use descriptive names:

```zexus
function test(shouldVerify, userData) {  // ✅ Works
    print(shouldVerify);
}
```

## Future Enhancements

Planned improvements:

1. **Dictionary Keys:** Allow keywords as map keys: `map["verify"]`
2. **Function Parameters:** Context-aware parameter names
3. **Import Aliases:** `import verify as verifyFunc`
4. **Destructuring:** `let { data, verify } = response`

## Technical Reference

### Modified Files

- **`src/zexus/parser/strategy_context.py`**
  - Line 2816: Changed type check to literal check
  - Enables keywords after DOT operator

### Related Keywords

All reserved keywords work as property names:
- `verify` - Verification statements
- `data` - Dataclass definitions
- `hash` - Hashing operations
- `log` - Logging statements
- `const` - Constant declarations
- `let` - Variable declarations
- `if`, `else`, `elif` - Control flow
- `while`, `for` - Loops
- `function`, `return` - Functions
- And all other reserved words

### Performance Impact

**Zero performance overhead:**
- Parser check is single literal existence test
- No runtime overhead
- No additional memory usage
- Compilation speed unchanged

## See Also

- [DATA Keyword](../DATA.md) - Dataclass definitions
- [Modifiers](../MODIFIERS.md) - Verified and immutable types
- [Property Access](../../ADVANCED_FEATURES.md#property-access) - Property access syntax
- [Parser Architecture](../../ARCHITECTURE.md#parser) - Parser design

---

**Status**: ✅ Implemented  
**Version**: 1.0.0  
**Last Updated**: December 24, 2025
