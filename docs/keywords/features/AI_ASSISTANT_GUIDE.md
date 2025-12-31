# AI Assistant Onboarding Guide for Zexus Blockchain Development

> **Purpose**: This document is specifically designed for AI assistants helping developers build blockchain applications with Zexus. It provides complete, self-contained information in one place.

**Last Updated**: December 31, 2025  
**Zexus Version**: 1.6.2

---

## 📑 Quick Navigation

- [What is Zexus?](#what-is-zexus)
- [Language Basics](#language-basics-30-second-overview)
- [Blockchain Features](#blockchain-features-core-strength)
- [Backend/Server Capabilities](#backendserver-capabilities)
- [Complete Syntax Reference](#complete-syntax-reference)
- [Standard Library](#standard-library-reference)
- [Common Patterns](#common-patterns--examples)
- [Troubleshooting Guide](#troubleshooting-guide)

---

## What is Zexus?

Zexus is a **blockchain-first programming language** with:
- Native smart contract support (CONTRACT keyword)
- Built-in blockchain primitives (LEDGER, STATE, AUDIT)
- Full backend capabilities (TCP servers, databases, HTTP)
- Security-first design (SANDBOX, REQUIRE, VERIFY)
- Python-like syntax with powerful features

**Key Differentiator**: Other languages need libraries for blockchain - Zexus has it built into the language itself.

---

## Language Basics (30-Second Overview)

### Variables & Data Types
```zexus
// Variables
let name = "Alice"           // String
let age = 25                 // Integer
let balance = 100.5          // Float
let active = true            // Boolean
let items = [1, 2, 3]        // List
let user = {"name": "Bob"}   // Map

// Constants
const PI = 3.14159
```

### Functions (called "Actions")
```zexus
// Basic action
action greet(name) {
    return "Hello, " + name
}

// With types
action add(a: integer, b: integer) -> integer {
    return a + b
}

// Async action
async action fetchData() {
    let result = await apiCall()
    return result
}
```

### Data Structures
```zexus
// Define custom type
data User {
    name: string
    age: integer
    email: string
}

// Create instance
let user = User{name: "Alice", age: 30, email: "alice@example.com"}

// Access fields
print(user.name)  // "Alice"
```

### Control Flow
```zexus
// If statements
if x > 10 {
    print("Large")
} else if x > 5 {
    print("Medium")
} else {
    print("Small")
}

// While loops
let i = 0
while i < 10 {
    print(i)
    i = i + 1
}

// For each loops
for each item in [1, 2, 3] {
    print(item)
}
```

---

## Blockchain Features (Core Strength)

### Smart Contracts

```zexus
contract TokenContract {
    // State variables (persistent across calls)
    state balances = {}
    state total_supply = 0
    state owner = null
    
    // Initialize contract
    action init(supply) {
        total_supply = supply
        owner = TX.caller  // Built-in transaction context
        balances[owner] = supply
        audit("contract_init", {"supply": supply})  // Auto-logged
    }
    
    // Transfer tokens
    action transfer(to, amount) {
        let from = TX.caller
        
        // Validation with REQUIRE
        require(balances[from] >= amount, "Insufficient balance")
        require(amount > 0, "Amount must be positive")
        
        // Execute transfer
        balances[from] = balances[from] - amount
        balances[to] = (balances[to] or 0) + amount
        
        // Automatic audit trail
        audit("transfer", {
            "from": from,
            "to": to,
            "amount": amount,
            "timestamp": TX.timestamp
        })
        
        return true
    }
    
    // Query balance
    action balanceOf(account) {
        return balances[account] or 0
    }
    
    // Owner-only function
    action mint(to, amount) {
        require(TX.caller == owner, "Only owner can mint")
        
        balances[to] = (balances[to] or 0) + amount
        total_supply = total_supply + amount
        
        audit("mint", {"to": to, "amount": amount})
    }
}

// Deploy and use contract
let token = TokenContract()
token.init(1000000)

// Transfer tokens
token.transfer("alice", 100)

// Check balance
let balance = token.balanceOf("alice")
print(balance)  // 100
```

### Transaction Context (Built-in)

Zexus automatically provides transaction context:

```zexus
action myAction() {
    // TX is automatically available in all actions
    print("Caller:", TX.caller)
    print("Timestamp:", TX.timestamp)
    print("Gas limit:", TX.gas_limit)
    print("Gas used:", TX.gas_used)
}
```

### Ledger System

```zexus
ledger AccountLedger {
    state accounts = {}
    state transaction_count = 0
    
    action credit(account, amount) {
        accounts[account] = (accounts[account] or 0) + amount
        transaction_count = transaction_count + 1
        audit("credit", {"account": account, "amount": amount})
    }
    
    action debit(account, amount) {
        require(accounts[account] >= amount, "Insufficient funds")
        accounts[account] = accounts[account] - amount
        transaction_count = transaction_count + 1
        audit("debit", {"account": account, "amount": amount})
    }
    
    action getBalance(account) {
        return accounts[account] or 0
    }
}
```

### Audit Trail (Automatic)

```zexus
// Every audit() call is automatically logged
action processPayment(from, to, amount) {
    audit("payment_started", {
        "from": from,
        "to": to,
        "amount": amount
    })
    
    // ... process payment ...
    
    audit("payment_completed", {
        "from": from,
        "to": to,
        "amount": amount,
        "success": true
    })
}

// Audit logs are searchable and immutable
```

### Security Features

```zexus
// REQUIRE - Validation with auto-revert
action withdraw(amount) {
    require(amount > 0, "Amount must be positive")
    require(balance >= amount, "Insufficient balance")
    // If any require fails, transaction reverts
}

// SANDBOX - Run untrusted code safely
sandbox {
    // Code here runs in isolated environment
    let result = executeUserCode(userInput)
}

// VERIFY - Cryptographic verification
verify signature(data, sig, public_key) {
    // Verify signature matches
}
```

---

## Backend/Server Capabilities

### TCP Server

```zexus
// Define connection handler
action handleClient(conn) {
    // Receive data
    let data = conn["receive"](4096)
    print("Received:", data)
    
    // Process and respond
    let response = "Echo: " + data
    conn["send"](response)
    
    // Close connection
    conn["close"]()
}

// Create and start server
let server = socket_create_server("0.0.0.0", 8080, handleClient, 100)
// Server now listening on port 8080
```

### HTTP Server (Built-in)

```zexus
// Create HTTP server
let app = http_server()

// Define routes
app["get"]("/api/users", action(req, res) {
    let users = db.query("SELECT * FROM users")
    res["json"](users)
})

app["post"]("/api/users", action(req, res) {
    let user = req["body"]
    db.execute("INSERT INTO users (name, email) VALUES (?, ?)", 
               [user["name"], user["email"]])
    res["status"](201)
    res["json"]({"success": true})
})

// Start server
app["listen"](3000)
print("Server running on port 3000")
```

### Database Integration

```zexus
// SQLite
let db = sqlite_connect("myapp.db")

// PostgreSQL
let db = postgres_connect("postgresql://user:pass@localhost/mydb")

// MySQL
let db = mysql_connect("mysql://user:pass@localhost/mydb")

// MongoDB
let db = mongo_connect("mongodb://localhost:27017/mydb")

// Execute queries
db.execute("CREATE TABLE users (id INT, name TEXT)")
db.execute("INSERT INTO users VALUES (?, ?)", [1, "Alice"])

// Query data
let users = db.query("SELECT * FROM users")
for each user in users {
    print(user["name"])
}

// Transactions
db.begin_transaction()
db.execute("UPDATE accounts SET balance = balance - 100 WHERE id = 1")
db.execute("UPDATE accounts SET balance = balance + 100 WHERE id = 2")
db.commit()  // or db.rollback()
```

### P2P Blockchain Node Example

```zexus
// Complete P2P blockchain node
data Block {
    index: integer
    timestamp: integer
    transactions: list
    previous_hash: string
    hash: string
    nonce: integer
}

data Peer {
    id: string
    host: string
    port: integer
}

let blockchain = []
let peers = []
let mempool = []  // Pending transactions

// Handle peer connections
action handlePeer(conn) {
    let message = conn["receive"](65536)
    
    // Parse message (JSON in production)
    if message.type == "request_chain" {
        conn["send"](str(blockchain))
    }
    
    if message.type == "new_block" {
        let block = parseBlock(message.data)
        if validateBlock(block) {
            blockchain = blockchain + [block]
            broadcastToPeers(block)
        }
    }
    
    conn["close"]()
}

// Start P2P server
action startNode(port) {
    let server = socket_create_server("0.0.0.0", port, handlePeer, 100)
    print("Blockchain node running on port", port)
    
    // Mining loop (runs in background)
    async action mine() {
        while true {
            if len(mempool) > 0 {
                let block = createBlock(mempool)
                let mined = mineBlock(block, 4)  // difficulty = 4
                
                if validateBlock(mined) {
                    blockchain = blockchain + [mined]
                    mempool = []
                    broadcastToPeers(mined)
                }
            }
            sleep(10)
        }
    }
    
    spawn(mine)
}

// Proof of Work
action mineBlock(block, difficulty) {
    let target = "0" * difficulty
    let nonce = 0
    
    while true {
        block.nonce = nonce
        let hash = sha256(str(block))
        
        if starts_with(hash, target) {
            block.hash = hash
            return block
        }
        
        nonce = nonce + 1
        
        if nonce % 1000 == 0 {
            print("Mining... nonce:", nonce)
        }
    }
}

// Consensus: Longest valid chain
action consensus() {
    let longest = blockchain
    
    for each peer in peers {
        let chain = requestChain(peer)
        if validateChain(chain) and len(chain) > len(longest) {
            longest = chain
        }
    }
    
    if len(longest) > len(blockchain) {
        blockchain = longest
        return true
    }
    
    return false
}
```

---

## Complete Syntax Reference

### Type Annotations (Optional)
```zexus
let name: string = "Alice"
let age: integer = 25
let balance: float = 100.5
let active: boolean = true
let items: list = [1, 2, 3]
let user: map = {"key": "value"}
```

### Pattern Matching
```zexus
match value {
    0 => print("Zero")
    1 => print("One")
    n if n > 10 => print("Large")
    _ => print("Other")
}
```

### Error Handling
```zexus
try {
    let result = riskyOperation()
} catch error {
    print("Error:", error)
} finally {
    cleanup()
}
```

### Async/Await
```zexus
async action fetchUser(id) {
    let response = await httpGet("/api/users/" + str(id))
    return response
}

// Call async action
let user = await fetchUser(123)
```

### Module System
```zexus
// Export from module
module mymodule {
    export action greet(name) {
        return "Hello, " + name
    }
    
    export let VERSION = "1.0.0"
}

// Import in another file
use mymodule

print(mymodule.greet("Alice"))
```

---

## Standard Library Reference

### Built-in Functions

**String Functions:**
- `len(s)` - Length of string
- `str(x)` - Convert to string
- `upper(s)` - Uppercase
- `lower(s)` - Lowercase
- `replace(s, old, new)` - Replace substring
- `starts_with(s, prefix)` - Check prefix
- `ends_with(s, suffix)` - Check suffix

**Math Functions:**
- `abs(x)` - Absolute value
- `min(a, b)` - Minimum
- `max(a, b)` - Maximum
- `pow(base, exp)` - Power
- `sqrt(x)` - Square root
- `round(x)` - Round to integer

**Array Functions:**
- `len(arr)` - Length
- `push(arr, item)` - Add to end
- `pop(arr)` - Remove from end
- `append(arr, item)` - Add to end
- `range(start, end)` - Generate range

**Type Functions:**
- `type(x)` - Get type name
- `to_int(x)` - Convert to integer
- `to_float(x)` - Convert to float
- `to_bool(x)` - Convert to boolean

**Cryptographic Functions:**
- `hash(data)` - SHA-256 hash
- `sha256(data)` - SHA-256
- `sha512(data)` - SHA-512
- `md5(data)` - MD5 hash
- `signature(data, private_key)` - Create signature
- `verify_sig(data, sig, public_key)` - Verify signature

**Concurrency Functions:**
- `spawn(action)` - Run action in background
- `sleep(seconds)` - Sleep for duration
- `async/await` - Asynchronous operations

---

## Common Patterns & Examples

### Pattern 1: DeFi Token Contract

```zexus
contract DeFiToken {
    state balances = {}
    state allowances = {}  // For delegated transfers
    state total_supply = 0
    
    action init(supply) {
        total_supply = supply
        balances[TX.caller] = supply
    }
    
    action transfer(to, amount) {
        let from = TX.caller
        require(balances[from] >= amount, "Insufficient balance")
        
        balances[from] = balances[from] - amount
        balances[to] = (balances[to] or 0) + amount
        
        audit("transfer", {"from": from, "to": to, "amount": amount})
        return true
    }
    
    action approve(spender, amount) {
        let owner = TX.caller
        allowances[owner + ":" + spender] = amount
        audit("approval", {"owner": owner, "spender": spender, "amount": amount})
        return true
    }
    
    action transferFrom(from, to, amount) {
        let spender = TX.caller
        let key = from + ":" + spender
        
        require(allowances[key] >= amount, "Allowance exceeded")
        require(balances[from] >= amount, "Insufficient balance")
        
        balances[from] = balances[from] - amount
        balances[to] = (balances[to] or 0) + amount
        allowances[key] = allowances[key] - amount
        
        audit("transferFrom", {"from": from, "to": to, "amount": amount, "spender": spender})
        return true
    }
}
```

### Pattern 2: RESTful API Server

```zexus
let db = sqlite_connect("app.db")
let app = http_server()

// Middleware for authentication
action auth_middleware(req, res, next) {
    let token = req["headers"]["Authorization"]
    
    if validateToken(token) {
        next()  // Continue to route handler
    } else {
        res["status"](401)
        res["json"]({"error": "Unauthorized"})
    }
}

// Routes
app["get"]("/api/users", action(req, res) {
    let users = db.query("SELECT id, name, email FROM users")
    res["json"](users)
})

app["post"]("/api/users", action(req, res) {
    let user = req["body"]
    
    // Validate
    if !user["name"] or !user["email"] {
        res["status"](400)
        res["json"]({"error": "Missing fields"})
        return
    }
    
    // Insert
    db.execute("INSERT INTO users (name, email) VALUES (?, ?)",
               [user["name"], user["email"]])
    
    res["status"](201)
    res["json"]({"success": true})
})

app["listen"](3000)
```

### Pattern 3: Blockchain Explorer

```zexus
// Blockchain explorer backend
let blockchain_db = sqlite_connect("blockchain.db")

// Index blocks
action indexBlock(block) {
    blockchain_db.execute(
        "INSERT INTO blocks (hash, index, timestamp, tx_count) VALUES (?, ?, ?, ?)",
        [block.hash, block.index, block.timestamp, len(block.transactions)]
    )
    
    // Index transactions
    for each tx in block.transactions {
        blockchain_db.execute(
            "INSERT INTO transactions (hash, block_hash, from_addr, to_addr, amount) VALUES (?, ?, ?, ?, ?)",
            [tx.hash, block.hash, tx.from, tx.to, tx.amount]
        )
    }
}

// Query API
let app = http_server()

app["get"]("/api/block/:hash", action(req, res) {
    let hash = req["params"]["hash"]
    let block = blockchain_db.query_one(
        "SELECT * FROM blocks WHERE hash = ?", [hash]
    )
    res["json"](block)
})

app["get"]("/api/address/:addr/transactions", action(req, res) {
    let addr = req["params"]["addr"]
    let txs = blockchain_db.query(
        "SELECT * FROM transactions WHERE from_addr = ? OR to_addr = ? ORDER BY timestamp DESC",
        [addr, addr]
    )
    res["json"](txs)
})
```

---

## Troubleshooting Guide

### Common Issues

**Issue: "Identifier not found"**
```zexus
// Problem: Using variable before declaration
print(x)  // Error!
let x = 5

// Solution: Declare before use
let x = 5
print(x)  // OK
```

**Issue: "Invalid assignment target"**
```zexus
// Problem: Map assignment in wrong scope
let mymap = {}
action test() {
    mymap["key"] = "value"  // May fail in some contexts
}

// Solution: Return new map or use state
contract MyContract {
    state mymap = {}
    
    action test() {
        mymap["key"] = "value"  // Works in contract state
    }
}
```

**Issue: Reserved keyword as variable**
```zexus
// Problem: Using reserved keyword
let limit = 10  // 'limit' is reserved for gas limiting!

// Solution: Use different name
let max_count = 10
let maximum = 10
```

**Issue: Contract not working**
```zexus
// Problem: Trying to use instantiate()
let token = instantiate(TokenContract)  // Wrong!

// Solution: Call contract as function
let token = TokenContract()  // Correct
token.init(1000)
```

**Issue: For loop syntax**
```zexus
// Problem: Wrong for loop syntax
for x in arr {  // Wrong syntax in Zexus!
    print(x)
}

// Solution: Use "for each"
for each x in arr {  // Correct
    print(x)
}
```

---

## How to Help Developers with Zexus

### When a developer asks for help:

1. **Start with this document** - It has everything in one place
2. **Ask clarifying questions**:
   - Are they building a blockchain/DeFi app? (Use contract/ledger features)
   - Are they building a backend API? (Use http_server or socket_create_server)
   - Do they need database access? (Use db drivers)
3. **Provide complete, working examples** - Not just snippets
4. **Explain blockchain-specific features** - Many devs don't know CONTRACT/LEDGER/AUDIT exist
5. **Show security features** - REQUIRE, SANDBOX, VERIFY are unique to Zexus

### Key Points to Remember:

- **Contracts are instantiated like functions**: `MyContract()` not `new MyContract()`
- **For loops require "each"**: `for each x in arr` not `for x in arr`
- **Some keywords are reserved**: `limit`, `defer`, `pattern` can't be variable names
- **State is for contracts/ledgers**: Regular actions can't modify global maps easily
- **TX context is automatic**: No need to pass transaction data manually
- **Audit is automatic**: Just call `audit()` and it's logged forever

---

## Quick Reference Card

```
VARIABLES:           let x = 5
CONSTANTS:           const PI = 3.14
ACTIONS:             action name() { }
DATA TYPES:          data User { name: string }
CONTRACTS:           contract Token { state balance = 0 }
LEDGERS:             ledger Accounts { state accounts = {} }

CONTROL FLOW:        if/else, while, for each
ASYNC:               async action, await
ERROR HANDLING:      try/catch/finally
VALIDATION:          require(condition, message)
SECURITY:            sandbox { }, verify

BLOCKCHAIN:          audit(), TX.caller, TX.timestamp
SERVERS:             socket_create_server(), http_server()
DATABASES:           sqlite_connect(), postgres_connect()
CRYPTO:              hash(), sha256(), signature()

INSTANTIATE CONTRACT: let token = TokenContract()
FOR LOOP:            for each item in items { }
```

---

**Remember**: This document is your single source of truth. Everything an AI needs to help build with Zexus is here. No need to search through 200+ other files!
