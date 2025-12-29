# THIS Keyword Documentation

## Overview

The `THIS` keyword provides **self-reference** in Zexus, allowing access to the current instance context within:

- **Smart Contracts**: Reference contract instance and its state
- **Data Classes**: Access instance properties and methods
- **Entity Methods**: Reference the entity instance
- **Object-Oriented Patterns**: Enable method chaining and encapsulation

## Syntax

```zexus
this.propertyName
this.methodName()
this[key]
```

The `THIS` keyword can be used anywhere within a contract action, data method, or entity method to reference the current instance.

## Behavior

### In Smart Contracts

```zexus
contract TokenContract {
    persistent storage balances: Map<Address, integer>
    persistent storage owner: Address
    
    action constructor() {
        this.owner = TX.caller  # Set owner to contract creator
    }
    
    action transfer(to: Address, amount: integer) {
        let senderBalance = this.balances[TX.caller]
        require senderBalance >= amount, "Insufficient balance"
        
        this.balances[TX.caller] = senderBalance - amount
        this.balances[to] = this.balances.get(to, 0) + amount
    }
    
    action getBalance(account: Address) -> integer {
        return this.balances.get(account, 0)
    }
}
```

### In Data Classes

```zexus
data Rectangle {
    width: number
    height: number
    
    area() {
        return this.width * this.height
    }
    
    perimeter() {
        return 2 * (this.width + this.height)
    }
    
    scale(factor) {
        this.width = this.width * factor
        this.height = this.height * factor
        return this  # Enable method chaining
    }
}

let rect = Rectangle(10, 5)
print rect.area()         # 50
print rect.perimeter()    # 30
rect.scale(2)
print rect.area()         # 200
```

### In Entity Methods

```zexus
entity User {
    name: string,
    email: string,
    age: integer
}

action createUser(name, email, age) {
    let user = User {
        name: name,
        email: email,
        age: age
    }
    return user
}

# Note: THIS is primarily for contract/data class methods
# Entity instances don't typically have methods with 'this'
```

## Key Features

### 1. Property Access
Access instance properties using dot notation:

```zexus
contract BankAccount {
    state balance: integer
    state accountNumber: string
    
    action deposit(amount: integer) {
        this.balance = this.balance + amount
        emit Deposit(this.accountNumber, amount)
    }
    
    action getBalance() -> integer {
        return this.balance
    }
}
```

### 2. Method Invocation
Call other methods on the same instance:

```zexus
data Calculator {
    value: number
    
    add(n) {
        this.value = this.value + n
        return this
    }
    
    multiply(n) {
        this.value = this.value * n
        return this
    }
    
    reset() {
        this.value = 0
        return this
    }
}

let calc = Calculator(10)
calc.add(5).multiply(2).add(3)  # Method chaining
print calc.value  # 33
```

### 3. State Modification
Modify instance state within methods:

```zexus
contract Counter {
    state count: integer
    
    action constructor() {
        this.count = 0
    }
    
    action increment() {
        this.count = this.count + 1
    }
    
    action decrement() {
        this.count = this.count - 1
    }
    
    action reset() {
        this.count = 0
    }
    
    action getValue() -> integer {
        return this.count
    }
}
```

## Usage Examples

### Example 1: ERC20 Token Contract

```zexus
contract ERC20Token {
    persistent storage totalSupply: integer
    persistent storage balances: Map<Address, integer>
    persistent storage allowances: Map<Address, Map<Address, integer>>
    persistent storage name: string
    persistent storage symbol: string
    
    action constructor(initialSupply: integer, tokenName: string, tokenSymbol: string) {
        this.totalSupply = initialSupply
        this.name = tokenName
        this.symbol = tokenSymbol
        this.balances[TX.caller] = initialSupply
    }
    
    action transfer(to: Address, amount: integer) -> boolean {
        let senderBalance = this.balances[TX.caller]
        require senderBalance >= amount, "Insufficient balance"
        
        this.balances[TX.caller] = senderBalance - amount
        this.balances[to] = this.balances.get(to, 0) + amount
        
        emit Transfer(TX.caller, to, amount)
        return true
    }
    
    action approve(spender: Address, amount: integer) -> boolean {
        if !this.allowances.has(TX.caller) {
            this.allowances[TX.caller] = {}
        }
        this.allowances[TX.caller][spender] = amount
        emit Approval(TX.caller, spender, amount)
        return true
    }
    
    action transferFrom(from: Address, to: Address, amount: integer) -> boolean {
        let allowance = this.allowances[from][TX.caller]
        require allowance >= amount, "Allowance exceeded"
        
        let fromBalance = this.balances[from]
        require fromBalance >= amount, "Insufficient balance"
        
        this.balances[from] = fromBalance - amount
        this.balances[to] = this.balances.get(to, 0) + amount
        this.allowances[from][TX.caller] = allowance - amount
        
        emit Transfer(from, to, amount)
        return true
    }
    
    action balanceOf(account: Address) -> integer {
        return this.balances.get(account, 0)
    }
}
```

### Example 2: Data Class with Computed Properties

```zexus
data Circle {
    radius: number
    
    area() {
        return 3.14159 * this.radius * this.radius
    }
    
    circumference() {
        return 2 * 3.14159 * this.radius
    }
    
    diameter() {
        return 2 * this.radius
    }
    
    scale(factor) {
        this.radius = this.radius * factor
        return this
    }
}

let circle = Circle(5)
print "Radius: " + string(circle.radius)
print "Area: " + string(circle.area())
print "Circumference: " + string(circle.circumference())

circle.scale(2)
print "New area: " + string(circle.area())
```

### Example 3: Multi-Signature Wallet

```zexus
contract MultiSigWallet {
    state owners: List<Address>
    state requiredSignatures: integer
    state transactions: Map<integer, Transaction>
    state confirmations: Map<integer, List<Address>>
    state transactionCount: integer
    
    action constructor(ownerList: List<Address>, required: integer) {
        require ownerList.length > 0, "Owners required"
        require required > 0 && required <= ownerList.length, "Invalid signature count"
        
        this.owners = ownerList
        this.requiredSignatures = required
        this.transactionCount = 0
    }
    
    action isOwner(account: Address) -> boolean {
        for each owner in this.owners {
            if owner == account {
                return true
            }
        }
        return false
    }
    
    action submitTransaction(to: Address, amount: integer) -> integer {
        require this.isOwner(TX.caller), "Not an owner"
        
        let txId = this.transactionCount
        this.transactions[txId] = {
            to: to,
            amount: amount,
            executed: false
        }
        this.confirmations[txId] = []
        this.transactionCount = this.transactionCount + 1
        
        emit TransactionSubmitted(txId, TX.caller, to, amount)
        return txId
    }
    
    action confirmTransaction(txId: integer) {
        require this.isOwner(TX.caller), "Not an owner"
        require this.transactions.has(txId), "Transaction does not exist"
        require !this.transactions[txId].executed, "Already executed"
        
        # Check if already confirmed
        for each confirmer in this.confirmations[txId] {
            require confirmer != TX.caller, "Already confirmed"
        }
        
        this.confirmations[txId].push(TX.caller)
        emit TransactionConfirmed(txId, TX.caller)
        
        # Check if enough confirmations
        if this.confirmations[txId].length >= this.requiredSignatures {
            this.executeTransaction(txId)
        }
    }
    
    action executeTransaction(txId: integer) {
        let tx = this.transactions[txId]
        require !tx.executed, "Already executed"
        require this.confirmations[txId].length >= this.requiredSignatures, "Not enough confirmations"
        
        this.transactions[txId].executed = true
        # Execute transfer logic here
        emit TransactionExecuted(txId, tx.to, tx.amount)
    }
}
```

### Example 4: Builder Pattern with Method Chaining

```zexus
data QueryBuilder {
    table: string
    selectFields: List<string>
    whereClause: string
    orderBy: string
    limitValue: integer
    
    select(fields: List<string>) {
        this.selectFields = fields
        return this
    }
    
    from(tableName: string) {
        this.table = tableName
        return this
    }
    
    where(condition: string) {
        this.whereClause = condition
        return this
    }
    
    order(field: string) {
        this.orderBy = field
        return this
    }
    
    limit(n: integer) {
        this.limitValue = n
        return this
    }
    
    build() -> string {
        let query = "SELECT " + join(this.selectFields, ", ")
        query = query + " FROM " + this.table
        
        if this.whereClause != null && this.whereClause != "" {
            query = query + " WHERE " + this.whereClause
        }
        
        if this.orderBy != null && this.orderBy != "" {
            query = query + " ORDER BY " + this.orderBy
        }
        
        if this.limitValue > 0 {
            query = query + " LIMIT " + string(this.limitValue)
        }
        
        return query
    }
}

# Usage with method chaining
let query = QueryBuilder()
    .select(["id", "name", "email"])
    .from("users")
    .where("age > 18")
    .order("name ASC")
    .limit(10)
    .build()

print query
# Output: SELECT id, name, email FROM users WHERE age > 18 ORDER BY name ASC LIMIT 10
```

### Example 5: State Machine

```zexus
data StateMachine {
    currentState: string
    states: Map<string, Map<string, string>>
    
    constructor(initialState: string) {
        this.currentState = initialState
        this.states = {}
    }
    
    addTransition(fromState: string, event: string, toState: string) {
        if !this.states.has(fromState) {
            this.states[fromState] = {}
        }
        this.states[fromState][event] = toState
        return this
    }
    
    trigger(event: string) -> boolean {
        if !this.states.has(this.currentState) {
            return false
        }
        
        let transitions = this.states[this.currentState]
        if !transitions.has(event) {
            return false
        }
        
        let newState = transitions[event]
        print "Transition: " + this.currentState + " -> " + newState + " (event: " + event + ")"
        this.currentState = newState
        return true
    }
    
    getState() -> string {
        return this.currentState
    }
}

# Traffic light example
let light = StateMachine("red")
    .addTransition("red", "timer", "green")
    .addTransition("green", "timer", "yellow")
    .addTransition("yellow", "timer", "red")

print "Initial: " + light.getState()  # red
light.trigger("timer")                 # red -> green
light.trigger("timer")                 # green -> yellow
light.trigger("timer")                 # yellow -> red
```

### Example 6: Access Control with THIS

```zexus
contract AccessControl {
    state owner: Address
    state admins: Map<Address, boolean>
    state moderators: Map<Address, boolean>
    
    action constructor() {
        this.owner = TX.caller
    }
    
    modifier onlyOwner {
        require TX.caller == this.owner, "Only owner can call this"
    }
    
    modifier onlyAdmin {
        require this.isAdmin(TX.caller), "Only admins can call this"
    }
    
    action isAdmin(account: Address) -> boolean {
        return account == this.owner || this.admins.get(account, false)
    }
    
    action isModerator(account: Address) -> boolean {
        return this.moderators.get(account, false)
    }
    
    action addAdmin(account: Address) modifier onlyOwner {
        this.admins[account] = true
        emit AdminAdded(account)
    }
    
    action removeAdmin(account: Address) modifier onlyOwner {
        this.admins[account] = false
        emit AdminRemoved(account)
    }
    
    action addModerator(account: Address) modifier onlyAdmin {
        this.moderators[account] = true
        emit ModeratorAdded(account)
    }
    
    action removeModerator(account: Address) modifier onlyAdmin {
        this.moderators[account] = false
        emit ModeratorRemoved(account)
    }
}
```

## Rules and Best Practices

### ✅ DO:

1. **Use THIS for instance properties**
   ```zexus
   contract MyContract {
       state balance: integer
       
       action updateBalance(amount) {
           this.balance = this.balance + amount  # ✅ Clear
       }
   }
   ```

2. **Enable method chaining with THIS**
   ```zexus
   data Builder {
       value: number
       
       add(n) {
           this.value = this.value + n
           return this  # ✅ Enables chaining
       }
   }
   ```

3. **Use THIS in modifiers and guards**
   ```zexus
   action protected() {
       require TX.caller == this.owner, "Unauthorized"  # ✅
   }
   ```

4. **Access contract state via THIS**
   ```zexus
   action getTotal() {
       return this.balance + this.pending  # ✅ Explicit
   }
   ```

### ❌ DON'T:

1. **Don't use THIS outside contracts/data classes**
   ```zexus
   action standalone() {
       let x = this.value  # ❌ Error: not in contract/data context
   }
   ```

2. **Don't confuse THIS with local variables**
   ```zexus
   action process() {
       let balance = 100
       this.balance = balance  # ✅ Different: instance vs local
   }
   ```

3. **Don't forget THIS for property access**
   ```zexus
   contract Counter {
       state count: integer
       
       action increment() {
           count = count + 1  # ❌ Won't work without 'this'
           this.count = this.count + 1  # ✅ Correct
       }
   }
   ```

## Comparison with Other Patterns

### THIS vs Direct State Access

```zexus
contract Example {
    state value: integer
    
    action method1() {
        # With THIS (explicit, recommended)
        this.value = 10
    }
    
    action method2() {
        # Without THIS (may not work in all contexts)
        value = 10  # ❌ May cause errors
    }
}
```

### THIS vs TX (Transaction Context)

| Feature | THIS | TX |
|---------|------|-----|
| Purpose | Current instance | Transaction context |
| Scope | Contract/Data instance | Global transaction |
| Properties | Instance state/methods | caller, value, timestamp, etc. |
| Mutability | Can modify instance | Read-only |

```zexus
contract Token {
    state balances: Map<Address, integer>
    
    action transfer(to, amount) {
        # THIS - instance state
        this.balances[TX.caller] -= amount
        
        # TX - transaction context
        let sender = TX.caller
        let value = TX.value
    }
}
```

## Error Handling

### Invalid Usage

```zexus
# ❌ Using THIS outside contract/data
action regularFunction() {
    let x = this.property  # Error: 'this' can only be used inside a contract, data method, or entity method
}
```

**Error message:**
```
'this' can only be used inside a contract, data method, or entity method
```

### Safe Patterns

```zexus
# ✅ Using THIS in contract
contract Safe {
    state value: integer
    
    action getValue() {
        return this.value  # Safe - inside contract
    }
}

# ✅ Using THIS in data method
data Point {
    x: number
    y: number
    
    distance() {
        return sqrt(this.x * this.x + this.y * this.y)  # Safe - in data method
    }
}
```

## Implementation Details

### Files Modified
- `src/zexus/zexus_token.py` - Added THIS token
- `src/zexus/lexer.py` - Added "this" keyword recognition
- `src/zexus/zexus_ast.py` - Added ThisExpression AST node
- `src/zexus/parser/parser.py` - Added parse_this() method
- `src/zexus/evaluator/core.py` - Added ThisExpression dispatch
- `src/zexus/evaluator/statements.py` - Added eval_this_expression()

### How It Works Internally

1. **Context Setup**: When entering a contract action or data method, instance is bound to `__contract_instance__` or `this` in environment
2. **Parsing**: `this` keyword creates a `ThisExpression` AST node
3. **Evaluation**: `eval_this_expression()` looks up:
   - `__contract_instance__` for contracts
   - `this` binding for data methods
   - Entity method context (future)
4. **Property Access**: `this.property` uses property access expression evaluation
5. **Error Handling**: Returns error if THIS used outside valid context

## Compatibility

- **Version:** Zexus v1.0.0+
- **Breaking Changes:** None
- **Backward Compatible:** Yes
- **Platform:** All supported platforms

## Summary

The `THIS` keyword enables object-oriented patterns in Zexus:

✅ **Instance reference** - Access current contract/data instance
✅ **Property access** - Read and modify instance properties
✅ **Method invocation** - Call instance methods
✅ **Method chaining** - Return this for fluent interfaces
✅ **State management** - Modify contract state
✅ **Clear semantics** - Explicit instance reference

Use `THIS` in:
- Smart contract actions to access contract state
- Data class methods to reference instance properties
- Builder patterns for method chaining
- Object-oriented design patterns

**Related Keywords**: CONTRACT, DATA, ENTITY, STATE, PERSISTENT, STORAGE  
**Category**: Object-Oriented Programming, Blockchain  
**Status**: ✅ Fully Implemented  
**Documentation**: Complete  
**Last Updated**: December 29, 2025
