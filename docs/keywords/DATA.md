# DATA Keyword - Dataclasses & Structured Types

## Overview

The `DATA` keyword enables creation of structured data types (dataclasses) with automatic method generation, validation, and serialization. It provides a more powerful and Zexus-native alternative to manual class definitions for data-centric objects.

**Key Features:**
- Automatic constructor generation
- Built-in toString(), toJSON(), fromJSON() methods
- Type validation and constraints
- Immutability support
- Pattern matching support
- Blockchain integration (verification, hashing)
- Default values and computed properties

## Syntax

```zexus
data TypeName {
    field1: type,
    field2: type = defaultValue,
    field3: type require constraint
}

data immutable TypeName {
    // All instances are immutable
}

data verified TypeName {
    // Blockchain verification enabled
}
```

## Parameters

- **TypeName**: The name of the data type (must be PascalCase by convention)
- **fields**: Comma-separated field definitions with optional types, defaults, and constraints
- **modifiers**: Optional keywords (`immutable`, `verified`) before the type name

### Field Definition Syntax

```zexus
fieldName: type                           // Required field
fieldName: type = defaultValue            // Optional with default
fieldName: type require constraint        // With validation
fieldName: type = value require constraint // Both default and validation
```

### Supported Types

- `string` - Text values
- `number` - Numeric values (int or float)
- `bool` - Boolean values
- `array` - Arrays
- `map` - Hash maps
- `any` - Any type (no validation)
- Custom data types

## Basic Usage

### Simple Dataclass

```zexus
data User {
    name: string,
    email: string,
    age: number
}

// Create instance
let user = User("Alice", "alice@example.com", 30);

// Access fields
print(user.name);    // "Alice"
print(user.age);     // 30

// Auto-generated toString()
print(user);         // User(name="Alice", email="alice@example.com", age=30)
```

### With Default Values

```zexus
data Config {
    host: string = "localhost",
    port: number = 8080,
    debug: bool = false
}

// Use defaults
let config1 = Config();                          // All defaults
let config2 = Config("0.0.0.0");                // Custom host
let config3 = Config("0.0.0.0", 3000, true);   // All custom
```

### With Validation

```zexus
data Email {
    address: string require /^[\w\.-]+@[\w\.-]+\.\w+$/,
    verified: bool = false
}

let email1 = Email("alice@example.com");  // ✅ Valid
let email2 = Email("invalid-email");      // ❌ Runtime error: validation failed
```

## Advanced Features

### Immutable Dataclasses

```zexus
data immutable Point {
    x: number,
    y: number
}

let p1 = Point(10, 20);
// p1.x = 30;  // ❌ Error: Cannot modify immutable dataclass

// Create new instance for changes
let p2 = Point(30, p1.y);
```

### Nested Dataclasses

```zexus
data Address {
    street: string,
    city: string,
    zip: string
}

data Person {
    name: string,
    address: Address
}

let addr = Address("123 Main St", "Springfield", "12345");
let person = Person("Bob", addr);

print(person.address.city);  // "Springfield"
```

### Auto-Serialization

```zexus
data Product {
    id: number,
    name: string,
    price: number
}

let product = Product(1, "Widget", 19.99);

// Auto-generated toJSON()
let json = product.toJSON();
print(json);  // {"id":1,"name":"Widget","price":19.99}

// Auto-generated fromJSON()
let loaded = Product.fromJSON(json);
print(loaded.name);  // "Widget"
```

### Pattern Matching

> **Status:** Planned - Basic structure works, but pattern matching syntax not yet implemented

```zexus
// FUTURE FEATURE - Pattern matching syntax not yet available
data Result {
    success: bool,
    value: any,
    error: string = ""
}

// Currently use if/else instead of match
function handleResult(result) {
    if (result.success) {
        print("Success: " + result.value);
    } else {
        print("Error: " + result.error);
    }
}

let r1 = Result(true, 42, "");
let r2 = Result(false, null, "Not found");

handleResult(r1);  // "Success: 42"
handleResult(r2);  // "Error: Not found"
```

## Blockchain Features

### Verified Dataclasses

```zexus
data verified Transaction {
    from: address,
    to: address,
    amount: number,
    signature: hash
}

let tx = Transaction(
    "0x1234...",
    "0x5678...",
    100,
    "0xabcd..."
);

// Auto-generated verification methods
if (tx.verify()) {
    print("Transaction is valid");
    let txHash = tx.hash();  // Cryptographic hash
    print("Hash: " + txHash);
}
```

### State Management

```zexus
data verified AccountState {
    balance: number,
    nonce: number,
    lastUpdate: number
}

let state = AccountState(1000, 0, timestamp());

// Verify state integrity
verify(state) {
    require(state.balance >= 0, "Negative balance");
    require(state.nonce >= 0, "Negative nonce");
}
```

## Auto-Generated Methods

Every dataclass automatically includes:

### Constructor
```zexus
TypeName(field1, field2, ..., fieldN)
```

### Instance Methods

```zexus
.toString()        // String representation
.toJSON()          // JSON serialization
.equals(other)     // Deep equality check
.clone()           // Create a copy
.hash()            // Cryptographic hash (for verified types)
.verify()          // Validation check (for verified types)
```

### Static Methods

```zexus
TypeName.fromJSON(jsonString)  // Deserialize from JSON
TypeName.default()              // Create with all default values
```

## Type Constraints

### Built-in Constraints

```zexus
data Person {
    name: string require /^[A-Za-z\s]+$/,           // Regex pattern
    age: number require age >= 0 && age <= 150,     // Range check
    email: string require len(email) > 0,           // Length check
    role: string require role in ["admin", "user"]  // Enum-like
}
```

### Custom Validators

```zexus
function isValidSSN(ssn) {
    return len(ssn) == 11 && substring(ssn, 3, 4) == "-";
}

data Employee {
    name: string,
    ssn: string require isValidSSN(ssn)
}
```

## Computed Properties

> **Status:** ✅ Implemented

Computed properties are automatically calculated fields that derive their values from other fields. They are evaluated lazily when accessed.

```zexus
data Rectangle {
    width: number,
    height: number,
    
    computed area => width * height,
    computed perimeter => 2 * (width + height)
}

let rect = Rectangle(10, 20);
print(rect.area);       // 200 (computed on access)
print(rect.perimeter);  // 60 (computed on access)
```

### Features:
- **Lazy Evaluation**: Computed only when accessed, not on construction
- **Access to Fields**: Can reference other fields by name
- **Expression-Based**: Use any valid Zexus expression
- **No Caching**: Re-computed on each access (future: add caching option)

### Examples:

```zexus
// Circle with multiple computed properties
data Circle {
    radius: number,
    computed diameter => radius * 2,
    computed circumference => 2 * 3.14159 * radius,
    computed area => 3.14159 * radius * radius
}

let circle = Circle(5);
log circle.diameter;       // 10
log circle.circumference;  // 31.4159
log circle.area;           // 78.53975

// Person with computed full name
data Person {
    firstName: string,
    lastName: string,
    computed fullName => firstName + " " + lastName
}

let person = Person("John", "Doe");
log person.fullName;  // "John Doe"
```

## Method Definitions

> **Status:** ✅ Implemented

Custom methods can be defined inside data blocks using the `method` keyword. Methods have access to instance fields via the `this` keyword.

```zexus
data Rectangle {
    width: number,
    height: number,
    
    method area() {
        return this.width * this.height;
    }
    
    method scale(factor) {
        return Rectangle(this.width * factor, this.height * factor);
    }
    
    method describe() {
        return "Rectangle " + string(this.width) + "x" + string(this.height);
    }
}

let rect = Rectangle(10, 20);
print rect.describe();  // "Rectangle 10x20"
print rect.area();      // 200

let scaled = rect.scale(2);
print scaled.describe(); // "Rectangle 20x40"
```

### Features:
- **this Keyword**: Access instance fields via `this.fieldName`
- **Parameters**: Methods can accept parameters
- **Return Values**: Use `return` statement to return values
- **Instance Methods**: Methods are bound to each instance
- **Works with Modifiers**: Compatible with `immutable` and `verified` data types

### Examples:

```zexus
// Counter with state-modifying methods
data Counter {
    value: number,
    
    method increment() {
        this.value = this.value + 1;
        return this.value;
    }
    
    method decrement() {
        this.value = this.value - 1;
        return this.value;
    }
    
    method reset() {
        this.value = 0;
        return this.value;
    }
}

let counter = Counter(10);
print counter.increment();  // 11
print counter.increment();  // 12
print counter.decrement();  // 11

// Methods with parameters
data Calculator {
    value: number,
    
    method add(x) {
        return this.value + x;
    }
    
    method multiply(x) {
        return this.value * x;
    }
}

let calc = Calculator(10);
print calc.add(5);       // 15
print calc.multiply(3);  // 30
```

## Operator Overloading

> **Status:** ✅ Implemented

Custom operators can be defined inside data blocks using the `operator` keyword. This allows dataclasses to define custom behavior for arithmetic, comparison, and logical operators.

```zexus
data Vector {
    x: number,
    y: number,
    
    operator +(other) {
        return Vector(this.x + other.x, this.y + other.y);
    }
    
    operator -(other) {
        return Vector(this.x - other.x, this.y - other.y);
    }
    
    operator *(scalar) {
        return Vector(this.x * scalar, this.y * scalar);
    }
    
    operator ==(other) {
        return this.x == other.x && this.y == other.y;
    }
}

let v1 = Vector(10, 20);
let v2 = Vector(5, 15);

let sum = v1 + v2;       // Vector(15, 35)
let diff = v1 - v2;      // Vector(5, 5)
let scaled = v1 * 2;     // Vector(20, 40)

if (v1 == v1) {
    print "Equal!";       // "Equal!"
}
```

### Supported Operators:
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparison**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Logical**: `&&`, `||` (not recommended for overloading)

### Features:
- **Custom Semantics**: Define what operators mean for your types
- **Type Safety**: Operators are checked at runtime
- **this Keyword**: Access instance fields via `this.fieldName`
- **Parameters**: Operator methods receive the right operand as parameter
- **Return Values**: Can return any type (typically same type for arithmetic)

### Examples:

```zexus
// Complex number arithmetic
data Complex {
    real: number,
    imag: number,
    
    operator +(other) {
        return Complex(this.real + other.real, this.imag + other.imag);
    }
    
    operator *(other) {
        // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        return Complex(
            this.real * other.real - this.imag * other.imag,
            this.real * other.imag + this.imag * other.real
        );
    }
}

let c1 = Complex(3, 4);
let c2 = Complex(1, 2);
let product = c1 * c2;    // Complex(-5, 10)

// Matrix operations
data Matrix2x2 {
    a: number, b: number,
    c: number, d: number,
    
    operator +(other) {
        return Matrix2x2(
            this.a + other.a, this.b + other.b,
            this.c + other.c, this.d + other.d
        );
    }
    
    operator *(other) {
        return Matrix2x2(
            this.a * other.a + this.b * other.c,
            this.a * other.b + this.b * other.d,
            this.c * other.a + this.d * other.c,
            this.c * other.b + this.d * other.d
        );
    }
}

// Money comparisons
data Money {
    amount: number,
    currency: string,
    
    operator ==(other) {
        return this.amount == other.amount && this.currency == other.currency;
    }
    
    operator <(other) {
        require(this.currency == other.currency, "Currency mismatch");
        return this.amount < other.amount;
    }
}

let price1 = Money(100, "USD");
let price2 = Money(150, "USD");

if (price1 < price2) {
    print "price1 is cheaper";
}
```

## Inheritance

> **Status:** ✅ Implemented

Dataclasses can extend other dataclasses using the `extends` keyword. Child classes inherit all fields, methods, computed properties, and operators from their parent, and can add new ones or override existing methods.

```zexus
data Animal {
    name: string,
    age: number,
    
    method speak() {
        return this.name + " makes a sound";
    }
    
    method info() {
        return this.name + " is " + string(this.age) + " years old";
    }
}

data Dog extends Animal {
    breed: string,
    
    method speak() {
        // Override parent method
        return this.name + " barks!";
    }
    
    method fetchBall() {
        // New method specific to Dog
        return this.name + " fetches the ball";
    }
}

let animal = Animal("Generic", 5);
print animal.speak();    // "Generic makes a sound"
print animal.info();     // "Generic is 5 years old"

let dog = Dog("Buddy", 3, "Golden Retriever");
print dog.speak();       // "Buddy barks!" (overridden)
print dog.info();        // "Buddy is 3 years old" (inherited)
print dog.fetchBall();   // "Buddy fetches the ball" (new method)
```

### Features:
- **Field Inheritance**: Child classes inherit all parent fields
- **Method Inheritance**: Child classes inherit all parent methods
- **Method Overriding**: Child methods override parent methods with same name
- **Multi-level Inheritance**: Support for inheritance chains (A → B → C)
- **Computed Properties**: Inherited from parent classes
- **Operator Inheritance**: Operators are inherited from parent
- **Constructor Arguments**: Parent fields come first, then child fields

### Examples:

```zexus
// Multi-level inheritance
data Vehicle {
    brand: string,
    year: number,
    
    method describe() {
        return string(this.year) + " " + this.brand;
    }
}

data Car extends Vehicle {
    doors: number,
    
    method honk() {
        return "Beep beep!";
    }
}

data ElectricCar extends Car {
    batteryCapacity: number,
    
    method charge() {
        return "Charging: " + string(this.batteryCapacity) + " kWh";
    }
}

let ev = ElectricCar("Tesla", 2024, 4, 100);
print ev.describe();     // "2024 Tesla" (from Vehicle)
print ev.honk();         // "Beep beep!" (from Car)
print ev.charge();       // "Charging: 100 kWh" (from ElectricCar)

// Inheritance with computed properties
data Shape {
    x: number,
    y: number,
    
    method getPosition() {
        return "(" + string(this.x) + ", " + string(this.y) + ")";
    }
}

data Rectangle extends Shape {
    width: number,
    height: number,
    
    computed area => this.width * this.height,
    computed perimeter => 2 * (this.width + this.height)
}

let rect = Rectangle(10, 20, 5, 3);
print rect.getPosition();  // "(10, 20)" (inherited method)
print rect.area;           // 15 (computed property)

// Inheritance with operator overloading
data Point2D {
    x: number,
    y: number,
    
    operator +(other) {
        return Point2D(this.x + other.x, this.y + other.y);
    }
}

data Point3D extends Point2D {
    z: number,
    
    operator +(other) {
        // Override to support 3D addition
        return Point3D(this.x + other.x, this.y + other.y, this.z + other.z);
    }
}

let p1 = Point3D(1, 2, 3);
let p2 = Point3D(4, 5, 6);
let sum = p1 + p2;  // Point3D(5, 7, 9)
```

## Decorators

> **Status:** ✅ Implemented

Decorators are special annotations that modify the behavior of dataclasses and methods. They are prefixed with `@` and placed before the target.

### Method Decorators

#### @logged - Automatic Logging

Logs method calls and their arguments automatically:

```zexus
data Calculator {
    value: number,
    
    @logged
    method add(x) {
        return this.value + x;
    }
}

let calc = Calculator(10);
calc.add(5);  // Logs: "📝 Calling add(5)"
```

#### @cached - Result Caching

Caches method results based on arguments to avoid redundant computation:

```zexus
data Fibonacci {
    @cached
    method calculate(n) {
        if (n <= 1) {
            return n;
        }
        return this.calculate(n - 1) + this.calculate(n - 2);
    }
}

let fib = Fibonacci();
fib.calculate(10);  // Computes and caches
fib.calculate(10);  // Returns cached (instant)
```

#### Multiple Decorators

```zexus
data Counter {
    count: number = 0,
    
    @logged
    @cached
    method increment() {
        this.count = this.count + 1;
        return this.count;
    }
}
```

### Class Decorators

#### @validated - Enhanced Validation

```zexus
@validated
data Email {
    address: string require /^[\w\.-]+@[\w\.-]+\.\w+$/,
    verified: bool = false
}

let email = Email("test@example.com", true);  // ✅ Valid
```

### Features:
- **@logged**: Automatic method call logging
- **@cached**: Result caching with argument-based keys
- **@validated**: Enhanced validation for dataclasses
- **Multiple Decorators**: Stack multiple decorators
- **Zero Overhead**: Only applied when decorator is present

## Future Enhancements - Not Yet Implemented

### Composition (Recommended Pattern)

```zexus
data Timestamped {
    createdAt: number,
    updatedAt: number
}

data Post {
    title: string,
    content: string,
    metadata: Timestamped
}

let post = Post(
    "Hello World",
    "First post!",
    Timestamped(timestamp(), timestamp())
);
```

## Best Practices

### 1. Use PascalCase for Type Names

```zexus
// ✅ Good
data UserAccount { ... }
data PaymentInfo { ... }

// ❌ Bad
data user_account { ... }
data paymentinfo { ... }
```

### 2. Prefer Immutability for Value Objects

```zexus
// ✅ Good for coordinates, money, dates, etc.
data immutable Point { x: number, y: number }
data immutable Money { amount: number, currency: string }

// ❌ Bad - mutable value objects lead to bugs
data Point { x: number, y: number }
```

### 3. Use Validation for Critical Data

```zexus
// ✅ Good - validate early
data Email {
    address: string require /^[\w\.-]+@[\w\.-]+\.\w+$/
}

// ❌ Bad - validation scattered in code
data Email {
    address: string
}
// ... later: if (!isValidEmail(email.address)) { ... }
```

### 4. Leverage Default Values

```zexus
// ✅ Good - sensible defaults
data HttpRequest {
    method: string = "GET",
    timeout: number = 30000,
    retry: bool = true
}

// ❌ Bad - forcing users to specify everything
data HttpRequest {
    method: string,
    timeout: number,
    retry: bool
}
```

### 5. Use Composition Over Inheritance

```zexus
// ✅ Good - flexible composition
data User {
    name: string,
    profile: UserProfile
}

data UserProfile {
    bio: string,
    avatar: string
}

// ❌ Bad - deep inheritance hierarchies
data User { name: string }
data PowerUser extends User { permissions: array }
data AdminUser extends PowerUser { level: number }
```

## Comparison with Manual Classes

### Manual Approach (Before DATA)

```zexus
function User(name, email, age) {
    return {
        name: name,
        email: email,
        age: age,
        toString: function() {
            return "User(" + this.name + ", " + this.email + ")";
        },
        toJSON: function() {
            return '{"name":"' + this.name + '","email":"' + this.email + '","age":' + this.age + '}';
        }
    };
}

let user = User("Alice", "alice@example.com", 30);
```

### DATA Keyword (New Approach)

```zexus
data User {
    name: string,
    email: string,
    age: number
}

let user = User("Alice", "alice@example.com", 30);
// All methods auto-generated!
```

**Benefits:**
- 90% less boilerplate code
- Automatic validation
- Type safety
- Built-in serialization
- Pattern matching support
- Consistent API across all data types

## Error Handling

```zexus
data SafeUser {
    name: string require len(name) > 0,
    age: number require age >= 0
}

try {
    let user = SafeUser("", -5);  // Multiple validation errors
} catch (error) {
    print("Validation failed: " + error);
    // "Validation failed: Field 'name' constraint violated"
}
```

## Performance Considerations

1. **Immutable types are optimized**: The compiler can make assumptions about immutable data
2. **Validation runs on construction**: Not on every field access
3. **Serialization is cached**: toJSON() results are memoized when possible
4. **Pattern matching is compiled**: Match statements are optimized at parse time

## Integration with Existing Features

### With LOG Keyword

```zexus
data Config {
    host: string,
    port: number
}

log >> "config_gen.zx";
print("data Config {");
print("    host: string = \"localhost\",");
print("    port: number = 8080");
print("}");
log > "/dev/stdout";

// Load and use generated config
log << "config_gen.zx";
let config = Config();
```

### With Async/Await

```zexus
data async Response {
    status: number,
    body: string,
    headers: map
}

async function fetchData(url) {
    let response = await fetch(url);
    return Response(
        response.status,
        await response.text(),
        response.headers
    );
}

let data = await fetchData("https://api.example.com");
```

### With Blockchain

```zexus
data verified Block {
    index: number,
    timestamp: number,
    data: string,
    previousHash: hash,
    nonce: number
}

function mineBlock(block) {
    while (!block.hash().startsWith("0000")) {
        block.nonce++;
    }
    return block;
}
```

## Examples

### Example 1: User Management

```zexus
data immutable User {
    id: number,
    username: string require /^[a-zA-Z0-9_]{3,20}$/,
    email: string require /^[\w\.-]+@[\w\.-]+\.\w+$/,
    createdAt: number = timestamp()
}

data UserDatabase {
    users: array = []
}

let db = UserDatabase();
let user1 = User(1, "alice_99", "alice@example.com");
let user2 = User(2, "bob_42", "bob@example.com");

db.users.push(user1);
db.users.push(user2);

print(db.users[0].username);  // "alice_99"
```

### Example 2: API Response Handling

```zexus
data Result {
    success: bool,
    data: any = null,
    error: string = "",
    statusCode: number = 200
}

function apiCall(endpoint) {
    try {
        let response = fetch(endpoint);
        return Result(true, response.data, "", response.status);
    } catch (error) {
        return Result(false, null, error.message, 500);
    }
}

let result = apiCall("/api/users");
match result {
    Result(true, data, _, _) => print("Success: " + data),
    Result(false, _, error, code) => print("Error " + code + ": " + error)
}
```

### Example 3: Blockchain Transaction

```zexus
data verified Transaction {
    from: address,
    to: address,
    amount: number require amount > 0,
    fee: number = 0,
    nonce: number,
    signature: hash
}

data Block {
    index: number,
    timestamp: number,
    transactions: array,
    previousHash: hash,
    hash: hash = ""
}

let tx = Transaction(
    "0x1234...",
    "0x5678...",
    100,
    1,
    0,
    "0xabcd..."
);

verify(tx) {
    require(tx.verify(), "Invalid signature");
    require(tx.amount > tx.fee, "Amount must exceed fee");
}

let block = Block(1, timestamp(), [tx], "0x0000...");
```

## Generic Types

> **Status:** ✅ Implemented

Generic types allow you to create reusable dataclasses that work with different types. Type parameters are specified in angle brackets and substituted when the type is instantiated.

### Basic Generic Type

```zexus
data Box<T> {
    value: T
}

let numBox = Box<number>(42);
print(numBox.value);  // 42

let strBox = Box<string>("hello");
print(strBox.value);  // hello
```

### Multiple Type Parameters

```zexus
data Pair<K, V> {
    key: K,
    value: V
}

let pair = Pair<string, number>("age", 25);
print(pair.key);    // age
print(pair.value);  // 25

data Triple<A, B, C> {
    first: A,
    second: B,
    third: C
}

let triple = Triple<string, number, bool>("test", 42, true);
```

### Generic with Methods

```zexus
data Container<T> {
    item: T,
    
    method get() {
        return this.item;
    },
    
    method set(newItem) {
        this.item = newItem;
    }
}

let container = Container<number>(100);
print(container.get());  // 100
```

### Generic with Computed Properties

```zexus
data Circle<T> {
    radius: T,
    
    computed area => this.radius * this.radius * 3.14159
}

let circle = Circle<number>(5);
print(circle.area);  // ~78.54
```

### Generic with Operator Overloading

```zexus
data Point<T> {
    x: T,
    y: T,
    
    operator +(other) {
        return Point<number>(this.x + other.x, this.y + other.y);
    }
}

let p1 = Point<number>(3, 4);
let p2 = Point<number>(1, 2);
let p3 = p1 + p2;
print(p3.x);  // 4
print(p3.y);  // 6
```

### Generic with Validation

```zexus
data Validated<T> {
    value: T require value != null
}

let validated = Validated<number>(100);
```

### Features:
- **Type Parameters**: Single or multiple (`T`, `K, V`, `A, B, C`)
- **Type Substitution**: Automatic replacement of type parameters
- **Full Feature Support**: Works with methods, computed properties, operators, validation
- **Type Safety**: Validates field types match the specified type argument
- **Caching**: Specialized types are cached for performance

### Limitations:
- Type arguments must be provided explicitly (no type inference yet)
- Type parameters cannot have constraints (e.g., `T extends Number`)
- Nested generic instantiation not yet supported (e.g., `Box<Pair<A, B>>`)

## Pattern Matching

> **Status:** ✅ Implemented

Pattern matching allows you to destructure and match against dataclass instances using powerful pattern syntax.

### Basic Pattern Matching

```zexus
data Point { x: number, y: number }

let p = Point(10, 20);

let result = match p {
    Point(0, 0) => "origin",
    Point(x, 0) => "on x-axis at " + x,
    Point(0, y) => "on y-axis at " + y,
    Point(x, y) => "at (" + x + ", " + y + ")"
};

print(result);  // "at (10, 20)"
```

### Wildcard Pattern

```zexus
data Shape {
    type: string,
    size: number
}

let shape = Shape("circle", 50);

let description = match shape {
    Shape("circle", _) => "It's a circle!",
    Shape("square", _) => "It's a square!",
    Shape(_, size) => "Unknown shape of size " + size
};

print(description);  // "It's a circle!"
```

### Variable Binding

```zexus
data User {
    name: string,
    age: number
}

let user = User("Alice", 30);

let greeting = match user {
    User("Bob", _) => "Hello Bob!",
    User(name, age) => "Hello " + name + ", you are " + age
};

print(greeting);  // "Hello Alice, you are 30"
```

### Literal Pattern Matching

```zexus
data Config {
    mode: string,
    port: number
}

let config = Config("production", 8080);

let status = match config {
    Config("production", 8080) => "Production mode on default port",
    Config("production", port) => "Production on port " + port,
    Config("development", _) => "Development mode",
    Config(mode, _) => mode + " mode"
};
```

### Complex Patterns with Computation

```zexus
data Rectangle {
    width: number,
    height: number,
    
    computed area => this.width * this.height
}

let rect = Rectangle(10, 20);

let info = match rect {
    Rectangle(w, h) => "Area: " + (w * h)
};

print(info);  // "Area: 200"
```

### Pattern Matching with Generic Types

```zexus
data Box<T> {
    value: T
}

let numBox = Box<number>(42);

let result = match numBox {
    Box(0) => "empty",
    Box(value) => "contains " + value
};

print(result);  // "contains 42"
```

### Features:
- **Constructor Patterns**: Match dataclass constructors `Point(x, y)`
- **Wildcard Patterns**: Use `_` to ignore values
- **Variable Binding**: Bind matched values to variables
- **Literal Matching**: Match exact values `Point(0, 0)`
- **Field Extraction**: Automatically extract field values in order
- **Generic Support**: Works with generic dataclasses
- **Computed Properties**: Access computed fields in patterns

### Pattern Syntax:
- **Type(field1, field2, ...)**: Constructor pattern with field bindings
- **Type(literal, ...)**: Match literal field values
- **Type(var, ...)**: Bind fields to variables
- **_**: Wildcard (matches anything, doesn't bind)

## Future Enhancements

Planned features for upcoming versions:

1. **Type Inference**: Automatic type parameter deduction from arguments
2. **Type Constraints**: `data Box<T extends Comparable> { ... }`
3. **Nested Generics**: `Box<Pair<string, number>>`
4. **Nested Patterns**: `Box(Point(x, y))` for nested destructuring
5. **Guard Clauses**: `Point(x, y) if x > 0 => ...`
6. **Array Patterns**: `[first, ...rest]` destructuring
7. **Private/Public Modifiers**: `private field: string`
8. **Readonly Fields**: Prevent mutation after construction

> **Current Implementation**: Production-grade dataclass with type validation, constraints, auto-generated methods (toString, toJSON, clone, equals, hash, verify), immutability, verification support, static default() method, computed properties, custom method definitions, operator overloading, inheritance with extends, decorators (@logged, @cached, @validated), generic types with full type substitution, and pattern matching with constructor patterns and wildcard support.

## See Also

- [CONST Keyword](CONST.md) - Immutable variables
- [LET Keyword](LET.md) - Variable declarations
- [LOG Keyword](LOG.md) - Code generation
- [BLOCKCHAIN Features](BLOCKCHAIN_FEATURES.md) - Verification and hashing
- [Type System](../TYPE_SYSTEM.md) - Type definitions and checking

## Currently Implemented Features

### ✅ Core Functionality
- [x] Basic dataclass definitions
- [x] Field type annotations (string, number, bool, array, map, any)
- [x] Default values for fields
- [x] Constraint validation (require clause)
- [x] Immutable modifier
- [x] Verified modifier

### ✅ Auto-Generated Methods
- [x] `toString()` - String representation
- [x] `toJSON()` - JSON serialization
- [x] `clone()` - Deep copy creation
- [x] `equals(other)` - Deep equality check
- [x] `hash()` - Cryptographic hash (verified types only)
- [x] `verify()` - Constraint re-validation (verified types only)

### ✅ Static Methods
- [x] `fromJSON(jsonString)` - Deserialize from JSON
- [x] `default()` - Create instance with all default values

### ✅ Type System
- [x] Type validation on construction
- [x] Runtime type checking
- [x] Constraint evaluation with field scope
- [x] Nested dataclass support

### ✅ Computed Properties
- [x] Lazy evaluation on property access
- [x] Access to instance fields in expressions
- [x] Support for complex expressions
- [x] Works with immutable and verified modifiers

### ✅ Method Definitions
- [x] Custom methods with `method` keyword
- [x] `this` keyword for accessing instance fields
- [x] Methods with parameters
- [x] Return values from methods
- [x] Instance method binding
- [x] Works with immutable and verified modifiers

### ✅ Operator Overloading
- [x] Custom operator definitions with `operator` keyword
- [x] Arithmetic operators (+, -, *, /, %)
- [x] Comparison operators (==, !=, <, >, <=, >=)
- [x] Access to `this` in operator methods
- [x] Return custom types from operators
- [x] Works with all data modifiers

### ✅ Inheritance
- [x] Extend parent dataclasses with `extends` keyword
- [x] Inherit fields from parent classes
- [x] Inherit methods from parent classes
- [x] Method overriding (child overrides parent)
- [x] Multi-level inheritance support (A → B → C)
- [x] Computed properties inheritance
- [x] Operator inheritance and overriding
- [x] Proper constructor argument ordering (parent fields first)

### ✅ Decorators
- [x] `@logged` decorator for automatic method call logging
- [x] `@cached` decorator for result caching
- [x] `@validated` decorator for class-level validation
- [x] Multiple decorator stacking
- [x] Method decorators
- [x] Class decorators
- [x] Zero overhead when not used

### ✅ Generic Types
- [x] Type parameter syntax (`data Box<T>`)
- [x] Multiple type parameters (`data Pair<K, V>`)
- [x] Type substitution in field definitions
- [x] Generic methods and computed properties
- [x] Generic operator overloading
- [x] Generic validation constraints
- [x] Specialized type caching for performance
- [x] Works with all data modifiers

### ✅ Pattern Matching
- [x] Match expressions with `match value { ... }`
- [x] Constructor patterns (`Point(x, y)`)
- [x] Wildcard patterns (`_` to ignore values)
- [x] Variable binding in patterns
- [x] Literal value matching
- [x] Field extraction in declaration order
- [x] Works with generic types
- [x] Works with computed properties

### ✅ Integration
- [x] Works with keyword-after-dot feature
- [x] Map-based implementation with String keys
- [x] Environment integration (const registration)
- [x] Error handling and validation messages

## Future Features (Not Yet Implemented)

### ⏳ Planned for Phase 3
- [ ] Nested pattern matching (`Box(Point(x, y))`)
- [ ] Pattern guard clauses (`Point(x, y) if x > 0`)
- [ ] Array/List pattern matching (`[first, ...rest]`)
- [ ] Type inference for generics (`Box(42)` infers `Box<number>`)
- [ ] Type constraints for generics (`data Box<T extends Comparable>`)
- [ ] Nested generic types (`Box<Pair<string, number>>`)
- [ ] Auto-documentation generation
- [ ] Private/public field modifiers (enhanced)
- [ ] Readonly fields (prevent mutation after construction)
- [ ] Abstract dataclasses (cannot be instantiated directly)
- [ ] Interface implementation (`data User implements Serializable`)

---

**Status**: ✅ Production Ready (Core Features)  
**Version**: 1.0.0  
**Last Updated**: December 24, 2025
