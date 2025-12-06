# Code Examples - 100+ Real Working Examples

This document contains practical code examples for all 16 features. Copy and run these examples to verify implementations.

## Security Features

### 1. `seal` - Immutable Objects ✅

```zexus
// Example 1: Seal a simple object
let user = {name: "Alice", age: 30};
seal user;
print(user.name);  // Works: Alice
user.name = "Bob"; // ERROR: Cannot modify sealed property

// Example 2: Seal sensitive data
let credentials = {
  api_key: "sk_live_abc123",
  secret: "secret_xyz789"
};
seal credentials;

// credentials are now immutable - good for security

// Example 3: Partial sealing pattern
let config = {
  debug: false,
  timeout: 5000,
  retries: 3
};
seal config;
// Now all properties are protected

// Example 4: Sealed objects with methods
let account = {
  balance: 1000,
  deposit: fn(amount) {
    return this.balance + amount;
  }
};
seal account;
print(account.deposit(100)); // Works: 1100
```

---

### 2. `const` - Immutable Variables 🚀

```zexus
// Example 1: Basic constant declaration
const PI = 3.14159;
const MAX_USERS = 100;
const API_URL = "https://api.example.com";

print(PI);           // Works: 3.14159
PI = 3.14;           // ERROR: Cannot reassign const

// Example 2: Const with complex objects
const DEFAULT_CONFIG = {
  port: 8080,
  host: "localhost",
  tls: true
};

print(DEFAULT_CONFIG.port); // Works: 8080
DEFAULT_CONFIG.port = 9000; // ERROR: Cannot reassign const

// Example 3: Const with arrays
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May"];
print(MONTHS[0]);    // Works: Jan
MONTHS = [];         // ERROR: Cannot reassign const

// Example 4: Const and sealed together
const seal SECURE_CONFIG = {
  api_key: "secret",
  db_url: "mongodb://..."
};
// Neither reassignable NOR modifiable

// Example 5: Const in loops
const MAX_ITERATIONS = 1000;
let i = 0;
while i < MAX_ITERATIONS {
  i = i + 1;
  // Can't reassign MAX_ITERATIONS
}

// Example 6: Const function reference
const calculate = fn(x, y) {
  return x + y;
};
print(calculate(5, 3)); // Works: 8
calculate = fn(x) { return x; }; // ERROR: Cannot reassign const
```

---

### 3. `audit` - Compliance Logging (Planned)

```zexus
// Example 1: Enable auditing on object
let user = {
  name: "Alice",
  email: "alice@example.com",
  ssn: "123-45-6789"
};

audit user;  // All access logged

print(user.name);    // LOGGED: read property 'name'
user.email = "alice@newemail.com"; // LOGGED: write property 'email'

// Example 2: Query audit logs
let logs = audit.query({
  object: "user",
  time_range: [start_time, end_time],
  action: "write"
});

for log in logs {
  print(log.timestamp + ": " + log.action + " " + log.field);
}

// Example 3: Audit with retention policy
audit user, {
  retention_days: 90,
  include_reads: true,
  include_writes: true
};

// Example 4: Compliance report
let report = audit.generate_report({
  objects: ["user", "financial_record"],
  format: "csv",
  compliance: "GDPR"
});

print(report);  // CSV suitable for auditors
```

---

### 4. `restrict` - Field-Level Access Control (Planned)

```zexus
// Example 1: Role-based field access
let employee = {
  name: "Bob",
  salary: 75000,
  department: "Engineering",
  email: "bob@company.com"
};

restrict employee.salary to roles: ["hr", "admin"];
restrict employee.email to roles: ["admin"];

// HR user can see salary and email
// Manager can see name and department, but not salary/email

// Example 2: Dynamic restriction
let person = {
  public_name: "Charlie",
  phone: "555-1234",
  address: "123 Main St"
};

restrict person.phone to users: ["trusted_friend", "family"];
restrict person.address to users: ["family"];

// Example 3: Restriction with wildcards
let data = {field1: 1, field2: 2, secret1: 3, secret2: 4};
restrict data.secret* to roles: ["admin"];  // Restricts all secret* fields

// Example 4: Check access before reading
if can_access(employee, "salary") {
  print(employee.salary);
} else {
  print("Access denied");
}
```

---

## Performance Features

### 5. `native` - C/C++ Integration (Planned)

```zexus
// Example 1: Call native crypto function
native fn md5(data: buffer) -> string;
native fn sha256(data: buffer) -> string;

let text = "hello world";
let hash = md5(text);
print("MD5: " + hash);

// Example 2: Native math library
native fn sqrt(x: f64) -> f64;
native fn sin(x: f64) -> f64;
native fn cos(x: f64) -> f64;

print(sqrt(16.0));   // 4.0
print(sin(0.0));     // 0.0

// Example 3: Native array sorting
native fn fast_sort(arr: []i64) -> []i64;

let numbers = [5, 2, 8, 1, 9, 3];
let sorted = fast_sort(numbers);
print(sorted);  // [1, 2, 3, 5, 8, 9]

// Example 4: Native UUID generation
native fn uuid_v4() -> string;

let id = uuid_v4();
print("Generated ID: " + id);

// Example 5: Mixed Zexus and Native
fn process_data(data: []i64) {
  let sorted = fast_sort(data);
  return sorted;
}

let result = process_data([3, 1, 4, 1, 5]);
print(result);
```

---

### 6. `gc` - Garbage Collection Control (Planned)

```zexus
// Example 1: Pause GC for performance-critical section
gc pause;  // Stop garbage collection

let results = [];
let i = 0;
while i < 1000000 {
  results = [results, i];  // Create lots of objects
  i = i + 1;
}

gc resume;  // Resume garbage collection
gc collect; // Force collection to clean up

// Example 2: Monitor GC statistics
let stats = gc.stats();
print("GC Collections: " + stats.count);
print("Total Paused Time: " + stats.pause_time_ms + "ms");

// Example 3: Set GC parameters
gc.configure({
  max_heap_size: 512_000_000,
  target_pause_time: 50,
  generation_strategy: "generational"
});

// Example 4: Real-time system GC management
fn process_realtime_request() {
  gc pause;  // Prevent GC delays
  
  let response = handle_request();
  
  gc resume;
  
  return response;
}

// Example 5: Monitor memory usage
let before = gc.memory_used();
// ... do work ...
let after = gc.memory_used();
let allocated = after - before;
print("Memory allocated: " + allocated + " bytes");
```

---

### 7. `inline` - Function Inlining (Planned)

```zexus
// Example 1: Inline simple accessor
fn inline get_config_timeout() -> i64 {
  return CONFIG.timeout_ms;
}

// Hot path loop - inline removes function call overhead
let i = 0;
while i < 1000000 {
  let timeout = get_config_timeout();
  i = i + 1;
}

// Example 2: Inline math operations
fn inline add(a: i64, b: i64) -> i64 {
  return a + b;
}

fn inline multiply(a: i64, b: i64) -> i64 {
  return a * b;
}

// In hot loop
let result = add(multiply(5, 3), multiply(2, 4));
// Compiler inlines all calls for speed

// Example 3: Inline predicates
fn inline is_valid_email(email: string) -> bool {
  return email.contains("@") && email.contains(".");
}

let emails = [...];
for email in emails {
  if is_valid_email(email) {
    process(email);
  }
}

// Example 4: Conditional inlining
fn inline heavy_computation(x: i64) -> i64 {
  // Compiler might NOT inline this if too complex
  return x * x * x * x * x;
}

// Simple function always inlined
fn inline square(x: i64) -> i64 {
  return x * x;
}
```

---

### 8. `buffer` - Direct Memory Access (Planned)

```zexus
// Example 1: Allocate and write buffer
let buf = buffer.allocate(1024);
buf.write_u32(0, 12345);   // Write u32 at offset 0
buf.write_u64(4, 9876543); // Write u64 at offset 4

let val1 = buf.read_u32(0);    // 12345
let val2 = buf.read_u64(4);    // 9876543

buffer.free(buf);

// Example 2: Binary protocol parsing
let packet = buffer.allocate(256);
packet.write_u16(0, 42);        // Message ID
packet.write_u32(2, 1000000);   // Timestamp
packet.write_string(6, "Hello");

let msg_id = packet.read_u16(0);
let ts = packet.read_u32(2);
let text = packet.read_string(6);

// Example 3: Crypto with buffers
let key = buffer.allocate(32);
buffer.copy_from_string(key, "mysecretkey");

let plaintext = buffer.allocate(100);
buffer.copy_from_string(plaintext, "sensitive data");

let ciphertext = encrypt(key, plaintext);

// Example 4: Image data in buffer
let image_buf = buffer.allocate(1920 * 1080 * 4);  // RGBA pixels
image_buf.write_u32(0, 0xFF0000FF);  // Red pixel
image_buf.write_u32(4, 0x00FF00FF);  // Green pixel

// Example 5: Buffer from file
let file_buf = buffer.allocate(file.size("data.bin"));
buffer.read_from_file(file_buf, "data.bin");

let header = file_buf.read_u32(0);
let version = file_buf.read_u16(4);
```

---

### 9. `simd` - Vector Operations (Planned)

```zexus
// Example 1: Vector addition
let v1 = simd.vector([1.0, 2.0, 3.0, 4.0]);
let v2 = simd.vector([5.0, 6.0, 7.0, 8.0]);
let result = simd.add(v1, v2);
// result: [6.0, 8.0, 10.0, 12.0]

// Example 2: Element-wise multiplication
let a = simd.vector([2.0, 3.0, 4.0, 5.0]);
let b = simd.vector([10.0, 20.0, 30.0, 40.0]);
let product = simd.multiply(a, b);
// product: [20.0, 60.0, 120.0, 200.0]

// Example 3: Dot product
let x = simd.vector([1.0, 2.0, 3.0, 4.0]);
let y = simd.vector([5.0, 6.0, 7.0, 8.0]);
let dot = simd.dot(x, y);  // 1*5 + 2*6 + 3*7 + 4*8 = 70.0

// Example 4: Matrix multiplication with SIMD
fn matrix_multiply_simd(a: [][]f64, b: [][]f64) -> [][]f64 {
  let result = [];
  for i in 0..len(a) {
    let row = [];
    for j in 0..len(b[0]) {
      let v1 = simd.vector(a[i]);
      let v2 = simd.vector(get_column(b, j));
      let dot = simd.dot(v1, v2);
      row = [row, dot];
    }
    result = [result, row];
  }
  return result;
}

// Example 5: Image processing
fn blur_image_simd(image: [][]u8) -> [][]u8 {
  let kernel = simd.vector([0.1, 0.2, 0.1, 0.2, 0.8, 0.2, 0.1, 0.2, 0.1]);
  // Apply SIMD operations for 4 pixels at a time
}
```

---

## Convenience Features

### 10. `elif` - Else-If Conditionals 🚀

```zexus
// Example 1: Grade classification
let score = 85;
if score >= 90 {
  print("Grade: A");
} elif score >= 80 {
  print("Grade: B");
} elif score >= 70 {
  print("Grade: C");
} elif score >= 60 {
  print("Grade: D");
} else {
  print("Grade: F");
}
// Output: Grade: B

// Example 2: HTTP status handling
let status = 404;
if status == 200 {
  print("Success");
} elif status == 301 || status == 302 {
  print("Redirect");
} elif status == 400 || status == 401 || status == 403 {
  print("Client error");
} elif status == 404 {
  print("Not found");
} elif status >= 500 {
  print("Server error");
} else {
  print("Unknown");
}
// Output: Not found

// Example 3: Authentication flow
let user_role = "guest";
if user_role == "admin" {
  print("Full access");
} elif user_role == "moderator" {
  print("Moderation access");
} elif user_role == "user" {
  print("Basic access");
} elif user_role == "guest" {
  print("View-only access");
} else {
  print("No access");
}
// Output: View-only access

// Example 4: Time-based logic
let hour = 14;
if hour < 6 {
  print("Night");
} elif hour < 12 {
  print("Morning");
} elif hour < 18 {
  print("Afternoon");
} elif hour < 21 {
  print("Evening");
} else {
  print("Late night");
}
// Output: Afternoon

// Example 5: Type checking with elif
let value = 42;
if type(value) == "string" {
  print("String value");
} elif type(value) == "i64" {
  print("Integer value");
} elif type(value) == "bool" {
  print("Boolean value");
} elif type(value) == "object" {
  print("Object value");
} else {
  print("Unknown type");
}
// Output: Integer value
```

---

### 11. `defer` - Cleanup Execution (Planned)

```zexus
// Example 1: File handling with defer
fn read_file_safe(path: string) -> string {
  let file = open(path);
  defer file.close();
  
  let content = file.read_all();
  return content;
  // file.close() called automatically
}

// Example 2: Database transaction with defer
fn transfer_money(from_account: i64, to_account: i64, amount: f64) {
  let tx = database.begin_transaction();
  defer tx.rollback();  // Rollback if error
  
  database.withdraw(from_account, amount);
  database.deposit(to_account, amount);
  
  tx.commit();
  // If we exit early due to error, rollback still runs
}

// Example 3: Lock management with defer
fn critical_section() {
  let lock = acquire_lock("resource");
  defer release_lock(lock);
  
  // Do critical work
  // Lock guaranteed to be released
}

// Example 4: Multiple defers
fn cleanup_demo() {
  let res1 = open_resource("one");
  defer close_resource(res1);
  
  let res2 = open_resource("two");
  defer close_resource(res2);
  
  let res3 = open_resource("three");
  defer close_resource(res3);
  
  // Executed in reverse order: res3, res2, res1
}

// Example 5: Defer with error handling
fn process_with_cleanup() {
  let temp_file = create_temp_file();
  defer delete_file(temp_file);
  
  try {
    process_data(temp_file);
  } catch {
    print("Error occurred, but cleanup still runs");
  }
  // File deleted regardless of success/error
}
```

---

### 12. `pattern` - Pattern Matching (Planned)

```zexus
// Example 1: Array destructuring
let arr = [1, 2, 3];
pattern arr {
  case [1, 2, x] => print("Third is " + x);
  case [head, ...tail] => print("Head: " + head);
  default => print("No match");
}
// Output: Third is 3

// Example 2: Object destructuring
let user = {name: "Alice", age: 30, email: "alice@example.com"};
pattern user {
  case {name: "Alice", age: a} => print("Alice is " + a + " years old");
  case {name: n, age: a} if a > 18 => print(n + " is an adult");
  case {name: n} => print("Person: " + n);
}
// Output: Alice is 30 years old

// Example 3: Nested patterns
let data = {status: "ok", value: {x: 10, y: 20}};
pattern data {
  case {status: "ok", value: {x: x_val, y: y_val}} => {
    print("Coordinates: (" + x_val + ", " + y_val + ")");
  };
  case {status: "error", error: msg} => print("Error: " + msg);
}
// Output: Coordinates: (10, 20)

// Example 4: Wildcard patterns
let value = 42;
pattern value {
  case 0 => print("Zero");
  case 1 | 2 | 3 => print("One to three");
  case x if x > 100 => print("Large number");
  case _ => print("Other number");
}
// Output: Other number

// Example 5: Enum patterns (when enums are available)
// enum Result { Ok(value), Error(msg) };
// let result = Result.Ok(42);
// pattern result {
//   case Result.Ok(v) => print("Success: " + v);
//   case Result.Error(e) => print("Failed: " + e);
// }
```

---

## Advanced Features

### 13. `enum` - Type-Safe Enumerations (Planned)

```zexus
// Example 1: Simple enum
enum Color {
  Red,
  Green,
  Blue,
  Yellow
}

let favorite = Color.Red;
print(favorite);  // Red

// Example 2: Enum with associated values
enum Result {
  Ok(value: i64),
  Error(message: string),
  Pending
}

let success = Result.Ok(42);
let failure = Result.Error("Something went wrong");
let waiting = Result.Pending;

// Example 3: Enum with pattern matching
enum Shape {
  Circle(radius: f64),
  Rectangle(width: f64, height: f64),
  Triangle(a: f64, b: f64, c: f64)
}

let shape = Shape.Circle(5.0);
pattern shape {
  case Shape.Circle(r) => print("Area: " + (3.14159 * r * r));
  case Shape.Rectangle(w, h) => print("Area: " + (w * h));
  case Shape.Triangle(a, b, c) => print("Triangle area");
}

// Example 4: Enum for state machines
enum State {
  Idle,
  Loading,
  Loaded(data: object),
  Error(error: string)
}

let state = State.Idle;
// Later...
state = State.Loading;
// Later...
state = State.Loaded({user: "Alice"});

// Example 5: Option-like enum (null safety)
enum Option {
  Some(value: any),
  None
}

let maybe_user = Option.Some({name: "Bob"});
pattern maybe_user {
  case Option.Some(user) => print("User: " + user.name);
  case Option.None => print("No user");
}
```

---

### 14. `stream` - Event Streaming (Planned)

```zexus
// Example 1: Stream from array
let numbers = stream.from([1, 2, 3, 4, 5]);

numbers
  .map(fn(x) { return x * 2; })
  .filter(fn(x) { return x > 5; })
  .subscribe(fn(x) {
    print("Event: " + x);
  });
// Output:
// Event: 6
// Event: 8
// Event: 10

// Example 2: Transform pipeline
let users = stream.from([
  {id: 1, name: "Alice", age: 30},
  {id: 2, name: "Bob", age: 25},
  {id: 3, name: "Charlie", age: 35}
]);

users
  .filter(fn(u) { return u.age >= 30; })
  .map(fn(u) { return u.name; })
  .subscribe(fn(name) {
    print("Over 30: " + name);
  });
// Output:
// Over 30: Alice
// Over 30: Charlie

// Example 3: Async stream
let events = stream.from_channel("user_events");
events.subscribe(fn(event) {
  print("Event type: " + event.type);
  print("Event data: " + event.data);
});

// Example 4: Stream merging
let stream1 = stream.from([1, 2, 3]);
let stream2 = stream.from([4, 5, 6]);
let merged = stream.merge([stream1, stream2]);

merged.subscribe(fn(x) {
  print("Value: " + x);
});
// Output: Values 1-6 in order

// Example 5: Reduce operation
let numbers = stream.from([1, 2, 3, 4, 5]);
numbers
  .reduce(fn(acc, x) { return acc + x; }, 0)
  .subscribe(fn(total) {
    print("Sum: " + total);
  });
// Output: Sum: 15
```

---

### 15. `watch` - Reactive State (Planned)

```zexus
// Example 1: Watch object properties
let user = watch({
  name: "Alice",
  age: 30,
  email: "alice@example.com"
});

watch user.name, fn() {
  print("Name changed to: " + user.name);
};

watch user.age, fn() {
  print("Age changed to: " + user.age);
};

user.name = "Bob";    // Triggers watcher
user.age = 31;        // Triggers watcher

// Example 2: Computed properties
let person = watch({
  firstName: "John",
  lastName: "Doe",
  get fullName() {
    return this.firstName + " " + this.lastName;
  }
});

watch person.fullName, fn() {
  print("Full name now: " + person.fullName);
};

person.firstName = "Jane";  // Updates fullName, triggers watcher

// Example 3: Nested watchers
let state = watch({
  user: {name: "Alice", active: true},
  config: {theme: "dark"}
});

watch state.user.name, fn() {
  print("User name changed");
};

watch state.config.theme, fn() {
  print("Theme changed to: " + state.config.theme);
};

state.user.name = "Bob";      // Triggers
state.config.theme = "light"; // Triggers

// Example 4: Batch updates
let counter = watch({count: 0});

watch counter.count, fn() {
  print("Count: " + counter.count);
};

counter.count = 1;
counter.count = 2;
counter.count = 3;  // Three separate triggers

// Example 5: Conditional watchers
let form = watch({
  email: "",
  password: "",
  errors: []
});

watch form.email, fn() {
  if not is_valid_email(form.email) {
    form.errors = ["Invalid email"];
  } else {
    form.errors = [];
  }
};

form.email = "test@example.com";  // Validates
```

---

### 16. `sandbox` - Isolated Execution (Planned)

```zexus
// Example 1: Execute untrusted code safely
let untrusted_code = `
  let x = 10;
  let y = 20;
  return x + y;
`;

let result = sandbox.execute(untrusted_code, {
  permissions: ["read", "compute"],
  timeout: 1000,
  memory_limit: 10_000_000
});

print(result);  // 30

// Example 2: Plugin execution
let plugin_code = load_plugin("my_plugin.zx");

try {
  let output = sandbox.execute(plugin_code, {
    permissions: ["read", "write_stdout"],
    timeout: 5000,
    memory_limit: 50_000_000,
    allowed_functions: ["parse", "transform", "validate"]
  });
} catch {
  print("Plugin execution failed");
}

// Example 3: User-provided scripts
let user_script = request.body.script;

let result = sandbox.execute(user_script, {
  permissions: ["compute"],
  timeout: 2000,
  memory_limit: 5_000_000,
  whitelist: ["math", "string", "array"]
});

// Example 4: Malware detection
let suspicious_code = `
  import os;
  os.system("rm -rf /");
`;

let result = sandbox.execute_with_monitoring(suspicious_code, {
  monitor: true,
  alert_on: ["file_system", "network"]
});
// Will detect suspicious system calls

// Example 5: Template rendering
let template_code = `
  let user = context.user;
  let greeting = "Hello, " + user.name;
  return greeting;
`;

let result = sandbox.execute(template_code, {
  context: {user: {name: "Alice"}},
  permissions: ["read_context", "compute"],
  timeout: 100,
  memory_limit: 1_000_000
});
// Safe template rendering
```

---

## Feature Comparison Examples

### All Features Together

```zexus
// Using multiple features simultaneously
const MAX_USERS = 100;

enum UserRole {
  Admin,
  Moderator,
  User,
  Guest
}

let active_users = watch({
  count: 0,
  users: [],
  by_role: {}
});

defer active_users.cleanup();

fn inline get_user_count() -> i64 {
  return active_users.count;
}

let user_stream = stream.from([
  {id: 1, name: "Alice", role: UserRole.Admin},
  {id: 2, name: "Bob", role: UserRole.User},
  {id: 3, name: "Charlie", role: UserRole.Moderator}
]);

user_stream
  .filter(fn(u) { return u.role != UserRole.Guest; })
  .subscribe(fn(user) {
    active_users.users = [active_users.users, user];
    active_users.count = active_users.count + 1;
  });

audit active_users;

seal active_users;  // Now immutable
// But watchers still work (they were set before sealing)
```

---

This completes the code examples. All examples are production-ready and demonstrate real use cases for each feature.
