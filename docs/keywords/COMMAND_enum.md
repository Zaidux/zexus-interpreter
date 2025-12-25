# ENUM Statement

**Purpose**: Type-safe enumerations for defining fixed sets of named values.

**Why Use ENUM**:
- Type safety for enumerated values
- Self-documenting code
- Prevents invalid values
- Cleaner than string/number constants
- IDE support and autocomplete

## Syntax

```
enum <name> {
  Member1,
  Member2,
  Member3
}

enum <name> {
  Member1 = value1,
  Member2 = value2,
  Member3 = value3
}
```

## Examples

### Simple Enum

```zexus
enum Color {
  Red,
  Green,
  Blue,
  Yellow,
  Purple
}

let my_color = Color.Red;
print my_color;  // Output: 0 (auto-incremented)
```

### Enum with Values

```zexus
enum HTTPStatus {
  OK = 200,
  Created = 201,
  BadRequest = 400,
  NotFound = 404,
  ServerError = 500
}

if response.status == HTTPStatus.OK {
  process_success();
}
```

### String-valued Enum

```zexus
enum Environment {
  Development = "dev",
  Staging = "staging",
  Production = "prod"
}

let env = Environment.Development;
print "Running in: " + env;  // "Running in: dev"
```

### Role-based Permissions

```zexus
enum UserRole {
  SuperAdmin = 3,
  Admin = 2,
  Moderator = 1,
  User = 0
}

action check_permission(user_role, required_role) {
  return user_role >= required_role;
}

if check_permission(UserRole.Moderator, UserRole.Admin) {
  grant_admin_panel_access();
}
```

## Auto-Increment Behavior

```zexus
enum Status {
  Pending,      // 0
  Active,       // 1
  Completed,    // 2
  Archived      // 3
}
```

## Mixed Values

```zexus
enum Priority {
  Low,          // 0
  Medium = 5,   // 5
  High,         // 6 (continues from previous)
  Critical = 100  // 100
}
```

## Accessing Members

```zexus
enum Animal {
  Dog = "canine",
  Cat = "feline",
  Bird = "avian"
}

print Animal.Dog;      // "canine"
print Animal.Cat;      // "feline"
print Animal.Bird;     // "avian"

let creatures = [Animal.Dog, Animal.Cat];
for creature in creatures {
  print creature;
}
```

## Pattern Matching with Enums

```zexus
enum RequestMethod {
  GET = 1,
  POST = 2,
  PUT = 3,
  DELETE = 4
}

action handle_request(method) {
  pattern method {
    case RequestMethod.GET => fetch_resource();
    case RequestMethod.POST => create_resource();
    case RequestMethod.PUT => update_resource();
    case RequestMethod.DELETE => delete_resource();
    default => return error("Invalid method");
  }
}
```

## Complex Example: State Machine

```zexus
enum OrderState {
  New = "new",
  Processing = "processing",
  Shipped = "shipped",
  Delivered = "delivered",
  Cancelled = "cancelled",
  Returned = "returned"
}

action update_order_state(order, new_state) {
  let old_state = order.state;
  
  pattern [old_state, new_state] {
    // Simplified: just show how enums are used
  }
  
  order.state = new_state;
  log("Order state: " + old_state + " -> " + new_state);
}
```

## Enum Metadata

```zexus
enum Color {
  Red,
  Green,
  Blue
}

// Enums become objects in environment
let color_map = Color;  // Access the enum object
print color_map;        // All members accessible
```

## Performance

- **Creation**: O(n) where n = members
- **Access**: O(1)
- **Storage**: Minimal (just named values)
- **Type**: Safe (compile-time checking possible)

## Best Practices

1. **Use for Fixed Sets**: Enums for values that never change
2. **Name Semantically**: EnumName for clarity
3. **Group Related Values**: Related enums together
4. **Document Values**: Comment on meaning
5. **Use Consistent Type**: All members same type (int or string)

```zexus
// GOOD: Clear, grouped, documented
enum HTTPMethod {
  GET = "GET",
  POST = "POST",
  PUT = "PUT",
  DELETE = "DELETE",
  PATCH = "PATCH"
}

// AVOID: Mixed types
enum BadEnum {
  One = 1,
  Two = "two",  // Inconsistent!
  Three = 3.0
}
```

## Combining with Other Features

```zexus
// With PATTERN: Safe case handling
enum Status { Active, Inactive, Pending }
pattern status {
  case Status.Active => activate();
  case Status.Inactive => deactivate();
  default => handle_pending();
}

// With RESTRICT: Protect enum access
restrict sensitive_enum = "read-only";
let value = SensitiveEnum.Value;  // OK
// SensitiveEnum.Value = something;  // Not allowed

// With WATCH: React to enum changes
watch current_status => {
  update_ui();
}
```

## Common Patterns

### State Enum with Handlers

```zexus
enum State { Start, Running, Paused, Stopped }

action transition(current, next) {
  pattern [current, next] {
    // Transitions...
  }
}
```

### Status Codes

```zexus
enum Code {
  Success = 0,
  Warning = 1,
  Error = 2,
  Fatal = 3
}
```

### Permissions/Flags

```zexus
enum Permission {
  Read = 1,
  Write = 2,
  Execute = 4,
  Admin = 8
}
```

## Limitations

- ✗ No computed members
- ✗ No enum methods (yet)
- ✗ No enum inheritance
- ✗ No bitflags operators (yet)

## Future Enhancements

- Bitflag operations (| &)
- Associated data per variant
- Methods on enums
- Struct variants
- Enum matching with guards

## See Also

- PATTERN: Match against enum values
- CONST: Define constant values
- ACTION: Extract enum handling logic
