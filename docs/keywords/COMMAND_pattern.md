# PATTERN Statement

**Purpose**: Pattern matching expressions for elegant conditional logic.

**Why Use PATTERN**:
- Readable alternative to if-else chains
- Matches values against multiple patterns
- Clean syntax for type/value discrimination
- Reduces code nesting
- Self-documenting code

## Syntax

```
pattern <expression> {
  case <pattern> => <action>;
  case <pattern> => <action>;
  default => <action>;
}
```

## Examples

### Simple Value Matching

```zexus
let status_code = 404;
pattern status_code {
  case 200 => print "OK";
  case 404 => print "Not Found";
  case 500 => print "Server Error";
  default => print "Unknown";
}
```

### String Patterns

```zexus
let color = "red";
pattern color {
  case "red" => set_rgb(255, 0, 0);
  case "green" => set_rgb(0, 255, 0);
  case "blue" => set_rgb(0, 0, 255);
  default => set_rgb(128, 128, 128);
}
```

### Enum Patterns

```zexus
enum UserRole { Admin, Moderator, User, Guest }
let role = UserRole.Admin;

pattern role {
  case UserRole.Admin => grant_all_permissions();
  case UserRole.Moderator => grant_moderate_permissions();
  case UserRole.User => grant_user_permissions();
  default => grant_guest_permissions();
}
```

### Complex Expressions

```zexus
let status = 200;
pattern status {
  case 200 => print "Success";
  case 201 => print "Created";
  case 400 => print "Bad Request";
  case 401 => print "Unauthorized";
  case 403 => print "Forbidden";
  case 404 => print "Not Found";
  case 500 => print "Server Error";
  default => print "Unknown";
}
```

## Match Logic

- **First Match**: Returns on first matching case
- **Default**: Optional, executes if no matches
- **LIFO**: No fall-through (not like C switch)
- **Type Safe**: Pattern types should match expression

## Advanced Patterns

### Range Patterns (Simulated)

```zexus
let score = 85;
pattern score {
  case 90 => print "A";
  case 80 => print "B";
  case 70 => print "C";
  case 60 => print "D";
  default => print "F";
}
```

### Nested Patterns

```zexus
let response = api_call();
pattern response.status {
  case 200 => {
    let data = parse_json(response.body);
    print "Data: " + data;
  }
  case 404 => {
    log_error("Resource not found");
    retry();
  }
  default => {
    log_error("API error: " + response.status);
  }
}
```

## Performance

- **Time**: O(n) where n = number of cases
- **Space**: O(1) additional space
- **Optimization**: Compiler may use jump tables for integer patterns

## Best Practices

1. **Cover All Cases**: Use default for uncovered cases
2. **Order by Frequency**: Put common cases first
3. **Keep Actions Simple**: Consider extracting to functions
4. **Document Intent**: Comment why certain patterns exist

```zexus
pattern event_type {
  case "click" => handle_click();        // Most common (90%)
  case "hover" => handle_hover();        // Moderately common (5%)
  case "focus" => handle_focus();        // Rare (4%)
  case "blur" => handle_blur();          // Rare (1%)
  default => log("Unknown event");
}
```

## Comparison with If-Else

**Pattern Matching (Preferred)**:
```zexus
pattern value {
  case 1 => action_one();
  case 2 => action_two();
  default => action_default();
}
```

**If-Else (Legacy)**:
```zexus
if value == 1 {
  action_one();
} elif value == 2 {
  action_two();
} else {
  action_default();
}
```

## Combining with Other Features

```zexus
// With SANDBOX: Safe pattern execution
sandbox("pattern-safe") {
  pattern value {
    case "safe" => process_safe_value();
    default => reject();
  }
}

// With TRAIL: Log pattern matches
trail *, "pattern_match";
pattern status {
  case 200 => print "Success";  // Traced
  default => print "Error";
}

// With RESTRICT: Control pattern execution
restrict sensitive_pattern = "read-only";
```

## Advanced Example: State Machine

```zexus
action process_order(order) {
  let state = order.state;
  
  pattern state {
    case "pending" => {
      validate_order(order);
      order.state = "validated";
    }
    case "validated" => {
      charge_payment(order);
      order.state = "charged";
    }
    case "charged" => {
      ship_items(order);
      order.state = "shipped";
    }
    case "shipped" => {
      print "Order already shipped";
    }
    default => {
      log_error("Unknown order state: " + state);
    }
  }
  
  return order;
}
```

## Limitations

- ✗ No wildcards or ranges (yet)
- ✗ No structural/destructuring patterns (yet)
- ✗ Single expression matching only
- ✗ No guard clauses (yet)

## Future Enhancements

- Wildcard patterns (_)
- Range patterns (1..10)
- Destructuring patterns
- Guard clauses (case x if condition =>)
- Tuple/struct patterns

## See Also

- IF/ELIF/ELSE: Traditional conditionals
- ACTION: Extract pattern actions to functions
