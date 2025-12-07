# WATCH Statement

**Purpose**: Reactive state management - execute code when watched variables change.

**Why Use WATCH**:
- React to state changes automatically
- Simplify reactive programming
- Update UI/derived state reactively
- Avoid manual change polling
- Self-documenting reactive intent

## Syntax

```
watch <expression> => {
  <reaction_code>
}

watch <expression> => <single_statement>;
```

## Examples

### Simple Variable Watch

```zexus
let count = 0;
watch count => {
  print "Count changed to: " + count;
}

count = 5;  // Triggers: "Count changed to: 5"
count = 10; // Triggers: "Count changed to: 10"
```

### UI Updates on State Change

```zexus
let user_name = "Guest";
watch user_name => {
  update_username_display(user_name);
}

user_name = "Alice";  // UI automatically updated
```

### Computed Values

```zexus
let width = 100;
let height = 200;

watch width => {
  area = width * height;
  perimeter = 2 * (width + height);
  update_dimensions_display();
}

watch height => {
  area = width * height;
  perimeter = 2 * (width + height);
  update_dimensions_display();
}
```

### Form Validation

```zexus
let email = "";
watch email => {
  if is_valid_email(email) {
    enable_submit_button();
    clear_email_error();
  } else {
    disable_submit_button();
    show_email_error("Invalid email format");
  }
}
```

### Data Synchronization

```zexus
let local_data = get_cached_data();
watch local_data => {
  sync_to_server(local_data);
  update_sync_timestamp();
}
```

## Watch with Expressions

### Nested Property Watch

```zexus
let user = { name: "Alice", age: 30 };

watch user.name => {
  print "User name changed to: " + user.name;
}

watch user.age => {
  if user.age >= 18 {
    grant_adult_features();
  } else {
    restrict_adult_features();
  }
}
```

### Conditional Watch

```zexus
let status = "inactive";
watch status => {
  if status == "active" {
    start_service();
  } elif status == "paused" {
    pause_service();
  } else {
    stop_service();
  }
}
```

### Complex Reaction

```zexus
let search_query = "";
watch search_query => {
  if len(search_query) > 2 {
    results = search_database(search_query);
    update_results_display(results);
    log_search(search_query);
  } else {
    clear_results_display();
  }
}
```

## Block Syntax

```zexus
let data = [];
watch data => {
  save_to_cache(data);
  update_ui(data);
  log("Data changed");
}
```

## Single Statement Syntax

```zexus
let count = 0;
watch count => increment_counter();

let enabled = true;
watch enabled => toggle_feature();
```

## Advanced Patterns

### Debounced Watch (Manual)

```zexus
let search_input = "";
let debounce_timer = null;

watch search_input => {
  if debounce_timer != null {
    cancel_timer(debounce_timer);
  }
  
  debounce_timer = schedule_after(500, () => {
    perform_search(search_input);
  });
}
```

### Watch with History

```zexus
let value = 0;
let history = [0];

watch value => {
  history.push(value);
  if len(history) > 100 {
    history.remove(0);  // Keep last 100 values
  }
}
```

### Multi-Variable Reaction

```zexus
let user_id = null;
let user_data = null;

watch user_id => {
  user_data = fetch_user(user_id);
}

watch user_data => {
  update_profile_display(user_data);
  load_user_preferences(user_data);
}
```

### State Machine with Watch

```zexus
enum State { Idle, Loading, Ready, Error }
let state = State.Idle;

watch state => {
  pattern state {
    case State.Idle => {
      show_idle_ui();
      disable_controls();
    }
    case State.Loading => {
      show_spinner();
      disable_controls();
    }
    case State.Ready => {
      hide_spinner();
      enable_controls();
      show_data();
    }
    case State.Error => {
      show_error_message();
      enable_retry_button();
    }
  }
}
```

## Performance Considerations

- **Frequency**: Called on every change to watched expression
- **Cost**: Should be O(1) to O(log n)
- **Overhead**: Minimal change detection
- **Avoiding Loops**: Don't modify watched variable in reaction

⚠️ **Warning - Infinite Loop**:
```zexus
let x = 0;
watch x => {
  x = x + 1;  // ❌ Changes x, triggers watch again!
}
```

✅ **Better Approach**:
```zexus
let x = 0;
let x_incremented = 0;

watch x => {
  x_incremented = x + 1;  // Don't modify watched var
}
```

## Best Practices

1. **Keep Reactions Fast**: Minimize computation
2. **Avoid Side Effects**: Prefer pure reactions
3. **Prevent Infinite Loops**: Don't modify watched variable
4. **Document Intent**: Comment what and why you're watching
5. **Use Appropriate Granularity**: Watch specific properties, not entire objects

```zexus
// GOOD: Watching specific property
watch user.profile_updated_at => {
  reload_user_profile();
}

// AVOID: Watching entire object (too broad)
watch user => {
  reload_everything();
}
```

## Combining with Other Features

```zexus
// With DEFER: Cleanup watch reactions
watch resource => {
  let temp = acquire_resource();
  defer release_resource(temp);
  
  process(resource, temp);
}

// With SANDBOX: Isolate watch reactions
watch untrusted_data => {
  sandbox("watch-reaction") {
    process_data(untrusted_data);
  }
}

// With TRAIL: Log watch triggers
trail *, "watch";
watch critical_value => {
  handle_critical(critical_value);  // Traced
}

// With RESTRICT: Control watched access
restrict monitored_value = "read-only";
watch monitored_value => {
  log_access(monitored_value);
}
```

## Real-World Example: Chat Application

```zexus
let messages = [];
let unread_count = 0;
let last_read_timestamp = now();

watch messages => {
  unread_count = count_unread(messages, last_read_timestamp);
  update_unread_badge(unread_count);
  scroll_to_latest();
  play_notification_sound();
}

watch unread_count => {
  if unread_count > 0 {
    show_notification("You have " + unread_count + " new messages");
  }
}
```

## Semantics

- **Execution**: Synchronous call when expression changes
- **Timing**: Immediately after assignment
- **Order**: Watches execute in registration order
- **Nesting**: Nested watches allowed but use caution

## Limitations

- ✗ No computed properties (yet)
- ✗ No change event details (what changed)
- ✗ No async reactions (yet)
- ✗ No conditional watches

## Future Enhancements

- Deep/shallow watch options
- Change event details (old value, new value)
- Async reactions with awaiting
- Computed properties with caching
- Watch with conditions
- Batch watch updates (micro-task queue)

## See Also

- STREAM: Handle events from external sources
- PATTERN: Conditional reactions to state changes
- DEFER: Cleanup in watch reactions
- ACTION: Extract watch logic to functions
