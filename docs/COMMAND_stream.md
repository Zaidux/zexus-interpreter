# STREAM Statement

**Purpose**: Event streaming and asynchronous event handling for reactive systems.

**Why Use STREAM**:
- Handle asynchronous events elegantly
- Decouple event producers from consumers
- Reactive programming patterns
- Real-time data processing
- Clean event handler registration

## Syntax

```
stream <stream_name> as <event_var> => {
  <handler_code>
}
```

## Examples

### Mouse Events

```zexus
stream clicks as event => {
  print "Clicked at: " + event.x + ", " + event.y;
}

stream hovers as event => {
  highlight_element(event.target);
}

stream drags as event => {
  update_position(event.target, event.x, event.y);
}
```

### API Responses

```zexus
stream api_responses as response => {
  if response.status == 200 {
    process_data(response.body);
  } else {
    handle_error(response.status);
  }
}

stream api_errors as error => {
  log_error("API error: " + error.message);
  retry_request();
}
```

### Message Queue

```zexus
stream messages as msg => {
  let processed = process_message(msg);
  send_acknowledgment(processed);
}

stream errors as error => {
  add_to_dead_letter_queue(error.message);
}
```

### WebSocket Events

```zexus
stream ws_connected as event => {
  print "Connected to server";
  send_identification();
}

stream ws_messages as msg => {
  handle_incoming_message(msg.data);
}

stream ws_disconnected as event => {
  print "Disconnected, attempting reconnect...";
  reconnect();
}
```

### Sensor Data

```zexus
stream temperature_readings as reading => {
  if reading.value > 100 {
    trigger_alarm("High temperature: " + reading.value);
  }
}

stream motion_detected as event => {
  record_timestamp();
  capture_frame();
}
```

## Multiple Handlers for Same Stream

```zexus
// Logging handler
stream user_actions as action => {
  log("User action: " + action.type);
}

// Analytics handler
stream user_actions as action => {
  track_analytics(action);
}

// UI update handler
stream user_actions as action => {
  update_ui(action);
}
```

## Stream Properties

**Async Nature**:
- Events arrive asynchronously
- Handler executes when event emitted
- Non-blocking (handler doesn't block producer)

**Event Object**:
- Contains event data
- Type/structure depends on stream
- Can extract properties with dot notation

**Handler Execution**:
- Runs in context of current environment
- Has access to all variables in scope
- Can modify global state (not recommended)

## Advanced Patterns

### Error Handling in Stream

```zexus
stream data_stream as data => {
  try {
    process_data(data);
  } catch error {
    log_error("Stream processing failed: " + error);
    fallback_handler(data);
  }
}
```

### Conditional Processing

```zexus
stream events as event => {
  if should_process(event) {
    process(event);
  } else if should_skip(event) {
    skip(event);
  } else {
    log_warning("Unexpected event: " + event.type);
  }
}
```

### Stream Filtering (Manual)

```zexus
stream all_events as event => {
  // Manual filtering
  if event.priority == "high" {
    handle_high_priority(event);
  }
}
```

### Event Batching

```zexus
let batch = [];
let batch_size = 100;

stream items as item => {
  batch.push(item);
  if len(batch) >= batch_size {
    process_batch(batch);
    batch = [];
  }
}
```

### Stream Chaining

```zexus
stream raw_events as event => {
  let transformed = transform_event(event);
  emit_to_next_stream(transformed);
}
```

## Performance Considerations

- **Throughput**: Depends on handler complexity
- **Latency**: Minimize handler execution time
- **Buffering**: Streams may buffer events if handler is slow
- **Memory**: Keep event backlog manageable

## Best Practices

1. **Keep Handlers Fast**: Process quickly, defer heavy work
2. **Handle Errors**: Always include error handling
3. **Avoid Side Effects**: Minimize global state changes
4. **Log Important Events**: Aid debugging and monitoring
5. **Document Stream Contracts**: What events contain

```zexus
// Good: Fast, focused handler
stream payments as payment => {
  if validate_payment(payment) {
    queue_for_processing(payment);  // Async queue
  }
}

// Avoid: Slow synchronous work
stream payments as payment => {
  connect_to_bank();       // Slow!
  charge_customer();       // Slow!
  update_accounting();     // Slow!
}
```

## Combining with Other Features

```zexus
// With DEFER: Cleanup stream handlers
stream events as event => {
  let resource = acquire_resource();
  defer release_resource(resource);
  
  process_with_resource(event, resource);
}

// With WATCH: React to stream state
watch stream_paused => {
  if stream_paused {
    log("Stream paused - buffering events");
  } else {
    log("Stream resumed");
  }
}

// With TRAIL: Log stream events
trail *, "stream_events";
stream critical_events as event => {
  process_critical(event);  // Traced
}

// With SANDBOX: Isolate stream handlers
stream untrusted_stream as event => {
  sandbox("stream-handler") {
    process_untrusted_data(event);
  }
}
```

## Real-World Example: Order Processing

```zexus
enum OrderStatus {
  Pending = "pending",
  Processing = "processing",
  Shipped = "shipped",
  Delivered = "delivered"
}

stream orders as order => {
  validate_order(order);
  order.status = OrderStatus.Processing;
  
  emit_to_warehouse(order);
  emit_to_accounting(order);
  emit_to_notification(order);
}

stream shipments as shipment => {
  update_order_status(shipment.order_id, OrderStatus.Shipped);
  notify_customer(shipment.order_id);
}

stream deliveries as delivery => {
  update_order_status(delivery.order_id, OrderStatus.Delivered);
  send_delivery_confirmation(delivery.order_id);
  trigger_feedback_request(delivery.order_id);
}
```

## Limitations

- ✗ No stream filtering/transformation (yet)
- ✗ No stream backpressure handling
- ✗ No stream composition (yet)
- ✗ Synchronous handler execution (async TBD)

## Future Enhancements

- Stream operators (.filter, .map, .reduce)
- Backpressure and flow control
- Stream composition and merging
- Async/await for handlers
- Stream replay
- Event sourcing support

## See Also

- WATCH: Monitor variable changes
- DEFER: Cleanup in stream handlers
- SANDBOX: Isolate untrusted handlers
- TRY/CATCH: Error handling in streams
