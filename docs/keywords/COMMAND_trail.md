**TRAIL**

- **Overview**: `trail` configures real-time event tracking for the interpreter. It can be used to enable live traces of specific event categories (audit, print, debug, * for all) with optional filters. Trail entries are intended for debugging, observability, and lightweight auditing.

- **Syntax**:

```
trail <event_type> [ , "<filter>" ] ;
```

- **Parameters**:
- **`event_type`**: Identifier or `*` for all events. Common values: `audit`, `print`, `debug`, `exec`.
- **`filter`** (optional): A string used to filter events (for example `"user:123"` to only include events related to a user id).

- **Examples**:

1) Turn on auditing trail:
```
trail audit;
```

2) Trace all events with a filter:
```
trail *, "user:42";
```

3) Enable print/debug tracing during development:
```
trail print;
trail debug, "module:payment";
```

- **Behavior & Return Value**:
- The `trail` statement registers a runtime `TrailConfig` with the event dispatcher and returns a `Map` summarizing the active trail (id, event_type, filter, started_at).
- Trails can be updated or removed by analogous statements (future enhancement: `untrail <id>`).

- **Integration Notes**:
- The runtime event dispatcher listens for trail configurations and emits matching events to configured sinks (console, log file, or in-memory buffer). The `TrailStatement` evaluator registers the configuration with the dispatcher.
- To minimize performance impact, tracing can be sampled or throttled.

- **Security & Privacy**:
- Trails may contain sensitive data; avoid enabling broad trails in production without proper access controls.
- Use filters to restrict the volume and sensitivity of traced events.

- **Notes for implementers**:
- Provide APIs to list and remove active trails, and to export or persist traced events to the `AuditLog` for long-term storage.
- Integrate with the `AuditLog` where events need permanent retention for compliance.