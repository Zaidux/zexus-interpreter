RESTRICT / SANDBOX / TRAIL - Implementation & Enforcement Patterns

Overview
--------
This document describes the runtime registries and enforcement patterns added to support the `restrict`, `sandbox`, and `trail` commands. It explains where registries live, how the evaluator registers policies and runs, and gives concrete code patterns to implement enforcement across the interpreter.

Files touched
------------
- `src/zexus/security.py` — added registries: `register_restriction`, `get_restriction`, `register_trail`, `register_sandbox_run`.
- `src/zexus/evaluator/statements.py` — evaluator now registers restrictions/trails and records sandbox runs when those statements execute.
- `docs/COMMAND_restrict.md`, `docs/COMMAND_sandbox.md`, `docs/COMMAND_trail.md` — user-facing docs added.

Runtime Registries
-------------------
SecurityContext (accessible via `get_security_context()`) now exposes lightweight registries:

- `register_restriction(target, field, restriction_type, author=None)` → returns entry `{id, target, field, restriction, author, timestamp}`.
- `get_restriction(target)` → returns latest restriction entry for `target` or `None`.
- `list_restrictions()` / `remove_restriction(id)`.

- `register_trail(event_type, filter_key=None)` → returns entry `{id, type, filter, enabled, timestamp}`.
- `list_trails()` / `remove_trail(id)`.

- `register_sandbox_run(parent_context=None, policy=None, result_summary=None)` → returns run entry with id and metadata. `list_sandbox_runs()` available.

Evaluator behavior
------------------
- When the interpreter evaluates `restrict obj.field = "rule";`, the `eval_restrict_statement` will:
  1. Validate the target exists
  2. Call `get_security_context().register_restriction(...)`
  3. Return a `Map` to the program including the registry `id` (if available)

- When evaluating `trail ...;`, the evaluator registers the trail via `register_trail(...)` so the runtime can wire observers.

- `sandbox { ... }` runs are recorded with `register_sandbox_run(...)` (minimal summary saved). Policies are not yet enforced automatically — the registry exists so enforcement hooks may consult it.

Enforcement patterns & examples
-------------------------------
Below are recommended patterns to enforce restrictions and to leverage trails and sandboxes across the interpreter.

1) Enforcing field-level restrictions during property access

Where to implement: central property access helpers — e.g. in `src/zexus/object.py` or `src/zexus/evaluator/statements.py` where `PropertyAccessExpression` is resolved.

Pattern (Python-like pseudocode):

```
from src.zexus.security import get_security_context

def resolve_property_access(env, obj_name, field_name, ctx):
    # Check restriction registry first
    sec = get_security_context()
    restriction = sec.get_restriction(f"{obj_name}.{field_name}")
    if restriction:
        rule = restriction['restriction']
        # Example rule: 'read-only' or 'admin-only' or 'redact'
        if rule == 'read-only' and ctx.request_type == 'write':
            raise EvaluationError('Write prohibited by restriction')
        if rule == 'admin-only' and not ctx.user_is_admin:
            raise EvaluationError('Admin privileges required')
        if rule == 'redact' and ctx.request_type == 'read':
            return '***REDACTED***'

    # Fallback: perform normal access
    return obj.data.get(field_name)
```

Notes:
- `ctx` above represents evaluation context metadata (authenticated user, request type, etc.). The registry is advisory; callers must consult it.
- For performance, you can cache `get_restriction` lookups for a given execution until policies change.

Implementation notes (what I implemented):
- `SecurityContext.register_restriction` and `get_restriction` were added to `src/zexus/security.py`.
- Property reads now consult `SecurityContext.get_restriction` in `src/zexus/evaluator/core.py`'s `PropertyAccessExpression` handler. Current behaviors implemented:
    - `redact` returns `"***REDACTED***"` for matching fields.
    - `admin-only` raises an `EvaluationError` unless `env.get('__is_admin__')` is truthy.
    - Other rules are advisory (no-op) until extended.
- Property writes (assignments) now consult `get_restriction` in `src/zexus/evaluator/statements.py`'s `eval_assignment_expression`. Current behaviors implemented:
    - `read-only` forbids writes and returns an `EvaluationError`.
    - `admin-only` requires `env.get('__is_admin__')` to be truthy to proceed.
    - Sealed properties are also checked and prevented from being modified.

2) Wiring trails into the event dispatcher

Where to implement: the event dispatcher or logging layer that emits runtime events (e.g. `print`, `audit`, or custom instrumentation).

Pattern:

```
sec = get_security_context()
for trail in sec.list_trails():
    if trail_matches_event(trail, event):
        emit_to_sink(trail, event)
```

Trail entries include `filter` which implementers should use to reduce sensitive data volume.

3) Enforcing sandbox policies

Where to implement: runtime helpers that expose builtins or host APIs to evaluated code. Before returning a builtin or executing an external API call, consult the sandbox policy for permissions.

Pattern:

```
policy = sec.get_sandbox_policy(name)  # optional helper implementers can add
if in_sandbox and not policy.allows('filesystem'):
    raise EvaluationError('Filesystem access not permitted in this sandbox')
```

Work left to fully enforce
--------------------------
- Hook property access sites to consult `SecurityContext.get_restriction` (see Pattern #1).
- Implement an event dispatcher that consumes `SecurityContext.list_trails()` and emits matching events to sinks.
- Expand sandbox policies with allowed API sets and resource quotas; have builtins consult those policies before executing.

What I implemented now:
- A lightweight event dispatcher `SecurityContext.emit_event(event_type, payload)` was added to `src/zexus/security.py` which:
    - Matches active trails (by type or `*`) and simple substring filter.
    - Appends trail events to the `AuditLog` in memory and prints matching events to stdout.
    - The evaluator now emits events for `print`, `audit`, and `debug` statements so trails get live traces.

Next enforcement steps (optional):
- Wire builtins and external APIs to consult sandbox policies before performing I/O.
- Add richer filter expressions and event sinks (file, network, push to observability backends).

Added in this change:
- Sandbox policy registry (`register_sandbox_policy`, `get_sandbox_policy`) in `src/zexus/security.py`.
    - A conservative default policy named `default` is registered automatically by the evaluator when running a `sandbox {}` block; it disallows file I/O builtins by default.
- Builtin enforcement in `src/zexus/evaluator/functions.py`: when a call happens inside a sandbox (`env.__in_sandbox__`), the evaluator consults the sandbox policy and blocks disallowed builtins with an `EvaluationError`.
- Trail sinks and persistence:
    - `register_trail_sink(type='file', path=...)`, `register_trail_sink(type='stdout')`, and `register_trail_sink(type='callback', callback=callable)` added to `SecurityContext`.
    - `emit_event()` now writes matching trail events to configured sinks and persists them to file (`AUDIT_DIR/trails.jsonl` by default) in JSONL format.

How to configure (examples):

```py
from src.zexus.security import get_security_context
ctx = get_security_context()
# Add a file sink
ctx.register_trail_sink('file', path='chain_data/trails.jsonl')
# Add stdout sink
ctx.register_trail_sink('stdout')
# Add custom callback sink
def push_to_remote(entry):
        # send entry to observability backend
        pass
ctx.register_trail_sink('callback', callback=push_to_remote)

# Register a sandbox policy that allows only safe builtins
ctx.register_sandbox_policy('read-only-sandbox', allowed_builtins=['now','timestamp','string','len'])
```

Behavioral notes:
- Trail sinks are best-effort: failures to write to sinks are caught and ignored to avoid interrupting program execution.
- Sandbox enforcement is applied only to builtin functions; user-defined functions and methods still run normally in the sandboxed environment but can be restricted by limiting available builtins and environment bindings.

Richer trail selectors & durable storage
---------------------------------------
- Trail filters now support three forms:
    - Substring match (default): `"module:payment"` matches if that substring appears in event payload.
    - Key:value match: `"user:42"` or `"module:payment"` will try to parse event payload JSON and match key/value.
    - Regex match: prefix the filter with `re:` to provide a regular expression, e.g. `re:^payment_.*$`.

- Durable sinks: in addition to stdout and JSONL file sink, you can now register a `sqlite` sink which persists trail events to a local SQLite DB for long-term storage and querying.

Inline sandbox policy selection
--------------------------------
You can now choose a sandbox policy inline when declaring a sandbox block:

```
sandbox("read-only-sandbox") {
    // code runs with the 'read-only-sandbox' policy applied
}

# or using key syntax
sandbox(policy = "read-only-sandbox") {
    // same
}
```

The parser recognizes both forms and the evaluator will set the sandbox environment's `__sandbox_policy__` accordingly.

Examples:

```py
ctx.register_trail_sink('stdout')
ctx.register_trail_sink('file', path='chain_data/trails.jsonl')
ctx.register_trail_sink('sqlite', db_path='chain_data/trails.db')
```

These sinks are best-effort; failures to write are ignored to avoid interrupting program execution.


Testing recommendations
-----------------------
- Add unit tests for `security.register_restriction` and `get_restriction`.
- Integration tests that run `restrict` then attempt a write to ensure enforcement logic blocks writes (once access-check hooks are added).
- Tests for trail registration and that the dispatcher forwards events to sinks when a trail is active.

Next steps
----------
I can:
- Implement enforcement hooks in `src/zexus/object.py` or evaluator property access points so `restrict` becomes effective, and wire a simple event dispatcher to honor `trail` registrations.
- Add unit + integration tests for the registries and the enforcement patterns above.

If you want that, tell me which enforcement you'd like first (property access or event dispatching), and I'll implement it and add tests.
