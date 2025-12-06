**RESTRICT**

- **Overview**: The `restrict` command applies field-level access control to object properties. It records a restriction policy for a given object property, which the runtime and security layer may consult when enforcing read/write/modify operations.

- **Syntax**:

```
restrict <object>.<field> = "<restriction_type>";
```

- **Parameters**:
- **`object.field`**: A property access expression identifying the target field (e.g., `user.email`).
- **`restriction_type`**: A string literal describing the restriction (examples: `"read-only"`, `"admin-only"`, `"redact"`).

- **Examples**:

1) Make `user.password` read-only:
```
restrict user.password = "read-only";
```

2) Require admin for writes and mark as sensitive:
```
restrict invoice.total = "admin-only";
restrict invoice.notes = "sensitive";
```

3) Use with other statements (policy recorded, execution continues):
```
restrict profile.ssn = "redact";
print("Restriction applied")
```

- **Behavior & Return Value**:
- `restrict` records a policy in the runtime's security registry and returns a `Map` object summarizing the policy (target, restriction_type, timestamp, id).
- It does not itself prevent operations; enforcement requires the runtime (or security layer) to consult recorded policies during reads/writes. Some built-in operations in the runtime will automatically consult the restriction registry when present.

- **Integration Notes**:
- The security subsystem (`security.py`) exposes an internal registry (`RestrictionRegistry`) which stores policies. The interpreter calls into that registry when `restrict` is evaluated.
- Parser and lexer register the `restrict` keyword and create a `RestrictStatement` AST node; this doc assumes those integrations are present.

- **Security Considerations**:
- Restriction policies are advisory until enforcement hooks exist for each operation type; ensure all sensitive APIs consult the registry.
- Policies should be exported/inspected by administrators; the `AuditLog` can be used to track when policies are added or changed.

- **Notes for implementers**:
- When adding enforcement, check property access paths for aliasing and dynamic proxies.
- Consider scoping policies by environment or module if needed (e.g., `sandbox` enforcement).