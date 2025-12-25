**SANDBOX**

- **Overview**: The `sandbox` command creates an isolated execution environment and executes a block of Zexus code inside it. The sandboxed environment has its own variable scope and may optionally restrict access to host resources (I/O, network, global state) depending on runtime enforcement.

- **Syntax**:

```
sandbox {
  // statements
}
```

- **Parameters**:
- **Block body**: Any valid sequence of Zexus statements. The body executes in an isolated `Environment` instance that inherits only selected bindings from the parent (implementation-defined).

- **Examples**:

1) Run math safely in a sandbox:
```
sandbox {
  const x = 42;
  print(x * 2);
}
```

2) Isolate side-effecting code (no host filesystem access expected):
```
sandbox {
  // This code cannot modify process-global state
  write_file("/tmp/test.txt", "hello")
}
```

3) Nested sandboxes and return values:
```
let result = sandbox {
  const secret = computeSecret();
  return secret + 1;
};
print(result);
```

- **Behavior & Return Value**:
- A `sandbox` statement evaluates its block in an isolated `Environment` and returns the block's final value (if any) to the outer scope.
- The exact isolation policy is driven by the runtime/security subsystem. By default the sandbox isolates variables and standard I/O; optional flags or deeper runtime hooks can be added later.

- **Integration Notes**:
- The evaluator for `SandboxStatement` creates a new `Environment` instance and executes the block, capturing its result and any thrown exceptions.
- The security layer may provide a `SandboxPolicy` to limit available APIs inside the sandbox (for example: disallowing `open`, `network_*`, or `external_bridge` access).

- **Security Considerations**:
- Sandboxing reduces risk from untrusted code but is not a substitute for OS-level isolation when dealing with arbitrary third-party code.
- When sandboxes are allowed to return objects that reference host resources, ensure those objects are wrapped or proxied to prevent escape.

- **Notes for implementers**:
- Provide options to configure the sandbox (resource quotas, allowed APIs, timeouts).
- Consider deterministic default policies and explicit opt-in for expanded capabilities.