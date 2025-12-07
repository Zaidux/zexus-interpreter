# Zexus Modifiers

This document describes the new modifier system for Zexus: a compact way to extend declarations without adding new keywords.

Supported modifiers (initial set):
- `public` — mark declaration for export/public visibility
- `private` — mark declaration as private to current module (not exported)
- `sealed` — prevent runtime mutation (for objects, used by evaluator)
- `async` — mark action as asynchronous (runtime may execute concurrently)
- `native` — mark that declaration uses native integration (requires capability)
- `inline` — mark that function should be inlined by optimizer
- `secure` — mark that declaration requires security/capability checks
- `pure` — mark that function is pure (no side-effects)

Usage examples:

- Mark an action as secure and asynchronous:

```
secure async action fetch_user() {
  // ...
}
```

- Expose an action publicly:

```
public action start() { ... }
```

Implementation notes:

- Modifiers are parsed as a leading sequence of identifiers before a declaration. The parser attaches a `modifiers` attribute on the AST node:

```py
node.modifiers = ['secure', 'async']
```

- Evaluator honors modifiers for actions by setting flags on the runtime `Action` object (e.g., `is_inlined`, `is_async`, `is_secure`).
- `public` will automatically export the symbol in the environment when encountered.
- `native` and `secure` are subject to capability checks when the action is executed; a plugin or capability enforcement layer can consult these flags.

Notes for plugin authors:
- Plugins can check `action_obj.is_secure` or `action_obj.required_capabilities` to enforce runtime checks.
- The modifiers system is intentionally lightweight and extensible; new modifiers can be supported by updating the lexer keyword map and evaluation hooks.

Security:
- `secure` is a tagging mechanism. Full enforcement requires the capability/security subsystem (see `ENHANCEMENT_PACKAGE/STRATEGIC_UPGRADES.md`) to be implemented and enabled.

