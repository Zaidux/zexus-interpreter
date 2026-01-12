# Strategy: `find` and `load` Keywords

## Goals
- Provide resilient resource discovery (`find`) when standard `use` resolution fails.
- Enable declarative configuration ingestion (`load`) for environment and external sources.
- Maintain deterministic behavior, clear diagnostics, and opt-in security controls.

## `find` Keyword

### Parsing and Syntax
- Support as expression form: `find "module.zx"`, `let cfg = find "configs/app.zx"`.
- Allow optional filters: `find "app.zx" in project`, `find "logger" near here` (stretch goal).
- Grammar: treat `find` as unary operator accepting string literal or identifier expression.

### Resolution Order
1. **Local Directory**: Resolve relative to current module path.
2. **Known Imports Cache**: Use compiler index of `use` statements and `zexus.json` entries.
3. **Project Scan**: Fallback to glob over allowed source roots (configurable).

### Ambiguity Handling
- If multiple matches, raise compile-time error listing candidates and request explicit path.
- Cache disambiguated results to avoid repeated scans.

### Integration Points
- Parser: extend expression grammar and AST node set (e.g., `FindExpression`).
- Compiler: treat as deferred module reference; use existing import pipeline with resolved absolute path.
- Runtime: no-op after compile; ensure module loader reuses resolved path to prevent duplicate loads.
- Tooling: update language server/completion to surface available targets.

### Safety and Performance
- Limit scan scope via `zexus.json` settings (e.g., `findPaths`).
- Memoize results per compilation unit; invalidate on file change events.
- Emit warning if fallback scan exceeds threshold (profiling hook).

## `load` Keyword

### Parsing and Syntax
- Expression form: `load env.APP_KEY`, `const creds = load "aws" from "secrets"`.
- Support dotted identifiers for env variables and nested config keys.
- Allow optional source clause: `load key from "./config.json"`.

### Resolution Semantics
1. **Environment Variables**: Default provider reads `os.environ`.
2. **File Providers**: Chainable loaders registered for `.env`, JSON, YAML (extensible).
3. **Library Providers**: Optional hooks for secrets managers; gated behind capability flags.

### Error Handling
- Missing value: raise descriptive runtime error with source hint and remediation suggestions.
- Invalid provider: compile-time error if source clause references unknown loader.
- Type coercion: defer to provider; expose helper utilities for common conversions.

### Integration Points
- Parser: add `LoadExpression` supporting optional `from` clause and dotted keys.
- Runtime: implement `LoadManager` that dispatches to provider registry.
- Configuration: extend `zexus.json` with `loadProviders`, `envFile`, `allowExternalSecrets`.
- Tests: create fixtures covering env fallback, file overrides, failure paths.

### Security Considerations
- Align with capability system; require explicit grants to access filesystem or external secrets.
- Sandbox default execution to `.env` file at project root unless configured.
- Audit logging for load accesses when security tracing is enabled.

## Implementation Phasing
1. **Parser & AST**: Introduce nodes for `find`/`load`; update syntax tests.
2. **Compiler & Loader**: Wire `find` resolution and module caching.
3. **Runtime Providers**: Build `LoadManager`, env loader, dotenv integration.
4. **Capabilities & Config**: Enforce security gates and configuration options.
5. **Tooling Updates**: Extend diagnostics, language server, documentation.

## Testing Plan
- Parser snapshots for new keywords.
- Unit tests for resolver caching, ambiguity errors, provider fallbacks.
- Integration tests executing scripts using `find` and `load` in actions, `let`, and nested expressions.
- Performance regression checks on large project scans and repeated `load` invocations.
