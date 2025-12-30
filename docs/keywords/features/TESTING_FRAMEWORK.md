# Testing Framework - Phase 1

**Status**: Planned
**Phase**: 1 - Build WITH Zexus
**Priority**: High
**Estimated Effort**: 2-3 months

## Overview

Build a native Zexus testing framework with:
- BDD-style test organization
- Assertions
- Mocking and spies
- Coverage reporting
- Parallel execution
- Watch mode

## Quick Example

```zexus
use {describe, it, expect, before_each, after_each} from "zexus/test"

describe("User Service", action() {
    let user_service = null
    
    before_each(action() {
        user_service = UserService()
    })
    
    it("creates a new user", action() {
        let user = user_service.create({
            name: "Alice",
            email: "alice@example.com"
        })
        
        expect(user.name).to_equal("Alice")
        expect(user.email).to_equal("alice@example.com")
        expect(user.id).to_be_defined()
    })
    
    it("validates email format", action() {
        expect(action() {
            user_service.create({
                name: "Bob",
                email: "invalid"
            })
        }).to_throw("Invalid email")
    })
})
```

## Features

### Assertions
```zexus
expect(value).to_equal(expected)
expect(value).to_be_greater_than(5)
expect(value).to_be_less_than(10)
expect(value).to_contain("substring")
expect(value).to_be_defined()
expect(value).to_be_null()
expect(fn).to_throw("error message")
```

### Mocking
```zexus
use {mock, spy} from "zexus/test"

let mock_db = mock({
    query: action() { return [{id: 1}] }
})

let spy_fn = spy()
spy_fn("test")
expect(spy_fn).to_have_been_called()
expect(spy_fn).to_have_been_called_with("test")
```

### Coverage
```bash
zx test --coverage
# Generates coverage report
```

### Watch Mode
```bash
zx test --watch
# Reruns tests on file changes
```

## Related Documentation

- [Ecosystem Strategy](../../ECOSYSTEM_STRATEGY.md)
- [@zexus/test Package (Phase 3)](../../packages/ZEXUS_TEST_PACKAGE.md)

---

**Status**: Planning
**Last Updated**: 2025-12-29
