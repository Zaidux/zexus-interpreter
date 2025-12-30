# Package Development Guide

**For Zexus Official Packages (@zexus/*)**

## Overview

This guide covers development of official Zexus packages that follow the batteries-included philosophy (Phase 3).

## Package Structure

All official packages follow this structure:

```
@zexus/package-name/
├── zexus.json          # Package manifest
├── README.md           # Documentation
├── LICENSE             # MIT License
├── src/                # Source code (.zx files)
│   ├── index.zx        # Main entry point
│   └── ...             # Other modules
├── tests/              # Test suite
│   └── ...             # Test files
├── examples/           # Usage examples
│   └── ...             # Example files
└── docs/               # Additional documentation
    └── ...             # Documentation files
```

## Package Manifest (zexus.json)

```json
{
  "name": "@zexus/package-name",
  "version": "1.0.0",
  "description": "Package description",
  "author": "Zexus Team",
  "license": "MIT",
  "main": "src/index.zx",
  "keywords": ["zexus", "category"],
  "repository": {
    "type": "git",
    "url": "https://github.com/zexus/package-name"
  },
  "dependencies": {
    "other-package": "^1.0.0"
  },
  "devDependencies": {
    "@zexus/test": "^1.0.0"
  }
}
```

## Development Guidelines

### 1. Code Quality

- Write idiomatic Zexus code
- Follow naming conventions (snake_case)
- Comprehensive documentation
- 100% test coverage
- Use type annotations

### 2. Testing

All packages must include tests:

```zexus
use {describe, it, expect} from "@zexus/test"
use {MyClass} from "../src/index"

describe("MyClass", action() {
    it("does something", action() {
        let instance = MyClass()
        expect(instance.value).to_equal(42)
    })
})
```

### 3. Documentation

- README with examples
- API documentation
- Usage guides
- Migration guides

### 4. Versioning

Follow Semantic Versioning (semver):
- MAJOR: Breaking changes
- MINOR: New features
- PATCH: Bug fixes

## Publishing

```bash
# Build package
zpm build

# Test package
zpm test

# Publish to registry
zpm publish
```

## Official Packages

### @zexus/web
Full-stack web framework

### @zexus/db
Database ORM and drivers

### @zexus/ai
Machine learning framework

### @zexus/gui
Cross-platform GUI framework

### @zexus/cli
CLI framework

### @zexus/test
Testing framework

See individual package documentation for details.

## Community Packages

Community members can create packages following the same structure.

Non-official packages should NOT use the `@zexus/` scope.

---

**Last Updated**: 2025-12-29
