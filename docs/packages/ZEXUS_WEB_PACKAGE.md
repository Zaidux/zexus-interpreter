# @zexus/web Package Specification

**Status**: Planned (Phase 3)
**Dependencies**: HTTP Server (Phase 1), HTTP Keywords (Phase 2)
**Priority**: High

## Overview

Complete full-stack web framework for Zexus combining:
- HTTP server from Phase 1
- Native HTTP keywords from Phase 2
- Additional web utilities

## Installation

```bash
zpm install @zexus/web
```

## Features

### HTTP Server
- Request/response handling
- Routing and middleware
- Static file serving
- Template rendering
- WebSocket support
- Session management
- Cookie parsing
- CORS support

### Authentication
- JWT tokens
- OAuth2 integration
- Session-based auth
- API key authentication

### Form Handling
- Form parsing
- File uploads
- Validation
- CSRF protection

### Template Engine
- HTML templating
- Variable interpolation
- Conditionals and loops
- Partials and layouts

## Quick Start

```zexus
use {Server, Router, middleware} from "@zexus/web"

let app = Server()
let router = Router()

# Middleware
app.use(middleware.logger())
app.use(middleware.cors())

# Routes
router.get("/", action(req, res) {
    res.send("Welcome!")
})

router.get("/users", action(req, res) {
    let users = fetch_users()
    res.json(users)
})

router.post("/users", action(req, res) {
    let user = create_user(req.body)
    res.status(201).json(user)
})

# Start server
app.use(router)
app.listen(3000)
```

## API Reference

See [API Documentation](./api/zexus-web.md) for complete reference.

## Examples

See [examples directory](../../examples/@zexus/web/) for usage examples.

---

**Status**: Planned
**Last Updated**: 2025-12-29
