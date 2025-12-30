# HTTP Native Keywords - Phase 2

**Status**: Future (After Phase 1)
**Phase**: 2 - Integrate INTO Zexus
**Priority**: High
**Dependencies**: HTTP Server (Phase 1)

## Overview

Make HTTP a **first-class language feature** by introducing native keywords for:
- HTTP servers
- HTTP clients
- Routing
- Request/response handling
- WebSocket support

## Why Native?

1. **Web-first philosophy** - Modern apps need HTTP
2. **Eliminate boilerplate** - Simpler, cleaner syntax
3. **Performance** - Language-level optimization
4. **Seamless async** - Built-in async/await integration

## Server Keyword

### Basic Syntax

```zexus
server app on port 8080 {
    route GET "/" {
        return "Hello, World!"
    }
    
    route GET "/users" {
        let users = fetch_users()
        return json(users)
    }
    
    route POST "/users" {
        let data = request.body
        verify is_email(data.email)
        
        let user = create_user(data)
        return created(user)
    }
}
```

### Full Syntax

```zexus
server app on port 8080 {
    # Configuration
    config {
        cors: true
        max_body_size: 1048576  # 1MB
        timeout: 30
    }
    
    # Middleware
    middleware auth {
        if !request.has_header("authorization") {
            return unauthorized()
        }
    }
    
    # Routes
    route GET "/api/users" {
        let users = database.query("SELECT * FROM users")
        return json(users)
    }
    
    route GET "/api/users/:id" {
        let id = params.id
        let user = database.find_user(id)
        
        if user == null {
            return not_found()
        }
        
        return json(user)
    }
    
    route POST "/api/users" middleware auth {
        let data = request.body
        
        # Validation
        verify is_email(data.email)
        restrict(data.name, {min_length: 1, max_length: 100})
        
        # Create user
        let user = database.create_user(data)
        
        return created(user)
    }
    
    route PUT "/api/users/:id" middleware auth {
        let id = params.id
        let data = request.body
        
        let user = database.update_user(id, data)
        return json(user)
    }
    
    route DELETE "/api/users/:id" middleware auth {
        let id = params.id
        database.delete_user(id)
        
        return no_content()
    }
}
```

### Request Object

Automatically available in route handlers:

```zexus
route POST "/api/data" {
    # Request properties
    let method = request.method          # "POST"
    let path = request.path              # "/api/data"
    let query = request.query            # URL parameters
    let headers = request.headers        # Request headers
    let body = request.body              # Request body
    let cookies = request.cookies        # Cookies
    
    # Computed properties
    let is_json = request.is_json        # true/false
    let content_type = request.content_type
    
    # Methods
    let data = request.json()            # Parse JSON body
    let form = request.form()            # Parse form data
    let file = request.file("upload")    # Get uploaded file
}
```

### Response Helpers

Built-in response helpers:

```zexus
route GET "/api/users" {
    let users = fetch_users()
    
    # JSON response (200)
    return json(users)
    
    # Created (201)
    return created(new_user)
    
    # No content (204)
    return no_content()
    
    # Not found (404)
    return not_found()
    
    # Unauthorized (401)
    return unauthorized()
    
    # Bad request (400)
    return bad_request("Invalid data")
    
    # Server error (500)
    return server_error("Something went wrong")
    
    # Custom response
    return response(418, "I'm a teapot")
    
    # Redirect
    return redirect("/login")
    
    # File download
    return file("report.pdf")
}
```

### Middleware

Define reusable middleware:

```zexus
# Global middleware
server app {
    # Logging
    middleware logger {
        print("[" + timestamp() + "] " + request.method + " " + request.path)
    }
    
    # CORS
    middleware cors {
        response.header("Access-Control-Allow-Origin", "*")
    }
    
    # Authentication
    middleware auth {
        let token = request.header("authorization")
        if !validate_token(token) {
            return unauthorized()
        }
        
        # Add user to request
        request.user = get_user_from_token(token)
    }
    
    # Apply to specific route
    route GET "/protected" middleware auth {
        return json({user: request.user})
    }
}
```

## HTTP Client Keyword

### Basic Usage

```zexus
# GET request
let response = http.get("https://api.example.com/users")
print(response.status)      # 200
print(response.body)        # Response text
let data = response.json()  # Parse JSON

# POST request
let response = http.post("https://api.example.com/users", {
    body: json({name: "Alice", email: "alice@example.com"}),
    headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer token123"
    }
})

# Other methods
let response = http.put(url, options)
let response = http.delete(url, options)
let response = http.patch(url, options)
```

### Async HTTP

```zexus
# Async HTTP requests
async action fetch_data() {
    let response = await http.get("https://api.example.com/data")
    return response.json()
}

# Parallel requests
let results = await Promise.all([
    http.get("https://api.example.com/users"),
    http.get("https://api.example.com/posts"),
    http.get("https://api.example.com/comments")
])
```

### Advanced Options

```zexus
let response = http.get("https://api.example.com/data", {
    headers: {
        "Authorization": "Bearer token",
        "User-Agent": "Zexus HTTP Client"
    },
    timeout: 5000,           # 5 seconds
    follow_redirects: true,
    max_redirects: 5,
    verify_ssl: true
})
```

## WebSocket Support

```zexus
server app on port 8080 {
    # WebSocket endpoint
    websocket "/ws" {
        on_connect {
            print("Client connected")
        }
        
        on_message(data) {
            print("Received: " + data)
            # Echo back
            send(data)
        }
        
        on_close {
            print("Client disconnected")
        }
    }
}

# WebSocket client
let ws = websocket.connect("ws://localhost:8080/ws")

ws.on_message(action(data) {
    print("Received: " + data)
})

ws.send("Hello, Server!")

ws.close()
```

## Static Files

```zexus
server app on port 8080 {
    # Serve static files
    static "/public" from "./public"
    
    # Or inline
    route GET "/assets/*" {
        return file(request.path)
    }
}
```

## Template Rendering

```zexus
server app {
    # Configure template engine
    templates from "./views"
    
    route GET "/" {
        return render("index.html", {
            title: "Home",
            users: fetch_users()
        })
    }
}
```

## Complete Example: REST API

```zexus
# REST API with HTTP native keywords
server api on port 3000 {
    config {
        cors: true
        max_body_size: 1048576
    }
    
    middleware logger {
        print("[" + request.method + "] " + request.path)
    }
    
    middleware auth {
        if request.path.starts_with("/api") {
            let token = request.header("authorization")
            if !validate_token(token) {
                return unauthorized()
            }
        }
    }
    
    # Public routes
    route GET "/" {
        return json({
            name: "My API",
            version: "1.0.0",
            status: "running"
        })
    }
    
    route POST "/auth/login" {
        let credentials = request.json()
        
        let user = authenticate(credentials.email, credentials.password)
        if user == null {
            return unauthorized()
        }
        
        let token = generate_token(user)
        return json({token: token})
    }
    
    # Protected routes
    route GET "/api/users" middleware auth {
        let users = database.query("SELECT * FROM users")
        return json(users)
    }
    
    route GET "/api/users/:id" middleware auth {
        let user = database.find_user(params.id)
        
        if user == null {
            return not_found()
        }
        
        return json(user)
    }
    
    route POST "/api/users" middleware auth {
        let data = request.json()
        
        verify is_email(data.email)
        
        let user = database.create_user(data)
        return created(user)
    }
    
    route PUT "/api/users/:id" middleware auth {
        let user = database.update_user(params.id, request.json())
        return json(user)
    }
    
    route DELETE "/api/users/:id" middleware auth {
        database.delete_user(params.id)
        return no_content()
    }
}
```

## Performance Benefits

Native HTTP keywords enable:

1. **Zero-copy** where possible
2. **Connection pooling** at language level
3. **HTTP/2** and **HTTP/3** support
4. **JIT optimization** for hot paths
5. **Async I/O** integration

## Migration from Phase 1

Phase 1 HTTP server code:

```zexus
use {create_server, get, post} from "zexus/http/server"

let app = create_server(8080)

get("/users", action(req, res) {
    res.json(fetch_users())
})

app.start()
```

Becomes Phase 2 native syntax:

```zexus
server app on port 8080 {
    route GET "/users" {
        return json(fetch_users())
    }
}
```

## Implementation Notes

### Compiler Support

Native HTTP keywords compile to optimized bytecode:

```
SERVER_START port
ROUTE_DEFINE method path handler
MIDDLEWARE_ADD name handler
SERVER_RUN
```

### Runtime Integration

HTTP server runs in separate async context:
- Non-blocking I/O
- Connection management
- Request parsing
- Response building

## Related Documentation

- [HTTP Server (Phase 1)](./features/HTTP_SERVER.md)
- [@zexus/web Package (Phase 3)](../packages/ZEXUS_WEB_PACKAGE.md)
- [Ecosystem Strategy](../ECOSYSTEM_STRATEGY.md)

---

**Status**: Planned (After Phase 1)
**Last Updated**: 2025-12-29
**Next Review**: Q3 2025
