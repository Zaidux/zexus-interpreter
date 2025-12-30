# HTTP Server - Phase 1

**Status**: Planned
**Phase**: 1 - Build WITH Zexus
**Priority**: High
**Estimated Effort**: 3-4 months

## Overview

Build a full-featured HTTP server **entirely in Zexus** to prove the language can handle:
- Network I/O
- Concurrent connections
- Protocol parsing
- Real-world performance requirements

## Goals

1. **Prove Zexus viability** for systems programming
2. **Establish patterns** for I/O and concurrency
3. **Performance baseline** for native keyword comparison (Phase 2)
4. **Foundation** for @zexus/web package (Phase 3)

## Architecture

### Module Structure

```
zexus/http/
├── server.zx          # Main server implementation
├── request.zx         # Request parsing
├── response.zx        # Response building
├── router.zx          # Routing logic
├── middleware.zx      # Middleware system
├── socket.zx          # TCP socket handling
└── parser.zx          # HTTP protocol parser
```

### Core Components

#### 1. TCP Socket Layer

```zexus
# socket.zx
module zexus.http.socket {
    export action create_socket(port: integer) -> Socket {
        # Use native socket primitives
        # Non-blocking I/O
        # Connection management
    }
    
    export action accept_connection(socket: Socket) -> Connection {
        # Accept incoming connections
        # Return connection object
    }
    
    export action read_from_socket(conn: Connection, buffer_size: integer) -> string {
        # Read data from connection
        # Handle partial reads
    }
    
    export action write_to_socket(conn: Connection, data: string) {
        # Write data to connection
        # Handle partial writes
    }
}
```

#### 2. HTTP Parser

```zexus
# parser.zx
module zexus.http.parser {
    export action parse_request(raw: string) -> Request {
        # Parse HTTP request line
        # Parse headers
        # Parse body
        # Handle chunked encoding
        # Validate HTTP/1.1 compliance
    }
    
    export action parse_headers(header_text: string) -> map {
        # Split into key-value pairs
        # Handle multi-line headers
        # Case-insensitive keys
    }
    
    export action parse_query_string(query: string) -> map {
        # Parse URL parameters
        # Handle URL decoding
    }
}
```

#### 3. Request Object

```zexus
# request.zx
data Request {
    method: string           # GET, POST, etc.
    path: string             # /api/users
    query: map               # URL parameters
    headers: map             # Request headers
    body: string             # Request body
    version: string          # HTTP/1.1
    
    # Computed properties
    get content_type() {
        return this.headers.get("content-type", "text/plain")
    }
    
    get is_json() {
        return this.content_type.contains("application/json")
    }
    
    # Methods
    json() {
        if this.is_json {
            return parse_json(this.body)
        }
        return null
    }
    
    param(name: string, default: any) {
        return this.query.get(name, default)
    }
}
```

#### 4. Response Builder

```zexus
# response.zx
data Response {
    status_code: integer = 200
    headers: map = {}
    body: string = ""
    
    # Fluent API
    status(code: integer) {
        this.status_code = code
        return this
    }
    
    header(key: string, value: string) {
        this.headers[key] = value
        return this
    }
    
    json(data: any) {
        this.headers["Content-Type"] = "application/json"
        this.body = stringify_json(data)
        return this
    }
    
    send(data: string) {
        this.body = data
        return this
    }
    
    html(content: string) {
        this.headers["Content-Type"] = "text/html"
        this.body = content
        return this
    }
    
    # Build HTTP response string
    build() -> string {
        let status_line = "HTTP/1.1 " + string(this.status_code) + " " + get_status_text(this.status_code)
        
        let header_lines = []
        for each key in keys(this.headers) {
            header_lines.push(key + ": " + this.headers[key])
        }
        header_lines.push("Content-Length: " + string(length(this.body)))
        
        return status_line + "\r\n" + 
               join(header_lines, "\r\n") + "\r\n\r\n" + 
               this.body
    }
}
```

#### 5. Router

```zexus
# router.zx
data Route {
    method: string
    path: string
    handler: action
    middleware: list = []
}

module zexus.http.router {
    let routes = []
    
    export action route(method: string, path: string, handler: action) {
        routes.push(Route({
            method: method,
            path: path,
            handler: handler
        }))
    }
    
    export action get(path: string, handler: action) {
        route("GET", path, handler)
    }
    
    export action post(path: string, handler: action) {
        route("POST", path, handler)
    }
    
    export action put(path: string, handler: action) {
        route("PUT", path, handler)
    }
    
    export action delete(path: string, handler: action) {
        route("DELETE", path, handler)
    }
    
    export action match_route(method: string, path: string) -> Route {
        for each route in routes {
            if route.method == method {
                # Simple string match for now
                # TODO: Pattern matching, parameters
                if route.path == path {
                    return route
                }
            }
        }
        return null
    }
}
```

#### 6. Middleware System

```zexus
# middleware.zx
module zexus.http.middleware {
    let middleware_stack = []
    
    export action use(middleware_fn: action) {
        middleware_stack.push(middleware_fn)
    }
    
    export action execute_middleware(req: Request, res: Response) -> boolean {
        for each middleware in middleware_stack {
            let result = middleware(req, res)
            if result == false {
                # Middleware stopped the chain
                return false
            }
        }
        return true
    }
}
```

#### 7. Server

```zexus
# server.zx
use {create_socket, accept_connection, read_from_socket, write_to_socket} from "./socket"
use {parse_request} from "./parser"
use {Request} from "./request"
use {Response} from "./response"
use {match_route} from "./router"
use {execute_middleware} from "./middleware"

data Server {
    port: integer
    socket: Socket = null
    running: boolean = false
    
    start() {
        this.socket = create_socket(this.port)
        this.running = true
        
        print("Server listening on port " + string(this.port))
        
        while this.running {
            # Accept connection
            let conn = accept_connection(this.socket)
            
            # Handle in async context
            async handle_connection(conn)
        }
    }
    
    stop() {
        this.running = false
        close_socket(this.socket)
    }
}

async action handle_connection(conn: Connection) {
    try {
        # Read request
        let raw_request = read_from_socket(conn, 4096)
        
        # Parse request
        let req = parse_request(raw_request)
        let res = Response()
        
        # Execute middleware
        let continue_chain = execute_middleware(req, res)
        
        if continue_chain {
            # Match route
            let route = match_route(req.method, req.path)
            
            if route != null {
                # Execute handler
                route.handler(req, res)
            } else {
                # 404 Not Found
                res.status(404).send("Not Found")
            }
        }
        
        # Send response
        let response_text = res.build()
        write_to_socket(conn, response_text)
        
    } catch (error) {
        # 500 Internal Server Error
        let error_response = Response()
            .status(500)
            .send("Internal Server Error")
        write_to_socket(conn, error_response.build())
    } finally {
        # Close connection
        close_connection(conn)
    }
}

export action create_server(port: integer) -> Server {
    return Server({port: port})
}
```

## Usage Example

### Basic Server

```zexus
use {create_server, get, post} from "zexus/http/server"

# Create server
let app = create_server(8080)

# Define routes
get("/", action(req, res) {
    res.send("Hello, World!")
})

get("/users/:id", action(req, res) {
    let user_id = req.params.id
    let user = fetch_user(user_id)
    res.json(user)
})

post("/users", action(req, res) {
    let user_data = req.json()
    verify is_email(user_data.email)
    
    let new_user = create_user(user_data)
    res.status(201).json(new_user)
})

# Start server
app.start()
```

### With Middleware

```zexus
use {create_server, use as middleware, get} from "zexus/http/server"

let app = create_server(3000)

# Logging middleware
middleware(action(req, res) {
    print("[" + timestamp() + "] " + req.method + " " + req.path)
    return true  # Continue chain
})

# Auth middleware
middleware(action(req, res) {
    let token = req.headers.get("authorization", "")
    if !validate_token(token) {
        res.status(401).json({error: "Unauthorized"})
        return false  # Stop chain
    }
    return true
})

# Route
get("/protected", action(req, res) {
    res.json({message: "You are authenticated!"})
})

app.start()
```

### REST API Example

```zexus
use {create_server, get, post, put, delete} from "zexus/http/server"

let app = create_server(8080)

# In-memory storage
let users = []
let next_id = 1

# GET /users - List all users
get("/users", action(req, res) {
    res.json(users)
})

# GET /users/:id - Get specific user
get("/users/:id", action(req, res) {
    let id = int(req.params.id)
    let user = users.find(action(u) { u.id == id })
    
    if user != null {
        res.json(user)
    } else {
        res.status(404).json({error: "User not found"})
    }
})

# POST /users - Create user
post("/users", action(req, res) {
    let data = req.json()
    
    # Validation
    verify is_email(data.email)
    restrict(data.name, {min_length: 1, max_length: 100})
    
    let user = {
        id: next_id,
        name: data.name,
        email: data.email,
        created_at: now()
    }
    
    users.push(user)
    next_id = next_id + 1
    
    res.status(201).json(user)
})

# PUT /users/:id - Update user
put("/users/:id", action(req, res) {
    let id = int(req.params.id)
    let data = req.json()
    
    let index = users.find_index(action(u) { u.id == id })
    
    if index >= 0 {
        users[index].name = data.name
        users[index].email = data.email
        res.json(users[index])
    } else {
        res.status(404).json({error: "User not found"})
    }
})

# DELETE /users/:id - Delete user
delete("/users/:id", action(req, res) {
    let id = int(req.params.id)
    let index = users.find_index(action(u) { u.id == id })
    
    if index >= 0 {
        let deleted = users.remove_at(index)
        res.json({message: "User deleted", user: deleted})
    } else {
        res.status(404).json({error: "User not found"})
    }
})

app.start()
```

## Performance Requirements

### Target Benchmarks

- **Throughput**: 10,000+ requests/second
- **Latency**: < 10ms average response time
- **Concurrency**: 1,000+ simultaneous connections
- **Memory**: < 100MB for idle server

### Optimization Strategies

1. **Connection pooling**
2. **Request/response object pooling**
3. **Async/await for I/O**
4. **VM-optimized hot paths**
5. **Minimal allocations**
6. **Efficient string building**

## Testing Strategy

### Unit Tests

```zexus
use {describe, it, expect} from "@zexus/test"
use {parse_request} from "zexus/http/parser"

describe("HTTP Parser", action() {
    it("parses GET request", action() {
        let raw = "GET /users HTTP/1.1\r\nHost: example.com\r\n\r\n"
        let req = parse_request(raw)
        
        expect(req.method).to_equal("GET")
        expect(req.path).to_equal("/users")
        expect(req.headers.host).to_equal("example.com")
    })
    
    it("parses POST with body", action() {
        let raw = "POST /users HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"name\":\"Alice\"}"
        let req = parse_request(raw)
        
        expect(req.method).to_equal("POST")
        expect(req.body).to_equal("{\"name\":\"Alice\"}")
    })
})
```

### Integration Tests

```zexus
# Test complete request/response cycle
use {create_server, get} from "zexus/http/server"

let app = create_server(8081)

get("/test", action(req, res) {
    res.json({success: true})
})

# Start server in background
async app.start()

# Make request
let response = http_get("http://localhost:8081/test")
expect(response.status).to_equal(200)
expect(response.body.success).to_equal(true)

app.stop()
```

### Load Tests

```bash
# Use external tools
ab -n 10000 -c 100 http://localhost:8080/
wrk -t4 -c100 -d30s http://localhost:8080/
```

## Security Considerations

### Built-in Security Features

1. **Request size limits**
   ```zexus
   let MAX_REQUEST_SIZE = 1048576  # 1MB
   ```

2. **Timeout handling**
   ```zexus
   let REQUEST_TIMEOUT = 30  # seconds
   ```

3. **Header validation**
   ```zexus
   action validate_headers(headers: map) {
       # Prevent header injection
       # Limit header count
       # Validate header names
   }
   ```

4. **Path traversal prevention**
   ```zexus
   action sanitize_path(path: string) {
       # Remove ../ 
       # Validate characters
   }
   ```

## Migration Path

### Phase 1 (Current)
Pure Zexus implementation

### Phase 2 (Future)
Native HTTP keywords replace core functionality:
```zexus
server app on port 8080 {
    route GET "/users" { /* ... */ }
}
```

### Phase 3 (Future)
@zexus/web package wraps both:
```zexus
use {Server} from "@zexus/web"
```

## Development Timeline

### Month 1: Foundation
- [ ] Socket layer
- [ ] HTTP parser
- [ ] Request/Response objects

### Month 2: Core Features
- [ ] Router
- [ ] Middleware system
- [ ] Basic server implementation

### Month 3: Advanced Features
- [ ] Performance optimization
- [ ] Error handling
- [ ] Security hardening

### Month 4: Polish & Testing
- [ ] Comprehensive tests
- [ ] Documentation
- [ ] Examples
- [ ] Benchmarks

## Success Criteria

- [ ] Passes HTTP/1.1 compliance tests
- [ ] Handles 10k+ req/sec
- [ ] < 10ms latency
- [ ] 100% test coverage
- [ ] Complete documentation
- [ ] 10+ real-world examples

## Related Documentation

- [Ecosystem Strategy](../../ECOSYSTEM_STRATEGY.md)
- [HTTP Keywords (Phase 2)](../HTTP_KEYWORDS.md)
- [@zexus/web Package (Phase 3)](../../packages/ZEXUS_WEB_PACKAGE.md)

---

**Status**: Planning
**Last Updated**: 2025-12-29
**Next Review**: Q1 2025
