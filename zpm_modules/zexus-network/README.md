# Zexus Network Library 🌐

A comprehensive networking library for the Zexus programming language, featuring HTTP clients, WebSockets, TCP/UDP sockets, servers, DNS utilities, and security features - all built with Zexus's powerful async/await system.

## 🚀 Features

### **HTTP Client**
- **Full HTTP/1.1 support** - GET, POST, PUT, DELETE, etc.
- **JSON-specific client** - Automatic serialization/deserialization
- **Async operations** - Non-blocking requests with await
- **Event-driven** - Request/response lifecycle events
- **Customizable** - Headers, timeouts, retries

### **WebSocket Client**
- **Real-time communication** - Bidirectional messaging
- **Auto-reconnect** - Exponential backoff on connection loss
- **Message events** - Reactive programming with events
- **JSON support** - Built-in JSON message handling

### **TCP/UDP Networking**
- **Raw socket support** - Low-level network programming
- **Async I/O** - Non-blocking send/receive operations
- **Connection management** - Automatic error handling and cleanup
- **Event system** - Connection, data, and error events

### **Server Capabilities**
- **TCP Server** - Handle multiple concurrent connections
- **HTTP Server** - Simple web server with routing
- **Async handlers** - Non-blocking request processing
- **Middleware support** - Request preprocessing

### **Network Utilities**
- **DNS resolution** - Hostname to IP address lookup
- **Ping utility** - Network connectivity testing
- **IP validation** - IPv4 and IPv6 address validation
- **Reverse DNS** - IP to hostname resolution

### **Security Features**
- **SSL/TLS support** - Secure socket connections
- **Rate limiting** - Protect against abuse
- **Request signing** - HMAC-based API authentication
- **Certificate support** - Client and server certificates

## 📦 Installation

```zexus
use "zexus-network" as net
```

🎯 Quick Start

HTTP Client Examples

```zexus
use "zexus-network" as net

action async fetch_data() {
    // Simple GET request
    let response = await net.http_get("https://api.example.com/data")
    print("Status: " + string(response.status))
    print("Body: " + response.body)
    
    // JSON API client
    let client = net.json_http_client("https://api.example.com")
    let user = await client.post_json("/users", {
        name: "John Doe",
        email: "john@example.com"
    })
    
    print("User created: " + string(user.id))
}

action async concurrent_requests() {
    // Make multiple requests concurrently
    let tasks = [
        spawn net.http_get("https://api.example.com/users"),
        spawn net.http_get("https://api.example.com/products"),
        spawn net.http_get("https://api.example.com/orders")
    ]
    
    let [users, products, orders] = await all(tasks)
    print("Fetched all data concurrently!")
}
```

WebSocket Examples

```zexus
action async websocket_chat() {
    let ws = net.websocket_client("wss://chat.example.com")
    
    // Handle incoming messages
    ws.on_message(action(message) {
        print("Chat message: " + string(message.data))
    })
    
    // Handle connection events
    net.on_event("websocket_open", action(event) {
        print("Connected to chat server!")
        spawn ws.send("Hello from Zexus!")
    })
    
    if await ws.connect() {
        // Send messages
        await ws.send_json({
            type: "message",
            content: "Hello WebSocket!",
            user: "zexus_user"
        })
        
        // Keep connection alive
        while true {
            await net.sleep(10)
            await ws.send_json({type: "ping"})
        }
    }
}
```

TCP Server Examples

```zexus
action async start_echo_server() {
    let server = net.tcp_server()
    
    server.on_connection(action async (client) {
        let address = client.get_address().toString()
        print("Client connected: " + address)
        
        // Echo back everything received
        while client.is_connected() {
            let data = await client.receive(1024)
            if len(data) > 0 {
                await client.send(data)  // Echo back to client
                print("Echoed " + string(len(data)) + " bytes to " + address)
            }
        }
        
        print("Client disconnected: " + address)
    })
    
    if await server.start(8080) {
        print("Echo server running on port 8080")
        
        // Run for 5 minutes then shutdown
        await net.sleep(300)
        await server.stop()
        print("Server stopped")
    }
}
```

Network Utilities

```zexus
action async network_diagnostics() {
    // DNS resolution
    let addresses = await net.resolve_hostname("google.com")
    print("Google.com IPs: " + string(addresses))
    
    // Ping test
    let result = await net.ping("google.com")
    if result.success {
        print("Ping successful! Latency: " + string(result.latency) + "ms")
    }
    
    // IP validation
    print("Valid IPs: " + string(net.is_valid_ipv4("192.168.1.1")) + ", " + 
          string(net.is_valid_ipv6("2001:db8::1")))
}
```

Security Examples

```zexus
action async secure_api_client() {
    let client = net.http_client("https://secure-api.example.com")
    
    // Rate limiting
    let limiter = net.rate_limiter(60, 100)  // 100 requests per minute
    
    action async make_limited_request() {
        if not limiter.is_rate_limited("api_client") {
            limiter.record_request("api_client")
            return await client.get("/data")
        } else {
            throw "Rate limit exceeded"
        }
    }
    
    // Request signing
    let secret = "my-api-secret"
    let request = {
        method: net.HttpMethod.POST,
        path: "/secure-endpoint",
        body: {data: "sensitive information"}
    }
    
    let signed_request = net.sign_request(request, secret)
    
    // Verify signature on server side
    let is_valid = net.verify_signature(signed_request, secret)
    print("Signature valid: " + string(is_valid))
}
```

📚 Complete API Reference

Core Types

```zexus
// Protocols
net.Protocol.TCP, net.Protocol.UDP, net.Protocol.HTTP, etc.

// HTTP Methods  
net.HttpMethod.GET, net.HttpMethod.POST, net.HttpMethod.PUT, etc.

// HTTP Status Codes
net.HttpStatus.OK, net.HttpStatus.NOT_FOUND, net.HttpStatus.INTERNAL_ERROR, etc.

// Network Address
let addr = net.ip_address("192.168.1.1", 8080, net.Protocol.TCP)
```

HTTP Client

```zexus
let client = net.http_client("https://api.example.com")
let response = await client.get("/path")
let json_client = net.json_http_client()
let data = await json_client.post_json("/path", {key: "value"})
```

WebSocket Client

```zexus
let ws = net.websocket_client("wss://echo.websocket.org")
ws.on_message(handler)
await ws.connect()
await ws.send("message")
await ws.send_json({data: "value"})
await ws.close()
```

Sockets

```zexus
let tcp = net.tcp_socket()
await tcp.connect(net.ip_address("example.com", 80))
await tcp.send([65, 66, 67])  // Send bytes
let data = await tcp.receive(1024)
await tcp.close()
```

Servers

```zexus
let server = net.tcp_server()
server.on_connection(handler)
await server.start(8080)
await server.stop()

let http_server = net.http_server()
http_server.route(net.HttpMethod.GET, "/", handler)
await http_server.start(3000)
```

Utilities

```zexus
await net.resolve_hostname("google.com")
await net.ping("example.com")
net.is_valid_ipv4("192.168.1.1")
net.is_valid_ipv6("2001:db8::1")
```

🔧 Advanced Usage

Event Handling

```zexus
// HTTP events
net.on_event("http_request_started", action(event) {
    print("Starting request to: " + event.request.url)
})

net.on_event("http_response_received", action(event) {
    print("Request completed in " + string(event.response.duration) + "s")
})

// WebSocket events
net.on_event("websocket_open", action(event) {
    print("WebSocket connected: " + event.url)
})

net.on_event("websocket_error", action(event) {
    print("WebSocket error: " + event.error)
})
```

Custom HTTP Headers & Timeouts

```zexus
let client = net.http_client("https://api.example.com", {
    "User-Agent": "MyApp/1.0",
    "Authorization": "Bearer token123"
})

net.set_network_config("timeout", 30)  // 30 second timeout
net.set_network_config("max_retries", 5)
```

Middleware for HTTP Server

```zexus
let server = net.http_server()

// Logging middleware
server.use(action async (request) {
    print(request.method + " " + request.path + " from " + request.remote_addr)
    return request  // Pass to next middleware
})

// Authentication middleware
server.use(action async (request) {
    let token = request.headers.get("Authorization", "")
    if not validate_token(token) {
        return {status: net.HttpStatus.UNAUTHORIZED, body: "Invalid token"}
    }
    return request
})

server.route(net.HttpMethod.GET, "/api/data", action(request) {
    return {status: net.HttpStatus.OK, body: "Protected data"}
})
```

SSL/TLS Secure Connections

```zexus
let secure_client = net.secure_socket("client.crt", "client.key")
await secure_client.connect(net.ip_address("secure-server.com", 443))

let secure_server = net.tcp_server()
// Configure server with certificates...
```

## 📁 File Structure

```

zexus-network/
├──index.zx                      # Main entry point
├──README.md                     # Documentation
└──network/                      # Network module directory
├── core.zx                   # Protocols, constants, base types
├── http.zx                   # HTTP client, requests, responses
├── websocket.zx              # WebSocket client
├── socket.zx                 # TCP/UDP sockets
├── server.zx                 # Server implementations
├── dns.zx                    # DNS resolution, network utilities
├── security.zx               # SSL/TLS, rate limiting, security
└── examples/
└── network_demo.zx       # Comprehensive usage examples

```

🛡️ Security Notes

· All connections support SSL/TLS when available
· Rate limiting prevents API abuse
· Request signing ensures message integrity
· Certificate validation for secure connections
· Input validation on all network operations

📈 Performance

· Async-first design - No blocking operations
· Connection pooling - Reuse HTTP connections
· Efficient buffering - Minimal memory usage
· Event-driven architecture - Scalable for high loads

🤝 Contributing

1. Follow Zexus protocol-based design patterns
2. Include async versions of all network operations
3. Add comprehensive event emission
4. Provide security best practices
5. Include usage examples

📄 License

MIT License - Feel free to use in your Zexus projects!

---

Zexus Network Library - Enterprise-grade networking with Zexus's powerful async/await system! 🚀

Build web servers, API clients, real-time applications, and more with clean, async-first code.

```

## **Key Benefits of This Structure:**

1. **Clean Entry Point** - Single `index.zx` at root level
2. **Modular Design** - Each file handles specific functionality
3. **Easy Imports** - Users just import the main library
4. **Comprehensive Coverage** - HTTP, WebSockets, TCP/UDP, DNS, security
5. **Zexus-Native** - Leverages async/await, events, protocols