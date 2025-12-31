# Zexus Database & Web Quick Reference

## 🗄️ Databases

### SQLite (No Dependencies)
```zexus
let db = sqlite_connect("app.db")          # or ":memory:" for in-memory
db["execute"]("CREATE TABLE ...")          # → true/false
db["execute"]("INSERT INTO ...", [params]) # → true/false  
let rows = db["query"]("SELECT ...")       # → List of Maps
let row = db["query_one"]("SELECT ...")    # → Map or null
db["begin"]()                               # Start transaction
db["commit"]()                              # Commit transaction
db["rollback"]()                            # Rollback transaction
db["close"]()                               # Close connection
```

### PostgreSQL (requires: pip install psycopg2-binary)
```zexus
let db = postgres_connect("dbname", "user", "password")
let db = postgres_connect("dbname", "user", "password", "host", 5432)
# Same methods as SQLite: execute, query, query_one, begin, commit, rollback, close
```

### MySQL (requires: pip install mysql-connector-python)
```zexus
let db = mysql_connect("dbname", "user", "password")
let db = mysql_connect("dbname", "user", "password", "host", 3306)
let id = db["last_insert_id"]()            # Get last insert ID
# Same methods as SQLite: execute, query, query_one, begin, commit, rollback, close
```

### MongoDB (requires: pip install pymongo)
```zexus
let db = mongo_connect("database")
let db = mongo_connect("database", "host", 27017, "user", "password")

# Insert
let id = db["insert_one"]("collection", {"name": "Alice"})
let ids = db["insert_many"]("collection", [{"name": "Bob"}, {...}])

# Find
let docs = db["find"]("collection")                    # All documents
let docs = db["find"]("collection", {"age": 30})      # With query
let doc = db["find_one"]("collection", {"name": "Alice"})

# Update
let count = db["update_one"]("collection", {"name": "Alice"}, {"$set": {"age": 31}})
let count = db["update_many"]("collection", {"age": {" $lt": 25}}, {"$inc": {"age": 1}})

# Delete
let count = db["delete_one"]("collection", {"name": "Bob"})
let count = db["delete_many"]("collection", {"age": {"$gt": 40}})

# Count
let total = db["count"]("collection")
let total = db["count"]("collection", {"age": {"$gte": 30}})

# Close
db["close"]()
```

## 🌐 HTTP Server

```zexus
let server = http_server(3000)                 # Create server on port 3000
let server = http_server(3000, "0.0.0.0")     # Bind to specific host

# Define routes
server["get"]("/path", action(req, res) {
    # req["method"]  - HTTP method
    # req["path"]    - Request path
    # req["headers"] - Headers map
    # req["body"]    - Request body
    # req["query"]   - Query parameters
    
    res["send"]("Hello!")              # Send text
    res["json"]({"key": "value"})      # Send JSON
    res["status"](404)                 # Set status code
})

server["post"]("/api/users", action(req, res) { ... })
server["put"]("/api/users/:id", action(req, res) { ... })
server["delete"]("/api/users/:id", action(req, res) { ... })

# Start server (runs in background)
server["listen"]()

# Stop server
server["stop"]()
```

## 🔌 Sockets/TCP

### Server
```zexus
let server = socket_listen(8080)               # Listen on port 8080
let server = socket_listen(8080, "0.0.0.0")   # Bind to specific host

let client = server["accept"]()                 # Accept connection (blocks)
let data = client["recv"](1024)                # Receive up to 1024 bytes
client["send"]("Hello Client!")                 # Send data
client["close"]()                               # Close client connection
server["close"]()                               # Close server
```

### Client
```zexus
let conn = socket_connect("localhost", 8080)   # Connect to server
conn["send"]("Hello Server!")                   # Send data
let response = conn["recv"](1024)              # Receive response
conn["close"]()                                 # Close connection
```

## 🧪 Testing Framework

```zexus
# Load test framework
eval_file("src/zexus/stdlib/test.zx")

# Assertions
assert_eq(actual, expected, "message")          # Equality check
assert_true(condition, "message")               # Boolean true
assert_false(condition, "message")              # Boolean false
assert_null(value, "message")                   # Null check
assert_type(value, "TypeName", "message")       # Type check

# Results
print_test_results()                            # Print pass/fail summary
```

## 🔗 HTTP Client

```zexus
let response = http_get("https://api.example.com/data")
let response = http_post("https://api.example.com/users", {"name": "Alice"})
let response = http_put("https://api.example.com/users/1", {"age": 31})
let response = http_delete("https://api.example.com/users/1")

# With headers and timeout
let headers = {"Authorization": "Bearer token"}
let response = http_get("https://api.example.com/secure", headers, 10)

# Response structure
# response["status"]  - HTTP status code
# response["headers"] - Response headers
# response["body"]    - Response body
# response["error"]   - Error message (if any)
```

## 📁 File Operations

```zexus
# Basic I/O
file_write_text("file.txt", "content")
let content = file_read_text("file.txt")
file_append("log.txt", "new line\n")

# JSON
file_write_json("data.json", {"key": "value"})
let data = file_read_json("data.json")

# File system
file_exists("file.txt")                         # → true/false
file_list_dir("path")                           # → List of filenames
fs_is_file("path")                              # → true/false
fs_is_dir("path")                               # → true/false
fs_mkdir("path/to/dir")                         # Create directory (recursive)
fs_copy("src", "dst")                           # Copy file or directory
fs_rename("old", "new")                         # Rename/move
fs_remove("file")                               # Remove file
fs_rmdir("dir", true)                           # Remove directory (recursive)
```

## 🔄 Concurrency

```zexus
# Async actions
async action fetch_data() {
    return 42
}

let task = fetch_data()
let result = await task

# Spawn background tasks
let task = spawn(fetch_data())
let result = await task

# Channels
channel<integer>[10] numbers
send(numbers, 42)
let value = receive(numbers)
close_channel(numbers)
```

## 📋 Complete Example: REST API with Database

```zexus
#!/usr/bin/env zexus
# Simple REST API

# Setup database
let db = sqlite_connect("api.db")
db["execute"]("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")

# Create server
let server = http_server(3000)

# GET /users - List all
server["get"]("/users", action(req, res) {
    let users = db["query"]("SELECT * FROM users")
    res["json"](users)
})

# POST /users - Create
server["post"]("/users", action(req, res) {
    # In real app, parse req["body"] JSON
    db["execute"]("INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')")
    res["json"]({"message": "User created"})
})

# Start server
server["listen"]()
print("API running on http://localhost:3000")

# Keep alive
sleep(3600)
```

## 🎯 Testing Example

```zexus
#!/usr/bin/env zexus
eval_file("src/zexus/stdlib/test.zx")

# Database tests
let db = sqlite_connect(":memory:")
db["execute"]("CREATE TABLE test (id INTEGER, name TEXT)")
db["execute"]("INSERT INTO test VALUES (1, 'Alice')")

let users = db["query"]("SELECT * FROM test")
assert_eq(len(users), 1, "Should have 1 user")
assert_eq(users[0]["name"], "Alice", "Name should be Alice")

print_test_results()
```

---

**For full documentation, see `docs/ECOSYSTEM_GUIDE.md`**
