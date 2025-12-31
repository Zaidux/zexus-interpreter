# Zexus Ecosystem Complete Guide

**Version**: 1.0.0  
**Date**: December 30, 2025  
**Status**: ✅ Production Ready

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Testing Framework](#testing-framework)
4. [HTTP Client](#http-client)
5. [HTTP Server](#http-server)
6. [Socket/TCP Primitives](#sockettcp-primitives)
7. [Database Drivers](#database-drivers)
   - [SQLite](#sqlite)
   - [PostgreSQL](#postgresql)
   - [MySQL](#mysql)
   - [MongoDB](#mongodb)
8. [File Operations](#file-operations)
9. [Concurrency](#concurrency)
10. [Examples](#examples)

---

## Overview

The Zexus Ecosystem provides a complete set of tools for building modern applications:

- **Testing Framework**: Write and run tests with assertions
- **HTTP Client**: Make HTTP requests (GET, POST, PUT, DELETE)
- **HTTP Server**: Build web servers with routing
- **Socket/TCP**: Low-level network programming
- **Databases**: SQLite, PostgreSQL, MySQL, MongoDB support
- **File I/O**: Complete file system operations
- **Concurrency**: Async/await, channels, spawn

**✅ 100% Complete** - All features implemented and tested!

---

## Installation

### Prerequisites

```bash
# Core Zexus (required)
git clone https://github.com/Zaidux/zexus-interpreter
cd zexus-interpreter

# Optional database drivers
pip install psycopg2-binary      # PostgreSQL
pip install mysql-connector-python  # MySQL
pip install pymongo              # MongoDB
# SQLite is built into Python - no installation needed
```

### Verify Installation

```bash
./zx-run --version
```

---

## Testing Framework

### Location
`src/zexus/stdlib/test.zx`

### Assertions

```zexus
# Load test framework
eval_file("src/zexus/stdlib/test.zx")

# Assertions
assert_eq(1 + 1, 2, "Addition works")
assert_true(5 > 3, "Comparison works")
assert_false(5 < 3, "Negative comparison")
assert_null(null, "Null check")
assert_type(42, "Integer", "Type checking")

# Results
print_test_results()  # Shows pass/fail summary
```

### Creating Test Files

```zexus
#!/usr/bin/env zexus
# mytest.zx

# Load framework
eval_file("src/zexus/stdlib/test.zx")

# Test suite
print("=== My Test Suite ===\n")

# Test 1
let result = my_function(5)
assert_eq(result, 10, "my_function doubles input")

# Test 2  
assert_true(file_exists("data.txt"), "Data file exists")

# Results
print_test_results()
```

### Run Tests

```bash
./zx-run mytest.zx
```

---

## HTTP Client

### Basic Requests

```zexus
# GET request
let response = http_get("https://api.example.com/users")
print("Status: " + string(response["status"]))
print("Body: " + response["body"])

# POST with JSON
let data = {"name": "Alice", "age": 30}
let response = http_post("https://api.example.com/users", data)

# PUT request
let update = {"age": 31}
let response = http_put("https://api.example.com/users/1", update)

# DELETE request
let response = http_delete("https://api.example.com/users/1")
```

### With Headers and Timeout

```zexus
# Custom headers
let headers = {"Authorization": "Bearer token123", "Content-Type": "application/json"}
let response = http_get("https://api.example.com/secure", headers, 10)

# Response structure
# response["status"]  - HTTP status code (200, 404, etc.)
# response["headers"] - Response headers
# response["body"]    - Response body
# response["error"]   - Error message (if any)
```

---

## HTTP Server

### Basic Server

```zexus
# Create server
let server = http_server(8080)

# Get methods
let get_fn = server["get"]
let post_fn = server["post"]
let listen_fn = server["listen"]
let stop_fn = server["stop"]

# Register routes
get_fn("/", action(req, res) {
    res["send"]("Hello World!")
})

get_fn("/api/status", action(req, res) {
    res["json"]({"status": "ok", "uptime": 3600})
})

post_fn("/api/users", action(req, res) {
    # Access request
    # req["method"]  - HTTP method
    # req["path"]    - Request path
    # req["headers"] - Request headers
    # req["body"]    - Request body
    # req["query"]   - Query parameters
    
    res["json"]({"message": "User created"})
})

# Start server (runs in background thread)
listen_fn()
print("Server running on http://localhost:8080")

# Keep alive
sleep(3600)

# Stop server
stop_fn()
```

### Response Methods

```zexus
# Send text
res["send"]("Plain text response")

# Send JSON
res["json"]({"key": "value"})

# Set status
res["status"](404)
res["send"]("Not Found")
```

---

## Socket/TCP Primitives

### TCP Server

```zexus
# Create server
let server = socket_listen(8080, "0.0.0.0")
let accept_fn = server["accept"]
let close_fn = server["close"]

print("Server listening on port 8080")

# Accept connection
let client = accept_fn()
let recv_fn = client["recv"]
let send_fn = client["send"]
let client_close_fn = client["close"]

# Receive data
let data = recv_fn(1024)
print("Received: " + data)

# Send response
send_fn("Echo: " + data)

# Close
client_close_fn()
close_fn()
```

### TCP Client

```zexus
# Connect to server
let client = socket_connect("localhost", 8080)
let send_fn = client["send"]
let recv_fn = client["recv"]
let close_fn = client["close"]

# Send data
send_fn("Hello Server!")

# Receive response
let response = recv_fn(1024)
print("Response: " + response)

# Close
close_fn()
```

---

## Database Drivers

### SQLite

#### Connect

```zexus
# In-memory database
let db = sqlite_connect(":memory:")

# File database
let db = sqlite_connect("myapp.db")

# Get methods
let exec_fn = db["execute"]
let query_fn = db["query"]
let query_one_fn = db["query_one"]
let close_fn = db["close"]
```

#### Create Table

```zexus
exec_fn("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, age INTEGER)")
```

#### Insert Data

```zexus
# Single insert
exec_fn("INSERT INTO users (name, email, age) VALUES ('Alice', 'alice@example.com', 30)")

# With parameters (prevents SQL injection)
let params = ["Bob", "bob@example.com", 25]
exec_fn("INSERT INTO users (name, email, age) VALUES (?, ?, ?)", params)
```

#### Query Data

```zexus
# Get all users
let users = query_fn("SELECT * FROM users")
print("Found " + string(len(users)) + " users")

# Get one user
let user = query_one_fn("SELECT * FROM users WHERE name = 'Alice'")
if user != null {
    print("Name: " + user["name"])
    print("Email: " + user["email"])
}

# With parameters
let params = ["alice@example.com"]
let user = query_one_fn("SELECT * FROM users WHERE email = ?", params)
```

#### Update and Delete

```zexus
# Update
exec_fn("UPDATE users SET age = 31 WHERE name = 'Alice'")

# Delete
exec_fn("DELETE FROM users WHERE age < 25")
```

#### Transactions

```zexus
let begin_fn = db["begin"]
let commit_fn = db["commit"]
let rollback_fn = db["rollback"]

# Start transaction
begin_fn()

# Multiple operations
exec_fn("INSERT INTO users ...")
exec_fn("UPDATE accounts ...")

# Commit or rollback
if success {
    commit_fn()
} else {
    rollback_fn()
}
```

---

### PostgreSQL

#### Connect

```zexus
# postgres_connect(database, user, password, host?, port?)
let db = postgres_connect("mydb", "postgres", "password123")

# With custom host/port
let db = postgres_connect("mydb", "user", "pass", "192.168.1.100", 5432)
```

#### Usage

PostgreSQL uses the **same interface as SQLite**:

```zexus
let exec_fn = db["execute"]
let query_fn = db["query"]
let query_one_fn = db["query_one"]
let commit_fn = db["commit"]
let rollback_fn = db["rollback"]
let close_fn = db["close"]

# Create table
exec_fn("CREATE TABLE products (id SERIAL PRIMARY KEY, name VARCHAR(100), price DECIMAL(10,2))")

# Insert with RETURNING (PostgreSQL feature)
exec_fn("INSERT INTO products (name, price) VALUES ('Widget', 19.99) RETURNING id")

# Query
let products = query_fn("SELECT * FROM products WHERE price > 10.00")
```

---

### MySQL

#### Connect

```zexus
# mysql_connect(database, user, password, host?, port?)
let db = mysql_connect("mydb", "root", "password123")

# With custom host/port
let db = mysql_connect("mydb", "user", "pass", "192.168.1.100", 3306)
```

#### Usage

MySQL uses the **same interface as SQLite/PostgreSQL**:

```zexus
let exec_fn = db["execute"]
let query_fn = db["query"]
let query_one_fn = db["query_one"]

# Create table
exec_fn("CREATE TABLE orders (id INT AUTO_INCREMENT PRIMARY KEY, customer VARCHAR(100), total DECIMAL(10,2))")

# Insert
exec_fn("INSERT INTO orders (customer, total) VALUES ('John Doe', 99.99)")

# Get last insert ID
let last_id_fn = db["last_insert_id"]
print("Inserted ID: " + string(last_id_fn()))

# Query
let orders = query_fn("SELECT * FROM orders")
```

---

### MongoDB

MongoDB uses a different (NoSQL) interface:

#### Connect

```zexus
# mongo_connect(database, host?, port?, username?, password?)
let db = mongo_connect("myapp")

# With authentication
let db = mongo_connect("myapp", "localhost", 27017, "admin", "password")

# Get methods
let insert_one_fn = db["insert_one"]
let insert_many_fn = db["insert_many"]
let find_fn = db["find"]
let find_one_fn = db["find_one"]
let update_one_fn = db["update_one"]
let update_many_fn = db["update_many"]
let delete_one_fn = db["delete_one"]
let delete_many_fn = db["delete_many"]
let count_fn = db["count"]
let close_fn = db["close"]
```

#### Insert Documents

```zexus
# Insert one
let user = {"name": "Alice", "email": "alice@example.com", "age": 30}
let id = insert_one_fn("users", user)
print("Inserted ID: " + id)

# Insert many
let users = [
    {"name": "Bob", "email": "bob@example.com", "age": 25},
    {"name": "Charlie", "email": "charlie@example.com", "age": 35}
]
let ids = insert_many_fn("users", users)
```

#### Find Documents

```zexus
# Find all
let users = find_fn("users")

# Find with query
let query = {"age": 30}
let users = find_fn("users", query)

# Find one
let user = find_one_fn("users", {"name": "Alice"})
if user != null {
    print("Found: " + user["name"])
}
```

#### Update Documents

```zexus
# Update one
let query = {"name": "Alice"}
let update = {"$set": {"age": 31}}
let count = update_one_fn("users", query, update)
print("Updated " + string(count) + " documents")

# Update many
let query = {"age": {"$lt": 30}}
let update = {"$inc": {"age": 1}}
let count = update_many_fn("users", query, update)
```

#### Delete Documents

```zexus
# Delete one
let count = delete_one_fn("users", {"name": "Bob"})

# Delete many
let count = delete_many_fn("users", {"age": {"$gt": 40}})
```

#### Count Documents

```zexus
# Count all
let total = count_fn("users")

# Count with query
let total = count_fn("users", {"age": {"$gte": 30}})
```

---

## File Operations

### Basic File I/O

```zexus
# Write text
file_write_text("data.txt", "Hello World!")

# Read text
let content = file_read_text("data.txt")

# Append
file_append("log.txt", "New log entry\n")

# Check existence
if file_exists("data.txt") {
    print("File exists")
}

# List directory
let files = file_list_dir(".")
```

### JSON Operations

```zexus
# Write JSON
let data = {"users": [{"name": "Alice", "age": 30}]}
file_write_json("data.json", data)

# Read JSON
let loaded = file_read_json("data.json")
```

### Extended File System

```zexus
# Check types
if fs_is_file("data.txt") {
    print("It's a file")
}

if fs_is_dir("folder") {
    print("It's a directory")
}

# Create directory
fs_mkdir("logs/2025/december")  # Creates all parents

# Copy
fs_copy("source.txt", "dest.txt")
fs_copy("source_dir", "dest_dir")  # Copies directories too

# Rename/move
fs_rename("old.txt", "new.txt")

# Remove
fs_remove("file.txt")
fs_rmdir("folder", true)  # true = recursive
```

---

## Concurrency

### Async Actions

```zexus
# Define async action
async action fetch_data() {
    return 42
}

# Call and await
let task = fetch_data()
let result = await task
print("Result: " + string(result))
```

### Spawn Background Tasks

```zexus
# Spawn task
async action background_job() {
    # Long-running work
    return "Done"
}

let task = spawn(background_job())
# Do other work...
let result = await task
```

### Channels

```zexus
# Create channel
channel<integer>[10] numbers

# Send
send(numbers, 42)
send(numbers, 99)

# Receive
let val1 = receive(numbers)
let val2 = receive(numbers)

# Close
close_channel(numbers)

# Receive from closed channel returns null
let val3 = receive(numbers)  # null
```

---

## Examples

### Complete Web Application

```zexus
#!/usr/bin/env zexus
# Simple REST API with SQLite

# Setup database
let db = sqlite_connect("app.db")
let exec_fn = db["execute"]
let query_fn = db["query"]

exec_fn("CREATE TABLE IF NOT EXISTS todos (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, done INTEGER)")

# Create HTTP server
let server = http_server(3000)
let get_fn = server["get"]
let post_fn = server["post"]
let listen_fn = server["listen"]

# GET /todos - List all todos
get_fn("/todos", action(req, res) {
    let todos = query_fn("SELECT * FROM todos")
    res["json"](todos)
})

# POST /todos - Create todo
post_fn("/todos", action(req, res) {
    # Note: req["body"] is a string, would need JSON parsing
    exec_fn("INSERT INTO todos (title, done) VALUES ('New Todo', 0)")
    res["json"]({"message": "Created"})
})

# Start server
listen_fn()
print("API running on http://localhost:3000")

sleep(3600)
```

### Testing Example

```zexus
#!/usr/bin/env zexus
# test_math.zx

eval_file("src/zexus/stdlib/test.zx")

print("=== Math Tests ===\n")

# Addition
assert_eq(2 + 2, 4, "Addition works")
assert_eq(10 + 5, 15, "Addition with larger numbers")

# Subtraction
assert_eq(10 - 5, 5, "Subtraction works")

# Multiplication
assert_eq(5 * 3, 15, "Multiplication works")

# Division
assert_eq(10 / 2, 5, "Division works")

# Type checking
assert_type(42, "Integer", "Integer type check")

print_test_results()
```

### MongoDB CRUD Example

```zexus
#!/usr/bin/env zexus
# MongoDB Example

let db = mongo_connect("blog")
let insert_one_fn = db["insert_one"]
let find_fn = db["find"]
let update_one_fn = db["update_one"]
let delete_one_fn = db["delete_one"]

# Create post
let post = {
    "title": "Hello MongoDB",
    "author": "Alice",
    "content": "This is my first post!",
    "tags": ["mongodb", "zexus"]
}
let id = insert_one_fn("posts", post)
print("Created post: " + id)

# Find posts
let posts = find_fn("posts", {"author": "Alice"})
print("Found " + string(len(posts)) + " posts")

# Update post
update_one_fn("posts", {"author": "Alice"}, {"$set": {"published": true}})

# Delete post
delete_one_fn("posts", {"author": "Alice"})
```

---

## API Reference Summary

### Testing
- `assert_eq(actual, expected, message)`
- `assert_true(value, message)`
- `assert_false(value, message)`
- `assert_null(value, message)`
- `assert_type(value, expected_type, message)`
- `print_test_results()`

### HTTP Client
- `http_get(url, headers?, timeout?)`
- `http_post(url, data, headers?, timeout?)`
- `http_put(url, data, headers?, timeout?)`
- `http_delete(url, headers?, timeout?)`

### HTTP Server
- `http_server(port, host?)`
- `server["get"](path, handler)`
- `server["post"](path, handler)`
- `server["listen"]()`
- `server["stop"]()`

### Sockets
- `socket_listen(port, host?)`
- `socket_connect(host, port)`
- `connection["send"](data)`
- `connection["recv"](bufsize)`
- `connection["close"]()`

### Databases (SQL)
- `sqlite_connect(path)`
- `postgres_connect(database, user, password, host?, port?)`
- `mysql_connect(database, user, password, host?, port?)`
- `db["execute"](query, params?)`
- `db["query"](query, params?)`
- `db["query_one"](query, params?)`
- `db["begin"]()`
- `db["commit"]()`
- `db["rollback"]()`
- `db["close"]()`

### Database (MongoDB)
- `mongo_connect(database, host?, port?, username?, password?)`
- `db["insert_one"](collection, document)`
- `db["insert_many"](collection, documents)`
- `db["find"](collection, query?)`
- `db["find_one"](collection, query?)`
- `db["update_one"](collection, query, update)`
- `db["update_many"](collection, query, update)`
- `db["delete_one"](collection, query)`
- `db["delete_many"](collection, query)`
- `db["count"](collection, query?)`
- `db["close"]()`

### File Operations
- `file_write_text(path, content)`
- `file_read_text(path)`
- `file_append(path, content)`
- `file_exists(path)`
- `file_list_dir(path)`
- `file_write_json(path, data)`
- `file_read_json(path)`
- `fs_is_file(path)`
- `fs_is_dir(path)`
- `fs_mkdir(path, parents?)`
- `fs_copy(src, dst)`
- `fs_rename(old, new)`
- `fs_remove(path)`
- `fs_rmdir(path, recursive?)`

### Concurrency
- `async action name() { ... }`
- `spawn(coroutine)`
- `await task`
- `channel<type>[size] name`
- `send(channel, value)`
- `receive(channel)`
- `close_channel(channel)`

---

## Support & Contributing

- **GitHub**: https://github.com/Zaidux/zexus-interpreter
- **Documentation**: See `docs/` folder
- **Issues**: GitHub Issues
- **Discord**: [Coming Soon]

---

**🎉 Happy Coding with Zexus!** 🎉
