# Database Drivers - Phase 1

**Status**: Planned (SQLite exists)
**Phase**: 1 - Build WITH Zexus
**Priority**: High
**Estimated Effort**: 4-6 months

## Overview

Build production-ready database drivers **entirely in Zexus** for:
- PostgreSQL
- MySQL
- MongoDB
- SQLite (already exists)

This proves Zexus can handle:
- Binary protocols
- Connection pooling
- Transaction management
- Real-world database workloads

## Goals

1. **Prove Zexus capabilities** for database integration
2. **Establish patterns** for connection management
3. **Foundation** for DATABASE native keyword (Phase 2)
4. **Core of @zexus/db** package (Phase 3)

## Architecture

### Module Structure

```
zexus/db/
├── common/
│   ├── connection.zx       # Connection interface
│   ├── pool.zx             # Connection pooling
│   ├── query.zx            # Query builder
│   └── transaction.zx      # Transaction management
├── postgres/
│   ├── driver.zx           # PostgreSQL driver
│   ├── protocol.zx         # Wire protocol
│   └── types.zx            # Type mapping
├── mysql/
│   ├── driver.zx           # MySQL driver
│   ├── protocol.zx         # Wire protocol
│   └── types.zx            # Type mapping
├── mongodb/
│   ├── driver.zx           # MongoDB driver
│   ├── bson.zx             # BSON encoding/decoding
│   └── commands.zx         # MongoDB commands
└── sqlite/
    └── driver.zx           # SQLite driver (exists)
```

## PostgreSQL Driver

### Wire Protocol Implementation

```zexus
# postgres/protocol.zx
module zexus.db.postgres.protocol {
    
    # Message types
    const MSG_QUERY = 'Q'
    const MSG_PARSE = 'P'
    const MSG_BIND = 'B'
    const MSG_EXECUTE = 'E'
    const MSG_SYNC = 'S'
    const MSG_TERMINATE = 'X'
    
    export action send_startup_message(socket: Socket, user: string, database: string) {
        # Protocol version 3.0
        let message = build_startup_message(user, database)
        write_to_socket(socket, message)
    }
    
    export action send_query(socket: Socket, sql: string) {
        # Simple query protocol
        let message = MSG_QUERY + int32_to_bytes(length(sql) + 5) + sql + "\0"
        write_to_socket(socket, message)
    }
    
    export action send_parse(socket: Socket, statement: string, sql: string) {
        # Prepared statement
        let message = build_parse_message(statement, sql)
        write_to_socket(socket, message)
    }
    
    export action receive_message(socket: Socket) -> Message {
        # Read message type
        let msg_type = read_bytes(socket, 1)
        
        # Read message length
        let length = bytes_to_int32(read_bytes(socket, 4))
        
        # Read message body
        let body = read_bytes(socket, length - 4)
        
        return Message({
            type: msg_type,
            body: body
        })
    }
    
    action build_startup_message(user: string, database: string) -> bytes {
        # Build PostgreSQL startup message
        # Protocol version 3.0
        let params = {
            "user": user,
            "database": database,
            "client_encoding": "UTF8"
        }
        
        # Encode as null-terminated key-value pairs
        # ... implementation
    }
}
```

### Connection Management

```zexus
# postgres/driver.zx
use {send_startup_message, send_query, receive_message} from "./protocol"

data PostgresConnection {
    host: string
    port: integer
    database: string
    user: string
    password: string
    socket: Socket = null
    transaction_active: boolean = false
    
    connect() {
        # Create TCP connection
        this.socket = create_tcp_socket(this.host, this.port)
        
        # Send startup message
        send_startup_message(this.socket, this.user, this.database)
        
        # Handle authentication
        this.authenticate()
        
        # Wait for ready message
        this.wait_for_ready()
    }
    
    authenticate() {
        let msg = receive_message(this.socket)
        
        match msg.type {
            case 'R':  # Authentication request
                # Handle MD5, SCRAM-SHA-256, etc.
                this.handle_auth(msg)
            default:
                throw "Unexpected message during auth"
        }
    }
    
    query(sql: string) -> QueryResult {
        # Send query
        send_query(this.socket, sql)
        
        # Receive results
        return this.receive_query_result()
    }
    
    prepare(name: string, sql: string) -> PreparedStatement {
        # Create prepared statement
        return PreparedStatement({
            connection: this,
            name: name,
            sql: sql
        })
    }
    
    begin_transaction() {
        if this.transaction_active {
            throw "Transaction already active"
        }
        this.query("BEGIN")
        this.transaction_active = true
    }
    
    commit() {
        if !this.transaction_active {
            throw "No active transaction"
        }
        this.query("COMMIT")
        this.transaction_active = false
    }
    
    rollback() {
        if !this.transaction_active {
            throw "No active transaction"
        }
        this.query("ROLLBACK")
        this.transaction_active = false
    }
    
    close() {
        # Send terminate message
        write_to_socket(this.socket, 'X')
        close_socket(this.socket)
    }
}

export action connect(connection_string: string) -> PostgresConnection {
    # Parse connection string
    # postgresql://user:password@host:port/database
    let parts = parse_connection_string(connection_string)
    
    let conn = PostgresConnection({
        host: parts.host,
        port: parts.port,
        database: parts.database,
        user: parts.user,
        password: parts.password
    })
    
    conn.connect()
    return conn
}
```

### Query Builder

```zexus
# common/query.zx
data QueryBuilder {
    table_name: string
    select_columns: list = ["*"]
    where_clauses: list = []
    order_by_clauses: list = []
    limit_value: integer = null
    offset_value: integer = null
    
    select(...columns) {
        this.select_columns = columns
        return this
    }
    
    where(column: string, operator: string, value: any) {
        this.where_clauses.push({
            column: column,
            operator: operator,
            value: value
        })
        return this
    }
    
    order_by(column: string, direction: string = "ASC") {
        this.order_by_clauses.push({
            column: column,
            direction: direction
        })
        return this
    }
    
    limit(n: integer) {
        this.limit_value = n
        return this
    }
    
    offset(n: integer) {
        this.offset_value = n
        return this
    }
    
    build() -> string {
        # Build SELECT statement
        let sql = "SELECT " + join(this.select_columns, ", ")
        sql = sql + " FROM " + this.table_name
        
        # WHERE clause
        if length(this.where_clauses) > 0 {
            let conditions = []
            for each clause in this.where_clauses {
                conditions.push(clause.column + " " + clause.operator + " ?")
            }
            sql = sql + " WHERE " + join(conditions, " AND ")
        }
        
        # ORDER BY
        if length(this.order_by_clauses) > 0 {
            let orders = []
            for each order in this.order_by_clauses {
                orders.push(order.column + " " + order.direction)
            }
            sql = sql + " ORDER BY " + join(orders, ", ")
        }
        
        # LIMIT
        if this.limit_value != null {
            sql = sql + " LIMIT " + string(this.limit_value)
        }
        
        # OFFSET
        if this.offset_value != null {
            sql = sql + " OFFSET " + string(this.offset_value)
        }
        
        return sql
    }
    
    get_params() -> list {
        # Extract parameter values
        return this.where_clauses.map(action(c) { c.value })
    }
}

export action query_builder(table: string) -> QueryBuilder {
    return QueryBuilder({table_name: table})
}
```

### Connection Pooling

```zexus
# common/pool.zx
data ConnectionPool {
    connection_factory: action
    min_connections: integer = 2
    max_connections: integer = 10
    idle_timeout: integer = 60
    
    available_connections: list = []
    active_connections: list = []
    total_connections: integer = 0
    
    initialize() {
        # Create minimum connections
        for each i in range(0, this.min_connections) {
            let conn = this.connection_factory()
            this.available_connections.push(conn)
            this.total_connections = this.total_connections + 1
        }
    }
    
    acquire() -> Connection {
        # Get available connection
        if length(this.available_connections) > 0 {
            let conn = this.available_connections.pop()
            this.active_connections.push(conn)
            return conn
        }
        
        # Create new connection if under limit
        if this.total_connections < this.max_connections {
            let conn = this.connection_factory()
            this.active_connections.push(conn)
            this.total_connections = this.total_connections + 1
            return conn
        }
        
        # Wait for available connection
        # TODO: Implement waiting with timeout
        throw "Connection pool exhausted"
    }
    
    release(conn: Connection) {
        # Move from active to available
        let index = this.active_connections.index_of(conn)
        if index >= 0 {
            this.active_connections.remove_at(index)
            this.available_connections.push(conn)
        }
    }
    
    close_all() {
        # Close all connections
        for each conn in this.available_connections {
            conn.close()
        }
        for each conn in this.active_connections {
            conn.close()
        }
        this.available_connections = []
        this.active_connections = []
        this.total_connections = 0
    }
}

export action create_pool(factory: action, options: map) -> ConnectionPool {
    let pool = ConnectionPool({
        connection_factory: factory,
        min_connections: options.get("min", 2),
        max_connections: options.get("max", 10),
        idle_timeout: options.get("idle_timeout", 60)
    })
    pool.initialize()
    return pool
}
```

## MySQL Driver

### Implementation Notes

- Similar architecture to PostgreSQL driver
- Different wire protocol (MySQL protocol)
- Authentication: mysql_native_password, caching_sha2_password
- Binary protocol for prepared statements
- Multiple result sets support

```zexus
# mysql/driver.zx
data MySQLConnection {
    # Similar to PostgresConnection
    # Different protocol implementation
}

export action connect(connection_string: string) -> MySQLConnection {
    # mysql://user:password@host:port/database
}
```

## MongoDB Driver

### BSON Support

```zexus
# mongodb/bson.zx
module zexus.db.mongodb.bson {
    export action encode(obj: any) -> bytes {
        # Encode Zexus object to BSON
        # Handle types: string, integer, float, boolean, null
        # Arrays, documents
        # ObjectId, Date, etc.
    }
    
    export action decode(data: bytes) -> any {
        # Decode BSON to Zexus object
    }
}
```

### MongoDB Operations

```zexus
# mongodb/driver.zx
use {encode, decode} from "./bson"

data MongoDBConnection {
    host: string
    port: integer
    database: string
    socket: Socket = null
    
    connect() {
        this.socket = create_tcp_socket(this.host, this.port)
        # MongoDB wire protocol handshake
    }
    
    collection(name: string) -> Collection {
        return Collection({
            connection: this,
            database: this.database,
            name: name
        })
    }
}

data Collection {
    connection: MongoDBConnection
    database: string
    name: string
    
    find(query: map) -> list {
        # Build find command
        let command = {
            "find": this.name,
            "filter": query
        }
        
        # Send command
        let result = this.connection.send_command(this.database, command)
        return result.cursor.firstBatch
    }
    
    insert_one(document: map) {
        let command = {
            "insert": this.name,
            "documents": [document]
        }
        return this.connection.send_command(this.database, command)
    }
    
    update_one(filter: map, update: map) {
        let command = {
            "update": this.name,
            "updates": [{
                "q": filter,
                "u": update
            }]
        }
        return this.connection.send_command(this.database, command)
    }
    
    delete_one(filter: map) {
        let command = {
            "delete": this.name,
            "deletes": [{
                "q": filter,
                "limit": 1
            }]
        }
        return this.connection.send_command(this.database, command)
    }
}
```

## Usage Examples

### PostgreSQL

```zexus
use {connect} from "zexus/db/postgres"

# Connect
let db = connect("postgresql://user:password@localhost:5432/myapp")

# Simple query
let users = db.query("SELECT * FROM users WHERE age > 18")

# Prepared statement
let stmt = db.prepare("get_user", "SELECT * FROM users WHERE id = $1")
let user = stmt.execute([42])

# Transaction
db.begin_transaction()
try {
    db.query("INSERT INTO users (name) VALUES ('Alice')")
    db.query("UPDATE accounts SET balance = balance + 100")
    db.commit()
} catch (error) {
    db.rollback()
    throw error
}

# Close
db.close()
```

### With Connection Pool

```zexus
use {connect, create_pool} from "zexus/db/postgres"

# Create pool
let pool = create_pool(
    action() { connect("postgresql://localhost/myapp") },
    {min: 2, max: 10}
)

# Use connection from pool
let conn = pool.acquire()
try {
    let result = conn.query("SELECT * FROM users")
    # ... process result
} finally {
    pool.release(conn)
}

# Cleanup
pool.close_all()
```

### Query Builder

```zexus
use {connect} from "zexus/db/postgres"
use {query_builder} from "zexus/db/common/query"

let db = connect("postgresql://localhost/myapp")

# Build query
let query = query_builder("users")
    .select("id", "name", "email")
    .where("age", ">", 18)
    .where("active", "=", true)
    .order_by("created_at", "DESC")
    .limit(10)

let sql = query.build()
let params = query.get_params()

let results = db.query(sql, params)
```

### MongoDB

```zexus
use {connect} from "zexus/db/mongodb"

# Connect
let client = connect("mongodb://localhost:27017/myapp")

# Get collection
let users = client.collection("users")

# Find documents
let active_users = users.find({active: true})

# Insert
users.insert_one({
    name: "Alice",
    email: "alice@example.com",
    age: 30,
    created_at: now()
})

# Update
users.update_one(
    {email: "alice@example.com"},
    {"$set": {age: 31}}
)

# Delete
users.delete_one({email: "alice@example.com"})
```

## Testing Strategy

### Unit Tests

```zexus
use {describe, it, expect} from "@zexus/test"
use {parse_connection_string} from "zexus/db/postgres"

describe("PostgreSQL Driver", action() {
    it("parses connection string", action() {
        let parts = parse_connection_string("postgresql://user:pass@localhost:5432/db")
        
        expect(parts.user).to_equal("user")
        expect(parts.password).to_equal("pass")
        expect(parts.host).to_equal("localhost")
        expect(parts.port).to_equal(5432)
        expect(parts.database).to_equal("db")
    })
})
```

### Integration Tests

```zexus
# Test with real database
use {connect} from "zexus/db/postgres"

let db = connect("postgresql://test:test@localhost:5432/testdb")

# Create test table
db.query("CREATE TABLE test_users (id SERIAL, name VARCHAR(100))")

# Insert
db.query("INSERT INTO test_users (name) VALUES ('Alice')")

# Query
let users = db.query("SELECT * FROM test_users")
expect(length(users)).to_equal(1)
expect(users[0].name).to_equal("Alice")

# Cleanup
db.query("DROP TABLE test_users")
db.close()
```

## Performance Requirements

### Target Benchmarks

- **Connection time**: < 50ms
- **Query latency**: < 5ms for simple queries
- **Throughput**: 1,000+ queries/second
- **Pool overhead**: < 1ms acquire/release

## Security Considerations

1. **SQL Injection Prevention**
   ```zexus
   # Use parameterized queries
   db.query("SELECT * FROM users WHERE id = $1", [user_id])
   ```

2. **Connection Security**
   ```zexus
   # SSL/TLS support
   let db = connect("postgresql://localhost/db?sslmode=require")
   ```

3. **Credential Management**
   ```zexus
   # Environment variables
   let db = connect(env_get("DATABASE_URL"))
   ```

## Migration to Native Keywords (Phase 2)

Eventually, database operations become native:

```zexus
# Phase 2: Native DATABASE keyword
database users {
    connection: "postgresql://localhost/myapp"
    
    query find_by_email(email: string) {
        SELECT * FROM users WHERE email = $email
    }
}

let user = users.find_by_email("alice@example.com")
```

## Development Timeline

### Month 1-2: PostgreSQL
- [ ] Wire protocol
- [ ] Connection management
- [ ] Query execution
- [ ] Prepared statements

### Month 3-4: MySQL
- [ ] Wire protocol
- [ ] Connection management
- [ ] Query execution
- [ ] Prepared statements

### Month 5: MongoDB
- [ ] BSON support
- [ ] Command execution
- [ ] CRUD operations

### Month 6: Polish
- [ ] Connection pooling
- [ ] Query builder
- [ ] Comprehensive tests
- [ ] Documentation

## Success Criteria

- [ ] All drivers pass official protocol tests
- [ ] Performance meets benchmarks
- [ ] 100% test coverage
- [ ] Complete documentation
- [ ] Production-ready

## Related Documentation

- [Ecosystem Strategy](../../ECOSYSTEM_STRATEGY.md)
- [DATABASE Keywords (Phase 2)](../DATABASE_KEYWORDS.md)
- [@zexus/db Package (Phase 3)](../../packages/ZEXUS_DB_PACKAGE.md)

---

**Status**: Planning
**Last Updated**: 2025-12-29
**Next Review**: Q1 2025
