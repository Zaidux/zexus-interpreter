# DATABASE Native Keywords - Phase 2

**Status**: Future (After Phase 1)
**Phase**: 2 - Integrate INTO Zexus
**Priority**: High
**Dependencies**: Database Drivers (Phase 1)

## Overview

Make database operations **first-class language features** with native keywords for:
- Database connections
- Query definitions
- Migrations
- Transactions
- Type-safe queries

## Why Native?

1. **Data-intensive apps** - Most apps need databases
2. **Type safety** - Compile-time query validation
3. **Performance** - Language-level optimization
4. **Simplicity** - Eliminate ORM boilerplate

## DATABASE Keyword

### Basic Syntax

```zexus
database users {
    connection: "postgresql://localhost/myapp"
    
    query find_all() {
        SELECT * FROM users
    }
    
    query find_by_email(email: string) {
        SELECT * FROM users WHERE email = $email
    }
    
    query create(name: string, email: string) {
        INSERT INTO users (name, email, created_at)
        VALUES ($name, $email, NOW())
        RETURNING *
    }
}

# Usage
let all_users = users.find_all()
let user = users.find_by_email("alice@example.com")
let new_user = users.create("Bob", "bob@example.com")
```

### Full Syntax

```zexus
database myapp {
    # Connection configuration
    connection: env_get("DATABASE_URL")
    
    # Connection pool
    pool {
        min: 2
        max: 10
        idle_timeout: 60
    }
    
    # Migrations
    migration "001_create_users" {
        up {
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        }
        
        down {
            DROP TABLE users
        }
    }
    
    # Queries
    query find_users(min_age: integer = 0) {
        SELECT * FROM users
        WHERE age >= $min_age
        ORDER BY created_at DESC
    }
    
    query create_user(name: string, email: string) {
        INSERT INTO users (name, email)
        VALUES ($name, $email)
        RETURNING *
    }
    
    query update_user(id: integer, name: string) {
        UPDATE users
        SET name = $name
        WHERE id = $id
        RETURNING *
    }
    
    query delete_user(id: integer) {
        DELETE FROM users WHERE id = $id
    }
    
    # Transactions
    transaction transfer_funds(from: integer, to: integer, amount: integer) {
        UPDATE accounts SET balance = balance - $amount WHERE id = $from;
        UPDATE accounts SET balance = balance + $amount WHERE id = $to;
    }
}

# Usage
let users = myapp.find_users(18)
let user = myapp.create_user("Alice", "alice@example.com")
myapp.transfer_funds(1, 2, 100)
```

## Model Definitions

```zexus
# Define models with native keyword
model User {
    table: "users"
    
    # Fields
    id: integer primary_key auto_increment
    name: string not_null max_length(100)
    email: string unique not_null
    age: integer default(0)
    created_at: datetime default(NOW())
    
    # Validations
    validate {
        is_email(this.email)
        this.age >= 0 and this.age <= 150
    }
    
    # Relationships
    has_many posts {
        foreign_key: "user_id"
    }
    
    belongs_to organization {
        foreign_key: "org_id"
    }
}

# Usage
let user = User.create({
    name: "Alice",
    email: "alice@example.com",
    age: 30
})

let users = User.where({age: {gt: 18}}).order_by("created_at", "DESC").limit(10)
let user = User.find(42)
user.name = "Alice Smith"
user.save()
user.delete()
```

## Query Builder

```zexus
# Fluent query building
let query = database.users
    .select("id", "name", "email")
    .where("age", ">", 18)
    .where("active", "=", true)
    .order_by("created_at", "DESC")
    .limit(10)
    .offset(0)

let results = query.execute()
```

## Transactions

```zexus
# Automatic transaction management
database myapp {
    transaction create_user_and_profile(user_data: map, profile_data: map) {
        # Both queries run in same transaction
        let user = INSERT INTO users (...) VALUES (...) RETURNING *;
        let profile = INSERT INTO profiles (user_id, ...) VALUES ($user.id, ...) RETURNING *;
        
        # If any fails, both rollback
        return {user: user, profile: profile}
    }
}

# Manual transactions
database.begin()
try {
    database.execute("INSERT INTO users ...")
    database.execute("INSERT INTO profiles ...")
    database.commit()
} catch (error) {
    database.rollback()
    throw error
}
```

## Migrations

```zexus
database myapp {
    migration "001_create_users" {
        up {
            CREATE TABLE users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        }
        
        down {
            DROP TABLE users
        }
    }
    
    migration "002_add_age_to_users" {
        up {
            ALTER TABLE users ADD COLUMN age INTEGER DEFAULT 0
        }
        
        down {
            ALTER TABLE users DROP COLUMN age
        }
    }
}

# Run migrations
database.migrate()

# Rollback
database.rollback()

# Check status
let status = database.migration_status()
```

## Type Safety

Native DATABASE keyword provides compile-time checking:

```zexus
database users {
    query find_by_id(id: integer) {
        SELECT * FROM users WHERE id = $id
    }
}

# Compile error: wrong type
users.find_by_id("not a number")  # ERROR: Expected integer, got string

# Compile error: wrong parameter count
users.find_by_id()  # ERROR: Missing required parameter 'id'

# Compile error: unknown query
users.find_by_name("Alice")  # ERROR: No query 'find_by_name' defined
```

**Note**: Type checking is performed on the Zexus function signature (parameter types), not on the SQL query itself. The compiler validates that calls to `find_by_id` receive an integer parameter. SQL syntax validation and result type checking are future enhancements that may leverage query analysis tools.

## Multiple Databases

```zexus
# Define multiple databases
database primary {
    connection: "postgresql://localhost/main"
}

database analytics {
    connection: "postgresql://localhost/analytics"
}

database cache {
    connection: "redis://localhost:6379"
}

# Use different databases
let users = primary.find_users()
let stats = analytics.get_stats()
cache.set("key", "value")
```

## NoSQL Support

```zexus
# MongoDB
database mongo {
    connection: "mongodb://localhost:27017/myapp"
    type: "mongodb"
    
    collection users {
        find(query: map) {
            # MongoDB query
        }
        
        insert_one(doc: map) {
            # Insert document
        }
        
        update_one(filter: map, update: map) {
            # Update document
        }
    }
}

# Usage
let users = mongo.users.find({age: {$gt: 18}})
mongo.users.insert_one({name: "Alice", age: 30})
```

## Complete Example

```zexus
# Define database with models and queries
database blog {
    connection: env_get("DATABASE_URL")
    
    # Users model
    model User {
        id: integer primary_key
        username: string unique not_null
        email: string unique not_null
        created_at: datetime
        
        has_many posts
        has_many comments
        
        validate {
            is_email(this.email)
            length(this.username) >= 3
        }
    }
    
    # Posts model
    model Post {
        id: integer primary_key
        title: string not_null
        content: text
        user_id: integer not_null
        created_at: datetime
        
        belongs_to user
        has_many comments
    }
    
    # Comments model
    model Comment {
        id: integer primary_key
        content: text not_null
        post_id: integer not_null
        user_id: integer not_null
        created_at: datetime
        
        belongs_to post
        belongs_to user
    }
    
    # Custom queries
    query recent_posts(limit: integer = 10) {
        SELECT p.*, u.username
        FROM posts p
        JOIN users u ON p.user_id = u.id
        ORDER BY p.created_at DESC
        LIMIT $limit
    }
    
    query popular_users(min_posts: integer = 5) {
        SELECT u.*, COUNT(p.id) as post_count
        FROM users u
        JOIN posts p ON u.id = p.user_id
        GROUP BY u.id
        HAVING COUNT(p.id) >= $min_posts
        ORDER BY post_count DESC
    }
}

# Usage
let user = blog.User.create({
    username: "alice",
    email: "alice@example.com"
})

let post = blog.Post.create({
    title: "My First Post",
    content: "Hello, World!",
    user_id: user.id
})

let recent = blog.recent_posts(5)
let popular = blog.popular_users(10)
```

## Performance Benefits

1. **Connection pooling** at language level
2. **Query caching** and **prepared statements**
3. **Compile-time optimization**
4. **Async I/O** integration
5. **Type-aware** query planning

## Migration from Phase 1

Phase 1 code:

```zexus
use {connect} from "zexus/db/postgres"

let db = connect("postgresql://localhost/myapp")
let users = db.query("SELECT * FROM users WHERE email = $1", ["alice@example.com"])
```

Becomes Phase 2 native:

```zexus
database myapp {
    connection: "postgresql://localhost/myapp"
    
    query find_by_email(email: string) {
        SELECT * FROM users WHERE email = $email
    }
}

let users = myapp.find_by_email("alice@example.com")
```

## Related Documentation

- [Database Drivers (Phase 1)](./features/DATABASE_DRIVERS.md)
- [@zexus/db Package (Phase 3)](../packages/ZEXUS_DB_PACKAGE.md)
- [Ecosystem Strategy](../ECOSYSTEM_STRATEGY.md)

---

**Status**: Planned (After Phase 1)
**Last Updated**: 2025-12-29
**Next Review**: Q3 2025
