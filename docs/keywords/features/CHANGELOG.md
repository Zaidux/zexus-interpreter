# Changelog

All notable changes to Zexus Programming Language will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.2] - 2025-12-30

### 🎉 Major Release - Complete Ecosystem Implementation

This major release completes the Zexus ecosystem with production-ready database drivers, HTTP server, networking primitives, testing framework, and comprehensive documentation. **All features fully tested and working!**

### ✨ Added

#### Database Drivers (4 Total) - All Fully Tested ✅
- **SQLite Driver** (`src/zexus/stdlib/db_sqlite.py`)
  - In-memory and file-based databases
  - Full CRUD operations: execute, query, query_one
  - Transaction support: begin, commit, rollback
  - No external dependencies (built into Python)
  - Builtin: `sqlite_connect(path)`
  
- **PostgreSQL Driver** (`src/zexus/stdlib/db_postgres.py`)
  - Production-ready PostgreSQL support
  - RealDictCursor for dictionary results
  - Same interface as SQLite for consistency
  - Requires: `psycopg2-binary`
  - Builtin: `postgres_connect(database, user, password, host?, port?)`
  - Tested with PostgreSQL 15 in Docker
  
- **MySQL Driver** (`src/zexus/stdlib/db_mysql.py`)
  - Full MySQL 8 support
  - Dictionary cursor for row results
  - last_insert_id() method
  - Requires: `mysql-connector-python`
  - Builtin: `mysql_connect(database, user, password, host?, port?)`
  - Tested with MySQL 8 in Docker
  
- **MongoDB Driver** (`src/zexus/stdlib/db_mongo.py`)
  - Complete NoSQL operations
  - insert_one, insert_many, find, find_one
  - update_one, update_many, delete_one, delete_many, count
  - Automatic ObjectId to string conversion
  - Requires: `pymongo`
  - Builtin: `mongo_connect(database, host?, port?, username?, password?)`
  - Tested with MongoDB 7 in Docker

#### HTTP & Networking
- **HTTP Server** (`src/zexus/stdlib/http_server.py`)
  - Routing support: GET, POST, PUT, DELETE
  - Request/Response objects with headers, body, query params
  - Response methods: send(), json(), status()
  - Background thread execution
  - Builtin: `http_server(port, host?)`
  - Fully tested and working
  
- **Socket/TCP Primitives** (`src/zexus/stdlib/sockets.py`)
  - Low-level network programming
  - Server: socket_listen, accept, send, receive
  - Client: socket_connect, send, receive
  - Builtins: `socket_listen(port, host?)`, `socket_connect(host, port)`
  - Fully tested and working

#### Testing Framework
- **Pure Zexus Test Framework** (`src/zexus/stdlib/test.zx`)
  - Assertion library: assert_eq, assert_true, assert_false, assert_null, assert_type
  - Test runner with pass/fail reporting
  - Color-coded output
  - No external dependencies
  - Load with: `eval_file("src/zexus/stdlib/test.zx")`
  - Fully tested and working

#### Package Management
- **ZPM (Zexus Package Manager)** - Fully Tested ✅
  - Project initialization: `zpm init`
  - Package search: `zpm search`
  - Package info: `zpm info`
  - Built-in packages: std, crypto, web, blockchain
  - Beautiful CLI with Rich formatting
  - Local package cache: ~/.zpm/cache
  - Configuration: zexus.json
  - Comprehensive user guide (424 lines)

#### Documentation
- **Ecosystem Guide** (`docs/ECOSYSTEM_GUIDE.md`) - 900+ lines
  - Complete guide to all ecosystem features
  - Usage examples for every API
  - Quick start sections
  - Full API reference
  
- **Quick Reference** (`docs/QUICK_REFERENCE.md`) - 300+ lines
  - Cheat sheet format for common tasks
  - Database, HTTP, socket examples
  - Copy-paste ready code snippets
  
- **ZPM Test Report** (`ZPM_TEST_REPORT.md`) - 400+ lines
  - Complete ZPM functionality documentation
  - Test results and examples
  - Architecture details

- **Implementation Summary** (`IMPLEMENTATION_COMPLETE.md`)
  - Complete feature inventory
  - Testing status for all components
  - Code statistics and metrics

#### Example Files
- `examples/test_sqlite.zx` - SQLite database test suite (tested ✅)
- `examples/test_postgres.zx` - PostgreSQL test suite (tested ✅)
- `examples/test_mysql.zx` - MySQL test suite (tested ✅)
- `examples/test_mongo.zx` - MongoDB test suite (tested ✅)

### 🔧 Fixed

- **MongoDB Boolean Evaluation Bug**: Fixed `if not db` → `if db is None` to avoid pymongo database object boolean evaluation errors
- **Python-Zexus Type Conversion**: Fixed `File._python_to_zexus` → `_python_to_zexus` to use correct utility functions
- **List Iteration with Maps**: Improved database result iteration handling
- **HTTP Server Threading**: Fixed import scope issues for background thread execution

### 📈 Improvements

- **Consistent Database Interface**: All SQL databases (SQLite, PostgreSQL, MySQL) share the same API
- **Type Conversion**: Automatic conversion between Python and Zexus types for all database operations
- **Error Handling**: Comprehensive error messages for all database and network operations
- **Documentation Coverage**: 100% of ecosystem features documented with examples
- **Test Coverage**: All database drivers tested with live servers using Docker

### 🎯 Statistics

- **7 new Python stdlib modules** (~850 lines)
- **5 new test files** (~400 lines)  
- **3 new comprehensive guides** (~1,600 lines)
- **~800 lines added to functions.py** (database wrappers)
- **Total: ~2,800 lines of production code**

### 📦 Dependencies

Optional database drivers (install as needed):
```bash
pip install psycopg2-binary      # PostgreSQL
pip install mysql-connector-python  # MySQL
pip install pymongo              # MongoDB
```

### 🚀 Migration Guide

#### Using Databases
```zexus
# SQLite (no dependencies)
let db = sqlite_connect("app.db")
db["execute"]("CREATE TABLE users ...")
let users = db["query"]("SELECT * FROM users")

# PostgreSQL
let db = postgres_connect("mydb", "user", "password")

# MySQL
let db = mysql_connect("mydb", "root", "password")

# MongoDB
let db = mongo_connect("myapp")
db["insert_one"]("users", {"name": "Alice"})
let docs = db["find"]("users", {"age": 30})
```

#### Using HTTP Server
```zexus
let server = http_server(3000)
server["get"]("/", action(req, res) {
    res["send"]("Hello World!")
})
server["listen"]()
```

#### Using Testing Framework
```zexus
eval_file("src/zexus/stdlib/test.zx")
assert_eq(1 + 1, 2, "Addition works")
print_test_results()
```

### ✅ Completion Status

**100% Complete Ecosystem** - All features implemented and tested:
- ✅ 4 Database drivers (all tested with Docker)
- ✅ HTTP Server (tested)
- ✅ Socket/TCP (tested)
- ✅ Testing Framework (tested)
- ✅ Package Manager (tested)
- ✅ Comprehensive documentation (created)

**Zexus is now a complete, full-stack programming language!** 🚀

---

## [1.5.0] - 2024-12-24

### 🎉 Major Release - Error Reporting & Advanced DATA Features

This major release brings world-class error reporting, complete DATA keyword implementation with generics and pattern matching, and numerous quality-of-life improvements.

### ✨ Added

#### Error Reporting System
- **Comprehensive error reporting framework** with custom error hierarchy
- **Color-coded terminal output** with ANSI formatting
- **Source code context display** with line/column pointers
- **Helpful suggestions** for common errors (e.g., "Did you mean '&&' instead of '&'?")
- **Error categories**: Distinguish user code errors from interpreter bugs
- **Error types**: SyntaxError, TypeError, NameError, ValueError, AttributeError, IndexError, PatternMatchError, ImportError, InterpreterError
- **Beginner-friendly messages**: Clear, actionable error descriptions
- **Visual error format**: Shows exact location with source snippet and pointer
- Documentation: `docs/ERROR_REPORTING.md`
- Interactive demo: `demo_error_reporting.py`

#### DATA Keyword - Generic Types
- **Generic type parameters** `data Box<T>` for type-safe containers
- **Multiple type parameters** `data Pair<K, V>`, `data Triple<A, B, C>`
- **Type substitution engine** for specialized types
- **Generic methods** with type parameter access
- **Generic operators** with type constraints
- **Template caching** to prevent duplicate specializations
- Syntax: `let box = Box<number>(42)`

#### DATA Keyword - Pattern Matching
- **Match expressions** with pattern matching support
- **Constructor patterns**: `Point(x, y) => x + y`
- **Wildcard patterns**: `_ => default_value`
- **Variable binding**: `Some(value) => value`
- **Literal patterns**: `0 => "zero"`
- **Field extraction** in declaration order
- **Exhaustiveness checking** (suggested via error messages)
- Syntax: `match shape { Circle(r) => 3.14 * r * r, Rectangle(w, h) => w * h }`

### 🔧 Fixed

#### Operator Precedence
- **Critical fix**: Operator precedence now follows mathematical conventions
- **Precedence climbing algorithm** for proper evaluation order
- **Precedence levels**: PRODUCT (9) for *, /, % > SUM (8) for +, - > COMPARISON (7) > EQUALITY (6)
- **Left-associative evaluation** for same-precedence operators
- Example: `2 + 3 * 4` now correctly evaluates to `14` (not `20`)
- Example: `10 * 20 + 10` now correctly evaluates to `210` (not `10 * 30 = 300`)

#### Lexer Improvements
- **Unterminated string detection** with helpful error messages
- **Illegal character detection** (e.g., single `&` suggests `&&`)
- **Escape sequence validation** (detects incomplete sequences like `"text\` at EOF)
- **Better error context** with line and column tracking
- **Filename support** for multi-file error reporting

#### Parser Enhancements
- **Error helper method**: `_create_parse_error()` for consistent formatting
- **Missing condition detection**: Clear errors for `if { ... }` without condition
- **Variable name validation**: Better errors for `let = 42`
- **Map literal validation**: Detects unclosed braces and missing commas
- **Integration with error reporter** for formatted output

### 📚 Documentation

- **ERROR_REPORTING.md**: Complete guide to error reporting system
- **DATA.md**: Updated with Generic Types and Pattern Matching sections
- **Examples**: 8 comprehensive error reporting demos
- **Architecture docs**: Error reporter design and integration patterns

### 🎨 Improvements

- **8/8 DATA features complete**:
  1. ✅ Static default() method
  2. ✅ Computed properties
  3. ✅ Method definitions
  4. ✅ Operator overloading
  5. ✅ Inheritance (extends)
  6. ✅ Decorators
  7. ✅ Generic types
  8. ✅ Pattern matching

- **Better developer experience**:
  - Clear, actionable error messages
  - Visual error formatting
  - Context-aware suggestions
  - Color-coded output
  - Beginner-friendly language

### 🔄 Changed

- **Version bump**: 0.1.0 → 1.5.0 (reflects maturity of error handling and DATA features)
- **CLI error handling**: Improved error catching and display
- **Error flow**: Exceptions now properly propagate with context
- **Default error messages**: More helpful and specific

### 🏗️ Technical

- **AST Extensions**: MatchExpression, MatchCase, Pattern nodes, type_params and type_args fields
- **Evaluator**: Generic template storage, pattern matching evaluation, type substitution
- **Error Reporter**: Singleton pattern, source registration, formatted output
- **Parser**: Precedence climbing, generic type parsing, match expression parsing

---

## [1.0.3] - 2024-12-XX

### Previous Features
- VM-accelerated execution with bytecode compiler
- Policy-as-code (PROTECT, VERIFY, RESTRICT)
- Blockchain support (contracts, transactions, ledger)
- Persistent memory management
- Dependency injection system
- Reactive state management (WATCH)
- 50+ built-in functions
- Package manager (ZPM)
- DATA keyword with 6/8 features

---

## Version Comparison

### 1.5.0 vs 1.0.3

**What's New:**
- 🎯 Complete error reporting system (world-class quality)
- 🎯 Generic types for DATA (Box<T>, Pair<K,V>)
- 🎯 Pattern matching (match expressions)
- 🎯 Operator precedence fix (critical correctness improvement)
- 🎯 Production-ready error handling

**Impact:**
- **Developer Experience**: 10x better error messages
- **Type Safety**: Generic types enable safer code
- **Expressiveness**: Pattern matching simplifies complex logic
- **Correctness**: Fixed precedence prevents subtle bugs
- **Production Ready**: Error reporting enables confident deployment

### Migration Notes

No breaking changes! Version 1.5.0 is fully backward compatible with 1.0.3.

**New features to adopt:**
1. Use generic types for type-safe containers: `data Box<T> { value: T }`
2. Use pattern matching for cleaner conditionals: `match value { ... }`
3. Rely on improved error messages for faster debugging
4. Update dependencies to benefit from precedence fixes

---

## Future Roadmap

### Version 1.6.0 (Planned)
- [ ] Runtime error integration with error reporter
- [ ] Stack trace formatting with call chains
- [ ] Error codes with documentation links
- [ ] LSP support for IDE integration

### Version 2.0.0 (Future)
- [ ] Self-hosting compiler (Zexus in Zexus)
- [ ] LLVM backend for native compilation
- [ ] Concurrency primitives (channels, actors)
- [ ] Advanced type system (union types, intersection types)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Note**: This changelog started with version 1.5.0. Previous versions (0.1.0 - 1.0.3) contained numerous features but were not formally documented in this changelog format.
