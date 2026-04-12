# Zexus Security Features

Zexus provides defense-in-depth security built into the language runtime. This document covers all security subsystems.

## Table of Contents
1. [Capability-Based Security](#capability-based-security)
2. [Mandatory Input Sanitization](#mandatory-input-sanitization)
3. [Resource Limits](#resource-limits)
4. [Integer Overflow Protection](#integer-overflow-protection)
5. [Contract Access Control](#contract-access-control)
6. [Virtual Filesystem Sandboxing](#virtual-filesystem-sandboxing)
7. [Taint Tracking](#taint-tracking)
8. [Audit Logging](#audit-logging)
9. [New in v1.8.4: Security Modules](#new-security-modules-v184)

---

## Capability-Based Security

Zexus uses a capability-based security model where code must be explicitly granted permissions to access system resources.

```zexus
// Request capabilities
@capabilities(["fs.read", "net.http"])

// Code without fs.write capability cannot write to disk
```

Capabilities are declared at module level and enforced at runtime. Attempting to perform an operation without the required capability raises a `SecurityError`.

---

## Mandatory Input Sanitization

User input is **taint-tracked** and must pass through the runtime's contextual sanitizers before reaching sensitive sinks. When you use context-aware helpers (SQL builders, templating, shell-safe args), sanitization is applied automatically; raw string concatenation still requires explicit sanitization.

### Supported Contexts
| Context | Sanitization Applied |
|---------|---------------------|
| SQL | Parameterized queries, quote escaping |
| HTML | Entity encoding (`<`, `>`, `&`, `"`, `'`) |
| URL | Percent-encoding of special characters |
| Shell | Argument escaping, metacharacter removal |
| Path | Directory traversal prevention (`../`) |

### How It Works
```zexus
let user_input = input("Enter name: ")

// Automatically sanitized when routed through context-aware helpers
let query = sql("SELECT * FROM users WHERE name = ${user_input}")

// Automatically sanitized for HTML context
let html = "<div>${user_input}</div>"
```

### Taint Tracking Integration
All values from external sources (user input, HTTP requests, file reads) are automatically marked as **tainted**. Tainted values must pass through sanitization before reaching sensitive sinks.

---

## Resource Limits

Automatic resource limits prevent denial-of-service attacks and runaway programs.

### Default Limits
| Resource | Default Limit | Environment Variable |
|----------|---------------|---------------------|
| Loop iterations | 1,000,000 | `ZEXUS_MAX_ITERATIONS` |
| Call stack depth | 1,000 | `ZEXUS_MAX_CALL_DEPTH` |
| Execution timeout | 30 seconds | `ZEXUS_EXEC_TIMEOUT` |
| VM stack depth | 50,000 | `ZEXUS_MAX_STACK_DEPTH` |

### Configuration
```zexus
// In code
@resource_limits({
    max_iterations: 500000,
    max_call_depth: 500,
    timeout: 10
})

// Or via environment
// ZEXUS_MAX_ITERATIONS=500000
```

### Behavior on Limit Reached
- **Loops**: `ResourceLimitError` raised with message indicating iteration count
- **Call depth**: `StackOverflowError` raised
- **Timeout**: `TimeoutError` raised, execution halted

---

## Integer Overflow Protection

All arithmetic operations are checked for overflow by default.

```zexus
let x = 2147483647   // Max 32-bit int
let y = x + 1        // Raises OverflowError (in strict mode)
                      // Or: silently wraps / uses BigInt (configurable)
```

### Modes
| Mode | Behavior |
|------|----------|
| `strict` (default) | Raises `OverflowError` on overflow |
| `wrap` | Wraps around (C-style) |
| `bigint` | Promotes to arbitrary-precision integer |

### Safe Math Functions
```zexus
use "math"
math.safe_add(a, b)    // Returns error instead of overflowing
math.safe_mul(a, b)
math.checked_div(a, b) // Checks for division by zero
```

---

## Contract Access Control

Smart contracts in Zexus enforce access control on state-modifying operations.

```zexus
contract Token {
    state owner = TX.caller
    state balances = {}

    @only_owner
    action mint(to, amount) {
        require_owner()
        state.balances[to] = (state.balances[to] or 0) + amount
    }

    action transfer(to, amount) {
        require(state.balances[TX.caller] >= amount, "Insufficient balance")
        state.balances[TX.caller] = state.balances[TX.caller] - amount
        state.balances[to] = (state.balances[to] or 0) + amount
    }
}
```

### Access Control Patterns
- `require_owner()` — Only contract deployer can call
- `has_role(role)` — Role-based access control
- `has_permission(perm)` — Fine-grained permission checks
- `require(condition, message)` — Custom assertions

---

## Virtual Filesystem Sandboxing

File system operations are sandboxed to prevent unauthorized access.

```zexus
@capabilities(["fs.read"])

// Reads are allowed within the sandbox root
let data = fs.read("data.txt")

// Attempting to escape the sandbox raises SecurityError
let secret = fs.read("/etc/passwd")  // SecurityError!
let escape = fs.read("../../etc/passwd")  // SecurityError! (path traversal blocked)
```

---

## Taint Tracking

Values from external sources are automatically tagged as tainted and tracked through the program.

```zexus
let input = http.get_param("user")  // Tainted
let clean = sanitize(input)          // Untainted after sanitization
let query = sql("SELECT * FROM users WHERE id = ${input}")  // Error: tainted value in SQL sink
```

---

## Audit Logging

Comprehensive audit logging for compliance (GDPR, HIPAA, SOC2).
Runtime audit logging is built into the language runtime and is separate from the `use "audit"` SAST module described later.

```zexus
audit.log("user_login", { user: "admin", ip: "10.0.0.1" })
audit.log("data_access", { table: "patients", columns: ["name", "ssn"] })
```

Audit events include timestamps, caller identity, operation type, and resource accessed.

---

## New Security Modules (v1.8.4)

### `use "fuzz"` — Fuzzing
Coverage-guided fuzzing with mutation strategies, contract fuzzer, and corpus management.

### `use "secrets"` — Secrets Management
Encrypted secret store with versioned rotation, envelope encryption, and audit log.

### `use "netsec"` — Network Security Tools
Port scanner, TLS certificate checker, DNS enumeration, HTTP security header analysis.

### `use "payloads"` — Exploit Payload Libraries
XSS, SQLi, SSRF, path traversal, command injection, XXE, and SSTI test vectors.

### `use "pentest"` — Penetration Testing Framework
Service enumeration, web fingerprinting, credential testing, structured reporting.

### `use "audit"` — SAST Scanner
Static analysis with 8 detection rules and SARIF output.

### `use "contract_audit"` — Contract Security Analyzer
Reentrancy, integer overflow, missing access control, unbounded loops, timestamp dependence detection.
