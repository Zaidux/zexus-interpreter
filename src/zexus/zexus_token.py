# zexus_token.py (ENHANCED WITH PHASE 1 TOKENS)

# Special Tokens
ILLEGAL = "ILLEGAL"
EOF = "EOF"

# Identifiers + Literals
IDENT = "IDENT"
INT = "INT"
STRING = "STRING"
FLOAT = "FLOAT"

# Operators
ASSIGN = "="
PLUS = "+"
MINUS = "-"
SLASH = "/"
STAR = "*"
BANG = "!"
LT = "<"
GT = ">"
EQ = "=="
NOT_EQ = "!="
MOD = "%"
DOT = "."
LTE = "<="
GTE = ">="
APPEND = ">>"           # Append operator for LOG: log >> file
IMPORT_OP = "<<"        # Import operator: << file or let x << file
AND = "&&"
OR = "||"
QUESTION = "?"          # Ternary operator: condition ? true_val : false_val
NULLISH = "??"          # Nullish coalescing: value ?? default

# Delimiters
COMMA = ","
SEMICOLON = ";"
COLON = ":"
LPAREN = "("
RPAREN = ")"
LBRACE = "{"
RBRACE = "}"
LBRACKET = "["
RBRACKET = "]"
AT = "@"                # Decorator symbol: @logged, @cached, etc.

# Backwards-compatible alias: some parts of the codebase expect the name
# ASTERISK for the multiplication token. Provide a stable alias to avoid
# NameError when older modules reference ASTERISK.
ASTERISK = STAR

# Keywords
LET = "LET"
CONST = "CONST"                # NEW: Const immutable variables
DATA = "DATA"                  # NEW: Dataclass definitions
PRINT = "PRINT"
IF = "IF"
THEN = "THEN"                  # NEW: Then for if-then-else expressions
ELIF = "ELIF"                  # NEW: Elif else-if conditionals
ELSE = "ELSE"
RETURN = "RETURN"
TRUE = "TRUE"
FALSE = "FALSE"
NULL = "NULL"
FOR = "FOR"
EACH = "EACH"
IN = "IN"
ACTION = "ACTION"
FUNCTION = "FUNCTION"
SCREEN = "SCREEN"
COMPONENT = "COMPONENT"
THEME = "THEME"
COLOR = "COLOR"
GRAPHICS = "GRAPHICS"
CANVAS = "CANVAS"
ANIMATION = "ANIMATION"
CLOCK = "CLOCK"
MAP = "MAP"
WHILE = "WHILE"
USE = "USE"
EXACTLY = "EXACTLY"
EMBEDDED = "EMBEDDED"
EXPORT = "EXPORT"
LAMBDA = "LAMBDA"
DEBUG = "DEBUG"      # NEW: Debug token
TRY = "TRY"          # NEW: Try token
CATCH = "CATCH"      # NEW: Catch token
CONTINUE = "CONTINUE" # NEW: Continue on error token
BREAK = "BREAK"      # NEW: Break loop statement
THROW = "THROW"      # NEW: Throw error statement
EXTERNAL = "EXTERNAL" # NEW: From token
FROM = "FROM"        # NEW: From token

# ASYNC / AWAIT / MODULE / EVENT / ENUM / PROTOCOL tokens
ASYNC = "ASYNC"
AWAIT = "AWAIT"
EVENT = "EVENT"
EMIT = "EMIT"
ENUM = "ENUM"
PROTOCOL = "PROTOCOL"
IMPORT = "IMPORT"

# SECURITY & ADVANCED FEATURES
ENTITY = "ENTITY"              # Entity declaration: entity User { ... }
VERIFY = "VERIFY"              # Verify checks: verify(action_name, ...conditions)
CONTRACT = "CONTRACT"          # Smart contracts: contract Token { ... }
PROTECT = "PROTECT"            # Protection guardrails: protect(action, rules)
SEAL = "SEAL"                  # Seal objects: seal identifier
AUDIT = "AUDIT"                # Audit log: audit data_name, action_type, timestamp
RESTRICT = "RESTRICT"          # Field-level access control: restrict obj.field = "rule"
SANDBOX = "SANDBOX"            # Isolated execution environment: sandbox { code }
TRAIL = "TRAIL"                # Real-time audit/debug/print trail: trail audit_or_print_or_debug

# CAPABILITY-BASED SECURITY (NEW)
CAPABILITY = "CAPABILITY"      # Define capabilities: capability(name, description, scope)
GRANT = "GRANT"                # Grant capabilities: grant entity_name capability(name)
REVOKE = "REVOKE"              # Revoke capabilities: revoke entity_name capability(name)
IMMUTABLE = "IMMUTABLE"        # Enforce immutability: immutable identifier

# DATA VALIDATION & SANITIZATION (NEW)
VALIDATE = "VALIDATE"          # Validate data: validate(value, schema, options)
SANITIZE = "SANITIZE"          # Sanitize input: sanitize(value, rules, encoding)

# COMPLEXITY & LARGE PROJECT MANAGEMENT (NEW)
INTERFACE = "INTERFACE"        # Define formal interface: interface Drawable { ... }
TYPE_ALIAS = "TYPE_ALIAS"      # Type alias: type_alias UserID = integer;
MODULE = "MODULE"              # Module definition: module database { ... }
USING = "USING"                # Resource management: using(file) { ... }
PACKAGE = "PACKAGE"            # Package/namespace: package myapp.utils { ... }

# CONCURRENCY & PERFORMANCE (NEW)
CHANNEL = "CHANNEL"            # Channel definition: channel<integer> data;
SEND = "SEND"                  # Send to channel: send(channel, value);
RECEIVE = "RECEIVE"            # Receive from channel: receive(channel);
ATOMIC = "ATOMIC"              # Atomic operations: atomic(counter++);

# RENDERER OPERATIONS (ADD THESE)
MIX = "MIX"                    # Color mixing: mix("blue", "red", 0.5)
RENDER = "RENDER"              # Render screen: render_screen("login")
ADD_TO = "ADD_TO"              # Add component: add_to_screen("login", "button")
SET_THEME = "SET_THEME"        # Set theme: set_theme("dark")
CREATE_CANVAS = "CREATE_CANVAS" # Create canvas: create_canvas(80, 25)
DRAW = "DRAW"                  # Draw operation: draw_line(x1, y1, x2, y2)

# PROPERTY TOKENS (ADD THESE)
WIDTH = "WIDTH"
HEIGHT = "HEIGHT"
X = "X"
Y = "Y"
TEXT = "TEXT"
BACKGROUND = "BACKGROUND"
BORDER = "BORDER"
STYLE = "STYLE"
RADIUS = "RADIUS"
FILL = "FILL"

# ADVANCED FEATURE TOKENS
MIDDLEWARE = "MIDDLEWARE"
AUTH = "AUTH"
THROTTLE = "THROTTLE"
CACHE = "CACHE"
INJECT = "INJECT"  # Dependency injection: inject DatabaseAPI

# BLOCKCHAIN & SMART CONTRACT TOKENS
LEDGER = "LEDGER"              # Immutable state ledger: ledger balances;
STATE = "STATE"                # State management: state counter = 0;
TX = "TX"                      # Transaction context: TX.caller, TX.timestamp
REVERT = "REVERT"              # Revert transaction: revert("reason");
REQUIRE = "REQUIRE"            # Require condition: require(condition, "error");
HASH = "HASH"                  # Hash function: hash(data, "SHA256");
SIGNATURE = "SIGNATURE"        # Create signature: signature(data, private_key);
VERIFY_SIG = "VERIFY_SIG"      # Verify signature: verify_sig(data, sig, public_key);
LIMIT = "LIMIT"                # Gas/resource limit: action transfer() limit 1000 { ... };
GAS = "GAS"                    # Gas tracking: gas_used(), gas_remaining()
PERSISTENT = "PERSISTENT"
STORAGE = "STORAGE"
REQUIRE = "REQUIRE"

# PERFORMANCE OPTIMIZATION TOKENS
NATIVE = "NATIVE"            # Call C/C++ code: native { "func_name", arg1, arg2 }
GC = "GC"                    # Control garbage collection: gc "collect" or gc "pause"
INLINE = "INLINE"            # Function inlining: inline func_name
BUFFER = "BUFFER"            # Direct memory access: buffer(ptr, size)
SIMD = "SIMD"                # Vector operations: simd(operation, vector1, vector2)

# CONVENIENCE FEATURES TOKENS
DEFER = "DEFER"              # Cleanup code execution: defer cleanup_code;
PATTERN = "PATTERN"          # Pattern matching: pattern value { case x => ...; }
MATCH = "MATCH"              # Match expression: match value { Point(x, y) => ... }

# ADVANCED FEATURES TOKENS
ENUM = "ENUM"                # Type-safe enumerations: enum Color { Red, Green, Blue }
STREAM = "STREAM"            # Event streaming: stream name as event => handler;
WATCH = "WATCH"              # Reactive state management: watch variable => reaction;
LOG = "LOG"                  # Output logging: log > filename.txt

# Modifiers (single keyword to extend declarations)
PUBLIC = "PUBLIC"
PRIVATE = "PRIVATE"
SEALED = "SEALED"
ASYNC = "ASYNC"
NATIVE = "NATIVE"
INLINE = "INLINE"
SECURE = "SECURE"
PURE = "PURE"
VIEW = "VIEW"                  # View function (alias for pure, read-only)
PAYABLE = "PAYABLE"            # Payable function (can receive tokens)
MODIFIER = "MODIFIER"          # Function modifier (like onlyOwner)

# Contract & Protocol
IMPLEMENTS = "IMPLEMENTS"      # Protocol implementation: contract X implements Y
THIS = "THIS"                  # Reference to current contract instance

class Token:
    def __init__(self, token_type, literal, line=None, column=None):
        self.type = token_type
        self.literal = literal
        self.line = line  # ✅ ADD line tracking
        self.column = column  # ✅ ADD column tracking
        
        # For backward compatibility with code expecting dict-like tokens
        self.value = literal  # Alias for literal

    def __repr__(self):
        if self.line and self.column:
            return f"Token({self.type}, '{self.literal}', line={self.line}, col={self.column})"
        return f"Token({self.type}, '{self.literal}')"
    
    def get(self, key, default=None):
        """Dict-like get method for backward compatibility"""
        if hasattr(self, key):
            return getattr(self, key)
        return default
    
    def __getitem__(self, key):
        """Allow dict-like access for compatibility"""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Token has no attribute '{key}'")
    
    def __contains__(self, key):
        """Check if token has attribute"""
        return hasattr(self, key)
    
    def to_dict(self):
        """Convert token to dictionary for compatibility"""
        return {
            'type': self.type,
            'literal': self.literal,
            'value': self.literal,
            'line': self.line,
            'column': self.column
        }