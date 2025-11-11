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
AND = "&&"
OR = "||"

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

# Backwards-compatible alias: some parts of the codebase expect the name
# ASTERISK for the multiplication token. Provide a stable alias to avoid
# NameError when older modules reference ASTERISK.
ASTERISK = STAR

# Keywords
LET = "LET"
PRINT = "PRINT"
IF = "IF"
ELSE = "ELSE"
RETURN = "RETURN"
TRUE = "TRUE"
FALSE = "FALSE"
FOR = "FOR"
EACH = "EACH"
IN = "IN"
ACTION = "ACTION"
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
PERSISTENT = "PERSISTENT"
REQUIRE = "REQUIRE"

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