# zexus_token.py (ENHANCED WITH LINE/COLUMN TRACKING)

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
MAP = "MAP"
WHILE = "WHILE"
USE = "USE"
EXACTLY = "EXACTLY"
EMBEDDED = "EMBEDDED"

class Token:
    def __init__(self, token_type, literal, line=None, column=None):
        self.type = token_type
        self.literal = literal
        self.line = line  # ✅ ADD line tracking
        self.column = column  # ✅ ADD column tracking

    def __repr__(self):
        if self.line and self.column:
            return f"Token({self.type}, '{self.literal}', line={self.line}, col={self.column})"
        return f"Token({self.type}, '{self.literal}')"