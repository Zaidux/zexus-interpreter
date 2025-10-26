# zexus_token.py (COMPLETE FIXED VERSION)

# This file defines all the possible token types in the Zexus language.

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
LT = "<"      # ✅ ADDED
GT = ">"      # ✅ ADDED
EQ = "=="
NOT_EQ = "!="
MOD = "%"      # Modulo operator
DOT = "."      # Dot operator for method calls
LTE = "<="      # Less than or equal
GTE = ">="      # Greater than or equal

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
EXACTLY = "EXACTLY"  # ✅ ADDED
EMBEDDED = "EMBEDDED"  # ADD THIS LINE

# The Token class represents a single token, holding its type and value.
class Token:
    def __init__(self, token_type, literal):
        self.type = token_type
        self.literal = literal

    def __repr__(self):
        return f"Token({self.type}, '{self.literal}')"