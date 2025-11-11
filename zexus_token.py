# zexus_token.py

# Core token types
ILLEGAL = "ILLEGAL"
EOF = "EOF"
SEMICOLON = ";"
COMMA = ","
DOT = "."
COLON = ":"

# Identifiers + literals
IDENT = "IDENT"
INT = "INT"
FLOAT = "FLOAT"
STRING = "STRING"
BOOL = "BOOL"
NULL = "NULL"

# Operators
ASSIGN = "="
PLUS = "+"
MINUS = "-"
BANG = "!"
STAR = "*"
SLASH = "/"
LT = "<"
GT = ">"
EQ = "=="
NOT_EQ = "!="

# Delimiters
LPAREN = "("
RPAREN = ")"
LBRACE = "{"
RBRACE = "}"
LBRACKET = "["
RBRACKET = "]"

# Keywords
FUNCTION = "FUNCTION"
ACTION = "action"
LET = "let"
TRUE = "true"
FALSE = "false"
IF = "if"
ELSE = "else"
RETURN = "return"
PRINT = "print"
FOR = "for"
EACH = "each"
IN = "in"
SCREEN = "screen"

# Module system keywords
USE = "use"
FROM = "from"
AS = "as"
EXPORT = "export"

class Token:
    def __init__(self, type_, literal):
        self.type = type_
        self.literal = literal

keywords = {
    "action": ACTION,
    "let": LET,
    "true": TRUE,
    "false": FALSE,
    "if": IF,
    "else": ELSE,
    "return": RETURN,
    "print": PRINT,
    "for": FOR,
    "each": EACH,
    "in": IN,
    "screen": SCREEN,
    "use": USE,
    "from": FROM,
    "as": AS,
    "export": EXPORT,
}