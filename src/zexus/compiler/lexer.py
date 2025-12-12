# src/zexus/compiler/lexer.py

"""
Enhanced Lexer for Zexus Compiler Phase
"""

from ..zexus_token import *

class Lexer:
    def __init__(self, source_code):
        self.input = source_code
        self.position = 0
        self.read_position = 0
        self.ch = ""
        self.line = 1
        self.column = 1
        self.read_char()

    def read_char(self):
        if self.read_position >= len(self.input):
            self.ch = ""
        else:
            self.ch = self.input[self.read_position]

        # Update line and column tracking
        if self.ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        self.position = self.read_position
        self.read_position += 1

    def peek_char(self):
        if self.read_position >= len(self.input):
            return ""
        else:
            return self.input[self.read_position]

    def next_token(self):
        self.skip_whitespace()

        # Skip single line comments
        if self.ch == '#' and self.peek_char() != '{':
            self.skip_comment()
            return self.next_token()

        tok = None
        current_line = self.line
        current_column = self.column

        if self.ch == '=':
            # Support '==' and arrow '=>' for compiler lexer as in interpreter
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(EQ, literal, current_line, current_column)
            elif self.peek_char() == '>':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(LAMBDA, literal, current_line, current_column)
            else:
                tok = Token(ASSIGN, self.ch, current_line, current_column)
        elif self.ch == '!':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(NOT_EQ, literal, current_line, current_column)
            else:
                tok = Token(BANG, self.ch, current_line, current_column)
        elif self.ch == '&':
            if self.peek_char() == '&':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(AND, literal, current_line, current_column)
            else:
                tok = Token(ILLEGAL, self.ch, current_line, current_column)
        elif self.ch == '|':
            if self.peek_char() == '|':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(OR, literal, current_line, current_column)
            else:
                tok = Token(ILLEGAL, self.ch, current_line, current_column)
        elif self.ch == '<':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(LTE, literal, current_line, current_column)
            else:
                tok = Token(LT, self.ch, current_line, current_column)
        elif self.ch == '>':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(GTE, literal, current_line, current_column)
            else:
                tok = Token(GT, self.ch, current_line, current_column)
        elif self.ch == '"':
            string_literal = self.read_string()
            tok = Token(STRING, string_literal, current_line, current_column)
        elif self.ch == '[':
            tok = Token(LBRACKET, self.ch, current_line, current_column)
        elif self.ch == ']':
            tok = Token(RBRACKET, self.ch, current_line, current_column)
        elif self.ch == '(':
            tok = Token(LPAREN, self.ch, current_line, current_column)
        elif self.ch == ')':
            tok = Token(RPAREN, self.ch, current_line, current_column)
        elif self.ch == '{':
            tok = Token(LBRACE, self.ch, current_line, current_column)
        elif self.ch == '}':
            tok = Token(RBRACE, self.ch, current_line, current_column)
        elif self.ch == ',':
            tok = Token(COMMA, self.ch, current_line, current_column)
        elif self.ch == ';':
            tok = Token(SEMICOLON, self.ch, current_line, current_column)
        elif self.ch == ':':
            tok = Token(COLON, self.ch, current_line, current_column)
        elif self.ch == '+':
            tok = Token(PLUS, self.ch, current_line, current_column)
        elif self.ch == '-':
            tok = Token(MINUS, self.ch, current_line, current_column)
        elif self.ch == '*':
            tok = Token(STAR, self.ch, current_line, current_column)
        elif self.ch == '/':
            tok = Token(SLASH, self.ch, current_line, current_column)
        elif self.ch == '%':
            tok = Token(MOD, self.ch, current_line, current_column)
        elif self.ch == '.':
            tok = Token(DOT, self.ch, current_line, current_column)
        elif self.ch == "":
            tok = Token(EOF, "", current_line, current_column)
        else:
            if self.is_letter(self.ch):
                literal = self.read_identifier()
                token_type = self.lookup_ident(literal)
                tok = Token(token_type, literal, current_line, current_column)
                return tok
            elif self.is_digit(self.ch):
                num_literal = self.read_number()
                if '.' in num_literal:
                    tok = Token(FLOAT, num_literal, current_line, current_column)
                else:
                    tok = Token(INT, num_literal, current_line, current_column)
                return tok
            else:
                tok = Token(ILLEGAL, self.ch, current_line, current_column)

        self.read_char()
        return tok

    def skip_comment(self):
        while self.ch != '\n' and self.ch != "":
            self.read_char()
        self.skip_whitespace()

    def read_string(self):
        start_position = self.position + 1
        while True:
            self.read_char()
            if self.ch == '"' or self.ch == "":
                break
        return self.input[start_position:self.position]

    def read_identifier(self):
        start_position = self.position
        while self.is_letter(self.ch) or self.is_digit(self.ch):
            self.read_char()
        return self.input[start_position:self.position]

    def read_number(self):
        start_position = self.position
        is_float = False

        # Read integer part
        while self.is_digit(self.ch):
            self.read_char()

        # Check for decimal point
        if self.ch == '.':
            is_float = True
            self.read_char()
            # Read fractional part
            while self.is_digit(self.ch):
                self.read_char()

        number_str = self.input[start_position:self.position]
        return number_str

    def lookup_ident(self, ident):
        keywords = {
            "let": LET,
            "print": PRINT,
            "if": IF,
            "else": ELSE,
            "true": TRUE,
            "false": FALSE,
            "return": RETURN,
            "for": FOR,
            "each": EACH,
            "in": IN,
            "action": ACTION,
            "async": ASYNC,
            "await": AWAIT,
            "enum": ENUM,
            "protocol": PROTOCOL,
            "interface": INTERFACE,
            "capability": CAPABILITY,
            "grant": GRANT,
            "revoke": REVOKE,
            "module": MODULE,
            "package": PACKAGE,
            "using": USING,
            "type_alias": TYPE_ALIAS,
            "while": WHILE,
            "use": USE,
            "exactly": EXACTLY,
            "embedded": EMBEDDED,
            "export": EXPORT,
            "lambda": LAMBDA,
            "debug": DEBUG,
            "try": TRY,
            "catch": CATCH,
            "external": EXTERNAL,
            "from": FROM,
            # Blockchain & Smart Contract keywords
            "ledger": LEDGER,
            "state": STATE,
            "tx": TX,
            "revert": REVERT,
            "hash": HASH,
            "signature": SIGNATURE,
            "verify_sig": VERIFY_SIG,
            "limit": LIMIT,
            "gas": GAS,
            "require": REQUIRE,
        }
        return keywords.get(ident, IDENT)

    def is_letter(self, char):
        return 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char == '_'

    def is_digit(self, char):
        return '0' <= char <= '9'

    def skip_whitespace(self):
        while self.ch in [' ', '\t', '\n', '\r']:
            self.read_char()