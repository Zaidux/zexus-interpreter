# lexer.py (ENHANCED WITH PHASE 1 KEYWORDS)
from .zexus_token import *

class Lexer:
    def __init__(self, source_code):
        self.input = source_code
        self.position = 0
        self.read_position = 0
        self.ch = ""
        self.in_embedded_block = False
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
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(EQ, literal)
                tok.line = current_line
                tok.column = current_column
            else:
                tok = Token(ASSIGN, self.ch)
                tok.line = current_line
                tok.column = current_column
        elif self.ch == '!':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(NOT_EQ, literal)
                tok.line = current_line
                tok.column = current_column
            else:
                tok = Token(BANG, self.ch)
                tok.line = current_line
                tok.column = current_column
        elif self.ch == '&':
            if self.peek_char() == '&':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(AND, literal)
                tok.line = current_line
                tok.column = current_column
            else:
                tok = Token(ILLEGAL, self.ch)
                tok.line = current_line
                tok.column = current_column
        elif self.ch == '|':
            if self.peek_char() == '|':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(OR, literal)
                tok.line = current_line
                tok.column = current_column
            else:
                tok = Token(ILLEGAL, self.ch)
                tok.line = current_line
                tok.column = current_column
        elif self.ch == '<':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(LTE, literal)
                tok.line = current_line
                tok.column = current_column
            else:
                tok = Token(LT, self.ch)
                tok.line = current_line
                tok.column = current_column
        elif self.ch == '>':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(GTE, literal)
                tok.line = current_line
                tok.column = current_column
            else:
                tok = Token(GT, self.ch)
                tok.line = current_line
                tok.column = current_column
        elif self.ch == '"':
            string_literal = self.read_string()
            tok = Token(STRING, string_literal)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '[':
            tok = Token(LBRACKET, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == ']':
            tok = Token(RBRACKET, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '(':
            tok = Token(LPAREN, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == ')':
            tok = Token(RPAREN, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '{':
            # Check if this might be start of embedded block
            lookback = self.input[max(0, self.position-10):self.position]
            if 'embedded' in lookback:
                self.in_embedded_block = True
            tok = Token(LBRACE, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '}':
            if self.in_embedded_block:
                self.in_embedded_block = False
            tok = Token(RBRACE, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == ',':
            tok = Token(COMMA, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == ';':
            tok = Token(SEMICOLON, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == ':':
            tok = Token(COLON, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '+':
            tok = Token(PLUS, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '-':
            tok = Token(MINUS, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '*':
            tok = Token(STAR, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '/':
            tok = Token(SLASH, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '%':
            tok = Token(MOD, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '.':
            tok = Token(DOT, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == "":
            tok = Token(EOF, "")
            tok.line = current_line
            tok.column = current_column
        else:
            if self.is_letter(self.ch):
                literal = self.read_identifier()

                if self.in_embedded_block:
                    token_type = IDENT
                else:
                    token_type = self.lookup_ident(literal)

                tok = Token(token_type, literal)
                tok.line = current_line
                tok.column = current_column
                return tok
            elif self.is_digit(self.ch):
                num_literal = self.read_number()
                if '.' in num_literal:
                    tok = Token(FLOAT, num_literal)
                else:
                    tok = Token(INT, num_literal)
                tok.line = current_line
                tok.column = current_column
                return tok
            else:
                if self.ch in ['\n', '\r']:
                    self.read_char()
                    return self.next_token()
                # For embedded code, treat unknown printable chars as IDENT
                if self.ch.isprintable():
                    literal = self.read_embedded_char()
                    tok = Token(IDENT, literal)
                    tok.line = current_line
                    tok.column = current_column
                    return tok
                tok = Token(ILLEGAL, self.ch)
                tok.line = current_line
                tok.column = current_column

        self.read_char()
        return tok

    def read_embedded_char(self):
        """Read a single character as identifier for embedded code compatibility"""
        char = self.ch
        self.read_char()
        return char

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
            "while": WHILE,
            "use": USE,
            "exactly": EXACTLY,
            "embedded": EMBEDDED,
            "export": EXPORT,
            "lambda": LAMBDA,
            "debug": DEBUG,      # NEW: Debug keyword
            "try": TRY,          # NEW: Try keyword  
            "catch": CATCH,      # NEW: Catch keyword
            "external": EXTERNAL, # NEW: External keyword
            "from": FROM,        # NEW: From keyword
        }
        return keywords.get(ident, IDENT)

    def is_letter(self, char):
        return 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char == '_'

    def is_digit(self, char):
        return '0' <= char <= '9'

    def skip_whitespace(self):
        while self.ch in [' ', '\t', '\n', '\r']:
            self.read_char()