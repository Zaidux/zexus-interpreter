# lexer.py (UPDATED VERSION)
from zexus_token import *

class Lexer:
    def __init__(self, source_code):
        self.input = source_code
        self.position = 0
        self.read_position = 0
        self.ch = ""
        self.read_char()

    def read_char(self):
        if self.read_position >= len(self.input):
            self.ch = ""
        else:
            self.ch = self.input[self.read_position]
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
        
        if self.ch == '=':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(EQ, literal)
            else:
                tok = Token(ASSIGN, self.ch)
        elif self.ch == '!':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(NOT_EQ, literal)
            else:
                tok = Token(BANG, self.ch)
        elif self.ch == '<':
            tok = Token(LT, self.ch)
        elif self.ch == '>':
            tok = Token(GT, self.ch)
        elif self.ch == '"':
            tok = Token(STRING, self.read_string())
        elif self.ch == '[':
            tok = Token(LBRACKET, self.ch)
        elif self.ch == ']':
            tok = Token(RBRACKET, self.ch)
        elif self.ch == '(':
            tok = Token(LPAREN, self.ch)
        elif self.ch == ')':
            tok = Token(RPAREN, self.ch)
        elif self.ch == '{':
            tok = Token(LBRACE, self.ch)
        elif self.ch == '}':
            tok = Token(RBRACE, self.ch)
        elif self.ch == ',':
            tok = Token(COMMA, self.ch)
        elif self.ch == ';':
            tok = Token(SEMICOLON, self.ch)
        elif self.ch == ':':
            tok = Token(COLON, self.ch)
        elif self.ch == '+':
            tok = Token(PLUS, self.ch)
        elif self.ch == '-':
            tok = Token(MINUS, self.ch)
        elif self.ch == '*':
            tok = Token(STAR, self.ch)
        elif self.ch == '/':
            tok = Token(SLASH, self.ch)
        elif self.ch == "":
            tok = Token(EOF, "")
        else:
            if self.is_letter(self.ch):
                literal = self.read_identifier()
                token_type = self.lookup_ident(literal)
                return Token(token_type, literal)
            elif self.is_digit(self.ch):
                num_literal = self.read_number()
                if '.' in num_literal:
                    return Token(FLOAT, num_literal)
                else:
                    return Token(INT, num_literal)
            else:
                # Instead of ILLEGAL, try to read as string or skip
                if self.ch in ['\n', '\r']:
                    self.read_char()
                    return self.next_token()
                tok = Token(ILLEGAL, self.ch)

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
            "while": WHILE,
            "use": USE,
            "exactly": EXACTLY,
            "embedded": EMBEDDED,  # ADD THIS
        }
        return keywords.get(ident, IDENT)

    def is_letter(self, char):
        return 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char == '_'

    def is_digit(self, char):
        return '0' <= char <= '9'

    def skip_whitespace(self):
        while self.ch in [' ', '\t', '\n', '\r']:
            self.read_char()