# lexer.py (ENHANCED WITH PHASE 1 KEYWORDS)
from .zexus_token import *
from .error_reporter import get_error_reporter, SyntaxError as ZexusSyntaxError

_LITERAL_KEYWORDS = {
    "true": TRUE,
    "false": FALSE,
    "null": NULL,
}

_STRICT_KEYWORDS = {
    'if', 'elif', 'else', 'while', 'for', 'each', 'in',
    'return', 'break', 'continue', 'throw', 'try', 'catch',
    'await', 'async', 'spawn', 'let', 'const', 'print',
    'use', 'find', 'load', 'export', 'import', 'debug', 'match', 'lambda',
    'case', 'default'
}

_CONTEXTS_ALLOWING_KEYWORD_IDENTS = {
    LET, CONST, DOT, COMMA, LBRACKET, COLON, ASSIGN
}

_KEYWORDS = {
    "let": LET,
    "const": CONST,
    "data": DATA,
    "print": PRINT,
    "if": IF,
    "then": THEN,
    "elif": ELIF,
    "else": ELSE,
    "true": TRUE,
    "false": FALSE,
    "null": NULL,
    "return": RETURN,
    "for": FOR,
    "each": EACH,
    "in": IN,
    "action": ACTION,
    "function": FUNCTION,
    "while": WHILE,
    "use": USE,
    "find": FIND,
    "load": LOAD,
    "exactly": EXACTLY,
    "embedded": EMBEDDED,
    "export": EXPORT,
    "lambda": LAMBDA,
    "debug": DEBUG,
    "try": TRY,
    "catch": CATCH,
    "continue": CONTINUE,
    "break": BREAK,
    "throw": THROW,
    "external": EXTERNAL,
    "screen": SCREEN,
    "component": COMPONENT,
    "theme": THEME,
    "color": COLOR,
    "canvas": CANVAS,
    "graphics": GRAPHICS,
    "animation": ANIMATION,
    "clock": CLOCK,
    "async": ASYNC,
    "await": AWAIT,
    "channel": CHANNEL,
    "send": SEND,
    "receive": RECEIVE,
    "atomic": ATOMIC,
    "event": EVENT,
    "emit": EMIT,
    "enum": ENUM,
    "protocol": PROTOCOL,
    "import": IMPORT,
    "public": PUBLIC,
    "private": PRIVATE,
    "sealed": SEALED,
    "secure": SECURE,
    "pure": PURE,
    "view": VIEW,
    "payable": PAYABLE,
    "modifier": MODIFIER,
    "entity": ENTITY,
    "verify": VERIFY,
    "contract": CONTRACT,
    "protect": PROTECT,
    "implements": IMPLEMENTS,
    "this": THIS,
    "as": AS,
    "interface": INTERFACE,
    "capability": CAPABILITY,
    "grant": GRANT,
    "revoke": REVOKE,
    "module": MODULE,
    "package": PACKAGE,
    "using": USING,
    "type_alias": TYPE_ALIAS,
    "seal": SEAL,
    "audit": AUDIT,
    "restrict": RESTRICT,
    "sandbox": SANDBOX,
    "trail": TRAIL,
    "middleware": MIDDLEWARE,
    "auth": AUTH,
    "throttle": THROTTLE,
    "cache": CACHE,
    "ledger": LEDGER,
    "state": STATE,
    "revert": REVERT,
    "limit": LIMIT,
    "persistent": PERSISTENT,
    "storage": STORAGE,
    "require": REQUIRE,
    "and": AND,
    "or": OR,
    "native": NATIVE,
    "gc": GC,
    "inline": INLINE,
    "buffer": BUFFER,
    "simd": SIMD,
    "defer": DEFER,
    "pattern": PATTERN,
    "match": MATCH,
    "case": CASE,
    "default": DEFAULT,
    "enum": ENUM,
    "stream": STREAM,
    "watch": WATCH,
    "log": LOG,
    "inject": INJECT,
    "validate": VALIDATE,
    "sanitize": SANITIZE,
}

_FUNCTION_DECL_KEYWORDS = {"action", "function"}

_FUNCTION_STATEMENT_BOUNDARIES = {
    None,
    SEMICOLON,
    LBRACE,
    RBRACE,
    RBRACKET,
    INT,
    STRING,
    FLOAT,
    RPAREN,
    TRUE,
    FALSE,
    NULL,
    RETURN,
    ASSIGN,
    ASYNC,
    EXPORT,
    PUBLIC,
    PRIVATE,
    SEALED,
    INLINE,
    SECURE,
    PURE,
    VIEW,
    PAYABLE,
    NATIVE,
}

_DATA_KEYWORD_CONTRACT_CONTEXTS = {
    SEMICOLON,
    LBRACE,
    RBRACE,
    RBRACKET,
    STRING,
    INT,
    FLOAT,
    TRUE,
    FALSE,
    NULL,
    PRIVATE,
    PUBLIC,
    SEALED,
    SECURE,
    PURE,
    VIEW,
    PAYABLE,
}

class Lexer:
    def __init__(self, source_code, filename="<stdin>"):
        self.input = source_code
        self.position = 0
        self.read_position = 0
        self.ch = ""
        self.in_embedded_block = False
        self.line = 1
        self.column = 1
        self.filename = filename
        # Hint for parser: when '(' starts a lambda parameter list that is
        # immediately followed by '=>', this flag will be set for the token
        # produced for that '('. Parser can check and consume accordingly.
        self._next_paren_has_lambda = False
        # Track last token type to enable context-aware keyword handling
        self.last_token_type = None
        # Track statement boundaries and nesting depth to disambiguate keywords vs identifiers
        self.at_statement_boundary = True
        self.paren_depth = 0
        self.bracket_depth = 0
        self.brace_depth = 0
        
        # Register source with error reporter
        self.error_reporter = get_error_reporter()
        self.error_reporter.register_source(filename, source_code)
        
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

        # CRITICAL FIX: Skip single line comments (both # and // styles)
        if self.ch == '#' and self.peek_char() != '{':
            self.skip_comment()
            return self.next_token()

        # NEW: Handle // style comments
        if self.ch == '/' and self.peek_char() == '/':
            self.skip_double_slash_comment()
            return self.next_token()

        tok = None
        current_line = self.line
        current_column = self.column

        if self.ch == '=':
            # Equality '=='
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(EQ, literal)
                tok.line = current_line
                tok.column = current_column
            # Arrow '=>' (treat as lambda shorthand)
            elif self.peek_char() == '>':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(LAMBDA, literal)
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
                # Single '&' is not supported - suggest using '&&'
                error = self.error_reporter.report_error(
                    ZexusSyntaxError,
                    f"Unexpected character '{self.ch}'",
                    line=current_line,
                    column=current_column,
                    filename=self.filename,
                    suggestion="Did you mean '&&' for logical AND?"
                )
                raise error
        elif self.ch == '|':
            if self.peek_char() == '|':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(OR, literal)
                tok.line = current_line
                tok.column = current_column
            else:
                # Single '|' is not supported - suggest using '||'
                error = self.error_reporter.report_error(
                    ZexusSyntaxError,
                    f"Unexpected character '{self.ch}'",
                    line=current_line,
                    column=current_column,
                    filename=self.filename,
                    suggestion="Did you mean '||' for logical OR?"
                )
                raise error
        elif self.ch == '<':
            if self.peek_char() == '=':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(LTE, literal)
                tok.line = current_line
                tok.column = current_column
            elif self.peek_char() == '<':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(IMPORT_OP, literal)
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
            elif self.peek_char() == '>':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(APPEND, literal)
                tok.line = current_line
                tok.column = current_column
            else:
                tok = Token(GT, self.ch)
                tok.line = current_line
                tok.column = current_column
        elif self.ch == '?':
            # Check for nullish coalescing '??'
            if self.peek_char() == '?':
                ch = self.ch
                self.read_char()
                literal = ch + self.ch
                tok = Token(NULLISH, literal)
                tok.line = current_line
                tok.column = current_column
            else:
                tok = Token(QUESTION, self.ch)
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
        elif self.ch == '@':
            tok = Token(AT, self.ch)
            tok.line = current_line
            tok.column = current_column
        elif self.ch == '(':
            # Quick char-level scan: detect if this '(' pairs with a ')' that
            # is followed by '=>' (arrow). If so, set a hint flag so parser
            # can treat the parentheses as a lambda-parameter list.
            try:
                src = self.input
                i = self.position
                depth = 0
                found = False
                while i < len(src):
                    c = src[i]
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                        if depth == 0:
                            # look ahead for '=>' skipping whitespace
                            j = i + 1
                            while j < len(src) and src[j].isspace():
                                j += 1
                            if j + 1 < len(src) and src[j] == '=' and src[j + 1] == '>':
                                found = True
                            break
                    i += 1
                self._next_paren_has_lambda = found
            except Exception:
                self._next_paren_has_lambda = False

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
            # Check if this is division or comment
            if self.peek_char() == '/':
                # It's a // comment, handle above
                self.skip_double_slash_comment()
                return self.next_token()
            else:
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
                self._finalize_token(tok)
                return tok
            elif self.is_digit(self.ch):
                num_literal = self.read_number()
                if '.' in num_literal:
                    tok = Token(FLOAT, num_literal)
                else:
                    tok = Token(INT, num_literal)
                tok.line = current_line
                tok.column = current_column
                self._finalize_token(tok)
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
                    self._finalize_token(tok)
                    return tok
                # Unknown character - report helpful error
                char_desc = f"'{self.ch}'" if self.ch.isprintable() else f"'\\x{ord(self.ch):02x}'"
                error = self.error_reporter.report_error(
                    ZexusSyntaxError,
                    f"Unexpected character {char_desc}",
                    line=current_line,
                    column=current_column,
                    filename=self.filename,
                    suggestion="Remove or replace this character with valid Zexus syntax."
                )
                raise error

        self.read_char()
        self._finalize_token(tok)
        return tok

    def _finalize_token(self, tok):
        """Update lexer state after producing a token."""
        if tok is None:
            return

        token_type = tok.type

        # Maintain nesting depth for parentheses and brackets to help newline handling
        if token_type == LPAREN:
            self.paren_depth += 1
        elif token_type == RPAREN:
            if self.paren_depth > 0:
                self.paren_depth -= 1
        elif token_type == LBRACKET:
            self.bracket_depth += 1
        elif token_type == RBRACKET:
            if self.bracket_depth > 0:
                self.bracket_depth -= 1
        elif token_type == LBRACE:
            self.brace_depth += 1
        elif token_type == RBRACE:
            if self.brace_depth > 0:
                self.brace_depth -= 1

        # Update last token type for context-aware keyword handling
        self.last_token_type = token_type

        # Determine whether the next non-whitespace token is at a statement boundary
        if token_type in {SEMICOLON, RBRACE, LBRACE, EOF}:
            self.at_statement_boundary = True
        elif token_type in {COMMA, DOT, ASSIGN, COLON, LPAREN, LBRACKET, AT}:
            self.at_statement_boundary = False
        elif token_type in {LET, CONST}:
            # Declarations expect an identifier next
            self.at_statement_boundary = False
        else:
            # Default: remain in the current statement
            self.at_statement_boundary = False

    def read_embedded_char(self):
        """Read a single character as identifier for embedded code compatibility"""
        char = self.ch
        self.read_char()
        return char

    def skip_comment(self):
        """Skip # style comments"""
        while self.ch != '\n' and self.ch != "":
            self.read_char()
        self.skip_whitespace()

    def skip_double_slash_comment(self):
        """Skip // style comments"""
        # Consume the first '/'
        self.read_char()
        # Consume the second '/'
        self.read_char()
        # Skip until end of line
        while self.ch != '\n' and self.ch != "":
            self.read_char()
        self.skip_whitespace()

    def read_string(self):
        start_position = self.position + 1
        start_line = self.line
        start_column = self.column
        result = []
        while True:
            self.read_char()
            if self.ch == "":
                # End of input - unclosed string
                error = self.error_reporter.report_error(
                    ZexusSyntaxError,
                    "Unterminated string literal",
                    line=start_line,
                    column=start_column,
                    filename=self.filename,
                    suggestion="Add a closing quote \" to terminate the string."
                )
                raise error
            elif self.ch == '\\':
                # Escape sequence - read next character
                self.read_char()
                if self.ch == '':
                    error = self.error_reporter.report_error(
                        ZexusSyntaxError,
                        "Incomplete escape sequence at end of file",
                        line=self.line,
                        column=self.column,
                        filename=self.filename,
                        suggestion="Remove the backslash or complete the escape sequence."
                    )
                    raise error
                # Map escape sequences to their actual characters
                escape_map = {
                    'n': '\n',
                    't': '\t',
                    'r': '\r',
                    '\\': '\\',
                    '"': '"',
                    "'": "'"
                }
                result.append(escape_map.get(self.ch, self.ch))
            elif self.ch == '"':
                # End of string
                break
            else:
                result.append(self.ch)
        return ''.join(result)

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
        # Always treat literal keywords as reserved regardless of context.
        literal_token = _LITERAL_KEYWORDS.get(ident)
        if literal_token is not None:
            return literal_token

        token = _KEYWORDS.get(ident)
        if token is None:
            return IDENT

        if ident in _FUNCTION_DECL_KEYWORDS:
            if self.last_token_type in _FUNCTION_STATEMENT_BOUNDARIES:
                return token
            return IDENT

        if ident == "data":
            if self.last_token_type in _DATA_KEYWORD_CONTRACT_CONTEXTS:
                return token
            return IDENT

        if ident in _STRICT_KEYWORDS:
            return token

        if not self.at_statement_boundary and self.last_token_type in _CONTEXTS_ALLOWING_KEYWORD_IDENTS:
            return IDENT

        return token

    def is_letter(self, char):
        return 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char == '_'

    def is_digit(self, char):
        return '0' <= char <= '9'

    def skip_whitespace(self):
        while self.ch in [' ', '\t', '\n', '\r']:
            if self.ch in ['\n', '\r']:
                # Treat newline as potential statement boundary when not inside paren/bracket expressions
                if self.paren_depth == 0 and self.bracket_depth == 0:
                    self.at_statement_boundary = True
            self.read_char()