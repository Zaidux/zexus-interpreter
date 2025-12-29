# lexer.py (ENHANCED WITH PHASE 1 KEYWORDS)
from .zexus_token import *
from .error_reporter import get_error_reporter, SyntaxError as ZexusSyntaxError

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
        return tok

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
        # keyword lookup mapping (string -> token constant)
        keywords = {
            "let": LET,
            "const": CONST,             # NEW: Const keyword for immutable variables
            "data": DATA,               # NEW: Data keyword for dataclass definitions
            "print": PRINT,
            "if": IF,
            "then": THEN,              # NEW: Then keyword for if-then-else expressions
            "elif": ELIF,               # NEW: Elif keyword for else-if conditionals
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
            "exactly": EXACTLY,
            "embedded": EMBEDDED,
            "export": EXPORT,
            "lambda": LAMBDA,
            "debug": DEBUG,      # DUAL-MODE: Works as both statement (debug x;) and function (debug(x))
            "try": TRY,          # NEW: Try keyword  
            "catch": CATCH,      # NEW: Catch keyword
            "continue": CONTINUE, # NEW: Continue on error keyword
            "break": BREAK,      # NEW: Break loop keyword
            "throw": THROW,      # NEW: Throw error keyword
            "external": EXTERNAL, # NEW: External keyword
            "from": FROM,        # NEW: From keyword
            "screen": SCREEN,         # NEW: renderer keyword
            "component": COMPONENT,   # NEW: renderer keyword
            "theme": THEME,           # NEW: renderer keyword
            "canvas": CANVAS,         # NEW (optional recognition)
            "graphics": GRAPHICS,     # NEW (optional recognition)
            "animation": ANIMATION,   # NEW (optional recognition)
            "clock": CLOCK,           # NEW (optional recognition)
            "async": ASYNC,
            "await": AWAIT,
            "channel": CHANNEL,       # NEW: Channel for concurrent communication
            "send": SEND,             # NEW: Send to channel
            "receive": RECEIVE,       # NEW: Receive from channel
            "atomic": ATOMIC,         # NEW: Atomic operations
            "event": EVENT,
            "emit": EMIT,
            "enum": ENUM,
            "protocol": PROTOCOL,
            "import": IMPORT,
            # Modifiers
            "public": PUBLIC,
            "private": PRIVATE,
            "sealed": SEALED,
            "secure": SECURE,
            "pure": PURE,
            "view": VIEW,
            "payable": PAYABLE,
            "modifier": MODIFIER,
            # NEW: Entity, Verify, Contract, Protect
            "entity": ENTITY,
            "verify": VERIFY,
            "contract": CONTRACT,
            "protect": PROTECT,
            "implements": IMPLEMENTS,
            "this": THIS,
            "interface": INTERFACE,
            "capability": CAPABILITY,  # NEW: Capability keyword for security
            "grant": GRANT,             # NEW: Grant keyword for capability grants
            "revoke": REVOKE,           # NEW: Revoke keyword for capability revocation
            "module": MODULE,           # NEW: Module keyword for code organization
            "package": PACKAGE,         # NEW: Package keyword for package definition
            "using": USING,             # NEW: Using keyword for resource management
            "type_alias": TYPE_ALIAS,   # NEW: Type alias keyword for type definitions
            "seal": SEAL,               # NEW: Seal keyword for immutable objects
            "audit": AUDIT,             # NEW: Audit keyword for compliance logging
            "restrict": RESTRICT,       # NEW: Restrict keyword for field-level access control
            "sandbox": SANDBOX,         # NEW: Sandbox keyword for isolated execution
            "trail": TRAIL,             # NEW: Trail keyword for real-time logging
            # Advanced features
            "middleware": MIDDLEWARE,
            "auth": AUTH,
            "throttle": THROTTLE,
            "cache": CACHE,
            # Blockchain & Smart Contract keywords
            "ledger": LEDGER,           # Immutable state ledger
            "state": STATE,             # State management
            "revert": REVERT,           # Revert transaction
            # NOTE: "tx" removed as keyword - users can use it as variable name
            # Only uppercase "TX" is reserved for transaction context
            "limit": LIMIT,             # Gas/resource limit
            # NOTE: hash, signature, verify_sig, gas are BUILTINS, not keywords
            # NEW: Persistent storage keywords
            "persistent": PERSISTENT,   # NEW: Persistent keyword
            "storage": STORAGE,         # NEW: Storage keyword
            "require": REQUIRE,         # Already defined in zexus_token.py
            # Logical operators as keywords (alternative to && and ||)
            "and": AND,                 # Logical AND (alternative to &&)
            "or": OR,                   # Logical OR (alternative to ||)
            # Performance optimization keywords
            "native": NATIVE,           # Performance: call C/C++ code
            "gc": GC,                   # Performance: control garbage collection
            "inline": INLINE,           # Performance: function inlining
            "buffer": BUFFER,           # Performance: direct memory access
            "simd": SIMD,               # Performance: vector operations
            "defer": DEFER,             # Convenience: cleanup code execution
            "pattern": PATTERN,         # Convenience: pattern matching
            "match": MATCH,             # Match expression for pattern matching
            "enum": ENUM,               # Advanced: type-safe enumerations
            "stream": STREAM,           # Advanced: event streaming
            "watch": WATCH,             # Advanced: reactive state management
            "log": LOG,                 # Output logging to file
            "inject": INJECT,           # Advanced: dependency injection
            "validate": VALIDATE,       # Data validation
            "sanitize": SANITIZE,       # Data sanitization
        }
        return keywords.get(ident, IDENT)

    def is_letter(self, char):
        return 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char == '_'

    def is_digit(self, char):
        return '0' <= char <= '9'

    def skip_whitespace(self):
        while self.ch in [' ', '\t', '\n', '\r']:
            self.read_char()