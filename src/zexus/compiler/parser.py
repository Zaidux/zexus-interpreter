# parser.py (TOLERANT MULTI-STRATEGY PARSER)
"""
Clean Production Parser for Zexus - Fixes Object Literal Issue
"""

from ..zexus_token import *
from .lexer import Lexer
from .zexus_ast import *

# Precedence constants
LOWEST, ASSIGN_PREC, EQUALS, LESSGREATER, SUM, PRODUCT, PREFIX, CALL, LOGICAL = 1, 2, 3, 4, 5, 6, 7, 8, 9

precedences = {
    EQ: EQUALS, NOT_EQ: EQUALS,
    LT: LESSGREATER, GT: LESSGREATER, LTE: LESSGREATER, GTE: LESSGREATER,
    PLUS: SUM, MINUS: SUM,
    SLASH: PRODUCT, STAR: PRODUCT, MOD: PRODUCT,
    AND: LOGICAL, OR: LOGICAL,
    LPAREN: CALL,
    DOT: CALL,
    ASSIGN: ASSIGN_PREC,
}

class ProductionParser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.errors = []
        self.cur_token = None
        self.peek_token = None
        
        # Parser function maps
        self.prefix_parse_fns = {
            IDENT: self.parse_identifier,
            INT: self.parse_integer_literal,
            FLOAT: self.parse_float_literal,
            STRING: self.parse_string_literal,
            BANG: self.parse_prefix_expression,
            MINUS: self.parse_prefix_expression,
            TRUE: self.parse_boolean,
            FALSE: self.parse_boolean,
            LPAREN: self.parse_grouped_expression,
            IF: self.parse_if_expression,
            LBRACKET: self.parse_list_literal,
            LBRACE: self.parse_map_literal,  # FIXED: Map literal parsing
            ACTION: self.parse_action_literal,
            EMBEDDED: self.parse_embedded_literal,
            LAMBDA: self.parse_lambda_expression,
            AWAIT: self.parse_await_expression,  # NEW: Await expression
        }
        
        self.infix_parse_fns = {
            PLUS: self.parse_infix_expression,
            MINUS: self.parse_infix_expression,
            SLASH: self.parse_infix_expression,
            STAR: self.parse_infix_expression,
            MOD: self.parse_infix_expression,
            EQ: self.parse_infix_expression,
            NOT_EQ: self.parse_infix_expression,
            LT: self.parse_infix_expression,
            GT: self.parse_infix_expression,
            LTE: self.parse_infix_expression,
            GTE: self.parse_infix_expression,
            AND: self.parse_infix_expression,
            OR: self.parse_infix_expression,
            ASSIGN: self.parse_assignment_expression,
            LPAREN: self.parse_call_expression,
            DOT: self.parse_method_call_expression,
            LAMBDA: self.parse_lambda_infix,  # support arrow-style lambdas: params => body
        }
        
        self.next_token()
        self.next_token()

    def parse_program(self):
        """Clean, efficient program parsing"""
        program = Program()
        while not self.cur_token_is(EOF):
            # Tolerant: skip stray semicolons between statements
            if self.cur_token_is(SEMICOLON):
                self.next_token()
                continue

            stmt = self.parse_statement()
            if stmt:
                program.statements.append(stmt)

            self.next_token()
        return program

    def parse_statement(self):
        """Parse statements with clear error reporting"""
        try:
            if self.cur_token_is(LET):
                return self.parse_let_statement()
            elif self.cur_token_is(RETURN):
                return self.parse_return_statement()
            elif self.cur_token_is(PRINT):
                return self.parse_print_statement()
            elif self.cur_token_is(FOR):
                return self.parse_for_each_statement()
            elif self.cur_token_is(ACTION):
                return self.parse_action_statement()
            elif self.cur_token_is(ASYNC):
                # support "async action name ..." or "action async name ..."
                # If "async action ..." then consume ASYNC and expect ACTION next
                self.next_token()
                if self.cur_token_is(ACTION):
                    return self.parse_action_statement(async_flag=True)
                # otherwise error
                self.errors.append(f"Line {self.cur_token.line}: Expected 'action' after 'async'")
                return None
            elif self.cur_token_is(EVENT):
                return self.parse_event_declaration()
            elif self.cur_token_is(EMIT):
                return self.parse_emit_statement()
            elif self.cur_token_is(ENUM):
                return self.parse_enum_declaration()
            elif self.cur_token_is(PROTOCOL):
                return self.parse_protocol_declaration()
            elif self.cur_token_is(IMPORT):
                return self.parse_import_statement()
            else:
                return self.parse_expression_statement()
        except Exception as e:
            self.errors.append(f"Line {self.cur_token.line}: Parse error - {str(e)}")
            return None

    def parse_let_statement(self):
        """Fixed: Properly handles object literals"""
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {self.cur_token.line}: Expected variable name after 'let'")
            return None
            
        name = Identifier(self.cur_token.literal)
        
        if not self.expect_peek(ASSIGN):
            return None
            
        self.next_token()
        value = self.parse_expression(LOWEST)
        
        return LetStatement(name=name, value=value)

    def parse_map_literal(self):
        """FIXED: Proper map literal parsing - this was the core issue!"""
        pairs = []

        # Must be called when current token is LBRACE
        if not self.cur_token_is(LBRACE):
            self.errors.append(f"Line {getattr(self.cur_token, 'line', 'unknown')}: parse_map_literal called on non-brace token")
            return None

        # Move inside the braces
        self.next_token()  # advance to token after '{'

        # Handle empty object case: {}
        if self.cur_token_is(RBRACE):
            return MapLiteral(pairs)

        # Parse key-value pairs
        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            # Parse key (can be string or identifier)
            if self.cur_token_is(STRING):
                key = StringLiteral(self.cur_token.literal)
            elif self.cur_token_is(IDENT):
                key = Identifier(self.cur_token.literal)
            else:
                self.errors.append(f"Line {getattr(self.cur_token, 'line', 'unknown')}: Object key must be string or identifier")
                return None

            # Expect colon (current peek should be COLON)
            if not self.expect_peek(COLON):
                return None

            # Move to value token and parse it
            self.next_token()
            value = self.parse_expression(LOWEST)
            if value is None:
                return None

            pairs.append((key, value))

            # Accept comma OR semicolon as separators; tolerate trailing separators
            if self.peek_token_is(COMMA) or self.peek_token_is(SEMICOLON):
                self.next_token()  # move to separator
                # advance to next token after separator (or closing brace)
                if self.peek_token_is(RBRACE):
                    self.next_token()  # move to RBRACE and break next loop iteration
                    break
                self.next_token()
                continue

            # If closing brace is the next token, consume it and finish
            if self.peek_token_is(RBRACE):
                self.next_token()  # advance to RBRACE
                break

            # Otherwise, try to advance; tolerant parsing
            self.next_token()

        # Final check: should be at a RBRACE token
        if not self.cur_token_is(RBRACE):
            self.errors.append(f"Line {getattr(self.cur_token, 'line', 'unknown')}: Expected '}}' to close object literal")
            return None

        # Note: we keep the closing brace consumed (caller behavior consistent)
        return MapLiteral(pairs)

    # Rest of parser methods (simplified for production)
    def parse_expression(self, precedence):
        if self.cur_token.type not in self.prefix_parse_fns:
            self.errors.append(f"Line {self.cur_token.line}: Unexpected token '{self.cur_token.literal}'")
            return None

        prefix = self.prefix_parse_fns[self.cur_token.type]
        left_exp = prefix()

        if left_exp is None:
            return None

        while (not self.peek_token_is(SEMICOLON) and 
               not self.peek_token_is(EOF) and 
               precedence <= self.peek_precedence()):

            if self.peek_token.type not in self.infix_parse_fns:
                return left_exp

            infix = self.infix_parse_fns[self.peek_token.type]
            self.next_token()
            left_exp = infix(left_exp)

            if left_exp is None:
                return None

        return left_exp

    def parse_identifier(self):
        return Identifier(value=self.cur_token.literal)

    def parse_integer_literal(self):
        try:
            return IntegerLiteral(value=int(self.cur_token.literal))
        except ValueError:
            self.errors.append(f"Line {self.cur_token.line}: Could not parse {self.cur_token.literal} as integer")
            return None

    def parse_float_literal(self):
        try:
            return FloatLiteral(value=float(self.cur_token.literal))
        except ValueError:
            self.errors.append(f"Line {self.cur_token.line}: Could not parse {self.cur_token.literal} as float")
            return None

    def parse_string_literal(self):
        return StringLiteral(value=self.cur_token.literal)

    def parse_boolean(self):
        return Boolean(value=self.cur_token_is(TRUE))

    def parse_list_literal(self):
        elements = self.parse_expression_list(RBRACKET)
        return ListLiteral(elements=elements)

    def parse_grouped_expression(self):
        self.next_token()
        exp = self.parse_expression(LOWEST)
        if not self.expect_peek(RPAREN):
            return None
        return exp

    def parse_prefix_expression(self):
        expression = PrefixExpression(operator=self.cur_token.literal, right=None)
        self.next_token()
        expression.right = self.parse_expression(PREFIX)
        return expression

    def parse_infix_expression(self, left):
        expression = InfixExpression(left=left, operator=self.cur_token.literal, right=None)
        precedence = self.cur_precedence()
        self.next_token()
        expression.right = self.parse_expression(precedence)
        return expression

    def parse_call_expression(self, function):
        arguments = self.parse_expression_list(RPAREN)
        return CallExpression(function=function, arguments=arguments)

    def parse_assignment_expression(self, left):
        if not isinstance(left, Identifier):
            self.errors.append(f"Line {self.cur_token.line}: Cannot assign to {type(left).__name__}")
            return None

        expression = AssignmentExpression(name=left, value=None)
        self.next_token()
        expression.value = self.parse_expression(LOWEST)
        return expression

    def parse_method_call_expression(self, left):
        if not self.expect_peek(IDENT):
            return None

        method = Identifier(self.cur_token.literal)

        if self.peek_token_is(LPAREN):
            self.next_token()
            arguments = self.parse_expression_list(RPAREN)
            return MethodCallExpression(object=left, method=method, arguments=arguments)
        else:
            return PropertyAccessExpression(object=left, property=method)

    def parse_expression_list(self, end):
        elements = []
        if self.peek_token_is(end):
            self.next_token()
            return elements

        self.next_token()
        elements.append(self.parse_expression(LOWEST))

        while self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            elements.append(self.parse_expression(LOWEST))

        if not self.expect_peek(end):
            return elements

        return elements

    # Statement parsing methods
    def parse_expression_statement(self):
        """Parse expression as a statement"""
        stmt = ExpressionStatement(expression=self.parse_expression(LOWEST))
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        return stmt

    def parse_return_statement(self):
        stmt = ReturnStatement(return_value=None)
        self.next_token()
        stmt.return_value = self.parse_expression(LOWEST)
        return stmt

    def parse_print_statement(self):
        stmt = PrintStatement(value=None)
        self.next_token()
        stmt.value = self.parse_expression(LOWEST)
        return stmt

    def parse_if_statement(self):
        self.next_token()  # Skip IF
        
        # Parse condition (with or without parentheses)
        if self.cur_token_is(LPAREN):
            self.next_token()
            condition = self.parse_expression(LOWEST)
            if self.cur_token_is(RPAREN):
                self.next_token()
        else:
            condition = self.parse_expression(LOWEST)

        if not condition:
            return None

        # Parse consequence
        consequence = self.parse_block()
        if not consequence:
            return None

        alternative = None
        if self.cur_token_is(ELSE):
            self.next_token()
            alternative = self.parse_block()

        return IfStatement(condition=condition, consequence=consequence, alternative=alternative)

    def parse_block(self):
        block = BlockStatement()

        # Handle different block styles
        if self.cur_token_is(LBRACE):
            self.next_token()  # Skip {

            while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
                # Skip stray semicolons between statements inside a block
                if self.cur_token_is(SEMICOLON):
                    self.next_token()
                    continue

                stmt = self.parse_statement()
                if stmt:
                    block.statements.append(stmt)

                # After parsing a statement, consume any trailing semicolons so they don't become unexpected tokens
                while self.peek_token_is(SEMICOLON):
                    self.next_token()  # move to semicolon
                    self.next_token()  # move past semicolon

                # Advance to next token if parser hasn't advanced to EOF or closing brace
                if not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
                    self.next_token()

            if self.cur_token_is(EOF):
                self.errors.append("Unclosed block (reached EOF)")
        else:
            # Single statement block
            # Tolerant: allow a trailing semicolon after single statement
            stmt = self.parse_statement()
            if stmt:
                block.statements.append(stmt)
                if self.peek_token_is(SEMICOLON):
                    self.next_token()  # consume semicolon

        return block

    def parse_for_each_statement(self):
        if not self.expect_peek(EACH):
            return None
            
        if not self.expect_peek(IDENT):
            return None
            
        item = Identifier(self.cur_token.literal)
        
        if not self.expect_peek(IN):
            return None
            
        self.next_token()
        iterable = self.parse_expression(LOWEST)
        
        body = self.parse_block()
        
        return ForEachStatement(item=item, iterable=iterable, body=body)

    def parse_action_statement(self, async_flag=False):
        """Parse action declaration; supports optional async modifier (action async name(...) { ... } or async action ...)"""
        # current token is ACTION (or we arrived here after consuming ASYNC)
        is_async = async_flag

        # Handle optional 'async' immediately after 'action': action async name ...
        if self.peek_token_is(IDENT) and self.peek_token.literal == "async":
            # unusual: peek is IDENT with literal "async" — but lexer maps "async" to ASYNC token;
            pass

        # If next token is ASYNC (action async name ...)
        if self.peek_token_is(ASYNC):
            self.next_token()
            is_async = True

        # Continue normal action parse
        if not self.expect_peek(IDENT):
            return None
        name = Identifier(self.cur_token.literal)

        parameters = []
        if self.peek_token_is(LPAREN):
            # Advance into the parameter list: consume '(' then move to first inner token
            self.next_token()  # move to LPAREN
            self.next_token()  # move to token after '('
            parameters = self.parse_parameter_list()
            # If current token is ')' (empty params case), advance past it.
            if self.cur_token_is(RPAREN):
                self.next_token()
            else:
                # Otherwise expect a closing paren next and consume it
                if not self.expect_peek(RPAREN):
                    return None
                # consume the ')'
                self.next_token()

        body = self.parse_block()
        # Create ActionStatement with is_async flag (add attribute)
        stmt = ActionStatement(name=name, parameters=parameters, body=body)
        # attach async flag if supported by AST
        setattr(stmt, "is_async", is_async)
        return stmt

    def parse_action_literal(self):
        if not self.expect_peek(LPAREN):
            return None
            
        parameters = self.parse_parameter_list()
        if parameters is None:
            return None

        body = self.parse_expression(LOWEST)
        # If action literal is used as an expression, callers may expect a function-like node.
        # Mark the ActionLiteral node to indicate expression-level function (helps lowering).
        action_lit = ActionLiteral(parameters=parameters, body=body)
        setattr(action_lit, "is_expression", True)
        return action_lit

    def parse_await_expression(self):
        # current token is AWAIT
        self.next_token()
        value = self.parse_expression(LOWEST)
        return AwaitExpression(expression=value)

    def parse_if_expression(self):
        """Parse if expression: if (condition) { consequence } else { alternative }"""
        expression = IfExpression(condition=None, consequence=None, alternative=None)

        if not self.expect_peek(LPAREN):
            return None

        self.next_token()
        expression.condition = self.parse_expression(LOWEST)

        if not self.expect_peek(RPAREN):
            return None

        if not self.expect_peek(LBRACE):
            return None

        expression.consequence = self.parse_block()

        if self.peek_token_is(ELSE):
            self.next_token()
            if not self.expect_peek(LBRACE):
                return None
            expression.alternative = self.parse_block()

        return expression

    def parse_embedded_literal(self):
        """Parse embedded code block: @{ language ... code ... }"""
        if not self.expect_peek(LBRACE):
            return None

        self.next_token()
        code_lines = []
        
        # Read until closing brace
        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            code_lines.append(self.cur_token.literal)
            self.next_token()
        
        if not self.cur_token_is(RBRACE):
            self.errors.append("Expected } after embedded code block")
            return None
        
        self.next_token()  # Skip closing brace
        
        code_content = ' '.join(code_lines)
        lines = code_content.strip().split('\n')
        
        if not lines:
            self.errors.append("Empty embedded code block")
            return None
        
        # First line is language, rest is code
        language_line = lines[0].strip() if lines else "unknown"
        language = language_line if language_line else "unknown"
        code = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
        
        return EmbeddedLiteral(language=language, code=code)

    def parse_lambda_expression(self):
        """Parse lambda/arrow function: x => body or lambda(x): body"""
        token = self.cur_token
        parameters = []

        self.next_token()

        if self.cur_token_is(LPAREN):
            self.next_token()
            parameters = self.parse_parameter_list()
            if not self.expect_peek(RPAREN):
                return None
        elif self.cur_token_is(IDENT):
            # Single parameter without parens
            parameters = [Identifier(self.cur_token.literal)]
            self.next_token()

        # Handle arrow or colon separator
        if self.cur_token_is(COLON):
            self.next_token()
        elif self.cur_token_is(LAMBDA):  # => token
            self.next_token()
        elif self.cur_token_is(MINUS) and self.peek_token_is(GT):
            self.next_token()  # Skip -
            self.next_token()  # Skip >

        body = self.parse_expression(LOWEST)
        return LambdaExpression(parameters=parameters, body=body)

    def parse_lambda_infix(self, left):
        """Parse arrow-style lambda when encountering leftside 'params' followed by =>

        Examples:
            x => x + 1
            (a, b) => a + b
        """
        # Build parameter list from `left` expression
        params = []
        if isinstance(left, Identifier):
            params = [left]
        else:
            try:
                if hasattr(left, 'elements'):
                    for el in left.elements:
                        if isinstance(el, Identifier):
                            params.append(el)
            except Exception:
                pass

        # Current token is LAMBDA; advance to body
        self.next_token()

        if self.cur_token_is(COLON):
            self.next_token()

        body = self.parse_expression(LOWEST)
        return LambdaExpression(parameters=params, body=body)

    def parse_action_literal(self):
        """Parse action literal: action (params) { body } or action (params) => expr"""
        if not self.expect_peek(LPAREN):
            return None
            
        parameters = self.parse_parameter_list()
        if parameters is None:
            return None

        # Expect closing paren
        if not self.expect_peek(RPAREN):
            return None
        
        self.next_token()

        # Action body can be a block or expression
        if self.cur_token_is(LBRACE):
            body = self.parse_block()
        else:
            # Expression body (shorthand)
            body = self.parse_expression(LOWEST)
        
        action_lit = ActionLiteral(parameters=parameters, body=body)
        return action_lit

    def parse_parameter_list(self):
        """Parse parameter list for functions"""
        parameters = []
        
        if self.cur_token_is(RPAREN):
            return parameters
        
        while not self.cur_token_is(RPAREN) and not self.cur_token_is(EOF):
            if self.cur_token_is(IDENT):
                parameters.append(Identifier(self.cur_token.literal))
                self.next_token()
            else:
                break
            
            # Handle comma separator
            if self.cur_token_is(COMMA):
                self.next_token()
            elif self.cur_token_is(RPAREN):
                break
            else:
                break
        
        return parameters

    def parse_event_declaration(self):
        if not self.expect_peek(IDENT):
            return None
        name = Identifier(self.cur_token.literal)
        body = self.parse_block()
        return EventDeclaration(name=name, properties=body)

    def parse_emit_statement(self):
        if not self.expect_peek(IDENT):
            return None
        name = Identifier(self.cur_token.literal)
        payload = None
        if self.peek_token_is(LPAREN):
            self.next_token()
            self.next_token()
            payload = self.parse_expression(LOWEST)
            if not self.expect_peek(RPAREN):
                return None
        elif self.peek_token_is(LBRACE):
            self.next_token()
            payload = self.parse_block()
        return EmitStatement(name=name, payload=payload)

    def parse_enum_declaration(self):
        if not self.expect_peek(IDENT):
            return None
        name = Identifier(self.cur_token.literal)
        if not self.expect_peek(LBRACE):
            return None
        # parse simple comma-separated identifiers or key:value pairs
        members = {}
        self.next_token()
        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            if self.cur_token_is(IDENT) or self.cur_token_is(STRING):
                key = self.cur_token.literal
                # optional colon value
                val = None
                if self.peek_token_is(COLON):
                    self.next_token()
                    self.next_token()
                    if self.cur_token_is(INT):
                        val = int(self.cur_token.literal)
                members[key] = val
            if self.peek_token_is(COMMA):
                self.next_token()
                self.next_token()
                continue
            self.next_token()
        if not self.cur_token_is(RBRACE):
            self.errors.append("Unclosed enum declaration")
            return None
        return EnumDeclaration(name=name, members=members)

    def parse_protocol_declaration(self):
        if not self.expect_peek(IDENT):
            return None
        name = Identifier(self.cur_token.literal)
        if not self.expect_peek(LBRACE):
            return None
        # parse simple list of method signatures (as identifiers)
        spec = {"methods": []}
        self.next_token()
        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            if self.cur_token_is(IDENT):
                spec["methods"].append(self.cur_token.literal)
            if self.peek_token_is(COMMA):
                self.next_token()
                self.next_token()
                continue
            self.next_token()
        if not self.cur_token_is(RBRACE):
            self.errors.append("Unclosed protocol declaration")
            return None
        return ProtocolDeclaration(name=name, spec=spec)

    def parse_import_statement(self):
        if not self.expect_peek(STRING):
            return None
        module_path = self.cur_token.literal
        alias = None
        if self.peek_token_is(IDENT) and self.peek_token.literal == "as":
            self.next_token()
            self.next_token()
            if self.cur_token_is(IDENT):
                alias = self.cur_token.literal
        return ImportStatement(module_path=module_path, alias=alias)

    # Token utilities
    def next_token(self):
        self.cur_token = self.peek_token
        self.peek_token = self.lexer.next_token()

    def cur_token_is(self, t):
        return self.cur_token.type == t

    def peek_token_is(self, t):
        return self.peek_token.type == t

    def expect_peek(self, t):
        if self.peek_token_is(t):
            self.next_token()
            return True
        self.errors.append(f"Line {self.cur_token.line}: Expected '{t}', got '{self.peek_token.type}'")
        return False

    def peek_precedence(self):
        return precedences.get(self.peek_token.type, LOWEST)

    def cur_precedence(self):
        return precedences.get(self.cur_token.type, LOWEST)

# --- Compatibility alias ----------------------------------------------------
# Provide the common name `Parser` for code that imports the compiler parser
# using older/alternate names.
Parser = ProductionParser