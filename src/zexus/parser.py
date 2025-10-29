# parser.py (COMPLETE FIXED VERSION)
from .zexus_token import *
from .lexer import Lexer
from .zexus_ast import *

# Precedence constants - FIXED: Assignment has LOW precedence (2)
LOWEST, ASSIGN_PREC, EQUALS, LESSGREATER, SUM, PRODUCT, PREFIX, CALL, LOGICAL = 1, 2, 3, 4, 5, 6, 7, 8, 9

precedences = {
    EQ: EQUALS, NOT_EQ: EQUALS,
    LT: LESSGREATER, GT: LESSGREATER, LTE: LESSGREATER, GTE: LESSGREATER,
    PLUS: SUM, MINUS: SUM,
    SLASH: PRODUCT, STAR: PRODUCT, MOD: PRODUCT,
    AND: LOGICAL, OR: LOGICAL,
    LPAREN: CALL,
    DOT: CALL,
    "=": ASSIGN_PREC,  # Assignment has LOW precedence (2)
}

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.errors = []
        self.cur_token = None
        self.peek_token = None

        # Debug: Check token mappings
        print(f"=== TOKEN DEBUG ===")
        print(f"ASSIGN constant: '{ASSIGN}'")
        print(f"'=' in precedences: {'=' in precedences}")
        print(f"Value for '=': {precedences.get('=', 'NOT FOUND')}")
        print("===================")

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
            LBRACE: self.parse_map_literal,
            ACTION: self.parse_action_literal,
            EMBEDDED: self.parse_embedded_literal,
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
        }
        self.next_token()
        self.next_token()

    def parse_assignment_expression(self, left):
        """Parse assignment expressions: identifier = value"""
        print(f"DEBUG: parse_assignment_expression called with left={left}")

        # Only allow assignment to identifiers
        if not isinstance(left, Identifier):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Cannot assign to {type(left).__name__}, only identifiers allowed")
            return None

        expression = AssignmentExpression(name=left, value=None)
        precedence = self.cur_precedence()
        print(f"DEBUG: assignment precedence={precedence}, cur_token={self.cur_token}")

        self.next_token()  # Move past the =

        # Parse the right-hand side with LOWEST precedence to allow full expressions
        print(f"DEBUG: parsing right-hand side of assignment")
        expression.value = self.parse_expression(LOWEST)
        print(f"DEBUG: assignment result: {expression}")
        return expression

    def parse_method_call_expression(self, left):
        """Parse: object.method(arguments)"""
        if not self.cur_token_is(DOT):
            return None

        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected method name after '.'")
            return None

        method = Identifier(self.cur_token.literal)

        if self.peek_token_is(LPAREN):
            self.next_token()
            arguments = self.parse_expression_list(RPAREN)
            return MethodCallExpression(object=left, method=method, arguments=arguments)
        else:
            return PropertyAccessExpression(object=left, property=method)

    def parse_program(self):
        program = Program()
        while not self.cur_token_is(EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                program.statements.append(stmt)
            self.next_token()
        return program

    def parse_statement(self):
        try:
            if self.cur_token_is(LET):
                return self.parse_let_statement()
            elif self.cur_token_is(RETURN):
                return self.parse_return_statement()
            elif self.cur_token_is(PRINT):
                return self.parse_print_statement()
            elif self.cur_token_is(FOR):
                return self.parse_for_each_statement()
            elif self.cur_token_is(SCREEN):
                return self.parse_screen_statement()
            elif self.cur_token_is(ACTION):
                return self.parse_action_statement()
            elif self.cur_token_is(IF):
                return self.parse_if_statement()
            elif self.cur_token_is(WHILE):
                return self.parse_while_statement()
            elif self.cur_token_is(USE):
                return self.parse_use_statement()
            elif self.cur_token_is(EXACTLY):
                return self.parse_exactly_statement()
            else:
                return self.parse_expression_statement()
        except Exception as e:
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Parse error: {str(e)}")
            self.recover_to_next_statement()
            return None

    def recover_to_next_statement(self):
        """Skip tokens until we find a statement boundary"""
        while not self.cur_token_is(EOF):
            if self.cur_token_is(SEMICOLON):
                return
            if self.peek_token_is(LET) or self.peek_token_is(RETURN) or self.peek_token_is(PRINT):
                return
            self.next_token()

    def parse_embedded_literal(self):
        """Parse: embedded {language code} """
        if not self.expect_peek(LBRACE):
            return None

        self.next_token()  # Move to token after {
        code_content = self.read_embedded_code_content()
        if code_content is None:
            return None

        lines = code_content.strip().split('\n')
        if not lines:
            self.errors.append("Empty embedded code block")
            return None

        language_line = lines[0].strip()
        language = language_line if language_line else "unknown"
        code = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ""
        return EmbeddedLiteral(language=language, code=code)

    def read_embedded_code_content(self):
        """Read content between { and } including nested braces"""
        start_position = self.lexer.position
        brace_count = 1

        while brace_count > 0 and not self.cur_token_is(EOF):
            self.next_token()
            if self.cur_token_is(LBRACE):
                brace_count += 1
            elif self.cur_token_is(RBRACE):
                brace_count -= 1

        if self.cur_token_is(EOF):
            self.errors.append("Unclosed embedded code block")
            return None

        end_position = self.lexer.position - len(self.cur_token.literal)
        content = self.lexer.input[start_position:end_position].strip()
        return content

    def parse_exactly_statement(self):
        """Parse: exactly block_name { ... } """
        if not self.expect_peek(IDENT):
            return None

        name = Identifier(self.cur_token.literal)

        if not self.expect_peek(LBRACE):
            return None

        body = self.parse_block_statement()
        return ExactlyStatement(name=name, body=body)

    def parse_for_each_statement(self):
        stmt = ForEachStatement(item=None, iterable=None, body=None)

        if not self.expect_peek(EACH):
            self.errors.append("Expected 'each' after 'for' in for-each loop")
            return None

        if not self.expect_peek(IDENT):
            self.errors.append("Expected identifier after 'each' in for-each loop")
            return None

        stmt.item = Identifier(value=self.cur_token.literal)

        if not self.expect_peek(IN):
            self.errors.append("Expected 'in' after item identifier in for-each loop")
            return None

        self.next_token()
        stmt.iterable = self.parse_expression(LOWEST)

        if not self.expect_peek(COLON):
            self.errors.append("Expected ':' after iterable in for-each loop")
            return None

        body = BlockStatement()
        self.next_token()
        stmt_body = self.parse_statement()
        if stmt_body:
            body.statements.append(stmt_body)

        stmt.body = body
        return stmt

    def parse_action_statement(self):
        if not self.expect_peek(IDENT):
            self.errors.append("Expected function name after 'action'")
            return None

        name = Identifier(self.cur_token.literal)

        if not self.expect_peek(LPAREN):
            self.errors.append("Expected '(' after function name")
            return None

        parameters = self.parse_action_parameters()
        if parameters is None:
            return None

        if not self.expect_peek(COLON):
            self.errors.append("Expected ':' after function parameters")
            return None

        body = BlockStatement()
        self.next_token()
        stmt = self.parse_statement()
        if stmt:
            body.statements.append(stmt)

        return ActionStatement(name=name, parameters=parameters, body=body)

    def parse_action_parameters(self):
        params = []
        if self.peek_token_is(RPAREN):
            self.next_token()
            return params

        self.next_token()
        if not self.cur_token_is(IDENT):
            self.errors.append("Expected parameter name")
            return None

        params.append(Identifier(self.cur_token.literal))

        while self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            if not self.cur_token_is(IDENT):
                self.errors.append("Expected parameter name after comma")
                return None
            params.append(Identifier(self.cur_token.literal))

        if not self.expect_peek(RPAREN):
            self.errors.append("Expected ')' after parameters")
            return None

        return params

    def parse_action_literal(self):
        if not self.expect_peek(LPAREN):
            return None

        parameters = self.parse_action_parameters()
        if parameters is None:
            return None

        if not self.expect_peek(COLON):
            return None

        body = BlockStatement()
        self.next_token()
        stmt = self.parse_statement()
        if stmt:
            body.statements.append(stmt)

        return ActionLiteral(parameters=parameters, body=body)

    def parse_if_statement(self):
        if not self.expect_peek(LPAREN):
            self.errors.append("Expected '(' after 'if'")
            return None

        self.next_token()
        condition = self.parse_expression(LOWEST)

        if not self.expect_peek(RPAREN):
            self.errors.append("Expected ')' after if condition")
            return None

        if not self.expect_peek(COLON):
            self.errors.append("Expected ':' after if condition")
            return None

        consequence = BlockStatement()
        self.next_token()
        stmt = self.parse_statement()
        if stmt:
            consequence.statements.append(stmt)

        alternative = None
        if self.peek_token_is(ELSE):
            self.next_token()
            if self.peek_token_is(IF):
                self.next_token()
                alternative = self.parse_if_statement()
            else:
                if not self.expect_peek(COLON):
                    self.errors.append("Expected ':' after 'else'")
                    return None
                alternative = BlockStatement()
                self.next_token()
                stmt = self.parse_statement()
                if stmt:
                    alternative.statements.append(stmt)

        return IfStatement(condition=condition, consequence=consequence, alternative=alternative)

    def parse_while_statement(self):
        if not self.expect_peek(LPAREN):
            self.errors.append("Expected '(' after 'while'")
            return None

        self.next_token()
        condition = self.parse_expression(LOWEST)

        if not self.expect_peek(RPAREN):
            self.errors.append("Expected ')' after while condition")
            return None

        if not self.expect_peek(COLON):
            self.errors.append("Expected ':' after while condition")
            return None

        body = BlockStatement()
        self.next_token()
        stmt = self.parse_statement()
        if stmt:
            body.statements.append(stmt)

        return WhileStatement(condition=condition, body=body)

    def parse_use_statement(self):
        if not self.expect_peek(IDENT):
            self.errors.append("Expected embedded reference after 'use'")
            return None

        embedded_ref = Identifier(self.cur_token.literal)

        if not self.expect_peek(DOT):
            self.errors.append("Expected '.' after embedded reference")
            return None

        if not self.expect_peek(IDENT):
            self.errors.append("Expected method name after '.'")
            return None

        method = self.cur_token.literal

        if not self.expect_peek(LPAREN):
            self.errors.append("Expected '(' after method name")
            return None

        arguments = self.parse_expression_list(RPAREN)
        return UseStatement(embedded_ref, method, arguments)

    def parse_screen_statement(self):
        stmt = ScreenStatement(name=None, body=None)

        if not self.expect_peek(IDENT):
            self.errors.append("Expected screen name after 'screen'")
            return None

        stmt.name = Identifier(value=self.cur_token.literal)

        if not self.expect_peek(LBRACE):
            self.errors.append("Expected '{' after screen name")
            return None

        stmt.body = self.parse_block_statement()
        return stmt

    def parse_return_statement(self):
        stmt = ReturnStatement(return_value=None)
        self.next_token()
        stmt.return_value = self.parse_expression(LOWEST)
        return stmt

    def parse_let_statement(self):
        stmt = LetStatement(name=None, value=None)

        if not self.expect_peek(IDENT):
            self.errors.append("Expected variable name after 'let'")
            return None

        stmt.name = Identifier(value=self.cur_token.literal)

        if not self.expect_peek(ASSIGN):
            self.errors.append("Expected '=' after variable name")
            return None

        self.next_token()
        stmt.value = self.parse_expression(LOWEST)

        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return stmt

    def parse_print_statement(self):
        stmt = PrintStatement(value=None)
        self.next_token()
        stmt.value = self.parse_expression(LOWEST)

        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return stmt

    def parse_expression_statement(self):
        stmt = ExpressionStatement(expression=self.parse_expression(LOWEST))
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        return stmt

    def parse_expression(self, precedence):
        print(f"DEBUG: parse_expression called with precedence={precedence}")
        print(f"DEBUG: cur_token={self.cur_token}, peek_token={self.peek_token}")

        # Check if current token has a prefix parse function
        if self.cur_token.type not in self.prefix_parse_fns:
            error_msg = f"Line {self.cur_token.line}:{self.cur_token.column} - No prefix parse function for {self.cur_token.type} found"
            print(f"DEBUG: {error_msg}")
            self.errors.append(error_msg)
            return None

        prefix = self.prefix_parse_fns[self.cur_token.type]
        print(f"DEBUG: calling prefix function for {self.cur_token.type}")
        left_exp = prefix()
        print(f"DEBUG: prefix result: {left_exp}")

        if left_exp is None:
            return None

        # Continue parsing while we have higher precedence infix operators
        while (not self.peek_token_is(SEMICOLON) and 
               not self.peek_token_is(EOF) and 
               precedence <= self.peek_precedence()):

            print(f"DEBUG: Inside while loop - peek_token={self.peek_token}, peek_precedence={self.peek_precedence()}")

            # Check if the next token has an infix parse function
            if self.peek_token.type not in self.infix_parse_fns:
                print(f"DEBUG: No infix function for {self.peek_token.type}, returning left_exp")
                return left_exp

            print(f"DEBUG: Found infix function for {self.peek_token.type}")
            # Get the infix function for the peek token
            infix = self.infix_parse_fns[self.peek_token.type]

            # Advance to the infix operator
            self.next_token()
            print(f"DEBUG: Advanced to infix token: {self.cur_token}")

            # Parse the infix expression
            print(f"DEBUG: Calling infix function with left={left_exp}")
            left_exp = infix(left_exp)
            print(f"DEBUG: infix result: {left_exp}")

            if left_exp is None:
                return None

        print(f"DEBUG: parse_expression returning: {left_exp}")
        return left_exp

    def parse_identifier(self):
        result = Identifier(value=self.cur_token.literal)
        print(f"DEBUG: parse_identifier returning: {result}")
        return result

    def parse_integer_literal(self):
        try:
            result = IntegerLiteral(value=int(self.cur_token.literal))
            print(f"DEBUG: parse_integer_literal returning: {result}")
            return result
        except ValueError:
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Could not parse {self.cur_token.literal} as integer")
            return None

    def parse_float_literal(self):
        try:
            result = FloatLiteral(value=float(self.cur_token.literal))
            print(f"DEBUG: parse_float_literal returning: {result}")
            return result
        except ValueError:
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Could not parse {self.cur_token.literal} as float")
            return None

    def parse_string_literal(self):
        result = StringLiteral(value=self.cur_token.literal)
        print(f"DEBUG: parse_string_literal returning: {result}")
        return result

    def parse_boolean(self):
        result = Boolean(value=self.cur_token_is(TRUE))
        print(f"DEBUG: parse_boolean returning: {result}")
        return result

    def parse_list_literal(self):
        list_lit = ListLiteral(elements=[])
        list_lit.elements = self.parse_expression_list(RBRACKET)
        return list_lit

    def parse_map_literal(self):
        pairs = []
        self.next_token()

        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            key = self.parse_expression(LOWEST)
            if not key:
                return None

            if not self.expect_peek(COLON):
                return None

            self.next_token()
            value = self.parse_expression(LOWEST)
            if not value:
                return None

            pairs.append((key, value))

            if self.peek_token_is(COMMA):
                self.next_token()
            self.next_token()

        return MapLiteral(pairs)

    def parse_call_expression(self, function):
        exp = CallExpression(function=function, arguments=[])
        exp.arguments = self.parse_expression_list(RPAREN)
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

    def parse_grouped_expression(self):
        self.next_token()
        exp = self.parse_expression(LOWEST)
        if not self.expect_peek(RPAREN):
            return None
        return exp

    def parse_if_expression(self):
        expression = IfExpression(condition=None, consequence=None, alternative=None)

        if not self.expect_peek(LPAREN):
            return None

        self.next_token()
        expression.condition = self.parse_expression(LOWEST)

        if not self.expect_peek(RPAREN):
            return None

        if not self.expect_peek(LBRACE):
            return None

        expression.consequence = self.parse_block_statement()

        if self.peek_token_is(ELSE):
            self.next_token()
            if not self.expect_peek(LBRACE):
                return None
            expression.alternative = self.parse_block_statement()

        return expression

    def parse_block_statement(self):
        block = BlockStatement()
        self.next_token()

        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                block.statements.append(stmt)
            self.next_token()

        return block

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

    # === TOKEN UTILITIES ===
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
        self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected next token to be {t}, got {self.peek_token.type} instead")
        return False

    def peek_precedence(self):
        result = precedences.get(self.peek_token.type, LOWEST)
        print(f"DEBUG: peek_precedence for '{self.peek_token.type}' = {result}")
        return result

    def cur_precedence(self):
        result = precedences.get(self.cur_token.type, LOWEST)
        print(f"DEBUG: cur_precedence for '{self.cur_token.type}' = {result}")
        return result