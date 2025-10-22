# parser.py (CLEAN WORKING VERSION)
from zexus_token import *
from lexer import Lexer
from zexus_ast import *

LOWEST, EQUALS, LESSGREATER, SUM, PRODUCT, PREFIX, CALL = 1, 2, 3, 4, 5, 6, 7
precedences = {
    EQ: EQUALS, NOT_EQ: EQUALS, LT: LESSGREATER, GT: LESSGREATER,
    PLUS: SUM, MINUS: SUM, SLASH: PRODUCT, STAR: PRODUCT,
    LPAREN: CALL,
}


class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.errors = []
        self.cur_token = None
        self.peek_token = None

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
        }

        self.infix_parse_fns = {
            PLUS: self.parse_infix_expression,
            MINUS: self.parse_infix_expression,
            SLASH: self.parse_infix_expression,
            STAR: self.parse_infix_expression,
            EQ: self.parse_infix_expression,
            NOT_EQ: self.parse_infix_expression,
            LT: self.parse_infix_expression,
            GT: self.parse_infix_expression,
            LPAREN: self.parse_call_expression,
        }

        self.next_token()
        self.next_token()

    # === PROGRAM ENTRY ===
    def parse_program(self):
        program = Program()
        while not self.cur_token_is(EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                program.statements.append(stmt)
            self.next_token()
        return program

    # === STATEMENT DISPATCH ===
    def parse_statement(self):
        if self.cur_token_is(LET):
            # Check for embedded code block
            if self.peek_token_is(LBRACE):
                return self.parse_embedded_code()
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
        else:
            return self.parse_expression_statement()

    # === ACTION STATEMENTS ===
    def parse_action_statement(self):
        if not self.expect_peek(IDENT):
            return None
        name = Identifier(self.cur_token.literal)
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
        return ActionStatement(name=name, parameters=parameters, body=body)

    def parse_action_parameters(self):
        params = []
        if self.peek_token_is(RPAREN):
            self.next_token()
            return params
        self.next_token()
        if not self.cur_token_is(IDENT):
            return None
        params.append(Identifier(self.cur_token.literal))
        while self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            if not self.cur_token_is(IDENT):
                return None
            params.append(Identifier(self.cur_token.literal))
        if not self.expect_peek(RPAREN):
            return None
        return params

    # === IF STATEMENTS ===
    def parse_if_statement(self):
        if not self.expect_peek(LPAREN):
            return None
        self.next_token()
        condition = self.parse_expression(LOWEST)
        if not self.expect_peek(RPAREN):
            return None
        if not self.expect_peek(COLON):
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
                    return None
                alternative = BlockStatement()
                self.next_token()
                stmt = self.parse_statement()
                if stmt:
                    alternative.statements.append(stmt)
        return IfStatement(condition=condition, consequence=consequence, alternative=alternative)

    # === WHILE LOOPS ===
    def parse_while_statement(self):
        if not self.expect_peek(LPAREN):
            return None
        self.next_token()
        condition = self.parse_expression(LOWEST)
        if not self.expect_peek(RPAREN):
            return None
        if not self.expect_peek(COLON):
            return None
        body = BlockStatement()
        self.next_token()
        stmt = self.parse_statement()
        if stmt:
            body.statements.append(stmt)
        return WhileStatement(condition=condition, body=body)

    # === EMBEDDED CODE HANDLING ===
    def parse_embedded_code(self):
        if not self.expect_peek(IDENT):
            return None
        name = Identifier(self.cur_token.literal)
        if not self.expect_peek(LBRACE):
            return None
        self.next_token()
        if not self.cur_token_is(IDENT):
            self.errors.append("Expected language identifier after '{'")
            return None
        language = self.cur_token.literal
        code = self.read_until_brace()
        if code is None:
            return None
        return EmbeddedCodeStatement(name, language, code)

    def parse_use_statement(self):
        if not self.expect_peek(IDENT):
            return None
        embedded_ref = Identifier(self.cur_token.literal)
        if not self.expect_peek(DOT):
            return None
        if not self.expect_peek(IDENT):
            return None
        method = self.cur_token.literal
        if not self.expect_peek(LPAREN):
            return None
        arguments = self.parse_expression_list(RPAREN)
        return UseStatement(embedded_ref, method, arguments)

    def read_until_brace(self):
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
        code = self.lexer.input[start_position:self.lexer.position]
        return code.strip()

    # === OTHER STATEMENTS ===
    def parse_screen_statement(self):
        stmt = ScreenStatement(name=None, body=None)
        if not self.expect_peek(IDENT):
            return None
        stmt.name = Identifier(value=self.cur_token.literal)
        if not self.expect_peek(LBRACE):
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
            return None
        stmt.name = Identifier(value=self.cur_token.literal)
        if not self.expect_peek(ASSIGN):
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

    def parse_for_each_statement(self):
        stmt = ForEachStatement(item=None, iterable=None, body=None)
        if not self.expect_peek(EACH):
            return None
        if not self.expect_peek(IDENT):
            return None
        stmt.item = Identifier(value=self.cur_token.literal)
        if not self.expect_peek(IN):
            return None
        self.next_token()
        stmt.iterable = self.parse_expression(LOWEST)
        if not self.expect_peek(LBRACE):
            return None
        stmt.body = self.parse_block_statement()
        return stmt

    def parse_expression_statement(self):
        stmt = ExpressionStatement(expression=self.parse_expression(LOWEST))
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        return stmt

    # === EXPRESSIONS ===
    def parse_expression(self, precedence):
        if self.cur_token.type not in self.prefix_parse_fns:
            return None
        prefix = self.prefix_parse_fns[self.cur_token.type]
        left_exp = prefix()
        while not self.peek_token_is(SEMICOLON) and precedence < self.peek_precedence():
            if self.peek_token.type not in self.infix_parse_fns:
                return left_exp
            infix = self.infix_parse_fns[self.peek_token.type]
            self.next_token()
            left_exp = infix(left_exp)
        return left_exp

    def parse_identifier(self):
        return Identifier(value=self.cur_token.literal)

    def parse_integer_literal(self):
        try:
            return IntegerLiteral(value=int(self.cur_token.literal))
        except ValueError:
            self.errors.append(f"Could not parse {self.cur_token.literal} as integer")
            return None

    def parse_float_literal(self):
        try:
            return FloatLiteral(value=float(self.cur_token.literal))
        except ValueError:
            self.errors.append(f"Could not parse {self.cur_token.literal} as float")
            return None

    def parse_string_literal(self):
        return StringLiteral(value=self.cur_token.literal)

    def parse_boolean(self):
        return Boolean(value=self.cur_token_is(TRUE))

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
        self.errors.append(f"Expected next token to be {t}, got {self.peek_token.type} instead")
        return False

    def peek_precedence(self):
        return precedences.get(self.peek_token.type, LOWEST)

    def cur_precedence(self):
        return precedences.get(self.cur_token.type, LOWEST)