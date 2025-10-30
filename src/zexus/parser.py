# parser.py (ULTRA-FLEXIBLE COMPLETE VERSION)
from .zexus_token import *
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
            EMBEDDED: self.parse_embedded_literal,
            LAMBDA: self.parse_lambda_expression,
            DEBUG: self.parse_debug_statement,
            TRY: self.parse_try_catch_statement,
            EXTERNAL: self.parse_external_declaration,
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

    # ULTRA-FLEXIBLE: Debug statement parsing
    def parse_debug_statement(self):
        """Parse debug statements: debug expression, debug("message"), debug "message" """
        token = self.cur_token
        self.next_token()  # consume 'debug'

        # Handle debug("message") syntax
        if self.cur_token_is(LPAREN):
            self.next_token()  # consume '('
            value = self.parse_expression(LOWEST)
            if not value:
                self.errors.append(f"Line {token.line}:{token.column} - Expected expression after 'debug('")
                return None
            if not self.expect_peek(RPAREN):
                return None
        else:
            # Handle debug expression or debug "message"
            value = self.parse_expression(LOWEST)
            if not value:
                self.errors.append(f"Line {token.line}:{token.column} - Expected expression after 'debug'")
                return None

        return DebugStatement(value=value)

    # ULTRA-FLEXIBLE: Try-catch statement parsing - NOW FLEXIBLE WITH : AND {}
    def parse_try_catch_statement(self):
        """Parse try-catch statements: try { code } catch error { handle } OR try: code catch error: handle"""
        try_token = self.cur_token

        # ULTRA FLEXIBLE: try block can be { } or : or single statement
        try_block = self._parse_flexible_block("try")
        if not try_block:
            return None

        # Expect catch
        if not self.expect_peek(CATCH):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected 'catch' after try block")
            return None

        # Parse catch parameter (ULTRA FLEXIBLE: with or without parentheses, or no parameter)
        error_var = None
        if self.peek_token_is(LPAREN):
            self.next_token()  # consume 'catch'
            self.next_token()  # consume '('
            if not self.cur_token_is(IDENT):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected error variable name after 'catch('")
                return None
            error_var = Identifier(self.cur_token.literal)
            if not self.expect_peek(RPAREN):
                return None
        elif self.peek_token_is(IDENT):
            self.next_token()  # consume 'catch'
            error_var = Identifier(self.cur_token.literal)
        else:
            # Allow catch without specific error variable - create a default one
            error_var = Identifier("error")
            # Don't consume any token - we're already at the catch block

        # Parse catch block (ULTRA FLEXIBLE: { } or : or single statement)
        catch_block = self._parse_flexible_block("catch")
        if not catch_block:
            return None

        return TryCatchStatement(
            try_block=try_block,
            error_variable=error_var,
            catch_block=catch_block
        )

    # NEW: Unified flexible block parser for all control structures
    def _parse_flexible_block(self, block_type):
        """Parse blocks that can use either : or { } syntax"""
        # Check for { } block
        if self.peek_token_is(LBRACE):
            if not self.expect_peek(LBRACE):
                return None
            return self.parse_block_statement()
        # Check for : (single statement block)
        elif self.peek_token_is(COLON):
            if not self.expect_peek(COLON):
                return None
            block = BlockStatement()
            self.next_token()
            stmt = self.parse_statement()
            if stmt:
                block.statements.append(stmt)
            return block
        else:
            # Single statement without colon or brace
            block = BlockStatement()
            self.next_token()
            stmt = self.parse_statement()
            if stmt:
                block.statements.append(stmt)
            return block

    # ULTRA-FLEXIBLE: External function declaration
    def parse_external_declaration(self):
        """Parse external function declarations: external action name() from "module" """
        token = self.cur_token

        if not self.expect_peek(ACTION):
            self.errors.append(f"Line {token.line}:{token.column} - Expected 'action' after 'external'")
            return None

        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected function name after 'external action'")
            return None

        name = Identifier(self.cur_token.literal)

        # Parse parameters (ULTRA FLEXIBLE: handle all cases)
        parameters = []

        # Case 1: No parameters at all
        if self.peek_token_is(FROM) or (self.peek_token_is(MINUS) and self.lexer.peek_char() == '>'):
            # No parameters, just proceed
            pass
        # Case 2: Parameters with parentheses
        elif self.peek_token_is(LPAREN):
            self.next_token()  # consume identifier
            if not self.expect_peek(LPAREN):
                return None
            parameters = self.parse_action_parameters()
            if parameters is None:
                return None
        # Case 3: Parameters without parentheses (single parameter)
        elif self.peek_token_is(IDENT):
            self.next_token()  # consume identifier
            parameters = [Identifier(self.cur_token.literal)]

        # Expect FROM keyword (ULTRA FLEXIBLE: handle ->, from, or even no keyword)
        if self.peek_token_is(MINUS) and self.lexer.peek_char() == '>':
            # Handle -> syntax
            self.next_token()  # consume current token
            self.next_token()  # consume '>'
        elif self.peek_token_is(FROM):
            self.next_token()  # consume 'from'
        else:
            # Allow implicit "from" - just look for string
            pass

        # Expect module string
        if not self.expect_peek(STRING):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected module path string")
            return None

        module_path = self.cur_token.literal

        return ExternalDeclaration(
            name=name,
            parameters=parameters,
            module_path=module_path
        )

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
            elif self.cur_token_is(EXPORT):
                return self.parse_export_statement()
            elif self.cur_token_is(DEBUG):
                return self.parse_debug_statement()
            elif self.cur_token_is(TRY):
                return self.parse_try_catch_statement()
            elif self.cur_token_is(EXTERNAL):
                return self.parse_external_declaration()
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
            next_keywords = [LET, RETURN, PRINT, FOR, ACTION, IF, WHILE, USE, EXPORT, DEBUG, TRY, EXTERNAL]
            if any(self.peek_token_is(kw) for kw in next_keywords):
                return
            self.next_token()

    # ULTRA-FLEXIBLE: Lambda expression parsing
    def parse_lambda_expression(self):
        """Parse lambda expressions in ALL common formats"""
        token = self.cur_token
        parameters = []

        start_line = self.cur_token.line
        start_column = self.cur_token.column

        self.next_token()  # consume 'lambda'

        try:
            # Case 1: No parameters - lambda: or lambda():
            if self.cur_token_is(COLON):
                # lambda: expression
                pass
            elif self.cur_token_is(LPAREN) and self.peek_token_is(RPAREN):
                # lambda(): expression  
                self.next_token()  # consume '('
                self.next_token()  # consume ')'
            elif self.cur_token_is(LPAREN):
                # Parameters with parentheses: lambda(x), lambda(x, y)
                self.next_token()  # consume '('
                parameters = self._parse_parameter_list()
                if not self.expect_peek(RPAREN):
                    return None
            elif self.cur_token_is(IDENT):
                # Parameters without parentheses: lambda x, lambda x y
                parameters = self._parse_parameter_list()
            else:
                # Allow lambda without explicit parameters
                pass

            # Expect colon or arrow (ULTRA FLEXIBLE)
            if self.cur_token_is(COLON):
                self.next_token()  # consume ':'
            elif self.cur_token_is(MINUS) and self.peek_token_is(GT):
                self.next_token()  # consume '-'
                self.next_token()  # consume '>'
            else:
                # Try to expect colon, but don't fail if we're already at the body
                if not (isinstance(self.cur_token, Token) and self.cur_token.type in [IDENT, INT, STRING, TRUE, FALSE, LPAREN, LBRACKET, LBRACE]):
                    if not self.expect_peek(COLON):
                        # Last resort: assume we're at the body already
                        pass

            body = self.parse_expression(LOWEST)
            return LambdaExpression(parameters=parameters, body=body)

        except Exception as e:
            self.errors.append(f"Line {start_line}:{start_column} - Error parsing lambda: {str(e)}")
            return None

    def _parse_parameter_list(self):
        """Helper to parse parameter lists for lambdas and functions"""
        parameters = []

        if not self.cur_token_is(IDENT):
            return parameters

        parameters.append(Identifier(self.cur_token.literal))

        while self.peek_token_is(COMMA):
            self.next_token()  # consume identifier
            self.next_token()  # consume ','
            if self.cur_token_is(IDENT):
                parameters.append(Identifier(self.cur_token.literal))
            else:
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected parameter name")
                return parameters

        return parameters

    def parse_assignment_expression(self, left):
        """Parse assignment expressions: identifier = value"""
        if not isinstance(left, Identifier):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Cannot assign to {type(left).__name__}, only identifiers allowed")
            return None

        expression = AssignmentExpression(name=left, value=None)
        self.next_token()  # Move past the =
        expression.value = self.parse_expression(LOWEST)
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

    def parse_export_statement(self):
        """Parse export statements: export function_name [to file1, file2] [with permission]"""
        token = self.cur_token

        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'export'")
            return None

        name = Identifier(self.cur_token.literal)

        # Parse optional 'to' clause for file restrictions
        allowed_files = []
        if self.peek_token_is(IDENT) and self.peek_token.literal == "to":
            self.next_token()  # consume identifier
            self.next_token()  # consume 'to'

            # Parse file list
            if not self.peek_token_is(STRING):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected file path after 'to'")
                return None

            while self.peek_token_is(STRING):
                self.next_token()
                allowed_files.append(self.cur_token.literal)
                if self.peek_token_is(COMMA):
                    self.next_token()
                else:
                    break

        # Parse optional 'with' clause for permissions
        permission = "read_only"
        if self.peek_token_is(IDENT) and self.peek_token.literal == "with":
            self.next_token()  # consume identifier
            self.next_token()  # consume 'with'

            if self.cur_token_is(STRING):
                permission = self.cur_token.literal
            else:
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected permission string after 'with'")
                return None

        return ExportStatement(name=name, allowed_files=allowed_files, permission=permission)

    # ULTRA-FLEXIBLE: If statement parsing - NOW FLEXIBLE WITH : AND {}
    def parse_if_statement(self):
        # Handle if with or without parentheses
        if self.peek_token_is(LPAREN):
            if not self.expect_peek(LPAREN):
                return None
            self.next_token()
            condition = self.parse_expression(LOWEST)
            if not self.expect_peek(RPAREN):
                return None
        else:
            self.next_token()
            condition = self.parse_expression(LOWEST)

        if not condition:
            return None

        # Parse consequence (ULTRA FLEXIBLE: { } or : or single statement)
        consequence = self._parse_flexible_block("if consequence")
        if not consequence:
            return None

        alternative = None
        if self.peek_token_is(ELSE):
            self.next_token()
            if self.peek_token_is(IF):
                self.next_token()
                alternative = self.parse_if_statement()
            else:
                # Parse alternative (ULTRA FLEXIBLE: { } or : or single statement)
                alternative = self._parse_flexible_block("else")
                if not alternative:
                    return None

        return IfStatement(condition=condition, consequence=consequence, alternative=alternative)

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

    # ULTRA-FLEXIBLE: For-each statement - NOW FLEXIBLE WITH : AND {}
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

        # Parse body (ULTRA FLEXIBLE: { } or : or single statement)
        body = self._parse_flexible_block("for-each")
        if not body:
            return None

        stmt.body = body
        return stmt

    # ULTRA-FLEXIBLE: Action statement - NOW FLEXIBLE WITH : AND {}
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

        # Parse body (ULTRA FLEXIBLE: { } or : or single statement)
        body = self._parse_flexible_block("action")
        if not body:
            return None

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

    # ULTRA-FLEXIBLE: While statement - NOW FLEXIBLE WITH : AND {}
    def parse_while_statement(self):
        if not self.expect_peek(LPAREN):
            self.errors.append("Expected '(' after 'while'")
            return None

        self.next_token()
        condition = self.parse_expression(LOWEST)

        if not self.expect_peek(RPAREN):
            self.errors.append("Expected ')' after while condition")
            return None

        # Parse body (ULTRA FLEXIBLE: { } or : or single statement)
        body = self._parse_flexible_block("while")
        if not body:
            return None

        return WhileStatement(condition=condition, body=body)

    def parse_use_statement(self):
        if not self.expect_peek(STRING):
            self.errors.append("Expected file path after 'use'")
            return None

        file_path = self.cur_token.literal

        alias = None
        if self.peek_token_is(IDENT) and self.peek_token.literal == "as":
            self.next_token()  # consume string
            self.next_token()  # consume 'as'
            if not self.expect_peek(IDENT):
                self.errors.append("Expected alias name after 'as'")
                return None
            alias = self.cur_token.literal

        return UseStatement(file_path=file_path, alias=alias)

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
        if self.cur_token.type not in self.prefix_parse_fns:
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Unexpected token '{self.cur_token.literal}'")
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
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Could not parse {self.cur_token.literal} as integer")
            return None

    def parse_float_literal(self):
        try:
            return FloatLiteral(value=float(self.cur_token.literal))
        except ValueError:
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Could not parse {self.cur_token.literal} as float")
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
        self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected next token to be {t}, got {self.peek_token.type} instead")
        return False

    def peek_precedence(self):
        return precedences.get(self.peek_token.type, LOWEST)

    def cur_precedence(self):
        return precedences.get(self.cur_token.type, LOWEST)