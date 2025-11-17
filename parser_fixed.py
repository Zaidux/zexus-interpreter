# parser.py (COMPLETE FIXED VERSION)
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
            INT: self.parse_number_literal,
            FLOAT: self.parse_number_literal,
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

    def parse_program(self):
        program = Program()
        while not self.cur_token_is(EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                program.statements.append(stmt)
            self.next_token()
        return program

    def parse_statement(self):
        print(f"DEBUG: parse_statement with token: {self.cur_token.type} -> '{self.cur_token.literal}'")
        
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
            print("DEBUG: ACTION token detected, calling parse_action_statement")
            return self.parse_action_statement()
        elif self.cur_token_is(USE):
            return self.parse_use_statement()
        elif self.cur_token_is(FROM):
            return self.parse_from_statement()
        elif self.cur_token_is(EXPORT):
            return self.parse_export_statement()
        elif self.cur_token_is(SEAL):
            return self.parse_seal_statement()
        else:
            return self.parse_expression_statement()

    def parse_action_statement(self):
        print("DEBUG: parse_action_statement called")
        # Store the action token
        action_token = self.cur_token
        
        # Parse the function name if present
        if not self.expect_peek(IDENT):
            print("DEBUG: Expected IDENT after action")
            return None
            
        func_name = Identifier(value=self.cur_token.literal)
        print(f"DEBUG: Function name: {func_name.value}")
        
        # Parse parameters
        if not self.expect_peek(LPAREN):
            print("DEBUG: Expected LPAREN after function name")
            return None
            
        self.next_token()  # Skip LPAREN
        parameters = self.parse_action_parameters()
        print(f"DEBUG: Parameters: {[p.value for p in parameters]}")
        
        # Parse function body
        if not self.expect_peek(LBRACE):
            print("DEBUG: Expected LBRACE after parameters")
            return None
            
        body = self.parse_block_statement()
        print(f"DEBUG: Body has {len(body.statements)} statements")
        
        # Create a let statement that assigns the function to the name
        action_literal = ActionLiteral(parameters=parameters, body=body)
        let_stmt = LetStatement(name=func_name, value=action_literal)
        
        print("DEBUG: Created LetStatement for function")
        return let_stmt

    def parse_action_literal(self):
        print("DEBUG: parse_action_literal called")
        lit = ActionLiteral(parameters=[], body=None)
        
        # Parse parameters
        if not self.expect_peek(LPAREN):
            return None
            
        self.next_token()  # Skip LPAREN
        lit.parameters = self.parse_action_parameters()
        
        # Parse function body  
        if not self.expect_peek(LBRACE):
            return None
            
        lit.body = self.parse_block_statement()
        return lit

    # ... REST OF THE PARSER METHODS REMAIN THE SAME ...
    # (I'm omitting them for brevity, but they should be copied from your current parser.py)
    
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

    def parse_number_literal(self):
        try:
            literal_value = self.cur_token.literal
            value = float(literal_value)
            if value.is_integer():
                return IntegerLiteral(value=int(value))
            else:
                return FloatLiteral(value=value)
        except (ValueError, TypeError) as e:
            self.errors.append(f"Could not parse '{self.cur_token.literal}' as number: {e}")
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

    def parse_action_parameters(self):
        params = []
        if self.peek_token_is(RPAREN):
            self.next_token()
            return params
        self.next_token()
        params.append(Identifier(value=self.cur_token.literal))
        while self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            params.append(Identifier(value=self.cur_token.literal))
        if not self.expect_peek(RPAREN):
            return None
        return params

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
        return False

    def peek_precedence(self):
        return precedences.get(self.peek_token.type, LOWEST)

    def cur_precedence(self):
        return precedences.get(self.cur_token.type, LOWEST)

    def parse_use_statement(self):
        # Handle simple use statement: use "module-name" [as alias]
        self.next_token()  # Skip 'use' keyword
        
        # Parse module name (as string or identifier)
        if self.cur_token_is(STRING):
            module_name = self.parse_string_literal()
        else:
            module_name = self.parse_identifier()
            
        # Check for alias with 'as' keyword
        alias = None
        if self.peek_token_is(AS):
            self.next_token()  # consume 'as'
            if not self.expect_peek(IDENT):
                return None
            alias = Identifier(value=self.cur_token.literal)
            
        return UseStatement(module_name=module_name, alias=alias)

    def parse_from_statement(self):
        # Handle from statement: from "module-name" use id1 [as alias1], id2 [as alias2], ...
        self.next_token()  # Skip 'from' keyword
        
        # Parse module name (as string or identifier)
        if self.cur_token_is(STRING):
            module_name = self.parse_string_literal()
        else:
            module_name = self.parse_identifier()
            
        # Expect 'use' keyword
        if not self.expect_peek(USE):
            return None
            
        # Parse import list
        imports = []
        while True:
            if not self.expect_peek(IDENT):
                break
                
            name = Identifier(value=self.cur_token.literal)
            alias = None
            
            # Check for alias
            if self.peek_token_is(AS):
                self.next_token()  # consume 'as'
                if not self.expect_peek(IDENT):
                    break
                alias = Identifier(value=self.cur_token.literal)
                
            imports.append((name, alias))
            
            # Continue if there's a comma
            if not self.peek_token_is(COMMA):
                break
            self.next_token()  # consume comma
            
        return FromStatement(module_name=module_name, imports=imports)

    def parse_export_statement(self):
        # Handle export statement: export let name = value | export action name() { ... }
        self.next_token()  # Skip 'export' keyword
        
        # Parse the declaration being exported
        if self.cur_token_is(LET):
            declaration = self.parse_let_statement()
        elif self.cur_token_is(ACTION):
            declaration = self.parse_action_statement()
        else:
            return None
            
        return ExportStatement(declaration=declaration)

    def parse_seal_statement(self):
        # Handle: seal <identifier | property-access>
        stmt = SealStatement(target=None)
        # move to token after 'seal'
        self.next_token()
        # parse target expression (typically an identifier or property access)
        stmt.target = self.parse_expression(LOWEST)
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        return stmt
