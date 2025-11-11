## src/zexus/parser.py
from .zexus_token import *
from .lexer import Lexer
from .zexus_ast import *
from .strategy_structural import StructuralAnalyzer
from .strategy_context import ContextStackParser
from .strategy_recovery import ErrorRecoveryEngine
from .config import config  # Import the config

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

class UltimateParser:
    def __init__(self, lexer, syntax_style=None, enable_advanced_strategies=None):
        self.lexer = lexer
        self.syntax_style = syntax_style or config.syntax_style
        self.enable_advanced_strategies = (
            enable_advanced_strategies 
            if enable_advanced_strategies is not None 
            else config.enable_advanced_parsing
        )
        self.errors = []
        self.cur_token = None
        self.peek_token = None

        # Multi-strategy architecture
        if self.enable_advanced_strategies:
            self._log("🚀 Initializing Ultimate Parser with Multi-Strategy Architecture...", "normal")
            self.structural_analyzer = StructuralAnalyzer()
            self.context_parser = ContextStackParser(self.structural_analyzer)
            self.error_recovery = ErrorRecoveryEngine(self.structural_analyzer, self.context_parser)
            self.block_map = {}
            self.use_advanced_parsing = True
        else:
            self.use_advanced_parsing = False

        # Traditional parser setup (fallback)
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
            LBRACE: self.parse_map_literal,  # CRITICAL: This handles { } objects
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
            LAMBDA: self.parse_lambda_infix,  # support arrow-style lambdas: params => body
            LPAREN: self.parse_call_expression,
            DOT: self.parse_method_call_expression,
        }
        self.next_token()
        self.next_token()

    def _log(self, message, level="normal"):
        """Controlled logging based on config"""
        if not config.enable_debug_logs:
            return
        if level == "verbose" and config.enable_debug_logs:
            print(message)
        elif level in ["normal", "minimal"]:
            print(message)

    def parse_program(self):
        """The tolerant parsing pipeline - FIXED"""
        if not self.use_advanced_parsing:
            return self._parse_traditional()

        try:
            self._log("🎯 Starting Tolerant Parsing Pipeline...", "normal")
            
            # Phase 1: Structural Analysis
            all_tokens = self._collect_all_tokens()
            self.block_map = self.structural_analyzer.analyze(all_tokens)
            
            if config.enable_debug_logs:
                self.structural_analyzer.print_structure()

            # Phase 2: Parse ALL blocks
            program = self._parse_all_blocks_tolerantly(all_tokens)

            # Fallback if advanced parsing fails
            if len(program.statements) == 0 and len(all_tokens) > 10:
                self._log("🔄 Advanced parsing found no statements, falling back to traditional...", "normal")
                return self._parse_traditional()

            self._log(f"✅ Parsing Complete: {len(program.statements)} statements, {len(self.errors)} errors", "minimal")
            return program
            
        except Exception as e:
            self._log(f"⚠️ Advanced parsing failed, falling back to traditional: {e}", "normal")
            self.use_advanced_parsing = False
            return self._parse_traditional()

    def parse_map_literal(self):
        """FIXED: Proper map literal parsing"""
        token = self.cur_token  # Current token is LBRACE
        pairs = []

        self._log(f"🔧 Parsing map literal at line {getattr(token, 'line', 'unknown')}", "verbose")

        # Skip the opening brace (current token)
        self.next_token()

        # Handle empty map: {}
        if self.cur_token_is(RBRACE):
            self.next_token()  # Skip }
            return MapLiteral(pairs=pairs)

        # Parse key-value pairs
        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            # Parse key
            if self.cur_token_is(STRING):
                key = StringLiteral(self.cur_token.literal)
            elif self.cur_token_is(IDENT):
                key = Identifier(self.cur_token.literal)
            else:
                self.errors.append(f"Line {getattr(self.cur_token, 'line', 'unknown')}: Object key must be string or identifier, got {getattr(self.cur_token, 'type', 'UNKNOWN')}")
                return None

            # Expect colon
            if not self.expect_peek(COLON):
                return None

            # Move to value token and parse value
            self.next_token()
            value = self.parse_expression(LOWEST)
            if value is None:
                return None

            pairs.append((key, value))

            # If there's a comma, consume it and advance to next key/value
            if self.peek_token_is(COMMA):
                self.next_token()  # consume comma
                self.next_token()  # move to next key (or closing brace)
                continue

            # If next is closing brace, consume it and break
            if self.peek_token_is(RBRACE):
                self.next_token()  # consume RBRACE
                break

            # Tolerant: advance to next token to try to continue parsing
            self.next_token()

        # Final check: ensure we ended on a closing brace (tolerant)
        if not self.cur_token_is(RBRACE):
            self.errors.append(f"Line {getattr(self.cur_token, 'line', 'unknown')}: Expected '}}'")
            return None

        # Move past closing brace
        self.next_token()

        self._log(f"✅ Successfully parsed map literal with {len(pairs)} pairs", "verbose")
        return MapLiteral(pairs=pairs)

    def _collect_all_tokens(self):
        """Collect all tokens for structural analysis"""
        tokens = []
        original_position = self.lexer.position
        original_cur = self.cur_token
        original_peek = self.peek_token

        # Reset lexer to beginning
        self.lexer.position = 0
        self.lexer.read_position = 0
        self.lexer.ch = ''
        self.lexer.read_char()

        # Collect all tokens
        while True:
            token = self.lexer.next_token()
            tokens.append(token)
            if token.type == EOF:
                break

        # Restore parser state
        self.lexer.position = original_position
        self.cur_token = original_cur
        self.peek_token = original_peek

        return tokens

    def _parse_all_blocks_tolerantly(self, all_tokens):
        """Parse ALL blocks without aggressive filtering - MAXIMUM TOLERANCE"""
        program = Program()
        parsed_count = 0
        error_count = 0

        # Parse ALL top-level blocks
        top_level_blocks = [
            block_id for block_id, block_info in self.block_map.items()
            if not block_info.get('parent')  # Only top-level blocks
        ]

        self._log(f"🔧 Parsing {len(top_level_blocks)} top-level blocks...", "normal")

        for block_id in top_level_blocks:
            block_info = self.block_map[block_id]
            try:
                statement = self.context_parser.parse_block(block_info, all_tokens)
                if statement:
                    program.statements.append(statement)
                    parsed_count += 1
                    if config.enable_debug_logs:  # Only show detailed parsing in verbose mode
                        stmt_type = type(statement).__name__
                        self._log(f"  ✅ Parsed: {stmt_type} at line {block_info['start_token'].line}", "verbose")
                        
            except Exception as e:
                error_msg = f"Line {block_info['start_token'].line}: {str(e)}"
                self.errors.append(error_msg)
                error_count += 1
                self._log(f"  ❌ Parse error: {error_msg}", "normal")

        # Traditional fallback if no blocks were parsed
        if parsed_count == 0 and top_level_blocks:
            self._log("🔄 No blocks parsed with context parser, trying traditional fallback...", "normal")
            for block_id in top_level_blocks[:3]:  # Try first 3 blocks
                block_info = self.block_map[block_id]
                try:
                    block_tokens = block_info['tokens']
                    if block_tokens:
                        block_code = ' '.join([t.literal for t in block_tokens if t.literal])
                        mini_lexer = Lexer(block_code)
                        mini_parser = UltimateParser(mini_lexer, self.syntax_style, False)
                        mini_program = mini_parser.parse_program()
                        if mini_program.statements:
                            program.statements.extend(mini_program.statements)
                            parsed_count += len(mini_program.statements)
                            self._log(f"  ✅ Traditional fallback parsed {len(mini_program.statements)} statements", "normal")
                except Exception as e:
                    self._log(f"  ❌ Traditional fallback also failed: {e}", "normal")

        return program

    def _parse_traditional(self):
        """Traditional recursive descent parsing (fallback)"""
        program = Program()
        while not self.cur_token_is(EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                program.statements.append(stmt)
            self.next_token()
        return program

    # === TOLERANT PARSER METHODS ===

    def parse_statement(self):
        """Parse statement with maximum tolerance"""
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
            elif self.cur_token_is(ENTITY):
                return self.parse_entity_statement()
            elif self.cur_token_is(VERIFY):
                return self.parse_verify_statement()
            elif self.cur_token_is(CONTRACT):
                return self.parse_contract_statement()
            elif self.cur_token_is(PROTECT):
                return self.parse_protect_statement()
            else:
                return self.parse_expression_statement()
        except Exception as e:
            # TOLERANT: Don't stop execution for parse errors, just log and continue
            error_msg = f"Line {self.cur_token.line}:{self.cur_token.column} - Parse error: {str(e)}"
            self.errors.append(error_msg)
            self._log(f"⚠️  {error_msg}", "normal")

            # Try to recover and continue
            self.recover_to_next_statement()
            return None

    def parse_block(self, block_type=""):
        """Unified block parser with maximum tolerance for both syntax styles"""
        # For universal syntax, require braces
        if self.syntax_style == "universal":
            if self.peek_token_is(LBRACE):
                if not self.expect_peek(LBRACE):
                    return None
                return self.parse_brace_block()
            else:
                # In universal mode, if no brace, treat as single statement
                return self.parse_single_statement_block()

        # For tolerable/auto mode, accept both styles
        if self.peek_token_is(LBRACE):
            if not self.expect_peek(LBRACE):
                return None
            return self.parse_brace_block()
        elif self.peek_token_is(COLON):
            if not self.expect_peek(COLON):
                return None
            return self.parse_single_statement_block()
        else:
            # TOLERANT: If no block indicator, assume single statement
            return self.parse_single_statement_block()

    def parse_brace_block(self):
        """Parse { } block with tolerance for missing closing brace"""
        block = BlockStatement()
        self.next_token()

        brace_count = 1
        while brace_count > 0 and not self.cur_token_is(EOF):
            if self.cur_token_is(LBRACE):
                brace_count += 1
            elif self.cur_token_is(RBRACE):
                brace_count -= 1
                if brace_count == 0:
                    break

            stmt = self.parse_statement()
            if stmt is not None:
                block.statements.append(stmt)
            self.next_token()

        # TOLERANT: Don't error if we hit EOF without closing brace
        if self.cur_token_is(EOF) and brace_count > 0:
            self.errors.append(f"Line {self.cur_token.line}: Unclosed block (reached EOF)")

        return block

    def parse_single_statement_block(self):
        """Parse a single statement as a block"""
        block = BlockStatement()
        # Don't consume the next token if it's the end of a structure
        if not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            stmt = self.parse_statement()
            if stmt:
                block.statements.append(stmt)
        return block

    def parse_if_statement(self):
        """Tolerant if statement parser"""
        # Skip IF token
        self.next_token()

        # Parse condition (with or without parentheses)
        if self.cur_token_is(LPAREN):
            self.next_token()  # Skip (
            condition = self.parse_expression(LOWEST)
            if self.cur_token_is(RPAREN):
                self.next_token()  # Skip )
        else:
            # No parentheses - parse expression directly
            condition = self.parse_expression(LOWEST)

        if not condition:
            self.errors.append("Expected condition after 'if'")
            return None

        # Parse consequence (flexible block style)
        consequence = self.parse_block("if")
        if not consequence:
            return None

        alternative = None
        if self.cur_token_is(ELSE):
            self.next_token()
            alternative = self.parse_block("else")

        return IfStatement(condition=condition, consequence=consequence, alternative=alternative)

    def parse_action_statement(self):
        """Tolerant action parser supporting both syntax styles"""
        if not self.expect_peek(IDENT):
            self.errors.append("Expected function name after 'action'")
            return None

        name = Identifier(self.cur_token.literal)

        # Parse parameters (with or without parentheses)
        parameters = []
        if self.peek_token_is(LPAREN):
            self.next_token()  # Skip to (
            self.next_token()  # Skip (
            parameters = self.parse_action_parameters()
            if parameters is None:
                return None
        elif self.peek_token_is(IDENT):
            # Single parameter without parentheses
            self.next_token()
            parameters = [Identifier(self.cur_token.literal)]

        # Parse body (flexible style)
        body = self.parse_block("action")
        if not body:
            return None

        return ActionStatement(name=name, parameters=parameters, body=body)

    def parse_let_statement(self):
        """Tolerant let statement parser"""
        stmt = LetStatement(name=None, value=None)

        if not self.expect_peek(IDENT):
            self.errors.append("Expected variable name after 'let'")
            return None

        stmt.name = Identifier(value=self.cur_token.literal)

        # TOLERANT: Allow both = and : for assignment
        if self.peek_token_is(ASSIGN) or (self.peek_token_is(COLON) and self.peek_token.literal == ":"):
            self.next_token()
        else:
            self.errors.append("Expected '=' or ':' after variable name")
            return None

        self.next_token()
        stmt.value = self.parse_expression(LOWEST)

        # TOLERANT: Semicolon is optional
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return stmt

    def parse_print_statement(self):
        """Tolerant print statement parser"""
        stmt = PrintStatement(value=None)
        self.next_token()
        stmt.value = self.parse_expression(LOWEST)

        # TOLERANT: Semicolon is optional
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return stmt

    def parse_try_catch_statement(self):
        """Enhanced try-catch parsing with structural awareness"""
        try_token = self.cur_token
        try_block = self.parse_block("try")
        if not try_block:
            return None

        if self.cur_token_is(CATCH):
            pass
        elif not self.expect_peek(CATCH):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected 'catch' after try block")
            return None

        error_var = None
        if self.peek_token_is(LPAREN):
            self.next_token()
            self.next_token()
            if not self.cur_token_is(IDENT):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected error variable name after 'catch('")
                return None
            error_var = Identifier(self.cur_token.literal)
            if not self.expect_peek(RPAREN):
                return None
        elif self.peek_token_is(IDENT):
            self.next_token()
            error_var = Identifier(self.cur_token.literal)
        else:
            error_var = Identifier("error")

        catch_block = self.parse_block("catch")
        if not catch_block:
            return None

        return TryCatchStatement(
            try_block=try_block,
            error_variable=error_var,
            catch_block=catch_block
        )

    def parse_debug_statement(self):
        token = self.cur_token
        self.next_token()

        # TOLERANT: Accept both debug expr and debug(expr)
        if self.cur_token_is(LPAREN):
            self.next_token()
            value = self.parse_expression(LOWEST)
            if not value:
                self.errors.append(f"Line {token.line}:{token.column} - Expected expression after 'debug('")
                return None
            if self.cur_token_is(RPAREN):
                self.next_token()
        else:
            value = self.parse_expression(LOWEST)
            if not value:
                self.errors.append(f"Line {token.line}:{token.column} - Expected expression after 'debug'")
                return None

        return DebugStatement(value=value)

    def parse_external_declaration(self):
        token = self.cur_token

        if not self.expect_peek(ACTION):
            self.errors.append(f"Line {token.line}:{token.column} - Expected 'action' after 'external'")
            return None

        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected function name after 'external action'")
            return None

        name = Identifier(self.cur_token.literal)

        parameters = []
        if self.peek_token_is(LPAREN):
            self.next_token()
            if not self.expect_peek(LPAREN):
                return None
            parameters = self.parse_action_parameters()
            if parameters is None:
                return None

        if not self.expect_peek(FROM):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected 'from' after external function declaration")
            return None

        if not self.expect_peek(STRING):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected module path string")
            return None

        module_path = self.cur_token.literal

        return ExternalDeclaration(
            name=name,
            parameters=parameters,
            module_path=module_path
        )

    def recover_to_next_statement(self):
        """Tolerant error recovery"""
        while not self.cur_token_is(EOF):
            if self.cur_token_is(SEMICOLON):
                return
            next_keywords = [LET, RETURN, PRINT, FOR, ACTION, IF, WHILE, USE, EXPORT, DEBUG, TRY, EXTERNAL]
            if any(self.peek_token_is(kw) for kw in next_keywords):
                return
            self.next_token()

    def parse_lambda_expression(self):
        token = self.cur_token
        parameters = []

        self.next_token()

        if self.cur_token_is(LPAREN):
            self.next_token()
            parameters = self._parse_parameter_list()
            if not self.expect_peek(RPAREN):
                return None
        elif self.cur_token_is(IDENT):
            parameters = self._parse_parameter_list()

        # Normalize possible separators between parameter list and body
        # Accept either ':' (keyword-style), '=>' (tokenized as LAMBDA) or '->' style
        if self.cur_token_is(COLON):
            # current token is ':', advance to the first token of the body
            self.next_token()
        elif self.cur_token_is(MINUS) and self.peek_token_is(GT):
            # support '->' (legacy), consume both '-' and '>' and advance to body
            self.next_token()
            self.next_token()
        else:
            # If a colon is the *next* token (peek), consume it and advance past it
            if self.peek_token_is(COLON):
                # move to colon
                self.next_token()
                # and move to the token after colon (the start of the body)
                self.next_token()
            # Otherwise, continue — body parsing will attempt to parse the current token

        body = self.parse_expression(LOWEST)
        return LambdaExpression(parameters=parameters, body=body)

    def parse_lambda_infix(self, left):
        """Parse arrow-style lambda when encountering leftside 'params' followed by =>

        Examples:
            x => x + 1
            (a, b) => a + b
        """
        # Current token is LAMBDA because caller advanced to it
        # Build parameter list from `left` expression
        params = []
        # Single identifier param
        if isinstance(left, Identifier):
            params = [left]
        else:
            # If left is a grouped expression returning a ListLiteral-like container
            # we'll attempt to extract identifiers from it (best-effort)
            try:
                if hasattr(left, 'elements'):
                    for el in left.elements:
                        if isinstance(el, Identifier):
                            params.append(el)
            except Exception:
                pass

        # Consume the LAMBDA token (already current)
        # The parse loop has already advanced current token to LAMBDA, so
        # now move to the body
        self.next_token()

        # Support optional colon or arrow-like separators were handled at lexing stage
        if self.cur_token_is(COLON):
            self.next_token()

        body = self.parse_expression(LOWEST)
        return LambdaExpression(parameters=params, body=body)

    def _parse_parameter_list(self):
        parameters = []

        if not self.cur_token_is(IDENT):
            return parameters

        parameters.append(Identifier(self.cur_token.literal))

        while self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            if self.cur_token_is(IDENT):
                parameters.append(Identifier(self.cur_token.literal))
            else:
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected parameter name")
                return parameters

        return parameters

    def parse_assignment_expression(self, left):
        if not isinstance(left, Identifier):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Cannot assign to {type(left).__name__}, only identifiers allowed")
            return None

        expression = AssignmentExpression(name=left, value=None)
        self.next_token()
        expression.value = self.parse_expression(LOWEST)
        return expression

    def parse_method_call_expression(self, left):
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

    def parse_export_statement(self):
        token = self.cur_token

        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'export'")
            return None

        name = Identifier(self.cur_token.literal)

        allowed_files = []
        if self.peek_token_is(IDENT) and self.peek_token.literal == "to":
            self.next_token()
            self.next_token()

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

        permission = "read_only"
        if self.peek_token_is(IDENT) and self.peek_token.literal == "with":
            self.next_token()
            self.next_token()

            if self.cur_token_is(STRING):
                permission = self.cur_token.literal
            else:
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected permission string after 'with'")
                return None
        return ExportStatement(name=name, allowed_files=allowed_files, permission=permission)

    def parse_embedded_literal(self):
        if not self.expect_peek(LBRACE):
            return None

        self.next_token()
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

        body = self.parse_block("for-each")
        if not body:
            return None

        stmt.body = body
        return stmt

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

    def parse_while_statement(self):
        if not self.expect_peek(LPAREN):
            self.errors.append("Expected '(' after 'while'")
            return None

        self.next_token()
        condition = self.parse_expression(LOWEST)

        if not self.expect_peek(RPAREN):
            self.errors.append("Expected ')' after while condition")
            return None

        body = self.parse_block("while")
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
            self.next_token()
            self.next_token()
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
        # Special-case: if this parenthesized group is followed by a lambda arrow
        # treat its contents as a parameter list for an arrow-style lambda: (a, b) => ...
        # The lexer sets a hint flag when it detects a ')' followed by '=>'. Use
        # that as a fast-path check to parse the contents as parameter identifiers.
        if getattr(self.lexer, '_next_paren_has_lambda', False) or self._lookahead_token_after_matching_paren() == LAMBDA:
            # Consume '('
            self.next_token()
            self.lexer._next_paren_has_lambda = False  # Clear lexer hint after consuming parenthesis
            params = []
            # If immediate RPAREN, empty params
            if self.cur_token_is(RPAREN):
                self.next_token()
                return ListLiteral(elements=params)

            # Collect identifiers separated by commas
            if self.cur_token_is(IDENT):
                params.append(Identifier(self.cur_token.literal))

            while self.peek_token_is(COMMA):
                self.next_token()  # move to comma
                self.next_token()  # move to next identifier
                if self.cur_token_is(IDENT):
                    params.append(Identifier(self.cur_token.literal))
                else:
                    self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected parameter name")
                    break

            # Expect closing paren
            if not self.expect_peek(RPAREN):
                return None

            # Return a ListLiteral-like node carrying identifiers for lambda parsing
            return ListLiteral(elements=params)

        # Default grouped expression behavior
        self.next_token()
        exp = self.parse_expression(LOWEST)
        if not self.expect_peek(RPAREN):
            return None
        return exp

    def _lookahead_token_after_matching_paren(self):
        """Character-level lookahead: detect if the matching ')' is followed by '=>' (arrow).

        This avoids consuming parser state by scanning the lexer's input string from the
        current position and counting parentheses. It's best-effort and ignores strings
        or escapes — suitable for parameter lists which are simple identifier lists.
        """
        lexer = self.lexer
        src = getattr(lexer, 'input', '')
        pos = getattr(lexer, 'position', 0)

        i = pos
        depth = 0
        length = len(src)

        while i < length:
            ch = src[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    # look ahead for '=>' skipping whitespace
                    j = i + 1
                    while j < length and src[j].isspace():
                        j += 1
                    if j + 1 < length and src[j] == '=' and src[j + 1] == '>':
                        return LAMBDA
                    return None
            i += 1

        return None

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
        return self.parse_brace_block()

    def parse_expression_list(self, end):
        # =====================================================
        # NEW STATEMENT PARSERS: ENTITY, VERIFY, CONTRACT, PROTECT
        # =====================================================

        def parse_entity_statement(self):
            """Parse entity declaration
        
            entity User {
                name: string,
                email: string,
                role: string = "user"
            }
            """
            from .zexus_ast import EntityStatement
        
            if not self.expect_peek(IDENT):
                return None
        
            entity_name = Identifier(self.cur_token.literal)
        
            if not self.expect_peek(LBRACE):
                return None
        
            properties = []
            while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
                self.next_token()
                if self.cur_token_is(RBRACE):
                    break
            
                # Parse property: name: type [= default]
                if self.cur_token_is(IDENT):
                    prop_name = self.cur_token.literal
                
                    if self.peek_token_is(COLON):
                        self.next_token()
                        self.next_token()
                        prop_type = self.cur_token.literal
                    
                        default_value = None
                        if self.peek_token_is(ASSIGN):
                            self.next_token()
                            self.next_token()
                            default_value = self.parse_expression(LOWEST)
                    
                        properties.append({
                            "name": prop_name,
                            "type": prop_type,
                            "default_value": default_value
                        })
                
                    if self.peek_token_is(COMMA):
                        self.next_token()
        
            self.expect_peek(RBRACE)
            return EntityStatement(entity_name, properties)

        def parse_verify_statement(self):
            """Parse verify statement
        
            verify(transfer_funds, [
                check_authenticated(),
                check_balance(amount)
            ])
            """
            from .zexus_ast import VerifyStatement
        
            if not self.expect_peek(LPAREN):
                return None
        
            self.next_token()
            target = self.parse_expression(LOWEST)
        
            if not self.expect_peek(COMMA):
                return None
        
            self.next_token()
            conditions = []
        
            if self.cur_token_is(LBRACKET):
                conditions = self.parse_expression_list(RBRACKET)
            else:
                conditions.append(self.parse_expression(LOWEST))
        
            if not self.expect_peek(RPAREN):
                return None
        
            return VerifyStatement(target, conditions)

        def parse_contract_statement(self):
            """Parse contract declaration
        
            contract Token {
                persistent storage balances: Map<Address, integer>
            
                action transfer(to: Address, amount: integer) -> boolean { ... }
            }
            """
            from .zexus_ast import ContractStatement
        
            if not self.expect_peek(IDENT):
                return None
        
            contract_name = Identifier(self.cur_token.literal)
        
            if not self.expect_peek(LBRACE):
                return None
        
            storage_vars = []
            actions = []
        
            while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
                self.next_token()
            
                if self.cur_token_is(RBRACE):
                    break
            
                # Check for persistent storage declaration
                if self.cur_token_is(IDENT) and self.cur_token.literal == "persistent":
                    self.next_token()
                    if self.cur_token_is(IDENT) and self.cur_token.literal == "storage":
                        self.next_token()
                        if self.cur_token_is(IDENT):
                            storage_name = self.cur_token.literal
                            storage_vars.append({"name": storage_name})
            
                # Check for action definition
                elif self.cur_token_is(ACTION):
                    action = self.parse_action_statement()
                    if action:
                        actions.append(action)
        
            self.expect_peek(RBRACE)
            return ContractStatement(contract_name, storage_vars, actions)

        def parse_protect_statement(self):
            """Parse protect statement
        
            protect(app, {
                rate_limit: 100,
                auth_required: true,
                require_https: true
            })
            """
            from .zexus_ast import ProtectStatement
        
            if not self.expect_peek(LPAREN):
                return None
        
            self.next_token()
            target = self.parse_expression(LOWEST)
        
            if not self.expect_peek(COMMA):
                return None
        
            self.next_token()
            rules = self.parse_expression(LOWEST)  # Expect a map literal
        
            enforcement_level = "strict"
            if self.peek_token_is(COMMA):
                self.next_token()
                self.next_token()
                if self.cur_token_is(STRING):
                    enforcement_level = self.cur_token.literal
        
            if not self.expect_peek(RPAREN):
                return None
        
            return ProtectStatement(target, rules, enforcement_level)

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

# Backward compatibility
Parser = UltimateParser