## src/zexus/parser.py
from ..zexus_token import *
from ..lexer import Lexer
from ..zexus_ast import *
from .strategy_structural import StructuralAnalyzer
from .strategy_context import ContextStackParser
from ..strategy_recovery import ErrorRecoveryEngine
from ..config import config  # Import the config

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
            elif self.cur_token_is(CONST):
                return self.parse_const_statement()
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
            elif self.cur_token_is(SEAL):
                return self.parse_seal_statement()
            elif self.cur_token_is(AUDIT):
                return self.parse_audit_statement()
            elif self.cur_token_is(RESTRICT):
                return self.parse_restrict_statement()
            elif self.cur_token_is(SANDBOX):
                return self.parse_sandbox_statement()
            elif self.cur_token_is(TRAIL):
                return self.parse_trail_statement()
            elif self.cur_token_is(NATIVE):
                return self.parse_native_statement()
            elif self.cur_token_is(GC):
                return self.parse_gc_statement()
            elif self.cur_token_is(INLINE):
                return self.parse_inline_statement()
            elif self.cur_token_is(BUFFER):
                return self.parse_buffer_statement()
            elif self.cur_token_is(SIMD):
                return self.parse_simd_statement()
            elif self.cur_token_is(DEFER):
                return self.parse_defer_statement()
            elif self.cur_token_is(PATTERN):
                return self.parse_pattern_statement()
            elif self.cur_token_is(ENUM):
                return self.parse_enum_statement()
            elif self.cur_token_is(STREAM):
                return self.parse_stream_statement()
            elif self.cur_token_is(WATCH):
                return self.parse_watch_statement()
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
        """Tolerant if statement parser with elif support"""
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

        # Parse elif clauses
        elif_parts = []
        while self.cur_token_is(ELIF):
            self.next_token()  # Move past elif
            
            # Parse elif condition (with or without parentheses)
            if self.cur_token_is(LPAREN):
                self.next_token()  # Skip (
                elif_condition = self.parse_expression(LOWEST)
                if self.cur_token_is(RPAREN):
                    self.next_token()  # Skip )
            else:
                # No parentheses - parse expression directly
                elif_condition = self.parse_expression(LOWEST)
            
            if not elif_condition:
                self.errors.append("Expected condition after 'elif'")
                return None
            
            # Parse elif consequence block
            elif_consequence = self.parse_block("elif")
            if not elif_consequence:
                return None
            
            elif_parts.append((elif_condition, elif_consequence))

        # Parse else clause
        alternative = None
        if self.cur_token_is(ELSE):
            self.next_token()
            alternative = self.parse_block("else")

        return IfStatement(condition=condition, consequence=consequence, elif_parts=elif_parts, alternative=alternative)

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

    def parse_const_statement(self):
        """Tolerant const statement parser - immutable variable declaration
        
        Syntax: const NAME = value;
        """
        stmt = ConstStatement(name=None, value=None)

        if not self.expect_peek(IDENT):
            self.errors.append("Expected variable name after 'const'")
            return None

        stmt.name = Identifier(value=self.cur_token.literal)

        # TOLERANT: Allow both = and : for assignment
        if self.peek_token_is(ASSIGN) or (self.peek_token_is(COLON) and self.peek_token.literal == ":"):
            self.next_token()
        else:
            self.errors.append("Expected '=' or ':' after variable name in const declaration")
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

        names = []

        # Support multiple forms: export { a, b }, export(a, b), export a, b ; export a:b; etc.
        if self.peek_token_is(LBRACE):
            # export { a, b, c }  -- tolerant manual consumption to avoid conflicts with other parsers
            if not self.expect_peek(LBRACE):
                return None
            # move into the first token inside the braces
            self.next_token()
            while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
                if self.cur_token_is(IDENT):
                    names.append(Identifier(self.cur_token.literal))
                # consume separators if present
                if self.peek_token_is(COMMA) or self.peek_token_is(SEMICOLON) or self.peek_token_is(COLON):
                    self.next_token()  # move to separator
                    self.next_token()  # move to token after separator
                    continue
                # otherwise advance
                self.next_token()
            # ensure we've consumed the closing brace
            if not self.cur_token_is(RBRACE):
                self.errors.append(f"Line {token.line}:{token.column} - Unterminated export block")
                return None

        elif self.peek_token_is(LPAREN):
            # export(a, b) -- tolerant manual parsing of identifiers
            if not self.expect_peek(LPAREN):
                return None
            # move into first token inside parens
            self.next_token()
            while not self.cur_token_is(RPAREN) and not self.cur_token_is(EOF):
                if self.cur_token_is(IDENT):
                    names.append(Identifier(self.cur_token.literal))
                if self.peek_token_is(COMMA) or self.peek_token_is(SEMICOLON) or self.peek_token_is(COLON):
                    self.next_token()
                    self.next_token()
                    continue
                self.next_token()
            if not self.cur_token_is(RPAREN):
                self.errors.append(f"Line {token.line}:{token.column} - Unterminated export(...)")
                return None

        else:
            # Single identifier or comma/sep separated list without braces
            if not self.expect_peek(IDENT):
                self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'export'")
                return None
            names.append(Identifier(self.cur_token.literal))
            # allow subsequent separators
            while self.peek_token_is(COMMA) or self.peek_token_is(SEMICOLON) or self.peek_token_is(COLON):
                self.next_token()
                if not self.expect_peek(IDENT):
                    self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after separator in export")
                    return None
                names.append(Identifier(self.cur_token.literal))

        # After names, optionally parse `to` allowed_files and `with` permission
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

        return ExportStatement(names=names, allowed_files=allowed_files, permission=permission)

    def parse_seal_statement(self):
        """Parse seal statement to mark a variable/object as immutable.
        
        Syntax: seal identifier
        """
        token = self.cur_token

        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'seal'")
            return None

        target = Identifier(self.cur_token.literal)
        return SealStatement(target=target)

    def parse_audit_statement(self):
        """Parse audit statement for compliance logging.
        
        Syntax: audit data_name, "action_type", [optional_timestamp];
        Examples:
            audit user_data, "access", timestamp;
            audit CONFIG, "modification", now;
        """
        token = self.cur_token

        # Expect identifier (data to audit)
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'audit'")
            return None

        data_name = Identifier(self.cur_token.literal)

        # Expect comma
        if not self.expect_peek(COMMA):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected ',' after data identifier in audit statement")
            return None

        # Expect action type (string literal)
        if not self.expect_peek(STRING):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected action type string in audit statement")
            return None

        action_type = StringLiteral(self.cur_token.literal)

        # Optional: timestamp
        timestamp = None
        if self.peek_token_is(COMMA):
            self.next_token()  # consume comma
            if not self.expect_peek(IDENT):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected timestamp identifier after comma in audit statement")
                return None
            timestamp = Identifier(self.cur_token.literal)

        # Expect semicolon
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return AuditStatement(data_name=data_name, action_type=action_type, timestamp=timestamp)

    def parse_restrict_statement(self):
        """Parse restrict statement for field-level access control.
        
        Syntax: restrict obj.field = "restriction_type";
        Examples:
            restrict user.password = "deny";
            restrict config.api_key = "admin-only";
            restrict data.sensitive = "read-only";
        """
        token = self.cur_token

        # Expect identifier.field pattern
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'restrict'")
            return None

        obj_name = Identifier(self.cur_token.literal)

        # Expect dot
        if not self.expect_peek(DOT):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '.' after identifier in restrict statement")
            return None

        # Expect field name
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected field name after '.' in restrict statement")
            return None

        field_name = Identifier(self.cur_token.literal)
        target = PropertyAccessExpression(obj_name, field_name)

        # Expect assignment
        if not self.expect_peek(ASSIGN):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '=' in restrict statement")
            return None

        # Expect restriction type (string literal)
        if not self.expect_peek(STRING):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected restriction type string in restrict statement")
            return None

        restriction_type = StringLiteral(self.cur_token.literal)

        # Expect semicolon
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return RestrictStatement(target=target, restriction_type=restriction_type)

    def parse_sandbox_statement(self):
        """Parse sandbox statement for isolated execution environments.
        
        Syntax: sandbox { code }
        Example:
            sandbox {
              let result = unsafe_operation();
              let data = risky_function();
            }
        """
        token = self.cur_token

        policy_name = None

        # Optional policy in parentheses: sandbox (policy = "name") { ... } or sandbox ("name") { ... }
        if self.peek_token_is(LPAREN):
            self.next_token()  # consume LPAREN
            # Accept either IDENT ASSIGN STRING or just STRING
            if self.peek_token_is(IDENT):
                self.next_token()
                if self.cur_token_is(IDENT) and self.peek_token_is(ASSIGN):
                    # consume ASSIGN
                    self.next_token()
                    if not self.expect_peek(STRING):
                        self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected policy string in sandbox()")
                        return None
                    policy_name = self.cur_token.literal
                else:
                    # treat the identifier as policy name
                    policy_name = self.cur_token.literal
            elif self.peek_token_is(STRING):
                self.next_token()
                policy_name = self.cur_token.literal
            else:
                # tolerate empty or unexpected
                pass

            # expect closing paren
            if not self.expect_peek(RPAREN):
                self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after sandbox policy")
                return None

        # Expect opening brace
        if not self.expect_peek(LBRACE):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '{{' after 'sandbox'")
            return None

        # Parse block body
        body = self.parse_block("sandbox")
        if body is None:
            return None

        return SandboxStatement(body=body, policy=policy_name)

    def parse_trail_statement(self):
        """Parse trail statement for real-time audit/debug/print tracking.
        
        Syntax:
            trail audit;           // follow all audit events
            trail print;           // follow all print statements
            trail debug;           // follow all debug output
            trail *, "pattern";    // trail all with filter
        """
        token = self.cur_token

        # Expect trail type (audit, print, debug, or *)
        if not self.expect_peek(IDENT):
            if not self.cur_token_is(STAR):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected trail type (audit, print, debug, or *)")
                return None
            trail_type = "*"
        else:
            trail_type = self.cur_token.literal

        # Optional filter
        filter_key = None
        if self.peek_token_is(COMMA):
            self.next_token()  # consume comma
            if not self.expect_peek(STRING):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected filter string after comma in trail statement")
                return None
            filter_key = StringLiteral(self.cur_token.literal)

        # Expect semicolon
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return TrailStatement(trail_type=trail_type, filter_key=filter_key)

    def parse_native_statement(self):
        """Parse native statement for calling C/C++ code.
        
        Syntax:
            native "libmath.so", "add_numbers"(x, y);
            native "libcrypto.so", "sha256"(data) as hash;
        """
        token = self.cur_token

        # Expect library name (string)
        if not self.expect_peek(STRING):
            self.errors.append(f"Line {token.line}:{token.column} - Expected library name string after 'native'")
            return None
        library_name = self.cur_token.literal

        # Expect comma
        if not self.expect_peek(COMMA):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ',' after library name in native statement")
            return None

        # Expect function name (string)
        if not self.expect_peek(STRING):
            self.errors.append(f"Line {token.line}:{token.column} - Expected function name string after comma in native statement")
            return None
        function_name = self.cur_token.literal

        # Expect opening paren for arguments
        if not self.expect_peek(LPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '(' after function name in native statement")
            return None

        # Parse arguments
        args = []
        if not self.peek_token_is(RPAREN):
            self.next_token()
            args.append(self.parse_expression(LOWEST))
            while self.peek_token_is(COMMA):
                self.next_token()
                self.next_token()
                args.append(self.parse_expression(LOWEST))

        # Expect closing paren
        if not self.expect_peek(RPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after arguments in native statement")
            return None

        # Optional: as alias
        alias = None
        if self.peek_token_is(AS):
            self.next_token()
            if not self.expect_peek(IDENT):
                self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'as' in native statement")
                return None
            alias = self.cur_token.literal

        # Optional semicolon
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return NativeStatement(library_name, function_name, args, alias)

    def parse_gc_statement(self):
        """Parse garbage collection statement.
        
        Syntax:
            gc "collect";
            gc "pause";
            gc "resume";
        """
        token = self.cur_token

        # Expect GC action (string)
        if not self.expect_peek(STRING):
            self.errors.append(f"Line {token.line}:{token.column} - Expected GC action string after 'gc'")
            return None
        action = self.cur_token.literal

        # Optional semicolon
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return GCStatement(action)

    def parse_inline_statement(self):
        """Parse inline statement for function inlining optimization.
        
        Syntax:
            inline my_function;
            inline critical_func;
        """
        token = self.cur_token

        # Expect function name (identifier)
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected function name after 'inline'")
            return None
        function_name = self.cur_token.literal

        # Optional semicolon
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return InlineStatement(function_name)

    def parse_buffer_statement(self):
        """Parse buffer statement for direct memory access.
        
        Syntax:
            buffer my_mem = allocate(1024);
            buffer my_mem.write(0, [1, 2, 3]);
            buffer my_mem.read(0, 4);
        """
        token = self.cur_token

        # Expect buffer name (identifier)
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected buffer name after 'buffer'")
            return None
        buffer_name = self.cur_token.literal

        # Optional operation (= allocate, .write, .read, etc.)
        operation = None
        arguments = []

        if self.peek_token_is(ASSIGN):
            self.next_token()
            # Expect allocate(...) or other operation
            if not self.expect_peek(IDENT):
                self.errors.append(f"Line {token.line}:{token.column} - Expected operation after '=' in buffer statement")
                return None
            operation = self.cur_token.literal

            # Expect opening paren
            if not self.expect_peek(LPAREN):
                self.errors.append(f"Line {token.line}:{token.column} - Expected '(' after operation in buffer statement")
                return None

            # Parse arguments
            if not self.peek_token_is(RPAREN):
                self.next_token()
                arguments.append(self.parse_expression(LOWEST))
                while self.peek_token_is(COMMA):
                    self.next_token()
                    self.next_token()
                    arguments.append(self.parse_expression(LOWEST))

            # Expect closing paren
            if not self.expect_peek(RPAREN):
                self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after arguments in buffer statement")
                return None

        elif self.peek_token_is(DOT):
            self.next_token()
            # Expect method name (read, write, free, etc.)
            if not self.expect_peek(IDENT):
                self.errors.append(f"Line {token.line}:{token.column} - Expected method name after '.' in buffer statement")
                return None
            operation = self.cur_token.literal

            # Expect opening paren
            if not self.expect_peek(LPAREN):
                self.errors.append(f"Line {token.line}:{token.column} - Expected '(' after method in buffer statement")
                return None

            # Parse arguments
            if not self.peek_token_is(RPAREN):
                self.next_token()
                arguments.append(self.parse_expression(LOWEST))
                while self.peek_token_is(COMMA):
                    self.next_token()
                    self.next_token()
                    arguments.append(self.parse_expression(LOWEST))

            # Expect closing paren
            if not self.expect_peek(RPAREN):
                self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after arguments in buffer statement")
                return None

        # Optional semicolon
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return BufferStatement(buffer_name, operation, arguments)

    def parse_simd_statement(self):
        """Parse SIMD statement for vector operations.
        
        Syntax:
            simd vector1 + vector2;
            simd matrix_mul(A, B);
            simd dot_product([1,2,3], [4,5,6]);
        """
        token = self.cur_token

        # Parse SIMD operation expression
        self.next_token()
        operation = self.parse_expression(LOWEST)

        if operation is None:
            self.errors.append(f"Line {token.line}:{token.column} - Expected expression in SIMD statement")
            return None

        # Optional semicolon
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return SIMDStatement(operation)

    def parse_defer_statement(self):
        """Parse defer statement - cleanup code execution.
        
        Syntax:
            defer close_file();
            defer cleanup();
            defer { cleanup1(); cleanup2(); }
        """
        token = self.cur_token

        # Parse deferred code (expression or block)
        if self.peek_token_is(LBRACE):
            self.next_token()
            code_block = self.parse_block("defer")
        else:
            self.next_token()
            code_block = self.parse_expression(LOWEST)

        if code_block is None:
            self.errors.append(f"Line {token.line}:{token.column} - Expected code after 'defer'")
            return None

        # Optional semicolon
        if self.peek_token_is(SEMICOLON):
            self.next_token()

        return DeferStatement(code_block)

    def parse_pattern_statement(self):
        """Parse pattern statement - pattern matching.
        
        Syntax:
            pattern value {
              case 1 => print "one";
              case 2 => print "two";
              default => print "other";
            }
        """
        token = self.cur_token

        # Parse expression to match against
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'pattern'")
            return None
        expression = Identifier(self.cur_token.literal)

        # Expect opening brace
        if not self.expect_peek(LBRACE):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '{{' after pattern expression")
            return None

        # Parse pattern cases
        cases = []
        self.next_token()

        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            # Expect 'case' or 'default'
            if self.cur_token.literal == "case":
                self.next_token()
                pattern = self.parse_expression(LOWEST)
                
                # Expect '=>'
                if not self.expect_peek(ASSIGN):  # Using = as stand-in for =>
                    self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '=>' in pattern case")
                    return None
                
                self.next_token()
                action = self.parse_expression(LOWEST)
                
                cases.append(PatternCase(pattern, action))
                
                # Optional semicolon
                if self.peek_token_is(SEMICOLON):
                    self.next_token()
            
            elif self.cur_token.literal == "default":
                self.next_token()
                
                # Expect '=>'
                if not self.expect_peek(ASSIGN):
                    self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '=>' in default case")
                    return None
                
                self.next_token()
                action = self.parse_expression(LOWEST)
                
                cases.append(PatternCase("default", action))
                
                # Optional semicolon
                if self.peek_token_is(SEMICOLON):
                    self.next_token()
                
                break  # Default should be last
            
            self.next_token()

        # Expect closing brace
        if not self.cur_token_is(RBRACE):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '}}' after pattern cases")
            return None

        return PatternStatement(expression, cases)

    def parse_enum_statement(self):
        """Parse enum statement - type-safe enumerations.
        
        Syntax:
            enum Color {
              Red,
              Green,
              Blue
            }
            
            enum Status {
              Active = 1,
              Inactive = 2
            }
        """
        token = self.cur_token

        # Expect enum name
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected enum name after 'enum'")
            return None
        enum_name = self.cur_token.literal

        # Expect opening brace
        if not self.expect_peek(LBRACE):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '{{' after enum name")
            return None

        # Parse enum members
        members = []
        self.next_token()

        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            if self.cur_token_is(IDENT):
                member_name = self.cur_token.literal
                member_value = None

                # Optional: = value
                if self.peek_token_is(ASSIGN):
                    self.next_token()
                    self.next_token()
                    if self.cur_token_is(INT):
                        member_value = int(self.cur_token.literal)
                    elif self.cur_token_is(STRING):
                        member_value = self.cur_token.literal

                members.append(EnumMember(member_name, member_value))

                # Skip comma and continue
                if self.peek_token_is(COMMA):
                    self.next_token()

            self.next_token()

        # Expect closing brace
        if not self.cur_token_is(RBRACE):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '}}' after enum members")
            return None

        return EnumStatement(enum_name, members)

    def parse_stream_statement(self):
        """Parse stream statement - event streaming.
        
        Syntax:
            stream clicks as event => {
              print "Clicked: " + event.x;
            }
        """
        token = self.cur_token

        # Expect stream name
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected stream name after 'stream'")
            return None
        stream_name = self.cur_token.literal

        # Expect 'as'
        if not self.expect_peek(AS):
            self.errors.append(f"Line {token.line}:{token.column} - Expected 'as' after stream name")
            return None

        # Expect event variable name
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected event variable name")
            return None
        event_var = Identifier(self.cur_token.literal)

        # Expect '=>'
        if not self.expect_peek(ASSIGN):  # Using = as stand-in for =>
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '=>' after event variable")
            return None

        # Expect block
        if not self.expect_peek(LBRACE):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '{{' for stream handler")
            return None

        handler = self.parse_block("stream")
        if handler is None:
            return None

        return StreamStatement(stream_name, event_var, handler)

    def parse_watch_statement(self):
        """Parse watch statement - reactive state management.
        
        Syntax:
            watch user_name => {
              update_ui();
            }
            
            watch count => print "Count: " + count;
        """
        token = self.cur_token

        # Parse watched expression
        self.next_token()
        watched_expr = self.parse_expression(LOWEST)

        if watched_expr is None:
            self.errors.append(f"Line {token.line}:{token.column} - Expected expression after 'watch'")
            return None

        # Expect '=>'
        if not self.expect_peek(ASSIGN):  # Using = as stand-in for =>
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '=>' in watch statement")
            return None

        # Parse reaction (block or expression)
        if self.peek_token_is(LBRACE):
            self.next_token()
            reaction = self.parse_block("watch")
        else:
            self.next_token()
            reaction = self.parse_expression(LOWEST)

        if reaction is None:
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected reaction after '=>'")
            return None

        return WatchStatement(watched_expr, reaction)

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
        """Enhanced use statement parser that handles multiple syntax styles"""
        token = self.cur_token
        
        # Check for brace syntax: use { Name1, Name2 } from './module.zx'
        if self.peek_token_is(LBRACE):
            return self.parse_use_with_braces()
        else:
            return self.parse_use_simple()

    def parse_use_with_braces(self):
        """Parse use statement with brace syntax: use { Name1, Name2 } from './module.zx'"""
        token = self.cur_token

        if not self.expect_peek(LBRACE):
            return None

        names = []

        # Parse names inside braces
        self.next_token()  # Move past {
        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            if self.cur_token_is(IDENT):
                names.append(Identifier(self.cur_token.literal))

            # Handle commas
            if self.peek_token_is(COMMA):
                self.next_token()  # Skip comma
            elif not self.peek_token_is(RBRACE):
                # If not comma or closing brace, it's probably an error but try to continue
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected ',' or '}}' in use statement")

            self.next_token()

        if not self.cur_token_is(RBRACE):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '}}' in use statement")
            return None

        # Expect 'from' after closing brace
        if not self.expect_peek(IDENT) or self.cur_token.literal != "from":
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected 'from' after import names")
            return None

        # Expect file path string
        if not self.expect_peek(STRING):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected file path after 'from'")
            return None

        file_path = self.cur_token.literal

        return UseStatement(file_path=file_path, names=names, is_named_import=True)

    def parse_use_simple(self):
        """Parse simple use statement: use './file.zx' as alias"""
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

        return UseStatement(file_path=file_path, alias=alias, is_named_import=False)

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

    def parse_entity_statement(self):
        """Parse entity declaration with maximum tolerance
        
        Supports:
        entity ZiverNode {
            rpc_server: JSONRPCServer
            ws_server: WebSocketRPCServer
            // ... other properties
        }
        """
        token = self.cur_token

        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected entity name after 'entity'")
            return None

        entity_name = Identifier(self.cur_token.literal)

        if not self.expect_peek(LBRACE):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '{{' after entity name")
            return None

        properties = []

        # Parse properties until we hit closing brace
        self.next_token()  # Move past {

        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            if self.cur_token_is(IDENT):
                prop_name = self.cur_token.literal

                # Expect colon after property name
                if not self.expect_peek(COLON):
                    self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected ':' after property name '{prop_name}'")
                    # Try to recover
                    self.recover_to_next_property()
                    continue

                self.next_token()  # Move past colon

                # Parse property type (can be identifier or built-in type)
                if self.cur_token_is(IDENT):
                    prop_type = self.cur_token.literal

                    properties.append({
                        "name": prop_name,
                        "type": prop_type
                    })

                    # Check for comma or new property
                    if self.peek_token_is(COMMA):
                        self.next_token()  # Skip comma
                    elif not self.peek_token_is(RBRACE) and self.peek_token_is(IDENT):
                        # Next property, no comma - tolerate this
                        pass

                else:
                    self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected type for property '{prop_name}'")
                    self.recover_to_next_property()
                    continue

            self.next_token()

        # Expect closing brace
        if not self.cur_token_is(RBRACE):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '}}' to close entity definition")
            # Tolerant: continue anyway
        else:
            self.next_token()  # Move past }

        return EntityStatement(name=entity_name, properties=properties)

    def recover_to_next_property(self):
        """Recover to the next property in entity definition"""
        while (not self.cur_token_is(RBRACE) and 
               not self.cur_token_is(EOF) and
               not (self.cur_token_is(IDENT) and self.peek_token_is(COLON))):
            self.next_token()

    def parse_verify_statement(self):
        """Parse verify statement
    
        verify(transfer_funds, [
            check_authenticated(),
            check_balance(amount)
        ])
        """
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