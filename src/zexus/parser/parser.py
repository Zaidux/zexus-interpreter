## src/zexus/parser.py
import tempfile
import os
from ..zexus_token import *
from ..lexer import Lexer
from ..zexus_ast import *
from .strategy_structural import StructuralAnalyzer
from .strategy_context import ContextStackParser
from ..strategy_recovery import ErrorRecoveryEngine
from ..config import config  # Import the config
from ..error_reporter import (
    get_error_reporter,
    SyntaxError as ZexusSyntaxError,
)

# Precedence constants
LOWEST, TERNARY, ASSIGN_PREC, NULLISH_PREC, LOGICAL, EQUALS, LESSGREATER, SUM, PRODUCT, PREFIX, CALL = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

precedences = {
    QUESTION: TERNARY,  # condition ? true : false (very low precedence)
    ASSIGN: ASSIGN_PREC,
    NULLISH: NULLISH_PREC,  # value ?? default
    OR: LOGICAL, AND: LOGICAL,
    EQ: EQUALS, NOT_EQ: EQUALS,
    LT: LESSGREATER, GT: LESSGREATER, LTE: LESSGREATER, GTE: LESSGREATER,
    PLUS: SUM, MINUS: SUM,
    SLASH: PRODUCT, STAR: PRODUCT, MOD: PRODUCT,
    LPAREN: CALL,
    LBRACKET: CALL,
    DOT: CALL,
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
        
        # Error reporter for better error messages
        self.error_reporter = get_error_reporter()
        self.filename = getattr(lexer, 'filename', '<stdin>')

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
            NULL: self.parse_null,
            THIS: self.parse_this,
            LPAREN: self.parse_grouped_expression,
            IF: self.parse_if_expression,
            LBRACKET: self.parse_list_literal,
            LBRACE: self.parse_map_literal,  # CRITICAL: This handles { } objects
            ACTION: self.parse_action_literal,
            FUNCTION: self.parse_function_literal,
            EMBEDDED: self.parse_embedded_literal,
            LAMBDA: self.parse_lambda_expression,
            DEBUG: self.parse_debug_statement,
            TRY: self.parse_try_catch_statement,
            EXTERNAL: self.parse_external_declaration,
            ASYNC: self.parse_async_expression,  # Support async <expression>
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
            QUESTION: self.parse_ternary_expression,  # condition ? true : false
            NULLISH: self.parse_nullish_expression,  # value ?? default
            ASSIGN: self.parse_assignment_expression,
            LAMBDA: self.parse_lambda_infix,  # support arrow-style lambdas: params => body
            LPAREN: self.parse_call_expression,
            LBRACKET: self.parse_index_expression,
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
    
    def _create_parse_error(self, message, suggestion=None, token=None):
        """
        Create a properly formatted parse error with context.
        
        Args:
            message: Error message
            suggestion: Optional helpful suggestion
            token: Token where error occurred (defaults to current token)
        
        Returns:
            ZexusSyntaxError ready to be raised or appended
        """
        if token is None:
            token = self.cur_token
        
        line = getattr(token, 'line', None)
        column = getattr(token, 'column', None)
        
        return self.error_reporter.report_error(
            ZexusSyntaxError,
            message,
            line=line,
            column=column,
            filename=self.filename,
            suggestion=suggestion
        )

    def parse_program(self):
        """The tolerant parsing pipeline - OPTIMIZED"""
        if not self.use_advanced_parsing:
            return self._parse_traditional()

        try:
            # OPTIMIZATION: Check if we already have tokens cached
            if not hasattr(self, '_cached_tokens'):
                self._cached_tokens = self._collect_all_tokens()
            
            all_tokens = self._cached_tokens

            # OPTIMIZATION: Only analyze structure if not done before
            if not hasattr(self, '_structure_analyzed'):
                self.block_map = self.structural_analyzer.analyze(all_tokens)
                self._structure_analyzed = True
                
                if config.enable_debug_logs:
                    self.structural_analyzer.print_structure()

            # Phase 2: Parse ALL blocks
            program = self._parse_all_blocks_tolerantly(all_tokens)

            # Fallback if advanced parsing fails
            if len(program.statements) == 0 and len(all_tokens) > 10:
                return self._parse_traditional()

            self._log(f"✅ Parsing Complete: {len(program.statements)} statements, {len(self.errors)} errors", "minimal")
            return program

        except Exception as e:
            self._log(f"⚠️ Advanced parsing failed, falling back to traditional: {e}", "normal")
            self.use_advanced_parsing = False
            return self._parse_traditional()

    def parse_map_literal(self):
        """Parse a map/object literal: { key: value, ... }"""
        # consume '{'
        self.next_token()

        pairs = []

        # Empty map
        if self.cur_token_is(RBRACE):
            self.next_token()
            return MapLiteral(pairs=pairs)

        # Collect key:value pairs with nesting support
        while not self.cur_token_is(EOF) and not self.cur_token_is(RBRACE):
            # Parse key
            if self.cur_token_is(STRING):
                key = StringLiteral(self.cur_token.literal)
                self.next_token()
            elif self.cur_token_is(IDENT):
                key = Identifier(self.cur_token.literal)
                self.next_token()
            else:
                # Tolerant: skip unexpected tokens until colon or next pair
                self.next_token()
                continue

            # Expect colon
            if not self.cur_token_is(COLON):
                # Tolerant: try to recover by searching forward
                while not self.cur_token_is(COLON) and not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
                    self.next_token()
                if not self.cur_token_is(COLON):
                    break

            # consume ':'
            self.next_token()

            # Parse value expression until comma or closing brace
            value_tokens = []
            nesting = 0
            while not self.cur_token_is(EOF) and not (self.cur_token_is(COMMA) and nesting == 0) and not (self.cur_token_is(RBRACE) and nesting == 0):
                if self.cur_token_is(LPAREN) or self.cur_token_is(LBRACE) or self.cur_token_is(LBRACKET):
                    nesting += 1
                elif self.cur_token_is(RPAREN) or self.cur_token_is(RBRACE) or self.cur_token_is(RBRACKET):
                    nesting -= 1
                value_tokens.append(self.cur_token)
                self.next_token()

            value_expr = self.parse_expression(LOWEST) if value_tokens else None
            pairs.append((key, value_expr))

            # If current token is comma, consume it and continue
            if self.cur_token_is(COMMA):
                self.next_token()

        # Expect closing brace
        if not self.cur_token_is(RBRACE):
            error = self._create_parse_error(
                "Expected '}' to close map literal",
                suggestion="Make sure all opening braces { have matching closing braces }"
            )
            raise error

        # consume '}'
        self.next_token()

        return MapLiteral(pairs=pairs)

    def _collect_all_tokens(self):
        """Collect all tokens for structural analysis - OPTIMIZED"""
        tokens = []
        original_position = self.lexer.position
        original_cur = self.cur_token
        original_peek = self.peek_token

        # Reset lexer to beginning
        self.lexer.position = 0
        self.lexer.read_position = 0
        self.lexer.ch = ''
        self.lexer.read_char()

        # OPTIMIZATION: Pre-allocate list with reasonable capacity
        tokens = []
        
        # OPTIMIZATION: Collect all tokens without logging overhead
        max_tokens = 100000
        iteration = 0
        while iteration < max_tokens:
            iteration += 1
            token = self.lexer.next_token()
            tokens.append(token)
            if token.type == EOF:
                break
        
        if iteration >= max_tokens:
            self._log(f"⚠️ WARNING: Hit token limit ({max_tokens}), possible lexer infinite loop", "normal")

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
        # Special case: Check for async expression (async <expr>) before parsing modifiers
        # This must come FIRST before modifier parsing
        if self.cur_token_is(ASYNC) and self.peek_token and self.peek_token.type not in {ACTION, FUNCTION}:
            # This is an async expression, not a modifier
            # Parse it as an expression statement containing AsyncExpression
            from ..zexus_ast import ExpressionStatement
            expr = self.parse_expression(LOWEST)
            node = ExpressionStatement(expression=expr)
            if self.peek_token_is(SEMICOLON):
                self.next_token()
            return node
        
        # Support optional leading modifiers: e.g. `secure async action foo {}`
        modifiers = []
        if self.cur_token and self.cur_token.type in {PUBLIC, PRIVATE, SEALED, ASYNC, NATIVE, INLINE, SECURE, PURE, VIEW, PAYABLE}:
            modifiers = self._parse_modifiers()
        try:
            node = None
            if self.cur_token_is(LET):
                node = self.parse_let_statement()
            elif self.cur_token_is(CONST):
                node = self.parse_const_statement()
            elif self.cur_token_is(DATA):
                node = self.parse_data_statement()
            elif self.cur_token_is(RETURN):
                node = self.parse_return_statement()
            elif self.cur_token_is(CONTINUE):
                node = self.parse_continue_statement()
            elif self.cur_token_is(BREAK):
                node = self.parse_break_statement()
            elif self.cur_token_is(THROW):
                node = self.parse_throw_statement()
            elif self.cur_token_is(PRINT):
                node = self.parse_print_statement()
            elif self.cur_token_is(FOR):
                node = self.parse_for_each_statement()
            elif self.cur_token_is(SCREEN):
                node = self.parse_screen_statement()
            elif self.cur_token_is(ACTION):
                node = self.parse_action_statement()
            elif self.cur_token_is(FUNCTION):
                node = self.parse_function_statement()
            elif self.cur_token_is(IF):
                node = self.parse_if_statement()
            elif self.cur_token_is(WHILE):
                node = self.parse_while_statement()
            elif self.cur_token_is(USE):
                node = self.parse_use_statement()
            elif self.cur_token_is(EXACTLY):
                node = self.parse_exactly_statement()
            elif self.cur_token_is(EXPORT):
                node = self.parse_export_statement()
            elif self.cur_token_is(DEBUG):
                node = self.parse_debug_statement()
            elif self.cur_token_is(TRY):
                node = self.parse_try_catch_statement()
            elif self.cur_token_is(EXTERNAL):
                node = self.parse_external_declaration()
            elif self.cur_token_is(ENTITY):
                node = self.parse_entity_statement()
            elif self.cur_token_is(VERIFY):
                node = self.parse_verify_statement()
            elif self.cur_token_is(CONTRACT):
                node = self.parse_contract_statement()
            elif self.cur_token_is(PROTECT):
                node = self.parse_protect_statement()
            elif self.cur_token_is(SEAL):
                node = self.parse_seal_statement()
            elif self.cur_token_is(AUDIT):
                node = self.parse_audit_statement()
            elif self.cur_token_is(RESTRICT):
                node = self.parse_restrict_statement()
            elif self.cur_token_is(SANDBOX):
                node = self.parse_sandbox_statement()
            elif self.cur_token_is(TRAIL):
                node = self.parse_trail_statement()
            elif self.cur_token_is(TX):
                node = self.parse_tx_statement()
            elif self.cur_token_is(NATIVE):
                node = self.parse_native_statement()
            elif self.cur_token_is(GC):
                node = self.parse_gc_statement()
            elif self.cur_token_is(INLINE):
                node = self.parse_inline_statement()
            elif self.cur_token_is(BUFFER):
                node = self.parse_buffer_statement()
            elif self.cur_token_is(SIMD):
                node = self.parse_simd_statement()
            elif self.cur_token_is(DEFER):
                node = self.parse_defer_statement()
            elif self.cur_token_is(PATTERN):
                node = self.parse_pattern_statement()
            elif self.cur_token_is(ENUM):
                node = self.parse_enum_statement()
            elif self.cur_token_is(STREAM):
                node = self.parse_stream_statement()
            elif self.cur_token_is(WATCH):
                print(f"[PARSE_STMT] Matched WATCH", file=sys.stderr, flush=True)
                node = self.parse_watch_statement()
            elif self.cur_token_is(EMIT):
                print(f"[PARSE_STMT] Matched EMIT", file=sys.stderr, flush=True)
                node = self.parse_emit_statement()
            elif self.cur_token_is(MODIFIER):
                print(f"[PARSE_STMT] Matched MODIFIER", file=sys.stderr, flush=True)
                node = self.parse_modifier_declaration()
            # === SECURITY STATEMENT HANDLERS ===
            elif self.cur_token_is(CAPABILITY):
                print(f"[PARSE_STMT] Matched CAPABILITY", file=sys.stderr, flush=True)
                node = self.parse_capability_statement()
            elif self.cur_token_is(GRANT):
                print(f"[PARSE_STMT] Matched GRANT", file=sys.stderr, flush=True)
                node = self.parse_grant_statement()
            elif self.cur_token_is(REVOKE):
                print(f"[PARSE_STMT] Matched REVOKE", file=sys.stderr, flush=True)
                node = self.parse_revoke_statement()
            elif self.cur_token_is(VALIDATE):
                print(f"[PARSE_STMT] Matched VALIDATE", file=sys.stderr, flush=True)
                node = self.parse_validate_statement()
            elif self.cur_token_is(SANITIZE):
                print(f"[PARSE_STMT] Matched SANITIZE", file=sys.stderr, flush=True)
                node = self.parse_sanitize_statement()
            elif self.cur_token_is(INJECT):
                print(f"[PARSE_STMT] Matched INJECT", file=sys.stderr, flush=True)
                node = self.parse_inject_statement()
            elif self.cur_token_is(IMMUTABLE):
                print(f"[PARSE_STMT] Matched IMMUTABLE", file=sys.stderr, flush=True)
                node = self.parse_immutable_statement()
            # === COMPLEXITY STATEMENT HANDLERS ===
            elif self.cur_token_is(INTERFACE):
                print(f"[PARSE_STMT] Matched INTERFACE", file=sys.stderr, flush=True)
                node = self.parse_interface_statement()
            elif self.cur_token_is(TYPE_ALIAS):
                print(f"[PARSE_STMT] Matched TYPE_ALIAS", file=sys.stderr, flush=True)
                node = self.parse_type_alias_statement()
            elif self.cur_token_is(MODULE):
                print(f"[PARSE_STMT] Matched MODULE", file=sys.stderr, flush=True)
                node = self.parse_module_statement()
            elif self.cur_token_is(PACKAGE):
                print(f"[PARSE_STMT] Matched PACKAGE", file=sys.stderr, flush=True)
                node = self.parse_package_statement()
            elif self.cur_token_is(USING):
                print(f"[PARSE_STMT] Matched USING", file=sys.stderr, flush=True)
                node = self.parse_using_statement()
            elif self.cur_token_is(CHANNEL):
                print(f"[PARSE_STMT] Matched CHANNEL", file=sys.stderr, flush=True)
                node = self.parse_channel_statement()
            elif self.cur_token_is(SEND):
                print(f"[PARSE_STMT] Matched SEND", file=sys.stderr, flush=True)
                node = self.parse_send_statement()
            elif self.cur_token_is(RECEIVE):
                print(f"[PARSE_STMT] Matched RECEIVE", file=sys.stderr, flush=True)
                node = self.parse_receive_statement()
            elif self.cur_token_is(ATOMIC):
                print(f"[PARSE_STMT] Matched ATOMIC", file=sys.stderr, flush=True)
                node = self.parse_atomic_statement()
            # === BLOCKCHAIN STATEMENT HANDLERS ===
            elif self.cur_token_is(LEDGER):
                print(f"[PARSE_STMT] Matched LEDGER", file=sys.stderr, flush=True)
                node = self.parse_ledger_statement()
            elif self.cur_token_is(STATE):
                print(f"[PARSE_STMT] Matched STATE", file=sys.stderr, flush=True)
                node = self.parse_state_statement()
            # REQUIRE is now handled by ContextStackParser for enhanced syntax support
            elif self.cur_token_is(REVERT):
                print(f"[PARSE_STMT] Matched REVERT", file=sys.stderr, flush=True)
                node = self.parse_revert_statement()
            elif self.cur_token_is(LIMIT):
                print(f"[PARSE_STMT] Matched LIMIT", file=sys.stderr, flush=True)
                node = self.parse_limit_statement()
            else:
                print(f"[PARSE_STMT] No match, falling back to expression statement", file=sys.stderr, flush=True)
                node = self.parse_expression_statement()

            if node is not None:
                return attach_modifiers(node, modifiers)
            return None
        except Exception as e:
            # TOLERANT: Don't stop execution for parse errors, just log and continue
            error_msg = f"Line {self.cur_token.line}:{self.cur_token.column} - Parse error: {str(e)}"
            self.errors.append(error_msg)
            self._log(f"⚠️  {error_msg}", "normal")

            # Try to recover and continue
            self.recover_to_next_statement()
            return None

    def _parse_modifiers(self):
        """Consume consecutive modifier tokens and return a list of modifier names."""
        mods = []
        # Accept modifiers in any order until we hit a non-modifier token
        while self.cur_token and self.cur_token.type in {PUBLIC, PRIVATE, SEALED, ASYNC, NATIVE, INLINE, SECURE, PURE, VIEW, PAYABLE}:
            # store the literal (e.g. 'secure') for readability
            mods.append(self.cur_token.literal if getattr(self.cur_token, 'literal', None) else self.cur_token.type)
            self.next_token()
        return mods

    def parse_block(self, block_type=""):
        """Unified block parser with maximum tolerance for both syntax styles"""
        # For universal syntax, require braces
        if self.syntax_style == "universal":
            # Accept a brace either as the current token or the peek token
            if self.cur_token_is(LBRACE) or self.peek_token_is(LBRACE):
                # If the current token is not the brace, advance to it
                if not self.cur_token_is(LBRACE):
                    if not self.expect_peek(LBRACE):
                        return None
                return self.parse_brace_block()
            else:
                # In universal mode, if no brace, treat as single statement
                return self.parse_single_statement_block()

        # For tolerable/auto mode, accept both styles
        # Accept a brace either as the current token or the peek token
        if self.cur_token_is(LBRACE) or self.peek_token_is(LBRACE):
            if not self.cur_token_is(LBRACE):
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
        import sys
        print(f"[BLOCK_START] Entering brace block, first token: {self.cur_token.type}={repr(self.cur_token.literal)}", file=sys.stderr, flush=True)

        brace_count = 1
        stmt_count = 0
        while brace_count > 0 and not self.cur_token_is(EOF):
            if self.cur_token_is(LBRACE):
                brace_count += 1
            elif self.cur_token_is(RBRACE):
                brace_count -= 1
                if brace_count == 0:
                    break

            print(f"[BLOCK_STMT] About to parse statement {stmt_count}, token: {self.cur_token.type}={repr(self.cur_token.literal)}", file=sys.stderr, flush=True)
            stmt = self.parse_statement()
            print(f"[BLOCK_STMT] Parsed statement {stmt_count}: {type(stmt).__name__ if stmt else 'None'}", file=sys.stderr, flush=True)
            if stmt is not None:
                block.statements.append(stmt)
            self.next_token()
            stmt_count += 1

        print(f"[BLOCK_END] Finished block with {len(block.statements)} statements", file=sys.stderr, flush=True)
        # TOLERANT: Don't error if we hit EOF without closing brace
        if self.cur_token_is(EOF) and brace_count > 0:
            self.errors.append(f"Line {self.cur_token.line}: Unclosed block (reached EOF)")

        return block

    def parse_single_statement_block(self):
        """Parse a single statement as a block
        
        Note: For advanced parsing mode (default), multi-statement indented blocks
        are handled by the StructuralAnalyzer + ContextStackParser pipeline.
        This method is primarily for traditional recursive descent parsing.
        """
        block = BlockStatement()
        # Don't consume the next token if it's the end of a structure
        if not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            stmt = self.parse_statement()
            if stmt:
                block.statements.append(stmt)
        return block

    def parse_if_statement(self):
        """Tolerant if statement parser with elif support"""
        import sys
        print(f"[PARSE_IF] Starting if statement parsing", file=sys.stderr, flush=True)
        # Skip IF token
        self.next_token()

        # Parse condition (with or without parentheses)
        if self.cur_token_is(LPAREN):
            self.next_token()  # Skip (
            condition = self.parse_expression(LOWEST)
            if not self.expect_peek(RPAREN):
                # Expected closing paren after condition
                return None
        else:
            # No parentheses - parse expression directly
            condition = self.parse_expression(LOWEST)

        if not condition:
            error = self._create_parse_error(
                "Expected condition after 'if'",
                suggestion="Add a condition expression: if (condition) { ... }"
            )
            raise error

        print(f"[PARSE_IF] Parsed condition, now at token: {self.cur_token.type}={repr(self.cur_token.literal)}", file=sys.stderr, flush=True)
        # Parse consequence (flexible block style)
        consequence = self.parse_block("if")
        print(f"[PARSE_IF] Parsed consequence block, now at token: {self.cur_token.type}={repr(self.cur_token.literal)}", file=sys.stderr, flush=True)
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
                if not self.expect_peek(RPAREN):
                    # Expected closing paren after elif condition
                    return None
            else:
                # No parentheses - parse expression directly
                elif_condition = self.parse_expression(LOWEST)
            
            if not elif_condition:
                error = self._create_parse_error(
                    "Expected condition after 'elif'",
                    suggestion="Add a condition expression: elif (condition) { ... }"
                )
                raise error
            
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

        # Parse optional return type: -> type
        return_type = None
        if self.peek_token_is(MINUS):
            # Check if this is -> (return type annotation)
            self.next_token()  # Move to MINUS
            if self.peek_token_is(GT):
                self.next_token()  # Move to GT
                self.next_token()  # Move to type identifier
                if self.cur_token_is(IDENT):
                    return_type = self.cur_token.literal

        # Parse body (flexible style)
        body = self.parse_block("action")
        if not body:
            return None

        # Note: is_async will be set by parse_statement if async modifier is present
        return ActionStatement(name=name, parameters=parameters, body=body, is_async=False, return_type=return_type)

    def parse_function_statement(self):
        """Tolerant function parser supporting both syntax styles"""
        if not self.expect_peek(IDENT):
            self.errors.append("Expected function name after 'function'")
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

        # Parse optional return type: -> type
        return_type = None
        if self.peek_token_is(MINUS):
            # Check if this is -> (return type annotation)
            self.next_token()  # Move to MINUS
            if self.peek_token_is(GT):
                self.next_token()  # Move to GT
                self.next_token()  # Move to type identifier
                if self.cur_token_is(IDENT):
                    return_type = self.cur_token.literal

        # Parse body (flexible style)
        body = self.parse_block("function")
        if not body:
            return None

        return FunctionStatement(name=name, parameters=parameters, body=body, return_type=return_type)

    def parse_let_statement(self):
        """Tolerant let statement parser"""
        stmt = LetStatement(name=None, value=None)

        if not self.expect_peek(IDENT):
            error = self._create_parse_error(
                "Expected variable name after 'let'",
                suggestion="Use 'let' to declare a variable: let myVariable = value"
            )
            raise error

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

    def parse_data_statement(self):
        """Parse data statement (dataclass definition)
        
        Syntax:
            data TypeName {
                field1: type,
                field2: type = default,
                field3: type require constraint
            }
            
            data immutable Point { x: number, y: number }
            data verified Transaction { from: address, to: address }
            data Box<T> { value: T }
            data Dog extends Animal { breed: string }
            @validated data Email { address: string }
        """
        from ..zexus_ast import DataStatement, DataField, Identifier
        
        # Parse decorators before 'data' keyword (if called from decorator context)
        decorators = []
        
        # Current token is DATA
        # Check for modifiers (immutable, verified, etc.)
        modifiers = []
        
        # Look ahead for modifiers before type name
        self.next_token()  # Move past 'data'
        
        while self.cur_token and self.cur_token_is(IDENT) and self.cur_token.literal in ["immutable", "verified"]:
            modifiers.append(self.cur_token.literal)
            self.next_token()
        
        # Type name
        if not self.cur_token_is(IDENT):
            self.errors.append("Expected type name after 'data'")
            return None
        
        type_name = self.cur_token.literal
        self.next_token()
        
        # Parse generic type parameters: <T, U, V>
        type_params = []
        if self.cur_token_is(LT):
            self.next_token()  # Skip <
            
            # Parse comma-separated type parameter names
            while not self.cur_token_is(GT) and not self.cur_token_is(EOF):
                if self.cur_token_is(IDENT):
                    type_params.append(self.cur_token.literal)
                    self.next_token()
                    
                    # Check for comma or closing >
                    if self.cur_token_is(COMMA):
                        self.next_token()  # Skip comma
                    elif self.cur_token_is(GT):
                        break  # Will advance past > below
                    else:
                        self.errors.append(f"Invalid type parameters: expected ',' or '>', got {self.cur_token.type}")
                        return None
                else:
                    self.errors.append(f"Invalid type parameter: expected identifier, got {self.cur_token.type}")
                    return None
            
            if self.cur_token_is(GT):
                self.next_token()  # Skip >
        
        # Check for inheritance: extends ParentType
        parent_type = None
        if self.cur_token_is(IDENT) and self.cur_token.literal == "extends":
            self.next_token()  # Skip 'extends'
            if self.cur_token_is(IDENT):
                parent_type = self.cur_token.literal
                self.next_token()
            else:
                self.errors.append("Expected parent type name after 'extends'")
                return None
        
        # Expect opening brace
        if not self.cur_token_is(LBRACE):
            self.errors.append(f"Expected '{{' after data type name, got {self.cur_token.type}")
            return None
        
        # Parse field definitions
        fields = []
        self.next_token()  # Move past {
        
        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            
            # Skip commas and semicolons
            if self.cur_token_is(COMMA) or self.cur_token_is(SEMICOLON):
                self.next_token()
                continue
            
            # Check for decorators: @logged, @cached, etc.
            field_decorators = []
            while self.cur_token_is(AT):
                self.next_token()  # Skip @
                if self.cur_token_is(IDENT):
                    field_decorators.append(self.cur_token.literal)
                    self.next_token()
            
            # Field must start with identifier (or 'method'/'operator'/'computed' keyword)
            if not self.cur_token_is(IDENT):
                self.next_token()
                continue
            
            # Check for special field types
            if self.cur_token.literal == "computed":
                # computed area => width * height
                self.next_token()  # Skip 'computed'
                if not self.cur_token_is(IDENT):
                    self.errors.append("Expected field name after 'computed'")
                    self.next_token()
                    continue
                
                field_name = self.cur_token.literal
                self.next_token()
                
                # Expect => (LAMBDA token)
                if not self.cur_token_is(LAMBDA):
                    self.errors.append(f"Expected '=>' after computed field name, got {self.cur_token.type}")
                    self.next_token()
                    continue
                
                self.next_token()  # Skip =>
                
                # Parse the computed expression
                computed_expr = self.parse_expression(LOWEST)
                # After parse_expression, cur_token is at the last token of the expression
                # We need to move to the next token (which should be a delimiter or closing brace)
                if not self.cur_token_is(RBRACE):
                    self.next_token()
                
                field = DataField(
                    name=field_name,
                    field_type=None,
                    default_value=None,
                    constraint=None,
                    computed=computed_expr,
                    decorators=field_decorators
                )
                fields.append(field)
                # Continue to next field
                continue
                
            elif self.cur_token.literal == "method":
                # method add(x) { return this.value + x; }
                self.next_token()  # Skip 'method'
                if not self.cur_token_is(IDENT):
                    self.errors.append("Expected method name after 'method'")
                    self.next_token()
                    continue
                
                method_name = self.cur_token.literal
                self.next_token()
                
                # Parse parameters
                method_params = []
                if self.cur_token_is(LPAREN):
                    self.next_token()  # Skip (
                    while not self.cur_token_is(RPAREN) and not self.cur_token_is(EOF):
                        if self.cur_token_is(IDENT):
                            method_params.append(self.cur_token.literal)
                            self.next_token()
                            # Skip optional type annotation: : type
                            if self.cur_token_is(COLON):
                                self.next_token()  # Skip :
                                self.skip_type_annotation()
                        if self.cur_token_is(COMMA):
                            self.next_token()
                    if self.cur_token_is(RPAREN):
                        self.next_token()  # Skip )
                
                # Parse method body
                if not self.cur_token_is(LBRACE):
                    self.errors.append("Expected '{' after method parameters")
                    self.next_token()
                    continue
                
                method_body_block = self.parse_block("method")
                if method_body_block:
                    field = DataField(
                        name=method_name,
                        method_body=method_body_block.statements,
                        method_params=method_params,
                        decorators=field_decorators
                    )
                    fields.append(field)
                # After parse_block, cur_token is at method body's }, move past it
                if self.cur_token_is(RBRACE):
                    self.next_token()
                # Continue to next field
                continue
                
            elif self.cur_token.literal == "action":
                # action get_value() -> T { return this.value; }
                # Same as method, just different keyword
                self.next_token()  # Skip 'action'
                if not self.cur_token_is(IDENT):
                    self.errors.append("Expected action name after 'action'")
                    self.next_token()
                    continue
                
                action_name = self.cur_token.literal
                self.next_token()
                
                # Parse parameters (with or without parentheses, with optional type annotations)
                action_params = []
                if self.cur_token_is(LPAREN):
                    self.next_token()  # Skip (
                    while not self.cur_token_is(RPAREN) and not self.cur_token_is(EOF):
                        if self.cur_token_is(IDENT):
                            action_params.append(self.cur_token.literal)
                            self.next_token()
                            # Skip optional type annotation: : type
                            if self.cur_token_is(COLON):
                                self.next_token()  # Skip :
                                self.skip_type_annotation()
                        if self.cur_token_is(COMMA):
                            self.next_token()
                    if self.cur_token_is(RPAREN):
                        self.next_token()  # Skip )
                
                # Skip optional return type: -> type
                if self.cur_token_is(MINUS):
                    self.next_token()  # Skip -
                    if self.cur_token_is(GT):
                        self.next_token()  # Skip >
                        self.skip_type_annotation()
                
                # Parse action body
                if not self.cur_token_is(LBRACE):
                    self.errors.append("Expected '{' after action parameters")
                    self.next_token()
                    continue
                
                action_body_block = self.parse_block("action")
                if action_body_block:
                    field = DataField(
                        name=action_name,
                        method_body=action_body_block.statements,
                        method_params=action_params,
                        decorators=field_decorators
                    )
                    fields.append(field)
                # After parse_block, cur_token is at action body's }, move past it
                if self.cur_token_is(RBRACE):
                    self.next_token()
                # Continue to next field
                continue
                
            elif self.cur_token.literal == "operator":
                # operator +(other) { return Vector(this.x + other.x, this.y + other.y); }
                self.next_token()  # Skip 'operator'
                
                # Get the operator symbol
                operator_symbol = None
                if self.cur_token.type in {PLUS, MINUS, STAR, SLASH, MOD, EQ, NOT_EQ, LT, GT, LTE, GTE}:
                    operator_symbol = self.cur_token.literal
                    self.next_token()
                else:
                    self.errors.append(f"Invalid operator symbol: {self.cur_token.literal}")
                    self.next_token()
                    continue
                
                # Parse parameters
                method_params = []
                if self.cur_token_is(LPAREN):
                    self.next_token()  # Skip (
                    while not self.cur_token_is(RPAREN) and not self.cur_token_is(EOF):
                        if self.cur_token_is(IDENT):
                            method_params.append(self.cur_token.literal)
                            self.next_token()
                            # Skip optional type annotation: : type
                            if self.cur_token_is(COLON):
                                self.next_token()  # Skip :
                                self.skip_type_annotation()
                        if self.cur_token_is(COMMA):
                            self.next_token()
                    if self.cur_token_is(RPAREN):
                        self.next_token()  # Skip )
                
                # Parse operator body
                if not self.cur_token_is(LBRACE):
                    self.errors.append("Expected '{' after operator parameters")
                    self.next_token()
                    continue
                
                operator_body_block = self.parse_block("operator")
                if operator_body_block:
                    field = DataField(
                        name=f"operator_{operator_symbol}",
                        operator=operator_symbol,
                        method_body=operator_body_block.statements,
                        method_params=method_params,
                        decorators=field_decorators
                    )
                    fields.append(field)
                # After parse_block, cur_token is at operator body's }, move past it
                if self.cur_token_is(RBRACE):
                    self.next_token()
                # Continue to next field
                continue
                
            else:
                # Regular field: name: type = default require constraint
                field_name = self.cur_token.literal
                self.next_token()
                
                # Check for type annotation
                field_type = None
                if self.cur_token_is(COLON):
                    self.next_token()  # Skip :
                    if self.cur_token_is(IDENT):
                        field_type = self.cur_token.literal
                        self.next_token()
                
                # Check for default value
                default_value = None
                if self.cur_token_is(ASSIGN):
                    self.next_token()  # Skip =
                    default_value = self.parse_expression(LOWEST)
                
                # Check for constraint (require clause)
                constraint = None
                if self.cur_token_is(IDENT) and self.cur_token.literal == "require":
                    self.next_token()  # Skip 'require'
                    constraint = self.parse_expression(LOWEST)
                
                field = DataField(
                    name=field_name,
                    field_type=field_type,
                    default_value=default_value,
                    constraint=constraint,
                    decorators=field_decorators
                )
                fields.append(field)
        
        # Note: After parsing, we may be at the dataclass's closing }
        # - For methods/operators: we advanced past their body's }, so we're already positioned correctly
        # - For regular fields: we're at the dataclass's }, no need to advance (caller handles it)
        
        return DataStatement(
            name=Identifier(type_name),
            fields=fields,
            modifiers=modifiers,
            parent=parent_type,
            decorators=decorators,
            type_params=type_params
        )

    def parse_print_statement(self):
        """Tolerant print statement parser with support for:
        - Single argument: print(message)
        - Multiple arguments: print(arg1, arg2, arg3)
        - Conditional print: print(condition, message) - exactly 2 args
        """
        import sys
        # Debug logging (fail silently if file operations fail)
        try:
            log_path = os.path.join(tempfile.gettempdir(), 'parser_log.txt')
            with open(log_path, 'a') as f:
                f.write(f"===  parse_print_statement CALLED ===\n")
                f.flush()
        except (IOError, OSError, PermissionError):
            pass  # Silently ignore debug logging errors
        
        stmt = PrintStatement(values=[])
        self.next_token()
        
        # Parse first expression
        first_expr = self.parse_expression(LOWEST)
        if first_expr:
            stmt.values.append(first_expr)
        
        # Parse additional comma-separated expressions
        while self.peek_token_is(COMMA):
            self.next_token()  # consume comma
            self.next_token()  # move to next expression
            expr = self.parse_expression(LOWEST)
            if expr:
                stmt.values.append(expr)
        
        # Check if this is conditional print (exactly 2 arguments)
        if len(stmt.values) == 2:
            # Conditional print: print(condition, message)
            stmt.condition = stmt.values[0]
            stmt.values = [stmt.values[1]]
            stmt.value = stmt.values[0]
        else:
            # Regular print: print(arg) or print(arg1, arg2, arg3, ...)
            # Keep backward compatibility with .value for single-expression prints
            stmt.value = stmt.values[0] if len(stmt.values) == 1 else None

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
        """Parse debug - dual mode: statement (debug x;) or function call (debug(x) or debug(cond, x))
        
        When debug is followed by (, it's treated as a function call expression.
        Supports:
        - debug(value) - regular debug
        - debug(condition, value) - conditional debug (exactly 2 args)
        - debug value; - statement mode
        """
        
        # DUAL-MODE: If followed by (, parse as function call with potential conditional
        if self.peek_token_is(LPAREN):
            # We need to parse the function call arguments
            self.next_token()  # Move to LPAREN
            self.next_token()  # Move to first argument
            
            # Collect arguments
            args = []
            if not self.cur_token_is(RPAREN):
                arg = self.parse_expression(LOWEST)
                if arg:
                    args.append(arg)
                
                while self.peek_token_is(COMMA):
                    self.next_token()  # consume comma
                    self.next_token()  # move to next arg
                    arg = self.parse_expression(LOWEST)
                    if arg:
                        args.append(arg)
            
            # Expect closing paren
            if not self.peek_token_is(RPAREN):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected ')' after debug arguments")
                return None
            self.next_token()  # consume RPAREN
            
            # Check if conditional debug (2 args) or regular debug (1 arg)
            if len(args) == 2:
                # Conditional debug: debug(condition, value)
                return DebugStatement(value=args[1], condition=args[0])
            elif len(args) == 1:
                # Regular debug: debug(value)
                return DebugStatement(value=args[0])
            else:
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - debug() requires 1 or 2 arguments")
                return None
        
        # Otherwise, it's a debug statement (debug x;)
        self.next_token()
        value = self.parse_expression(LOWEST)
        if not value:
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected expression after 'debug'")
            return None

        return DebugStatement(value=value)

    def parse_external_declaration(self):
        token = self.cur_token

        # Support simple syntax: external identifier;
        if self.peek_token_is(IDENT):
            self.next_token()
            name = Identifier(self.cur_token.literal)
            return ExternalDeclaration(
                name=name,
                parameters=[],
                module_path=""
            )
        
        # Full syntax: external action identifier from "module";
        if not self.expect_peek(ACTION):
            self.errors.append(f"Line {token.line}:{token.column} - Expected identifier or 'action' after 'external'")
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

        # After a dot, allow keywords to be used as property/method names
        # This enables t.verify(), obj.data, etc. even though verify/data are keywords
        self.next_token()
        
        # Accept any token with a literal as a property name (IDENT or keywords)
        # This allows using reserved keywords like 'verify', 'data', 'hash', etc. as property names
        if self.cur_token.literal:
            method = Identifier(self.cur_token.literal)
        else:
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected property/method name after '.'")
            return None

        if self.peek_token_is(LPAREN):
            self.next_token()
            arguments = self.parse_expression_list(RPAREN)
            return MethodCallExpression(object=left, method=method, arguments=arguments)
        else:
            return PropertyAccessExpression(object=left, property=method)

    def parse_export_statement(self):
        token = self.cur_token

        # Check for syntactic sugar: export action name() {} or export function name() {}
        if self.peek_token_is(ACTION) or self.peek_token_is(FUNCTION):
            self.next_token()  # Move to ACTION/FUNCTION token
            
            # Parse the action/function normally
            if self.cur_token_is(ACTION):
                func_stmt = self.parse_action_statement()
            else:  # FUNCTION
                func_stmt = self.parse_function_statement()
            
            if func_stmt is None:
                return None
            
            # Extract the function name for export
            func_name = func_stmt.name.value if hasattr(func_stmt.name, 'value') else str(func_stmt.name)
            
            # Create a compound statement: the function definition + export
            # We'll use a BlockStatement to hold both
            from ..zexus_ast import BlockStatement, ExportStatement
            export_stmt = ExportStatement(names=[Identifier(func_name)])
            
            # Return a block containing both statements
            return BlockStatement(statements=[func_stmt, export_stmt])

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

    def parse_tx_statement(self):
        """Parse transaction block statement.
        
        Syntax:
            tx {
                balance = balance - amount;
                recipient_balance = recipient_balance + amount;
            }
        """
        # Consume 'tx' keyword
        self.next_token()
        
        # Expect opening brace
        if not self.cur_token_is(LBRACE):
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected '{{' after 'tx'")
            return None
        
        # Parse block body
        body = self.parse_block_statement()
        
        if body is None:
            return None
        
        return TxStatement(body=body)

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
            
            watch count => print("Count: " + count);
        """
        token = self.cur_token

        # Parse watched expression
        self.next_token()
        watched_expr = self.parse_expression(LOWEST)

        if watched_expr is None:
            self.errors.append(f"Line {token.line}:{token.column} - Expected expression after 'watch'")
            return None

        # Expect '=>' (LAMBDA token)
        if not self.expect_peek(LAMBDA):
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

        return WatchStatement(reaction=reaction, watched_expr=watched_expr)

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

        # Normalize several possible entry points: caller may call this
        # with cur_token at LPAREN, at the first parameter, or at RPAREN.
        if self.cur_token_is(LPAREN):
            # advance to the token after '('
            self.next_token()

        # If we are immediately at ')' then it's an empty parameter list
        if self.cur_token_is(RPAREN):
            self.next_token()
            return params

        # Now expect an identifier for the first parameter
        if not self.cur_token_is(IDENT):
            self.errors.append("Expected parameter name")
            return None

        params.append(Identifier(self.cur_token.literal))
        
        # Skip optional type annotation: : type
        if self.peek_token_is(COLON):
            self.next_token()  # Move to :
            self.next_token()  # Move to type (skip it)

        while self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            if not self.cur_token_is(IDENT):
                self.errors.append("Expected parameter name after comma")
                return None
            params.append(Identifier(self.cur_token.literal))
            
            # Skip optional type annotation: : type
            if self.peek_token_is(COLON):
                self.next_token()  # Move to :
                self.next_token()  # Move to type (skip it)

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

    def parse_function_literal(self):
        """Parse function literal expression: function(params) { body } or function(params) : stmt
        Returns an ActionLiteral which is compatible with function execution"""
        if not self.expect_peek(LPAREN):
            return None

        parameters = self.parse_action_parameters()
        if parameters is None:
            return None

        # After parse_action_parameters, cur_token is already at the token after ')'
        # Check for colon (traditional action-style) or curly brace (function-style)
        if self.cur_token_is(COLON):
            # Traditional action style: function(x) : stmt
            body = BlockStatement()
            self.next_token()
            stmt = self.parse_statement()
            if stmt:
                body.statements.append(stmt)

            return ActionLiteral(parameters=parameters, body=body)
        
        elif self.cur_token_is(LBRACE):
            # Function-style with braces: function(x) { stmts }
            # cur_token is already at {, so parse the brace block directly
            body = self.parse_brace_block()
            if not body:
                return None
            return ActionLiteral(parameters=parameters, body=body)
        
        # Also handle peek variants for backwards compatibility if cur_token is still at )
        elif self.peek_token_is(COLON):
            if not self.expect_peek(COLON):
                return None
            body = BlockStatement()
            self.next_token()
            stmt = self.parse_statement()
            if stmt:
                body.statements.append(stmt)
            return ActionLiteral(parameters=parameters, body=body)
        
        elif self.peek_token_is(LBRACE):
            self.next_token()  # Move to {
            body = self.parse_brace_block()
            if not body:
                return None
            return ActionLiteral(parameters=parameters, body=body)
        
        else:
            self.errors.append("Expected ':' or '{' after function parameters")
            return None

    def parse_while_statement(self):
        """Tolerant while statement parser (with or without parentheses)"""
        self.next_token()  # Move past WHILE token
        
        # Parse condition (with or without parentheses)
        if self.cur_token_is(LPAREN):
            self.next_token()  # Skip (
            condition = self.parse_expression(LOWEST)
            # After parse_expression, check if RPAREN is current or peek token
            if self.cur_token_is(RPAREN):
                self.next_token()  # Skip ) - it's already in cur_token
            elif self.peek_token_is(RPAREN):
                self.next_token()  # Advance to )
                self.next_token()  # Skip )
            else:
                self.errors.append("Expected ')' after while condition")
                return None
        else:
            # No parentheses - parse expression directly
            condition = self.parse_expression(LOWEST)
        
        if not condition:
            self.errors.append("Expected condition after 'while'")
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

    def parse_continue_statement(self):
        """Parse CONTINUE statement - enables error recovery mode."""
        stmt = ContinueStatement()
        self.next_token()  # consume CONTINUE token
        return stmt

    def parse_break_statement(self):
        """Parse BREAK statement - exits current loop."""
        stmt = BreakStatement()
        self.next_token()  # consume BREAK token
        return stmt

    def parse_throw_statement(self):
        """Parse THROW statement - throws an error."""
        self.next_token()  # consume THROW token
        # Parse error message expression
        message = self.parse_expression(LOWEST)
        stmt = ThrowStatement(message=message)
        return stmt

    def parse_expression_statement(self):
        stmt = ExpressionStatement(expression=self.parse_expression(LOWEST))
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        return stmt

    def parse_expression(self, precedence):
        # Special handling for DEBUG token in expression context
        # If DEBUG is followed by (, treat it as identifier for function call
        # Otherwise, it will be parsed as a debug statement
        if self.cur_token.type == DEBUG and self.peek_token_is(LPAREN):
            # Convert DEBUG token to identifier in function call context
            left_exp = Identifier(value="debug")
        elif self.cur_token.type not in self.prefix_parse_fns:
            self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Unexpected token '{self.cur_token.literal}'")
            return None
        else:
            prefix = self.prefix_parse_fns[self.cur_token.type]
            left_exp = prefix()

        if left_exp is None:
            return None

        # Stop parsing when we hit closing delimiters or terminators
        # This prevents the parser from trying to parse beyond expression boundaries
        while (not self.peek_token_is(SEMICOLON) and 
               not self.peek_token_is(EOF) and
               not self.peek_token_is(RPAREN) and
               not self.peek_token_is(RBRACE) and
               not self.peek_token_is(RBRACKET) and 
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
        # Allow DEBUG keyword to be used as identifier in expression contexts
        # This enables debug(value) function calls while keeping debug value; statements
        if self.cur_token.type == DEBUG:
            return Identifier(value="debug")
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
        lit = getattr(self.cur_token, 'literal', '')
        val = True if isinstance(lit, str) and lit.lower() == 'true' else False
        # Transient trace to diagnose boolean parsing
        try:
            if lit.lower() == 'false':
                import traceback as _tb
                stack = ''.join(_tb.format_stack(limit=4)[-2:])
                print(f"[PARSE_BOOL_TRACE] false token at position {self.lexer.position}: literal={lit}, val={val}\n{stack}")
        except Exception:
            pass
        return Boolean(value=val)

    def parse_null(self):
        """Parse null literal"""
        from ..zexus_ast import NullLiteral
        return NullLiteral()

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
    
    def parse_async_expression(self):
        """Parse async expression: async <expression>
        
        Example: async producer()
        This executes the expression asynchronously in a background thread.
        """
        from ..zexus_ast import AsyncExpression
        # Consume 'async' token
        self.next_token()
        # Parse the expression to execute asynchronously
        expr = self.parse_expression(PREFIX)
        return AsyncExpression(expression=expr)

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

    

        # Default grouped expression behavior
        self.next_token()
        exp = self.parse_expression(LOWEST)
        if not self.expect_peek(RPAREN):
            return None
        return exp

    def parse_index_expression(self, left):
        """Parse index expressions like obj[expr] and convert them to PropertyAccessExpression."""
        # current token is LBRACKET (parser calls this after advancing to that token)
        # Move to the first token inside the brackets
        self.next_token()
        index_expr = self.parse_expression(LOWEST)
        # Expect closing bracket
        if not self.expect_peek(RBRACKET):
            return None
        return PropertyAccessExpression(object=left, property=index_expr)

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
        """Parse if expression - handles both statement form and expression form
        
        Statement form: if (condition) { ... } else { ... }
        Expression form: if condition then value else value
        """
        expression = IfExpression(condition=None, consequence=None, alternative=None)
        
        # Check if next token is LPAREN (statement form) or not (expression form)
        if self.peek_token_is(LPAREN):
            # Statement form: if (condition) { ... }
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
        else:
            # Expression form: if condition then value else value
            self.next_token()  # Move to condition
            expression.condition = self.parse_expression(LOWEST)
            
            if expression.condition is None:
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected condition after 'if'")
                return None
            
            # Expect THEN
            if not self.expect_peek(THEN):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected 'then' after if condition")
                return None
            
            # Parse consequence expression
            self.next_token()
            consequence_exp = self.parse_expression(LOWEST)
            if consequence_exp is None:
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected expression after 'then'")
                return None
            
            # Wrap the consequence expression in an ExpressionStatement for compatibility
            from ..zexus_ast import ExpressionStatement, BlockStatement
            consequence_stmt = ExpressionStatement(expression=consequence_exp)
            consequence_block = BlockStatement()
            consequence_block.statements = [consequence_stmt]
            expression.consequence = consequence_block
            
            # Expect ELSE
            if not self.expect_peek(ELSE):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected 'else' in if-then-else expression")
                return None
            
            # Parse alternative expression
            self.next_token()
            alternative_exp = self.parse_expression(LOWEST)
            if alternative_exp is None:
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected expression after 'else'")
                return None
            
            # Wrap the alternative expression in an ExpressionStatement
            alternative_stmt = ExpressionStatement(expression=alternative_exp)
            alternative_block = BlockStatement()
            alternative_block.statements = [alternative_stmt]
            expression.alternative = alternative_block
            
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

        # Check for inheritance: extends ParentEntity
        parent = None
        if self.peek_token_is(IDENT) and self.peek_token.literal == "extends":
            self.next_token()  # Move to 'extends'
            if not self.expect_peek(IDENT):
                self.errors.append(f"Line {self.cur_token.line}:{self.cur_token.column} - Expected parent entity name after 'extends'")
                return None
            parent = Identifier(self.cur_token.literal)

        if not self.expect_peek(LBRACE):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '{{' after entity name")
            return None

        properties = []
        methods = []

        # Parse properties and methods until we hit closing brace
        self.next_token()  # Move past {

        while not self.cur_token_is(RBRACE) and not self.cur_token_is(EOF):
            # Check if this is an action/method definition
            if self.cur_token_is(ACTION) or self.cur_token_is(FUNCTION):
                method = self.parse_action_statement() if self.cur_token_is(ACTION) else self.parse_function_statement()
                if method:
                    methods.append(method)
                continue
            
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
            # Consume the closing brace
            self.next_token()

        return EntityStatement(name=entity_name, properties=properties, parent=parent, methods=methods)

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
        
        contract QuantumCrypto implements QuantumResistantCrypto { ... }
        """
        if not self.expect_peek(IDENT):
            return None

        contract_name = Identifier(self.cur_token.literal)
        
        # Check for implements clause
        implements = None
        if self.peek_token_is(IMPLEMENTS):
            self.next_token()  # consume 'implements'
            if self.expect_peek(IDENT):
                implements = Identifier(self.cur_token.literal)

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
        
        # Create body block with storage vars and actions
        body = BlockStatement()
        body.statements = storage_vars + actions
        
        return ContractStatement(contract_name, body, modifiers=[], implements=implements)

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

    def skip_type_annotation(self):
        """Skip a type annotation, handling generic types like List<T>, Map<K,V>, etc.
        
        Assumes cur_token is on the first token of the type (e.g., 'List', 'string', etc.)
        After calling, cur_token will be on the token after the complete type annotation.
        """
        if not self.cur_token_is(IDENT):
            return
        
        # Skip the type name
        self.next_token()
        
        # Check for generic type parameters: <...>
        if self.cur_token_is(LT):
            # We have a generic type, need to skip the entire <...> part
            depth = 1
            self.next_token()  # Skip <
            
            while depth > 0 and not self.cur_token_is(EOF):
                if self.cur_token_is(LT):
                    depth += 1
                elif self.cur_token_is(GT):
                    depth -= 1
                self.next_token()
            
            # Check if we exited due to EOF with unmatched brackets
            if depth > 0 and self.cur_token_is(EOF):
                self.errors.append(f"Malformed type annotation: unmatched '<' in generic type")
        
        # cur_token is now on the token after the type annotation

    # === SECURITY STATEMENT PARSERS ===

    def parse_capability_statement(self):
        """Parse capability statement - grant/check capabilities"""
        token = self.cur_token
        self.next_token()
        
        # capability name { ... }
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected capability name")
            return None
        
        cap_name = Identifier(self.cur_token.literal)
        self.next_token()
        
        # Simple capability registration
        return CapabilityStatement(name=cap_name)

    def parse_grant_statement(self):
        """Parse grant statement - grant capability to entity"""
        token = self.cur_token
        self.next_token()
        
        # grant entity capability
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected entity name")
            return None
        
        entity_name = Identifier(self.cur_token.literal)
        self.next_token()
        
        # Expect capability
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected capability name")
            return None
        
        capability = Identifier(self.cur_token.literal)
        self.next_token()
        
        return GrantStatement(entity_name=entity_name, capability=capability)

    def parse_revoke_statement(self):
        """Parse revoke statement - revoke capability from entity"""
        token = self.cur_token
        self.next_token()
        
        # revoke entity capability
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected entity name")
            return None
        
        entity_name = Identifier(self.cur_token.literal)
        self.next_token()
        
        # Expect capability
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected capability name")
            return None
        
        capability = Identifier(self.cur_token.literal)
        self.next_token()
        
        return RevokeStatement(entity_name=entity_name, capability=capability)

    def parse_validate_statement(self):
        """Parse validate statement - validate data"""
        token = self.cur_token
        self.next_token()
        
        # validate data_expr using schema_expr
        data_expr = self.parse_expression(LOWEST)
        if data_expr is None:
            self.errors.append(f"Line {token.line}:{token.column} - Expected expression to validate")
            return None
        
        # Expect 'using'
        if not (self.cur_token_is(IDENT) and self.cur_token.literal == 'using'):
            self.errors.append(f"Line {token.line}:{token.column} - Expected 'using' after validate")
        else:
            self.next_token()
        
        schema_expr = self.parse_expression(LOWEST)
        
        return ValidateStatement(data=data_expr, schema=schema_expr)

    def parse_sanitize_statement(self):
        """Parse sanitize statement - sanitize data"""
        token = self.cur_token
        self.next_token()
        
        # sanitize data_expr as encoding_type
        data_expr = self.parse_expression(LOWEST)
        if data_expr is None:
            self.errors.append(f"Line {token.line}:{token.column} - Expected expression to sanitize")
            return None
        
        # Expect 'as'
        encoding_type = None
        if self.cur_token_is(IDENT) and self.cur_token.literal == 'as':
            self.next_token()
            if self.cur_token_is(IDENT):
                encoding_type = self.cur_token.literal
                self.next_token()
        
        return SanitizeStatement(data=data_expr, encoding=encoding_type)

    def parse_inject_statement(self):
        """Parse inject statement - dependency injection"""
        token = self.cur_token
        self.next_token()
        
        # inject dependency_name
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected dependency name after 'inject'")
            return None
        
        dependency_name = self.cur_token.literal
        dependency = Identifier(value=dependency_name)
        self.next_token()
        
        # Semicolon is optional
        if self.cur_token_is(SEMICOLON):
            self.next_token()
        
        return InjectStatement(dependency=dependency)

    def parse_immutable_statement(self):
        """Parse immutable statement - declare immutable variables"""
        token = self.cur_token
        self.next_token()
        
        # immutable let/const name = value
        target = None
        if self.cur_token_is(LET):
            self.next_token()
            if self.cur_token_is(IDENT):
                target = Identifier(self.cur_token.literal)
                self.next_token()
        elif self.cur_token_is(CONST):
            self.next_token()
            if self.cur_token_is(IDENT):
                target = Identifier(self.cur_token.literal)
                self.next_token()
        elif self.cur_token_is(IDENT):
            target = Identifier(self.cur_token.literal)
            self.next_token()
        
        if target is None:
            self.errors.append(f"Line {token.line}:{token.column} - Expected variable after immutable")
            return None
        
        value = None
        if self.cur_token_is(ASSIGN):
            self.next_token()
            value = self.parse_expression(LOWEST)
        
        return ImmutableStatement(target=target, value=value)

    # === COMPLEXITY STATEMENT PARSERS ===

    def parse_interface_statement(self):
        """Parse interface definition statement"""
        token = self.cur_token
        self.next_token()
        
        # interface Name { method1; method2; }
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected interface name")
            return None
        
        interface_name = Identifier(self.cur_token.literal)
        self.next_token()
        
        methods = []
        properties = {}
        
        if self.cur_token_is(LBRACE):
            self.next_token()
            while not self.cur_token_is(RBRACE) and self.cur_token.type != EOF:
                if self.cur_token_is(IDENT):
                    methods.append(self.cur_token.literal)
                    self.next_token()
                    # Skip to next method
                    while not self.cur_token_is(SEMICOLON) and not self.cur_token_is(RBRACE):
                        self.next_token()
                    if self.cur_token_is(SEMICOLON):
                        self.next_token()
                else:
                    self.next_token()
        
        return InterfaceStatement(name=interface_name, methods=methods, properties=properties)

    def parse_type_alias_statement(self):
        """Parse type alias statement"""
        token = self.cur_token
        self.next_token()
        
        # type_alias Name = type_expr
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected type alias name")
            return None
        
        alias_name = Identifier(self.cur_token.literal)
        self.next_token()
        
        if not self.cur_token_is(ASSIGN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '=' in type alias")
            return None
        
        self.next_token()
        base_type = self.parse_expression(LOWEST)
        
        return TypeAliasStatement(name=alias_name, base_type=base_type)

    def parse_module_statement(self):
        """Parse module definition statement"""
        token = self.cur_token
        self.next_token()
        
        # module Name { body }
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected module name")
            return None
        
        module_name = Identifier(self.cur_token.literal)
        self.next_token()
        
        body = self.parse_block()
        
        return ModuleStatement(name=module_name, body=body)

    def parse_package_statement(self):
        """Parse package definition statement"""
        token = self.cur_token
        self.next_token()
        
        # package name.path { body }
        package_parts = []
        while self.cur_token_is(IDENT):
            package_parts.append(self.cur_token.literal)
            self.next_token()
            if self.cur_token_is(DOT):
                self.next_token()
            else:
                break
        
        if not package_parts:
            self.errors.append(f"Line {token.line}:{token.column} - Expected package name")
            return None
        
        package_name = Identifier('.'.join(package_parts))
        
        body = self.parse_block()
        
        return PackageStatement(name=package_name, body=body)

    def parse_using_statement(self):
        """Parse using statement for resource management"""
        token = self.cur_token
        self.next_token()
        
        # using(resource = expr) { body }
        if not self.cur_token_is(LPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '(' after using")
            return None
        
        self.next_token()
        
        # Parse resource assignment
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected resource name")
            return None
        
        resource_name = Identifier(self.cur_token.literal)
        self.next_token()
        
        if not self.cur_token_is(ASSIGN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '=' in using clause")
            return None
        
        self.next_token()
        resource_expr = self.parse_expression(LOWEST)
        
        if not self.cur_token_is(RPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after using clause")
            return None
        
        self.next_token()
        body = self.parse_block()
        
        return UsingStatement(resource_name=resource_name, resource_expr=resource_expr, body=body)

    def parse_channel_statement(self):
        """Parse channel declaration: channel<type> name; or channel<type> name = expr;"""
        token = self.cur_token
        self.next_token()  # consume CHANNEL
        
        # Parse channel<type>
        if not self.cur_token_is(LT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '<' after channel")
            return None
        
        self.next_token()
        element_type = self.parse_type_expression()
        
        if not self.cur_token_is(GT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '>' in channel type")
            return None
        
        self.next_token()
        
        # Parse channel name
        if not self.cur_token_is(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected channel name")
            return None
        
        name = self.cur_token.literal
        self.next_token()
        
        # Optional capacity specification
        capacity = None
        if self.cur_token_is(ASSIGN):
            self.next_token()
            capacity = self.parse_expression(LOWEST)
        
        if not self.cur_token_is(SEMICOLON):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ';' after channel declaration")
            return None
        
        self.next_token()
        return ChannelStatement(name=name, element_type=element_type, capacity=capacity)

    def parse_send_statement(self):
        """Parse send statement: send(channel, value);"""
        token = self.cur_token
        self.next_token()  # consume SEND
        
        if not self.cur_token_is(LPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '(' after send")
            return None
        
        self.next_token()
        channel_expr = self.parse_expression(LOWEST)
        
        if not self.cur_token_is(COMMA):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ',' after channel in send")
            return None
        
        self.next_token()
        value_expr = self.parse_expression(LOWEST)
        
        if not self.cur_token_is(RPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after send arguments")
            return None
        
        self.next_token()
        
        if not self.cur_token_is(SEMICOLON):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ';' after send statement")
            return None
        
        self.next_token()
        return SendStatement(channel_expr=channel_expr, value_expr=value_expr)

    def parse_receive_statement(self):
        """Parse receive statement: receive(channel); or var = receive(channel);"""
        token = self.cur_token
        self.next_token()  # consume RECEIVE
        
        if not self.cur_token_is(LPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '(' after receive")
            return None
        
        self.next_token()
        channel_expr = self.parse_expression(LOWEST)
        
        if not self.cur_token_is(RPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after receive argument")
            return None
        
        self.next_token()
        
        # Optional target assignment
        target = None
        
        if not self.cur_token_is(SEMICOLON):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ';' after receive statement")
            return None
        
        self.next_token()
        return ReceiveStatement(channel_expr=channel_expr, target=target)

    def parse_atomic_statement(self):
        """Parse atomic statement: atomic { body } or atomic(expr)"""
        token = self.cur_token
        self.next_token()  # consume ATOMIC
        
        body = None
        expr = None
        
        if self.cur_token_is(LBRACE):
            # atomic { body }
            body = self.parse_block()
        elif self.cur_token_is(LPAREN):
            # atomic(expr)
            self.next_token()
            expr = self.parse_expression(LOWEST)
            
            if not self.cur_token_is(RPAREN):
                self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after atomic expression")
                return None
            
            self.next_token()
            
            if not self.cur_token_is(SEMICOLON):
                self.errors.append(f"Line {token.line}:{token.column} - Expected ';' after atomic expression")
                return None
            
            self.next_token()
        else:
            self.errors.append(f"Line {token.line}:{token.column} - Expected '{{' or '(' after atomic")
            return None
        
        return AtomicStatement(body=body, expr=expr)

    # === BLOCKCHAIN STATEMENT PARSING ===
    
    def parse_ledger_statement(self):
        """Parse ledger statement: ledger NAME = value;
        
        Declares immutable ledger variable with version tracking.
        """
        token = self.cur_token
        
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'ledger'")
            return None
        
        name = Identifier(value=self.cur_token.literal)
        
        if not self.expect_peek(ASSIGN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '=' after ledger name")
            return None
        
        self.next_token()
        initial_value = self.parse_expression(LOWEST)
        
        # Semicolon is optional
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        
        return LedgerStatement(name=name, initial_value=initial_value)
    
    def parse_state_statement(self):
        """Parse state statement: state NAME = value;
        
        Declares mutable contract state variable.
        """
        token = self.cur_token
        
        if not self.expect_peek(IDENT):
            self.errors.append(f"Line {token.line}:{token.column} - Expected identifier after 'state'")
            return None
        
        name = Identifier(value=self.cur_token.literal)
        
        if not self.expect_peek(ASSIGN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '=' after state name")
            return None
        
        self.next_token()
        initial_value = self.parse_expression(LOWEST)
        
        # Semicolon is optional
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        
        return StateStatement(name=name, initial_value=initial_value)
    
    def parse_require_statement(self):
        """Parse require statement: require(condition, message);
        
        Asserts condition, reverts transaction if false.
        """
        token = self.cur_token
        
        if not self.expect_peek(LPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '(' after 'require'")
            return None
        
        self.next_token()
        condition = self.parse_expression(LOWEST)
        
        message = None
        if self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            message = self.parse_expression(LOWEST)
        
        if not self.expect_peek(RPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after require arguments")
            return None
        
        # Semicolon is optional
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        
        return RequireStatement(condition=condition, message=message)
    
    def parse_revert_statement(self):
        """Parse revert statement: revert(reason);
        
        Reverts transaction with optional reason.
        """
        token = self.cur_token
        
        reason = None
        if self.peek_token_is(LPAREN):
            self.next_token()
            self.next_token()
            reason = self.parse_expression(LOWEST)
            
            if not self.expect_peek(RPAREN):
                self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after revert reason")
                return None
        
        # Semicolon is optional
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        
        return RevertStatement(reason=reason)
    
    def parse_limit_statement(self):
        """Parse limit statement: limit(gas_amount);
        
        Sets gas limit for operation.
        """
        token = self.cur_token
        
        if not self.expect_peek(LPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected '(' after 'limit'")
            return None
        
        self.next_token()
        gas_limit = self.parse_expression(LOWEST)
        
        if not self.expect_peek(RPAREN):
            self.errors.append(f"Line {token.line}:{token.column} - Expected ')' after limit amount")
            return None
        
        # Semicolon is optional
        if self.peek_token_is(SEMICOLON):
            self.next_token()
        
        return LimitStatement(gas_limit=gas_limit)
    
    # === BLOCKCHAIN EXPRESSION PARSING ===
    
    def parse_tx_expression(self):
        """Parse TX expression: tx.caller, tx.timestamp, tx.gas_used, etc.
        
        Access transaction context properties.
        """
        if not self.expect_peek(DOT):
            # Just 'tx' by itself returns the TX object
            return TXExpression(property_name=None)
        
        if not self.expect_peek(IDENT):
            self.errors.append("Expected property name after 'tx.'")
            return None
        
        property_name = self.cur_token.literal
        return TXExpression(property_name=property_name)
    
    def parse_hash_expression(self):
        """Parse hash expression: hash(data, algorithm)
        
        Cryptographic hash function.
        """
        if not self.expect_peek(LPAREN):
            self.errors.append("Expected '(' after 'hash'")
            return None
        
        self.next_token()
        data = self.parse_expression(LOWEST)
        
        algorithm = StringLiteral(value="SHA256")  # Default algorithm
        if self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            algorithm = self.parse_expression(LOWEST)
        
        if not self.expect_peek(RPAREN):
            self.errors.append("Expected ')' after hash arguments")
            return None
        
        return HashExpression(data=data, algorithm=algorithm)
    
    def parse_signature_expression(self):
        """Parse signature expression: signature(data, private_key, algorithm)
        
        Creates digital signature.
        """
        if not self.expect_peek(LPAREN):
            self.errors.append("Expected '(' after 'signature'")
            return None
        
        self.next_token()
        data = self.parse_expression(LOWEST)
        
        if not self.expect_peek(COMMA):
            self.errors.append("Expected ',' after data in signature")
            return None
        
        self.next_token()
        private_key = self.parse_expression(LOWEST)
        
        algorithm = StringLiteral(value="ECDSA")  # Default algorithm
        if self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            algorithm = self.parse_expression(LOWEST)
        
        if not self.expect_peek(RPAREN):
            self.errors.append("Expected ')' after signature arguments")
            return None
        
        return SignatureExpression(data=data, private_key=private_key, algorithm=algorithm)
    
    def parse_verify_sig_expression(self):
        """Parse verify_sig expression: verify_sig(data, signature, public_key, algorithm)
        
        Verifies digital signature.
        """
        if not self.expect_peek(LPAREN):
            self.errors.append("Expected '(' after 'verify_sig'")
            return None
        
        self.next_token()
        data = self.parse_expression(LOWEST)
        
        if not self.expect_peek(COMMA):
            self.errors.append("Expected ',' after data in verify_sig")
            return None
        
        self.next_token()
        signature = self.parse_expression(LOWEST)
        
        if not self.expect_peek(COMMA):
            self.errors.append("Expected ',' after signature in verify_sig")
            return None
        
        self.next_token()
        public_key = self.parse_expression(LOWEST)
        
        algorithm = StringLiteral(value="ECDSA")  # Default algorithm
        if self.peek_token_is(COMMA):
            self.next_token()
            self.next_token()
            algorithm = self.parse_expression(LOWEST)
        
        if not self.expect_peek(RPAREN):
            self.errors.append("Expected ')' after verify_sig arguments")
            return None
        
        return VerifySignatureExpression(data=data, signature=signature, public_key=public_key, algorithm=algorithm)
    
    def parse_gas_expression(self):
        """Parse gas expression: gas or gas.used or gas.remaining
        
        Access gas tracking information.
        """
        if not self.peek_token_is(DOT):
            # Just 'gas' by itself - returns GasExpression with no property
            return GasExpression(property_name=None)
        
        self.next_token()  # consume DOT
        
        if not self.expect_peek(IDENT):
            self.errors.append("Expected property name after 'gas.'")
            return None
        
        property_name = self.cur_token.literal
        return GasExpression(property_name=property_name)

    def parse_ternary_expression(self, condition):
        """Parse ternary expression: condition ? true_value : false_value"""
        from ..zexus_ast import TernaryExpression
        
        self.next_token()  # consume '?'
        true_value = self.parse_expression(LOWEST)
        
        if not self.expect_peek(COLON):
            self.errors.append("Expected ':' in ternary expression")
            return None
        
        self.next_token()  # consume ':'
        false_value = self.parse_expression(LOWEST)
        
        return TernaryExpression(condition, true_value, false_value)

    def parse_nullish_expression(self, left):
        """Parse nullish coalescing: value ?? default"""
        from ..zexus_ast import NullishExpression
        
        self.next_token()  # consume '??'
        right = self.parse_expression(NULLISH_PREC)
        
        return NullishExpression(left, right)

    def parse_this(self):
        """Parse 'this' expression for contract self-reference"""
        return ThisExpression()
    
    def parse_emit_statement(self):
        """Parse emit statement
        
        emit Transfer(from, to, amount);
        emit StateChange(\"balance_updated\", new_balance);
        """
        if not self.expect_peek(IDENT):
            return None
        
        event_name = Identifier(self.cur_token.literal)
        
        # Parse optional arguments
        arguments = []
        if self.peek_token_is(LPAREN):
            self.next_token()  # consume '('
            arguments = self.parse_expression_list(RPAREN)
        
        return EmitStatement(event_name, arguments)
    
    def parse_modifier_declaration(self):
        """Parse modifier declaration
        
        modifier onlyOwner {
            require(TX.caller == owner, \"Not owner\");
        }
        """
        if not self.expect_peek(IDENT):
            return None
        
        name = Identifier(self.cur_token.literal)
        
        # Parse optional parameters
        parameters = []
        if self.peek_token_is(LPAREN):
            self.next_token()  # consume '('
            parameters = self.parse_function_parameters()
        
        # Parse body
        if not self.expect_peek(LBRACE):
            return None
        
        body = self.parse_block_statement()
        
        return ModifierDeclaration(name, parameters, body)

# Backward compatibility
Parser = UltimateParser
