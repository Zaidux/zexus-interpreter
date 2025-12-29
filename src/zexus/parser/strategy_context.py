# strategy_context.py (FINAL FIXED VERSION)
import sys
import tempfile
import os
from ..zexus_token import *
from ..zexus_ast import *
from ..config import config as zexus_config
from types import SimpleNamespace # Helper for AST node creation

# Import Parser for nested parsing (needed for LOG statement)
# Note: This is imported at runtime to avoid circular dependency
def get_parser_class():
    """Lazy import Parser class to avoid circular dependency"""
    from .parser import Parser
    return Parser

# Local helper to control debug printing according to user config
def ctx_debug(msg, data=None, level='debug'):
    try:
        if not zexus_config.should_log(level):
            return
    except Exception:
        return
    if data is not None:
        print(f"🔍 [CTX DEBUG] {msg}: {data}")
    else:
        print(f"🔍 [CTX DEBUG] {msg}")

# Helper function for parser debug output - OPTIMIZED
def parser_debug(msg):
    # OPTIMIZATION: Check config flag to avoid string formatting overhead
    if hasattr(zexus_config, 'enable_parser_debug') and zexus_config.enable_parser_debug:
        print(msg)
    elif zexus_config.should_log('debug'):
        print(msg)

# Helper class to create objects that behave like AST nodes (dot notation access)
class AstNodeShim:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __repr__(self):
        return f"AstNodeShim({self.__dict__})"

class ContextStackParser:
    def __init__(self, structural_analyzer):
        self.structural_analyzer = structural_analyzer
        self.current_context = ['global']
        self.context_rules = {
            'function': self._parse_function_context,
            FUNCTION: self._parse_function_statement_context,
            ACTION: self._parse_action_statement_context,
            ASYNC: self._parse_async_expression_block,  # Handle async expressions
            'try_catch': self._parse_try_catch_context,
            'try_catch_statement': self._parse_try_catch_statement,
            'conditional': self._parse_conditional_context,
            'loop': self._parse_loop_context,
            'screen': self._parse_screen_context,
            'brace_block': self._parse_brace_block_context,
            'paren_block': self._parse_paren_block_context,
            'statement_block': self._parse_statement_block_context,
            'bracket_block': self._parse_brace_block_context,
            # DIRECT handlers for specific statement types
            IF: self._parse_statement_block_context,
            FOR: self._parse_statement_block_context,
            WHILE: self._parse_statement_block_context,
            RETURN: self._parse_statement_block_context,
            CONTINUE: self._parse_statement_block_context,
            DEFER: self._parse_statement_block_context,
            ENUM: self._parse_statement_block_context,
            SANDBOX: self._parse_statement_block_context,
            'let_statement': self._parse_let_statement_block,
            LET: self._parse_let_statement_block,  # Handle LET token type from structural analyzer
            'const_statement': self._parse_const_statement_block,
            CONST: self._parse_const_statement_block,  # Handle CONST token type
            'print_statement': self._parse_print_statement_block,
            PRINT: self._parse_print_statement_block,  # Handle PRINT token type from structural analyzer
            'debug_statement': self._parse_debug_statement_block,
            DEBUG: self._parse_debug_statement_block,
            'assignment_statement': self._parse_assignment_statement,
            'function_call_statement': self._parse_function_call_statement,
            'entity_statement': self._parse_entity_statement_block,
            'USE': self._parse_use_statement_block,
            'use_statement': self._parse_use_statement_block,  # Fix: add lowercase version
            # Added contract handling
            'contract_statement': self._parse_contract_statement_block,
            # NEW: Security statement handlers
            CAPABILITY: self._parse_capability_statement,
            GRANT: self._parse_grant_statement,
            REVOKE: self._parse_revoke_statement,
            VALIDATE: self._parse_validate_statement,
            SANITIZE: self._parse_sanitize_statement,
            IMMUTABLE: self._parse_immutable_statement,
            # NEW: Complexity management handlers
            INTERFACE: self._parse_interface_statement,
            TYPE_ALIAS: self._parse_type_alias_statement,
            MODULE: self._parse_module_statement,
            PACKAGE: self._parse_package_statement,
            USING: self._parse_using_statement,
            # CONCURRENCY handlers
            CHANNEL: self._parse_channel_statement,
            SEND: self._parse_send_statement,
            RECEIVE: self._parse_receive_statement,
            ATOMIC: self._parse_atomic_statement,
            # BLOCKCHAIN handlers
            LEDGER: self._parse_ledger_statement,
            STATE: self._parse_state_statement,
            PERSISTENT: self._parse_persistent_statement,
            REQUIRE: self._parse_require_statement,
            REVERT: self._parse_revert_statement,
            LIMIT: self._parse_limit_statement,
            # REACTIVE handlers
            STREAM: self._parse_stream_statement,
            WATCH: self._parse_watch_statement,
            # POLICY-AS-CODE handlers
            PROTECT: self._parse_protect_statement,
            VERIFY: self._parse_verify_statement,
            RESTRICT: self._parse_restrict_statement,
            # ENTERPRISE FEATURE handlers
            MIDDLEWARE: self._parse_middleware_statement,
            AUTH: self._parse_auth_statement,
            THROTTLE: self._parse_throttle_statement,
            CACHE: self._parse_cache_statement,
            # DEPENDENCY INJECTION handlers
            INJECT: self._parse_inject_statement,
            VALIDATE: self._parse_validate_statement,
            SANITIZE: self._parse_sanitize_statement,
        }

    def push_context(self, context_type, context_name=None):
        """Push a new context onto the stack"""
        context_str = f"{context_type}:{context_name}" if context_name else context_type
        self.current_context.append(context_str)
        ctx_debug(f"📥 [Context] Pushed: {context_str}", level='debug')

    def pop_context(self):
        """Pop the current context from the stack"""
        if len(self.current_context) > 1:
            popped = self.current_context.pop()
            ctx_debug(f"📤 [Context] Popped: {popped}", level='debug')
            return popped
        return None

    def get_current_context(self):
        """Get the current parsing context"""
        return self.current_context[-1] if self.current_context else 'global'

    def parse_block(self, block_info, all_tokens):
        """Parse a block with context awareness"""
        block_type = block_info.get('subtype', block_info['type'])
        context_name = block_info.get('name', 'anonymous')

        self.push_context(block_type, context_name)

        try:
            # Early exit: if a block has no meaningful tokens, skip parsing it
            tokens = block_info.get('tokens', []) or []
            def _meaningful(tok):
                lit = getattr(tok, 'literal', None)
                # treat identifiers, strings, numbers and structural tokens as meaningful
                if tok.type in {IDENT, STRING, INT, FLOAT, LBRACE, RBRACE, LPAREN, RPAREN, LBRACKET, RBRACKET, COMMA, DOT, SEMICOLON, ASSIGN, LAMBDA}:
                    return True
                return not (lit is None or lit == '')

            if not any(_meaningful(t) for t in tokens):
                ctx_debug(f"Skipping empty/insignificant block tokens for {block_type}", level='debug')
                return None
            # Use appropriate parsing strategy for this context
            if block_type in self.context_rules:
                result = self.context_rules[block_type](block_info, all_tokens)
            else:
                result = self._parse_generic_block(block_info, all_tokens)

            # CRITICAL: Don't wrap Statement nodes, only wrap Expressions
            if result is not None:
                if isinstance(result, Statement):
                    parser_debug(f"  ✅ Parsed: {type(result).__name__} at line {block_info.get('start_token', {}).get('line', 'unknown')}")
                    # If we got a BlockStatement but it has no inner statements,
                    # attempt to populate it from the block tokens (best-effort).
                    if isinstance(result, BlockStatement) and not getattr(result, 'statements', None):
                        tokens = block_info.get('tokens', [])
                        if tokens:
                            print(f"  🔧 Populating empty BlockStatement from {len(tokens)} tokens")
                            parsed_stmts = self._parse_block_statements(tokens)
                            result.statements = parsed_stmts
                            parser_debug(f"  ✅ Populated BlockStatement with {len(parsed_stmts)} statements")
                    return result
                elif isinstance(result, Expression):
                    parser_debug(f"  ✅ Parsed: ExpressionStatement at line {block_info.get('start_token', {}).get('line', 'unknown')}")
                    return ExpressionStatement(result)
                else:
                    result = self._ensure_statement_node(result, block_info)
                    if result:
                        parser_debug(f"  ✅ Parsed: {type(result).__name__} at line {block_info.get('start_token', {}).get('line', 'unknown')}")
                    return result
            else:
                parser_debug(f"  ⚠️ No result for {block_type} at line {block_info.get('start_token', {}).get('line', 'unknown')}")
                return None

        except Exception as e:
            parser_debug(f"⚠️ [Context] Error parsing {block_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.pop_context()

    def _ensure_statement_node(self, node, block_info):
        """Ensure the node is a proper Statement"""
        # If it's already a Statement, return it
        if isinstance(node, Statement):
            return node

        # If it's an Expression, wrap it
        if isinstance(node, Expression):
            return ExpressionStatement(node)

        # If it's a list, process each item
        elif isinstance(node, list):
            statements = []
            for item in node:
                if isinstance(item, Expression):
                    statements.append(ExpressionStatement(item))
                elif isinstance(item, Statement):
                    statements.append(item)

            if len(statements) > 1:
                block = BlockStatement()
                block.statements = statements
                return block
            elif len(statements) == 1:
                return statements[0]
            else:
                return BlockStatement()

        # Unknown type, return empty block
        return BlockStatement()

    # === DIRECT STATEMENT PARSERS - THESE RETURN ACTUAL STATEMENTS ===

    def _parse_let_statement_block(self, block_info, all_tokens):
        """Parse let statement block with robust method chain handling"""
        parser_debug("🔧 [Context] Parsing let statement")
        tokens = block_info['tokens']

        if len(tokens) < 4:
            parser_debug("  ❌ Invalid let statement: too few tokens")
            return None

        if tokens[1].type != IDENT:
            parser_debug("  ❌ Invalid let statement: expected identifier after 'let'")
            return None

        variable_name = tokens[1].literal
        parser_debug(f"  📝 Variable: {variable_name}")

        # Check for << operator (file import): let code << "file.ext"
        if len(tokens) >= 4 and tokens[2].type == IMPORT_OP:
            parser_debug(f"  📝 File import syntax detected: << operator")
            # Get the filepath (next token after <<)
            if tokens[3].type in [STRING, IDENT]:
                filepath_token = tokens[3]
                if filepath_token.type == STRING:
                    filepath_expr = StringLiteral(filepath_token.literal)
                else:
                    filepath_expr = Identifier(filepath_token.literal)
                
                parser_debug(f"  ✅ Let with file import: {variable_name} << {filepath_token.literal}")
                # Create a special FileImportExpression
                from ..zexus_ast import FileImportExpression
                value_expression = FileImportExpression(filepath_expr)
                return LetStatement(
                    name=Identifier(variable_name),
                    value=value_expression,
                    type_annotation=None
                )
            else:
                parser_debug(f"  ❌ Expected filename after << operator")
                return None

        # Check for type annotation (name: Type = value) or colon syntax (name : value)
        type_annotation = None
        colon_as_assign = False
        if len(tokens) > 2 and tokens[2].type == COLON:
            # Check if next token is IDENT (type annotation) or value (colon syntax)
            if len(tokens) > 3 and tokens[3].type == IDENT:
                # Could be type annotation - check if followed by =
                if len(tokens) > 4 and tokens[4].type == ASSIGN:
                    # Type annotation: let x : Type = value
                    type_annotation = tokens[3].literal
                    parser_debug(f"  📝 Type annotation: {type_annotation}")
                else:
                    # Colon as assignment: let x : identifier_value
                    colon_as_assign = True
                    parser_debug(f"  📝 Colon assignment syntax detected")
            else:
                # Colon as assignment: let x : 42
                colon_as_assign = True
                parser_debug(f"  📝 Colon assignment syntax detected")

        # Find equals sign or use colon as assignment
        if colon_as_assign:
            equals_index = 2  # Treat colon as equals
        else:
            start_index = 4 if type_annotation else 2
            equals_index = -1
            for i in range(start_index, len(tokens)):
                if tokens[i].type == ASSIGN:
                    equals_index = i
                    break

            if equals_index == -1:
                parser_debug("  ❌ Invalid let statement: no assignment operator")
                return None

        # Collect RHS tokens with proper nesting support
        value_tokens = []
        nesting = 0
        j = equals_index + 1

        while j < len(tokens):
            t = tokens[j]

            # Track nested structures
            if t.type in {LPAREN, LBRACE, LBRACKET}:
                nesting += 1
            elif t.type in {RPAREN, RBRACE, RBRACKET}:
                nesting -= 1

            # Only check statement boundaries when not in nested structure
            if nesting == 0:
                # Stop at explicit terminators
                if t.type == SEMICOLON:
                    break
                # Check for statement starters that should break
                # Context-sensitive: IF followed by THEN is an expression, not a statement
                if t.type in {LET, PRINT, FOR, WHILE, RETURN, CONTINUE, ACTION, TRY, EXTERNAL, SCREEN, EXPORT, USE, DEBUG}:
                    prev = tokens[j-1] if j > 0 else None
                    # Allow if part of method chain OR if DEBUG followed by ( (function call)
                    allow_method_chain = prev and prev.type == DOT
                    allow_debug_call = t.type == DEBUG and j + 1 < len(tokens) and tokens[j + 1].type == LPAREN
                    if not (allow_method_chain or allow_debug_call):
                        break
                # IF is only a statement starter if NOT followed by THEN
                elif t.type == IF:
                    prev = tokens[j-1] if j > 0 else None
                    # Allow if part of method chain
                    if prev and prev.type == DOT:
                        pass  # Method chain, continue
                    else:
                        # Check if this is if-then-else expression
                        is_if_expression = False
                        for k in range(j + 1, len(tokens)):
                            if tokens[k].type == THEN:
                                is_if_expression = True
                                break
                            # LPAREN right after IF indicates statement form: if (...) { }
                            # But LPAREN after IDENT is a function call: if exists(...) then ...
                            elif tokens[k].type == LPAREN and k == j + 1:
                                # LPAREN immediately after IF = statement form
                                break
                            elif tokens[k].type in {LBRACE, COLON}:
                                # Other statement form indicators
                                break
                        if not is_if_expression:
                            # Statement form IF, break here
                            break

            value_tokens.append(t)
            j += 1

        parser_debug(f"  📝 Value tokens: {[t.literal for t in value_tokens]}")

        # Parse the value expression
        if not value_tokens:
            parser_debug("  ❌ No value tokens found")
            return None

        # Special case: map literal
        if value_tokens[0].type == LBRACE:
            parser_debug("  🗺️ Detected map literal")
            value_expression = self._parse_map_literal(value_tokens)
        else:
            value_expression = self._parse_expression(value_tokens)

        if value_expression is None:
            parser_debug("  ❌ Could not parse value expression")
            return None

        type_msg = f" : {type_annotation}" if type_annotation else ""
        parser_debug(f"  ✅ Let statement: {variable_name}{type_msg} = {type(value_expression).__name__}")
        return LetStatement(
            name=Identifier(variable_name),
            value=value_expression,
            type_annotation=Identifier(type_annotation) if type_annotation else None
        )

    def _parse_const_statement_block(self, block_info, all_tokens):
        """Parse const statement block with robust method chain handling (mirrors let)"""
        parser_debug("🔧 [Context] Parsing const statement")
        tokens = block_info['tokens']

        if len(tokens) < 4:
            parser_debug("  ❌ Invalid const statement: too few tokens")
            return None

        if tokens[1].type != IDENT:
            parser_debug("  ❌ Invalid const statement: expected identifier after 'const'")
            return None

        variable_name = tokens[1].literal
        parser_debug(f"  📝 Variable: {variable_name}")

        # Check for << operator (file import)
        if len(tokens) > 2 and tokens[2].type == IMPORT_OP:
            parser_debug(f"  📂 Detected const << file import")
            if len(tokens) < 4:
                parser_debug("  ❌ Invalid const << statement: missing filename")
                return None
            
            # Parse the filename expression (everything after <<)
            filename_tokens = []
            j = 3
            while j < len(tokens):
                if tokens[j].type == SEMICOLON:
                    break
                filename_tokens.append(tokens[j])
                j += 1
            
            if not filename_tokens:
                parser_debug("  ❌ Invalid const << statement: empty filename")
                return None
            
            filename_expr = self._parse_expression(filename_tokens)
            if filename_expr is None:
                parser_debug("  ❌ Could not parse filename expression")
                return None
            
            parser_debug(f"  ✅ Const file import: {variable_name} << file")
            return ConstStatement(
                name=Identifier(variable_name),
                value=FileImportExpression(filepath=filename_expr)
            )

        equals_index = -1
        for i, token in enumerate(tokens):
            if token.type == ASSIGN:
                equals_index = i
                break

        if equals_index == -1:
            parser_debug("  ❌ Invalid const statement: no assignment operator")
            return None

        # Collect RHS tokens with proper nesting support
        value_tokens = []
        nesting = 0
        j = equals_index + 1

        while j < len(tokens):
            t = tokens[j]

            # Track nested structures
            if t.type in {LPAREN, LBRACE, LBRACKET}:
                nesting += 1
            elif t.type in {RPAREN, RBRACE, RBRACKET}:
                nesting -= 1

            # Only check statement boundaries when not in nested structure
            if nesting == 0:
                # Stop at explicit terminators
                if t.type == SEMICOLON:
                    break
                # Allow method chains but stop at other statement starters
                if t.type in {LET, CONST, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY, EXTERNAL, SCREEN, EXPORT, USE, DEBUG}:
                    prev = tokens[j-1] if j > 0 else None
                    if not (prev and prev.type == DOT):  # Allow if part of method chain
                        break

            value_tokens.append(t)
            j += 1

        parser_debug(f"  📝 Value tokens: {[t.literal for t in value_tokens]}")

        # Parse the value expression
        if not value_tokens:
            parser_debug("  ❌ No value tokens found")
            return None

        # Special case: map literal
        if value_tokens[0].type == LBRACE:
            parser_debug("  🗺️ Detected map literal")
            value_expression = self._parse_map_literal(value_tokens)
        else:
            value_expression = self._parse_expression(value_tokens)

        if value_expression is None:
            parser_debug("  ❌ Could not parse value expression")
            return None

        parser_debug(f"  ✅ Const statement: {variable_name} = {type(value_expression).__name__}")
        return ConstStatement(
            name=Identifier(variable_name),
            value=value_expression
        )

    def _parse_data_statement(self, tokens):
        """Parse data statement (dataclass definition)
        
        @validated
        data User {
            name: string,
            email: string = "default",
            age: number require age >= 0
        }
        
        data immutable Point { x: number, y: number }
        data verified Transaction { from: address, to: address, amount: number }
        """
        parser_debug("🔧 [Context] Parsing data statement")
        parser_debug(f"  📝 Got {len(tokens)} tokens: {[t.literal for t in tokens]}")
        
        if len(tokens) < 4:  # data TypeName { }
            parser_debug("  ❌ Invalid data statement: too few tokens")
            return None
        
        # Parse decorators before 'data' keyword: @validated, @logged, etc.
        decorators = []
        idx = 0
        
        # Skip any @ decorators at the beginning
        while idx < len(tokens) and tokens[idx].type == AT:
            idx += 1  # Skip @
            if idx < len(tokens) and tokens[idx].type == IDENT:
                decorators.append(tokens[idx].literal)
                parser_debug(f"  🎨 Found decorator: @{tokens[idx].literal}")
                idx += 1
            else:
                parser_debug("  ❌ Invalid decorator: expected name after @")
        
        # Now we should be at 'data' keyword
        if idx >= len(tokens) or tokens[idx].type != DATA:
            parser_debug(f"  ❌ Expected 'data' keyword at index {idx}, got {tokens[idx].type if idx < len(tokens) else 'EOF'}")
            return None
        idx += 1  # Skip 'data'
        
        # Check for modifiers (immutable, verified, etc.)
        modifiers = []
        
        # Parse modifiers before type name
        while idx < len(tokens) and tokens[idx].type == IDENT and tokens[idx].literal in ["immutable", "verified"]:
            modifiers.append(tokens[idx].literal)
            parser_debug(f"  🏷️ Found modifier: {tokens[idx].literal}")
            idx += 1
        
        # Type name
        if idx >= len(tokens) or tokens[idx].type != IDENT:
            parser_debug("  ❌ Invalid data statement: expected type name")
            return None
        
        type_name = tokens[idx].literal
        parser_debug(f"  📝 Type name: {type_name}")
        idx += 1
        
        # Parse generic type parameters: <T, U, V>
        type_params = []
        if idx < len(tokens) and tokens[idx].type == LT:
            idx += 1  # Skip <
            parser_debug("  🔤 Parsing generic type parameters")
            
            # Parse comma-separated type parameter names
            while idx < len(tokens):
                if tokens[idx].type == GT:
                    idx += 1  # Skip >
                    break
                
                if tokens[idx].type == IDENT:
                    type_params.append(tokens[idx].literal)
                    parser_debug(f"    ✅ Type parameter: {tokens[idx].literal}")
                    idx += 1
                    
                    # Check for comma or closing >
                    if idx < len(tokens):
                        if tokens[idx].type == COMMA:
                            idx += 1  # Skip comma
                        elif tokens[idx].type == GT:
                            continue  # Will break on next iteration
                        else:
                            parser_debug(f"  ❌ Invalid type parameters: expected ',' or '>', got {tokens[idx].type}")
                            return None
                else:
                    parser_debug(f"  ❌ Invalid type parameter: expected identifier, got {tokens[idx].type}")
                    return None
            
            if type_params:
                parser_debug(f"  🔤 Generic type parameters: {type_params}")
        
        # Check for inheritance: extends ParentType
        parent_type = None
        if idx < len(tokens) and tokens[idx].type == IDENT and tokens[idx].literal == "extends":
            idx += 1
            if idx < len(tokens) and tokens[idx].type == IDENT:
                parent_type = tokens[idx].literal
                parser_debug(f"  🔗 Extends: {parent_type}")
                idx += 1
            else:
                parser_debug("  ❌ Invalid extends: expected parent type name")
        
        # Find opening brace
        if idx >= len(tokens) or tokens[idx].type != LBRACE:
            parser_debug("  ❌ Invalid data statement: expected {")
            return None
        idx += 1
        
        # Find closing brace
        brace_count = 1
        body_start = idx
        body_end = idx
        
        while idx < len(tokens):
            if tokens[idx].type == LBRACE:
                brace_count += 1
            elif tokens[idx].type == RBRACE:
                brace_count -= 1
                if brace_count == 0:
                    body_end = idx
                    break
            idx += 1
        
        if brace_count != 0:
            parser_debug("  ❌ Invalid data statement: unmatched braces")
            return None
        
        # Parse field definitions
        body_tokens = tokens[body_start:body_end]
        fields = self._parse_data_fields(body_tokens)
        
        parser_debug(f"  ✅ Data statement: {type_name} with {len(fields)} fields")
        return DataStatement(
            name=Identifier(type_name),
            fields=fields,
            modifiers=modifiers,
            parent=parent_type,
            decorators=decorators,
            type_params=type_params
        )
    
    def _parse_data_fields(self, tokens):
        """Parse field definitions in a dataclass body
        
        name: string,
        email: string = "default",
        age: number require age >= 0,
        computed area => width * height
        """
        from ..zexus_ast import DataField
        
        parser_debug(f"  🔍 Parsing data fields from tokens: {[t.literal for t in tokens]}")
        fields = []
        i = 0
        
        while i < len(tokens):
            # Skip whitespace and commas
            if tokens[i].type in {COMMA, SEMICOLON}:
                i += 1
                continue
            
            # Check for decorators: @logged, @cached, etc.
            decorators = []
            while i < len(tokens) and tokens[i].type == AT:
                i += 1  # Skip @
                if i < len(tokens) and tokens[i].type == IDENT:
                    decorators.append(tokens[i].literal)
                    parser_debug(f"    🎨 Found decorator: @{tokens[i].literal}")
                    i += 1
            
            # Field must start with identifier (or 'method'/'operator' keyword for custom definitions)
            if i >= len(tokens) or tokens[i].type != IDENT:
                i += 1
                continue
            
            # Check for operator overloading: operator + (other) { ... }
            if tokens[i].literal == "operator":
                parser_debug(f"    🔧 Operator overloading detected")
                i += 1
                
                # Get the operator symbol
                operator_symbol = None
                if i < len(tokens):
                    # Could be +, -, *, /, ==, etc.
                    if tokens[i].type in {PLUS, MINUS, STAR, SLASH, MOD, EQ, NOT_EQ, LT, GT, LTE, GTE}:
                        operator_symbol = tokens[i].literal
                        parser_debug(f"      Operator: {operator_symbol}")
                        i += 1
                    else:
                        parser_debug(f"      ❌ Invalid operator symbol: {tokens[i].literal}")
                        i += 1
                        continue
                
                # Parse parameters: (other)
                method_params = []
                if i < len(tokens) and tokens[i].type == LPAREN:
                    i += 1
                    while i < len(tokens) and tokens[i].type != RPAREN:
                        if tokens[i].type == IDENT:
                            method_params.append(tokens[i].literal)
                            i += 1
                        elif tokens[i].type == COMMA:
                            i += 1
                        else:
                            i += 1
                    if i < len(tokens) and tokens[i].type == RPAREN:
                        i += 1
                
                parser_debug(f"        Parameters: {method_params}")
                
                # Parse operator body: { ... }
                method_body = None
                if i < len(tokens) and tokens[i].type == LBRACE:
                    # Find matching closing brace
                    brace_count = 1
                    body_start = i + 1
                    i += 1
                    
                    while i < len(tokens) and brace_count > 0:
                        if tokens[i].type == LBRACE:
                            brace_count += 1
                        elif tokens[i].type == RBRACE:
                            brace_count -= 1
                        i += 1
                    
                    body_tokens = tokens[body_start:i-1]
                    if body_tokens:
                        # Parse the body as a block of statements
                        method_body = self._parse_block_statements(body_tokens)
                        parser_debug(f"        Body: {len(method_body)} statements")
                
                # Create a DataField for the operator
                fields.append(DataField(
                    name=f"__operator_{operator_symbol}__",
                    field_type=None,
                    default_value=None,
                    constraint=None,
                    computed=None,
                    method_body=method_body,
                    method_params=method_params,
                    operator=operator_symbol,
                    decorators=decorators
                ))
                continue
            
            # Check for method definition: method name(params) { ... }
            if tokens[i].literal == "method":
                parser_debug(f"    🔧 Method definition detected")
                i += 1
                
                if i >= len(tokens) or tokens[i].type != IDENT:
                    parser_debug(f"      ❌ Expected method name")
                    i += 1
                    continue
                
                method_name = tokens[i].literal
                parser_debug(f"      Method: {method_name}")
                i += 1
                
                # Parse parameters: (param1, param2, ...)
                method_params = []
                if i < len(tokens) and tokens[i].type == LPAREN:
                    i += 1
                    while i < len(tokens) and tokens[i].type != RPAREN:
                        if tokens[i].type == IDENT:
                            method_params.append(tokens[i].literal)
                            i += 1
                        elif tokens[i].type == COMMA:
                            i += 1
                        else:
                            i += 1
                    if i < len(tokens) and tokens[i].type == RPAREN:
                        i += 1
                
                parser_debug(f"        Parameters: {method_params}")
                
                # Parse method body: { ... }
                method_body = None
                if i < len(tokens) and tokens[i].type == LBRACE:
                    # Find matching closing brace
                    brace_count = 1
                    body_start = i + 1
                    i += 1
                    
                    while i < len(tokens) and brace_count > 0:
                        if tokens[i].type == LBRACE:
                            brace_count += 1
                        elif tokens[i].type == RBRACE:
                            brace_count -= 1
                        i += 1
                    
                    body_tokens = tokens[body_start:i-1]
                    if body_tokens:
                        # Parse the body as a block of statements
                        method_body = self._parse_block_statements(body_tokens)
                        parser_debug(f"        Body: {len(method_body)} statements")
                
                # Create a DataField for the method
                fields.append(DataField(
                    name=method_name,
                    field_type=None,
                    default_value=None,
                    constraint=None,
                    computed=None,
                    method_body=method_body,
                    method_params=method_params,
                    decorators=decorators
                ))
                continue
            
            field_name = tokens[i].literal
            parser_debug(f"    📌 Field: {field_name}")
            i += 1
            
            field_type = None
            default_value = None
            constraint = None
            computed = None
            
            # Check for type annotation: name: type
            if i < len(tokens) and tokens[i].type == COLON:
                i += 1
                if i < len(tokens) and tokens[i].type == IDENT:
                    field_type = tokens[i].literal
                    parser_debug(f"      Type: {field_type}")
                    i += 1
            
            # Check for default value: = value
            if i < len(tokens) and tokens[i].type == ASSIGN:
                i += 1
                # Collect value tokens until comma, require, or computed
                value_tokens = []
                while i < len(tokens) and tokens[i].type not in {COMMA, SEMICOLON}:
                    if tokens[i].type in {REQUIRE} or (tokens[i].type == IDENT and tokens[i].literal in ["computed"]):
                        break
                    value_tokens.append(tokens[i])
                    i += 1
                
                if value_tokens:
                    default_value = self._parse_expression(value_tokens)
                    parser_debug(f"      Default: {default_value}")
            
            # Check for require constraint: require expression
            if i < len(tokens) and tokens[i].type == REQUIRE:
                i += 1
                # Collect constraint tokens until comma
                constraint_tokens = []
                while i < len(tokens) and tokens[i].type not in {COMMA, SEMICOLON}:
                    if tokens[i].type == IDENT and tokens[i].literal == "computed":
                        break
                    constraint_tokens.append(tokens[i])
                    i += 1
                
                if constraint_tokens:
                    constraint = self._parse_expression(constraint_tokens)
                    parser_debug(f"      Constraint: {constraint}")
            
            # Check for computed property: computed => expression
            if i < len(tokens) and tokens[i].type == IDENT and tokens[i].literal == "computed":
                i += 1
                # Expect => (LAMBDA token)
                if i < len(tokens) and tokens[i].type == LAMBDA:
                    i += 1
                    # Collect expression tokens
                    expr_tokens = []
                    while i < len(tokens) and tokens[i].type not in {COMMA, SEMICOLON}:
                        expr_tokens.append(tokens[i])
                        i += 1
                    
                    if expr_tokens:
                        computed = self._parse_expression(expr_tokens)
                        parser_debug(f"      Computed: {computed}")
            
            fields.append(DataField(
                name=field_name,
                field_type=field_type,
                default_value=default_value,
                constraint=constraint,
                computed=computed
            ))
        
        parser_debug(f"  ✅ Parsed {len(fields)} fields")
        return fields

    def _parse_print_statement_block(self, block_info, all_tokens):
        """Parse print statement block with support for:
        - Single argument: print(message) or print(expr)
        - Multiple arguments: print(arg1, arg2, arg3)
        - Conditional print: print(condition, message)
        
        If exactly 2 arguments are provided, treat first as condition, second as message.
        """
        # Debug logging (fail silently if file operations fail)
        try:
            log_path = os.path.join(tempfile.gettempdir(), 'context_parser_log.txt')
            with open(log_path, 'a') as f:
                f.write(f"=== _parse_print_statement_block CALLED ===\n")
                f.flush()
        except (IOError, OSError, PermissionError):
            pass  # Silently ignore debug logging errors
        
        parser_debug("🔧 [Context] Parsing print statement")
        tokens = block_info['tokens']

        if len(tokens) < 2:
            return PrintStatement(values=[])

        # Get all tokens after PRINT keyword
        expression_tokens = tokens[1:]
        
        # If the tokens start with ( and end with ), strip them (they're the print() parens, not nesting)
        tokens_to_parse = expression_tokens
        if len(expression_tokens) >= 2 and expression_tokens[0].type == LPAREN and expression_tokens[-1].type == RPAREN:
            # Strip outer parentheses
            tokens_to_parse = expression_tokens[1:-1]
        
        # Split by commas to get individual expressions
        values = []
        current_expr_tokens = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        
        for token in tokens_to_parse:
            # Track nesting depth
            if token.type == LPAREN:
                paren_depth += 1
            elif token.type == RPAREN:
                paren_depth -= 1
            elif token.type == LBRACKET:
                bracket_depth += 1
            elif token.type == RBRACKET:
                bracket_depth -= 1
            elif token.type == LBRACE:
                brace_depth += 1
            elif token.type == RBRACE:
                brace_depth -= 1
            
            # If we find a comma at depth 0, it's a separator
            if token.type == COMMA and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                # Parse the accumulated tokens as an expression
                if current_expr_tokens:
                    expr = self._parse_expression(current_expr_tokens)
                    if expr:
                        values.append(expr)
                    current_expr_tokens = []
            else:
                current_expr_tokens.append(token)
        
        # Parse the last expression
        if current_expr_tokens:
            expr = self._parse_expression(current_expr_tokens)
            if expr:
                values.append(expr)
        
        # If no values found, use empty string for backward compatibility
        if not values:
            values = [StringLiteral("")]
        
        # Check if this is conditional print: exactly 2 arguments
        if len(values) == 2:
            # Conditional print: print(condition, message)
            return PrintStatement(values=[values[1]], condition=values[0])
        else:
            # Regular print: print(arg1, arg2, ...) or print(single_arg)
            return PrintStatement(values=values)

    def _parse_debug_statement_block(self, block_info, all_tokens):
        """Parse debug statement block - RETURNS DebugStatement (logs with metadata)
        
        Supports:
        - Statement mode: debug value;
        - Function mode: debug(value) or debug(condition, value)
        
        If exactly 2 arguments in function mode, treat as conditional debug.
        """
        # Import DebugStatement at the top to avoid UnboundLocalError
        from ..zexus_ast import DebugStatement
        
        parser_debug("🔧 [Context] Parsing debug statement")
        tokens = block_info['tokens']

        # Check if this is actually a function call: debug(...)
        if len(tokens) >= 2 and tokens[1].type == LPAREN:
            parser_debug("  ℹ️ DEBUG followed by ( - parsing as function call with potential condition")
            
            # Find the matching RPAREN
            tokens_to_parse = tokens[1:]  # All tokens starting from LPAREN
            if len(tokens_to_parse) >= 2 and tokens_to_parse[0].type == LPAREN and tokens_to_parse[-1].type == RPAREN:
                # Extract arguments between parentheses
                arg_tokens = tokens_to_parse[1:-1]
                
                # Split by commas at depth 0 to get individual arguments
                args = []
                current_arg = []
                paren_depth = 0
                bracket_depth = 0
                brace_depth = 0
                
                for token in arg_tokens:
                    if token.type == LPAREN:
                        paren_depth += 1
                    elif token.type == RPAREN:
                        paren_depth -= 1
                    elif token.type == LBRACKET:
                        bracket_depth += 1
                    elif token.type == RBRACKET:
                        bracket_depth -= 1
                    elif token.type == LBRACE:
                        brace_depth += 1
                    elif token.type == RBRACE:
                        brace_depth -= 1
                    
                    if token.type == COMMA and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                        if current_arg:
                            expr = self._parse_expression(current_arg)
                            if expr:
                                args.append(expr)
                            current_arg = []
                    else:
                        current_arg.append(token)
                
                # Parse last argument
                if current_arg:
                    expr = self._parse_expression(current_arg)
                    if expr:
                        args.append(expr)
                
                # Check if conditional debug (2 args) or regular debug (1 arg)
                if len(args) == 2:
                    # Conditional debug: debug(condition, value)
                    return DebugStatement(value=args[1], condition=args[0])
                elif len(args) == 1:
                    # Regular debug: debug(value)
                    return DebugStatement(value=args[0])
                else:
                    parser_debug("  ❌ DEBUG function call requires 1 or 2 arguments")
                    return None
            
            # Fallback: treat as regular expression statement
            expression = self._parse_expression(tokens)
            return ExpressionStatement(expression) if expression else None

        # Otherwise, it's a statement: debug value;
        if len(tokens) < 2:
            parser_debug("  ❌ DEBUG statement requires a value")
            return None

        expression_tokens = tokens[1:]
        expression = self._parse_expression(expression_tokens)

        if expression is None:
            parser_debug("  ❌ Could not parse DEBUG statement value")
            return None

        # Import DebugStatement from zexus_ast
        from ..zexus_ast import DebugStatement
        return DebugStatement(value=expression)

    def _parse_assignment_statement(self, block_info, all_tokens):
        """Parse assignment statement - RETURNS AssignmentExpression"""
        parser_debug("🔧 [Context] Parsing assignment statement")
        tokens = block_info['tokens']

        # Find the ASSIGN operator
        assign_idx = None
        for i, tok in enumerate(tokens):
            if tok.type == ASSIGN:
                assign_idx = i
                break
        
        if assign_idx is None or assign_idx == 0:
            parser_debug("  ❌ Invalid assignment: no assignment operator or nothing before it")
            return None

        # Parse the left-hand side (could be identifier or property access)
        lhs_tokens = tokens[:assign_idx]
        
        # Check if this is a property access (e.g., obj.property or obj["key"])
        target_expr = None
        if len(lhs_tokens) == 1:
            # Simple identifier assignment
            target_expr = Identifier(lhs_tokens[0].literal)
        else:
            # Could be property access or index access
            target_expr = self._parse_expression(lhs_tokens)
        
        if target_expr is None:
            parser_debug("  ❌ Could not parse assignment target")
            return None
        
        # CRITICAL FIX: only collect RHS tokens up to statement boundary
        # Track nesting depth to avoid stopping on braces inside nested structures
        value_tokens = []
        stop_types = {SEMICOLON}  # RBRACE removed - handle with nesting instead
        statement_starters = {LET, CONST, PRINT, FOR, IF, WHILE, RETURN, CONTINUE, ACTION, TRY, EXTERNAL, SCREEN, EXPORT, USE, DEBUG, AUDIT, RESTRICT, SANDBOX, TRAIL, NATIVE, GC, INLINE, BUFFER, SIMD, DEFER, PATTERN, ENUM, STREAM, WATCH, CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE, INTERFACE, TYPE_ALIAS, MODULE, PACKAGE, USING}
        j = assign_idx + 1
        nesting_depth = 0
        while j < len(tokens):
            t = tokens[j]
            
            # Track nesting for brackets, braces, and parens
            if t.type in {LBRACKET, LBRACE, LPAREN}:
                nesting_depth += 1
            elif t.type in {RBRACKET, RBRACE, RPAREN}:
                nesting_depth -= 1
                
                # If we close more than we opened, we've hit outer scope closing brace
                if nesting_depth < 0:
                    break
            
            # Only check for statement boundaries when not nested
            if nesting_depth == 0:
                if t.type in stop_types:
                    break
                    
                # Context-sensitive IF handling: IF followed by THEN is an expression, not a statement
                if t.type == IF:
                    # Look ahead to check if this is if-then-else expression
                    is_if_expression = False
                    for k in range(j + 1, len(tokens)):
                        if tokens[k].type == THEN:
                            is_if_expression = True
                            break
                        elif tokens[k].type in {LBRACE, LPAREN, SEMICOLON}:
                            # These indicate statement form, not expression
                            break
                    if not is_if_expression:
                        # This is a statement-form IF, stop here
                        break
                    # Otherwise, it's an if-then-else expression, include it
                elif t.type in statement_starters:
                    # Other statement starters always break
                    break
            
            value_tokens.append(t)
            j += 1

        # Check if this is a map literal
        if value_tokens and value_tokens[0].type == LBRACE:
            parser_debug("  🗺️ Detected map literal in assignment")
            value_expression = self._parse_map_literal(value_tokens)
        else:
            value_expression = self._parse_expression(value_tokens)

        if value_expression is None:
            parser_debug("  ❌ Could not parse assignment value")
            return None

        return AssignmentExpression(
            name=target_expr,
            value=value_expression
        )

    def _parse_function_call_statement(self, block_info, all_tokens):
        """Parse function call as a statement - RETURNS ExpressionStatement"""
        parser_debug("🔧 [Context] Parsing function call statement")
        tokens = block_info['tokens']

        if len(tokens) < 3 or tokens[1].type != LPAREN:
            parser_debug("  ❌ Invalid function call: no parentheses")
            return None

        function_name = tokens[0].literal
        inner_tokens = tokens[2:-1] if tokens and tokens[-1].type == RPAREN else tokens[2:]
        arguments = self._parse_argument_list(inner_tokens)

        call_expression = CallExpression(Identifier(function_name), arguments)
        return ExpressionStatement(call_expression)

    def _parse_entity_statement_block(self, block_info, all_tokens):
        """Parse entity declaration block with properties and methods"""
        parser_debug("🔧 [Context] Parsing entity statement")
        tokens = block_info['tokens']

        if len(tokens) < 4:  # entity Name { ... }
            return None

        entity_name = tokens[1].literal if tokens[1].type == IDENT else "Unknown"
        parser_debug(f"  📝 Entity: {entity_name}")

        # Check for inheritance: entity User extends BaseEntity { ... }
        parent_name = None
        idx = 2  # Start after entity name
        if idx < len(tokens) and tokens[idx].type == IDENT and tokens[idx].literal == "extends":
            idx += 1
            if idx < len(tokens) and tokens[idx].type == IDENT:
                parent_name = tokens[idx].literal
                parser_debug(f"  🔗 Extends: {parent_name}")
                idx += 1
            else:
                parser_debug("  ❌ Invalid extends: expected parent entity name")

        # Parse properties and methods between braces
        properties = []
        methods = []
        brace_start = -1
        brace_end = -1
        brace_count = 0

        for i, token in enumerate(tokens):
            if token.type == LBRACE:
                if brace_count == 0:
                    brace_start = i
                brace_count += 1
            elif token.type == RBRACE:
                brace_count -= 1
                if brace_count == 0:
                    brace_end = i
                    break

        if brace_start != -1 and brace_end != -1:
            # Parse properties, injections, and actions inside braces
            i = brace_start + 1
            while i < brace_end:
                # Check for inject statements (dependency injection)
                if tokens[i].type == INJECT:
                    parser_debug(f"  🔌 Found inject statement")
                    i += 1  # Move past INJECT
                    
                    # Get dependency name
                    if i < brace_end and tokens[i].type == IDENT:
                        dep_name = tokens[i].literal
                        i += 1
                        
                        # Skip optional type annotation (: Type)
                        if i < brace_end and tokens[i].type == COLON:
                            i += 1  # Skip colon
                            if i < brace_end and tokens[i].type == IDENT:
                                i += 1  # Skip type
                        
                        # Add as a special property with inject flag
                        properties.append(AstNodeShim(
                            name=Identifier(dep_name),
                            type=Identifier("injected"),
                            default_value=None,
                            is_injected=True
                        ))
                        parser_debug(f"  🔌 Injected dependency: {dep_name}")
                    continue
                
                # Check for action (method) definitions
                if tokens[i].type == ACTION:
                    # Parse action like in contract parser
                    action_start = i
                    i += 1  # Move past ACTION
                    
                    # Get action name (accept any token with a literal, including keywords)
                    if i < brace_end and tokens[i].literal:
                        action_name = tokens[i].literal
                        parser_debug(f"  📝 Found method: {action_name}")
                        i += 1
                        
                        # Collect parameters
                        param_list = []
                        if i < brace_end and tokens[i].type == LPAREN:
                            i += 1
                            paren_depth = 1
                            while i < brace_end and paren_depth > 0:
                                if tokens[i].type == LPAREN:
                                    paren_depth += 1
                                    i += 1
                                elif tokens[i].type == RPAREN:
                                    paren_depth -= 1
                                    if paren_depth == 0:
                                        break
                                    i += 1
                                elif tokens[i].type == IDENT:
                                    # Add parameter name
                                    param_list.append(Identifier(tokens[i].literal))
                                    i += 1
                                    # Skip type annotation (: type)
                                    if i < brace_end and tokens[i].type == COLON:
                                        i += 1  # Skip colon
                                        if i < brace_end and tokens[i].type == IDENT:
                                            i += 1  # Skip type name
                                    # Skip comma if present
                                    if i < brace_end and tokens[i].type == COMMA:
                                        i += 1
                                else:
                                    i += 1
                            i += 1  # Skip closing paren
                        
                        # Skip return type annotation if present (-> type)
                        if i < brace_end and tokens[i].type == MINUS and i + 1 < brace_end and tokens[i + 1].type == GT:
                            i += 2  # Skip ->
                            if i < brace_end and tokens[i].type == IDENT:
                                i += 1  # Skip return type
                        
                        # Find action body (between braces)
                        if i < brace_end and tokens[i].type == LBRACE:
                            action_brace_start = i
                            action_brace_count = 1
                            i += 1
                            while i < brace_end and action_brace_count > 0:
                                if tokens[i].type == LBRACE:
                                    action_brace_count += 1
                                elif tokens[i].type == RBRACE:
                                    action_brace_count -= 1
                                i += 1
                            action_brace_end = i - 1
                            
                            # Parse action body
                            body_tokens = tokens[action_brace_start + 1:action_brace_end]
                            body_statements = self._parse_block_statements(body_tokens)
                            
                            # Create block statement
                            block_stmt = BlockStatement()
                            block_stmt.statements = body_statements
                            
                            # Create action statement
                            action_stmt = ActionStatement(
                                name=Identifier(action_name),
                                parameters=param_list,
                                body=block_stmt,
                                return_type=None
                            )
                            methods.append(action_stmt)
                            parser_debug(f"  ✅ Parsed method: {action_name}")
                    continue
                
                # Parse regular properties
                if tokens[i].type == IDENT:
                    prop_name = tokens[i].literal
                    parser_debug(f"  📝 Found property name: {prop_name}")

                    # Look for colon and type
                    if i + 1 < brace_end and tokens[i + 1].type == COLON:
                        if i + 2 < brace_end:
                            prop_type = tokens[i + 2].literal
                            # Use AstNodeShim so evaluator can use .name.value
                            properties.append(AstNodeShim(
                                name=Identifier(prop_name),
                                type=Identifier(prop_type),
                                default_value=None
                            ))
                            parser_debug(f"  📝 Property: {prop_name}: {prop_type}")
                            i += 3
                            continue

                i += 1

        return EntityStatement(
            name=Identifier(entity_name),
            properties=properties,
            methods=methods,
            parent=Identifier(parent_name) if parent_name else None
        )

    def _parse_contract_statement_block(self, block_info, all_tokens):
        """Parse contract declaration block - FINAL FIXED VERSION"""
        parser_debug("🔧 [Context] Parsing contract statement")
        tokens = block_info['tokens']

        if len(tokens) < 3:
            return None

        # 1. Extract Name
        contract_name = tokens[1].literal if tokens[1].type == IDENT else "UnknownContract"
        parser_debug(f"  📝 Contract Name: {contract_name}")

        # 2. Identify Block Boundaries
        brace_start = -1
        brace_end = -1
        brace_count = 0

        for i, token in enumerate(tokens):
            if token.type == LBRACE:
                if brace_count == 0: brace_start = i
                brace_count += 1
            elif token.type == RBRACE:
                brace_count -= 1
                if brace_count == 0:
                    brace_end = i
                    break

        # List to hold storage vars (Properties) and actions
        storage_vars = []
        actions = []

        if brace_start != -1 and brace_end != -1:
            # 3. Parse Internals
            i = brace_start + 1
            while i < brace_end:
                token = tokens[i]

                # A. Handle Actions (Methods)
                if token.type == ACTION:
                    # Find the end of this action block
                    action_start = i
                    action_brace_nest = 0
                    action_brace_start_found = False
                    action_end = -1

                    j = i
                    while j < brace_end:
                        if tokens[j].type == LBRACE:
                            action_brace_nest += 1
                            action_brace_start_found = True
                        elif tokens[j].type == RBRACE:
                            action_brace_nest -= 1
                            if action_brace_start_found and action_brace_nest == 0:
                                action_end = j
                                break
                        j += 1

                    if action_end != -1:
                        action_tokens = tokens[action_start:action_end+1]

                        # Parse Action Name
                        act_name = "anonymous"
                        if action_start + 1 < len(tokens) and tokens[action_start+1].type == IDENT:
                            act_name = tokens[action_start+1].literal

                        # Parse Parameters
                        params = []
                        paren_start = -1
                        paren_end = -1
                        for k, tk in enumerate(action_tokens):
                            if tk.type == LPAREN: paren_start = k; break

                        if paren_start != -1:
                            depth = 0
                            for k in range(paren_start, len(action_tokens)):
                                if action_tokens[k].type == LPAREN: depth += 1
                                elif action_tokens[k].type == RPAREN:
                                    depth -= 1
                                    if depth == 0: paren_end = k; break

                            if paren_end > paren_start:
                                param_tokens = action_tokens[paren_start+1:paren_end]
                                for pk in param_tokens:
                                    if pk.type == IDENT:
                                        params.append(Identifier(pk.literal))

                        # Parse Body
                        body_block = BlockStatement()
                        act_brace_start = -1
                        for k, tk in enumerate(action_tokens):
                            if tk.type == LBRACE: act_brace_start = k; break

                        if act_brace_start != -1:
                             body_tokens = action_tokens[act_brace_start+1:-1]
                             body_block.statements = self._parse_block_statements(body_tokens)

                        actions.append(ActionStatement(
                            name=Identifier(act_name),
                            parameters=params,
                            body=body_block
                        ))

                        i = action_end + 1
                        continue

                # B. Handle Persistent Storage Variables
                elif token.type == PERSISTENT:
                    # Check if next token is STORAGE
                    if i + 1 < brace_end and tokens[i + 1].type == STORAGE:
                        # Move to identifier after "persistent storage"
                        i += 2
                        if i < brace_end and tokens[i].type == IDENT:
                            prop_name = tokens[i].literal
                            prop_type = "any"
                            default_val = None

                            current_idx = i + 1
                            if current_idx < brace_end and tokens[current_idx].type == COLON:
                                current_idx += 1
                                if current_idx < brace_end and tokens[current_idx].type == IDENT:
                                    prop_type = tokens[current_idx].literal
                                    current_idx += 1

                            # Check for default/initial value
                            if current_idx < brace_end and tokens[current_idx].type == ASSIGN:
                                current_idx += 1
                                if current_idx < brace_end:
                                    val_token = tokens[current_idx]
                                    if val_token.type == STRING:
                                        default_val = StringLiteral(val_token.literal)
                                    elif val_token.type == INT:
                                        default_val = IntegerLiteral(int(val_token.literal))
                                    elif val_token.type == FLOAT:
                                        default_val = FloatLiteral(float(val_token.literal))
                                    elif val_token.type == IDENT:
                                        default_val = Identifier(val_token.literal)
                                    current_idx += 1

                            # CRITICAL FIX: Use AstNodeShim so evaluator can access .name and .initial_value via dot notation
                            storage_vars.append(AstNodeShim(
                                name=Identifier(prop_name),
                                type=Identifier(prop_type),
                                initial_value=default_val, # For Contract evaluator
                                default_value=default_val  # For Entity evaluator (fallback compatibility)
                            ))

                            i = current_idx
                            continue

                # C. Handle State Variables (Properties)
                elif token.type == IDENT:
                    prop_name = token.literal

                    if i + 1 < brace_end and tokens[i+1].type == COLON:
                        prop_type = "any"
                        default_val = None

                        current_idx = i + 2
                        if current_idx < brace_end and tokens[current_idx].type == IDENT:
                            prop_type = tokens[current_idx].literal
                            current_idx += 1

                        # Check for default/initial value
                        if current_idx < brace_end and tokens[current_idx].type == ASSIGN:
                             current_idx += 1
                             if current_idx < brace_end:
                                 val_token = tokens[current_idx]
                                 if val_token.type == STRING:
                                     default_val = StringLiteral(val_token.literal)
                                 elif val_token.type == INT:
                                     default_val = IntegerLiteral(int(val_token.literal))
                                 elif val_token.type == IDENT:
                                     default_val = Identifier(val_token.literal)
                                 current_idx += 1

                        # CRITICAL FIX: Use AstNodeShim so evaluator can access .name and .initial_value via dot notation
                        # The evaluator uses `storage_var_node.name.value` and `storage_var_node.initial_value`
                        storage_vars.append(AstNodeShim(
                            name=Identifier(prop_name),
                            type=Identifier(prop_type),
                            initial_value=default_val, # For Contract evaluator
                            default_value=default_val  # For Entity evaluator (fallback compatibility)
                        ))

                        i = current_idx
                        continue

                i += 1

        # 4. Inject Name property if missing (Fixes runtime error)
        has_name = any(p.name.value == 'name' for p in storage_vars)
        if not has_name:
            storage_vars.append(AstNodeShim(
                name=Identifier("name"),
                type=Identifier("string"),
                initial_value=StringLiteral(contract_name),
                default_value=StringLiteral(contract_name)
            ))

        # 5. Create body BlockStatement containing storage vars and actions
        # Convert storage_vars to LetStatements for body
        body_statements = []
        
        # Add storage vars as state declarations
        for storage_var in storage_vars:
            body_statements.append(storage_var)
        
        # Add actions
        body_statements.extend(actions)
        
        body_block = BlockStatement()
        body_block.statements = body_statements
        
        # Also store storage_vars and actions as attributes for backward compatibility
        contract_stmt = ContractStatement(
            name=Identifier(contract_name),
            body=body_block,
            modifiers=None
        )
        
        # Add backward compatibility attributes
        contract_stmt.storage_vars = storage_vars
        contract_stmt.actions = actions
        
        return contract_stmt

    # === FIXED USE STATEMENT PARSERS ===
    def _parse_use_statement_block(self, block_info, all_tokens):
        """Enhanced use statement parser that handles both syntax styles"""
        tokens = block_info['tokens']
        parser_debug(f"    📝 Found use statement: {[t.literal for t in tokens]}")

        # Check for brace syntax: use { Name1, Name2 } from './module.zx'
        has_braces = any(t.type == LBRACE for t in tokens)

        if has_braces:
            return self._parse_use_with_braces(tokens)
        else:
            return self._parse_use_simple(tokens)

    def _parse_use_with_braces(self, tokens):
        """Parse use { names } from 'path' syntax"""
        names = []
        file_path = None

        # Find the brace section
        brace_start = -1
        brace_end = -1
        for i, token in enumerate(tokens):
            if token.type == LBRACE:
                brace_start = i
                break

        if brace_start != -1:
            # Extract names from inside braces
            i = brace_start + 1
            while i < len(tokens) and tokens[i].type != RBRACE:
                if tokens[i].type == IDENT:
                    names.append(Identifier(tokens[i].literal))
                i += 1
            brace_end = i

        # Find 'from' and file path
        if brace_end != -1 and brace_end + 1 < len(tokens):
            for i in range(brace_end + 1, len(tokens)):
                # FIX: Check for FROM token type OR identifier 'from'
                is_from = (tokens[i].type == FROM) or (tokens[i].type == IDENT and tokens[i].literal == 'from')

                if is_from:
                    if i + 1 < len(tokens) and tokens[i + 1].type == STRING:
                        file_path = tokens[i + 1].literal
                        parser_debug(f"    📝 Found import path: {file_path}")
                    break

        return UseStatement(
            file_path=file_path or "",
            names=names,
            is_named_import=True
        )

    def _parse_use_simple(self, tokens):
        """Parse simple use 'path' [as alias] syntax"""
        file_path = None
        alias = None

        for i, token in enumerate(tokens):
            if token.type == STRING:
                file_path = token.literal
            elif token.type == IDENT and token.literal == 'as':
                if i + 1 < len(tokens) and tokens[i + 1].type == IDENT:
                    alias = tokens[i + 1].literal

        return UseStatement(
            file_path=file_path or "",
            alias=alias,
            is_named_import=False
        )

    def _parse_statement_block_context(self, block_info, all_tokens):
        """Parse standalone statement blocks - use direct parsers where available"""
        subtype = block_info.get('subtype', 'unknown')
        # print(f"🔧 [Context] Parsing statement block: {subtype} (type: {type(subtype)})")

        # Use the direct parser methods
        if subtype == 'let_statement':
            return self._parse_let_statement_block(block_info, all_tokens)
        elif subtype == 'const_statement':
            return self._parse_const_statement_block(block_info, all_tokens)
        elif subtype == 'print_statement':
            return self._parse_print_statement_block(block_info, all_tokens)
        elif subtype == 'function_call_statement':
            return self._parse_function_call_statement(block_info, all_tokens)
        elif subtype == 'assignment_statement':
            return self._parse_assignment_statement(block_info, all_tokens)
        elif subtype == 'try_catch_statement':
            return self._parse_try_catch_statement(block_info, all_tokens)
        elif subtype == 'entity_statement':
            return self._parse_entity_statement_block(block_info, all_tokens)
        elif subtype == 'contract_statement':
             return self._parse_contract_statement_block(block_info, all_tokens)
        elif subtype == 'USE':
            return self._parse_use_statement_block(block_info, all_tokens)
        elif subtype == 'use_statement': # Fix subtype mismatch
            return self._parse_use_statement_block(block_info, all_tokens)
        elif subtype in {IF, FOR, WHILE, RETURN, CONTINUE, DEFER, ENUM, SANDBOX}:
            # Use the existing logic in _parse_block_statements which handles these keywords
            # print(f"🎯 [Context] Calling _parse_block_statements for subtype={subtype}")
            # print(f"🎯 [Context] block_info['tokens'] has {len(block_info.get('tokens', []))} tokens")
            stmts = self._parse_block_statements(block_info['tokens'])
            # print(f"🎯 [Context] Got {len(stmts) if stmts else 0} statements back")
            return stmts[0] if stmts else None
        else:
            return self._parse_generic_statement_block(block_info, all_tokens)

    def _parse_generic_statement_block(self, block_info, all_tokens):
        """Parse generic statement block - RETURNS ExpressionStatement"""
        tokens = block_info['tokens']
        expression = self._parse_expression(tokens)
        if expression:
            return ExpressionStatement(expression)
        return None

    # === TRY-CATCH STATEMENT PARSER ===

    def _parse_try_catch_statement(self, block_info, all_tokens):
        """Parse try-catch statement block - RETURNS TryCatchStatement"""
        parser_debug("🔧 [Context] Parsing try-catch statement block")

        tokens = block_info['tokens']

        try_block = self._parse_try_block(tokens)
        error_var = self._extract_catch_variable(tokens)
        catch_block = self._parse_catch_block(tokens)

        return TryCatchStatement(
            try_block=try_block,
            error_variable=error_var,
            catch_block=catch_block
        )

    def _parse_try_block(self, tokens):
        """Parse the try block from tokens"""
        # print("  🔧 [Try] Parsing try block")
        try_start = -1
        try_end = -1
        brace_count = 0
        in_try = False

        for i, token in enumerate(tokens):
            if token.type == TRY:
                in_try = True
            elif in_try and token.type == LBRACE:
                if brace_count == 0:
                    try_start = i + 1
                brace_count += 1
            elif in_try and token.type == RBRACE:
                brace_count -= 1
                if brace_count == 0:
                    try_end = i
                    break

        if try_start != -1 and try_end != -1 and try_end > try_start:
            try_tokens = tokens[try_start:try_end]
            # print(f"  🔧 [Try] Found {len(try_tokens)} tokens in try block: {[t.literal for t in try_tokens]}")
            try_block_statements = self._parse_block_statements(try_tokens)
            block = BlockStatement()
            block.statements = try_block_statements
            return block

        parser_debug("  ⚠️ [Try] Could not find try block content")
        return BlockStatement()

    def _parse_catch_block(self, tokens):
        """Parse the catch block from tokens"""
        # print("  🔧 [Catch] Parsing catch block")
        catch_start = -1
        catch_end = -1
        brace_count = 0
        in_catch = False

        for i, token in enumerate(tokens):
            if token.type == CATCH:
                in_catch = True
            elif in_catch and token.type == LBRACE:
                if brace_count == 0:
                    catch_start = i + 1
                brace_count += 1
            elif in_catch and token.type == RBRACE:
                brace_count -= 1
                if brace_count == 0:
                    catch_end = i
                    break

        if catch_start != -1 and catch_end != -1 and catch_end > catch_start:
            catch_tokens = tokens[catch_start:catch_end]
            # print(f"  🔧 [Catch] Found {len(catch_tokens)} tokens in catch block: {[t.literal for t in catch_tokens]}")
            catch_block_statements = self._parse_block_statements(catch_tokens)
            block = BlockStatement()
            block.statements = catch_block_statements
            return block

        parser_debug("  ⚠️ [Catch] Could not find catch block content")
        return BlockStatement()

    def _parse_block_statements(self, tokens):
        """Parse statements from a block of tokens"""
        if not tokens:
            return []
        
        statements = []
        i = 0
        # Common statement-starter tokens used by several heuristics and fallbacks
        statement_starters = {LET, CONST, DATA, PRINT, FOR, IF, WHILE, RETURN, CONTINUE, ACTION, FUNCTION, TRY, EXTERNAL, SCREEN, EXPORT, USE, DEBUG, ENTITY, CONTRACT, VERIFY, PROTECT, PERSISTENT, STORAGE, AUDIT, RESTRICT, SANDBOX, TRAIL, NATIVE, GC, INLINE, BUFFER, SIMD, DEFER, PATTERN, ENUM, STREAM, WATCH, LOG, CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE, INTERFACE, TYPE_ALIAS, MODULE, PACKAGE, USING, MIDDLEWARE, AUTH, THROTTLE, CACHE, REQUIRE}
        
        # Safety: track loop iterations to prevent infinite loops
        max_iterations = len(tokens) * 10  # Very generous limit
        iteration_count = 0
        last_i = -1
        
        while i < len(tokens):
            iteration_count += 1
            if iteration_count > max_iterations:
                parser_debug(f"⚠️ WARNING: Excessive iterations ({iteration_count}) in _parse_block_statements, possible infinite loop")
                parser_debug(f"   Current i={i}, len(tokens)={len(tokens)}, token={tokens[i].type if i < len(tokens) else 'EOF'}")
                break
            
            # Detect if we're stuck (i hasn't changed)
            if i == last_i:
                parser_debug(f"⚠️ WARNING: Parser stuck at position {i}, token type {tokens[i].type}")
                i += 1  # Force progress
                continue
            last_i = i
            
            token = tokens[i]

            # PRINT statement heuristic
            if token.type == PRINT:
                j = i + 1
                nesting = 0
                while j < len(tokens):
                    t = tokens[j]
                    if t.type in [LPAREN, LBRACKET, LBRACE]:
                        nesting += 1
                    elif t.type in [RPAREN, RBRACKET, RBRACE]:
                        nesting -= 1
                        # Stop after closing the top-level expression
                        if nesting <= 0:
                            j += 1  # Include the closing paren
                            break
                    elif nesting == 0 and t.type in [SEMICOLON]:
                        break
                    # Stop at statement keywords when not nested
                    elif nesting == 0 and t.type in statement_starters and j > i + 1:
                        break
                    j += 1

                print_tokens = tokens[i:j]
                if zexus_config.enable_debug_logs:
                    parser_debug(f"    📝 Found print statement: {[t.literal for t in print_tokens]}")

                if len(print_tokens) > 1:
                    # Parse the expression tokens (skip PRINT keyword)
                    expr_tokens = print_tokens[1:]
                    
                    # If tokens start with ( and end with ), strip them (print() syntax)
                    tokens_to_parse = expr_tokens
                    if len(expr_tokens) >= 2 and expr_tokens[0].type == LPAREN and expr_tokens[-1].type == RPAREN:
                        tokens_to_parse = expr_tokens[1:-1]
                    
                    # Split by commas to get multiple arguments
                    values = []
                    current_expr_tokens = []
                    paren_depth = 0
                    bracket_depth = 0
                    brace_depth = 0
                    
                    for token in tokens_to_parse:
                        if token.type == LPAREN:
                            paren_depth += 1
                        elif token.type == RPAREN:
                            paren_depth -= 1
                        elif token.type == LBRACKET:
                            bracket_depth += 1
                        elif token.type == RBRACKET:
                            bracket_depth -= 1
                        elif token.type == LBRACE:
                            brace_depth += 1
                        elif token.type == RBRACE:
                            brace_depth -= 1
                        
                        # Comma at depth 0 is a separator
                        if token.type == COMMA and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                            if current_expr_tokens:
                                expr = self._parse_expression(current_expr_tokens)
                                if expr:
                                    values.append(expr)
                                current_expr_tokens = []
                        else:
                            current_expr_tokens.append(token)
                    
                    # Parse the last expression
                    if current_expr_tokens:
                        expr = self._parse_expression(current_expr_tokens)
                        if expr:
                            values.append(expr)
                    
                    # If no values found, use empty string
                    if not values:
                        values = [StringLiteral("")]
                    
                    statements.append(PrintStatement(values=values))

                i = j

            # LET statement heuristic
            elif token.type == LET:
                j = i + 1
                nesting = 0
                paren_nesting = 0  # Track parentheses separately for expressions
                has_equals = False  # Track if we've seen the = sign
                
                while j < len(tokens):
                    t = tokens[j]
                    
                    # Track nesting
                    if t.type == LPAREN:
                        paren_nesting += 1
                    elif t.type == RPAREN:
                        paren_nesting -= 1
                    elif t.type == LBRACE:
                        nesting += 1
                    elif t.type == RBRACE:
                        nesting -= 1
                        if nesting < 0:
                            break
                    elif t.type == ASSIGN:
                        has_equals = True
                    
                    # Stop at semicolon (at nesting 0 and paren_nesting 0)
                    if nesting == 0 and paren_nesting == 0 and t.type == SEMICOLON:
                        break
                    
                    # CRITICAL: Detect start of new IDENT-based statements
                    # ONLY after we've collected a complete LET (has variable name, =, and value)
                    # The value is complete when we're back at paren_nesting 0 and nesting 0
                    # IMPORTANT: We need at least let var = value (minimum 4 tokens after initial =)
                    # AND we must have already seen the = sign AND be past it
                    if nesting == 0 and paren_nesting == 0 and has_equals and t.type == IDENT:
                        # We've seen the = sign and are back at nesting level 0
                        # Check if this IDENT starts a new statement
                        # BUT: Ensure we've collected at least one value token after the =
                        # Count tokens since the = sign
                        equals_pos = -1
                        for eq_idx in range(i, j):
                            if tokens[eq_idx].type == ASSIGN:
                                equals_pos = eq_idx
                                break
                        
                        # Only check for new statements if we're at least 2 positions past the =
                        # (This allows for at least one value token after =)
                        if equals_pos >= 0 and j > equals_pos + 1:
                            # NEW: Check if token is on a new line (line-based statement boundary)
                            prev_line = tokens[j-1].line if j > 0 else 0
                            curr_line = t.line
                            is_new_line = curr_line > prev_line
                            
                            lookahead_idx = j + 1
                            is_new_statement = False
                            
                            # If on a new line and current token could start a statement, it's likely a new statement
                            if is_new_line and lookahead_idx < len(tokens):
                                next_tok = tokens[lookahead_idx]
                                # IDENT followed by DOT (method call) or LPAREN (function call) on new line
                                if next_tok.type in {DOT, LPAREN}:
                                    is_new_statement = True
                            
                            # Original checks (keep for non-newline cases)
                            if not is_new_statement and lookahead_idx < len(tokens):
                                next_tok = tokens[lookahead_idx]
                                # Function call: ident(
                                # BUT NOT if previous token is DOT (method call continuation)
                                prev_tok = tokens[j-1] if j > 0 else None
                                is_method_call_continuation = prev_tok and prev_tok.type == DOT
                                
                                if next_tok.type == LPAREN and not is_method_call_continuation:
                                    is_new_statement = True
                                # Simple assignment: ident =
                                # BUT NOT if it's part of type annotation (before first =)
                                elif next_tok.type == ASSIGN and j > equals_pos + 1:
                                    is_new_statement = True
                                # Property assignment: ident.prop...=
                                elif next_tok.type == DOT:
                                    # Scan through property chain to find ASSIGN
                                    k = lookahead_idx + 1
                                    while k < len(tokens) and k < j + 10:
                                        if tokens[k].type == IDENT:
                                            k += 1
                                            if k < len(tokens):
                                                if tokens[k].type == ASSIGN:
                                                    is_new_statement = True
                                                    break
                                                elif tokens[k].type == DOT:
                                                    k += 1  # Continue chain
                                                else:
                                                    break
                                        else:
                                            break
                            
                            if is_new_statement:
                                break
                    
                    # Stop when we hit another statement starter (at nesting 0)
                    # EXCEPT: Allow IF when followed by THEN (if-then-else expression)
                    if nesting == 0 and paren_nesting == 0 and t.type in statement_starters and j > i + 1:
                        # Check if this is IF followed by THEN (expression form)
                        if t.type == IF:
                            # Look ahead for THEN to determine if this is an expression
                            is_if_expression = False
                            for k in range(j + 1, len(tokens)):
                                if tokens[k].type == THEN:
                                    is_if_expression = True
                                    break
                                elif tokens[k].type in {LBRACE, LPAREN, SEMICOLON}:
                                    # Statement indicators before THEN
                                    break
                            # print(f"[LET_IF_DEBUG] Found IF at j={j}, is_if_expression={is_if_expression}")
                            if is_if_expression:
                                # This is if-then-else expression, continue collecting
                                j += 1
                                continue
                        # Not an if-expression, so break
                        break
                    
                    j += 1

                let_tokens = tokens[i:j]
                parser_debug(f"    📝 Found let statement: {[t.literal for t in let_tokens]}")

                if len(let_tokens) >= 4 and let_tokens[1].type == IDENT:
                    var_name = let_tokens[1].literal
                    # Attempt to parse assigned value if present
                    equals_idx = -1
                    for k, tk in enumerate(let_tokens):
                        if tk.type == ASSIGN:
                            equals_idx = k
                            break

                    if equals_idx != -1 and equals_idx + 1 < len(let_tokens):
                        value_tokens = let_tokens[equals_idx + 1:]
                        if value_tokens and value_tokens[0].type == LBRACE:
                            value_expr = self._parse_map_literal(value_tokens)
                        else:
                            value_expr = self._parse_expression(value_tokens)
                        if value_expr is None:
                            value_expr = Identifier("undefined_var")
                    else:
                        value_expr = Identifier("undefined_var")

                    statements.append(LetStatement(Identifier(var_name), value_expr))

                i = j

            # DATA statement heuristic (dataclass definition)
            elif token.type == DATA:
                j = i + 1
                brace_nesting = 0
                # Find the complete data block
                while j < len(tokens):
                    if tokens[j].type == LBRACE:
                        brace_nesting += 1
                    elif tokens[j].type == RBRACE:
                        brace_nesting -= 1
                        if brace_nesting == 0:
                            j += 1  # Include the closing brace
                            break
                    j += 1
                
                data_tokens = tokens[i:j]
                parser_debug(f"    📝 Found data statement: {[t.literal for t in data_tokens]}")
                
                stmt = self._parse_data_statement(data_tokens)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            # USE statement heuristic (fallback for non-structural detection)
            elif token.type == USE:
                # This is kept for backward compatibility or nested uses
                # The structural analyzer should now catch top-level uses
                j = i + 1
                while j < len(tokens) and tokens[j].type not in [SEMICOLON]:
                    # Need to handle brace groups for complex uses
                    if tokens[j].type == LBRACE:
                        while j < len(tokens) and tokens[j].type != RBRACE:
                            j += 1
                    j += 1

                use_tokens = tokens[i:j]
                parser_debug(f"    📝 Found use statement (heuristic): {[t.literal for t in use_tokens]}")

                # Reuse the sophisticated parser
                block_info = {'tokens': use_tokens}
                stmt = self._parse_use_statement_block(block_info, tokens)
                if stmt:
                    statements.append(stmt)

                i = j
                continue

            # EXPORT statement heuristic
            elif token.type == EXPORT:
                # Check for syntactic sugar: export action/function
                if i + 1 < len(tokens) and tokens[i + 1].type in [ACTION, FUNCTION]:
                    # This is "export action name() {}" or "export function name() {}"
                    # Parse the action/function definition first
                    j = i + 2  # Start after EXPORT ACTION/FUNCTION
                    
                    # Find the function name
                    if j < len(tokens) and tokens[j].type == IDENT:
                        func_name = tokens[j].literal
                        j += 1
                        
                        # Skip parameters (find matching parens)
                        if j < len(tokens) and tokens[j].type == LPAREN:
                            paren_depth = 1
                            j += 1
                            while j < len(tokens) and paren_depth > 0:
                                if tokens[j].type == LPAREN:
                                    paren_depth += 1
                                elif tokens[j].type == RPAREN:
                                    paren_depth -= 1
                                j += 1
                        
                        # Find the body block
                        if j < len(tokens) and tokens[j].type == LBRACE:
                            brace_depth = 1
                            j += 1
                            while j < len(tokens) and brace_depth > 0:
                                if tokens[j].type == LBRACE:
                                    brace_depth += 1
                                elif tokens[j].type == RBRACE:
                                    brace_depth -= 1
                                j += 1
                        
                        # Now parse the function tokens (excluding EXPORT)
                        func_tokens = tokens[i + 1:j]
                        parser_debug(f"    📝 Found export action/function: {func_name}")
                        
                        # Parse using _parse_block_statements since we have tokens
                        func_stmts = self._parse_block_statements(func_tokens)
                        
                        if func_stmts:
                            # Add the function statement
                            statements.extend(func_stmts)
                            # Add export statement for the function name
                            statements.append(ExportStatement(names=[Identifier(func_name)]))
                        
                        i = j
                        continue
                
                # Standard export syntax
                j = i + 1
                # if the export uses a brace block, include the whole brace section
                if j < len(tokens) and tokens[j].type == LBRACE:
                    brace_nest = 0
                    while j < len(tokens):
                        if tokens[j].type == LBRACE:
                            brace_nest += 1
                        elif tokens[j].type == RBRACE:
                            brace_nest -= 1
                            if brace_nest == 0:
                                j += 1
                                break
                        j += 1
                else:
                    while j < len(tokens) and tokens[j].type not in [SEMICOLON]:
                        j += 1

                export_tokens = tokens[i:j]
                parser_debug(f"    📝 Found export statement: {[t.literal for t in export_tokens]}")

                # Extract identifier names from the token slice (tolerant)
                names = []
                k = 1
                while k < len(export_tokens):
                    tk = export_tokens[k]
                    # stop at 'to' or 'with' clause
                    if tk.type == IDENT and tk.literal not in ('to', 'with', 'default'):
                        names.append(Identifier(tk.literal))
                    k += 1

                statements.append(ExportStatement(names=names))
                i = j
                continue

            # ENTITY statement heuristic
            elif token.type == ENTITY:
                j = i + 1
                while j < len(tokens):
                    # Skip until end of entity block (brace balanced)
                    if tokens[j].type == LBRACE:
                        nest = 1
                        j += 1
                        while j < len(tokens) and nest > 0:
                            if tokens[j].type == LBRACE:
                                nest += 1
                            elif tokens[j].type == RBRACE:
                                nest -= 1
                            j += 1
                        break
                    j += 1

                entity_tokens = tokens[i:j]
                block_info = {'tokens': entity_tokens}
                stmt = self._parse_entity_statement_block(block_info, tokens)
                if stmt:
                    statements.append(stmt)
                i = j
                continue

            # EXTERNAL statement heuristic
            elif token.type == EXTERNAL:
                j = i + 1
                # Simple syntax: external identifier;
                # Full syntax: external action identifier from "module";
                while j < len(tokens) and tokens[j].type not in [SEMICOLON]:
                    j += 1

                external_tokens = tokens[i:j]
                parser_debug(f"    📝 Found external statement: {[t.literal for t in external_tokens]}")

                # Parse using the main parser's parse_external_declaration (lazy import Parser)
                Parser = get_parser_class()
                temp_parser = Parser(external_tokens)
                temp_parser.next_token()
                stmt = temp_parser.parse_external_declaration()
                if stmt:
                    statements.append(stmt)
                else:
                    print(f"    ⚠️ Failed to parse external statement")

                i = j
                continue

            # ACTION (function-like) statement heuristic
            elif token.type == ACTION:
                j = i + 1
                stmt_tokens = [token]
                brace_nest = 0
                paren_nest = 0
                # Collect until the matching closing brace for the action body
                while j < len(tokens):
                    tj = tokens[j]
                    stmt_tokens.append(tj)
                    if tj.type == LPAREN:
                        paren_nest += 1
                    elif tj.type == RPAREN:
                        if paren_nest > 0:
                            paren_nest -= 1
                    elif tj.type == LBRACE:
                        brace_nest += 1
                    elif tj.type == RBRACE:
                        brace_nest -= 1
                        if brace_nest == 0:
                            j += 1
                            break
                    j += 1

                parser_debug(f"    📝 Found action statement: {[t.literal for t in stmt_tokens]}")

                # Extract name, params and body
                action_name = None
                params = []
                body_block = BlockStatement()

                if len(stmt_tokens) >= 2 and stmt_tokens[1].type == IDENT:
                    action_name = stmt_tokens[1].literal

                # find parameter list
                paren_start = None
                paren_end = None
                for k, tk in enumerate(stmt_tokens):
                    if tk.type == LPAREN:
                        paren_start = k
                        break
                if paren_start is not None:
                    depth = 0
                    for k in range(paren_start, len(stmt_tokens)):
                        if stmt_tokens[k].type == LPAREN:
                            depth += 1
                        elif stmt_tokens[k].type == RPAREN:
                            depth -= 1
                            if depth == 0:
                                paren_end = k
                                break
                if paren_start is not None and paren_end is not None and paren_end > paren_start + 1:
                    inner = stmt_tokens[paren_start+1:paren_end]
                    # collect identifiers as parameters
                    cur = []
                    for tk in inner:
                        if tk.type == IDENT:
                            params.append(Identifier(tk.literal))

                # find body tokens between the outermost braces
                brace_start = None
                brace_end = None
                for k, tk in enumerate(stmt_tokens):
                    if tk.type == LBRACE:
                        brace_start = k
                        break
                if brace_start is not None:
                    depth = 0
                    for k in range(brace_start, len(stmt_tokens)):
                        if stmt_tokens[k].type == LBRACE:
                            depth += 1
                        elif stmt_tokens[k].type == RBRACE:
                            depth -= 1
                            if depth == 0:
                                brace_end = k
                                break
                if brace_start is not None and brace_end is not None and brace_end > brace_start + 1:
                    inner_body = stmt_tokens[brace_start+1:brace_end]
                    body_block.statements = self._parse_block_statements(inner_body)

                statements.append(ActionStatement(
                    name=Identifier(action_name if action_name else 'anonymous'),
                    parameters=params,
                    body=body_block
                ))

                i = j
                continue
            
            # FUNCTION statement heuristic (similar to ACTION)
            elif token.type == FUNCTION:
                j = i + 1
                stmt_tokens = [token]
                brace_nest = 0
                paren_nest = 0
                # Collect until the matching closing brace for the function body
                while j < len(tokens):
                    tj = tokens[j]
                    stmt_tokens.append(tj)
                    if tj.type == LPAREN:
                        paren_nest += 1
                    elif tj.type == RPAREN:
                        if paren_nest > 0:
                            paren_nest -= 1
                    elif tj.type == LBRACE:
                        brace_nest += 1
                    elif tj.type == RBRACE:
                        brace_nest -= 1
                        if brace_nest == 0:
                            j += 1
                            break
                    j += 1

                parser_debug(f"    📝 Found function statement: {[t.literal for t in stmt_tokens]}")

                # Extract name, params and body
                function_name = None
                params = []
                body_block = BlockStatement()

                if len(stmt_tokens) >= 2 and stmt_tokens[1].type == IDENT:
                    function_name = stmt_tokens[1].literal

                # find parameter list
                paren_start = None
                paren_end = None
                for k, tk in enumerate(stmt_tokens):
                    if tk.type == LPAREN:
                        paren_start = k
                        break
                if paren_start is not None:
                    depth = 0
                    for k in range(paren_start, len(stmt_tokens)):
                        if stmt_tokens[k].type == LPAREN:
                            depth += 1
                        elif stmt_tokens[k].type == RPAREN:
                            depth -= 1
                            if depth == 0:
                                paren_end = k
                                break
                    if paren_end is not None:
                        param_tokens = stmt_tokens[paren_start + 1:paren_end]
                        params = [Identifier(t.literal) for t in param_tokens if t.type == IDENT]

                # find body
                brace_start = None
                for k, tk in enumerate(stmt_tokens):
                    if tk.type == LBRACE:
                        brace_start = k
                        break
                if brace_start is not None and brace_start + 1 < len(stmt_tokens):
                    inner_body = stmt_tokens[brace_start + 1:-1]
                    body_block.statements = self._parse_block_statements(inner_body)

                statements.append(FunctionStatement(
                    name=Identifier(function_name if function_name else 'anonymous'),
                    parameters=params,
                    body=body_block
                ))

                i = j
                continue

            # MODULE statement heuristic
            elif token.type == MODULE:
                j = i + 1
                stmt_tokens = [token]
                brace_nest = 0
                
                # Collect until the matching closing brace for the module body
                while j < len(tokens):
                    tj = tokens[j]
                    stmt_tokens.append(tj)
                    if tj.type == LBRACE:
                        brace_nest += 1
                    elif tj.type == RBRACE:
                        brace_nest -= 1
                        if brace_nest == 0:
                            j += 1
                            break
                    j += 1
                
                parser_debug(f"    📝 Found module statement: {[t.literal for t in stmt_tokens]}")
                
                module_name = None
                body_block = BlockStatement()
                
                if len(stmt_tokens) >= 2 and stmt_tokens[1].type == IDENT:
                    module_name = stmt_tokens[1].literal
                
                # find body
                brace_start = None
                for k, tk in enumerate(stmt_tokens):
                    if tk.type == LBRACE:
                        brace_start = k
                        break
                if brace_start is not None and brace_start + 1 < len(stmt_tokens):
                    inner_body = stmt_tokens[brace_start + 1:-1]
                    body_block.statements = self._parse_block_statements(inner_body)
                
                statements.append(ModuleStatement(
                    name=Identifier(module_name if module_name else 'anonymous'),
                    body=body_block
                ))
                
                i = j
                continue

            # PACKAGE statement heuristic
            elif token.type == PACKAGE:
                j = i + 1
                stmt_tokens = [token]
                brace_nest = 0
                
                # Collect until the matching closing brace for the package body
                while j < len(tokens):
                    tj = tokens[j]
                    stmt_tokens.append(tj)
                    if tj.type == LBRACE:
                        brace_nest += 1
                    elif tj.type == RBRACE:
                        brace_nest -= 1
                        if brace_nest == 0:
                            j += 1
                            break
                    j += 1
                
                parser_debug(f"    📝 Found package statement: {[t.literal for t in stmt_tokens]}")
                
                package_name = ""
                k = 1
                while k < len(stmt_tokens) and stmt_tokens[k].type != LBRACE:
                    package_name += stmt_tokens[k].literal
                    k += 1
                
                # find body
                brace_start = None
                for k, tk in enumerate(stmt_tokens):
                    if tk.type == LBRACE:
                        brace_start = k
                        break
                if brace_start is not None and brace_start + 1 < len(stmt_tokens):
                    inner_body = stmt_tokens[brace_start + 1:-1]
                    body_block = BlockStatement()
                    body_block.statements = self._parse_block_statements(inner_body)
                    body = body_block
                else:
                    body = BlockStatement()
                
                statements.append(PackageStatement(
                    name=Identifier(package_name if package_name else 'anonymous'),
                    body=body
                ))
                
                i = j
                continue

            elif token.type == WATCH:
                j = i + 1
                stmt_tokens = [token]
                brace_nest = 0
                
                # Collect until the matching closing brace for the watch body
                while j < len(tokens):
                    tj = tokens[j]
                    stmt_tokens.append(tj)
                    if tj.type == LBRACE:
                        brace_nest += 1
                    elif tj.type == RBRACE:
                        brace_nest -= 1
                        if brace_nest == 0:
                            j += 1
                            break
                    j += 1
                
                parser_debug(f"    📝 Found watch statement: {[t.literal for t in stmt_tokens]}")
                
                block_info = {'tokens': stmt_tokens}
                stmt = self._parse_watch_statement(block_info, tokens)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue
            
            # LOG statement heuristic: log > filename OR log >> filename OR log << filename
            elif token.type == LOG:
                j = i + 1
                # Check for <<, >>, or > operator
                if j < len(tokens) and tokens[j].type == IMPORT_OP:
                    # Import mode: log << file (import and execute code from file)
                    j += 1
                    
                    # Get filepath (can be STRING or IDENT)
                    if j < len(tokens) and tokens[j].type in [STRING, IDENT]:
                        filepath_token = tokens[j]
                        parser_debug(f"    📝 Found import log statement: log << {filepath_token.literal}")
                        
                        # Create a simple string literal or identifier expression
                        if filepath_token.type == STRING:
                            filepath_expr = StringLiteral(filepath_token.literal)
                        else:
                            filepath_expr = Identifier(filepath_token.literal)
                        
                        statements.append(ImportLogStatement(filepath_expr))
                        j += 1
                    else:
                        parser_debug(f"    ⚠️ Expected filepath after 'log <<'")
                    
                    i = j
                    continue
                    
                # Expect > or >> symbol for output redirection
                append_mode = True  # Default to append
                if j < len(tokens) and tokens[j].type == APPEND:
                    # Explicit append: log >> file
                    append_mode = True
                    j += 1
                elif j < len(tokens) and tokens[j].type == GT:
                    # Write mode: log > file (still appends within scope, but clearer intent)
                    append_mode = True  # Keep append for scope safety
                    j += 1
                else:
                    parser_debug(f"    ⚠️ Expected '>', '>>', or '<<' after 'log'")
                    i = j
                    continue
                
                # Get filepath (can be STRING or IDENT)
                if j < len(tokens) and tokens[j].type in [STRING, IDENT]:
                    filepath_token = tokens[j]
                    mode_str = ">>" if tokens[i+1].type == APPEND else ">"
                    parser_debug(f"    📝 Found log statement: log {mode_str} {filepath_token.literal}")
                    
                    # Create a simple string literal or identifier expression
                    if filepath_token.type == STRING:
                        filepath_expr = StringLiteral(filepath_token.literal)
                    else:
                        filepath_expr = Identifier(filepath_token.literal)
                    
                    statements.append(LogStatement(filepath_expr, append_mode))
                    j += 1
                else:
                    parser_debug(f"    ⚠️ Expected filepath after 'log {mode_str}'")
                
                i = j
                continue
            
            elif token.type == IF:
                # Check if this is an if-then-else expression or an if statement
                # Look ahead for THEN token to determine
                is_expression_form = False
                for k in range(i + 1, len(tokens)):
                    if tokens[k].type == THEN:
                        is_expression_form = True
                        break
                    elif tokens[k].type in [LBRACE, COLON, LPAREN]:
                        # Found statement indicators before THEN
                        break
                
                if is_expression_form:
                    # This is if-then-else expression - skip parsing it as statement
                    # It will be parsed as part of the containing expression
                    parser_debug(f"    ℹ️ Skipping IF at {i} - detected as if-then-else expression")
                    i += 1
                    continue
                
                # Parse IF statement directly here
                j = i + 1
                
                # Collect condition tokens (between IF and { or :)
                cond_tokens = []
                paren_depth = 0
                skipped_outer_paren = False
                
                while j < len(tokens) and tokens[j].type not in [LBRACE, COLON]:
                    # Handle outer parentheses for the condition
                    if tokens[j].type == LPAREN:
                        if len(cond_tokens) == 0 and paren_depth == 0:
                            # This might be wrapping the whole condition - skip it tentatively
                            j += 1
                            paren_depth += 1
                            skipped_outer_paren = True
                            continue
                        else:
                            paren_depth += 1
                    
                    elif tokens[j].type == RPAREN:
                        paren_depth -= 1
                        # Only break if we're closing the tentatively skipped outer paren
                        # AND there are more tokens after it (not end of condition)
                        if paren_depth == 0 and skipped_outer_paren and len(cond_tokens) > 0:
                            j += 1
                            # Check if next token is { or : (end of condition)
                            if j < len(tokens) and tokens[j].type in [LBRACE, COLON]:
                                break
                            # Otherwise continue collecting - the paren wasn't wrapping the whole condition
                            skipped_outer_paren = False
                            continue
                    
                    cond_tokens.append(tokens[j])
                    j += 1
                
                # print(f"  [IF_COND] Condition tokens: {[t.literal for t in cond_tokens]}")
                
                # Parse condition expression
                condition = self._parse_expression(cond_tokens) if cond_tokens else Identifier("true")
                
                # Collect consequence block tokens
                if j < len(tokens) and tokens[j].type == LBRACE:
                    # Brace-style block
                    j += 1  # Skip LBRACE
                    inner_tokens = []
                    depth = 1
                    while j < len(tokens) and depth > 0:
                        if tokens[j].type == LBRACE:
                            depth += 1
                        elif tokens[j].type == RBRACE:
                            depth -= 1
                            if depth == 0:
                                break
                        inner_tokens.append(tokens[j])
                        j += 1
                    
                    consequence = BlockStatement()
                    consequence.statements = self._parse_block_statements(inner_tokens)
                    j += 1  # Skip closing RBRACE
                elif j < len(tokens) and tokens[j].type == COLON:
                    # Colon-style block - collect until next statement keyword or dedent
                    j += 1  # Skip COLON
                    inner_tokens = []
                    # Collect tokens until we hit a keyword that starts a new statement at the same level
                    while j < len(tokens):
                        if tokens[j].type in [IF, ELIF, ELSE, WHILE, FOR, ACTION, FUNCTION, LET, CONST, RETURN, CONTINUE, USE, EXPORT]:
                            # Found a new statement, stop here
                            break
                        inner_tokens.append(tokens[j])
                        j += 1
                    
                    consequence = BlockStatement()
                    consequence.statements = self._parse_block_statements(inner_tokens)
                else:
                    consequence = BlockStatement()
                
                # Check for elif/else
                elif_parts = []
                alternative = None
                
                while j < len(tokens) and tokens[j].type in [ELIF, ELSE]:
                    if tokens[j].type == ELIF:
                        j += 1
                        # Parse elif condition
                        elif_cond_tokens = []
                        elif_paren_depth = 0
                        elif_skipped_outer = False
                        while j < len(tokens) and tokens[j].type not in [LBRACE, COLON]:
                            if tokens[j].type == LPAREN:
                                if len(elif_cond_tokens) == 0 and elif_paren_depth == 0:
                                    j += 1
                                    elif_paren_depth += 1
                                    elif_skipped_outer = True
                                    continue
                                else:
                                    elif_paren_depth += 1
                            elif tokens[j].type == RPAREN:
                                elif_paren_depth -= 1
                                if elif_paren_depth == 0 and elif_skipped_outer and len(elif_cond_tokens) > 0:
                                    j += 1
                                    if j < len(tokens) and tokens[j].type in [LBRACE, COLON]:
                                        break
                                    elif_skipped_outer = False
                                    continue
                            elif_cond_tokens.append(tokens[j])
                            j += 1
                        
                        elif_cond = self._parse_expression(elif_cond_tokens) if elif_cond_tokens else Identifier("true")
                        
                        # Collect elif block
                        if j < len(tokens) and tokens[j].type == LBRACE:
                            # Brace-style
                            j += 1
                            elif_inner = []
                            depth = 1
                            while j < len(tokens) and depth > 0:
                                if tokens[j].type == LBRACE:
                                    depth += 1
                                elif tokens[j].type == RBRACE:
                                    depth -= 1
                                    if depth == 0:
                                        break
                                elif_inner.append(tokens[j])
                                j += 1
                            elif_block = BlockStatement()
                            elif_block.statements = self._parse_block_statements(elif_inner)
                            j += 1
                        elif j < len(tokens) and tokens[j].type == COLON:
                            # Colon-style
                            j += 1
                            elif_inner = []
                            while j < len(tokens):
                                if tokens[j].type in [IF, ELIF, ELSE, WHILE, FOR, ACTION, FUNCTION, LET, CONST, RETURN, CONTINUE, USE, EXPORT]:
                                    break
                                elif_inner.append(tokens[j])
                                j += 1
                            elif_block = BlockStatement()
                            elif_block.statements = self._parse_block_statements(elif_inner)
                        else:
                            elif_block = BlockStatement()
                        
                        elif_parts.append((elif_cond, elif_block))
                    
                    elif tokens[j].type == ELSE:
                        j += 1
                        # Collect else block
                        if j < len(tokens) and tokens[j].type == LBRACE:
                            # Brace-style
                            j += 1
                            else_inner = []
                            depth = 1
                            while j < len(tokens) and depth > 0:
                                if tokens[j].type == LBRACE:
                                    depth += 1
                                elif tokens[j].type == RBRACE:
                                    depth -= 1
                                    if depth == 0:
                                        break
                                else_inner.append(tokens[j])
                                j += 1
                            alternative = BlockStatement()
                            alternative.statements = self._parse_block_statements(else_inner)
                            j += 1
                        elif j < len(tokens) and tokens[j].type == COLON:
                            # Colon-style
                            j += 1
                            else_inner = []
                            while j < len(tokens):
                                if tokens[j].type in [IF, ELIF, ELSE, WHILE, FOR, ACTION, FUNCTION, LET, CONST, RETURN, CONTINUE, USE, EXPORT]:
                                    break
                                else_inner.append(tokens[j])
                                j += 1
                            alternative = BlockStatement()
                            alternative.statements = self._parse_block_statements(else_inner)
                        break
                
                stmt = IfStatement(
                    condition=condition,
                    consequence=consequence,
                    elif_parts=elif_parts,
                    alternative=alternative
                )
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            elif token.type == TRY:
                j = i + 1
                stmt_tokens = [token]
                
                # Collect try block
                brace_nest = 0
                while j < len(tokens):
                    t = tokens[j]
                    stmt_tokens.append(t)
                    if t.type == LBRACE:
                        brace_nest += 1
                    elif t.type == RBRACE:
                        brace_nest -= 1
                        if brace_nest == 0:
                            j += 1
                            break
                    j += 1
                
                # Check for catch
                if j < len(tokens) and tokens[j].type == CATCH:
                    stmt_tokens.append(tokens[j])
                    j += 1
                    
                    # Optional error variable (catch (e))
                    if j < len(tokens) and tokens[j].type == LPAREN:
                        while j < len(tokens) and tokens[j].type != LBRACE:
                            stmt_tokens.append(tokens[j])
                            j += 1
                    
                    # Collect catch block
                    brace_nest = 0
                    while j < len(tokens):
                        t = tokens[j]
                        stmt_tokens.append(t)
                        if t.type == LBRACE:
                            brace_nest += 1
                        elif t.type == RBRACE:
                            brace_nest -= 1
                            if brace_nest == 0:
                                j += 1
                                break
                        j += 1
                
                block_info = {'tokens': stmt_tokens}
                stmt = self._parse_try_catch_statement(block_info, tokens)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            elif token.type == RETURN:
                # Parse RETURN statement directly
                j = i + 1
                value_tokens = []
                nesting = 0  # Track brace/paren nesting
                
                # Collect tokens until semicolon at depth 0, or next statement at depth 0
                # This properly handles: return function() { return 42; };
                while j < len(tokens):
                    t = tokens[j]
                    
                    # Track nesting for braces, parens, brackets
                    if t.type in {LPAREN, LBRACE, LBRACKET}:
                        nesting += 1
                    elif t.type in {RPAREN, RBRACE, RBRACKET}:
                        nesting -= 1
                        # Don't go negative - if we hit RBRACE at nesting 0, stop
                        if nesting < 0:
                            break
                    
                    # Only check termination conditions at nesting level 0
                    if nesting == 0:
                        if t.type == SEMICOLON:
                            break
                        # Don't break on statement starters that are inside braces
                        # Only break if it's truly a new statement (e.g., not FUNCTION inside return expr)
                        if t.type in statement_starters and t.type not in {FUNCTION, ACTION, RETURN}:
                            break
                    
                    value_tokens.append(t)
                    j += 1
                
                # Parse the return value
                value = None
                if value_tokens:
                    value = self._parse_expression(value_tokens)
                
                stmt = ReturnStatement(value)
                if stmt:
                    statements.append(stmt)
                
                # Skip trailing semicolon if present
                if j < len(tokens) and tokens[j].type == SEMICOLON:
                    j += 1
                
                i = j
                continue

            elif token.type == CONTINUE:
                # Parse CONTINUE statement directly (simple statement, no value)
                stmt = ContinueStatement()
                statements.append(stmt)
                
                # Skip to next token (and skip semicolon if present)
                j = i + 1
                if j < len(tokens) and tokens[j].type == SEMICOLON:
                    j += 1
                
                i = j
                continue

            elif token.type == BREAK:
                # Parse BREAK statement directly (simple statement, no value)
                stmt = BreakStatement()
                statements.append(stmt)
                
                # Skip to next token (and skip semicolon if present)
                j = i + 1
                if j < len(tokens) and tokens[j].type == SEMICOLON:
                    j += 1
                
                i = j
                continue

            elif token.type == THROW:
                # Parse THROW statement - throw error_message
                j = i + 1
                # Collect message expression tokens until semicolon or newline
                msg_tokens = []
                while j < len(tokens) and tokens[j].type not in {SEMICOLON, RBRACE}:
                    msg_tokens.append(tokens[j])
                    j += 1
                
                # Parse message expression
                if msg_tokens:
                    msg_expr = self._parse_expression(msg_tokens)
                else:
                    msg_expr = StringLiteral(value="Error")
                
                stmt = ThrowStatement(message=msg_expr)
                statements.append(stmt)
                
                # Skip semicolon if present
                if j < len(tokens) and tokens[j].type == SEMICOLON:
                    j += 1
                
                i = j
                continue

            elif token.type == WHILE:
                # Parse WHILE statement directly
                j = i + 1
                stmt_tokens = [token]
                
                # Collect condition tokens (between WHILE and {)
                cond_tokens = []
                paren_depth = 0
                has_parens = False
                
                # Check if condition has parentheses
                if j < len(tokens) and tokens[j].type == LPAREN:
                    has_parens = True
                    j += 1
                    paren_depth = 1
                    
                    # Collect condition tokens inside parens
                    while j < len(tokens) and paren_depth > 0:
                        if tokens[j].type == LPAREN:
                            paren_depth += 1
                            cond_tokens.append(tokens[j])
                        elif tokens[j].type == RPAREN:
                            paren_depth -= 1
                            if paren_depth == 0:
                                j += 1  # Skip closing paren
                                break
                            cond_tokens.append(tokens[j])
                        else:
                            cond_tokens.append(tokens[j])
                        j += 1
                else:
                    # No parentheses - collect tokens until we hit {
                    while j < len(tokens) and tokens[j].type != LBRACE:
                        cond_tokens.append(tokens[j])
                        j += 1
                
                # Parse condition
                condition = self._parse_expression(cond_tokens) if cond_tokens else Identifier("true")
                
                # Collect body block (between { and })
                body_block = BlockStatement()
                if j < len(tokens) and tokens[j].type == LBRACE:
                    j += 1  # Skip opening brace
                    body_tokens = []
                    brace_nest = 1
                    
                    while j < len(tokens) and brace_nest > 0:
                        if tokens[j].type == LBRACE:
                            brace_nest += 1
                        elif tokens[j].type == RBRACE:
                            brace_nest -= 1
                            if brace_nest == 0:
                                j += 1  # Skip closing brace
                                break
                        body_tokens.append(tokens[j])
                        j += 1
                    
                    # Recursively parse body statements
                    body_block.statements = self._parse_block_statements(body_tokens)
                
                parser_debug(f"    📝 Found while statement with {len(body_block.statements)} body statements")
                
                stmt = WhileStatement(condition=condition, body=body_block)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            elif token.type == FOR:
                # Parse FOR EACH statement directly
                j = i + 1
                stmt_tokens = [token]
                
                # Expect EACH keyword
                if j < len(tokens) and tokens[j].type == EACH:
                    j += 1
                    
                    # Collect iterator variable name
                    item_name = None
                    if j < len(tokens) and tokens[j].type == IDENT:
                        item_name = tokens[j].literal
                        j += 1
                    
                    # Expect IN keyword
                    if j < len(tokens) and tokens[j].type == IN:
                        j += 1
                        
                        # Collect iterable expression tokens (until {)
                        iterable_tokens = []
                        while j < len(tokens) and tokens[j].type != LBRACE:
                            iterable_tokens.append(tokens[j])
                            j += 1
                        
                        # Parse iterable
                        iterable = self._parse_expression(iterable_tokens) if iterable_tokens else Identifier("[]")
                        
                        # Collect body block (between { and })
                        body_block = BlockStatement()
                        if j < len(tokens) and tokens[j].type == LBRACE:
                            j += 1  # Skip opening brace
                            body_tokens = []
                            brace_nest = 1
                            
                            while j < len(tokens) and brace_nest > 0:
                                if tokens[j].type == LBRACE:
                                    brace_nest += 1
                                elif tokens[j].type == RBRACE:
                                    brace_nest -= 1
                                    if brace_nest == 0:
                                        j += 1  # Skip closing brace
                                        break
                                body_tokens.append(tokens[j])
                                j += 1
                            
                            # Recursively parse body statements
                            body_block.statements = self._parse_block_statements(body_tokens)
                        
                        parser_debug(f"    📝 Found for each statement with {len(body_block.statements)} body statements")
                        
                        stmt = ForEachStatement(
                            item=Identifier(item_name if item_name else 'item'),
                            iterable=iterable,
                            body=body_block
                        )
                        if stmt:
                            statements.append(stmt)
                
                i = j
                continue

            elif token.type == DEFER:
                # Parse DEFER statement directly
                j = i + 1
                stmt_tokens = [token]
                
                # Collect the code block (between { and })
                code_block = BlockStatement()
                if j < len(tokens) and tokens[j].type == LBRACE:
                    j += 1  # Skip opening brace
                    block_tokens = []
                    brace_nest = 1
                    
                    while j < len(tokens) and brace_nest > 0:
                        if tokens[j].type == LBRACE:
                            brace_nest += 1
                        elif tokens[j].type == RBRACE:
                            brace_nest -= 1
                            if brace_nest == 0:
                                j += 1  # Skip closing brace
                                break
                        block_tokens.append(tokens[j])
                        j += 1
                    
                    # Recursively parse code block statements
                    code_block.statements = self._parse_block_statements(block_tokens)
                
                parser_debug(f"    📝 Found defer statement with {len(code_block.statements)} statements")
                
                stmt = DeferStatement(code_block=code_block)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            elif token.type == ENUM:
                # Parse ENUM statement directly
                j = i + 1
                stmt_tokens = [token]
                
                # Get enum name
                enum_name = None
                if j < len(tokens) and tokens[j].type == IDENT:
                    enum_name = tokens[j].literal
                    j += 1
                
                # Parse members between { and }
                members = []
                if j < len(tokens) and tokens[j].type == LBRACE:
                    j += 1  # Skip opening brace
                    
                    while j < len(tokens) and tokens[j].type != RBRACE:
                        if tokens[j].type == IDENT:
                            member_name = tokens[j].literal
                            member_value = None
                            j += 1
                            
                            # Check for = value
                            if j < len(tokens) and tokens[j].type == ASSIGN:
                                j += 1  # Skip =
                                if j < len(tokens) and tokens[j].type in [INT, STRING]:
                                    member_value = tokens[j].literal
                                    j += 1
                            
                            members.append(EnumMember(member_name, member_value))
                        
                        # Skip commas
                        if j < len(tokens) and tokens[j].type == COMMA:
                            j += 1
                        else:
                            break
                    
                    if j < len(tokens) and tokens[j].type == RBRACE:
                        j += 1  # Skip closing brace
                
                parser_debug(f"    📝 Found enum '{enum_name}' with {len(members)} members")
                
                stmt = EnumStatement(enum_name, members)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            elif token.type == SANDBOX:
                # Parse SANDBOX statement directly
                j = i + 1
                
                # Parse body block between { and }
                body = None
                if j < len(tokens) and tokens[j].type == LBRACE:
                    j += 1  # Skip opening brace
                    block_tokens = []
                    brace_nest = 1
                    
                    while j < len(tokens) and brace_nest > 0:
                        if tokens[j].type == LBRACE:
                            brace_nest += 1
                        elif tokens[j].type == RBRACE:
                            brace_nest -= 1
                            if brace_nest == 0:
                                j += 1  # Skip closing brace
                                break
                        block_tokens.append(tokens[j])
                        j += 1
                    
                    # Recursively parse body statements
                    body_statements = self._parse_block_statements(block_tokens)
                    body = BlockStatement()
                    body.statements = body_statements
                
                parser_debug(f"    📝 Found sandbox statement with {len(body.statements) if body else 0} statements")
                
                stmt = SandboxStatement(body=body, policy=None)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            elif token.type == REQUIRE:
                # Parse REQUIRE statement: require(condition, message) or require condition { tolerance_block }
                j = i + 1
                
                # Collect tokens until semicolon OR until after tolerance block closes
                require_tokens = [token]
                brace_nest = 0
                while j < len(tokens):
                    tj = tokens[j]
                    
                    # Track brace nesting for tolerance blocks
                    if tj.type == LBRACE:
                        brace_nest += 1
                    elif tj.type == RBRACE:
                        brace_nest -= 1
                    
                    require_tokens.append(tj)
                    j += 1
                    
                    # Stop at semicolon when not inside braces
                    if tj.type == SEMICOLON and brace_nest == 0:
                        break
                    
                    # Stop after tolerance block closes (if there was one)
                    if brace_nest == 0 and len(require_tokens) > 1 and require_tokens[-2].type == RBRACE:
                        break
                
                parser_debug(f"    📝 Found require statement with {len(require_tokens)} tokens: {[t.literal for t in require_tokens[:20]]}")
                
                # Use the handler to parse it
                block_info = {'tokens': require_tokens}
                stmt = self._parse_require_statement(block_info, tokens)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            elif token.type == ATOMIC:
                # Parse ATOMIC statement: atomic { statements }
                j = i + 1
                
                # Collect tokens until after closing brace
                atomic_tokens = [token]
                brace_nest = 0
                while j < len(tokens):
                    tj = tokens[j]
                    if tj.type == LBRACE:
                        brace_nest += 1
                    elif tj.type == RBRACE:
                        brace_nest -= 1
                        if brace_nest == 0:
                            atomic_tokens.append(tj)
                            j += 1
                            break
                    atomic_tokens.append(tj)
                    j += 1
                
                parser_debug(f"    📝 Found atomic statement with {len(atomic_tokens)} tokens")
                
                # Use the handler to parse it
                block_info = {'tokens': atomic_tokens}
                stmt = self._parse_atomic_statement(block_info, tokens)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            elif token.type == ASYNC:
                # Parse ASYNC expression: async <expression>
                j = i + 1
                
                # Collect tokens for the async expression (usually just async + ident + parens)
                async_tokens = [token]
                nesting = 0
                while j < len(tokens):
                    tj = tokens[j]
                    if tj.type == LPAREN:
                        nesting += 1
                    elif tj.type == RPAREN:
                        nesting -= 1
                        async_tokens.append(tj)
                        if nesting == 0:
                            j += 1
                            break
                    elif nesting == 0 and tj.type in statement_starters:
                        break
                    async_tokens.append(tj)
                    j += 1
                
                parser_debug(f"    📝 Found async expression with {len(async_tokens)} tokens")
                
                # Use the handler to parse it
                block_info = {'tokens': async_tokens}
                stmt = self._parse_async_expression_block(block_info, tokens)
                if stmt:
                    # Wrap in ExpressionStatement since async expressions are expressions
                    statements.append(ExpressionStatement(stmt))
                
                i = j
                continue

            # Fallback: attempt to parse as expression
            else:
                j = i
                run_tokens = []
                nesting = 0
                while j < len(tokens):
                    t = tokens[j]
                    
                    # Before adding this token, check if it starts a NEW assignment statement
                    # Only check when we've completed a previous statement (e.g., after function call)
                    # Don't check if the last token was DOT (we're in the middle of property access)
                    if nesting == 0 and len(run_tokens) > 0 and t.type == IDENT:
                        # Only detect new assignment if previous token suggests end of previous statement
                        # E.g., after RPAREN (end of function call) or after a complete value
                        prev_token = run_tokens[-1] if run_tokens else None
                        if prev_token and prev_token.type not in {DOT, LPAREN, LBRACKET, LBRACE}:
                            # Check if this starts a new statement (assignment or function call)
                            k = j + 1
                            is_new_statement_start = False
                            
                            if k < len(tokens):
                                next_tok = tokens[k]
                                # Function call: ident(
                                if next_tok.type == LPAREN:
                                    is_new_statement_start = True
                                # Assignment: ident = or ident.prop =
                                elif next_tok.type == ASSIGN:
                                    is_new_statement_start = True
                                elif next_tok.type == DOT:
                                    # Property assignment: scan for ASSIGN
                                    while k < len(tokens) and k < j + 10:
                                        if tokens[k].type == DOT:
                                            k += 1
                                            if k < len(tokens) and tokens[k].type == IDENT:
                                                k += 1
                                            else:
                                                break
                                        elif tokens[k].type == ASSIGN:
                                            is_new_statement_start = True
                                            break
                                        else:
                                            break
                            
                            if is_new_statement_start:
                                break
                    
                    # update nesting for parentheses/brackets/braces
                    if t.type in {LPAREN, LBRACE, LBRACKET}:
                        nesting += 1
                    elif t.type in {RPAREN, RBRACE, RBRACKET}:
                        if nesting > 0:
                            nesting -= 1

                    # stop at top-level statement terminators or starters
                    # BUT: Don't stop at keywords that appear after a DOT (they're method/property names)
                    if nesting == 0:
                        # Check if this is a keyword after a dot (method/property access)
                        is_after_dot = (len(run_tokens) > 0 and run_tokens[-1].type == DOT)
                        if not is_after_dot and (t.type in [SEMICOLON, LBRACE, RBRACE] or t.type in statement_starters):
                            break

                    run_tokens.append(t)
                    j += 1

                if run_tokens:
                    # Check if this is an assignment (contains ASSIGN token)
                    has_assign = any(t.type == ASSIGN for t in run_tokens)
                    if has_assign:
                        # Parse as assignment statement
                        block_info = {'tokens': run_tokens}
                        stmt = self._parse_assignment_statement(block_info, tokens)
                        if stmt:
                            # Assignment parser returns AssignmentExpression, wrap in ExpressionStatement
                            if not isinstance(stmt, Statement):
                                stmt = ExpressionStatement(stmt)
                            statements.append(stmt)
                    else:
                        # Parse as regular expression
                        expr = self._parse_expression(run_tokens)
                        if expr:
                            statements.append(ExpressionStatement(expr))
                # Advance to the token after the run (or by one to avoid infinite loop)
                if j == i:
                    i += 1
                else:
                    i = j

        # print(f"    ✅ Parsed {len(statements)} statements from block")
        return statements

    # === MAP LITERAL PARSING ===

    def _parse_map_literal(self, tokens):
        """Parse a map literal { key: value, ... }"""
        # parser_debug("  🗺️ [Map] Parsing map literal")

        if not tokens or tokens[0].type != LBRACE:
            parser_debug("  ❌ [Map] Not a map literal - no opening brace")
            return None

        pairs_list = []
        i = 1  # Skip opening brace

        while i < len(tokens) and tokens[i].type != RBRACE:
            key_token = tokens[i]

            # Expect colon after key
            if i + 1 < len(tokens) and tokens[i + 1].type == COLON:
                value_start = i + 2
                value_tokens = []

                j = value_start
                nesting = 0
                while j < len(tokens):
                    t = tokens[j]
                    if t.type == LBRACE or t.type == LBRACKET or t.type == LPAREN:
                        nesting += 1
                    elif t.type == RBRACE or t.type == RBRACKET or t.type == RPAREN:
                        if nesting > 0:
                            nesting -= 1
                        elif t.type == RBRACE and nesting == 0:
                            # Found the closing brace of the map (or end of value if comma follows)
                            break
                    
                    if nesting == 0 and t.type == COMMA:
                        break
                        
                    value_tokens.append(t)
                    j += 1

                value_expr = self._parse_expression(value_tokens)
                if value_expr:
                    if key_token.type == IDENT:
                        key_node = Identifier(key_token.literal)
                    elif key_token.type == STRING:
                        key_node = StringLiteral(key_token.literal)
                    else:
                        key_node = StringLiteral(key_token.literal)

                    pairs_list.append((key_node, value_expr))
                    # print(f"  🗺️ [Map] Added pair: {key_token.literal} -> {type(value_expr).__name__}")

                i = j
                if i < len(tokens) and tokens[i].type == COMMA:
                    i += 1
            else:
                # Skip token if it's unexpected (robust parsing)
                i += 1

        map_literal = MapLiteral(pairs_list)
        # print(f"  🗺️ [Map] Successfully parsed map with {len(pairs_list)} pairs")
        return map_literal

    # === EXPRESSION PARSING METHODS ===

    def _parse_paren_block_context(self, block_info, all_tokens):
        """Parse parentheses block - return proper statements where appropriate"""
        parser_debug("🔧 [Context] Parsing parentheses block")
        tokens = block_info['tokens']
        if len(tokens) < 3:
            return None

        context = self.get_current_context()
        start_idx = block_info.get('start_index', 0)

        if start_idx > 0 and all_tokens[start_idx - 1].type == PRINT:
            return self._parse_print_statement(block_info, all_tokens)
        elif start_idx > 0 and all_tokens[start_idx - 1].type == IDENT:
            return self._parse_function_call(block_info, all_tokens)
        else:
            expression = self._parse_generic_paren_expression(block_info, all_tokens)
            if expression:
                return ExpressionStatement(expression)
            return None

    def _parse_print_statement(self, block_info, all_tokens):
        """Parse print statement with support for multiple comma-separated arguments"""
        parser_debug("🔧 [Context] Parsing print statement with enhanced expression boundary detection")
        tokens = block_info['tokens']

        # Need at least PRINT token + one value token
        if len(tokens) < 2:
            return PrintStatement(values=[])

        # Collect tokens up to a statement boundary
        inner_tokens = []
        statement_terminators = {SEMICOLON, RBRACE}
        statement_starters = {LET, CONST, PRINT, FOR, IF, WHILE, RETURN, CONTINUE, ACTION, TRY, AUDIT, RESTRICT, SANDBOX, TRAIL, NATIVE, GC, INLINE, BUFFER, SIMD, DEFER, PATTERN, ENUM, STREAM, WATCH, CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE, INTERFACE, TYPE_ALIAS, MODULE, PACKAGE, USING}
        nesting_level = 0

        for token in tokens[1:]:  # Skip the PRINT token
            # Track nesting level for parentheses/braces
            if token.type in {LPAREN, LBRACE}:
                nesting_level += 1
            elif token.type in {RPAREN, RBRACE}:
                nesting_level -= 1
                if nesting_level < 0:  # Found closing without opening
                    break

            # Only check for boundaries when not inside nested structure
            if nesting_level == 0:
                if token.type in statement_terminators or token.type in statement_starters:
                    break

            inner_tokens.append(token)

        if not inner_tokens:
            return PrintStatement(values=[])

        parser_debug(f"  📝 Print statement tokens: {[t.literal for t in inner_tokens]}")
        
        # NEW: Handle comma-separated arguments
        # If the tokens start with ( and end with ), strip them (they're the print() parens, not nesting)
        tokens_to_parse = inner_tokens
        if len(inner_tokens) >= 2 and inner_tokens[0].type == LPAREN and inner_tokens[-1].type == RPAREN:
            # Strip outer parentheses
            tokens_to_parse = inner_tokens[1:-1]
        
        # Split by commas to get individual expressions
        values = []
        current_expr_tokens = []
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        
        for token in tokens_to_parse:
            # Track nesting depth
            if token.type == LPAREN:
                paren_depth += 1
            elif token.type == RPAREN:
                paren_depth -= 1
            elif token.type == LBRACKET:
                bracket_depth += 1
            elif token.type == RBRACKET:
                bracket_depth -= 1
            elif token.type == LBRACE:
                brace_depth += 1
            elif token.type == RBRACE:
                brace_depth -= 1
            
            # If we find a comma at depth 0, it's a separator
            if token.type == COMMA and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                # Parse the accumulated tokens as an expression
                if current_expr_tokens:
                    expr = self._parse_expression(current_expr_tokens)
                    if expr:
                        values.append(expr)
                    current_expr_tokens = []
            else:
                current_expr_tokens.append(token)
        
        # Parse the last expression
        if current_expr_tokens:
            expr = self._parse_expression(current_expr_tokens)
            if expr:
                values.append(expr)
        
        # If no values found, use empty string for backward compatibility
        if not values:
            values = [StringLiteral("")]
        
        parser_debug(f"  ✅ Parsed {len(values)} print arguments")
        return PrintStatement(values=values)

    def _parse_return_statement(self, block_info, all_tokens):
        """Parse return statement"""
        parser_debug("🔧 [Context] Parsing return statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != RETURN:
            return None
        
        # If only return token, return None
        if len(tokens) <= 1:
            return ReturnStatement(Identifier("null"))
        
        # Parse the return value expression
        value_tokens = tokens[1:]
        parser_debug(f"  📝 Return value tokens: {[t.literal for t in value_tokens]}")
        
        value_expr = self._parse_expression(value_tokens)
        return ReturnStatement(value_expr if value_expr else Identifier("null"))

    def _parse_expression(self, tokens):
        """Parse a full expression with operator precedence handling"""
        if not tokens or len(tokens) == 0:
            return StringLiteral("")
        
        # Handle if-then-else expression (high precedence, check early)
        # Pattern: if <condition> then <value1> else <value2>
        if tokens[0].type == IF:
            # Look for THEN and ELSE tokens at nesting level 0
            then_index = -1
            else_index = -1
            nesting = 0
            for idx, t in enumerate(tokens):
                if t.type in {LPAREN, LBRACE, LBRACKET}:
                    nesting += 1
                elif t.type in {RPAREN, RBRACE, RBRACKET}:
                    nesting -= 1
                elif nesting == 0:
                    if t.type == THEN and then_index == -1:
                        then_index = idx
                    elif t.type == ELSE and else_index == -1 and then_index != -1:
                        else_index = idx
                        break
            
            if then_index > 0 and else_index > then_index:
                # Valid if-then-else expression
                condition_tokens = tokens[1:then_index]
                consequence_tokens = tokens[then_index+1:else_index]
                alternative_tokens = tokens[else_index+1:]
                
                condition = self._parse_expression(condition_tokens)
                consequence_exp = self._parse_expression(consequence_tokens)
                alternative_exp = self._parse_expression(alternative_tokens)
                
                if condition and consequence_exp and alternative_exp:
                    # Wrap expressions in ExpressionStatements within BlockStatements
                    consequence_stmt = ExpressionStatement(expression=consequence_exp)
                    consequence_block = BlockStatement()
                    consequence_block.statements = [consequence_stmt]
                    
                    alternative_stmt = ExpressionStatement(expression=alternative_exp)
                    alternative_block = BlockStatement()
                    alternative_block.statements = [alternative_stmt]
                    
                    return IfExpression(
                        condition=condition,
                        consequence=consequence_block,
                        alternative=alternative_block
                    )
        
        # Handle ASSIGN (lowest precedence)
        assign_index = -1
        nesting = 0
        for idx, t in enumerate(tokens):
            if t.type in {LPAREN, LBRACE, LBRACKET}:
                nesting += 1
            elif t.type in {RPAREN, RBRACE, RBRACKET}:
                nesting -= 1
            elif t.type == ASSIGN and nesting == 0:
                assign_index = idx
                break
        
        if assign_index > 0 and assign_index < len(tokens) - 1:
            left_tokens = tokens[:assign_index]
            right_tokens = tokens[assign_index+1:]
            
            # For assignments, left side should be a simple identifier
            # Don't parse it as a full expression to avoid creating InfixExpression
            if len(left_tokens) == 1 and left_tokens[0].type == IDENT:
                left = Identifier(left_tokens[0].literal)
            else:
                left = self._parse_expression(left_tokens)
            
            right = self._parse_expression(right_tokens)
            
            if left and right:
                return AssignmentExpression(name=left, value=right)
            return left or right
        
        # Handle ternary operator ? : (very low precedence, after assignment before OR)
        question_index = -1
        colon_index = -1
        nesting = 0
        for idx, t in enumerate(tokens):
            if t.type in {LPAREN, LBRACE, LBRACKET}:
                nesting += 1
            elif t.type in {RPAREN, RBRACE, RBRACKET}:
                nesting -= 1
            elif t.type == QUESTION and nesting == 0 and question_index == -1:
                question_index = idx
            elif t.type == COLON and nesting == 0 and question_index != -1 and colon_index == -1:
                colon_index = idx
                break  # Found complete ternary
        
        if question_index > 0 and colon_index > question_index + 1 and colon_index < len(tokens) - 1:
            condition = self._parse_expression(tokens[:question_index])
            true_value = self._parse_expression(tokens[question_index+1:colon_index])
            false_value = self._parse_expression(tokens[colon_index+1:])
            
            if condition and true_value and false_value:
                return TernaryExpression(condition=condition, true_value=true_value, false_value=false_value)
            return condition or true_value or false_value
        
        # Handle nullish coalescing ?? (after ternary, before OR)
        nullish_index = -1
        nesting = 0
        for idx, t in enumerate(tokens):
            if t.type in {LPAREN, LBRACE, LBRACKET}:
                nesting += 1
            elif t.type in {RPAREN, RBRACE, RBRACKET}:
                nesting -= 1
            elif t.type == NULLISH and nesting == 0:
                nullish_index = idx
                break
        
        if nullish_index > 0 and nullish_index < len(tokens) - 1:
            left = self._parse_expression(tokens[:nullish_index])
            right = self._parse_expression(tokens[nullish_index+1:])
            if left and right:
                return NullishExpression(left=left, right=right)
            return left or right
        
        # Handle logical OR (lowest precedence)
        or_index = -1
        nesting = 0
        for idx, t in enumerate(tokens):
            if t.type in {LPAREN, LBRACE, LBRACKET}:
                nesting += 1
            elif t.type in {RPAREN, RBRACE, RBRACKET}:
                nesting -= 1
            elif t.type == OR and nesting == 0:
                or_index = idx
                break  # Take the first OR at depth 0
        
        if or_index > 0 and or_index < len(tokens) - 1:  # Valid split point
            left = self._parse_expression(tokens[:or_index])
            right = self._parse_expression(tokens[or_index+1:])
            if left and right:
                return InfixExpression(left=left, operator="||", right=right)
            return left or right
        
        # Handle logical AND (next lowest precedence)
        and_index = -1
        nesting = 0
        for idx, t in enumerate(tokens):
            if t.type in {LPAREN, LBRACE, LBRACKET}:
                nesting += 1
            elif t.type in {RPAREN, RBRACE, RBRACKET}:
                nesting -= 1
            elif t.type == AND and nesting == 0:
                and_index = idx
                break
        
        if and_index > 0 and and_index < len(tokens) - 1:  # Valid split point
            left = self._parse_expression(tokens[:and_index])
            right = self._parse_expression(tokens[and_index+1:])
            if left and right:
                return InfixExpression(left=left, operator="&&", right=right)
            return left or right
        
        # Continue with rest of expression parsing (comparison, arithmetic, etc)
        return self._parse_comparison_and_above(tokens)

    def _parse_comparison_and_above(self, tokens):
        """Parse comparisons and arithmetic operators (higher precedence than AND/OR)"""
        if not tokens:
            return StringLiteral("")

        # Special cases first
        # Handle unary prefix operators (!, -, etc.)
        if tokens[0].type == BANG:
            # Parse the remainder as the operand of the NOT expression
            right_expr = self._parse_expression(tokens[1:]) if len(tokens) > 1 else Boolean(True)
            return PrefixExpression("!", right_expr)
        
        if tokens[0].type == MINUS:
            # Parse the remainder as the operand of the prefix expression
            right_expr = self._parse_expression(tokens[1:]) if len(tokens) > 1 else IntegerLiteral(0)
            return PrefixExpression("-", right_expr)

        if tokens[0].type == LBRACE:
            return self._parse_map_literal(tokens)
        if tokens[0].type == LBRACKET:
            return self._parse_list_literal(tokens)
        if tokens[0].type == LAMBDA:
            return self._parse_lambda(tokens)
        if tokens[0].type == MATCH:
            return self._parse_match_expression(tokens)
        if tokens[0].type == FUNCTION or tokens[0].type == ACTION:
            return self._parse_function_literal(tokens)
        if tokens[0].type == SANDBOX:
            return self._parse_sandbox_expression(tokens)
        if tokens[0].type == SANITIZE:
            return self._parse_sanitize_expression(tokens)
        if tokens[0].type == AWAIT:
            return self._parse_await_expression(tokens)

        # Main expression parser with chaining
        i = 0
        n = len(tokens)
        current_expr = None
        nesting = 0

        # Helper to parse a primary expression at current position
        def parse_primary():
            nonlocal i
            if i >= n:
                return None

            t = tokens[i]
            if t.type == LPAREN:  # Parenthesized expression
                i += 1
                start = i
                depth = 1
                while i < n and depth > 0:
                    if tokens[i].type == LPAREN:
                        depth += 1
                    elif tokens[i].type == RPAREN:
                        depth -= 1
                    i += 1
                if depth == 0:  # Found closing paren
                    inner = self._parse_expression(tokens[start:i-1])
                    return inner if inner else StringLiteral("")
                return StringLiteral("")

            elif t.type == IDENT or t.type in {SEND, RECEIVE, DEBUG}:  # Identifier or function call (including concurrency/debug keywords)
                name = t.literal if t.type == IDENT else ("send" if t.type == SEND else ("receive" if t.type == RECEIVE else "debug"))
                i += 1
                
                # Check for generic type arguments: Box<number>
                type_args = []
                if i < n and tokens[i].type == LT:
                    # Need to disambiguate < for generics vs less-than operator
                    # Heuristic: if followed by type name and > or comma, it's a generic
                    lookahead = i + 1
                    is_generic = False
                    
                    if lookahead < n and tokens[lookahead].type == IDENT:
                        # Look for closing > or comma
                        temp_idx = lookahead + 1
                        while temp_idx < n:
                            if tokens[temp_idx].type == GT:
                                is_generic = True
                                break
                            elif tokens[temp_idx].type == COMMA:
                                is_generic = True
                                break
                            elif tokens[temp_idx].type == IDENT:
                                temp_idx += 1
                            else:
                                break
                    
                    if is_generic:
                        i += 1  # Skip <
                        # Parse comma-separated type arguments
                        while i < n:
                            if tokens[i].type == GT:
                                i += 1  # Skip >
                                break
                            
                            if tokens[i].type == IDENT:
                                type_args.append(tokens[i].literal)
                                i += 1
                                
                                # Check for comma or closing >
                                if i < n:
                                    if tokens[i].type == COMMA:
                                        i += 1  # Skip comma
                                    elif tokens[i].type == GT:
                                        continue  # Will break on next iteration
                                    else:
                                        break  # Invalid syntax, stop parsing type args
                            else:
                                break
                
                # Check for immediate function call
                if i < n and tokens[i].type == LPAREN:
                    i += 1  # Skip LPAREN
                    args = []
                    # Collect argument expressions
                    while i < n and tokens[i].type != RPAREN:
                        start = i
                        depth = 0
                        # Find end of current argument
                        while i < n:
                            if tokens[i].type in {LPAREN, LBRACE, LBRACKET}:
                                depth += 1
                            elif tokens[i].type in {RPAREN, RBRACE, RBRACKET}:
                                depth -= 1
                                if depth < 0:  # Found closing of call
                                    break
                            elif tokens[i].type == COMMA and depth == 0:
                                break
                            i += 1
                        # Parse the argument expression
                        if start < i:
                            arg = self._parse_expression(tokens[start:i])
                            if arg:
                                args.append(arg)
                        if i < n and tokens[i].type == COMMA:
                            i += 1  # Skip comma
                    if i < n and tokens[i].type == RPAREN:
                        i += 1  # Skip RPAREN
                    return CallExpression(Identifier(name), args, type_args=type_args)
                else:
                    return Identifier(name)

            # Literals
            else:
                i += 1
                return self._parse_single_token_expression(t)

        # Start with primary expression
        current_expr = parse_primary()
        if not current_expr:
            return StringLiteral("")

        # Repeatedly parse chained operations
        while i < n:
            t = tokens[i]

            # Method call or property access
            if t.type == DOT and i + 1 < n:
                i += 1  # Skip DOT
                if i >= n:
                    break

                name_token = tokens[i]
                # Allow keywords as property/method names (e.g., t.verify, obj.data)
                # Check if token has a literal (IDENT or any keyword)
                if not name_token.literal:
                    break

                i += 1  # Skip name
                # Method call: expr.name(args)
                if i < n and tokens[i].type == LPAREN:
                    i += 1  # Skip LPAREN
                    args = []
                    # Parse arguments same as function call
                    while i < n and tokens[i].type != RPAREN:
                        start = i
                        depth = 0
                        while i < n:
                            if tokens[i].type in {LPAREN, LBRACE, LBRACKET}:
                                depth += 1
                            elif tokens[i].type in {RPAREN, RBRACE, RBRACKET}:
                                depth -= 1
                                if depth < 0:
                                    break
                            elif tokens[i].type == COMMA and depth == 0:
                                break
                            i += 1
                        if start < i:
                            arg = self._parse_expression(tokens[start:i])
                            if arg:
                                args.append(arg)
                        if i < n and tokens[i].type == COMMA:
                            i += 1
                    if i < n and tokens[i].type == RPAREN:
                        i += 1
                    current_expr = MethodCallExpression(
                        object=current_expr,
                        method=Identifier(name_token.literal),
                        arguments=args
                    )
                else:
                    # Property access: expr.name
                    current_expr = PropertyAccessExpression(
                        object=current_expr,
                        property=Identifier(name_token.literal)
                    )
                continue

            # Direct function call on expression
            if t.type == LPAREN:
                i += 1  # Skip LPAREN
                args = []
                while i < n and tokens[i].type != RPAREN:
                    start = i
                    depth = 0
                    while i < n:
                        if tokens[i].type in {LPAREN, LBRACE, LBRACKET}:
                            depth += 1
                        elif tokens[i].type in {RPAREN, RBRACE, RBRACKET}:
                            depth -= 1
                            if depth < 0:
                                break
                        elif tokens[i].type == COMMA and depth == 0:
                            break
                        i += 1
                    if start < i:
                        arg = self._parse_expression(tokens[start:i])
                        if arg:
                            args.append(arg)
                    if i < n and tokens[i].type == COMMA:
                        i += 1
                if i < n and tokens[i].type == RPAREN:
                    i += 1
                current_expr = CallExpression(
                    function=current_expr,
                    arguments=args
                )
                continue

            # Bracket-index access: expr[ key ] -> PropertyAccessExpression with parsed key
            if t.type == LBRACKET:
                i += 1  # Skip LBRACKET
                start = i
                depth = 0
                while i < n:
                    if tokens[i].type in {LBRACKET, LPAREN, LBRACE}:
                        depth += 1
                    elif tokens[i].type in {RBRACKET, RPAREN, RBRACE}:
                        if depth == 0:
                            break
                        depth -= 1
                    i += 1
                inner_tokens = tokens[start:i]
                # If there's a closing RBRACKET, skip it
                if i < n and tokens[i].type == RBRACKET:
                    i += 1
                prop_expr = self._parse_expression(inner_tokens) if inner_tokens else Identifier('')
                current_expr = PropertyAccessExpression(
                    object=current_expr,
                    property=prop_expr
                )
                continue

            # Binary operators (comparisons and arithmetic - but NOT AND/OR which are handled above)
            if t.type in {PLUS, MINUS, ASTERISK, SLASH, MOD,
                         LT, GT, EQ, NOT_EQ, LTE, GTE}:
                # Get operator precedence
                op_precedence = self._get_operator_precedence(t.type)
                i += 1  # Skip operator
                
                # Parse right side with proper precedence climbing
                # We need to find where the right operand ends based on precedence
                right_start = i
                right_end = i
                
                # Find the extent of the right operand based on precedence
                # Lower precedence operators should end our right operand
                depth = 0
                while right_end < n:
                    tt = tokens[right_end]
                    
                    # Track nesting depth
                    if tt.type in {LPAREN, LBRACKET, LBRACE}:
                        depth += 1
                    elif tt.type in {RPAREN, RBRACKET, RBRACE}:
                        depth -= 1
                        if depth < 0:  # We've hit a closing bracket from outer context
                            break
                    
                    # If we're not nested and we hit an operator with same or lower precedence, stop
                    if depth == 0 and tt.type in {PLUS, MINUS, ASTERISK, SLASH, MOD, LT, GT, EQ, NOT_EQ, LTE, GTE}:
                        next_precedence = self._get_operator_precedence(tt.type)
                        # For left-associative operators, stop if next has same or lower precedence
                        if next_precedence <= op_precedence:
                            break
                    
                    right_end += 1
                
                # Parse the right operand
                right = self._parse_comparison_and_above(tokens[right_start:right_end]) if right_start < right_end else None
                
                if right:
                    current_expr = InfixExpression(
                        left=current_expr,
                        operator=t.literal,
                        right=right
                    )
                    i = right_end  # Continue from where right operand ended
                    continue
                else:
                    break

            # No more chaining possible
            break

        return current_expr
    
    def _get_operator_precedence(self, token_type):
        """Get operator precedence for proper parsing
        Higher numbers = higher precedence (evaluated first)
        """
        # Precedence levels (matching parser.py conventions)
        PRODUCT = 9   # *, /, %
        SUM = 8       # +, -
        COMPARISON = 7  # <, >, <=, >=
        EQUALITY = 6    # ==, !=
        
        if token_type in {ASTERISK, SLASH, MOD}:
            return PRODUCT
        elif token_type in {PLUS, MINUS}:
            return SUM
        elif token_type in {LT, GT, LTE, GTE}:
            return COMPARISON
        elif token_type in {EQ, NOT_EQ}:
            return EQUALITY
        else:
            return 1  # Lowest precedence

    def _parse_single_token_expression(self, token):
        """Parse a single token into an expression"""
        if token.type == STRING:
            return StringLiteral(token.literal)
        elif token.type == INT:
            try:
                return IntegerLiteral(int(token.literal))
            except Exception:
                return IntegerLiteral(0)
        elif token.type == FLOAT:
            try:
                return FloatLiteral(float(token.literal))
            except Exception:
                return FloatLiteral(0.0)
        elif token.type == IDENT:
            # Special case: 'this' should be ThisExpression
            if token.literal == 'this':
                from ..zexus_ast import ThisExpression
                return ThisExpression()
            return Identifier(token.literal)
        elif token.type == THIS:
            from ..zexus_ast import ThisExpression
            return ThisExpression()
        elif token.type == TRUE:
            # Derive value from literal text for safety
            lit = getattr(token, 'literal', 'true')
            val = True if isinstance(lit, str) and lit.lower() == 'true' else False
            return Boolean(val)
        elif token.type == FALSE:
            # Derive value from literal text for safety
            lit = getattr(token, 'literal', 'false')
            val = False if isinstance(lit, str) and lit.lower() == 'false' else True
            return Boolean(val)
        elif token.type == NULL:
            return NullLiteral()
        else:
            return StringLiteral(token.literal)

    def _parse_compound_expression(self, tokens):
        """Parse compound expressions with multiple tokens (best-effort)"""
        expression_parts = []
        i = 0

        while i < len(tokens):
            token = tokens[i]
            if token.type == IDENT and i + 1 < len(tokens) and tokens[i + 1].type == LPAREN:
                func_name = token.literal
                arg_tokens = self._extract_nested_tokens(tokens, i + 1)
                arguments = self._parse_argument_list(arg_tokens)
                expression_parts.append(CallExpression(Identifier(func_name), arguments))
                # advance by nested tokens length + 2 (function name and parentheses)
                i += len(arg_tokens) + 2
            else:
                expression_parts.append(self._parse_single_token_expression(token))
                i += 1

        if len(expression_parts) > 0:
            # Return first part as a best-effort expression (more advanced combining could be added)
            return expression_parts[0]
        else:
            return StringLiteral("")

    def _extract_nested_tokens(self, tokens, start_index):
        """Extract tokens inside nested parentheses/brackets/braces"""
        if start_index >= len(tokens) or tokens[start_index].type != LPAREN:
            return []

        nested_tokens = []
        depth = 1
        i = start_index + 1

        while i < len(tokens) and depth > 0:
            token = tokens[i]
            if token.type == LPAREN:
                depth += 1
            elif token.type == RPAREN:
                depth -= 1

            if depth > 0:
                nested_tokens.append(token)
            i += 1

        return nested_tokens

    def _parse_list_literal(self, tokens):
        """Parse a list literal [a, b, c] from a token list"""
        if not tokens or tokens[0].type != LBRACKET:
            parser_debug("  ❌ [List] Not a list literal")
            return None

        elements = []
        i = 1
        cur = []
        nesting = 0
        while i < len(tokens):
            t = tokens[i]
            if t.type in {LBRACKET, LPAREN, LBRACE}:
                nesting += 1
                cur.append(t)
            elif t.type == RBRACKET:
                # Only RBRACKET can close the list literal
                if nesting > 0:
                    nesting -= 1
                    cur.append(t)
                else:
                    # reached closing bracket of the list
                    if cur:
                        elem = self._parse_expression(cur)
                        elements.append(elem)
                    break
            elif t.type in {RPAREN, RBRACE}:
                # These just close nested expressions, not the list itself
                if nesting > 0:
                    nesting -= 1
                cur.append(t)
            elif t.type == COMMA and nesting == 0:
                if cur:
                    elem = self._parse_expression(cur)
                    elements.append(elem)
                    cur = []
            else:
                cur.append(t)
            i += 1

        parser_debug(f"  ✅ Parsed list with {len(elements)} elements")
        return ListLiteral(elements)

    def _parse_lambda(self, tokens):
        """Parse a lambda expression from tokens starting with LAMBDA (keyword-style)

        Supports forms:
          lambda x: x + 1
          lambda (x, y): x + y
        """
        print("  🔧 [Lambda] Parsing lambda expression (keyword-style)")
        if not tokens or tokens[0].type != LAMBDA:
            return None

        i = 1
        params = []

        # parenthesized params
        if i < len(tokens) and tokens[i].type == LPAREN:
            # collect tokens inside parentheses
            nested = self._extract_nested_tokens(tokens, i)
            j = 0
            cur_ident = None
            while j < len(nested):
                tk = nested[j]
                if tk.type == IDENT:
                    params.append(Identifier(tk.literal))
                j += 1
            i += len(nested) + 2
        # single identifier param
        elif i < len(tokens) and tokens[i].type == IDENT:
            params.append(Identifier(tokens[i].literal))
            i += 1

        # Accept ':' or '=>' or '-' '>' sequence
        if i < len(tokens) and tokens[i].type == COLON:
            i += 1
        elif i < len(tokens) and tokens[i].type == MINUS and i + 1 < len(tokens) and tokens[i + 1].type == GT:
            i += 2
        elif i < len(tokens) and tokens[i].type == LAMBDA:
            # defensive: allow repeated LAMBDA token produced by lexer for '=>'
            i += 1

        # Remaining tokens are body
        body_tokens = tokens[i:]
        body = self._parse_expression(body_tokens) if body_tokens else StringLiteral("")
        return LambdaExpression(parameters=params, body=body)

    def _parse_match_expression(self, tokens):
        """Parse a match expression for pattern matching
        
        match value {
            Point(x, y) => x + y,
            User(name, _) => name,
            42 => "the answer",
            _ => "default"
        }
        """
        from ..zexus_ast import (MatchExpression, MatchCase, ConstructorPattern, 
                                 VariablePattern, WildcardPattern, LiteralPattern)
        
        parser_debug("🔧 [Match] Parsing match expression")
        
        if not tokens or tokens[0].type != MATCH:
            return None
        
        idx = 1
        
        # Parse the value expression (until {)
        value_end = idx
        while value_end < len(tokens) and tokens[value_end].type != LBRACE:
            value_end += 1
        
        if value_end >= len(tokens):
            parser_debug("  ❌ Match expression: expected {")
            return None
        
        value_tokens = tokens[idx:value_end]
        value_expr = self._parse_expression(value_tokens)
        
        if not value_expr:
            parser_debug("  ❌ Match expression: failed to parse value")
            return None
        
        # Find matching closing brace
        idx = value_end + 1  # Skip {
        brace_count = 1
        body_start = idx
        
        while idx < len(tokens):
            if tokens[idx].type == LBRACE:
                brace_count += 1
            elif tokens[idx].type == RBRACE:
                brace_count -= 1
                if brace_count == 0:
                    break
            idx += 1
        
        if brace_count != 0:
            parser_debug("  ❌ Match expression: unmatched braces")
            return None
        
        body_end = idx
        body_tokens = tokens[body_start:body_end]
        
        # Parse match cases
        cases = []
        i = 0
        
        while i < len(body_tokens):
            # Skip commas and semicolons
            if body_tokens[i].type in {COMMA, SEMICOLON}:
                i += 1
                continue
            
            # Find the => separator
            arrow_idx = -1
            depth = 0
            for j in range(i, len(body_tokens)):
                if body_tokens[j].type in {LPAREN, LBRACE, LBRACKET}:
                    depth += 1
                elif body_tokens[j].type in {RPAREN, RBRACE, RBRACKET}:
                    depth -= 1
                elif body_tokens[j].type == LAMBDA and depth == 0:  # => is tokenized as LAMBDA
                    arrow_idx = j
                    break
            
            if arrow_idx == -1:
                # No more cases
                break
            
            # Parse pattern (from i to arrow_idx)
            pattern_tokens = body_tokens[i:arrow_idx]
            pattern = self._parse_pattern(pattern_tokens)
            
            if not pattern:
                parser_debug(f"  ❌ Failed to parse pattern: {[t.literal for t in pattern_tokens]}")
                i = arrow_idx + 1
                continue
            
            # Find result expression end (comma, semicolon, or end of body)
            result_start = arrow_idx + 1
            result_end = result_start
            depth = 0
            
            while result_end < len(body_tokens):
                if body_tokens[result_end].type in {LPAREN, LBRACE, LBRACKET}:
                    depth += 1
                elif body_tokens[result_end].type in {RPAREN, RBRACE, RBRACKET}:
                    depth -= 1
                elif body_tokens[result_end].type in {COMMA, SEMICOLON} and depth == 0:
                    break
                result_end += 1
            
            # Parse result expression
            result_tokens = body_tokens[result_start:result_end]
            result_expr = self._parse_expression(result_tokens) if result_tokens else NullLiteral()
            
            cases.append(MatchCase(pattern=pattern, result=result_expr))
            parser_debug(f"  ✅ Parsed match case: {pattern} => {result_expr}")
            
            i = result_end
        
        parser_debug(f"  ✅ Match expression with {len(cases)} cases")
        return MatchExpression(value=value_expr, cases=cases)
    
    def _parse_pattern(self, tokens):
        """Parse a pattern for pattern matching
        
        Patterns:
          Point(x, y)    - Constructor pattern with bindings
          _              - Wildcard pattern
          x, name        - Variable pattern
          42, "hello"    - Literal pattern
        """
        from ..zexus_ast import (ConstructorPattern, VariablePattern, 
                                 WildcardPattern, LiteralPattern, IntegerLiteral, StringLiteral)
        
        if not tokens:
            return None
        
        # Wildcard pattern: _
        if len(tokens) == 1 and tokens[0].type == IDENT and tokens[0].literal == "_":
            return WildcardPattern()
        
        # Literal patterns: 42, "hello", true, false
        if len(tokens) == 1:
            t = tokens[0]
            if t.type == INT:
                return LiteralPattern(IntegerLiteral(int(t.literal)))
            elif t.type == STRING:
                return LiteralPattern(StringLiteral(t.literal))
            elif t.type == TRUE:
                return LiteralPattern(Boolean(True))
            elif t.type == FALSE:
                return LiteralPattern(Boolean(False))
        
        # Constructor pattern: Point(x, y)
        if tokens[0].type == IDENT and len(tokens) > 1 and tokens[1].type == LPAREN:
            constructor_name = tokens[0].literal
            
            # Find matching closing paren
            paren_count = 0
            start = 1
            end = 1
            
            for i in range(1, len(tokens)):
                if tokens[i].type == LPAREN:
                    paren_count += 1
                elif tokens[i].type == RPAREN:
                    paren_count -= 1
                    if paren_count == 0:
                        end = i
                        break
            
            # Parse bindings (comma-separated patterns inside parentheses)
            binding_tokens = tokens[start+1:end]
            bindings = []
            
            if binding_tokens:
                # Split by commas at depth 0
                current_binding = []
                depth = 0
                
                for t in binding_tokens:
                    if t.type in {LPAREN, LBRACE, LBRACKET}:
                        depth += 1
                        current_binding.append(t)
                    elif t.type in {RPAREN, RBRACE, RBRACKET}:
                        depth -= 1
                        current_binding.append(t)
                    elif t.type == COMMA and depth == 0:
                        if current_binding:
                            binding_pattern = self._parse_pattern(current_binding)
                            if binding_pattern:
                                bindings.append(binding_pattern)
                        current_binding = []
                    else:
                        current_binding.append(t)
                
                # Don't forget the last binding
                if current_binding:
                    binding_pattern = self._parse_pattern(current_binding)
                    if binding_pattern:
                        bindings.append(binding_pattern)
            
            return ConstructorPattern(constructor_name=constructor_name, bindings=bindings)
        
        # Variable pattern: single identifier (not a wildcard)
        if len(tokens) == 1 and tokens[0].type == IDENT:
            return VariablePattern(name=tokens[0].literal)
        
        return None

    def _parse_function_literal(self, tokens):
        """Parse a function or action literal expression (anonymous function)
        
        Supports forms:
          function(x) { return x * 2; }
          action(x, y) { return x + y; }
          function() { return 42; }
        """
        print("  🔧 [Function Literal] Parsing function/action literal")
        if not tokens or tokens[0].type not in {FUNCTION, ACTION}:
            return None
        
        i = 1
        params = []
        
        # Collect parameters from parentheses
        if i < len(tokens) and tokens[i].type == LPAREN:
            i += 1
            while i < len(tokens) and tokens[i].type != RPAREN:
                if tokens[i].type == IDENT:
                    params.append(Identifier(tokens[i].literal))
                i += 1
            if i < len(tokens) and tokens[i].type == RPAREN:
                i += 1
        
        # Extract body tokens (from { to })
        body = BlockStatement()
        if i < len(tokens) and tokens[i].type == LBRACE:
            # Collect all tokens until matching closing brace
            brace_count = 0
            start = i
            while i < len(tokens):
                if tokens[i].type == LBRACE:
                    brace_count += 1
                elif tokens[i].type == RBRACE:
                    brace_count -= 1
                i += 1
                if brace_count == 0:
                    break
            
            # Parse body statements
            body_tokens = tokens[start+1:i-1]  # Exclude braces
            if body_tokens:
                parsed_stmts = self._parse_block_statements(body_tokens)
                body.statements = parsed_stmts
        
        # Return as ActionLiteral (same as lambda for function expressions)
        return ActionLiteral(parameters=params, body=body)

    def _parse_sandbox_expression(self, tokens):
        """Parse a sandbox expression from tokens starting with SANDBOX
        
        Supports form:
          sandbox { code }
        
        Returns a SandboxStatement which can be evaluated as an expression.
        """
        print("  🔧 [Sandbox Expression] Parsing sandbox expression")
        if not tokens or tokens[0].type != SANDBOX:
            return None
        
        i = 1
        body = None
        
        # Parse body block between { and }
        if i < len(tokens) and tokens[i].type == LBRACE:
            i += 1  # Skip opening brace
            block_tokens = []
            brace_nest = 1
            
            while i < len(tokens) and brace_nest > 0:
                if tokens[i].type == LBRACE:
                    brace_nest += 1
                elif tokens[i].type == RBRACE:
                    brace_nest -= 1
                    if brace_nest == 0:
                        break
                block_tokens.append(tokens[i])
                i += 1
            
            # Parse body statements
            if block_tokens:
                body_statements = self._parse_block_statements(block_tokens)
                body = BlockStatement()
                body.statements = body_statements
        
        # Return SandboxStatement (can be used as expression that returns value)
        return SandboxStatement(body=body, policy=None)

    def _parse_sanitize_expression(self, tokens):
        """Parse a sanitize expression from tokens starting with SANITIZE
        
        Supports forms:
          sanitize data, "html"
          sanitize data, "email"
          sanitize user_input, encoding_var
        
        Returns a SanitizeStatement which can be evaluated as an expression.
        """
        print("  🔧 [Sanitize Expression] Parsing sanitize expression")
        if not tokens or tokens[0].type != SANITIZE:
            return None
        
        # Find comma separating data and encoding
        comma_idx = -1
        for i in range(1, len(tokens)):
            if tokens[i].type == COMMA:
                comma_idx = i
                break
        
        if comma_idx == -1:
            # No encoding specified, use default
            data_tokens = tokens[1:]
            data = self._parse_expression(data_tokens)
            encoding = None
        else:
            # Parse data and encoding
            data_tokens = tokens[1:comma_idx]
            encoding_tokens = tokens[comma_idx+1:]
            
            data = self._parse_expression(data_tokens)
            encoding = self._parse_expression(encoding_tokens)
        
        # Return SanitizeStatement (can be used as expression that returns value)
        return SanitizeStatement(data=data, rules=None, encoding=encoding)

    def _parse_await_expression(self, tokens):
        """Parse an await expression from tokens starting with AWAIT
        
        Supports form:
          await expression
        
        Returns an AwaitExpression AST node.
        """
        
        if not tokens or tokens[0].type != AWAIT:
            return None
        
        # Skip AWAIT token
        expr_tokens = tokens[1:]
        
        if not expr_tokens:
            return None
        
        # Parse the expression to await
        expression = self._parse_expression(expr_tokens)
        
        if not expression:
            return None
        
        return AwaitExpression(expression)

    def _parse_argument_list(self, tokens):
        """Parse comma-separated argument list with improved nesting support"""
        parser_debug("  🔍 Parsing argument list")
        arguments = []
        current_arg = []
        nesting_level = 0

        for token in tokens:
            # Track nesting level for parentheses/braces
            if token.type in {LPAREN, LBRACE}:
                nesting_level += 1
            elif token.type in {RPAREN, RBRACE}:
                nesting_level -= 1

            # Only treat commas as separators when not inside nested structures
            if token.type == COMMA and nesting_level == 0:
                if current_arg:
                    arg_expr = self._parse_expression(current_arg)
                    parser_debug(f"  📝 Parsed argument: {type(arg_expr).__name__ if arg_expr else 'None'}")
                    arguments.append(arg_expr)
                    current_arg = []
            else:
                current_arg.append(token)

        # Handle last argument
        if current_arg:
            arg_expr = self._parse_expression(current_arg)
            parser_debug(f"  📝 Parsed final argument: {type(arg_expr).__name__ if arg_expr else 'None'}")
            arguments.append(arg_expr)

        # Filter out None arguments by replacing with empty string literal
        arguments = [arg if arg is not None else StringLiteral("") for arg in arguments]
        parser_debug(f"  ✅ Parsed {len(arguments)} arguments total")
        return arguments

    def _parse_function_call(self, block_info, all_tokens):
        """Parse function call expression with arguments"""
        start_idx = block_info.get('start_index', 0)
        if start_idx > 0:
            function_name = all_tokens[start_idx - 1].literal
            tokens = block_info['tokens']

            if len(tokens) >= 3:
                inner_tokens = tokens[1:-1]
                arguments = self._parse_argument_list(inner_tokens)
                return CallExpression(Identifier(function_name), arguments)
            else:
                return CallExpression(Identifier(function_name), [])
        return None

    def _parse_generic_paren_expression(self, block_info, all_tokens):
        """Parse generic parenthesized expression with full expression parsing"""
        tokens = block_info['tokens']
        inner_tokens = tokens[1:-1] if len(tokens) > 2 else []

        if not inner_tokens:
            return None

        return self._parse_expression(inner_tokens)

    # === REST OF THE CONTEXT METHODS ===

    def _parse_loop_context(self, block_info, all_tokens):
        """Parse loop blocks (for/while) with context awareness"""
        parser_debug("🔧 [Context] Parsing loop block")
        return BlockStatement()

    def _parse_screen_context(self, block_info, all_tokens):
        """Parse screen blocks with context awareness"""
        return ScreenStatement(
            name=Identifier(block_info.get('name', 'anonymous')),
            body=BlockStatement()
        )

    def _parse_try_catch_context(self, block_info, all_tokens):
        """Parse try-catch block with full context awareness"""
        parser_debug("🔧 [Context] Parsing try-catch block with context awareness")
        error_var = self._extract_catch_variable(block_info['tokens'])
        return TryCatchStatement(
            try_block=BlockStatement(),
            error_variable=error_var,
            catch_block=BlockStatement()
        )

    def _parse_function_context(self, block_info, all_tokens):
        """Parse function block with context awareness"""
        params = self._extract_function_parameters(block_info, all_tokens)
        return ActionStatement(
            name=Identifier(block_info.get('name', 'anonymous')),
            parameters=params,
            body=BlockStatement()
        )

    def _parse_action_statement_context(self, block_info, all_tokens):
        """Parse action statement: [async] action name(params) [-> return_type] { body }"""
        tokens = block_info.get('tokens', [])
        if not tokens:
            return None
        
        # Check for async modifier
        is_async = False
        i = 0
        if tokens[i].type == ASYNC:
            is_async = True
            i += 1
        
        # Now expect ACTION keyword
        if i >= len(tokens) or tokens[i].type != ACTION:
            return None
        i += 1
        
        # Extract action name (next IDENT after ACTION)
        name = None
        params = []
        return_type = None
        body_tokens = []
        
        while i < len(tokens):
            if tokens[i].type == IDENT:
                name = Identifier(tokens[i].literal)
                break
            i += 1
        
        # Extract parameters from parentheses
        if name and i < len(tokens):
            i += 1
            if i < len(tokens) and tokens[i].type == LPAREN:
                # Collect tokens until RPAREN, handling type annotations
                i += 1
                while i < len(tokens) and tokens[i].type != RPAREN:
                    if tokens[i].type == IDENT:
                        # This is a parameter name
                        params.append(Identifier(tokens[i].literal))
                        i += 1
                        
                        # Check for type annotation: : type
                        if i < len(tokens) and tokens[i].type == COLON:
                            i += 1  # Skip COLON
                            if i < len(tokens) and tokens[i].type == IDENT:
                                i += 1  # Skip type name
                        
                        # Check for comma (more parameters)
                        if i < len(tokens) and tokens[i].type == COMMA:
                            i += 1  # Skip COMMA
                    else:
                        # Unexpected token, skip it
                        i += 1
                
                # Now i should be at RPAREN
                if i < len(tokens) and tokens[i].type == RPAREN:
                    i += 1  # Skip RPAREN
        
        # Check for return type annotation: -> type
        if i < len(tokens) and tokens[i].type == MINUS:
            if i + 1 < len(tokens) and tokens[i + 1].type == GT:
                # Found -> return type annotation
                i += 2  # Skip - and >
                if i < len(tokens) and tokens[i].type == IDENT:
                    return_type = tokens[i].literal
                    i += 1
        
        # Extract body tokens (everything from { to })
        if i < len(tokens):
            body_tokens = tokens[i:]
        
        # Parse body as a block statement
        body = BlockStatement()
        if body_tokens:
            # Skip opening brace if present
            start = 0
            if body_tokens and body_tokens[0].type == LBRACE:
                start = 1
            if start < len(body_tokens) and body_tokens[-1].type == RBRACE:
                body_tokens = body_tokens[start:-1]
            else:
                body_tokens = body_tokens[start:]
            
            # Parse statements from body tokens
            if body_tokens:
                parsed_stmts = self._parse_block_statements(body_tokens)
                body.statements = parsed_stmts
        
        return ActionStatement(
            name=name or Identifier('anonymous'),
            parameters=params,
            body=body,
            is_async=is_async,
            return_type=return_type
        )

    def _parse_async_expression_block(self, block_info, all_tokens):
        """Parse async expression: async <expression>
        
        Example: async producer()
        This creates an AsyncExpression that executes the expression in a background thread.
        """
        from ..zexus_ast import AsyncExpression, ExpressionStatement
        import sys
        
        tokens = block_info.get('tokens', [])
        if not tokens or tokens[0].type != ASYNC:
            return None
        
        # The tokens are [ASYNC, ...expression tokens...]
        # We can just call parse_async_expression from the main parser!
        # But wait, that expects to see ASYNC as cur_token. We need to create a mini-parser.
        #
        # Actually, simpler approach: Just parse the expression part and wrap it in AsyncExpression
        expr_tokens = tokens[1:]  # Skip ASYNC
        
        if not expr_tokens:
            return None
        
        # Create a mini-parser for the expression tokens
        from ..parser.parser import UltimateParser
        from ..zexus_token import Token, EOF
        
        class TokenLexer:
            """A lexer that reads from a list of tokens instead of source code"""
            def __init__(self, tokens):
                self.tokens = tokens + [Token(EOF, 'EOF', 0, 0)]  # Add EOF token
                self.pos = 0
            
            def next_token(self):
                if self.pos < len(self.tokens):
                    tok = self.tokens[self.pos]
                    self.pos += 1
                    return tok
                return Token(EOF, 'EOF', 0, 0)
        
        token_lexer = TokenLexer(expr_tokens)
        mini_parser = UltimateParser(token_lexer, enable_advanced_strategies=False)
        
        # The parser already primes itself in __init__ by calling next_token() twice!
        # So cur_token is already set to the first token, peek_token to the second.
        
        expr = mini_parser.parse_expression(1)  # LOWEST precedence
        
        if expr:
            # Wrap in AsyncExpression
            async_expr = AsyncExpression(expression=expr)
            return ExpressionStatement(async_expr)
        
        return None

    def _parse_function_statement_context(self, block_info, all_tokens):
        """Parse function statement: function name(params) [-> return_type] { body }"""
        tokens = block_info.get('tokens', [])
        if not tokens or tokens[0].type != FUNCTION:
            return None
        
        # Extract function name (next IDENT after FUNCTION)
        name = None
        params = []
        return_type = None
        body_tokens = []
        
        i = 1
        while i < len(tokens):
            if tokens[i].type == IDENT:
                name = Identifier(tokens[i].literal)
                break
            i += 1
        
        # Extract parameters from parentheses
        if name and i < len(tokens):
            i += 1
            if i < len(tokens) and tokens[i].type == LPAREN:
                # Collect tokens until RPAREN
                i += 1
                while i < len(tokens) and tokens[i].type != RPAREN:
                    if tokens[i].type == IDENT:
                        params.append(Identifier(tokens[i].literal))
                    i += 1
                i += 1  # Skip RPAREN
        
        # Check for return type annotation: -> type
        if i < len(tokens) and tokens[i].type == MINUS:
            if i + 1 < len(tokens) and tokens[i + 1].type == GT:
                # Found -> return type annotation
                i += 2  # Skip - and >
                if i < len(tokens) and tokens[i].type == IDENT:
                    return_type = tokens[i].literal
                    i += 1
        
        # Extract body tokens (everything from { to })
        if i < len(tokens):
            body_tokens = tokens[i:]
        
        # Parse body as a block statement
        body = BlockStatement()
        if body_tokens:
            # Skip opening brace if present
            start = 0
            if body_tokens and body_tokens[0].type == LBRACE:
                start = 1
            if start < len(body_tokens) and body_tokens[-1].type == RBRACE:
                body_tokens = body_tokens[start:-1]
            else:
                body_tokens = body_tokens[start:]
            
            # Parse statements from body tokens
            if body_tokens:
                parsed_stmts = self._parse_block_statements(body_tokens)
                body.statements = parsed_stmts
        
        return FunctionStatement(
            name=name or Identifier('anonymous'),
            parameters=params,
            body=body,
            return_type=return_type
        )

    def _parse_conditional_context(self, block_info, all_tokens):
        """Parse if/else blocks with context awareness"""
        parser_debug("🔧 [Context] Parsing conditional block")
        condition = self._extract_condition(block_info, all_tokens)

        # Collect following `elif` parts and `else` alternative by scanning tokens
        elif_parts = []
        alternative = None

        end_idx = block_info.get('end_index', block_info.get('start_index', 0))
        i = end_idx + 1
        n = len(all_tokens)

        # Helper to collect a brace-delimited block starting at index `start` (pointing at LBRACE)
        def collect_brace_inner(start_index):
            j = start_index
            if j >= n or all_tokens[j].type != LBRACE:
                # find next LBRACE
                while j < n and all_tokens[j].type != LBRACE:
                    j += 1
                if j >= n:
                    return [], j

            depth = 0
            inner = []
            while j < n:
                tok = all_tokens[j]
                if tok.type == LBRACE:
                    depth += 1
                    if depth > 1:
                        inner.append(tok)
                elif tok.type == RBRACE:
                    depth -= 1
                    if depth == 0:
                        return inner, j + 1
                    inner.append(tok)
                else:
                    if depth >= 1:
                        inner.append(tok)
                j += 1

            return inner, j

        while i < n:
            t = all_tokens[i]

            # Skip non-significant tokens
            if t.type in {SEMICOLON}:
                i += 1
                continue

            # Handle ELIF
            if t.type == ELIF:
                # Collect condition tokens until the following LBRACE
                cond_tokens = []
                j = i + 1
                while j < n and all_tokens[j].type != LBRACE:
                    # stop if we hit another control keyword
                    if all_tokens[j].type in {ELIF, ELSE, IF}:
                        break
                    cond_tokens.append(all_tokens[j])
                    j += 1

                cond_expr = self._parse_expression(cond_tokens) if cond_tokens else Identifier("true")

                # Collect the block inner tokens and parse into statements
                inner_tokens, next_idx = collect_brace_inner(j)
                block_stmt = BlockStatement()
                block_stmt.statements = self._parse_block_statements(inner_tokens)

                elif_parts.append((cond_expr, block_stmt))

                i = next_idx
                continue

            # Handle ELSE
            if t.type == ELSE:
                # Collect block following else
                j = i + 1
                inner_tokens, next_idx = collect_brace_inner(j)
                alt_block = BlockStatement()
                alt_block.statements = self._parse_block_statements(inner_tokens)
                alternative = alt_block
                i = next_idx
                break

            # If we hit a top-level closing brace or another unrelated statement starter, stop
            if t.type == RBRACE or t.type in {LET, CONST, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY, EXTERNAL}:
                break

            i += 1

        # Build the IfStatement with parsed block statements
        consequence_block = BlockStatement()
        # Parse the main consequence block tokens from block_info if available
        main_inner = []
        # The block_info may include tokens for the if-block; try to extract inner tokens
        b_tokens = block_info.get('tokens', [])
        if b_tokens and b_tokens[0].type == LBRACE:
            # tokens include braces; extract inner slice
            main_inner = b_tokens[1:-1]
        elif b_tokens:
            # If not braced, attempt to parse as statements directly
            main_inner = b_tokens

        consequence_block.statements = self._parse_block_statements(main_inner)

        return IfStatement(
            condition=condition,
            consequence=consequence_block,
            elif_parts=elif_parts,
            alternative=alternative
        )

    def _parse_brace_block_context(self, block_info, all_tokens):
        """Parse generic brace block with context awareness"""
        parser_debug("🔧 [Context] Parsing brace block")
        return BlockStatement()

    def _parse_generic_block(self, block_info, all_tokens):
        """Fallback parser for unknown block types - intelligently detects statement type"""
        tokens = block_info.get('tokens', [])
        if not tokens:
            return BlockStatement()
        
        # Debug: log what we're trying to parse
        parser_debug(f"  🔍 [Generic] Parsing generic block with tokens: {[t.literal for t in tokens]}")
        
        # Check if this is a LET statement
        if tokens[0].type == LET:
            parser_debug(f"  🎯 [Generic] Detected let statement")
            return self._parse_let_statement_block(block_info, all_tokens)
        
        # Check if this is a CONST statement
        if tokens[0].type == CONST:
            parser_debug(f"  🎯 [Generic] Detected const statement")
            return self._parse_const_statement_block(block_info, all_tokens)
        
        # Check if this is a DATA statement (dataclass definition)
        # Can be: data User { ... } or @validated data User { ... }
        has_decorator = tokens[0].type == AT
        data_index = 0
        
        # Skip decorators to find 'data' keyword
        while data_index < len(tokens) and tokens[data_index].type == AT:
            data_index += 1  # Skip @
            if data_index < len(tokens) and tokens[data_index].type == IDENT:
                data_index += 1  # Skip decorator name
        
        if data_index < len(tokens) and tokens[data_index].type == DATA:
            parser_debug(f"  🎯 [Generic] Detected data statement (decorators: {has_decorator})")
            return self._parse_data_statement(tokens)
        
        # Check if this is a LOG statement (log >, log >>, or log <<)
        if tokens[0].type == LOG:
            parser_debug(f"  🎯 [Generic] Detected log statement")
            # Parse using _parse_block_statements which handles LOG
            statements = self._parse_block_statements(tokens)
            if statements and len(statements) > 0:
                return statements[0]  # Return the first (and likely only) statement
            return None
        
        # Check if this is an assignment statement (identifier = value OR property.access = value)
        # Look for ASSIGN token anywhere in the statement
        assign_idx = None
        for i, tok in enumerate(tokens):
            if tok.type == ASSIGN:
                assign_idx = i
                break
        
        if assign_idx is not None and assign_idx > 0:
            parser_debug(f"  🎯 [Generic] Detected assignment statement (assign at index {assign_idx})")
            return self._parse_assignment_statement(block_info, all_tokens)
        
        # Check if this is a print statement
        if tokens[0].type == PRINT:
            parser_debug(f"  🎯 [Generic] Detected print statement")
            return self._parse_print_statement(block_info, all_tokens)
        
        # Check if this is a return statement
        if tokens[0].type == RETURN:
            parser_debug(f"  🎯 [Generic] Detected return statement")
            return self._parse_return_statement(block_info, all_tokens)
        
        # Check if this is a require statement
        if tokens[0].type == REQUIRE:
            parser_debug(f"  🎯 [Generic] Detected require statement")
            return self._parse_require_statement(block_info, all_tokens)
        
        # Check if this is an external declaration
        if tokens[0].type == EXTERNAL:
            parser_debug(f"  🎯 [Generic] Detected external declaration")
            # Manual parsing for simple syntax: external identifier;
            if len(tokens) >= 2 and tokens[1].type == IDENT:
                name = Identifier(tokens[1].literal)
                stmt = ExternalDeclaration(
                    name=name,
                    parameters=[],
                    module_path=""
                )
                return stmt
            # Fall through if parsing fails
        
        # Check if it's a function call (identifier followed by parentheses)
        if tokens[0].type == IDENT and len(tokens) >= 2 and tokens[1].type == LPAREN:
            parser_debug(f"  🎯 [Generic] Detected function call")
            # Parse as expression and wrap in ExpressionStatement
            expr = self._parse_expression(tokens)
            if expr:
                return ExpressionStatement(expr)
        
        # Try to parse as a simple expression
        parser_debug(f"  🎯 [Generic] Attempting to parse as expression")
        expr = self._parse_expression(tokens)
        if expr:
            return ExpressionStatement(expr)
        
        # Fallback: return empty block
        return BlockStatement()

    # Helper methods
    def _extract_catch_variable(self, tokens):
        """Extract the error variable from catch block"""
        for i, token in enumerate(tokens):
            if token.type == CATCH and i + 1 < len(tokens):
                # catch (err) style
                if tokens[i + 1].type == LPAREN and i + 2 < len(tokens):
                    if tokens[i + 2].type == IDENT:
                        return Identifier(tokens[i + 2].literal)
                # catch err style
                elif tokens[i + 1].type == IDENT:
                    return Identifier(tokens[i + 1].literal)
        return Identifier("error")

    def _extract_function_parameters(self, block_info, all_tokens):
        """Extract function parameters from function signature"""
        params = []
        start_idx = block_info.get('start_index', 0)
        # Scan backward to find preceding '('
        for i in range(max(0, start_idx - 50), start_idx):
            if i < len(all_tokens) and all_tokens[i].type == LPAREN:
                j = i + 1
                while j < len(all_tokens) and all_tokens[j].type != RPAREN:
                    if all_tokens[j].type == IDENT:
                        params.append(Identifier(all_tokens[j].literal))
                    j += 1
                break
        return params

    def _extract_condition(self, block_info, all_tokens):
        """Extract condition from conditional statements"""
        start_idx = block_info.get('start_index', 0)
        for i in range(max(0, start_idx - 20), start_idx):
            if i < len(all_tokens) and all_tokens[i].type == LPAREN:
                j = i + 1
                condition_tokens = []
                while j < len(all_tokens) and all_tokens[j].type != RPAREN:
                    condition_tokens.append(all_tokens[j])
                    j += 1
                if condition_tokens:
                    # Attempt to parse the whole condition expression
                    cond_expr = self._parse_expression(condition_tokens)
                    return cond_expr if cond_expr is not None else Identifier("true")
                break
        return Identifier("true")

    # === NEW SECURITY STATEMENT HANDLERS ===

    def _parse_capability_statement(self, block_info, all_tokens):
        """Parse capability definition statement
        
        capability read_file = {
          description: "Read file system",
          scope: "io"
        };
        """
        parser_debug("🔧 [Context] Parsing capability statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            parser_debug("  ❌ Invalid capability statement: expected name")
            return None
        
        if tokens[0].type != CAPABILITY:
            parser_debug("  ❌ Expected CAPABILITY keyword")
            return None
        
        if tokens[1].type != IDENT:
            parser_debug("  ❌ Expected capability name")
            return None
        
        cap_name = tokens[1].literal
        print(f"  📋 Capability: {cap_name}")
        
        # Look for definition block
        definition = None
        for i in range(2, len(tokens)):
            if tokens[i].type == LBRACE:
                # Extract map/definition
                definition = self._parse_map_literal(tokens[i:])
                break
        
        return CapabilityStatement(
            name=Identifier(cap_name),
            definition=definition
        )

    def _parse_grant_statement(self, block_info, all_tokens):
        """Parse grant statement
        
        grant user1 {
          read_file,
          read_network
        };
        """
        parser_debug("🔧 [Context] Parsing grant statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            parser_debug("  ❌ Invalid grant statement")
            return None
        
        if tokens[0].type != GRANT:
            parser_debug("  ❌ Expected GRANT keyword")
            return None
        
        if tokens[1].type != IDENT:
            parser_debug("  ❌ Expected entity name after grant")
            return None
        
        entity_name = tokens[1].literal
        print(f"  👤 Entity: {entity_name}")
        
        # Parse capabilities list
        capabilities = []
        i = 2
        while i < len(tokens):
            if tokens[i].type == IDENT:
                capabilities.append(Identifier(tokens[i].literal))
            elif tokens[i].type == LPAREN:
                # function call style: capability(name)
                if i + 2 < len(tokens) and tokens[i + 1].type == IDENT:
                    capabilities.append(FunctionCall(
                        Identifier("capability"),
                        [Identifier(tokens[i + 1].literal)]
                    ))
                    i += 2
            i += 1
        
        print(f"  🔑 Capabilities: {len(capabilities)}")
        
        return GrantStatement(
            entity_name=Identifier(entity_name),
            capabilities=capabilities
        )

    def _parse_revoke_statement(self, block_info, all_tokens):
        """Parse revoke statement (mirrors grant)"""
        parser_debug("🔧 [Context] Parsing revoke statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            parser_debug("  ❌ Invalid revoke statement")
            return None
        
        if tokens[0].type != REVOKE:
            parser_debug("  ❌ Expected REVOKE keyword")
            return None
        
        if tokens[1].type != IDENT:
            parser_debug("  ❌ Expected entity name after revoke")
            return None
        
        entity_name = tokens[1].literal
        print(f"  👤 Entity: {entity_name}")
        
        # Parse capabilities list (same as grant)
        capabilities = []
        i = 2
        while i < len(tokens):
            if tokens[i].type == IDENT:
                capabilities.append(Identifier(tokens[i].literal))
            elif tokens[i].type == LPAREN:
                if i + 2 < len(tokens) and tokens[i + 1].type == IDENT:
                    capabilities.append(FunctionCall(
                        Identifier("capability"),
                        [Identifier(tokens[i + 1].literal)]
                    ))
                    i += 2
            i += 1
        
        print(f"  🔑 Capabilities: {len(capabilities)}")
        
        return RevokeStatement(
            entity_name=Identifier(entity_name),
            capabilities=capabilities
        )

    def _parse_validate_statement(self, block_info, all_tokens):
        """Parse validate statement
        
        validate user_input, {
          name: string,
          email: email,
          age: number(18, 120)
        };
        """
        parser_debug("🔧 [Context] Parsing validate statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            parser_debug("  ❌ Invalid validate statement")
            return None
        
        # Parse: validate <expr>, <schema>
        comma_idx = -1
        for i, t in enumerate(tokens):
            if t.type == COMMA:
                comma_idx = i
                break
        
        if comma_idx == -1:
            parser_debug("  ⚠️ No schema provided for validate")
            # Single argument: validate(expr) with implicit schema
            data = self._parse_expression(tokens[1:])
            return ValidateStatement(data, {})
        
        # Split into data and schema
        data_tokens = tokens[1:comma_idx]
        schema_tokens = tokens[comma_idx + 1:]
        
        data = self._parse_expression(data_tokens)
        schema = self._parse_expression(schema_tokens)
        
        print(f"  ✓ Validate: {type(data).__name__} against {type(schema).__name__}")
        
        return ValidateStatement(data, schema)

    def _parse_sanitize_statement(self, block_info, all_tokens):
        """Parse sanitize statement
        
        sanitize user_input, {
          encoding: "html",
          rules: ["remove_scripts"]
        };
        """
        parser_debug("🔧 [Context] Parsing sanitize statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            parser_debug("  ❌ Invalid sanitize statement")
            return None
        
        # Parse: sanitize <expr>, <options>
        comma_idx = -1
        for i, t in enumerate(tokens):
            if t.type == COMMA:
                comma_idx = i
                break
        
        # Data to sanitize
        data_tokens = tokens[1:comma_idx if comma_idx != -1 else None]
        data = self._parse_expression(data_tokens)
        
        # Options (if provided)
        rules = None
        encoding = None
        if comma_idx != -1:
            options_tokens = tokens[comma_idx + 1:]
            options = self._parse_expression(options_tokens)
            # Extract encoding and rules from options if it's a map
            if isinstance(options, Map):
                for key, val in options.pairs:
                    if isinstance(key, StringLiteral):
                        if key.value == "encoding":
                            encoding = val
                        elif key.value == "rules":
                            rules = val
        
        print(f"  🧹 Sanitize: {type(data).__name__}")
        
        return SanitizeStatement(data, rules, encoding)

    def _parse_immutable_statement(self, block_info, all_tokens):
        """Parse immutable statement
        
        immutable const user = { name: "Alice" };
        immutable let config = load_config();
        """
        parser_debug("🔧 [Context] Parsing immutable statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            parser_debug("  ❌ Invalid immutable statement")
            return None
        
        if tokens[0].type != IMMUTABLE:
            parser_debug("  ❌ Expected IMMUTABLE keyword")
            return None
        
        # Check if next is LET, CONST, or IDENT
        if tokens[1].type in {LET, CONST}:
            # immutable let/const name = value
            if len(tokens) < 4 or tokens[2].type != IDENT:
                parser_debug("  ❌ Invalid immutable declaration")
                return None
            
            var_name = tokens[2].literal
            target = Identifier(var_name)
            
            # Extract value if present
            value = None
            if len(tokens) > 3 and tokens[3].type == ASSIGN:
                value_tokens = tokens[4:]
                value = self._parse_expression(value_tokens)
            
            print(f"  🔒 Immutable: {var_name}")
            return ImmutableStatement(target, value)
        
        elif tokens[1].type == IDENT:
            # immutable identifier
            var_name = tokens[1].literal
            target = Identifier(var_name)
            
            # Check for assignment
            value = None
            if len(tokens) > 2 and tokens[2].type == ASSIGN:
                value_tokens = tokens[3:]
                value = self._parse_expression(value_tokens)
            
            print(f"  🔒 Immutable: {var_name}")
            return ImmutableStatement(target, value)
        
        else:
            parser_debug("  ❌ Expected LET, CONST, or identifier after IMMUTABLE")
            return None
    # === COMPLEXITY & LARGE PROJECT MANAGEMENT HANDLERS ===

    def _parse_interface_statement(self, block_info, all_tokens):
        """Parse interface definition statement
        
        interface Drawable {
            draw(canvas);
            get_bounds();
        };
        """
        parser_debug("🔧 [Context] Parsing interface statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2 or tokens[0].type != INTERFACE:
            parser_debug("  ❌ Expected INTERFACE keyword")
            return None
        
        if tokens[1].type != IDENT:
            parser_debug("  ❌ Expected interface name")
            return None
        
        interface_name = tokens[1].literal
        print(f"  📋 Interface: {interface_name}")
        
        methods = []
        properties = {}
        
        # Parse interface body
        for i in range(2, len(tokens)):
            if tokens[i].type == LBRACE:
                # Find matching closing brace
                j = i + 1
                brace_count = 1
                method_tokens = []
                
                while j < len(tokens) and brace_count > 0:
                    if tokens[j].type == LBRACE:
                        brace_count += 1
                    elif tokens[j].type == RBRACE:
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    
                    method_tokens.append(tokens[j])
                    j += 1
                
                # Parse method signatures
                for k, tok in enumerate(method_tokens):
                    if tok.type == IDENT and k + 1 < len(method_tokens) and method_tokens[k + 1].type == LPAREN:
                        # Found a method
                        method_name = tok.literal
                        methods.append(method_name)
                        print(f"    📝 Method: {method_name}()")
                
                break
        
        return InterfaceStatement(
            name=Identifier(interface_name),
            methods=methods,
            properties=properties
        )

    def _parse_type_alias_statement(self, block_info, all_tokens):
        """Parse type alias statement
        
        type_alias UserID = integer;
        type_alias Point = { x: float, y: float };
        """
        parser_debug("🔧 [Context] Parsing type_alias statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 4 or tokens[0].type != TYPE_ALIAS:
            parser_debug("  ❌ Invalid type_alias statement")
            return None
        
        if tokens[1].type != IDENT:
            parser_debug("  ❌ Expected type name")
            return None
        
        type_name = tokens[1].literal
        
        if tokens[2].type != ASSIGN:
            parser_debug("  ❌ Expected '=' in type_alias")
            return None
        
        # Parse the base type
        base_type_tokens = tokens[3:]
        base_type = self._parse_expression(base_type_tokens)
        
        parser_debug(f"  📝 Type alias: {type_name}")
        
        return TypeAliasStatement(
            name=Identifier(type_name),
            base_type=base_type
        )

    def _parse_module_statement(self, block_info, all_tokens):
        """Parse module definition statement
        
        module database {
            internal function connect() { ... }
            public function query(sql) { ... }
        }
        """
        parser_debug("🔧 [Context] Parsing module statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2 or tokens[0].type != MODULE:
            parser_debug("  ❌ Expected MODULE keyword")
            return None
        
        if tokens[1].type != IDENT:
            parser_debug("  ❌ Expected module name")
            return None
        
        module_name = tokens[1].literal
        # print(f"  📦 Module: {module_name}")
        
        # Parse module body
        body = None
        for i in range(2, len(tokens)):
            if tokens[i].type == LBRACE:
                # Extract body tokens
                j = i + 1
                brace_count = 1
                body_tokens = []
                
                while j < len(tokens) and brace_count > 0:
                    if tokens[j].type == LBRACE:
                        brace_count += 1
                    elif tokens[j].type == RBRACE:
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    
                    body_tokens.append(tokens[j])
                    j += 1
                
                # Parse body as block statement
                if body_tokens:
                    body_block = BlockStatement()
                    body_block.statements = self._parse_block_statements(body_tokens)
                    body = body_block
                
                break
        
        if not body:
            body = BlockStatement()
        
        return ModuleStatement(
            name=Identifier(module_name),
            body=body
        )

    def _parse_package_statement(self, block_info, all_tokens):
        """Parse package definition statement
        
        package myapp.database {
            module connection { ... }
            module query { ... }
        }
        """
        parser_debug("🔧 [Context] Parsing package statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2 or tokens[0].type != PACKAGE:
            parser_debug("  ❌ Expected PACKAGE keyword")
            return None
        
        # Parse package name (may be dotted)
        package_name = ""
        i = 1
        while i < len(tokens) and tokens[i].type == IDENT:
            if package_name:
                package_name += "."
            package_name += tokens[i].literal
            i += 1
            
            # Check for dot
            if i < len(tokens) and tokens[i].type == DOT:
                i += 1
        
        print(f"  📦 Package: {package_name}")
        
        # Parse package body
        body = None
        for j in range(i, len(tokens)):
            if tokens[j].type == LBRACE:
                # Extract body tokens
                k = j + 1
                brace_count = 1
                body_tokens = []
                
                while k < len(tokens) and brace_count > 0:
                    if tokens[k].type == LBRACE:
                        brace_count += 1
                    elif tokens[k].type == RBRACE:
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    
                    body_tokens.append(tokens[k])
                    k += 1
                
                # Parse body
                if body_tokens:
                    body_block = BlockStatement()
                    body_block.statements = self._parse_block_statements(body_tokens)
                    body = body_block
                
                break
        
        if not body:
            body = BlockStatement()
        
        return PackageStatement(
            name=Identifier(package_name),
            body=body
        )

    def _parse_using_statement(self, block_info, all_tokens):
        """Parse using statement for resource management
        
        using(file = open("data.txt")) {
            content = file.read();
            process(content);
        }
        """
        print("�� [Context] Parsing using statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2 or tokens[0].type != USING:
            parser_debug("  ❌ Expected USING keyword")
            return None
        
        # Parse: using(name = expr) { body }
        if tokens[1].type != LPAREN:
            parser_debug("  ❌ Expected '(' after using")
            return None
        
        # Find closing paren and equals
        close_paren_idx = -1
        equals_idx = -1
        paren_count = 1
        
        for i in range(2, len(tokens)):
            if tokens[i].type == LPAREN:
                paren_count += 1
            elif tokens[i].type == RPAREN:
                paren_count -= 1
                if paren_count == 0:
                    close_paren_idx = i
                    break
            elif tokens[i].type == ASSIGN and paren_count == 1:
                equals_idx = i
        
        if close_paren_idx == -1 or equals_idx == -1:
            parser_debug("  ❌ Invalid using statement syntax")
            return None
        
        # Extract resource name
        if tokens[2].type != IDENT:
            parser_debug("  ❌ Expected resource name")
            return None
        
        resource_name = tokens[2].literal
        
        # Extract resource expression
        resource_tokens = tokens[equals_idx + 1:close_paren_idx]
        resource_expr = self._parse_expression(resource_tokens)
        
        # Parse body
        body = BlockStatement()
        for i in range(close_paren_idx + 1, len(tokens)):
            if tokens[i].type == LBRACE:
                # Extract body tokens
                j = i + 1
                brace_count = 1
                body_tokens = []
                
                while j < len(tokens) and brace_count > 0:
                    if tokens[j].type == LBRACE:
                        brace_count += 1
                    elif tokens[j].type == RBRACE:
                        brace_count -= 1
                        if brace_count == 0:
                            break
                    
                    body_tokens.append(tokens[j])
                    j += 1
                
                # Parse body statements
                if body_tokens:
                    body.statements = self._parse_block_statements(body_tokens)
                
                break
        
        print(f"  🔓 Resource: {resource_name}")
        
        return UsingStatement(
            resource_name=Identifier(resource_name),
            resource_expr=resource_expr,
            body=body
        )

    # === CONCURRENCY & PERFORMANCE HANDLERS ===
    def _parse_channel_statement(self, block_info, all_tokens):
        """Parse channel declaration in context-aware mode

        Examples:
          channel<integer> numbers;
          channel messages;
          channel<string>[10] buffered_messages;
        """
        parser_debug("🔧 [Context] Parsing channel statement")
        tokens = block_info.get('tokens', [])

        if not tokens or tokens[0].type != CHANNEL:
            parser_debug("  ❌ Expected CHANNEL keyword")
            return None

        i = 1
        element_type = None
        capacity = None
        name = None

        # Optional generic type: < type_expr >
        if i < len(tokens) and tokens[i].type == LT:
            # collect tokens until GT
            j = i + 1
            type_tokens = []
            while j < len(tokens) and tokens[j].type != GT:
                type_tokens.append(tokens[j])
                j += 1
            if j >= len(tokens) or tokens[j].type != GT:
                parser_debug("  ❌ Unterminated generic type for channel")
                return None
            element_type = self._parse_expression(type_tokens) if type_tokens else None
            i = j + 1

        # Optional capacity in brackets after type or before name
        if i < len(tokens) and tokens[i].type == LBRACKET:
            # expect INT then RBRACKET
            if i+1 < len(tokens) and tokens[i+1].type == INT:
                try:
                    cap_value = int(tokens[i+1].literal)
                    # Wrap in IntegerLiteral node for evaluator
                    from ..zexus_ast import IntegerLiteral
                    capacity = IntegerLiteral(cap_value)
                except Exception:
                    capacity = None
                i += 2
                if i < len(tokens) and tokens[i].type == RBRACKET:
                    i += 1
                else:
                    parser_debug("  ❌ Expected ']' after channel capacity")
                    return None

        # Name
        if i < len(tokens) and tokens[i].type == IDENT:
            name = Identifier(tokens[i].literal)
            i += 1
        else:
            parser_debug("  ❌ Expected channel name")
            return None

        parser_debug(f"  ✅ Channel: {name.value}, type={type(element_type).__name__}, capacity={capacity}")
        return ChannelStatement(name=name, element_type=element_type, capacity=capacity)

    def _parse_send_statement(self, block_info, all_tokens):
        """Parse send(channel, value) statements."""
        parser_debug("🔧 [Context] Parsing send statement")
        tokens = block_info.get('tokens', [])
        if len(tokens) < 3 or tokens[0].type != SEND or tokens[1].type != LPAREN:
            parser_debug("  ❌ Invalid send statement")
            return None

        inner = tokens[2:-1] if tokens and tokens[-1].type == RPAREN else tokens[2:]
        args = self._parse_argument_list(inner)
        if not args or len(args) < 2:
            parser_debug("  ❌ send requires (channel, value)")
            return None

        channel_expr = args[0]
        value_expr = args[1]
        return SendStatement(channel_expr=channel_expr, value_expr=value_expr)

    def _parse_receive_statement(self, block_info, all_tokens):
        """Parse receive(channel) statements. Assignment handled elsewhere."""
        parser_debug("🔧 [Context] Parsing receive statement")
        tokens = block_info.get('tokens', [])
        if len(tokens) < 3 or tokens[0].type != RECEIVE or tokens[1].type != LPAREN:
            parser_debug("  ❌ Invalid receive statement")
            return None

        inner = tokens[2:-1] if tokens and tokens[-1].type == RPAREN else tokens[2:]
        # parse single channel expression
        channel_expr = self._parse_expression(inner)
        if channel_expr is None:
            parser_debug("  ❌ Could not parse channel expression for receive")
            return None

        return ReceiveStatement(channel_expr=channel_expr, target=None)

    def _parse_atomic_statement(self, block_info, all_tokens):
        """Parse atomic blocks or single-expression atomics."""
        parser_debug("🔧 [Context] Parsing atomic statement")
        tokens = block_info.get('tokens', [])
        if not tokens or tokens[0].type != ATOMIC:
            parser_debug("  ❌ Expected ATOMIC keyword")
            return None

        # atomic { ... }
        if len(tokens) > 1 and tokens[1].type == LBRACE:
            # body tokens between braces
            inner = tokens[2:-1] if tokens and tokens[-1].type == RBRACE else tokens[2:]
            stmts = self._parse_block_statements(inner)
            body = BlockStatement()
            body.statements = stmts
            return AtomicStatement(body=body)

        # atomic(expr) or atomic expr
        inner = tokens[1:-1] if len(tokens) > 2 and tokens[-1].type == RPAREN else tokens[1:]
        if inner:
            expr = self._parse_expression(inner)
            return AtomicStatement(expr=expr)

        parser_debug("  ❌ Empty atomic statement")
        return None
    # === BLOCKCHAIN STATEMENT PARSERS ===
    
    def _parse_ledger_statement(self, block_info, all_tokens):
        """Parse ledger NAME = value; statements."""
        parser_debug("🔧 [Context] Parsing ledger statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != LEDGER:
            parser_debug("  ❌ Expected LEDGER keyword")
            return None
        
        # ledger NAME = value
        if len(tokens) < 4 or tokens[1].type != IDENT or tokens[2].type != ASSIGN:
            parser_debug("  ❌ Invalid ledger syntax, expected: ledger NAME = value")
            return None
        
        name = Identifier(tokens[1].literal)
        
        # Parse value expression (from token 3 onwards, excluding semicolon)
        value_tokens = tokens[3:]
        if value_tokens and value_tokens[-1].type == SEMICOLON:
            value_tokens = value_tokens[:-1]
        
        initial_value = self._parse_expression(value_tokens) if value_tokens else None
        
        parser_debug(f"  ✅ Ledger: {name.value}")
        return LedgerStatement(name=name, initial_value=initial_value)
    
    def _parse_state_statement(self, block_info, all_tokens):
        """Parse state NAME = value; statements."""
        parser_debug("🔧 [Context] Parsing state statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != STATE:
            parser_debug("  ❌ Expected STATE keyword")
            return None
        
        # state NAME = value
        if len(tokens) < 4 or tokens[1].type != IDENT or tokens[2].type != ASSIGN:
            parser_debug("  ❌ Invalid state syntax, expected: state NAME = value")
            return None
        
        name = Identifier(tokens[1].literal)
        
        # Parse value expression (from token 3 onwards, excluding semicolon)
        value_tokens = tokens[3:]
        if value_tokens and value_tokens[-1].type == SEMICOLON:
            value_tokens = value_tokens[:-1]
        
        initial_value = self._parse_expression(value_tokens) if value_tokens else None
        
        parser_debug(f"  ✅ State: {name.value}")
        return StateStatement(name=name, initial_value=initial_value)
    
    def _parse_persistent_statement(self, block_info, all_tokens):
        """Parse persistent storage NAME = value; statements.
        
        Forms:
          persistent storage config = { "network": "mainnet" };
          persistent storage balances: map = {};
          persistent storage owner: string;
        """
        parser_debug("🔧 [Context] Parsing persistent statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != PERSISTENT:
            parser_debug("  ❌ Expected PERSISTENT keyword")
            return None
        
        # Expect STORAGE after PERSISTENT
        if len(tokens) < 2 or tokens[1].type != STORAGE:
            parser_debug("  ❌ Expected STORAGE keyword after PERSISTENT")
            return None
        
        # persistent storage NAME = value
        # persistent storage NAME: TYPE = value
        if len(tokens) < 3 or tokens[2].type != IDENT:
            parser_debug("  ❌ Expected identifier after 'persistent storage'")
            return None
        
        name = Identifier(tokens[2].literal)
        type_annotation = None
        initial_value = None
        
        idx = 3
        
        # Check for type annotation (: TYPE)
        if idx < len(tokens) and tokens[idx].type == COLON:
            idx += 1
            if idx < len(tokens) and tokens[idx].type == IDENT:
                type_annotation = tokens[idx].literal
                idx += 1
        
        # Check for initial value (= expression)
        if idx < len(tokens) and tokens[idx].type == ASSIGN:
            idx += 1
            # Parse value expression (from idx onwards, excluding semicolon)
            value_tokens = tokens[idx:]
            if value_tokens and value_tokens[-1].type == SEMICOLON:
                value_tokens = value_tokens[:-1]
            
            initial_value = self._parse_expression(value_tokens) if value_tokens else None
        
        parser_debug(f"  ✅ Persistent storage: {name.value}")
        return PersistentStatement(name=name, type_annotation=type_annotation, initial_value=initial_value)
    
    def _parse_require_statement(self, block_info, all_tokens):
        """Parse require statement with enhanced features.
        
        Forms:
        1. require(condition, message)
        2. require condition, message
        3. require condition { tolerance_block }
        4. require:type condition, message
        5. require \"file.zx\" imported, message
        6. require module \"name\" available, message
        """
        parser_debug("🔧 [Context] Parsing require statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != REQUIRE:
            parser_debug("  ❌ Expected REQUIRE keyword")
            return None
        
        # Check for requirement type: require:balance, require:gas, etc.
        requirement_type = None
        start_idx = 1
        if len(tokens) > 1 and tokens[1].type == COLON:
            if len(tokens) > 2 and tokens[2].type == IDENT:
                requirement_type = tokens[2].literal
                start_idx = 3
                parser_debug(f"  📌 Requirement type: {requirement_type}")
        
        # Check for tolerance block: look for LBRACE after condition
        # We want the FIRST LBRACE (not nested inside the condition)
        tolerance_block = None
        block_start = None
        paren_depth = 0
        for i in range(start_idx, len(tokens)):
            if tokens[i].type == LPAREN:
                paren_depth += 1
            elif tokens[i].type == RPAREN:
                paren_depth -= 1
            elif tokens[i].type == LBRACE and paren_depth == 0:
                block_start = i
                break
        
        if block_start is not None:
            print(f"⚙️ [REQUIRE PARSER] Found LBRACE at index {block_start}")
            # Extract and parse tolerance block
            block_tokens = tokens[block_start:]
            tokens = tokens[:block_start]  # Remove block from main tokens
            print(f"⚙️ [REQUIRE PARSER] Block tokens: {[t.literal for t in block_tokens[:10]]}")
            
            # Parse as BlockStatement: strip outer braces and parse statements
            if block_tokens and block_tokens[0].type == LBRACE:
                # Remove outer braces
                inner_tokens = block_tokens[1:]
                if inner_tokens and inner_tokens[-1].type == RBRACE:
                    inner_tokens = inner_tokens[:-1]
                print(f"⚙️ [REQUIRE PARSER] Inner tokens (no braces): {len(inner_tokens)} tokens")
                
                # Parse statements inside the block
                print(f"⚙️ [REQUIRE PARSER] About to call _parse_block_statements")
                statements = self._parse_block_statements(inner_tokens)
                print(f"⚙️ [REQUIRE PARSER] Got {len(statements) if statements else 0} statements back")
                tolerance_block = BlockStatement()
                tolerance_block.statements = statements
                print(f"⚙️ [REQUIRE PARSER] Created BlockStatement with {len(statements)} statements")
                parser_debug(f"  🧩 Found tolerance block with {len(statements)} statements")
            else:
                tolerance_block = None
        
        # Check for file/module dependencies
        file_path = None
        module_name = None
        
        # Pattern: require \"file.zx\" imported
        if start_idx < len(tokens) and tokens[start_idx].type == STRING:
            file_path = tokens[start_idx].literal
            start_idx += 1
            # Look for 'imported' keyword
            if start_idx < len(tokens) and tokens[start_idx].type == IDENT and tokens[start_idx].literal == 'imported':
                start_idx += 1
                # Look for comma and message
                message = None
                if start_idx < len(tokens) and tokens[start_idx].type == COMMA:
                    start_idx += 1
                    if start_idx < len(tokens):
                        message_tokens = tokens[start_idx:]
                        message = self._parse_expression(message_tokens)
                
                parser_debug(f"  ✅ File dependency: {file_path}")
                return RequireStatement(
                    condition=None,
                    message=message,
                    file_path=file_path,
                    requirement_type='file'
                )
        
        # Pattern: require module \"name\" available
        if start_idx < len(tokens) and tokens[start_idx].type == IDENT and tokens[start_idx].literal == 'module':
            start_idx += 1
            if start_idx < len(tokens) and tokens[start_idx].type == STRING:
                module_name = tokens[start_idx].literal
                start_idx += 1
                # Look for 'available' keyword
                if start_idx < len(tokens) and tokens[start_idx].type == IDENT and tokens[start_idx].literal == 'available':
                    start_idx += 1
                    # Look for comma and message
                    message = None
                    if start_idx < len(tokens) and tokens[start_idx].type == COMMA:
                        start_idx += 1
                        if start_idx < len(tokens):
                            message_tokens = tokens[start_idx:]
                            message = self._parse_expression(message_tokens)
                    
                    parser_debug(f"  ✅ Module dependency: {module_name}")
                    return RequireStatement(
                        condition=None,
                        message=message,
                        module_name=module_name,
                        requirement_type='module'
                    )
        
        # Check for parenthesized form: require(condition, message)
        if start_idx < len(tokens) and tokens[start_idx].type == LPAREN:
            # Extract tokens between LPAREN and RPAREN
            inner = tokens[start_idx+1:-1] if len(tokens) > start_idx+1 and tokens[-1].type == RPAREN else tokens[start_idx+1:]
            
            # Split by comma to get condition and optional message
            args = self._parse_argument_list(inner)
            
            if not args:
                parser_debug("  ❌ require() needs at least one argument")
                return None
            
            condition = args[0]
            message = args[1] if len(args) > 1 else None
            
            parser_debug(f"  ✅ Require() with {len(args)} arguments")
            return RequireStatement(
                condition=condition,
                message=message,
                tolerance_block=tolerance_block,
                requirement_type=requirement_type
            )
        
        # Non-parenthesized form: require condition, message
        # Find comma for message
        comma_idx = None
        paren_depth = 0
        for i, tok in enumerate(tokens[start_idx:], start_idx):
            if tok.type in {LPAREN, LBRACKET}:
                paren_depth += 1
            elif tok.type in {RPAREN, RBRACKET}:
                paren_depth -= 1
            elif tok.type == COMMA and paren_depth == 0:
                comma_idx = i
                break
        
        if comma_idx:
            cond_tokens = tokens[start_idx:comma_idx]
            msg_tokens = tokens[comma_idx+1:]
            condition = self._parse_expression(cond_tokens) if cond_tokens else None
            message = self._parse_expression(msg_tokens) if msg_tokens else None
        else:
            cond_tokens = tokens[start_idx:]
            condition = self._parse_expression(cond_tokens) if cond_tokens else None
            message = None
        
        if not condition:
            parser_debug("  ❌ require needs a condition")
            return None
        
        parser_debug(f"  ✅ Require statement")
        print(f"⚙️ [REQUIRE PARSER] Creating RequireStatement with tolerance_block={type(tolerance_block).__name__ if tolerance_block else 'None'}")
        stmt = RequireStatement(
            condition=condition,
            message=message,
            tolerance_block=tolerance_block,
            requirement_type=requirement_type
        )
        print(f"⚙️ [REQUIRE PARSER] Created stmt.tolerance_block={type(stmt.tolerance_block).__name__ if stmt.tolerance_block else 'None'}")
        return stmt
    
    def _parse_revert_statement(self, block_info, all_tokens):
        """Parse revert(reason) statements."""
        parser_debug("🔧 [Context] Parsing revert statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != REVERT:
            parser_debug("  ❌ Expected REVERT keyword")
            return None
        
        reason = None
        
        # revert() or revert(reason)
        if len(tokens) > 1 and tokens[1].type == LPAREN:
            inner = tokens[2:-1] if len(tokens) > 2 and tokens[-1].type == RPAREN else tokens[2:]
            if inner:
                reason = self._parse_expression(inner)
        
        parser_debug("  ✅ Revert statement")
        return RevertStatement(reason=reason)
    
    def _parse_limit_statement(self, block_info, all_tokens):
        """Parse limit(amount) statements."""
        parser_debug("🔧 [Context] Parsing limit statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != LIMIT or (len(tokens) > 1 and tokens[1].type != LPAREN):
            parser_debug("  ❌ Expected limit()")
            return None
        
        # Extract tokens between LPAREN and RPAREN
        inner = tokens[2:-1] if len(tokens) > 2 and tokens[-1].type == RPAREN else tokens[2:]
        
        gas_limit = self._parse_expression(inner) if inner else None
        
        if gas_limit is None:
            parser_debug("  ❌ limit needs a gas amount")
            return None
        
        parser_debug("  ✅ Limit statement")
        return LimitStatement(amount=gas_limit)

    def _parse_stream_statement(self, block_info, all_tokens):
        """Parse stream statement.
        
        Form: stream name as event_var => { handler }
        Example: stream clicks as event => { print event.x; }
        """
        parser_debug("🔧 [Context] Parsing stream statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != STREAM:
            parser_debug("  ❌ Expected STREAM keyword")
            return None
        
        # stream name as event_var => { handler }
        # Find 'as' keyword (or AS token if it exists)
        as_idx = -1
        for i, t in enumerate(tokens):
            if t.literal.lower() == 'as' or t.type == IDENT and t.literal == 'as':
                as_idx = i
                break
        
        if as_idx == -1:
            parser_debug("  ❌ Expected 'as' in stream statement")
            return None
        
        # Extract stream name (between STREAM and 'as')
        stream_name = tokens[1].literal if as_idx > 1 else "unknown"
        
        # Extract event variable name (after 'as')
        event_var = None
        if as_idx + 1 < len(tokens):
            event_var = Identifier(value=tokens[as_idx + 1].literal)
        
        # Find '=>' arrow
        arrow_idx = -1
        for i in range(as_idx + 2, len(tokens)):
            if tokens[i].literal == '=>':
                arrow_idx = i
                break
        
        if arrow_idx == -1:
            parser_debug("  ❌ Expected '=>' in stream statement")
            return None
        
        # Parse handler block (after '=>')
        handler_tokens = tokens[arrow_idx + 1:]
        if handler_tokens and handler_tokens[0].type == LBRACE:
            inner = handler_tokens[1:-1] if handler_tokens[-1].type == RBRACE else handler_tokens[1:]
            stmts = self._parse_block_statements(inner)
            handler = BlockStatement()
            handler.statements = stmts
        else:
            parser_debug("  ❌ Expected block after '=>'")
            return None
        
        parser_debug(f"  ✅ Stream statement: {stream_name} as {event_var}")
        return StreamStatement(
            stream_name=stream_name,
            event_var=event_var,
            handler=handler
        )
    
    def _parse_watch_statement(self, block_info, all_tokens):
        """Parse watch statement.
        
        Forms:
        1. watch { ... }  (Implicit dependencies)
        2. watch expr => { ... } (Explicit dependencies)
        """
        parser_debug("🔧 [Context] Parsing watch statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != WATCH:
            parser_debug("  ❌ Expected WATCH keyword")
            return None
            
        # Check for form 1: watch { ... }
        if len(tokens) > 1 and tokens[1].type == LBRACE:
            # Extract body
            inner = tokens[2:-1] if tokens[-1].type == RBRACE else tokens[2:]
            stmts = self._parse_block_statements(inner)
            body = BlockStatement()
            body.statements = stmts
            
            parser_debug("  ✅ Watch statement (implicit)")
            return WatchStatement(reaction=body, watched_expr=None)
            
        # Check for form 2: watch expr => ...
        # Find '=>' (tokenized as LAMBDA)
        
        arrow_idx = -1
        for i, t in enumerate(tokens):
            if t.type == LAMBDA or t.literal == '=>':
                arrow_idx = i
                break
                
        if arrow_idx != -1:
            expr_tokens = tokens[1:arrow_idx]
            reaction_tokens = tokens[arrow_idx+1:]
            
            watched_expr = self._parse_expression(expr_tokens)
            
            # Parse reaction
            if reaction_tokens and reaction_tokens[0].type == LBRACE:
                inner = reaction_tokens[1:-1] if reaction_tokens[-1].type == RBRACE else reaction_tokens[1:]
                stmts = self._parse_block_statements(inner)
                reaction = BlockStatement()
                reaction.statements = stmts
            else:
                # Single expression reaction
                reaction_expr = self._parse_expression(reaction_tokens)
                reaction = reaction_expr
                
            parser_debug("  ✅ Watch statement (explicit)")
            return WatchStatement(reaction=reaction, watched_expr=watched_expr)
            
        parser_debug("  ❌ Invalid watch syntax")
        return None

    def _parse_protect_statement(self, block_info, all_tokens):
        """Parse protect statement.
        
        Form: protect(<target>, <rules>, <level>)
        Example: protect(transfer_funds, {rate_limit: 10}, "strict")
        """
        parser_debug("🔧 [Context] Parsing protect statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != PROTECT:
            parser_debug("  ❌ Expected PROTECT keyword")
            return None
        
        # Check if it's function-call style: protect(...)
        if len(tokens) > 1 and tokens[1].type == LPAREN:
            # Find matching RPAREN
            paren_depth = 0
            rparen_idx = -1
            for i in range(1, len(tokens)):
                if tokens[i].type == LPAREN:
                    paren_depth += 1
                elif tokens[i].type == RPAREN:
                    paren_depth -= 1
                    if paren_depth == 0:
                        rparen_idx = i
                        break
            
            if rparen_idx == -1:
                parser_debug("  ❌ Unmatched parentheses in protect")
                return None
            
            # Parse arguments: target, rules, [enforcement_level]
            args_tokens = tokens[2:rparen_idx]
            args = []
            current_arg = []
            depth = 0
            
            for t in args_tokens:
                if t.type in (LPAREN, LBRACE, LBRACKET):
                    depth += 1
                    current_arg.append(t)
                elif t.type in (RPAREN, RBRACE, RBRACKET):
                    depth -= 1
                    current_arg.append(t)
                elif t.type == COMMA and depth == 0:
                    if current_arg:
                        args.append(current_arg)
                        current_arg = []
                else:
                    current_arg.append(t)
            
            if current_arg:
                args.append(current_arg)
            
            # Parse target (first argument)
            target = self._parse_expression(args[0]) if len(args) > 0 else None
            
            # Parse rules (second argument - should be a map literal)
            rules = self._parse_expression(args[1]) if len(args) > 1 else None
            
            # Parse enforcement level (third argument - optional string)
            enforcement_level = None
            if len(args) > 2:
                level_expr = self._parse_expression(args[2])
                if isinstance(level_expr, StringLiteral):
                    enforcement_level = level_expr.value
            
            parser_debug("  ✅ Protect statement (function-call style)")
            stmt = ProtectStatement(target=target, rules=rules)
            stmt.enforcement_level = enforcement_level
            return stmt
        
        # Old style: protect <target> { <rules> }
        # Find LBRACE to separate target from rules
        brace_idx = -1
        for i, t in enumerate(tokens):
            if t.type == LBRACE:
                brace_idx = i
                break
        
        if brace_idx == -1:
            parser_debug("  ❌ Expected { for protect rules")
            return None
        
        # Parse target
        target_tokens = tokens[1:brace_idx]
        target = self._parse_expression(target_tokens)
        
        # Parse rules (inner block)
        inner = tokens[brace_idx+1:-1] if tokens[-1].type == RBRACE else tokens[brace_idx+1:]
        rules = self._parse_block_statements(inner)
        rules_block = BlockStatement()
        rules_block.statements = rules
        
        parser_debug("  ✅ Protect statement (block style)")
        return ProtectStatement(target=target, rules=rules_block)

    def _parse_middleware_statement(self, block_info, all_tokens):
        """Parse middleware statement.
        
        Form: middleware(name, action(req, res) { ... })
        Example: middleware("authenticate", action(request, response) { ... })
        """
        parser_debug("🔧 [Context] Parsing middleware statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != MIDDLEWARE:
            parser_debug("  ❌ Expected MIDDLEWARE keyword")
            return None
        
        # Expect LPAREN after middleware
        if len(tokens) < 2 or tokens[1].type != LPAREN:
            parser_debug("  ❌ Expected ( after middleware")
            return None
        
        # Find matching RPAREN
        paren_depth = 0
        rparen_idx = -1
        for i in range(1, len(tokens)):
            if tokens[i].type == LPAREN:
                paren_depth += 1
            elif tokens[i].type == RPAREN:
                paren_depth -= 1
                if paren_depth == 0:
                    rparen_idx = i
                    break
        
        if rparen_idx == -1:
            parser_debug("  ❌ Unmatched parentheses")
            return None
        
        # Parse arguments: name, handler
        args_tokens = tokens[2:rparen_idx]
        
        # Find comma separating name and handler
        comma_idx = -1
        depth = 0
        for i, tok in enumerate(args_tokens):
            if tok.type in {LPAREN, LBRACE, LBRACKET}:
                depth += 1
            elif tok.type in {RPAREN, RBRACE, RBRACKET}:
                depth -= 1
            elif tok.type == COMMA and depth == 0:
                comma_idx = i
                break
        
        if comma_idx == -1:
            parser_debug("  ❌ Expected comma between name and handler")
            return None
        
        name_tokens = args_tokens[:comma_idx]
        handler_tokens = args_tokens[comma_idx+1:]
        
        name = self._parse_expression(name_tokens)
        handler = self._parse_expression(handler_tokens)
        
        parser_debug("  ✅ Middleware statement")
        return MiddlewareStatement(name=name, handler=handler)

    def _parse_auth_statement(self, block_info, all_tokens):
        """Parse auth statement.
        
        Form: auth { provider: "oauth2", scopes: ["read", "write"] }
        """
        parser_debug("🔧 [Context] Parsing auth statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != AUTH:
            parser_debug("  ❌ Expected AUTH keyword")
            return None
        
        # Expect LBRACE after auth
        if len(tokens) < 2 or tokens[1].type != LBRACE:
            parser_debug("  ❌ Expected { after auth")
            return None
        
        # Find matching RBRACE
        brace_depth = 0
        rbrace_idx = -1
        for i in range(1, len(tokens)):
            if tokens[i].type == LBRACE:
                brace_depth += 1
            elif tokens[i].type == RBRACE:
                brace_depth -= 1
                if brace_depth == 0:
                    rbrace_idx = i
                    break
        
        if rbrace_idx == -1:
            parser_debug("  ❌ Unmatched braces")
            return None
        
        # Parse config map
        config_tokens = tokens[2:rbrace_idx]
        config = self._parse_map_literal_tokens(config_tokens)
        
        parser_debug("  ✅ Auth statement")
        return AuthStatement(config=config)

    def _parse_throttle_statement(self, block_info, all_tokens):
        """Parse throttle statement.
        
        Form: throttle(target, { requests_per_minute: 100 })
        """
        parser_debug("🔧 [Context] Parsing throttle statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != THROTTLE:
            parser_debug("  ❌ Expected THROTTLE keyword")
            return None
        
        # Expect LPAREN after throttle
        if len(tokens) < 2 or tokens[1].type != LPAREN:
            parser_debug("  ❌ Expected ( after throttle")
            return None
        
        # Find matching RPAREN
        paren_depth = 0
        rparen_idx = -1
        for i in range(1, len(tokens)):
            if tokens[i].type == LPAREN:
                paren_depth += 1
            elif tokens[i].type == RPAREN:
                paren_depth -= 1
                if paren_depth == 0:
                    rparen_idx = i
                    break
        
        if rparen_idx == -1:
            parser_debug("  ❌ Unmatched parentheses")
            return None
        
        # Parse arguments: target, limits
        args_tokens = tokens[2:rparen_idx]
        
        # Find comma separating target and limits
        comma_idx = -1
        depth = 0
        for i, tok in enumerate(args_tokens):
            if tok.type in {LPAREN, LBRACE, LBRACKET}:
                depth += 1
            elif tok.type in {RPAREN, RBRACE, RBRACKET}:
                depth -= 1
            elif tok.type == COMMA and depth == 0:
                comma_idx = i
                break
        
        if comma_idx == -1:
            parser_debug("  ❌ Expected comma between target and limits")
            return None
        
        target_tokens = args_tokens[:comma_idx]
        limits_tokens = args_tokens[comma_idx+1:]
        
        target = self._parse_expression(target_tokens)
        limits = self._parse_expression(limits_tokens)
        
        parser_debug("  ✅ Throttle statement")
        return ThrottleStatement(target=target, limits=limits)

    def _parse_cache_statement(self, block_info, all_tokens):
        """Parse cache statement.
        
        Form: cache(target, { ttl: 3600 })
        """
        parser_debug("🔧 [Context] Parsing cache statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != CACHE:
            parser_debug("  ❌ Expected CACHE keyword")
            return None
        
        # Expect LPAREN after cache
        if len(tokens) < 2 or tokens[1].type != LPAREN:
            parser_debug("  ❌ Expected ( after cache")
            return None
        
        # Find matching RPAREN
        paren_depth = 0
        rparen_idx = -1
        for i in range(1, len(tokens)):
            if tokens[i].type == LPAREN:
                paren_depth += 1
            elif tokens[i].type == RPAREN:
                paren_depth -= 1
                if paren_depth == 0:
                    rparen_idx = i
                    break
        
        if rparen_idx == -1:
            parser_debug("  ❌ Unmatched parentheses")
            return None
        
        # Parse arguments: target, policy
        args_tokens = tokens[2:rparen_idx]
        
        # Find comma separating target and policy
        comma_idx = -1
        depth = 0
        for i, tok in enumerate(args_tokens):
            if tok.type in {LPAREN, LBRACE, LBRACKET}:
                depth += 1
            elif tok.type in {RPAREN, RBRACE, RBRACKET}:
                depth -= 1
            elif tok.type == COMMA and depth == 0:
                comma_idx = i
                break
        
        if comma_idx == -1:
            parser_debug("  ❌ Expected comma between target and policy")
            return None
        
        target_tokens = args_tokens[:comma_idx]
        policy_tokens = args_tokens[comma_idx+1:]
        
        target = self._parse_expression(target_tokens)
        policy = self._parse_expression(policy_tokens)
        
        parser_debug("  ✅ Cache statement")
        return CacheStatement(target=target, policy=policy)

    def _parse_verify_statement(self, block_info, all_tokens):
        """Parse verify statement with extended syntax.
        
        Forms:
        1. verify condition, "message"
        2. verify (condition)
        3. verify(target, [conditions])
        4. verify:mode condition, "message"
        5. verify condition { logic_block }
        6. verify:data value matches pattern, "message"
        7. verify:db value exists_in "table", "message"
        8. verify:env "VAR" is_set, "message"
        9. verify:access condition { action_block }
        
        Examples:
        - verify false, "Access denied"
        - verify (TX.caller == self.owner)
        - verify:data email matches email_pattern, "Invalid email"
        - verify:access user.role == "admin" { block_request(); }
        - verify:db user_id exists_in "users", "User not found"
        - verify:env "API_KEY" is_set, "API_KEY required"
        - verify condition { log_error("Failed"); return false; }
        """
        parser_debug("🔧 [Context] Parsing verify statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != VERIFY:
            parser_debug("  ❌ Expected VERIFY keyword")
            return None
        
        # Check for mode syntax: verify:mode
        mode = None
        start_idx = 1
        if len(tokens) > 1 and tokens[1].type == COLON:
            # Extract mode
            if len(tokens) > 2 and tokens[2].type == IDENT:
                mode = tokens[2].literal
                start_idx = 3
                parser_debug(f"  📌 Detected mode: {mode}")
        
        # Check for logic block at the end
        logic_block = None
        message = None
        block_start = None
        block_end_idx = None  # Track where block ends
        
        # Find the block
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i].type == LBRACE:
                block_start = i
                # Find matching RBRACE
                brace_count = 0
                for j in range(i, len(tokens)):
                    if tokens[j].type == LBRACE:
                        brace_count += 1
                    elif tokens[j].type == RBRACE:
                        brace_count -= 1
                        if brace_count == 0:
                            block_end_idx = j
                            break
                break
        
        # Check for comma AFTER the block (if block exists)
        if block_end_idx and block_end_idx + 1 < len(tokens) and tokens[block_end_idx + 1].type == COMMA:
            # This is the pattern: verify { ... }, "message"
            # Parse the block
            block_tokens = tokens[block_start:block_end_idx+1]
            brace_count = 0
            block_end = None
            for i, tok in enumerate(block_tokens):
                if tok.type == LBRACE:
                    brace_count += 1
                elif tok.type == RBRACE:
                    brace_count -= 1
                    if brace_count == 0:
                        block_end = i
                        break
            
            if block_end and block_end > 1:
                inner_tokens = block_tokens[1:block_end]
                
                # Split inner tokens by commas to get individual conditions
                condition_token_groups = []
                current_group = []
                paren_depth = 0
                
                for tok in inner_tokens:
                    if tok.type == LPAREN:
                        paren_depth += 1
                    elif tok.type == RPAREN:
                        paren_depth -= 1
                    
                    if tok.type == COMMA and paren_depth == 0:
                        # Comma at depth 0 separates conditions
                        if current_group:
                            condition_token_groups.append(current_group)
                            current_group = []
                    else:
                        current_group.append(tok)
                
                # Add the last group
                if current_group:
                    condition_token_groups.append(current_group)
                
                # Parse each condition as an expression and wrap in ExpressionStatement
                from ..zexus_ast import ExpressionStatement
                inner_statements = []
                for group in condition_token_groups:
                    if group:
                        expr = self._parse_expression(group)
                        if expr:
                            inner_statements.append(ExpressionStatement(expr))
                
                logic_block = BlockStatement()
                logic_block.statements = inner_statements
            else:
                logic_block = BlockStatement()
            
            # Parse the message
            message_tokens = tokens[block_end_idx + 2:]
            message = self._parse_expression(message_tokens) if message_tokens else None
            
            parser_debug(f"  ✅ Verify block with {len(logic_block.statements) if logic_block else 0} conditions and message")
            return VerifyStatement(condition=None, message=message, logic_block=logic_block)
        
        # Handle logic block WITHOUT comma (verify condition { ... })
        if block_start:
            # Extract and parse logic block
            block_tokens = tokens[block_start:block_end_idx+1] if block_end_idx else tokens[block_start:]
            tokens = tokens[:block_start]  # Remove block from main tokens
            
            # Parse the block: skip LBRACE and find matching RBRACE
            brace_count = 0
            block_end = None
            for i, tok in enumerate(block_tokens):
                if tok.type == LBRACE:
                    brace_count += 1
                elif tok.type == RBRACE:
                    brace_count -= 1
                    if brace_count == 0:
                        block_end = i
                        break
            
            if block_end and block_end > 1:
                # Extract tokens between braces
                inner_tokens = block_tokens[1:block_end]
                inner_statements = self._parse_block_statements(inner_tokens)
                logic_block = BlockStatement()
                logic_block.statements = inner_statements
            else:
                logic_block = BlockStatement()
            
            parser_debug(f"  🧩 Found logic block with {len(logic_block.statements) if logic_block else 0} statements")
        
        # Parse based on mode
        if mode == 'data':
            return self._parse_verify_data(tokens[start_idx:], logic_block)
        elif mode == 'access':
            return self._parse_verify_access(tokens[start_idx:], logic_block)
        elif mode == 'db':
            return self._parse_verify_db(tokens[start_idx:], logic_block)
        elif mode == 'env':
            return self._parse_verify_env(tokens[start_idx:], logic_block)
        elif mode == 'pattern':
            return self._parse_verify_pattern(tokens[start_idx:], logic_block)
        
        # Check for comma-separated format: verify condition, message
        comma_idx = None
        paren_depth = 0
        for i, tok in enumerate(tokens[start_idx:], start_idx):
            if tok.type in {LPAREN, LBRACKET}:
                paren_depth += 1
            elif tok.type in {RPAREN, RBRACKET}:
                paren_depth -= 1
            elif tok.type == COMMA and paren_depth == 0:
                comma_idx = i
                break
        
        if comma_idx:
            # Format: verify condition, message
            cond_tokens = tokens[start_idx:comma_idx]
            msg_tokens = tokens[comma_idx+1:]
            
            condition = self._parse_expression(cond_tokens) if cond_tokens else None
            message = self._parse_expression(msg_tokens) if msg_tokens else None
            
            parser_debug(f"  ✅ Verify statement (simple assertion) with message")
            return VerifyStatement(condition=condition, message=message, logic_block=logic_block)
        else:
            # Format: verify (condition) or verify(target, [...])
            inner = tokens[2:-1] if len(tokens) > 2 and tokens[-1].type == RPAREN else tokens[start_idx:]
            
            condition = self._parse_expression(inner) if inner else None
            
            if condition is None:
                parser_debug("  ❌ verify needs a condition")
                return None
            
            parser_debug("  ✅ Verify statement (parenthesized)")
            return VerifyStatement(condition=condition, logic_block=logic_block)
    
    def _parse_verify_data(self, tokens, logic_block):
        """Parse verify:data - data/format verification"""
        parser_debug("  📊 Parsing verify:data")
        # Format: verify:data value matches pattern, "message"
        #         verify:data value is_type "string", "message"
        
        # Find keywords: matches, is_type, exists_in, etc.
        keyword_idx = None
        keyword = None
        for i, tok in enumerate(tokens):
            if tok.type == IDENT and tok.literal in ['matches', 'is_type', 'is', 'equals']:
                keyword_idx = i
                keyword = tok.literal
                break
        
        if not keyword_idx:
            parser_debug("  ❌ No data verification keyword found")
            return None
        
        value_tokens = tokens[:keyword_idx]
        value_expr = self._parse_expression(value_tokens) if value_tokens else None
        
        # Find comma for message
        comma_idx = None
        for i, tok in enumerate(tokens[keyword_idx+1:], keyword_idx+1):
            if tok.type == COMMA:
                comma_idx = i
                break
        
        if comma_idx:
            pattern_tokens = tokens[keyword_idx+1:comma_idx]
            message_tokens = tokens[comma_idx+1:]
        else:
            pattern_tokens = tokens[keyword_idx+1:]
            message_tokens = []
        
        pattern = self._parse_expression(pattern_tokens) if pattern_tokens else None
        message = self._parse_expression(message_tokens) if message_tokens else None
        
        parser_debug(f"  ✅ verify:data with keyword '{keyword}'")
        return VerifyStatement(
            mode='data',
            condition=value_expr,
            pattern=pattern,
            message=message,
            logic_block=logic_block,
            verify_type=keyword
        )
    
    def _parse_verify_access(self, tokens, logic_block):
        """Parse verify:access - access control with blocking"""
        parser_debug("  🔒 Parsing verify:access")
        # Format: verify:access condition { block }
        
        # Find comma if present
        comma_idx = None
        for i, tok in enumerate(tokens):
            if tok.type == COMMA:
                comma_idx = i
                break
        
        if comma_idx:
            cond_tokens = tokens[:comma_idx]
            message_tokens = tokens[comma_idx+1:]
            message = self._parse_expression(message_tokens) if message_tokens else None
        else:
            cond_tokens = tokens
            message = None
        
        condition = self._parse_expression(cond_tokens) if cond_tokens else None
        
        parser_debug(f"  ✅ verify:access with action block")
        return VerifyStatement(
            mode='access',
            condition=condition,
            message=message,
            action_block=logic_block
        )
    
    def _parse_verify_db(self, tokens, logic_block):
        """Parse verify:db - database verification"""
        parser_debug("  🗄️ Parsing verify:db")
        # Format: verify:db value exists_in "table", "message"
        #         verify:db value unique_in "table", "message"
        
        # Find keywords: exists_in, unique_in
        keyword_idx = None
        keyword = None
        for i, tok in enumerate(tokens):
            if tok.type == IDENT and tok.literal in ['exists_in', 'unique_in', 'matches_in']:
                keyword_idx = i
                keyword = tok.literal
                break
        
        if not keyword_idx:
            parser_debug("  ❌ No db verification keyword found")
            return None
        
        value_tokens = tokens[:keyword_idx]
        value_expr = self._parse_expression(value_tokens) if value_tokens else None
        
        # Find commas
        comma_indices = []
        for i, tok in enumerate(tokens[keyword_idx+1:], keyword_idx+1):
            if tok.type == COMMA:
                comma_indices.append(i)
        
        # Parse table and message
        if len(comma_indices) >= 1:
            table_tokens = tokens[keyword_idx+1:comma_indices[0]]
            message_tokens = tokens[comma_indices[0]+1:] if len(comma_indices) > 0 else []
        else:
            table_tokens = tokens[keyword_idx+1:]
            message_tokens = []
        
        table = self._parse_expression(table_tokens) if table_tokens else None
        message = self._parse_expression(message_tokens) if message_tokens else None
        
        parser_debug(f"  ✅ verify:db with keyword '{keyword}'")
        return VerifyStatement(
            mode='db',
            condition=value_expr,
            db_table=table,
            db_query=keyword,  # exists_in, unique_in, etc.
            message=message,
            logic_block=logic_block
        )
    
    def _parse_verify_env(self, tokens, logic_block):
        """Parse verify:env - environment variable verification"""
        parser_debug("  🌍 Parsing verify:env")
        # Format: verify:env "VAR_NAME" is_set, "message"
        #         verify:env "VAR_NAME" equals "value", "message"
        
        # Find keywords: is_set, equals, matches
        keyword_idx = None
        keyword = None
        for i, tok in enumerate(tokens):
            if tok.type == IDENT and tok.literal in ['is_set', 'equals', 'matches', 'exists']:
                keyword_idx = i
                keyword = tok.literal
                break
        
        if not keyword_idx:
            # Simple form: verify:env "VAR_NAME", "message"
            comma_idx = None
            for i, tok in enumerate(tokens):
                if tok.type == COMMA:
                    comma_idx = i
                    break
            
            if comma_idx:
                var_tokens = tokens[:comma_idx]
                msg_tokens = tokens[comma_idx+1:]
            else:
                var_tokens = tokens
                msg_tokens = []
            
            env_var = self._parse_expression(var_tokens) if var_tokens else None
            message = self._parse_expression(msg_tokens) if msg_tokens else None
            
            parser_debug(f"  ✅ verify:env (simple check)")
            return VerifyStatement(
                mode='env',
                env_var=env_var,
                message=message,
                logic_block=logic_block,
                verify_type='is_set'
            )
        
        # Complex form with keyword
        var_tokens = tokens[:keyword_idx]
        env_var = self._parse_expression(var_tokens) if var_tokens else None
        
        # Find comma
        comma_idx = None
        for i, tok in enumerate(tokens[keyword_idx+1:], keyword_idx+1):
            if tok.type == COMMA:
                comma_idx = i
                break
        
        if comma_idx:
            expected_tokens = tokens[keyword_idx+1:comma_idx]
            message_tokens = tokens[comma_idx+1:]
        else:
            expected_tokens = tokens[keyword_idx+1:]
            message_tokens = []
        
        expected_value = self._parse_expression(expected_tokens) if expected_tokens else None
        message = self._parse_expression(message_tokens) if message_tokens else None
        
        parser_debug(f"  ✅ verify:env with keyword '{keyword}'")
        return VerifyStatement(
            mode='env',
            env_var=env_var,
            expected_value=expected_value,
            message=message,
            logic_block=logic_block,
            verify_type=keyword
        )
    
    def _parse_verify_pattern(self, tokens, logic_block):
        """Parse verify:pattern - pattern matching verification"""
        parser_debug("  🎯 Parsing verify:pattern")
        # Format: verify:pattern value matches "/regex/", "message"
        
        # Find 'matches' keyword
        matches_idx = None
        for i, tok in enumerate(tokens):
            if tok.type == IDENT and tok.literal == 'matches':
                matches_idx = i
                break
        
        if not matches_idx:
            parser_debug("  ❌ No 'matches' keyword found")
            return None
        
        value_tokens = tokens[:matches_idx]
        value_expr = self._parse_expression(value_tokens) if value_tokens else None
        
        # Find comma
        comma_idx = None
        for i, tok in enumerate(tokens[matches_idx+1:], matches_idx+1):
            if tok.type == COMMA:
                comma_idx = i
                break
        
        if comma_idx:
            pattern_tokens = tokens[matches_idx+1:comma_idx]
            message_tokens = tokens[comma_idx+1:]
        else:
            pattern_tokens = tokens[matches_idx+1:]
            message_tokens = []
        
        pattern = self._parse_expression(pattern_tokens) if pattern_tokens else None
        message = self._parse_expression(message_tokens) if message_tokens else None
        
        parser_debug(f"  ✅ verify:pattern")
        return VerifyStatement(
            mode='pattern',
            condition=value_expr,
            pattern=pattern,
            message=message,
            logic_block=logic_block
        )

    def _parse_restrict_statement(self, block_info, all_tokens):
        """Parse restrict statement (already exists in AST).
        
        This method is kept for compatibility, but RestrictStatement
        is already handled in the parser. 
        """
        parser_debug("🔧 [Context] Parsing restrict statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != RESTRICT:
            parser_debug("  ❌ Expected RESTRICT keyword")
            return None
        
        # Let the existing restrict parsing handle this
        # Just return None to fall through to default handling
        return None

    def _parse_inject_statement(self, block_info, all_tokens):
        """Parse inject statement.
        
        Form: inject <dependency_name>
        Example: inject DatabaseAPI
        """
        parser_debug("🔧 [Context] Parsing inject statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != INJECT:
            parser_debug("  ❌ Expected INJECT keyword")
            return None
        
        if len(tokens) < 2 or tokens[1].type != IDENT:
            parser_debug("  ❌ inject needs a dependency name")
            return None
        
        dependency_name = tokens[1].literal
        
        parser_debug(f"  ✅ Inject statement: {dependency_name}")
        return InjectStatement(dependency=Identifier(value=dependency_name))

    def _parse_validate_statement(self, block_info, all_tokens):
        """Parse validate statement.
        
        Form: validate ( <value>, <schema> )
        Example: validate (user_input, email_schema)
        """
        parser_debug("🔧 [Context] Parsing validate statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != VALIDATE:
            parser_debug("  ❌ Expected VALIDATE keyword")
            return None
        
        # Extract tokens between LPAREN and RPAREN
        inner = tokens[2:-1] if len(tokens) > 2 and tokens[-1].type == RPAREN else tokens[2:]
        
        # Split by COMMA to get value and schema
        comma_idx = -1
        for i, t in enumerate(inner):
            if t.type == COMMA:
                comma_idx = i
                break
        
        if comma_idx == -1:
            parser_debug("  ❌ validate needs value and schema")
            return None
        
        value_tokens = inner[:comma_idx]
        schema_tokens = inner[comma_idx+1:]
        
        data = self._parse_expression(value_tokens)
        schema = self._parse_expression(schema_tokens)
        
        parser_debug("  ✅ Validate statement")
        return ValidateStatement(data=data, schema=schema)

    def _parse_sanitize_statement(self, block_info, all_tokens):
        """Parse sanitize statement.
        
        Form: sanitize ( <value>, <rules> )
        Example: sanitize (user_input, html_rules)
        """
        parser_debug("🔧 [Context] Parsing sanitize statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != SANITIZE:
            parser_debug("  ❌ Expected SANITIZE keyword")
            return None
        
        # Extract tokens between LPAREN and RPAREN
        inner = tokens[2:-1] if len(tokens) > 2 and tokens[-1].type == RPAREN else tokens[2:]
        
        # Split by COMMA to get value and rules
        comma_idx = -1
        for i, t in enumerate(inner):
            if t.type == COMMA:
                comma_idx = i
                break
        
        if comma_idx == -1:
            parser_debug("  ❌ sanitize needs value and rules")
            return None
        
        value_tokens = inner[:comma_idx]
        rules_tokens = inner[comma_idx+1:]
        
        data = self._parse_expression(value_tokens)
        rules = self._parse_expression(rules_tokens)
        
        parser_debug("  ✅ Sanitize statement")
        return SanitizeStatement(data=data, rules=rules)