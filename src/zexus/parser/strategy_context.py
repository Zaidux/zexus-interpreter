# strategy_context.py (FINAL FIXED VERSION)
from ..zexus_token import *
from ..zexus_ast import *
from ..config import config as zexus_config
from types import SimpleNamespace # Helper for AST node creation

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

# Helper function for parser debug output
def parser_debug(msg):
    if zexus_config.should_log('debug'):
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
            DEFER: self._parse_statement_block_context,
            ENUM: self._parse_statement_block_context,
            SANDBOX: self._parse_statement_block_context,
            'let_statement': self._parse_let_statement_block,
            'const_statement': self._parse_const_statement_block,
            'print_statement': self._parse_print_statement_block,
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

        # Check for type annotation (name: Type = value)
        type_annotation = None
        if len(tokens) > 2 and tokens[2].type == COLON:
            if len(tokens) > 3 and tokens[3].type == IDENT:
                type_annotation = tokens[3].literal
                parser_debug(f"  📝 Type annotation: {type_annotation}")
            else:
                parser_debug("  ❌ Invalid let statement: expected type after colon")
                return None

        # Find equals sign (skip type annotation if present)
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
                # Allow method chains but stop at other statement starters
                if t.type in {LET, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY, EXTERNAL, SCREEN, EXPORT, USE, DEBUG}:
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

    def _parse_print_statement_block(self, block_info, all_tokens):
        """Parse print statement block - RETURNS PrintStatement"""
        parser_debug("🔧 [Context] Parsing print statement")
        tokens = block_info['tokens']

        if len(tokens) < 2:
            return PrintStatement(StringLiteral(""))

        expression_tokens = tokens[1:]
        expression = self._parse_expression(expression_tokens)

        if expression is None:
            expression = StringLiteral("")

        return PrintStatement(expression)

    def _parse_assignment_statement(self, block_info, all_tokens):
        """Parse assignment statement - RETURNS AssignmentExpression"""
        parser_debug("🔧 [Context] Parsing assignment statement")
        tokens = block_info['tokens']

        if len(tokens) < 3 or tokens[1].type != ASSIGN:
            parser_debug("  ❌ Invalid assignment: no assignment operator")
            return None

        variable_name = tokens[0].literal
        # CRITICAL FIX: only collect RHS tokens up to statement boundary
        value_tokens = []
        stop_types = {SEMICOLON, RBRACE}
        statement_starters = {LET, CONST, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY, EXTERNAL, SCREEN, EXPORT, USE, DEBUG, AUDIT, RESTRICT, SANDBOX, TRAIL, NATIVE, GC, INLINE, BUFFER, SIMD, DEFER, PATTERN, ENUM, STREAM, WATCH, CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE, INTERFACE, TYPE_ALIAS, MODULE, PACKAGE, USING}
        j = 2
        while j < len(tokens):
            t = tokens[j]
            if t.type in stop_types or t.type in statement_starters:
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
            name=Identifier(variable_name),
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
        """Parse entity declaration block"""
        parser_debug("🔧 [Context] Parsing entity statement")
        tokens = block_info['tokens']

        if len(tokens) < 4:  # entity Name { ... }
            return None

        entity_name = tokens[1].literal if tokens[1].type == IDENT else "Unknown"
        parser_debug(f"  📝 Entity: {entity_name}")

        # Parse properties between braces
        properties = []
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
            # Parse properties inside braces
            i = brace_start + 1
            while i < brace_end:
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
            properties=properties
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

                        print(f"  ⚡ Found Contract Action: {act_name}")
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
            print(f"  ⚡ Injecting .name property for runtime compatibility: {contract_name}")
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
        print(f"    📝 Found use statement: {[t.literal for t in tokens]}")

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
                        print(f"    📝 Found import path: {file_path}")
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
        print(f"🔧 [Context] Parsing statement block: {subtype} (type: {type(subtype)})")

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
        elif subtype in {IF, FOR, WHILE, RETURN, DEFER, ENUM, SANDBOX}:
            # Use the existing logic in _parse_block_statements which handles these keywords
            print(f"🎯 [Context] Calling _parse_block_statements for subtype={subtype}")
            print(f"🎯 [Context] block_info['tokens'] has {len(block_info.get('tokens', []))} tokens")
            stmts = self._parse_block_statements(block_info['tokens'])
            print(f"🎯 [Context] Got {len(stmts) if stmts else 0} statements back")
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
        print("  🔧 [Try] Parsing try block")
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
            print(f"  🔧 [Try] Found {len(try_tokens)} tokens in try block: {[t.literal for t in try_tokens]}")
            try_block_statements = self._parse_block_statements(try_tokens)
            block = BlockStatement()
            block.statements = try_block_statements
            return block

        parser_debug("  ⚠️ [Try] Could not find try block content")
        return BlockStatement()

    def _parse_catch_block(self, tokens):
        """Parse the catch block from tokens"""
        print("  🔧 [Catch] Parsing catch block")
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
            print(f"  🔧 [Catch] Found {len(catch_tokens)} tokens in catch block: {[t.literal for t in catch_tokens]}")
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
        statement_starters = {LET, CONST, PRINT, FOR, IF, WHILE, RETURN, ACTION, FUNCTION, TRY, EXTERNAL, SCREEN, EXPORT, USE, DEBUG, ENTITY, CONTRACT, VERIFY, PROTECT, PERSISTENT, STORAGE, AUDIT, RESTRICT, SANDBOX, TRAIL, NATIVE, GC, INLINE, BUFFER, SIMD, DEFER, PATTERN, ENUM, STREAM, WATCH, CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE, INTERFACE, TYPE_ALIAS, MODULE, PACKAGE, USING, MIDDLEWARE, AUTH, THROTTLE, CACHE, REQUIRE}
        while i < len(tokens):
            token = tokens[i]

            # PRINT statement heuristic
            if token.type == PRINT:
                j = i + 1
                while j < len(tokens) and tokens[j].type not in [SEMICOLON, LBRACE, RBRACE]:
                    j += 1

                print_tokens = tokens[i:j]
                print(f"    📝 Found print statement: {[t.literal for t in print_tokens]}")

                if len(print_tokens) > 1:
                    # Fast-path: if the print contains exactly a single string literal
                    # (e.g. print "hello"; or print("hello");), treat it as a literal.
                    if len(print_tokens) == 2 and print_tokens[1].type == STRING:
                        statements.append(PrintStatement(StringLiteral(print_tokens[1].literal)))
                    else:
                        # Otherwise parse the full expression (handles concatenation and variables)
                        expr = self._parse_expression(print_tokens[1:])
                        if expr:
                            statements.append(PrintStatement(expr))
                        else:
                            statements.append(PrintStatement(StringLiteral("")))

                i = j

            # LET statement heuristic
            elif token.type == LET:
                j = i + 1
                nesting = 0
                while j < len(tokens):
                    t = tokens[j]
                    if t.type == LBRACE:
                        nesting += 1
                    elif t.type == RBRACE:
                        nesting -= 1
                        if nesting < 0:
                            break
                    
                    if nesting == 0 and t.type == SEMICOLON:
                        break
                    
                    j += 1

                let_tokens = tokens[i:j]
                print(f"    📝 Found let statement: {[t.literal for t in let_tokens]}")

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
                print(f"    📝 Found use statement (heuristic): {[t.literal for t in use_tokens]}")

                # Reuse the sophisticated parser
                block_info = {'tokens': use_tokens}
                stmt = self._parse_use_statement_block(block_info, tokens)
                if stmt:
                    statements.append(stmt)

                i = j
                continue

            # EXPORT statement heuristic
            elif token.type == EXPORT:
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
                print(f"    📝 Found export statement: {[t.literal for t in export_tokens]}")

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
                print(f"    📝 Found external statement: {[t.literal for t in external_tokens]}")

                # Parse using the main parser's parse_external_declaration
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

                print(f"    📝 Found action statement: {[t.literal for t in stmt_tokens]}")

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

                print(f"    📝 Found function statement: {[t.literal for t in stmt_tokens]}")

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
                
                print(f"    📝 Found module statement: {[t.literal for t in stmt_tokens]}")
                
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
                
                print(f"    📝 Found package statement: {[t.literal for t in stmt_tokens]}")
                
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
                
                print(f"    📝 Found watch statement: {[t.literal for t in stmt_tokens]}")
                
                block_info = {'tokens': stmt_tokens}
                stmt = self._parse_watch_statement(block_info, tokens)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue
            
            elif token.type == IF:
                # Parse IF statement directly here
                j = i + 1
                
                # Collect condition tokens (between IF and {)
                cond_tokens = []
                paren_depth = 0
                
                while j < len(tokens) and tokens[j].type != LBRACE:
                    # Handle outer parentheses for the condition
                    if tokens[j].type == LPAREN:
                        if len(cond_tokens) == 0 and paren_depth == 0:
                            # Skip the very first opening paren if it wraps the whole condition
                            j += 1
                            paren_depth += 1
                            continue
                        else:
                            paren_depth += 1
                    
                    elif tokens[j].type == RPAREN:
                        paren_depth -= 1
                        # If we hit a closing paren that matches the initial skipped one
                        if paren_depth == 0 and len(cond_tokens) > 0:
                            j += 1
                            break
                    
                    cond_tokens.append(tokens[j])
                    j += 1
                
                print(f"  [IF_COND] Condition tokens: {[t.literal for t in cond_tokens]}")
                
                # Parse condition expression
                condition = self._parse_expression(cond_tokens) if cond_tokens else Identifier("true")
                
                # Collect consequence block tokens (inside { })
                if j < len(tokens) and tokens[j].type == LBRACE:
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
                        while j < len(tokens) and tokens[j].type != LBRACE:
                            if tokens[j].type == LPAREN and len(elif_cond_tokens) == 0:
                                j += 1
                                continue
                            elif tokens[j].type == RPAREN and len(elif_cond_tokens) > 0:
                                j += 1
                                break
                            elif_cond_tokens.append(tokens[j])
                            j += 1
                        
                        elif_cond = self._parse_expression(elif_cond_tokens) if elif_cond_tokens else Identifier("true")
                        
                        # Collect elif block
                        if j < len(tokens) and tokens[j].type == LBRACE:
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
                        else:
                            elif_block = BlockStatement()
                        
                        elif_parts.append((elif_cond, elif_block))
                    
                    elif tokens[j].type == ELSE:
                        j += 1
                        # Collect else block
                        if j < len(tokens) and tokens[j].type == LBRACE:
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
                
                print(f"    📝 Found while statement with {len(body_block.statements)} body statements")
                
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
                        
                        print(f"    📝 Found for each statement with {len(body_block.statements)} body statements")
                        
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
                
                print(f"    📝 Found defer statement with {len(code_block.statements)} statements")
                
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
                
                print(f"    📝 Found enum '{enum_name}' with {len(members)} members")
                
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
                
                print(f"    📝 Found sandbox statement with {len(body.statements) if body else 0} statements")
                
                stmt = SandboxStatement(body=body, policy=None)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            elif token.type == REQUIRE:
                # Parse REQUIRE statement: require(condition, message) or require condition, message
                j = i + 1
                
                # Collect tokens until semicolon
                require_tokens = [token]
                while j < len(tokens) and tokens[j].type != SEMICOLON:
                    require_tokens.append(tokens[j])
                    j += 1
                
                print(f"    📝 Found require statement: {[t.literal for t in require_tokens]}")
                
                # Use the handler to parse it
                block_info = {'tokens': require_tokens}
                stmt = self._parse_require_statement(block_info, tokens)
                if stmt:
                    statements.append(stmt)
                
                i = j
                continue

            # Fallback: attempt to parse as expression
            else:
                j = i
                run_tokens = []
                nesting = 0
                while j < len(tokens):
                    t = tokens[j]
                    # update nesting for parentheses/brackets/braces
                    if t.type in {LPAREN, LBRACE, LBRACKET}:
                        nesting += 1
                    elif t.type in {RPAREN, RBRACE, RBRACKET}:
                        if nesting > 0:
                            nesting -= 1

                    # stop at top-level statement terminators or starters
                    if nesting == 0 and (t.type in [SEMICOLON, LBRACE, RBRACE] or t.type in statement_starters):
                        break

                    run_tokens.append(t)
                    j += 1

                if run_tokens:
                    expr = self._parse_expression(run_tokens)
                    if expr:
                        statements.append(ExpressionStatement(expr))
                # Advance to the token after the run (or by one to avoid infinite loop)
                if j == i:
                    i += 1
                else:
                    i = j

        print(f"    ✅ Parsed {len(statements)} statements from block")
        return statements

    # === MAP LITERAL PARSING ===

    def _parse_map_literal(self, tokens):
        """Parse a map literal { key: value, ... }"""
        parser_debug("  🗺️ [Map] Parsing map literal")

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
                    print(f"  🗺️ [Map] Added pair: {key_token.literal} -> {type(value_expr).__name__}")

                i = j
                if i < len(tokens) and tokens[i].type == COMMA:
                    i += 1
            else:
                # Skip token if it's unexpected (robust parsing)
                i += 1

        map_literal = MapLiteral(pairs_list)
        print(f"  🗺️ [Map] Successfully parsed map with {len(pairs_list)} pairs")
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
        """Parse print statement with sophisticated expression parsing and boundary detection"""
        parser_debug("🔧 [Context] Parsing print statement with enhanced expression boundary detection")
        tokens = block_info['tokens']

        # Need at least PRINT token + one value token
        if len(tokens) < 2:
            return PrintStatement(StringLiteral(""))

        # Collect tokens up to a statement boundary
        inner_tokens = []
        statement_terminators = {SEMICOLON, RBRACE}
        statement_starters = {LET, CONST, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY, AUDIT, RESTRICT, SANDBOX, TRAIL, NATIVE, GC, INLINE, BUFFER, SIMD, DEFER, PATTERN, ENUM, STREAM, WATCH, CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE, INTERFACE, TYPE_ALIAS, MODULE, PACKAGE, USING}
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
            return PrintStatement(StringLiteral(""))

        parser_debug(f"  📝 Print statement tokens: {[t.literal for t in inner_tokens]}")
        expression = self._parse_expression(inner_tokens)
        parser_debug(f"  ✅ Parsed print expression: {type(expression).__name__ if expression else 'None'}")
        return PrintStatement(expression if expression is not None else StringLiteral(""))

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
        # Handle unary prefix minus (e.g., -3, -x)
        # We do this before other checks so negative numbers are parsed as PrefixExpression
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
        if tokens[0].type == FUNCTION or tokens[0].type == ACTION:
            return self._parse_function_literal(tokens)
        if tokens[0].type == SANDBOX:
            return self._parse_sandbox_expression(tokens)
        if tokens[0].type == SANITIZE:
            return self._parse_sanitize_expression(tokens)

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

            elif t.type == IDENT or t.type in {SEND, RECEIVE}:  # Identifier or function call (including concurrency keywords)
                name = t.literal
                i += 1
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
                    return CallExpression(Identifier(name), args)
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
                if name_token.type != IDENT:
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
            if t.type in {PLUS, MINUS, ASTERISK, SLASH, 
                         LT, GT, EQ, NOT_EQ, LTE, GTE}:
                i += 1  # Skip operator
                right = self._parse_comparison_and_above(tokens[i:])
                if right:
                    current_expr = InfixExpression(
                        left=current_expr,
                        operator=t.literal,
                        right=right
                    )
                break  # Stop after handling one binary operator

            # No more chaining possible
            break

        return current_expr

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
            return Identifier(token.literal)
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
        print("  🔧 [List] Parsing list literal")
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
            elif t.type in {RBRACKET, RPAREN, RBRACE}:
                if nesting > 0:
                    nesting -= 1
                    cur.append(t)
                else:
                    # reached closing bracket of the list
                    if cur:
                        elem = self._parse_expression(cur)
                        elements.append(elem)
                    break
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
        print(f"🔧 [Context] Parsing screen: {block_info.get('name', 'anonymous')}")
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
        print(f"🔧 [Context] Parsing function: {block_info.get('name', 'anonymous')}")
        params = self._extract_function_parameters(block_info, all_tokens)
        return ActionStatement(
            name=Identifier(block_info.get('name', 'anonymous')),
            parameters=params,
            body=BlockStatement()
        )

    def _parse_action_statement_context(self, block_info, all_tokens):
        """Parse action statement: action name(params) { body }"""
        tokens = block_info.get('tokens', [])
        if not tokens or tokens[0].type != ACTION:
            return None
        
        # Extract action name (next IDENT after ACTION)
        name = None
        params = []
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
            body=body
        )

    def _parse_function_statement_context(self, block_info, all_tokens):
        """Parse function statement: function name(params) { body }"""
        tokens = block_info.get('tokens', [])
        if not tokens or tokens[0].type != FUNCTION:
            return None
        
        # Extract function name (next IDENT after FUNCTION)
        name = None
        params = []
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
            body=body
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
        
        # Check if this is an assignment statement (identifier = value)
        if len(tokens) >= 3 and tokens[0].type == IDENT and tokens[1].type == ASSIGN:
            parser_debug(f"  🎯 [Generic] Detected assignment statement")
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
        print(f"  📦 Module: {module_name}")
        
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
                    capacity = int(tokens[i+1].literal)
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
        """Parse require(condition, message) statements."""
        parser_debug("🔧 [Context] Parsing require statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != REQUIRE or (len(tokens) > 1 and tokens[1].type != LPAREN):
            parser_debug("  ❌ Expected require()")
            return None
        
        # Extract tokens between LPAREN and RPAREN
        inner = tokens[2:-1] if len(tokens) > 2 and tokens[-1].type == RPAREN else tokens[2:]
        
        # Split by comma to get condition and optional message
        args = self._parse_argument_list(inner)
        
        if not args:
            parser_debug("  ❌ require needs at least one argument")
            return None
        
        condition = args[0]
        message = args[1] if len(args) > 1 else None
        
        parser_debug(f"  ✅ Require with {len(args)} arguments")
        return RequireStatement(condition=condition, message=message)
    
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
        # Find '=>' (ASSIGN in this context usually, or maybe we need to check for it)
        # The lexer might not produce a specific ARROW token, usually it's ASSIGN or similar.
        # But wait, the parser uses ASSIGN for => in parse_watch_statement.
        
        arrow_idx = -1
        for i, t in enumerate(tokens):
            if t.literal == '=>': # Look for => specifically
                arrow_idx = i
                break
            elif t.type == ASSIGN: # Fallback to ASSIGN token
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
        
        Form: protect <target> { <rules> }
        Example: protect Profile.update_email { verify(...) restrict(...) }
        """
        parser_debug("🔧 [Context] Parsing protect statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != PROTECT:
            parser_debug("  ❌ Expected PROTECT keyword")
            return None
        
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
        
        parser_debug("  ✅ Protect statement")
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
        """Parse verify statement.
        
        Forms:
        1. verify condition, "message"
        2. verify (condition)
        3. verify(target, [conditions])
        
        Examples:
        - verify false, "Access denied"
        - verify (TX.caller == self.owner)
        - verify(transfer_funds, [check_auth()])
        """
        parser_debug("🔧 [Context] Parsing verify statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != VERIFY:
            parser_debug("  ❌ Expected VERIFY keyword")
            return None
        
        # Check for comma-separated format: verify condition, message
        comma_idx = None
        paren_depth = 0
        for i, tok in enumerate(tokens[1:], 1):
            if tok.type in {LPAREN, LBRACKET}:
                paren_depth += 1
            elif tok.type in {RPAREN, RBRACKET}:
                paren_depth -= 1
            elif tok.type == COMMA and paren_depth == 0:
                comma_idx = i
                break
        
        if comma_idx:
            # Format: verify condition, message
            cond_tokens = tokens[1:comma_idx]
            msg_tokens = tokens[comma_idx+1:]
            
            condition = self._parse_expression(cond_tokens) if cond_tokens else None
            message = self._parse_expression(msg_tokens) if msg_tokens else None
            
            parser_debug(f"  ✅ Verify statement (simple assertion) with message")
            return VerifyStatement(condition=condition, message=message)
        else:
            # Format: verify (condition) or verify(target, [...])
            inner = tokens[2:-1] if len(tokens) > 2 and tokens[-1].type == RPAREN else tokens[1:]
            
            condition = self._parse_expression(inner) if inner else None
            
            if condition is None:
                parser_debug("  ❌ verify needs a condition")
                return None
            
            parser_debug("  ✅ Verify statement (parenthesized)")
            return VerifyStatement(condition=condition)

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