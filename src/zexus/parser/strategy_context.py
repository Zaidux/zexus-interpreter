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
            'let_statement': self._parse_let_statement_block,
            'const_statement': self._parse_const_statement_block,
            'print_statement': self._parse_print_statement_block,
            'assignment_statement': self._parse_assignment_statement,
            'function_call_statement': self._parse_function_call_statement,
            'entity_statement': self._parse_entity_statement_block,
            'USE': self._parse_use_statement_block,
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
                    print(f"  ✅ Parsed: {type(result).__name__} at line {block_info.get('start_token', {}).get('line', 'unknown')}")
                    # If we got a BlockStatement but it has no inner statements,
                    # attempt to populate it from the block tokens (best-effort).
                    if isinstance(result, BlockStatement) and not getattr(result, 'statements', None):
                        tokens = block_info.get('tokens', [])
                        if tokens:
                            print(f"  🔧 Populating empty BlockStatement from {len(tokens)} tokens")
                            parsed_stmts = self._parse_block_statements(tokens)
                            result.statements = parsed_stmts
                            print(f"  ✅ Populated BlockStatement with {len(parsed_stmts)} statements")
                    return result
                elif isinstance(result, Expression):
                    print(f"  ✅ Parsed: ExpressionStatement at line {block_info.get('start_token', {}).get('line', 'unknown')}")
                    return ExpressionStatement(result)
                else:
                    result = self._ensure_statement_node(result, block_info)
                    if result:
                        print(f"  ✅ Parsed: {type(result).__name__} at line {block_info.get('start_token', {}).get('line', 'unknown')}")
                    return result
            else:
                print(f"  ⚠️ No result for {block_type} at line {block_info.get('start_token', {}).get('line', 'unknown')}")
                return None

        except Exception as e:
            print(f"⚠️ [Context] Error parsing {block_type}: {e}")
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
        print("🔧 [Context] Parsing let statement")
        tokens = block_info['tokens']

        if len(tokens) < 4:
            print("  ❌ Invalid let statement: too few tokens")
            return None

        if tokens[1].type != IDENT:
            print("  ❌ Invalid let statement: expected identifier after 'let'")
            return None

        variable_name = tokens[1].literal
        print(f"  📝 Variable: {variable_name}")

        equals_index = -1
        for i, token in enumerate(tokens):
            if token.type == ASSIGN:
                equals_index = i
                break

        if equals_index == -1:
            print("  ❌ Invalid let statement: no assignment operator")
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

        print(f"  📝 Value tokens: {[t.literal for t in value_tokens]}")

        # Parse the value expression
        if not value_tokens:
            print("  ❌ No value tokens found")
            return None

        # Special case: map literal
        if value_tokens[0].type == LBRACE:
            print("  🗺️ Detected map literal")
            value_expression = self._parse_map_literal(value_tokens)
        else:
            value_expression = self._parse_expression(value_tokens)

        if value_expression is None:
            print("  ❌ Could not parse value expression")
            return None

        print(f"  ✅ Let statement: {variable_name} = {type(value_expression).__name__}")
        return LetStatement(
            name=Identifier(variable_name),
            value=value_expression
        )

    def _parse_const_statement_block(self, block_info, all_tokens):
        """Parse const statement block with robust method chain handling (mirrors let)"""
        print("🔧 [Context] Parsing const statement")
        tokens = block_info['tokens']

        if len(tokens) < 4:
            print("  ❌ Invalid const statement: too few tokens")
            return None

        if tokens[1].type != IDENT:
            print("  ❌ Invalid const statement: expected identifier after 'const'")
            return None

        variable_name = tokens[1].literal
        print(f"  📝 Variable: {variable_name}")

        equals_index = -1
        for i, token in enumerate(tokens):
            if token.type == ASSIGN:
                equals_index = i
                break

        if equals_index == -1:
            print("  ❌ Invalid const statement: no assignment operator")
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

        print(f"  📝 Value tokens: {[t.literal for t in value_tokens]}")

        # Parse the value expression
        if not value_tokens:
            print("  ❌ No value tokens found")
            return None

        # Special case: map literal
        if value_tokens[0].type == LBRACE:
            print("  🗺️ Detected map literal")
            value_expression = self._parse_map_literal(value_tokens)
        else:
            value_expression = self._parse_expression(value_tokens)

        if value_expression is None:
            print("  ❌ Could not parse value expression")
            return None

        print(f"  ✅ Const statement: {variable_name} = {type(value_expression).__name__}")
        return ConstStatement(
            name=Identifier(variable_name),
            value=value_expression
        )

    def _parse_print_statement_block(self, block_info, all_tokens):
        """Parse print statement block - RETURNS PrintStatement"""
        print("🔧 [Context] Parsing print statement")
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
        print("🔧 [Context] Parsing assignment statement")
        tokens = block_info['tokens']

        if len(tokens) < 3 or tokens[1].type != ASSIGN:
            print("  ❌ Invalid assignment: no assignment operator")
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
            print("  🗺️ Detected map literal in assignment")
            value_expression = self._parse_map_literal(value_tokens)
        else:
            value_expression = self._parse_expression(value_tokens)

        if value_expression is None:
            print("  ❌ Could not parse assignment value")
            return None

        return AssignmentExpression(
            name=Identifier(variable_name),
            value=value_expression
        )

    def _parse_function_call_statement(self, block_info, all_tokens):
        """Parse function call as a statement - RETURNS ExpressionStatement"""
        print("🔧 [Context] Parsing function call statement")
        tokens = block_info['tokens']

        if len(tokens) < 3 or tokens[1].type != LPAREN:
            print("  ❌ Invalid function call: no parentheses")
            return None

        function_name = tokens[0].literal
        inner_tokens = tokens[2:-1] if tokens and tokens[-1].type == RPAREN else tokens[2:]
        arguments = self._parse_argument_list(inner_tokens)

        call_expression = CallExpression(Identifier(function_name), arguments)
        return ExpressionStatement(call_expression)

    def _parse_entity_statement_block(self, block_info, all_tokens):
        """Parse entity declaration block"""
        print("🔧 [Context] Parsing entity statement")
        tokens = block_info['tokens']

        if len(tokens) < 4:  # entity Name { ... }
            return None

        entity_name = tokens[1].literal if tokens[1].type == IDENT else "Unknown"
        print(f"  📝 Entity: {entity_name}")

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
                    print(f"  📝 Found property name: {prop_name}")

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
                            print(f"  📝 Property: {prop_name}: {prop_type}")
                            i += 3
                            continue

                i += 1

        return EntityStatement(
            name=Identifier(entity_name),
            properties=properties
        )

    def _parse_contract_statement_block(self, block_info, all_tokens):
        """Parse contract declaration block - FINAL FIXED VERSION"""
        print("🔧 [Context] Parsing contract statement")
        tokens = block_info['tokens']

        if len(tokens) < 3:
            return None

        # 1. Extract Name
        contract_name = tokens[1].literal if tokens[1].type == IDENT else "UnknownContract"
        print(f"  📝 Contract Name: {contract_name}")

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

        # 5. Return ContractStatement with strict positional arguments
        return ContractStatement(
            name=Identifier(contract_name),
            storage_vars=storage_vars,
            actions=actions
        )

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
        print(f"🔧 [Context] Parsing statement block: {block_info.get('subtype', 'unknown')}")

        subtype = block_info.get('subtype', 'unknown')

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
        elif subtype == 'use_statement': # Fix subtype mismatch
            return self._parse_use_statement_block(block_info, all_tokens)
        elif subtype == 'USE':
            return self._parse_use_statement_block(block_info, all_tokens)
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
        print("🔧 [Context] Parsing try-catch statement block")

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

        print("  ⚠️ [Try] Could not find try block content")
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

        print("  ⚠️ [Catch] Could not find catch block content")
        return BlockStatement()

    def _parse_block_statements(self, tokens):
        """Parse statements from a block of tokens"""
        if not tokens:
            return []

        statements = []
        i = 0
        # Common statement-starter tokens used by several heuristics and fallbacks
        # Common statement-starter tokens used by several heuristics and fallbacks
        statement_starters = {LET, CONST, PRINT,     FOR, IF, WHILE, RETURN, ACTION, TRY,   EXTERNAL, SCREEN, EXPORT, USE, DEBUG,   ENTITY, CONTRACT, VERIFY, PROTECT, PERSISTENT, STORAGE, AUDIT, RESTRICT, SANDBOX, TRAIL, NATIVE, GC, INLINE, BUFFER, SIMD, DEFER, PATTERN, ENUM, STREAM, WATCH, CAPABILITY, GRANT, REVOKE, VALIDATE, SANITIZE, IMMUTABLE, INTERFACE, TYPE_ALIAS, MODULE, PACKAGE, USING}
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
                while j < len(tokens) and tokens[j].type not in [SEMICOLON, LBRACE, RBRACE]:
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
            
            elif token.type == IF:
                # Parse IF statement directly here
                j = i + 1
                
                # Collect condition tokens (between IF and {)
                cond_tokens = []
                while j < len(tokens) and tokens[j].type != LBRACE:
                    if tokens[j].type == LPAREN and len(cond_tokens) == 0:
                        j += 1
                        continue
                    elif tokens[j].type == RPAREN and len(cond_tokens) > 0:
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

            elif token.type == RETURN:
                # Parse RETURN statement directly
                j = i + 1
                value_tokens = []
                
                # Collect tokens until semicolon, closing brace, or next statement
                while j < len(tokens) and tokens[j].type not in [SEMICOLON, RBRACE]:
                    if tokens[j].type in statement_starters and tokens[j].type != RETURN:
                        break
                    value_tokens.append(tokens[j])
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
        print("  🗺️ [Map] Parsing map literal")

        if not tokens or tokens[0].type != LBRACE:
            print("  ❌ [Map] Not a map literal - no opening brace")
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
                while j < len(tokens) and tokens[j].type not in [COMMA, RBRACE]:
                    value_tokens.append(tokens[j])
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
        print("🔧 [Context] Parsing parentheses block")
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
        print("🔧 [Context] Parsing print statement with enhanced expression boundary detection")
        tokens = block_info['tokens']

        if len(tokens) < 3:
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

        print(f"  📝 Print statement tokens: {[t.literal for t in inner_tokens]}")
        expression = self._parse_expression(inner_tokens)
        print(f"  ✅ Parsed print expression: {type(expression).__name__ if expression else 'None'}")
        return PrintStatement(expression if expression is not None else StringLiteral(""))

    def _parse_return_statement(self, block_info, all_tokens):
        """Parse return statement"""
        print("🔧 [Context] Parsing return statement")
        tokens = block_info.get('tokens', [])
        
        if not tokens or tokens[0].type != RETURN:
            return None
        
        # If only return token, return None
        if len(tokens) <= 1:
            return ReturnStatement(Identifier("null"))
        
        # Parse the return value expression
        value_tokens = tokens[1:]
        print(f"  📝 Return value tokens: {[t.literal for t in value_tokens]}")
        
        value_expr = self._parse_expression(value_tokens)
        return ReturnStatement(value_expr if value_expr else Identifier("null"))

    def _parse_expression(self, tokens):
        """Parse a full expression with operator precedence handling"""
        if not tokens or len(tokens) == 0:
            return StringLiteral("")
        
        # Handle logical OR (lowest precedence)
        or_index = -1
        paren_depth = 0
        for idx, t in enumerate(tokens):
            if t.type == LPAREN:
                paren_depth += 1
            elif t.type == RPAREN:
                paren_depth -= 1
            elif t.type == OR and paren_depth == 0:
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
        paren_depth = 0
        for idx, t in enumerate(tokens):
            if t.type == LPAREN:
                paren_depth += 1
            elif t.type == RPAREN:
                paren_depth -= 1
            elif t.type == AND and paren_depth == 0:
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
        if tokens[0].type == LBRACE:
            return self._parse_map_literal(tokens)
        if tokens[0].type == LBRACKET:
            return self._parse_list_literal(tokens)
        if tokens[0].type == LAMBDA:
            return self._parse_lambda(tokens)
        if tokens[0].type == FUNCTION:
            return self._parse_function_literal(tokens)

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

            elif t.type == IDENT:  # Identifier or function call
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
            print("  ❌ [List] Not a list literal")
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

        # If there is a trailing element
        if cur:
            elem = self._parse_expression(cur)
            elements.append(elem)

        print(f"  ✅ Parsed list with {len(elements)} elements")
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
        """Parse a function literal expression (anonymous function)
        
        Supports forms:
          function(x) { return x * 2; }
          function(x, y) { return x + y; }
          function() { return 42; }
        """
        print("  🔧 [Function Literal] Parsing function literal")
        if not tokens or tokens[0].type != FUNCTION:
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

    def _parse_argument_list(self, tokens):
        """Parse comma-separated argument list with improved nesting support"""
        print("  🔍 Parsing argument list")
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
                    print(f"  📝 Parsed argument: {type(arg_expr).__name__ if arg_expr else 'None'}")
                    arguments.append(arg_expr)
                    current_arg = []
            else:
                current_arg.append(token)

        # Handle last argument
        if current_arg:
            arg_expr = self._parse_expression(current_arg)
            print(f"  📝 Parsed final argument: {type(arg_expr).__name__ if arg_expr else 'None'}")
            arguments.append(arg_expr)

        # Filter out None arguments by replacing with empty string literal
        arguments = [arg if arg is not None else StringLiteral("") for arg in arguments]
        print(f"  ✅ Parsed {len(arguments)} arguments total")
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
        print("🔧 [Context] Parsing loop block")
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
        print("🔧 [Context] Parsing try-catch block with context awareness")
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
        print("🔧 [Context] Parsing conditional block")
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
        print("🔧 [Context] Parsing brace block")
        return BlockStatement()

    def _parse_generic_block(self, block_info, all_tokens):
        """Fallback parser for unknown block types - intelligently detects statement type"""
        tokens = block_info.get('tokens', [])
        if not tokens:
            return BlockStatement()
        
        # Debug: log what we're trying to parse
        print(f"  🔍 [Generic] Parsing generic block with tokens: {[t.literal for t in tokens]}")
        
        # Check if this is a LET statement
        if tokens[0].type == LET:
            print(f"  🎯 [Generic] Detected let statement")
            return self._parse_let_statement_block(block_info, all_tokens)
        
        # Check if this is a CONST statement
        if tokens[0].type == CONST:
            print(f"  🎯 [Generic] Detected const statement")
            return self._parse_const_statement_block(block_info, all_tokens)
        
        # Check if this is an assignment statement (identifier = value)
        if len(tokens) >= 3 and tokens[0].type == IDENT and tokens[1].type == ASSIGN:
            print(f"  🎯 [Generic] Detected assignment statement")
            return self._parse_assignment_statement(block_info, all_tokens)
        
        # Check if this is a print statement
        if tokens[0].type == PRINT:
            print(f"  🎯 [Generic] Detected print statement")
            return self._parse_print_statement(block_info, all_tokens)
        
        # Check if this is a return statement
        if tokens[0].type == RETURN:
            print(f"  🎯 [Generic] Detected return statement")
            return self._parse_return_statement(block_info, all_tokens)
        
        # Check if it's a function call (identifier followed by parentheses)
        if tokens[0].type == IDENT and len(tokens) >= 2 and tokens[1].type == LPAREN:
            print(f"  🎯 [Generic] Detected function call")
            # Parse as expression and wrap in ExpressionStatement
            expr = self._parse_expression(tokens)
            if expr:
                return ExpressionStatement(expr)
        
        # Try to parse as a simple expression
        print(f"  🎯 [Generic] Attempting to parse as expression")
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
        print("🔧 [Context] Parsing capability statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            print("  ❌ Invalid capability statement: expected name")
            return None
        
        if tokens[0].type != CAPABILITY:
            print("  ❌ Expected CAPABILITY keyword")
            return None
        
        if tokens[1].type != IDENT:
            print("  ❌ Expected capability name")
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
        print("🔧 [Context] Parsing grant statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            print("  ❌ Invalid grant statement")
            return None
        
        if tokens[0].type != GRANT:
            print("  ❌ Expected GRANT keyword")
            return None
        
        if tokens[1].type != IDENT:
            print("  ❌ Expected entity name after grant")
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
        print("🔧 [Context] Parsing revoke statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            print("  ❌ Invalid revoke statement")
            return None
        
        if tokens[0].type != REVOKE:
            print("  ❌ Expected REVOKE keyword")
            return None
        
        if tokens[1].type != IDENT:
            print("  ❌ Expected entity name after revoke")
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
        print("🔧 [Context] Parsing validate statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            print("  ❌ Invalid validate statement")
            return None
        
        # Parse: validate <expr>, <schema>
        comma_idx = -1
        for i, t in enumerate(tokens):
            if t.type == COMMA:
                comma_idx = i
                break
        
        if comma_idx == -1:
            print("  ⚠️ No schema provided for validate")
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
        print("🔧 [Context] Parsing sanitize statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            print("  ❌ Invalid sanitize statement")
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
        print("🔧 [Context] Parsing immutable statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2:
            print("  ❌ Invalid immutable statement")
            return None
        
        if tokens[0].type != IMMUTABLE:
            print("  ❌ Expected IMMUTABLE keyword")
            return None
        
        # Check if next is LET, CONST, or IDENT
        if tokens[1].type in {LET, CONST}:
            # immutable let/const name = value
            if len(tokens) < 4 or tokens[2].type != IDENT:
                print("  ❌ Invalid immutable declaration")
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
            print("  ❌ Expected LET, CONST, or identifier after IMMUTABLE")
            return None
    # === COMPLEXITY & LARGE PROJECT MANAGEMENT HANDLERS ===

    def _parse_interface_statement(self, block_info, all_tokens):
        """Parse interface definition statement
        
        interface Drawable {
            draw(canvas);
            get_bounds();
        };
        """
        print("🔧 [Context] Parsing interface statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2 or tokens[0].type != INTERFACE:
            print("  ❌ Expected INTERFACE keyword")
            return None
        
        if tokens[1].type != IDENT:
            print("  ❌ Expected interface name")
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
        print("🔧 [Context] Parsing type_alias statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 4 or tokens[0].type != TYPE_ALIAS:
            print("  ❌ Invalid type_alias statement")
            return None
        
        if tokens[1].type != IDENT:
            print("  ❌ Expected type name")
            return None
        
        type_name = tokens[1].literal
        
        if tokens[2].type != ASSIGN:
            print("  ❌ Expected '=' in type_alias")
            return None
        
        # Parse the base type
        base_type_tokens = tokens[3:]
        base_type = self._parse_expression(base_type_tokens)
        
        print(f"  📝 Type alias: {type_name}")
        
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
        print("🔧 [Context] Parsing module statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2 or tokens[0].type != MODULE:
            print("  ❌ Expected MODULE keyword")
            return None
        
        if tokens[1].type != IDENT:
            print("  ❌ Expected module name")
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
        print("🔧 [Context] Parsing package statement")
        tokens = block_info.get('tokens', [])
        
        if len(tokens) < 2 or tokens[0].type != PACKAGE:
            print("  ❌ Expected PACKAGE keyword")
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
            print("  ❌ Expected USING keyword")
            return None
        
        # Parse: using(name = expr) { body }
        if tokens[1].type != LPAREN:
            print("  ❌ Expected '(' after using")
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
            print("  ❌ Invalid using statement syntax")
            return None
        
        # Extract resource name
        if tokens[2].type != IDENT:
            print("  ❌ Expected resource name")
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
        print("🔧 [Context] Parsing channel statement")
        tokens = block_info.get('tokens', [])

        if not tokens or tokens[0].type != CHANNEL:
            print("  ❌ Expected CHANNEL keyword")
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
                print("  ❌ Unterminated generic type for channel")
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
                    print("  ❌ Expected ']' after channel capacity")
                    return None

        # Name
        if i < len(tokens) and tokens[i].type == IDENT:
            name = Identifier(tokens[i].literal)
            i += 1
        else:
            print("  ❌ Expected channel name")
            return None

        print(f"  ✅ Channel: {name.value}, type={type(element_type).__name__}, capacity={capacity}")
        return ChannelStatement(name=name, element_type=element_type, capacity=capacity)

    def _parse_send_statement(self, block_info, all_tokens):
        """Parse send(channel, value) statements."""
        print("🔧 [Context] Parsing send statement")
        tokens = block_info.get('tokens', [])
        if len(tokens) < 3 or tokens[0].type != SEND or tokens[1].type != LPAREN:
            print("  ❌ Invalid send statement")
            return None

        inner = tokens[2:-1] if tokens and tokens[-1].type == RPAREN else tokens[2:]
        args = self._parse_argument_list(inner)
        if not args or len(args) < 2:
            print("  ❌ send requires (channel, value)")
            return None

        channel_expr = args[0]
        value_expr = args[1]
        return SendStatement(channel_expr=channel_expr, value_expr=value_expr)

    def _parse_receive_statement(self, block_info, all_tokens):
        """Parse receive(channel) statements. Assignment handled elsewhere."""
        print("🔧 [Context] Parsing receive statement")
        tokens = block_info.get('tokens', [])
        if len(tokens) < 3 or tokens[0].type != RECEIVE or tokens[1].type != LPAREN:
            print("  ❌ Invalid receive statement")
            return None

        inner = tokens[2:-1] if tokens and tokens[-1].type == RPAREN else tokens[2:]
        # parse single channel expression
        channel_expr = self._parse_expression(inner)
        if channel_expr is None:
            print("  ❌ Could not parse channel expression for receive")
            return None

        return ReceiveStatement(channel_expr=channel_expr, target=None)

    def _parse_atomic_statement(self, block_info, all_tokens):
        """Parse atomic blocks or single-expression atomics."""
        print("🔧 [Context] Parsing atomic statement")
        tokens = block_info.get('tokens', [])
        if not tokens or tokens[0].type != ATOMIC:
            print("  ❌ Expected ATOMIC keyword")
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

        print("  ❌ Empty atomic statement")
        return None
