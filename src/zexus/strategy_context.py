# strategy_context.py (PRODUCTION-READY VERSION)
from .zexus_token import *
from .zexus_ast import *
from .config import config as zexus_config

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

class ContextStackParser:
    def __init__(self, structural_analyzer):
        self.structural_analyzer = structural_analyzer
        self.current_context = ['global']
        self.context_rules = {
            'function': self._parse_function_context,
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
            'print_statement': self._parse_print_statement_block,
            'assignment_statement': self._parse_assignment_statement,
            'function_call_statement': self._parse_function_call_statement,
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
        statement_starters = {LET, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY, EXTERNAL, SCREEN, EXPORT, USE, DEBUG}
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

    def _parse_statement_block_context(self, block_info, all_tokens):
        """Parse standalone statement blocks - use direct parsers where available"""
        print(f"🔧 [Context] Parsing statement block: {block_info.get('subtype', 'unknown')}")

        subtype = block_info.get('subtype', 'unknown')

        # Use the direct parser methods
        if subtype == 'let_statement':
            return self._parse_let_statement_block(block_info, all_tokens)
        elif subtype == 'print_statement':
            return self._parse_print_statement_block(block_info, all_tokens)
        elif subtype == 'function_call_statement':
            return self._parse_function_call_statement(block_info, all_tokens)
        elif subtype == 'assignment_statement':
            return self._parse_assignment_statement(block_info, all_tokens)
        elif subtype == 'try_catch_statement':
            return self._parse_try_catch_statement(block_info, all_tokens)
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

        print(f"    📝 Parsing {len(tokens)} tokens into statements")

        statements = []
        i = 0
        # Common statement-starter tokens used by several heuristics and fallbacks
        statement_starters = {LET, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY, EXTERNAL, SCREEN, EXPORT, USE, DEBUG}
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
                    string_literal = None
                    for t in print_tokens:
                        if t.type == STRING:
                            string_literal = StringLiteral(t.literal)
                            break

                    if string_literal:
                        statements.append(PrintStatement(string_literal))
                    else:
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

            # USE statement heuristic
            elif token.type == USE:
                j = i + 1
                while j < len(tokens) and tokens[j].type not in [SEMICOLON, LBRACE, RBRACE]:
                    j += 1

                use_tokens = tokens[i:j]
                print(f"    📝 Found use statement: {[t.literal for t in use_tokens]}")

                # Expect a string literal as module path
                module_path = None
                alias = None
                if len(use_tokens) >= 2 and use_tokens[1].type == STRING:
                    module_path = StringLiteral(use_tokens[1].literal)

                # Look for 'as' alias pattern
                for k in range(2, len(use_tokens)):
                    if use_tokens[k].type == IDENT and use_tokens[k].literal == 'as' and k + 1 < len(use_tokens):
                        if use_tokens[k+1].type == IDENT:
                            alias = Identifier(use_tokens[k+1].literal)
                        break

                statements.append(UseStatement(module_path, alias))
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
                    if tk.type == IDENT and tk.literal not in ('to', 'with'):
                        names.append(Identifier(tk.literal))
                    k += 1

                statements.append(ExportStatement(names=names))
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
            else:
                # Fallback: collect a run of tokens until a statement boundary
                # and attempt to parse them as a single expression. This reduces
                # token fragmentation caused by the structural analyzer splitting
                # long expressions into many tiny blocks.
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
        statement_starters = {LET, PRINT, FOR, IF, WHILE, RETURN, ACTION, TRY}
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

    def _parse_expression(self, tokens):
        """Parse a full expression with robust method chaining and property access"""
        if not tokens:
            return StringLiteral("")

        print(f"  🔍 Parsing expression from tokens: {[t.literal for t in tokens]}")

        # Special cases first
        if tokens[0].type == LBRACE:
            return self._parse_map_literal(tokens)
        if tokens[0].type == LBRACKET:
            return self._parse_list_literal(tokens)
        if tokens[0].type == LAMBDA:
            return self._parse_lambda(tokens)

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

            # Binary operators
            if t.type in {PLUS, MINUS, ASTERISK, SLASH, 
                         LT, GT, EQ, NOT_EQ, LTE, GTE}:
                i += 1  # Skip operator
                right = self._parse_expression(tokens[i:])
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

        # Handle string concatenation or infix operators
        for i, token in enumerate(tokens):
            if token.type in {PLUS, MINUS, ASTERISK, SLASH, 
                            LT, GT, EQ, NOT_EQ, LTE, GTE, LAMBDA}:
                left_tokens = tokens[:i]
                right_tokens = tokens[i + 1:]
                # Arrow-style lambda: treat LAMBDA as a lambda operator where
                # left side are parameters and right side is the body.
                if token.type == LAMBDA:
                    # Parse left as parameters (ident or parenthesized list)
                    params = []
                    if left_tokens:
                        # single identifier
                        if len(left_tokens) == 1 and left_tokens[0].type == IDENT:
                            params = [Identifier(left_tokens[0].literal)]
                        else:
                            # try to parse as a list of idents (e.g. (a,b))
                            first = left_tokens[0]
                            if first.type == LBRACKET or first.type == LPAREN:
                                inner = left_tokens[1:-1]
                                # split on commas
                                cur = []
                                ids = []
                                for t in inner:
                                    if t.type == COMMA:
                                        if cur and cur[0].type == IDENT:
                                            ids.append(Identifier(cur[0].literal))
                                        cur = []
                                    else:
                                        cur.append(t)
                                if cur and cur[0].type == IDENT:
                                    ids.append(Identifier(cur[0].literal))
                                params = ids

                    body_expr = self._parse_expression(right_tokens)
                    return LambdaExpression(parameters=params, body=body_expr)

                left_expr = self._parse_expression(left_tokens)
                right_expr = self._parse_expression(right_tokens)
                return InfixExpression(left_expr, token.literal, right_expr)

        # Single token expressions
        if len(tokens) == 1:
            return self._parse_single_token_expression(tokens[0])

        # Compound expressions (best-effort)
        expr = self._parse_compound_expression(tokens)
        print(f"  ✅ Parsed expression result: {type(expr).__name__ if expr else 'None'}")
        return expr

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
            return Boolean(True)
        elif token.type == FALSE:
            return Boolean(False)
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

    def _parse_conditional_context(self, block_info, all_tokens):
        """Parse if/else blocks with context awareness"""
        print("🔧 [Context] Parsing conditional block")
        condition = self._extract_condition(block_info, all_tokens)
        return IfStatement(
            condition=condition,
            consequence=BlockStatement(),
            alternative=None
        )

    def _parse_brace_block_context(self, block_info, all_tokens):
        """Parse generic brace block with context awareness"""
        print("🔧 [Context] Parsing brace block")
        return BlockStatement()

    def _parse_generic_block(self, block_info, all_tokens):
        """Fallback parser for unknown block types"""
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