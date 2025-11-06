# strategy_context.py (COMPLETE FIXED VERSION)
from .zexus_token import *
from .zexus_ast import *

class ContextStackParser:
    def __init__(self, structural_analyzer):
        self.structural_analyzer = structural_analyzer
        self.current_context = ['global']
        self.context_rules = {
            'function': self._parse_function_context,
            'try_catch': self._parse_try_catch_context,
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
        print(f"📥 [Context] Pushed: {context_str}")

    def pop_context(self):
        """Pop the current context from the stack"""
        if len(self.current_context) > 1:
            popped = self.current_context.pop()
            print(f"📤 [Context] Popped: {popped}")
            return popped
        return None

    def get_current_context(self):
        """Get the current parsing context"""
        return self.current_context[-1] if self.current_context else 'global'

    def parse_block(self, block_info, all_tokens):
        """Parse a block with context awareness - FIXED"""
        block_type = block_info.get('subtype', block_info['type'])
        context_name = block_info.get('name', 'anonymous')

        self.push_context(block_type, context_name)

        try:
            # Use appropriate parsing strategy for this context
            if block_type in self.context_rules:
                result = self.context_rules[block_type](block_info, all_tokens)
            else:
                result = self._parse_generic_block(block_info, all_tokens)

            # CRITICAL FIX: Don't wrap Statement nodes, only wrap Expressions
            if result is not None:
                # If it's already a Statement, return it as-is
                if isinstance(result, Statement):
                    print(f"  ✅ Parsed: {type(result).__name__} at line {block_info['start_token'].line}")
                    return result
                # If it's an Expression, wrap it in ExpressionStatement
                elif isinstance(result, Expression):
                    print(f"  ✅ Parsed: ExpressionStatement at line {block_info['start_token'].line}")
                    return ExpressionStatement(result)
                # If it's something else, try to ensure it's a statement
                else:
                    result = self._ensure_statement_node(result, block_info)
                    if result:
                        print(f"  ✅ Parsed: {type(result).__name__} at line {block_info['start_token'].line}")
                    return result
            else:
                print(f"  ⚠️ No result for {block_type} at line {block_info['start_token'].line}")
                return None

        except Exception as e:
            print(f"⚠️ [Context] Error parsing {block_type}: {e}")
            return None
        finally:
            self.pop_context()

    def _ensure_statement_node(self, node, block_info):
        """Ensure the node is a proper Statement - FIXED"""
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
        """Parse let statement block - FIXED to handle map literals"""
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

        value_tokens = tokens[equals_index + 1:]
        print(f"  📝 Value tokens: {[t.literal for t in value_tokens]}")

        # CRITICAL FIX: Check if this is a map literal and parse it properly
        if value_tokens and value_tokens[0].type == LBRACE:
            print("  🗺️  Parsing as map literal...")
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

    def _parse_map_literal(self, tokens):
        """Parse map literal from tokens - FIXED"""
        print("  🔧 [Context] Parsing map literal from tokens")
        
        if not tokens or tokens[0].type != LBRACE:
            print("  ❌ Not a valid map literal - no opening brace")
            return None

        pairs = []
        i = 1  # Skip opening brace
        
        while i < len(tokens) and tokens[i].type != RBRACE:
            # Parse key
            if tokens[i].type == STRING:
                key = StringLiteral(tokens[i].literal)
            elif tokens[i].type == IDENT:
                key = Identifier(tokens[i].literal)
            else:
                print(f"  ❌ Invalid map key: {tokens[i].type}")
                return None

            # Expect colon
            i += 1
            if i >= len(tokens) or tokens[i].type != COLON:
                print("  ❌ Expected colon after map key")
                return None

            # Parse value
            i += 1
            if i >= len(tokens):
                print("  ❌ Expected value after colon")
                return None

            # Parse value expression (could be simple value or nested structure)
            value_start = i
            value_end = i
            
            # Find the end of this value (comma or closing brace)
            brace_count = 0
            while value_end < len(tokens) and tokens[value_end].type != RBRACE:
                if tokens[value_end].type == COMMA and brace_count == 0:
                    break
                if tokens[value_end].type == LBRACE:
                    brace_count += 1
                elif tokens[value_end].type == RBRACE:
                    brace_count -= 1
                value_end += 1

            value_tokens = tokens[i:value_end]
            value_expression = self._parse_expression(value_tokens)
            
            if value_expression is None:
                print("  ❌ Could not parse map value")
                return None

            pairs.append((key, value_expression))
            i = value_end

            # Skip comma if present
            if i < len(tokens) and tokens[i].type == COMMA:
                i += 1
            
            # This check is crucial to prevent infinite loop if the RBRACE is missing
            if i == len(tokens) and tokens[i-1].type != RBRACE:
                print("  ⚠️ Warning: Reached end of tokens without closing map brace.")
                break

        print(f"  ✅ Parsed map literal with {len(pairs)} pairs")
        
        # Create MapLiteral - it is now imported from .zexus_ast
        return MapLiteral(pairs=pairs)


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
        value_tokens = tokens[2:]
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
        inner_tokens = tokens[2:-1] if tokens[-1].type == RPAREN else tokens[2:]
        arguments = self._parse_argument_list(inner_tokens)

        call_expression = CallExpression(Identifier(function_name), arguments)
        return ExpressionStatement(call_expression)

    def _parse_statement_block_context(self, block_info, all_tokens):
        """Parse standalone statement blocks - FIXED to use direct parsers"""
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
        else:
            return self._parse_generic_statement_block(block_info, all_tokens)

    def _parse_generic_statement_block(self, block_info, all_tokens):
        """Parse generic statement block - RETURNS ExpressionStatement"""
        tokens = block_info['tokens']
        expression = self._parse_expression(tokens)
        if expression:
            return ExpressionStatement(expression)
        return None

    # === EXPRESSION PARSING METHODS ===

    def _parse_paren_block_context(self, block_info, all_tokens):
        """Parse parentheses block - FIXED to return proper statements"""
        print("🔧 [Context] Parsing parentheses block")
        tokens = block_info['tokens']
        if len(tokens) < 3:
            return None

        context = self.get_current_context()
        start_idx = block_info['start_index']

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
        """Parse print statement with sophisticated expression parsing"""
        print("🔧 [Context] Parsing print statement with expression")
        tokens = block_info['tokens']

        if len(tokens) < 3:
            return PrintStatement(StringLiteral(""))

        inner_tokens = tokens[1:-1]

        if not inner_tokens:
            return PrintStatement(StringLiteral(""))

        expression = self._parse_expression(inner_tokens)
        return PrintStatement(expression)

    def _parse_expression(self, tokens):
        """Parse a full expression from tokens"""
        if not tokens:
            return StringLiteral("")

        # Handle string concatenation: "a" + "b" + "c"
        for i, token in enumerate(tokens):
            if token.type == PLUS:
                left_tokens = tokens[:i]
                right_tokens = tokens[i+1:]
                left_expr = self._parse_expression(left_tokens)
                right_expr = self._parse_expression(right_tokens)
                return InfixExpression(left_expr, "+", right_expr)

        # Handle function calls: string(variable)
        if len(tokens) >= 3 and tokens[0].type == IDENT and tokens[1].type == LPAREN:
            function_name = tokens[0].literal
            arg_tokens = self._extract_nested_tokens(tokens, 1)
            arguments = self._parse_argument_list(arg_tokens)
            return CallExpression(Identifier(function_name), arguments)
        
        # Handle map literals in expressions: { key: value }
        if tokens[0].type == LBRACE:
            return self._parse_map_literal(tokens)

        # Handle single token expressions
        if len(tokens) == 1:
            return self._parse_single_token_expression(tokens[0])

        # Handle complex expressions by creating a compound representation
        return self._parse_compound_expression(tokens)

    def _parse_single_token_expression(self, token):
        """Parse a single token into an expression"""
        if token.type == STRING:
            return StringLiteral(token.literal)
        elif token.type == INT:
            return IntegerLiteral(int(token.literal))
        elif token.type == FLOAT:
            return FloatLiteral(float(token.literal))
        elif token.type == IDENT:
            return Identifier(token.literal)
        elif token.type == TRUE:
            return Boolean(True)
        elif token.type == FALSE:
            return Boolean(False)
        else:
            return StringLiteral(token.literal)

    def _parse_compound_expression(self, tokens):
        """Parse compound expressions with multiple tokens"""
        expression_parts = []
        i = 0

        while i < len(tokens):
            token = tokens[i]
            if token.type == IDENT and i + 1 < len(tokens) and tokens[i+1].type == LPAREN:
                func_name = token.literal
                arg_tokens = self._extract_nested_tokens(tokens, i+1)
                arguments = self._parse_argument_list(arg_tokens)
                expression_parts.append(CallExpression(Identifier(func_name), arguments))
                i += len(arg_tokens) + 2
            else:
                expression_parts.append(self._parse_single_token_expression(token))
                i += 1

        if len(expression_parts) > 1:
            return expression_parts[0]
        elif expression_parts:
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

    def _parse_argument_list(self, tokens):
        """Parse comma-separated argument list"""
        arguments = []
        current_arg = []
        brace_count = 0
        paren_count = 0
        bracket_count = 0

        for token in tokens:
            if token.type == LBRACE: brace_count += 1
            elif token.type == RBRACE: brace_count -= 1
            elif token.type == LPAREN: paren_count += 1
            elif token.type == RPAREN: paren_count -= 1
            elif token.type == LBRACKET: bracket_count += 1
            elif token.type == RBRACKET: bracket_count -= 1

            if token.type == COMMA and brace_count == 0 and paren_count == 0 and bracket_count == 0:
                if current_arg:
                    arguments.append(self._parse_expression(current_arg))
                    current_arg = []
            else:
                current_arg.append(token)

        if current_arg:
            arguments.append(self._parse_expression(current_arg))

        return arguments

    def _parse_function_call(self, block_info, all_tokens):
        """Parse function call expression with arguments"""
        start_idx = block_info['start_index']
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
        print(f"🔧 [Context] Parsing screen: {block_info.get(