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
        """Parse let statement block - RETURNS LetStatement"""
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
        
        value_expression = self._parse_expression(value_tokens)
        if value_expression is None:
            print("  ❌ Could not parse value expression")
            return None
            
        print(f"  ✅ Let statement: {variable_name} = {value_expression}")
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

        for token in tokens:
            if token.type == COMMA:
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
                if tokens[i + 1].type == LPAREN and i + 2 < len(tokens):
                    if tokens[i + 2].type == IDENT:
                        return Identifier(tokens[i + 2].literal)
                elif tokens[i + 1].type == IDENT:
                    return Identifier(tokens[i + 1].literal)
        return Identifier("error")

    def _extract_function_parameters(self, block_info, all_tokens):
        """Extract function parameters from function signature"""
        params = []
        start_idx = block_info['start_index']
        for i in range(max(0, start_idx - 10), start_idx):
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
        start_idx = block_info['start_index']
        for i in range(max(0, start_idx - 5), start_idx):
            if i < len(all_tokens) and all_tokens[i].type == LPAREN:
                j = i + 1
                while j < len(all_tokens) and all_tokens[j].type != RPAREN:
                    if all_tokens[j].type == IDENT:
                        return Identifier(all_tokens[j].literal)
                    j += 1
                break
        return Identifier("true")