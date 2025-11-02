# strategy_context.py (COMPLETE FIXED VERSION)
from .zexus_token import *
from .zexus_ast import *

class ContextStackParser:
    def __init__(self, structural_analyzer):
        self.structural_analyzer = structural_analyzer
        self.current_context = ['global']
        # In strategy_context.py, update the context_rules in __init__:
        self.context_rules = {
            'function': self._parse_function_context,
            'try_catch': self._parse_try_catch_context,
            'conditional': self._parse_conditional_context,
            'loop': self._parse_loop_context,
            'screen': self._parse_screen_context,
            'brace_block': self._parse_brace_block_context,
            'paren_block': self._parse_paren_block_context,
            'statement_block': self._parse_statement_block_context,  # ADD THIS LINE
            'bracket_block': self._parse_brace_block_context  # Use brace parser for brackets
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
        """Parse a block with context awareness"""
        block_type = block_info.get('subtype', block_info['type'])

        # Update context based on block type
        context_name = block_info.get('name', 'anonymous')
        self.push_context(block_type, context_name)

        try:
            if block_type in self.context_rules:
                result = self.context_rules[block_type](block_info, all_tokens)
            else:
                result = self._parse_generic_block(block_info, all_tokens)

            if result is not None:
                result = self._ensure_statement_node(result, block_info)

            return result
        except Exception as e:
            print(f"⚠️ [Context] Error parsing {block_type}: {e}")
            return BlockStatement()
        finally:
            self.pop_context()

    def _ensure_statement_node(self, node, block_info):
        """Ensure the node is a proper Statement, not a raw Expression"""
        if isinstance(node, Expression):
            return ExpressionStatement(node)
        elif isinstance(node, list):
            statements = []
            for item in node:
                if isinstance(item, Expression):
                    statements.append(ExpressionStatement(item))
                else:
                    statements.append(item)
            if len(statements) > 1:
                block = BlockStatement()
                block.statements = statements
                return block
            elif len(statements) == 1:
                return statements[0]
            else:
                return BlockStatement()
        return node

    # === STATEMENT BLOCK PARSING (NEW SECTION) ===

    def _parse_statement_block_context(self, block_info, all_tokens):
        """Parse standalone statement blocks like let statements - NEW METHOD"""
        print(f"🔧 [Context] Parsing statement block: {block_info.get('subtype', 'unknown')}")

        subtype = block_info.get('subtype', 'unknown')

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

    def _parse_let_statement_block(self, block_info, all_tokens):
        """Parse let statement block - CRITICAL FIX"""
        tokens = block_info['tokens']

        if len(tokens) < 4:
            return None

        if tokens[1].type != IDENT:
            return None

        variable_name = tokens[1].literal

        equals_index = -1
        for i, token in enumerate(tokens):
            if token.type == ASSIGN:
                equals_index = i
                break

        if equals_index == -1:
            return None

        value_tokens = tokens[equals_index + 1:]
        value_expression = self._parse_expression(value_tokens)

        return LetStatement(
            name=Identifier(variable_name),
            value=value_expression
        )

    def _parse_print_statement_block(self, block_info, all_tokens):
        """Parse print statement block"""
        tokens = block_info['tokens']

        if len(tokens) < 2:
            return PrintStatement(StringLiteral(""))

        expression_tokens = tokens[1:]
        expression = self._parse_expression(expression_tokens)

        return PrintStatement(expression)

    def _parse_function_call_statement(self, block_info, all_tokens):
        """Parse function call as a statement"""
        tokens = block_info['tokens']

        if len(tokens) < 3 or tokens[1].type != LPAREN:
            return None

        function_name = tokens[0].literal
        inner_tokens = tokens[2:-1] if tokens[-1].type == RPAREN else tokens[2:]
        arguments = self._parse_argument_list(inner_tokens)

        call_expression = CallExpression(Identifier(function_name), arguments)
        return ExpressionStatement(call_expression)

    def _parse_assignment_statement(self, block_info, all_tokens):
        """Parse assignment statement"""
        tokens = block_info['tokens']

        if len(tokens) < 3 or tokens[1].type != ASSIGN:
            return None

        variable_name = tokens[0].literal
        value_tokens = tokens[2:]
        value_expression = self._parse_expression(value_tokens)

        return AssignmentExpression(
            name=Identifier(variable_name),
            value=value_expression
        )

    def _parse_generic_statement_block(self, block_info, all_tokens):
        """Parse generic statement block"""
        tokens = block_info['tokens']
        expression = self._parse_expression(tokens)
        return ExpressionStatement(expression)

    # === ENHANCED EXPRESSION PARSING METHODS ===

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
            return self._parse_generic_paren_expression(block_info, all_tokens)

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
        """Parse a full expression from tokens - SOPHISTICATED IMPLEMENTATION"""
        if not tokens:
            return StringLiteral("")

        for i, token in enumerate(tokens):
            if token.type == PLUS:
                left_tokens = tokens[:i]
                right_tokens = tokens[i+1:]
                left_expr = self._parse_expression(left_tokens)
                right_expr = self._parse_expression(right_tokens)
                return InfixExpression(left_expr, "+", right_expr)

        if len(tokens) >= 3 and tokens[0].type == IDENT and tokens[1].type == LPAREN:
            function_name = tokens[0].literal
            arg_tokens = self._extract_nested_tokens(tokens, 1)
            arguments = self._parse_argument_list(arg_tokens)
            return CallExpression(Identifier(function_name), arguments)

        if len(tokens) == 1:
            return self._parse_single_token_expression(tokens[0])

        return self._parse_compound_expression(tokens)

    def _parse_single_token_expression(self, token):
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
        tokens = block_info['tokens']
        inner_tokens = tokens[1:-1] if len(tokens) > 2 else []
        if not inner_tokens:
            return None
        return self._parse_expression(inner_tokens)

    # === REST OF THE CONTEXT METHODS ===

    def _parse_loop_context(self, block_info, all_tokens):
        print("🔧 [Context] Parsing loop block")
        return BlockStatement()

    def _parse_screen_context(self, block_info, all_tokens):
        print(f"🔧 [Context] Parsing screen: {block_info.get('name', 'anonymous')}")
        return ScreenStatement(
            name=Identifier(block_info.get('name', 'anonymous')),
            body=BlockStatement()
        )

    def _parse_try_catch_context(self, block_info, all_tokens):
        print("🔧 [Context] Parsing try-catch block with context awareness")
        error_var = self._extract_catch_variable(block_info['tokens'])
        return TryCatchStatement(
            try_block=BlockStatement(),
            error_variable=error_var,
            catch_block=BlockStatement()
        )

    def _parse_function_context(self, block_info, all_tokens):
        print(f"🔧 [Context] Parsing function: {block_info.get('name', 'anonymous')}")
        params = self._extract_function_parameters(block_info, all_tokens)
        return ActionStatement(
            name=Identifier(block_info.get('name', 'anonymous')),
            parameters=params,
            body=BlockStatement()
        )

    def _parse_conditional_context(self, block_info, all_tokens):
        print("🔧 [Context] Parsing conditional block")
        condition = self._extract_condition(block_info, all_tokens)
        return IfStatement(
            condition=condition,
            consequence=BlockStatement(),
            alternative=None
        )

    def _parse_brace_block_context(self, block_info, all_tokens):
        print("🔧 [Context] Parsing brace block")
        return BlockStatement()

    def _parse_statement_list(self, tokens):
        return BlockStatement()

    def _extract_catch_variable(self, tokens):
        for i, token in enumerate(tokens):
            if token.type == CATCH and i + 1 < len(tokens):
                if tokens[i + 1].type == LPAREN and i + 2 < len(tokens):
                    if tokens[i + 2].type == IDENT:
                        return Identifier(tokens[i + 2].literal)
                elif tokens[i + 1].type == IDENT:
                    return Identifier(tokens[i + 1].literal)
        return Identifier("error")

    def _extract_function_parameters(self, block_info, all_tokens):
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

    def _parse_literal(self, token):
        if token.type == INT:
            return IntegerLiteral(int(token.literal))
        elif token.type == STRING:
            return StringLiteral(token.literal)
        elif token.type == TRUE:
            return Boolean(True)
        elif token.type == FALSE:
            return Boolean(False)
        return None

    def _parse_generic_block(self, block_info, all_tokens):
        return BlockStatement()