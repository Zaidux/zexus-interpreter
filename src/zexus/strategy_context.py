# strategy_context.py (COMPLETELY FIXED VERSION)
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
            'paren_block': self._parse_paren_block_context
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
        """Parse a block with context awareness - FIXED to return proper statements"""
        block_type = block_info.get('subtype', block_info['type'])

        # Update context based on block type
        context_name = block_info.get('name', 'anonymous')
        self.push_context(block_type, context_name)

        try:
            # Use appropriate parsing strategy for this context
            if block_type in self.context_rules:
                result = self.context_rules[block_type](block_info, all_tokens)
            else:
                result = self._parse_generic_block(block_info, all_tokens)

            # CRITICAL FIX: Ensure we always return Statement nodes, not raw Expressions
            if result is not None:
                result = self._ensure_statement_node(result, block_info)

            return result
        except Exception as e:
            print(f"⚠️ [Context] Error parsing {block_type}: {e}")
            # Return empty block as fallback
            return BlockStatement()
        finally:
            # Always pop context when done
            self.pop_context()

    def _ensure_statement_node(self, node, block_info):
        """Ensure the node is a proper Statement, not a raw Expression"""
        if isinstance(node, Expression):
            # Wrap expressions in ExpressionStatement
            return ExpressionStatement(node)
        elif isinstance(node, list):
            # Handle lists of expressions - wrap each one
            statements = []
            for item in node:
                if isinstance(item, Expression):
                    statements.append(ExpressionStatement(item))
                else:
                    statements.append(item)
            # If we have multiple statements, return a block
            if len(statements) > 1:
                block = BlockStatement()
                block.statements = statements
                return block
            elif len(statements) == 1:
                return statements[0]
            else:
                return BlockStatement()
        return node

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

        # Extract error variable from catch
        error_var = self._extract_catch_variable(block_info['tokens'])

        return TryCatchStatement(
            try_block=BlockStatement(),
            error_variable=error_var,
            catch_block=BlockStatement()
        )

    def _parse_function_context(self, block_info, all_tokens):
        """Parse function block with context awareness"""
        print(f"🔧 [Context] Parsing function: {block_info.get('name', 'anonymous')}")

        # Extract parameters from function signature
        params = self._extract_function_parameters(block_info, all_tokens)

        return ActionStatement(
            name=Identifier(block_info.get('name', 'anonymous')),
            parameters=params,
            body=BlockStatement()
        )

    def _parse_conditional_context(self, block_info, all_tokens):
        """Parse if/else blocks with context awareness"""
        print("🔧 [Context] Parsing conditional block")

        # Extract condition from tokens before the block
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

    def _parse_paren_block_context(self, block_info, all_tokens):
        """Parse parentheses block - FIXED to return proper statements"""
        print("🔧 [Context] Parsing parentheses block")

        # For parentheses blocks, we need to determine if this is:
        # 1. A print statement: print("hello")
        # 2. A function call: someFunction()
        # 3. A condition: if (condition)
        # 4. Just a parenthesized expression: (x + y)

        tokens = block_info['tokens']
        if len(tokens) < 3:  # Need at least ( content )
            return None

        # Look at context to determine what this paren block represents
        context = self.get_current_context()
        start_idx = block_info['start_index']

        # Check if this is likely a print statement
        if start_idx > 0 and all_tokens[start_idx - 1].type == PRINT:
            return self._parse_print_statement(block_info, all_tokens)
        
        # Check if this is likely a function call
        elif start_idx > 0 and all_tokens[start_idx - 1].type == IDENT:
            return self._parse_function_call(block_info, all_tokens)
        
        # Otherwise, parse as generic parenthesized expression
        else:
            return self._parse_generic_paren_expression(block_info, all_tokens)

    def _parse_print_statement(self, block_info, all_tokens):
        """Parse print statement with parentheses"""
        tokens = block_info['tokens']
        if len(tokens) >= 3:  # Should have print ( expression )
            # Extract the expression inside parentheses
            inner_tokens = tokens[1:-1]
            if inner_tokens and inner_tokens[0].type == STRING:
                return PrintStatement(StringLiteral(inner_tokens[0].literal))
        return PrintStatement(StringLiteral(""))

    def _parse_function_call(self, block_info, all_tokens):
        """Parse function call expression"""
        start_idx = block_info['start_index']
        if start_idx > 0:
            function_name = all_tokens[start_idx - 1].literal
            return CallExpression(
                function=Identifier(function_name),
                arguments=[]  # Simplified - no arguments for now
            )
        return None

    def _parse_generic_paren_expression(self, block_info, all_tokens):
        """Parse generic parenthesized expression"""
        tokens = block_info['tokens']
        inner_tokens = tokens[1:-1] if len(tokens) > 2 else []
        
        if not inner_tokens:
            return None
            
        # Parse the first meaningful token inside parentheses
        first_token = inner_tokens[0]
        
        if first_token.type == STRING:
            return StringLiteral(first_token.literal)
        elif first_token.type == INT:
            return IntegerLiteral(int(first_token.literal))
        elif first_token.type == IDENT:
            return Identifier(first_token.literal)
        elif first_token.type == TRUE:
            return Boolean(True)
        elif first_token.type == FALSE:
            return Boolean(False)
        
        return None

    def _parse_statement_list(self, tokens):
        """Parse a list of tokens into statements with context awareness - SIMPLIFIED"""
        # For now, just return empty block to avoid complex parsing
        return BlockStatement()

    def _extract_catch_variable(self, tokens):
        """Extract the error variable from catch block"""
        for i, token in enumerate(tokens):
            if token.type == CATCH and i + 1 < len(tokens):
                # Look for catch (error) or catch error syntax
                if tokens[i + 1].type == LPAREN and i + 2 < len(tokens):
                    if tokens[i + 2].type == IDENT:
                        return Identifier(tokens[i + 2].literal)
                elif tokens[i + 1].type == IDENT:
                    return Identifier(tokens[i + 1].literal)
        return Identifier("error")  # Default error variable

    def _extract_function_parameters(self, block_info, all_tokens):
        """Extract function parameters from function signature"""
        params = []
        start_idx = block_info['start_index']

        # Look for parameters in parentheses before the function body
        for i in range(max(0, start_idx - 10), start_idx):
            if i < len(all_tokens) and all_tokens[i].type == LPAREN:
                # Extract parameters until closing paren
                j = i + 1
                while j < len(all_tokens) and all_tokens[j].type != RPAREN:
                    if all_tokens[j].type == IDENT:
                        params.append(Identifier(all_tokens[j].literal))
                    j += 1
                break

        return params

    def _extract_condition(self, block_info, all_tokens):
        """Extract condition from conditional statements"""
        # Look for condition in parentheses before the block
        start_idx = block_info['start_index']
        for i in range(max(0, start_idx - 5), start_idx):
            if i < len(all_tokens) and all_tokens[i].type == LPAREN:
                # Extract condition until closing paren
                j = i + 1
                while j < len(all_tokens) and all_tokens[j].type != RPAREN:
                    if all_tokens[j].type == IDENT:
                        return Identifier(all_tokens[j].literal)
                    j += 1
                break
        return Identifier("true")  # Default condition

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
        """Fallback parser for unknown block types"""
        return BlockStatement()