# strategy_context.py
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
        print(f"📥 [Context] Pushed: {context_str} | Stack: {self.current_context}")
    
    def pop_context(self):
        """Pop the current context from the stack"""
        if len(self.current_context) > 1:
            popped = self.current_context.pop()
            print(f"📤 [Context] Popped: {popped} | Stack: {self.current_context}")
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
            # Use appropriate parsing strategy for this context
            if block_type in self.context_rules:
                result = self.context_rules[block_type](block_info, all_tokens)
            else:
                result = self._parse_generic_block(block_info, all_tokens)
                
            return result
        finally:
            # Always pop context when done
            self.pop_context()
    
    def _parse_try_catch_context(self, block_info, all_tokens):
        """Parse try-catch block with full context awareness"""
        print("🔧 [Context] Parsing try-catch block with context awareness")
        
        # Extract try and catch sections from structural analysis
        try_section = block_info.get('try_section')
        catch_section = block_info.get('catch_section')
        
        if not try_section or not catch_section:
            # Fallback: extract from tokens manually
            try_section, catch_section = self._extract_try_catch_sections(block_info['tokens'])
        
        # Parse try block in try context
        self.push_context('try_block')
        try_block = self._parse_statement_list(try_section['tokens']) if try_section else BlockStatement()
        self.pop_context()
        
        # Parse catch block in catch context  
        self.push_context('catch_block')
        catch_block = self._parse_statement_list(catch_section['tokens']) if catch_section else BlockStatement()
        self.pop_context()
        
        # Extract error variable from catch
        error_var = self._extract_catch_variable(block_info['tokens'])
        
        return TryCatchStatement(
            try_block=try_block,
            error_variable=error_var,
            catch_block=catch_block
        )
    
    def _parse_function_context(self, block_info, all_tokens):
        """Parse function block with context awareness"""
        print(f"🔧 [Context] Parsing function: {block_info.get('name', 'anonymous')}")
        
        # Extract parameters from function signature
        params = self._extract_function_parameters(block_info, all_tokens)
        
        # Parse function body
        self.push_context('function_body')
        body = self._parse_statement_list(block_info['tokens'])
        self.pop_context()
        
        return ActionStatement(
            name=Identifier(block_info.get('name', 'anonymous')),
            parameters=params,
            body=body
        )
    
    def _parse_conditional_context(self, block_info, all_tokens):
        """Parse if/else blocks with context awareness"""
        print("🔧 [Context] Parsing conditional block")
        
        # Extract condition from tokens before the block
        condition = self._extract_condition(block_info, all_tokens)
        
        # Parse consequence (if body)
        self.push_context('conditional_body')
        consequence = self._parse_statement_list(block_info['tokens'])
        self.pop_context()
        
        # Check for else block in nested blocks
        alternative = None
        for nested in block_info.get('nested_blocks', []):
            if (nested.get('subtype') == 'conditional' and 
                any(t.type == ELSE for t in nested['tokens'])):
                alternative = self._parse_statement_list(nested['tokens'])
                break
        
        return IfStatement(
            condition=condition,
            consequence=consequence,
            alternative=alternative
        )
    
    def _parse_brace_block_context(self, block_info, all_tokens):
        """Parse generic brace block with context awareness"""
        print("🔧 [Context] Parsing brace block")
        return self._parse_statement_list(block_info['tokens'])
    
    def _parse_paren_block_context(self, block_info, all_tokens):
        """Parse parentheses block (usually conditions/parameters)"""
        print("🔧 [Context] Parsing parentheses block")
        
        # For parentheses blocks, parse as expression list
        expressions = []
        i = 1  # Skip opening paren
        while i < len(block_info['tokens']) - 1:  # Skip closing paren
            token = block_info['tokens'][i]
            if token.type not in [COMMA, RPAREN]:
                # Create simple identifier expressions for now
                if token.type == IDENT:
                    expressions.append(Identifier(token.literal))
                elif token.type in [INT, STRING, TRUE, FALSE]:
                    expressions.append(self._parse_literal(token))
            i += 1
        
        return expressions[0] if len(expressions) == 1 else expressions
    
    def _parse_statement_list(self, tokens):
        """Parse a list of tokens into statements with context awareness"""
        statements = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            # Skip block delimiters for statement parsing
            if token.type in [LBRACE, RBRACE, LPAREN, RPAREN, LBRACKET, RBRACKET]:
                i += 1
                continue
            
            # Parse based on token type and current context
            statement = self._parse_statement_by_context(token, tokens, i)
            if statement:
                statements.append(statement)
                # Advance index based on statement length
                i = statement.get('end_index', i + 1)
            else:
                # Skip problematic tokens but continue
                print(f"⚠️ [Context] Skipping token in {self.get_current_context()}: {token}")
                i += 1
        
        return BlockStatement(statements)
    
    def _parse_statement_by_context(self, token, tokens, start_index):
        """Parse a statement based on current context"""
        current_context = self.get_current_context()
        
        # Different parsing rules based on context
        if 'try_block' in current_context:
            return self._parse_try_context_statement(token, tokens, start_index)
        elif 'catch_block' in current_context:
            return self._parse_catch_context_statement(token, tokens, start_index)
        elif 'function_body' in current_context:
            return self._parse_function_statement(token, tokens, start_index)
        else:
            return self._parse_generic_statement(token, tokens, start_index)
    
    def _parse_try_context_statement(self, token, tokens, start_index):
        """Parse statements in try block context - more forgiving"""
        if token.type == LET:
            return self._parse_let_statement(tokens, start_index)
        elif token.type == IF:
            return self._parse_if_statement(tokens, start_index)
        elif token.type == RETURN:
            return self._parse_return_statement(tokens, start_index)
        elif token.type == DEBUG:
            return self._parse_debug_statement(tokens, start_index)
        else:
            # In try context, attempt to parse as expression
            return self._parse_expression_statement(tokens, start_index)
    
    def _parse_catch_context_statement(self, token, tokens, start_index):
        """Parse statements in catch block context - error handling focus"""
        if token.type == LET:
            return self._parse_let_statement(tokens, start_index)
        elif token.type == RETURN:
            return self._parse_return_statement(tokens, start_index)
        elif token.type == DEBUG:
            return self._parse_debug_statement(tokens, start_index)
        else:
            # In catch context, allow various error handling patterns
            return self._parse_expression_statement(tokens, start_index)
    
    def _extract_try_catch_sections(self, tokens):
        """Extract try and catch sections from tokens"""
        try_section = None
        catch_section = None
        
        # Find try block
        for i, token in enumerate(tokens):
            if token.type == TRY:
                # Find opening brace after try
                for j in range(i + 1, len(tokens)):
                    if tokens[j].type == LBRACE:
                        try_start = j
                        # Find matching closing brace
                        brace_count = 1
                        for k in range(j + 1, len(tokens)):
                            if tokens[k].type == LBRACE:
                                brace_count += 1
                            elif tokens[k].type == RBRACE:
                                brace_count -= 1
                                if brace_count == 0:
                                    try_section = {
                                        'tokens': tokens[try_start:k + 1]
                                    }
                                    break
                        break
                break
        
        # Find catch block
        for i, token in enumerate(tokens):
            if token.type == CATCH:
                # Find opening brace after catch
                for j in range(i + 1, len(tokens)):
                    if tokens[j].type == LBRACE:
                        catch_start = j
                        # Find matching closing brace
                        brace_count = 1
                        for k in range(j + 1, len(tokens)):
                            if tokens[k].type == LBRACE:
                                brace_count += 1
                            elif tokens[k].type == RBRACE:
                                brace_count -= 1
                                if brace_count == 0:
                                    catch_section = {
                                        'tokens': tokens[catch_start:k + 1]
                                    }
                                    break
                        break
                break
        
        return try_section, catch_section
    
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
    
    # Placeholder methods for statement parsing (to be integrated with existing parser)
    def _parse_let_statement(self, tokens, start_index):
        # Simplified implementation - integrate with existing parser later
        if start_index + 2 < len(tokens):
            name = Identifier(tokens[start_index + 1].literal)
            return LetStatement(name=name, value=None)
        return None
    
    def _parse_if_statement(self, tokens, start_index):
        return IfStatement(condition=None, consequence=BlockStatement(), alternative=None)
    
    def _parse_return_statement(self, tokens, start_index):
        return ReturnStatement(return_value=None)
    
    def _parse_debug_statement(self, tokens, start_index):
        return DebugStatement(value=None)
    
    def _parse_expression_statement(self, tokens, start_index):
        return ExpressionStatement(expression=None)
    
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
    
    def _parse_generic_statement(self, token, tokens, start_index):
        """Fallback statement parser"""
        return self._parse_expression_statement(tokens, start_index)