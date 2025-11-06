# strategy_structural.py (FIXED VERSION - with improved try-catch parsing)
from .zexus_token import *

class StructuralAnalyzer:
    def __init__(self):
        self.blocks = {}
        self.block_counter = 0

    def analyze(self, tokens):
        """First pass: understand the complete structure of the code - FIXED"""
        print("🔍 [Structural Analysis] Mapping code structure...")

        self.blocks = {}
        self.block_counter = 0
        stack = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Skip EOF for structural analysis
            if token.type == EOF:
                i += 1
                continue

            # Track block starts
            if token.type in [LBRACE, LPAREN, LBRACKET]:
                block_info = {
                    'id': f"block_{self.block_counter}",
                    'type': self._get_block_type(token.type),
                    'start_index': i,
                    'start_token': token,
                    'end_index': None,
                    'tokens': [token],
                    'nested_blocks': [],
                    'parent': stack[-1]['id'] if stack else None
                }
                stack.append(block_info)
                self.block_counter += 1
                i += 1
                continue

            # Track block ends  
            if token.type in [RBRACE, RPAREN, RBRACKET] and stack:
                block = stack.pop()
                block['end_index'] = i
                block['end_token'] = token
                block['tokens'].append(token)

                # Identify block specifics (function, try-catch, etc.)
                block = self._identify_block_details(block, tokens)

                if stack:
                    # Nested block - add to parent
                    stack[-1]['nested_blocks'].append(block)
                else:
                    # Top-level block
                    self.blocks[block['id']] = block

                i += 1
                continue

            # Track content for current block
            if stack:
                stack[-1]['tokens'].append(token)
                i += 1
                continue

            # === CRITICAL FIX: Handle standalone statements like 'let x = 42' ===
            if self._is_statement_start(token):
                # Start a new statement block
                statement_info = self._create_statement_block(tokens, i)
                if statement_info:
                    self.blocks[statement_info['id']] = statement_info
                    i = statement_info['end_index'] + 1  # Skip to after this statement
                    continue

            i += 1

        # Handle any unclosed blocks (error recovery)
        self._handle_unclosed_blocks(stack, tokens)

        print(f"✅ Found {len(self.blocks)} top-level blocks with {sum(len(b['nested_blocks']) for b in self.blocks.values())} nested blocks")
        return self.blocks

    def _is_statement_start(self, token):
        """Check if this token starts a statement - FIXED to include let statements"""
        statement_starters = [LET, PRINT, RETURN, IF, WHILE, FOR, ACTION, SCREEN, 
                            USE, EXPORT, DEBUG, TRY, EXTERNAL, IDENT]
        return token.type in statement_starters

    def _create_statement_block(self, tokens, start_index):
        """Create a block for a standalone statement - FIXED"""
        token = tokens[start_index]

        # CRITICAL FIX: Check for try-catch first before breaking into statements
        if token.type == TRY:
            return self._parse_try_catch_statement(tokens, start_index)

        # Determine statement type for other statements
        if token.type == LET:
            return self._parse_let_statement(tokens, start_index)
        elif token.type == PRINT:
            return self._parse_print_statement(tokens, start_index)
        elif token.type == IDENT:
            # Could be assignment or function call
            return self._parse_expression_statement(tokens, start_index)
        else:
            # Generic statement
            return self._parse_generic_statement(tokens, start_index)

    def _parse_try_catch_statement(self, tokens, start_index):
        """Parse a try-catch statement as a single block - IMPROVED FIX"""
        print(f"🔍 [Structural] Parsing try-catch statement at index {start_index}")
        
        statement_tokens = []
        i = start_index
        brace_count = 0
        in_try_block = False
        in_catch_block = False
        catch_found = False
        
        while i < len(tokens):
            current_token = tokens[i]
            statement_tokens.append(current_token)
            
            print(f"  📝 Token {i}: {current_token.type} = '{current_token.literal}'")
            
            # Track braces to find the complete try-catch structure
            if current_token.type == LBRACE:
                brace_count += 1
            elif current_token.type == RBRACE:
                brace_count -= 1
                
            # Check if we're starting try block
            if current_token.type == TRY and not in_try_block:
                in_try_block = True
                print("  🔧 [Try] Found try keyword")
                
            # Check if we're starting catch block  
            if current_token.type == CATCH and not catch_found:
                catch_found = True
                print("  🔧 [Catch] Found catch keyword")
                
            # Mark when we enter catch block (after catch keyword and opening brace)
            if catch_found and not in_catch_block and brace_count > 0:
                in_catch_block = True
                print("  🔧 [Catch] Entered catch block")
                
            # CRITICAL FIX: Don't stop for statement starters inside try-catch blocks
            # Only stop when we've completed the entire try-catch structure
            if (in_try_block and catch_found and in_catch_block and 
                brace_count == 0 and current_token.type == RBRACE):
                print(f"  ✅ Ending try-catch statement at token {i}: complete structure captured")
                break
                
            # Also stop if we hit EOF
            if current_token.type == EOF:
                print(f"  ⚠️ Ending try-catch at token {i}: EOF")
                break
                
            i += 1
            
        # CRITICAL FIX: If we didn't capture enough tokens, try to capture more
        # This handles cases where the try-catch structure is complex
        if len(statement_tokens) < 10 and i < len(tokens) - 1:
            print("  🔄 Try-catch too short, capturing more tokens...")
            max_tokens = min(50, len(tokens) - i)  # Don't go beyond available tokens
            for j in range(max_tokens):
                if i + j >= len(tokens):
                    break
                    
                current_token = tokens[i + j]
                statement_tokens.append(current_token)
                print(f"  📝 Additional token {i + j}: {current_token.type} = '{current_token.literal}'")
                
                if current_token.type == LBRACE:
                    brace_count += 1
                elif current_token.type == RBRACE:
                    brace_count -= 1
                    
                # Stop if we complete the structure
                if (in_try_block and catch_found and in_catch_block and 
                    brace_count == 0 and current_token.type == RBRACE):
                    print(f"  ✅ Completed try-catch structure at token {i + j}")
                    i = i + j  # Update position
                    break

        block_id = f"block_{self.block_counter}"
        self.block_counter += 1

        block_info = {
            'id': block_id,
            'type': 'statement_block',
            'subtype': 'try_catch_statement',
            'start_index': start_index,
            'end_index': i,
            'start_token': tokens[start_index],
            'end_token': tokens[i] if i < len(tokens) else tokens[-1],
            'tokens': statement_tokens,
            'nested_blocks': [],
            'parent': None
        }

        print(f"  📦 Final try-catch statement tokens: {[t.literal for t in statement_tokens]}")
        return block_info

    def _parse_let_statement(self, tokens, start_index):
        """Parse a let statement into a block - CRITICAL FIX"""
        print(f"🔍 [Structural] Parsing let statement at index {start_index}")
        statement_tokens = []
        i = start_index

        # Collect tokens until end of statement
        while i < len(tokens):
            current_token = tokens[i]
            statement_tokens.append(current_token)

            print(f"  📝 Token {i}: {current_token.type} = '{current_token.literal}'")

            # CRITICAL FIX: Don't stop at { - this could be a map literal!
            if (current_token.type == SEMICOLON or 
                (i > start_index and self._is_statement_start(current_token) and current_token.type != IDENT) or
                current_token.type == EOF):
                print(f"  🛑 Ending let statement at token {i}: {current_token.type}")
                break

            # Special handling for map literals - if we hit {, continue collecting
            if current_token.type == LBRACE:
                print(f"  🗺️ Found map literal start at token {i}, continuing...")
                # Continue to collect the entire map

            i += 1

        # CRITICAL: Make sure we have enough tokens for "let x = value"
        if len(statement_tokens) < 4:
            print(f"  ❌ Let statement too short: only {len(statement_tokens)} tokens")
            print(f"  📋 Tokens: {[t.literal for t in statement_tokens]}")
            # Try to collect more tokens if we stopped too early
            if i < len(tokens) - 1:
                print("  🔄 Trying to collect more tokens...")
                while i < len(tokens) and len(statement_tokens) < 10:  # Increased limit for maps
                    current_token = tokens[i]
                    statement_tokens.append(current_token)
                    print(f"  📝 Additional token {i}: {current_token.type} = '{current_token.literal}'")

                    if (current_token.type == SEMICOLON or 
                        current_token.type == EOF):
                        break
                    i += 1

        # Create the statement block
        block_id = f"block_{self.block_counter}"
        self.block_counter += 1

        block_info = {
            'id': block_id,
            'type': 'statement_block',
            'subtype': 'let_statement',
            'start_index': start_index,
            'end_index': i - 1,  # Adjust since we break before incrementing
            'start_token': tokens[start_index],
            'end_token': tokens[i - 1] if i > 0 else tokens[-1],
            'tokens': statement_tokens,
            'nested_blocks': [],
            'parent': None
        }

        # Extract variable name for better identification
        if len(statement_tokens) > 1 and statement_tokens[1].type == IDENT:
            block_info['name'] = statement_tokens[1].literal
            block_info['variable_name'] = statement_tokens[1].literal
            print(f"  ✅ Let statement for variable: {statement_tokens[1].literal}")
        else:
            print(f"  ⚠️ Could not extract variable name from tokens: {[t.literal for t in statement_tokens]}")

        print(f"  📦 Final let statement tokens: {[t.literal for t in statement_tokens]}")
        return block_info

    def _parse_print_statement(self, tokens, start_index):
        """Parse a print statement into a block"""
        print(f"🔍 [Structural] Parsing print statement at index {start_index}")
        statement_tokens = []
        i = start_index

        while i < len(tokens):
            current_token = tokens[i]
            statement_tokens.append(current_token)

            print(f"  📝 Token {i}: {current_token.type} = '{current_token.literal}'")

            if (current_token.type == SEMICOLON or 
                (i > start_index and self._is_statement_start(current_token)) or
                current_token.type == EOF):
                print(f"  🛑 Ending print statement at token {i}: {current_token.type}")
                break

            i += 1

        block_id = f"block_{self.block_counter}"
        self.block_counter += 1

        print(f"  📦 Final print statement tokens: {[t.literal for t in statement_tokens]}")
        return {
            'id': block_id,
            'type': 'statement_block',
            'subtype': 'print_statement',
            'start_index': start_index,
            'end_index': i - 1,
            'start_token': tokens[start_index],
            'end_token': tokens[i - 1] if i > 0 else tokens[-1],
            'tokens': statement_tokens,
            'nested_blocks': [],
            'parent': None
        }

    def _parse_expression_statement(self, tokens, start_index):
        """Parse an expression statement (function call, assignment, etc.)"""
        statement_tokens = []
        i = start_index

        while i < len(tokens):
            current_token = tokens[i]
            statement_tokens.append(current_token)

            if (current_token.type == SEMICOLON or 
                (i > start_index and self._is_statement_start(current_token)) or
                current_token.type == EOF):
                break

            i += 1

        block_id = f"block_{self.block_counter}"
        self.block_counter += 1

        # Determine what kind of expression this is
        subtype = 'expression_statement'
        if len(statement_tokens) > 1 and statement_tokens[1].type == ASSIGN:
            subtype = 'assignment_statement'
        elif (len(statement_tokens) > 1 and statement_tokens[1].type == LPAREN and
              statement_tokens[0].type == IDENT):
            subtype = 'function_call_statement'

        return {
            'id': block_id,
            'type': 'statement_block',
            'subtype': subtype,
            'start_index': start_index,
            'end_index': i - 1,
            'start_token': tokens[start_index],
            'end_token': tokens[i - 1] if i > 0 else tokens[-1],
            'tokens': statement_tokens,
            'nested_blocks': [],
            'parent': None
        }

    def _parse_generic_statement(self, tokens, start_index):
        """Parse any other kind of statement"""
        statement_tokens = []
        i = start_index

        while i < len(tokens):
            current_token = tokens[i]
            statement_tokens.append(current_token)

            if (current_token.type == SEMICOLON or 
                (i > start_index and self._is_statement_start(current_token)) or
                current_token.type == EOF):
                break

            i += 1

        block_id = f"block_{self.block_counter}"
        self.block_counter += 1

        return {
            'id': block_id,
            'type': 'statement_block',
            'subtype': 'generic_statement',
            'start_index': start_index,
            'end_index': i - 1,
            'start_token': tokens[start_index],
            'end_token': tokens[i - 1] if i > 0 else tokens[-1],
            'tokens': statement_tokens,
            'nested_blocks': [],
            'parent': None
        }

    def _get_block_type(self, token_type):
        """Map token type to block type"""
        if token_type == LBRACE:
            return 'brace_block'
        elif token_type == LPAREN:
            return 'paren_block' 
        elif token_type == LBRACKET:
            return 'bracket_block'
        return 'unknown'

    def _identify_block_details(self, block, all_tokens):
        """Identify what kind of block this is based on content"""
        tokens = block['tokens']

        # Look at tokens before this block for context
        start_idx = block['start_index']
        context_tokens = []
        if start_idx > 0:
            # Look back up to 5 tokens for context
            context_start = max(0, start_idx - 5)
            context_tokens = all_tokens[context_start:start_idx]

        # Function detection: action name(...) { ... }
        if (any(t.type == ACTION for t in context_tokens) and
            any(t.type == IDENT for t in context_tokens) and
            any(t.type == LPAREN for t in context_tokens)):

            # Find the action name
            for i, token in enumerate(context_tokens):
                if token.type == ACTION and i + 1 < len(context_tokens):
                    if context_tokens[i + 1].type == IDENT:
                        block['subtype'] = 'function'
                        block['name'] = context_tokens[i + 1].literal
                        block['action_token'] = token
                        break

        # Try-catch detection
        elif any(t.type == TRY for t in tokens) and any(t.type == CATCH for t in tokens):
            block['subtype'] = 'try_catch'

            # Extract try and catch sections
            try_section = self._extract_try_section(tokens)
            catch_section = self._extract_catch_section(tokens)

            if try_section:
                block['try_section'] = try_section
            if catch_section:
                block['catch_section'] = catch_section

        # Conditional detection (if/else)
        elif any(t.type == IF for t in tokens) or any(t.type == ELSE for t in tokens):
            block['subtype'] = 'conditional'

        # Loop detection (for/while)
        elif any(t.type == FOR for t in tokens) or any(t.type == WHILE for t in tokens):
            block['subtype'] = 'loop'

        # Screen detection
        elif any(t.type == SCREEN for t in context_tokens):
            block['subtype'] = 'screen'
            for i, token in enumerate(context_tokens):
                if token.type == SCREEN and i + 1 < len(context_tokens):
                    if context_tokens[i + 1].type == IDENT:
                        block['name'] = context_tokens[i + 1].literal
                        break

        # Debug statement detection
        elif any(t.type == DEBUG for t in tokens):
            block['subtype'] = 'debug'

        # External declaration detection
        elif any(t.type == EXTERNAL for t in context_tokens):
            block['subtype'] = 'external'

        return block

    def _extract_try_section(self, tokens):
        """Extract the try block section from tokens"""
        try_start = None
        try_end = None

        for i, token in enumerate(tokens):
            if token.type == TRY:
                try_start = i
                # Look for the opening brace after try
                for j in range(i + 1, len(tokens)):
                    if tokens[j].type == LBRACE:
                        try_start = j
                        break
                break

        if try_start is not None:
            # Find matching closing brace for try block
            brace_count = 0
            for i in range(try_start, len(tokens)):
                if tokens[i].type == LBRACE:
                    brace_count += 1
                elif tokens[i].type == RBRACE:
                    brace_count -= 1
                    if brace_count == 0:
                        try_end = i
                        break

            if try_end is not None:
                return {
                    'start': try_start,
                    'end': try_end,
                    'tokens': tokens[try_start:try_end + 1]
                }

        return None

    def _extract_catch_section(self, tokens):
        """Extract the catch block section from tokens"""
        catch_start = None

        for i, token in enumerate(tokens):
            if token.type == CATCH:
                catch_start = i
                # Look for the opening brace after catch
                for j in range(i + 1, len(tokens)):
                    if tokens[j].type == LBRACE:
                        catch_start = j
                        break
                break

        if catch_start is not None:
            # Find matching closing brace for catch block
            brace_count = 0
            for i in range(catch_start, len(tokens)):
                if tokens[i].type == LBRACE:
                    brace_count += 1
                elif tokens[i].type == RBRACE:
                    brace_count -= 1
                    if brace_count == 0:
                        catch_end = i
                        return {
                            'start': catch_start,
                            'end': catch_end,
                            'tokens': tokens[catch_start:catch_end + 1]
                        }

        return None

    def _handle_unclosed_blocks(self, stack, tokens):
        """Handle any blocks that weren't properly closed"""
        for block in stack:
            block['end_index'] = len(tokens) - 1
            block['end_token'] = tokens[-1] if tokens else None
            block['unclosed'] = True
            self.blocks[block['id']] = block
            print(f"⚠️  Unclosed {block['type']} block starting at line {block['start_token'].line}")

    def get_block_hierarchy(self):
        """Return the block structure as a hierarchical tree"""
        hierarchy = []
        for block_id, block in self.blocks.items():
            if not block.get('parent'):
                hierarchy.append(self._build_block_tree(block))
        return hierarchy

    def _build_block_tree(self, block):
        """Build a tree structure for a block and its nested blocks"""
        node = {
            'id': block['id'],
            'type': block['type'],
            'subtype': block.get('subtype', 'unknown'),
            'name': block.get('name', 'anonymous'),
            'start_line': block['start_token'].line if block['start_token'] else 0,
            'end_line': block['end_token'].line if block['end_token'] else 0,
            'children': []
        }

        for nested in block.get('nested_blocks', []):
            node['children'].append(self._build_block_tree(nested))

        return node

    def print_structure(self):
        """Print the discovered structure for debugging"""
        print("\n📊 CODE STRUCTURE ANALYSIS:")
        print("=" * 50)

        for block_id, block in self.blocks.items():
            self._print_block(block)

        print("=" * 50)

    def _print_block(self, block, indent=0):
        """Recursively print block information"""
        indent_str = "  " * indent
        block_type = block.get('subtype', block['type'])
        name = block.get('name', 'anonymous')
        start_line = block['start_token'].line if block['start_token'] else '?'
        end_line = block['end_token'].line if block['end_token'] else '?'

        print(f"{indent_str}└── {block_type}: {name} (lines {start_line}-{end_line})")

        for nested in block.get('nested_blocks', []):
            self._print_block(nested, indent + 1)