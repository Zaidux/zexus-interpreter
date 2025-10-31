# strategy_structural.py
from .zexus_token import *

class StructuralAnalyzer:
    def __init__(self):
        self.blocks = {}
        self.block_counter = 0
        
    def analyze(self, tokens):
        """First pass: understand the complete structure of the code"""
        print("🔍 [Structural Analysis] Mapping code structure...")
        
        self.blocks = {}
        self.block_counter = 0
        stack = []
        
        for i, token in enumerate(tokens):
            # Skip EOF for structural analysis
            if token.type == EOF:
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
                
            # Track content for current block
            elif stack:
                stack[-1]['tokens'].append(token)
            
            # Track block ends  
            if token.type in [RBRACE, RPAREN, RBRACKET] and stack:
                block = stack.pop()
                block['end_index'] = i
                block['end_token'] = token
                
                # Identify block specifics (function, try-catch, etc.)
                block = self._identify_block_details(block, tokens)
                
                if stack:
                    # Nested block - add to parent
                    stack[-1]['nested_blocks'].append(block)
                else:
                    # Top-level block
                    self.blocks[block['id']] = block
        
        # Handle any unclosed blocks (error recovery)
        self._handle_unclosed_blocks(stack, tokens)
        
        print(f"✅ Found {len(self.blocks)} top-level blocks with {sum(len(b['nested_blocks']) for b in self.blocks.values())} nested blocks")
        return self.blocks
    
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