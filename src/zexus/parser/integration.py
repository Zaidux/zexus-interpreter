"""
Parser integration for Phase 1 (Modifiers) and Phase 6 (Metaprogramming).

Applies modifier parsing and macro expansion during AST construction.
"""

from typing import Any, List, Optional
from ..zexus_ast import FunctionLiteral, ActionLiteral
from ..evaluator.integration import get_integration


class ParserIntegration:
    """Integration hooks for parser with modifier and metaprogramming support."""
    
    @staticmethod
    def attach_modifiers(node: Any, modifiers: List[str]) -> Any:
        """Attach modifiers to AST node.
        
        Phase 1: Modifier System
        """
        if not hasattr(node, 'modifiers'):
            node.modifiers = []
        
        if isinstance(modifiers, list):
            node.modifiers.extend(modifiers)
        else:
            node.modifiers.append(modifiers)
        
        return node
    
    @staticmethod
    def parse_modifiers(tokens: List[str]) -> tuple:
        """Parse modifier tokens from token list.
        
        Returns: (modifiers, remaining_tokens)
        """
        modifier_tokens = {
            'PUBLIC', 'PRIVATE', 'SEALED', 'ASYNC', 
            'NATIVE', 'INLINE', 'SECURE', 'PURE'
        }
        
        modifiers = []
        remaining = []
        
        for token in tokens:
            if isinstance(token, str) and token.upper() in modifier_tokens:
                modifiers.append(token.upper())
            else:
                remaining.append(token)
        
        return modifiers, remaining
    
    @staticmethod
    def apply_macro_expansion(ast_node: Any) -> Any:
        """Apply metaprogramming macros to AST.
        
        Phase 6: Metaprogramming
        """
        try:
            integration = get_integration()
            return integration.meta_registry.apply_macros(ast_node)
        except Exception:
            # If metaprogramming fails, return original node
            return ast_node
    
    @staticmethod
    def extract_function_signature(func_node: Any) -> Optional[dict]:
        """Extract function signature information.
        
        Returns: {'name': str, 'params': List[str], 'modifiers': List[str]}
        """
        if isinstance(func_node, (FunctionLiteral, ActionLiteral)):
            params = []
            if hasattr(func_node, 'parameters') and func_node.parameters:
                params = [p.value if hasattr(p, 'value') else str(p) for p in func_node.parameters]
            
            modifiers = getattr(func_node, 'modifiers', [])
            
            return {
                'name': getattr(func_node, 'name', 'anonymous'),
                'params': params,
                'modifiers': modifiers,
                'has_modifiers': len(modifiers) > 0
            }
        
        return None
