"""Symbol provider for Zexus LSP."""

from typing import List, Dict, Any
try:
    from pygls.lsp.types import (DocumentSymbol, SymbolKind, Range, Position)
    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False
    # Define minimal stubs when pygls not available
    class SymbolKind:
        pass


class SymbolProvider:
    """Provides document symbols for outline view."""

    def get_symbols(self, doc_info: Dict[str, Any]) -> List:
        """Get document symbols from AST."""
        if not PYGLS_AVAILABLE:
            return []
        
        symbols = []
        ast = doc_info.get('ast')
        
        if not ast:
            return symbols
        
        # TODO: Walk AST and extract symbols
        # For now, return empty list
        
        return symbols
