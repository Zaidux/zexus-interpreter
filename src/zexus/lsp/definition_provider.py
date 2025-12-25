"""Definition provider for Zexus LSP."""

from typing import List, Dict, Any, Optional
try:
    from pygls.lsp.types import Position
    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False


class DefinitionProvider:
    """Provides go-to-definition for Zexus code."""

    def get_definition(self, uri: str, position: Position, doc_info: Dict[str, Any]) -> Optional[List]:
        """Get definition location for symbol at position."""
        if not PYGLS_AVAILABLE:
            return None
        
        # TODO: Implement go-to-definition by analyzing AST
        # For now, return None
        
        return None
