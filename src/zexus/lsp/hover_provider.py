"""Hover provider for Zexus LSP."""

from typing import Dict, Any, Optional
try:
    from pygls.lsp.types import Hover, MarkupContent, MarkupKind, Position, Range
    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False


class HoverProvider:
    """Provides hover information for Zexus code."""

    def get_hover(self, position: Position, doc_info: Dict[str, Any]) -> Optional[Hover]:
        """Get hover information for the given position."""
        if not PYGLS_AVAILABLE:
            return None
        
        text = doc_info.get('text', '')
        
        # Get word at position
        lines = text.split('\n')
        if position.line >= len(lines):
            return None
        
        line = lines[position.line]
        if position.character >= len(line):
            return None
        
        # Find word boundaries
        start = position.character
        end = position.character
        
        while start > 0 and (line[start - 1].isalnum() or line[start - 1] == '_'):
            start -= 1
        
        while end < len(line) and (line[end].isalnum() or line[end] == '_'):
            end += 1
        
        word = line[start:end]
        
        if not word:
            return None
        
        # Provide documentation for keywords and built-ins
        docs = {
            'let': 'Declare a mutable variable',
            'const': 'Declare an immutable constant',
            'action': 'Define a function/action',
            'contract': 'Define a smart contract',
            'entity': 'Define a data structure',
            'verify': 'Runtime verification with custom logic',
            'protect': 'Apply security policy to function',
            'print': 'Print value to console',
            'len': 'Get length of collection',
            # Add more...
        }
        
        if word in docs:
            return Hover(
                contents=MarkupContent(
                    kind=MarkupKind.Markdown,
                    value=f'**{word}**\n\n{docs[word]}'
                ),
                range=Range(
                    start=Position(line=position.line, character=start),
                    end=Position(line=position.line, character=end)
                )
            )
        
        return None
