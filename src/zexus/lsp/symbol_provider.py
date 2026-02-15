"""Symbol provider for Zexus LSP — document outline and symbol search."""

from typing import List, Dict, Any, Optional

try:
    from pygls.lsp.types import (DocumentSymbol, SymbolKind, Range, Position)
    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False
    # Define minimal stubs when pygls not available
    class SymbolKind:
        Function = 12
        Variable = 13
        Constant = 14
        Class = 5
        Struct = 23
        Enum = 10
        Interface = 11
        Module = 2
        Package = 4
        TypeParameter = 26


# Map AST node type name → (SymbolKind, detail label)
_SYMBOL_MAP = {
    'ActionStatement':       (SymbolKind.Function,      'action'),
    'FunctionStatement':     (SymbolKind.Function,      'function'),
    'PureFunctionStatement': (SymbolKind.Function,      'pure function'),
    'LetStatement':          (SymbolKind.Variable,      'let'),
    'ConstStatement':        (SymbolKind.Constant,      'const'),
    'EntityStatement':       (SymbolKind.Class,         'entity'),
    'ContractStatement':     (SymbolKind.Class,         'contract'),
    'DataStatement':         (SymbolKind.Struct,        'data'),
    'EnumStatement':         (SymbolKind.Enum,          'enum'),
    'InterfaceStatement':    (SymbolKind.Interface,     'interface'),
    'ProtocolStatement':     (SymbolKind.Interface,     'protocol'),
    'TypeAliasStatement':    (SymbolKind.TypeParameter, 'type'),
    'ModuleStatement':       (SymbolKind.Module,        'module'),
    'PackageStatement':      (SymbolKind.Package,       'package'),
    'ScreenStatement':       (SymbolKind.Function,      'screen'),
    'ComponentStatement':    (SymbolKind.Function,      'component'),
    'MiddlewareStatement':   (SymbolKind.Function,      'middleware'),
    'PatternStatement':      (SymbolKind.Function,      'pattern'),
    'StreamStatement':       (SymbolKind.Variable,      'stream'),
    'ModifierDeclaration':   (SymbolKind.Function,      'modifier'),
}


def _get_name(node) -> Optional[str]:
    """Extract the string name from an AST definition node."""
    name = getattr(node, 'name', None)
    if name is None:
        return None
    if hasattr(name, 'value'):
        return str(name.value)
    return str(name)


def _find_name_line(tokens, name: str, start_line: int = 0) -> int:
    """Return the 0-based line number where ``name`` appears as an IDENT
    in the token stream, starting from ``start_line``.  Returns
    ``start_line`` if not found.
    """
    for tok in tokens:
        ttype = getattr(tok, 'type', '')
        lit = getattr(tok, 'literal', '')
        line = getattr(tok, 'line', 0)
        if ttype == 'IDENT' and lit == name and (line - 1) >= start_line:
            return max(0, line - 1)
    return start_line


def _make_range(line: int, col: int = 0, end_line: Optional[int] = None, length: int = 1):
    """Build a pygls Range.  Lines/cols are 0-based."""
    if end_line is None:
        end_line = line
    return Range(
        start=Position(line=line, character=col),
        end=Position(line=end_line, character=col + length),
    )


def _node_to_symbol(node, tokens, text_lines) -> Optional[DocumentSymbol]:
    """Convert an AST definition node into a ``DocumentSymbol``.

    Returns ``None`` if the node does not map to a symbol.
    """
    if not PYGLS_AVAILABLE:
        return None

    node_type = type(node).__name__
    entry = _SYMBOL_MAP.get(node_type)
    if entry is None:
        return None

    kind, detail = entry
    name = _get_name(node)
    if not name:
        return None

    # Locate the name in the token stream
    line = _find_name_line(tokens, name)
    symbol_range = _make_range(line, length=len(name))
    selection_range = symbol_range

    # Build children for container types (entity, contract, enum, data)
    children = []

    if node_type == 'EntityStatement':
        # Properties
        for prop in (getattr(node, 'properties', None) or []):
            prop_name = _get_name(prop)
            if prop_name:
                pline = _find_name_line(tokens, prop_name, line)
                children.append(DocumentSymbol(
                    name=prop_name,
                    kind=SymbolKind.Variable,
                    range=_make_range(pline, length=len(prop_name)),
                    selection_range=_make_range(pline, length=len(prop_name)),
                    detail='property',
                ))
        # Methods
        for method in (getattr(node, 'methods', None) or []):
            child_sym = _node_to_symbol(method, tokens, text_lines)
            if child_sym:
                children.append(child_sym)

    elif node_type == 'ContractStatement':
        body = getattr(node, 'body', None)
        stmts = getattr(body, 'statements', None) if body else None
        if isinstance(stmts, list):
            for stmt in stmts:
                child_sym = _node_to_symbol(stmt, tokens, text_lines)
                if child_sym:
                    children.append(child_sym)

    elif node_type == 'EnumStatement':
        for member in (getattr(node, 'members', None) or []):
            m_name = _get_name(member) or getattr(member, 'value', None)
            if isinstance(m_name, str):
                pass
            elif hasattr(m_name, 'value'):
                m_name = m_name.value
            if m_name:
                mline = _find_name_line(tokens, str(m_name), line)
                children.append(DocumentSymbol(
                    name=str(m_name),
                    kind=SymbolKind.Constant,
                    range=_make_range(mline, length=len(str(m_name))),
                    selection_range=_make_range(mline, length=len(str(m_name))),
                    detail='member',
                ))

    elif node_type == 'DataStatement':
        for field in (getattr(node, 'fields', None) or []):
            f_name = _get_name(field)
            if f_name:
                fline = _find_name_line(tokens, f_name, line)
                children.append(DocumentSymbol(
                    name=f_name,
                    kind=SymbolKind.Variable,
                    range=_make_range(fline, length=len(f_name)),
                    selection_range=_make_range(fline, length=len(f_name)),
                    detail='field',
                ))

    return DocumentSymbol(
        name=name,
        kind=kind,
        range=symbol_range,
        selection_range=selection_range,
        detail=detail,
        children=children if children else None,
    )


class SymbolProvider:
    """Provides document symbols for outline view."""

    def get_symbols(self, doc_info: Dict[str, Any]) -> List:
        """Get document symbols from AST."""
        if not PYGLS_AVAILABLE:
            return []

        symbols = []
        ast = doc_info.get('ast')
        tokens = doc_info.get('tokens', [])
        text = doc_info.get('text', '')
        text_lines = text.split('\n') if text else []

        if not ast:
            return symbols

        # Walk top-level statements
        stmts = getattr(ast, 'statements', [])
        if not isinstance(stmts, list):
            return symbols

        for stmt in stmts:
            sym = _node_to_symbol(stmt, tokens, text_lines)
            if sym:
                symbols.append(sym)

        return symbols
