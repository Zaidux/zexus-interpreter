"""Definition provider for Zexus LSP — go-to-definition support."""

from typing import List, Dict, Any, Optional

try:
    from pygls.lsp.types import Position, Location, Range
    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False


# AST node types that introduce named definitions
_DEFINITION_NODE_TYPES = (
    'ActionStatement', 'FunctionStatement', 'PureFunctionStatement',
    'LetStatement', 'ConstStatement',
    'EntityStatement', 'ContractStatement', 'DataStatement',
    'EnumStatement', 'InterfaceStatement', 'ProtocolStatement',
    'TypeAliasStatement', 'ModuleStatement', 'PackageStatement',
    'ScreenStatement', 'ComponentStatement', 'MiddlewareStatement',
    'PatternStatement', 'StreamStatement', 'ModifierDeclaration',
)


def _get_name(node) -> Optional[str]:
    """Extract the string name from an AST definition node."""
    name = getattr(node, 'name', None)
    if name is None:
        return None
    if hasattr(name, 'value'):
        return str(name.value)
    return str(name)


def _collect_definitions(node, defs: Dict[str, list]):
    """Walk the AST recursively and collect all named definitions.

    ``defs`` maps name → list of AST nodes (same name may be defined
    more than once, e.g. multiple ``let`` re-assignments).
    """
    if node is None:
        return
    node_type = type(node).__name__

    if node_type in _DEFINITION_NODE_TYPES:
        name = _get_name(node)
        if name:
            defs.setdefault(name, []).append(node)

    # Recurse into child nodes that may contain more definitions
    for attr in ('statements', 'body', 'consequence', 'alternative',
                 'methods', 'properties', 'cases', 'block', 'try_block',
                 'catch_block', 'finally_block', 'members'):
        child = getattr(node, attr, None)
        if child is None:
            continue
        if isinstance(child, list):
            for c in child:
                _collect_definitions(c, defs)
        else:
            # BlockStatement, Program, etc.
            stmts = getattr(child, 'statements', None)
            if isinstance(stmts, list):
                for c in stmts:
                    _collect_definitions(c, defs)
            else:
                _collect_definitions(child, defs)


def _find_token_position(tokens, name: str, definition_keywords=None):
    """Scan the token list for the *first* definition of ``name``.

    We look for a keyword token (``action``, ``let``, ``const``, ``entity``,
    ``contract``, ``data``, ``enum``, ``interface``, ``protocol``, ``type``,
    ``module``, ``package``, ``screen``, ``component``, ``middleware``,
    ``pattern``, ``stream``, ``pure``) followed (possibly after a type
    annotation) by an ``IDENT`` token whose ``literal`` matches ``name``.
    Returns ``(line, column)`` (0-based) or ``None``.
    """
    if definition_keywords is None:
        definition_keywords = {
            'action', 'let', 'const', 'entity', 'contract', 'data',
            'enum', 'interface', 'protocol', 'type', 'module', 'package',
            'screen', 'component', 'middleware', 'pattern', 'stream',
            'pure', 'fn', 'function', 'def', 'class', 'struct', 'modifier',
        }

    want_ident = False
    for tok in tokens:
        lit = getattr(tok, 'literal', '')
        ttype = getattr(tok, 'type', '')

        if lit in definition_keywords:
            want_ident = True
            continue

        if want_ident and ttype == 'IDENT' and lit == name:
            line = getattr(tok, 'line', 0)
            col = getattr(tok, 'column', 0)
            # Token lines are 1-based; LSP uses 0-based
            return (max(0, line - 1), max(0, col))

        if want_ident and ttype not in ('IDENT', 'COLON', ':', 'LBRACKET', 'RBRACKET',
                                          'LT', 'GT', 'COMMA', 'STRING'):
            # Not part of a type annotation — reset
            want_ident = False

    return None


def _word_at_position(text: str, line: int, character: int) -> Optional[str]:
    """Extract the identifier under the cursor."""
    lines = text.split('\n')
    if line >= len(lines):
        return None
    row = lines[line]
    if character >= len(row):
        return None

    # Walk left
    start = character
    while start > 0 and (row[start - 1].isalnum() or row[start - 1] == '_'):
        start -= 1
    # Walk right
    end = character
    while end < len(row) and (row[end].isalnum() or row[end] == '_'):
        end += 1

    word = row[start:end]
    return word if word else None


class DefinitionProvider:
    """Provides go-to-definition for Zexus code."""

    def get_definition(self, uri: str, position, doc_info: Dict[str, Any]) -> Optional[List]:
        """Return a list of ``Location`` objects for the definition of the
        symbol under the cursor, or ``None`` if not found.
        """
        if not PYGLS_AVAILABLE:
            return None

        text = doc_info.get('text', '')
        ast = doc_info.get('ast')
        tokens = doc_info.get('tokens', [])

        if not text:
            return None

        line = position.line
        character = position.character
        word = _word_at_position(text, line, character)
        if not word:
            return None

        # 1. Collect all definitions from the AST
        defs: Dict[str, list] = {}
        if ast:
            _collect_definitions(ast, defs)

        if word not in defs:
            return None

        # 2. Find the first definition position in the token stream
        pos = _find_token_position(tokens, word)
        if pos is None:
            return None

        def_line, def_col = pos
        location = Location(
            uri=uri,
            range=Range(
                start=Position(line=def_line, character=def_col),
                end=Position(line=def_line, character=def_col + len(word)),
            ),
        )
        return [location]
