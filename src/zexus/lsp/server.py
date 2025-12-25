#!/usr/bin/env python3
"""
Zexus Language Server
Implements LSP for Zexus language providing IntelliSense and other features.
"""

import logging
import sys
from typing import List, Optional

# Configure logging first before any imports that might fail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

try:
    from pygls.server import LanguageServer
    from pygls.lsp.methods import (
        TEXT_DOCUMENT_DID_OPEN,
        TEXT_DOCUMENT_DID_CHANGE,
        TEXT_DOCUMENT_DID_SAVE,
        TEXT_DOCUMENT_COMPLETION,
        TEXT_DOCUMENT_HOVER,
        TEXT_DOCUMENT_DEFINITION,
        TEXT_DOCUMENT_REFERENCES,
        TEXT_DOCUMENT_DOCUMENT_SYMBOL,
        TEXT_DOCUMENT_FORMATTING,
        TEXT_DOCUMENT_SIGNATURE_HELP,
    )
    from pygls.lsp.types import (
        CompletionItem,
        CompletionItemKind,
        CompletionList,
        CompletionParams,
        DidOpenTextDocumentParams,
        DidChangeTextDocumentParams,
        DidSaveTextDocumentParams,
        Hover,
        HoverParams,
        Location,
        MarkupContent,
        MarkupKind,
        Position,
        Range,
        TextDocumentPositionParams,
        DocumentSymbol,
        DocumentSymbolParams,
        SymbolKind,
        DocumentFormattingParams,
        TextEdit,
        SignatureHelp,
        SignatureInformation,
        ParameterInformation,
        SignatureHelpParams,
    )
    PYGLS_AVAILABLE = True
except ImportError:
    PYGLS_AVAILABLE = False
    logger.error("pygls not installed. Install with: pip install pygls")

# Import Zexus modules - these should always be available if zexus is installed
try:
    from zexus.lexer import Lexer
    from zexus.parser import Parser
    from zexus.evaluator.core import Evaluator
    from .completion_provider import CompletionProvider
    from .symbol_provider import SymbolProvider
    from .hover_provider import HoverProvider
    from .definition_provider import DefinitionProvider
    ZEXUS_AVAILABLE = True
except ImportError as e:
    ZEXUS_AVAILABLE = False
    logger.error(f"Zexus modules not available: {e}")
    logger.error("Make sure Zexus is properly installed: pip install -e .")


if PYGLS_AVAILABLE:
    if not ZEXUS_AVAILABLE:
        logger.error("Error: Zexus modules not available. LSP server cannot start.")
        raise RuntimeError("Zexus modules not available. LSP server cannot start.")
    
    class ZexusLanguageServer(LanguageServer):
        """Zexus Language Server implementation."""

        def __init__(self):
            super().__init__('zexus-language-server', 'v1.5.0')
            self.completion_provider = CompletionProvider()
            self.symbol_provider = SymbolProvider()
            self.hover_provider = HoverProvider()
            self.definition_provider = DefinitionProvider()
            self.documents = {}  # Store parsed documents


    server = ZexusLanguageServer()


    @server.feature(TEXT_DOCUMENT_DID_OPEN)
    async def did_open(ls: ZexusLanguageServer, params: DidOpenTextDocumentParams):
        """Handle document open event."""
        logger.info(f"Document opened: {params.text_document.uri}")
        uri = params.text_document.uri
        text = params.text_document.text
        
        # Parse the document
        try:
            lexer = Lexer(text)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            ast = parser.parse()
            ls.documents[uri] = {
                'text': text,
                'ast': ast,
                'tokens': tokens
            }
            logger.info(f"Document parsed successfully: {uri}")
        except Exception as e:
            logger.error(f"Error parsing document {uri}: {e}")
            ls.documents[uri] = {
                'text': text,
                'ast': None,
                'tokens': []
            }


    @server.feature(TEXT_DOCUMENT_DID_CHANGE)
    async def did_change(ls: ZexusLanguageServer, params: DidChangeTextDocumentParams):
        """Handle document change event."""
        uri = params.text_document.uri
        changes = params.content_changes
        
        if changes:
            text = changes[0].text
            # Re-parse the document
            try:
                lexer = Lexer(text)
                tokens = lexer.tokenize()
                parser = Parser(tokens)
                ast = parser.parse()
                ls.documents[uri] = {
                    'text': text,
                    'ast': ast,
                    'tokens': tokens
                }
            except Exception as e:
                logger.error(f"Error parsing document {uri}: {e}")
                ls.documents[uri] = {
                    'text': text,
                    'ast': None,
                    'tokens': []
                }


    @server.feature(TEXT_DOCUMENT_DID_SAVE)
    async def did_save(ls: ZexusLanguageServer, params: DidSaveTextDocumentParams):
        """Handle document save event."""
        logger.info(f"Document saved: {params.text_document.uri}")


    @server.feature(TEXT_DOCUMENT_COMPLETION)
    async def completions(ls: ZexusLanguageServer, params: CompletionParams) -> CompletionList:
        """Provide completion items."""
        uri = params.text_document.uri
        position = params.position
        
        doc_info = ls.documents.get(uri, {})
        text = doc_info.get('text', '')
        
        items = ls.completion_provider.get_completions(text, position, doc_info)
        return CompletionList(is_incomplete=False, items=items)


    @server.feature(TEXT_DOCUMENT_HOVER)
    async def hover(ls: ZexusLanguageServer, params: HoverParams) -> Optional[Hover]:
        """Provide hover information."""
        uri = params.text_document.uri
        position = params.position
        
        doc_info = ls.documents.get(uri, {})
        
        return ls.hover_provider.get_hover(position, doc_info)


    @server.feature(TEXT_DOCUMENT_DEFINITION)
    async def definition(ls: ZexusLanguageServer, params: TextDocumentPositionParams) -> Optional[List[Location]]:
        """Provide go-to-definition."""
        uri = params.text_document.uri
        position = params.position
        
        doc_info = ls.documents.get(uri, {})
        
        return ls.definition_provider.get_definition(uri, position, doc_info)


    @server.feature(TEXT_DOCUMENT_DOCUMENT_SYMBOL)
    async def document_symbols(ls: ZexusLanguageServer, params: DocumentSymbolParams) -> List[DocumentSymbol]:
        """Provide document symbols for outline view."""
        uri = params.text_document.uri
        doc_info = ls.documents.get(uri, {})
        
        return ls.symbol_provider.get_symbols(doc_info)


    @server.feature(TEXT_DOCUMENT_FORMATTING)
    async def formatting(ls: ZexusLanguageServer, params: DocumentFormattingParams) -> List[TextEdit]:
        """Format document."""
        # TODO: Implement code formatting
        return []


    @server.feature(TEXT_DOCUMENT_SIGNATURE_HELP)
    async def signature_help(ls: ZexusLanguageServer, params: SignatureHelpParams) -> Optional[SignatureHelp]:
        """Provide signature help for function calls."""
        uri = params.text_document.uri
        position = params.position
        
        doc_info = ls.documents.get(uri, {})
        text = doc_info.get('text', '')
        
        # Get current line
        lines = text.split('\n')
        if position.line >= len(lines):
            return None
        
        current_line = lines[position.line][:position.character]
        
        # Simple signature help for built-in functions
        # TODO: Make this more sophisticated by parsing AST
        builtins_signatures = {
            'print': SignatureInformation(
                label='print(value)',
                documentation='Print value to console',
                parameters=[ParameterInformation(label='value', documentation='Value to print')]
            ),
            'len': SignatureInformation(
                label='len(collection)',
                documentation='Get length of collection',
                parameters=[ParameterInformation(label='collection', documentation='Collection to measure')]
            ),
            # Add more built-ins...
        }
        
        # Find which function is being called
        for func_name, sig in builtins_signatures.items():
            if func_name + '(' in current_line:
                return SignatureHelp(
                    signatures=[sig],
                    active_signature=0,
                    active_parameter=0
                )
        
        return None


    def main():
        """Start the language server."""
        logger.info("Starting Zexus Language Server...")
        server.start_io()


    if __name__ == '__main__':
        main()
else:
    def main():
        print("Error: pygls is not installed. Please install it with: pip install pygls", file=sys.stderr)
        sys.exit(1)
