"""Zexus Language Server Protocol implementation."""

# Import only if pygls is available
try:
    from .server import ZexusLanguageServer
    __all__ = ['ZexusLanguageServer']
except ImportError:
    # pygls not available, LSP features won't work
    __all__ = []
