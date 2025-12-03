# src/zexus/parser/__init__.py - UPDATE
"""
Parser module for Zexus language.
"""

# Use absolute imports inside the package
try:
    from .parser import Parser, UltimateParser
    from .strategy_context import StrategyContext
    from .strategy_structural import StructuralStrategy
except ImportError as e:
    print(f"Warning: Could not import parser modules: {e}")
    # Define placeholders
    class Parser: pass
    class UltimateParser: pass
    class StrategyContext: pass
    class StructuralStrategy: pass

__all__ = ["Parser", "UltimateParser", "StrategyContext", "StructuralStrategy"]