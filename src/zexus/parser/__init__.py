# src/zexus/parser/__init__.py
"""
Parser module for Zexus language.
"""

# Use absolute imports inside the package
try:
    from .parser import Parser, UltimateParser
    # Import the actual class names defined in the strategy files
    from .strategy_context import ContextStackParser
    from .strategy_structural import StructuralAnalyzer
except ImportError as e:
    print(f"Warning: Could not import parser modules: {e}")
    # Define placeholders
    class Parser: pass
    class UltimateParser: pass
    class ContextStackParser: pass
    class StructuralAnalyzer: pass

# Aliases for backward compatibility or external references
StrategyContext = ContextStackParser
StructuralStrategy = StructuralAnalyzer

__all__ = [
    "Parser", 
    "UltimateParser", 
    "ContextStackParser", 
    "StructuralAnalyzer",
    "StrategyContext",
    "StructuralStrategy"
]