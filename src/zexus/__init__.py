# src/zexus/__init__.py
"""
Zexus Programming Language
A declarative, intent-based programming language for modern applications.
"""

__version__ = "1.5.0"
__author__ = "Ziver Labs"
__email__ = "ziverofficial567@gmail.com"

from .lexer import Lexer
from .parser import Parser
# UPDATED: Import from new evaluator structure
from .evaluator import evaluate
from .object import (
    Environment, Object, Integer, Float, String, Boolean, Null, 
    List, Map, Action, Builtin, ReturnValue, EmbeddedCode
)

# For backward compatibility, you can alias eval_node to evaluate if needed
# but better to update callers to use evaluate
eval_node = evaluate  # Alias for backward compatibility

__all__ = [
    "Lexer", "Parser", "evaluate", "eval_node", "Environment",  # UPDATED
    "Object", "Integer", "Float", "String", "Boolean", 
    "Null", "List", "Map", "Action", "Builtin", "ReturnValue", "EmbeddedCode"
]