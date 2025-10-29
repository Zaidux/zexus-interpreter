# ~/zexus-interpreter/src/zexus/__init__.py
"""
Zexus Programming Language
A declarative, intent-based programming language for modern applications.
"""

__version__ = "0.1.0"
__author__ = "Ziver Labs"
__email__ = "dev@ziver.io"

from .lexer import Lexer
from .parser import Parser
from .evaluator import eval_node, Environment
from .object import (
    Object, Integer, Float, String, Boolean, Null, 
    List, Map, Action, Builtin, ReturnValue
)

__all__ = [
    "Lexer", "Parser", "eval_node", "Environment",
    "Object", "Integer", "Float", "String", "Boolean", 
    "Null", "List", "Map", "Action", "Builtin", "ReturnValue"
]
