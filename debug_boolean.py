#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexer import Lexer
from parser import Parser
from zexus_ast import *

def debug_boolean():
    source_code = "if (true && false): print 'test'"
    lexer = Lexer(source_code)
    parser = Parser(lexer)
    program = parser.parse_program()
    
    # Walk through the AST and print node types
    def walk_node(node, depth=0):
        indent = "  " * depth
        print(f"{indent}{type(node).__name__}")
        if hasattr(node, '__dict__'):
            for key, value in node.__dict__.items():
                if isinstance(value, (list, Node)):
                    print(f"{indent}  {key}:")
                    if isinstance(value, list):
                        for item in value:
                            walk_node(item, depth + 2)
                    else:
                        walk_node(value, depth + 2)
    
    for stmt in program.statements:
        walk_node(stmt)

if __name__ == "__main__":
    debug_boolean()
