#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexer import Lexer
from parser import Parser

def debug_assignment(filename):
    with open(filename, 'r') as f:
        source_code = f.read()
    
    lexer = Lexer(source_code)
    parser = Parser(lexer)
    
    print(f"=== PARSING {filename} ===")
    program = parser.parse_program()
    
    if parser.errors:
        print("PARSER ERRORS:")
        for error in parser.errors:
            print(f"  {error}")
    else:
        print("PARSING SUCCESSFUL!")
        # Print the AST to see what's being generated
        for i, stmt in enumerate(program.statements):
            print(f"Statement {i}: {stmt}")
    
    return program

if __name__ == "__main__":
    debug_assignment("test_grand_finale.zx")
