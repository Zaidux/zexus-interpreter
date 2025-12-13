# test_parser_debug.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexer import Lexer
from parser import Parser
from zexus_ast import *

def test_assignment_parsing():
    source = "x = 10"
    lexer = Lexer(source)
    parser = Parser(lexer)
    
    print("=== Testing assignment parsing ===")
    print(f"Source: '{source}'")
    
    # Parse as expression statement
    stmt = parser.parse_expression_statement()
    
    if parser.errors:
        print("PARSER ERRORS:")
        for error in parser.errors:
            print(f"  {error}")
    else:
        print("SUCCESS: No parser errors")
        if stmt and stmt.expression:
            print(f"Parsed expression type: {type(stmt.expression).__name__}")
            if isinstance(stmt.expression, AssignmentExpression):
                print(f"  Left: {stmt.expression.name.value}")
                print(f"  Right: {stmt.expression.value.value}")
            else:
                print(f"  Expression: {stmt.expression}")

if __name__ == "__main__":
    test_assignment_parsing()
