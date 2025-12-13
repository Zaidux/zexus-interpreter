# test_advanced_vs_traditional.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser import UltimateParser

def test_both_parsers():
    """Test both advanced and traditional parsing"""
    code = 'let data = {"name": "test", "value": 123}'
    
    print("=== Testing Advanced Parser ===")
    lexer1 = Lexer(code)
    parser1 = UltimateParser(lexer1, enable_advanced_strategies=True)
    program1 = parser1.parse_program()
    
    print(f"Advanced - Statements: {len(program1.statements)}, Errors: {len(parser1.errors)}")
    if program1.statements:
        stmt = program1.statements[0]
        if hasattr(stmt, 'value'):
            print(f"Advanced - Value type: {type(stmt.value).__name__}")
    
    print("\n=== Testing Traditional Parser ===")
    lexer2 = Lexer(code)
    parser2 = UltimateParser(lexer2, enable_advanced_strategies=False)
    program2 = parser2.parse_program()
    
    print(f"Traditional - Statements: {len(program2.statements)}, Errors: {len(parser2.errors)}")
    if program2.statements:
        stmt = program2.statements[0]
        if hasattr(stmt, 'value'):
            print(f"Traditional - Value type: {type(stmt.value).__name__}")

if __name__ == "__main__":
    test_both_parsers()
