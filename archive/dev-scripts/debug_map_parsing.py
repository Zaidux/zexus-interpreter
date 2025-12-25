# debug_map_parsing.py (FIXED)
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser import UltimateParser

def debug_map_parsing():
    """Debug exactly what's happening with map parsing"""
    code = 'let data = {"name": "test", "value": 123}'
    
    print("=== DEBUGGING MAP PARSING ===")
    print(f"Code: {code}")
    
    # Test with advanced parsing
    print("\n--- ADVANCED PARSER ---")
    lexer = Lexer(code)
    
    # Fixed: Convert tokens properly
    tokens = []
    while True:
        token = lexer.next_token()
        tokens.append(token)
        if token.type == 'EOF':
            break
            
    print("Tokens:", [(t.type, t.literal) for t in tokens if t.type != 'EOF'])
    
    lexer = Lexer(code)  # Reset lexer
    parser = UltimateParser(lexer, enable_advanced_strategies=True)
    program = parser.parse_program()
    
    print(f"Statements: {len(program.statements)}")
    print(f"Errors: {len(parser.errors)}")
    
    if parser.errors:
        print("ERRORS:")
        for error in parser.errors:
            print(f"  - {error}")
    
    if program.statements:
        stmt = program.statements[0]
        print(f"Statement type: {type(stmt).__name__}")
        if hasattr(stmt, 'value'):
            print(f"Value type: {type(stmt.value).__name__}")
            print(f"Value: {stmt.value}")
            if hasattr(stmt.value, 'pairs'):
                print(f"Map pairs: {stmt.value.pairs}")

if __name__ == "__main__":
    debug_map_parsing()