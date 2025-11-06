# test_map_fix.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser import UltimateParser

def test_map_literal():
    """Test the map literal parsing fix"""
    print("=== Testing Map Literal Fix ===")
    
    # Test cases
    test_cases = [
        'let data = {"name": "test", "value": 123}',
        'let empty = {}',
        'let user = {name: "john", age: 25}',
    ]
    
    for i, code in enumerate(test_cases, 1):
        print(f"\n🔬 Test {i}: {code}")
        lexer = Lexer(code)
        parser = UltimateParser(lexer)
        program = parser.parse_program()
        
        print(f"  Statements: {len(program.statements)}")
        print(f"  Errors: {len(parser.errors)}")
        
        if parser.errors:
            for error in parser.errors:
                print(f"  ❌ {error}")
        
        if program.statements:
            stmt = program.statements[0]
            print(f"  Statement: {type(stmt).__name__}")
            
            if hasattr(stmt, 'value'):
                value_type = type(stmt.value).__name__
                print(f"  Value type: {value_type}")
                
                if value_type == 'MapLiteral':
                    print(f"  ✅ SUCCESS: Map literal parsed correctly!")
                    print(f"  Pairs: {len(stmt.value.pairs)}")
                    for key, value in stmt.value.pairs:
                        print(f"    {key}: {value}")
                else:
                    print(f"  ❌ FAILED: Expected MapLiteral, got {value_type}")

if __name__ == "__main__":
    test_map_literal()
