# test_map_only.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser import UltimateParser

def test_map_only():
    """Test ONLY the map literal parsing without advanced strategies"""
    print("=== Testing Map Literal (Simple Parser) ===")
    
    # Test with advanced parsing disabled to see the raw parser behavior
    code = 'let data = {"name": "test", "value": 123}'
    
    print(f"Code: {code}")
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer, enable_advanced_strategies=False)  # Disable advanced parsing
    
    program = parser.parse_program()
    
    print(f"Statements: {len(program.statements)}")
    print(f"Errors: {len(parser.errors)}")
    
    if parser.errors:
        for error in parser.errors:
            print(f"❌ {error}")
    
    if program.statements:
        stmt = program.statements[0]
        print(f"Statement type: {type(stmt).__name__}")
        
        if hasattr(stmt, 'value'):
            value_type = type(stmt.value).__name__
            print(f"Value type: {value_type}")
            
            if value_type == 'MapLiteral':
                print("✅ SUCCESS: Map literal parsed correctly!")
                print(f"Pairs: {len(stmt.value.pairs)}")
                for key, value in stmt.value.pairs:
                    print(f"  {key}: {value}")
            else:
                print(f"❌ FAILED: Expected MapLiteral, got {value_type}")
                print(f"Value: {stmt.value}")

if __name__ == "__main__":
    test_map_only()
