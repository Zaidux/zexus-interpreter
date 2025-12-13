# test_basic.py
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_lexer():
    """Test the lexer directly"""
    from zexus.lexer import Lexer
    
    code = 'let data = {"name": "test", "value": 123}'
    lexer = Lexer(code)
    
    print("✅ Lexer tokens:")
    while True:
        token = lexer.next_token()
        print(f"  {token.type}: '{token.literal}'")
        if token.type == 'EOF':
            break

def test_parser():
    """Test the parser with map literals"""
    from zexus.lexer import Lexer
    from zexus.parser import UltimateParser
    
    code = 'let data = {"name": "test", "value": 123}'
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    print(f"✅ Parser test:")
    print(f"  Statements: {len(program.statements)}")
    print(f"  Errors: {len(parser.errors)}")
    
    if program.statements:
        stmt = program.statements[0]
        print(f"  Statement type: {type(stmt).__name__}")
        if hasattr(stmt, 'value') and hasattr(stmt.value, 'pairs'):
            print(f"  Map pairs: {len(stmt.value.pairs)}")
            for key, value in stmt.value.pairs:
                print(f"    {key}: {value}")

if __name__ == "__main__":
    print("=== Testing Zexus Phase 1 ===")
    test_lexer()
    print()
    test_parser()