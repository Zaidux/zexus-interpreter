#!/usr/bin/env python3  
# check_compiler_parser.py

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_compiler_components():
    print("=== Testing Compiler Components ===")
    
    # Test if we can import compiler modules
    try:
        from zexus.compiler.lexer import Lexer
        from zexus.compiler.parser import Parser
        print("✅ Compiler modules imported successfully")
        
        # Test lexer
        code = 'print(1)'
        lexer = Lexer(code)
        token = lexer.next_token()
        print(f"✅ Lexer first token: {token.type} = {token.literal}")
        
        # Test parser
        parser = Parser(lexer)
        program = parser.parse_program()
        print(f"✅ Parser parsed {len(program.statements)} statements")
        
    except Exception as e:
        print(f"❌ Compiler import/execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_compiler_components()
