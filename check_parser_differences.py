#!/usr/bin/env python3
# check_parser_differences.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_parsers():
    print("=== Parser Comparison ===")
    
    # Test code that fails in compiler
    test_cases = [
        'print(string(42));',  # with semicolon
        'print(string(42))',   # without semicolon  
        'let x = 10 / 0',      # division (for try-catch)
        'let m = {a: 1, b: 2}' # object literal
    ]
    
    for code in test_cases:
        print(f"\n--- Testing: {repr(code)} ---")
        
        # Test with interpreter parser
        try:
            from zexus.lexer import Lexer as ILexer
            from zexus.parser import Parser as IParser
            
            ilexer = ILexer(code)
            iparser = IParser(ilexer, 'universal', enable_advanced_strategies=True)
            iprogram = iparser.parse_program()
            print(f"✅ Interpreter: {len(iprogram.statements)} statements")
            if iparser.errors:
                print(f"   Interpreter errors: {iparser.errors}")
        except Exception as e:
            print(f"❌ Interpreter failed: {e}")
        
        # Test with compiler parser
        try:
            from zexus.compiler.lexer import Lexer as CLexer
            from zexus.compiler.parser import parser as cparser
            
            clexer = CLexer(code)
            cprogram = cparser.parse(clexer)
            print(f"✅ Compiler: {len(cprogram.statements)} statements")
        except Exception as e:
            print(f"❌ Compiler failed: {e}")

if __name__ == "__main__":
    check_parsers()
