#!/usr/bin/env python3
# compare_lexers.py

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer as InterpreterLexer
from zexus.compiler.lexer import Lexer as CompilerLexer

def compare_lexers(source_code, description):
    print(f"\n=== {description} ===")
    print(f"Code: {repr(source_code)}")
    
    print("\n--- INTERPRETER LEXER ---")
    il = InterpreterLexer(source_code)
    while True:
        token = il.next_token()
        print(f"  {token.type}: {repr(token.literal)} (line {token.line}, col {token.column})")
        if token.type == "EOF":
            break
    
    print("\n--- COMPILER LEXER ---")
    try:
        cl = CompilerLexer(source_code)
        while True:
            token = cl.next_token()
            print(f"  {token.type}: {repr(token.literal)} (line {token.line}, col {token.column})")
            if token.type == "EOF":
                break
    except Exception as e:
        print(f"  ERROR: {e}")

# Test cases
test_cases = [
    ('print(1)', 'Simple print'),
    ('let x = 2', 'Variable assignment'),
    ('string(3)', 'String function'),
    ('// comment\nprint(1)', 'With comment'),
    ('print("hello")', 'String literal'),
]

for code, desc in test_cases:
    compare_lexers(code, desc)
