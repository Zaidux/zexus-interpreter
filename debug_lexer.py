# Create debug_lexer.py to see the token stream
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexer import Lexer

def debug_lexer(filename):
    with open(filename, 'r') as f:
        source_code = f.read()
    
    print("=== LEXER TOKEN STREAM ===")
    lexer = Lexer(source_code)
    
    token_count = 0
    while True:
        token = lexer.next_token()
        print(f"Token {token_count}: {token}")
        token_count += 1
        if token.type == 'EOF':
            break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_lexer.py <filename.zx>")
        sys.exit(1)
    debug_lexer(sys.argv[1])
