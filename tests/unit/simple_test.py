import sys
import os

# Add the src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import modules directly without going through __init__.py
from zexus.lexer import Lexer

# Test the lexer
code = 'let x = 5'
lexer = Lexer(code)

print("✅ Simple lexer test:")
token = lexer.next_token()
while token.type != 'EOF':
    print(f"  {token.type}: '{token.literal}'")
    token = lexer.next_token()
