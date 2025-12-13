import sys
import os

# Add the src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from zexus.lexer import Lexer
from zexus.zexus_token import MODULE, IDENT

code = 'module connection'
lexer = Lexer(code)

print(f"Lexing: '{code}'")
token = lexer.next_token()
while token.type != 'EOF':
    print(f"  {token.type}: '{token.literal}'")
    token = lexer.next_token()
