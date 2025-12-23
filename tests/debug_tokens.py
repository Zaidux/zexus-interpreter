#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.zexus_token import EOF

code = """if x > 3:
    print("yes")
"""

lexer = Lexer(code)
tokens = []
while True:
    tok = lexer.next_token()
    tokens.append(tok)
    print(f"{tok.type:15s} {repr(tok.literal)}")
    if tok.type == EOF:
        break
