#!/usr/bin/env python3
from src.zexus.lexer import Lexer

code = """action test() {
    data[key] = {"count": 0};
}
"""

lexer = Lexer(code)
tokens = []
while True:
    tok = lexer.next_token()
    tokens.append(tok)
    print(f'{len(tokens):3d}. {tok.type:15s} {repr(tok.literal):20s}')
    if tok.type == 'EOF':
        break
