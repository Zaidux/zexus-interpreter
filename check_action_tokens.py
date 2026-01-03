#!/usr/bin/env python3

from src.zexus.lexer import Lexer

code = """action test(key) {
    data[key] = {"count": 0};
    return data[key];
}"""

lexer = Lexer(code)
tokens = []
while True:
    tok = lexer.next_token()
    tokens.append(tok)
    if tok.type == 'EOF':
        break

print(f"Total tokens: {len(tokens)}")
for i, tok in enumerate(tokens):
    print(f"{i:3}: {tok.type:15} {repr(tok.literal)}")
