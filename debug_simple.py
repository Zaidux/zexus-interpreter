#!/usr/bin/env python3
from lexer import Lexer

code = '''
action test_if():
    if (5 > 3):
        return "Condition works"
'''

print("=== TOKEN DEBUG ===")
lexer = Lexer(code)

while True:
    token = lexer.next_token()
    print(f"Token: {token.type:10} -> '{token.literal}'")
    if token.type == 'EOF':
        break
