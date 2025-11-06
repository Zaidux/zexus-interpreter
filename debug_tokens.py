import sys
import os
sys.path.insert(0, 'src')

from zexus.lexer import Lexer

code = """
try {
    let x = 10 / 0
} catch((error)) {
    print("Error: " + string(error))
}
"""

lexer = Lexer(code)
tokens = lexer.tokenize_advanced()

print("=== Token Analysis ===")
for i, token in enumerate(tokens):
    print(f"{i:2d}: {token.type:15} = '{token.value}'")
