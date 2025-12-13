import sys
import os
sys.path.insert(0, 'src')

from zexus.lexer import Lexer
from zexus.parser import Parser

code = """
try {
    let x = 10 / 0
} catch((error)) {
    print("Error: " + string(error))
}
"""

print("=== Testing Parser ===")
print("Code:")
print(code)

lexer = Lexer(code)
tokens = lexer.tokenize()
print(f"Tokens: {[(t.type, t.value) for t in tokens]}")

parser = Parser(tokens)
ast = parser.parse()
print(f"Parse successful: {ast is not None}")
if ast:
    print(f"Number of statements: {len(ast.statements)}")
    for i, stmt in enumerate(ast.statements):
        print(f"Statement {i}: {type(stmt).__name__}")
