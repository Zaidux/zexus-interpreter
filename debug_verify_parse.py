import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.strategy_context import ContextStackParser

code = """
verify {
    username != "",
    len(username) >= 3
}, "Failed"
"""

lexer = Lexer(code)
parser = ContextStackParser(lexer)
result = parser.parse_program()

print("AST:")
print(result)
print("\nStatements:")
for stmt in result.statements:
    print(f"Type: {type(stmt).__name__}")
    print(f"  condition: {stmt.condition}")
    print(f"  message: {stmt.message}")
    print(f"  logic_block: {stmt.logic_block}")
    if stmt.logic_block:
        print(f"  logic_block type: {type(stmt.logic_block).__name__}")
        print(f"  logic_block.statements: {stmt.logic_block.statements}")
