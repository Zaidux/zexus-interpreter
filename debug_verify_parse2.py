import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.strategy_context import StructuralAnalyzer, ContextStackParser

code = """
let username = "a"
verify {
    username != "",
    len(username) >= 3
}, "Failed"
"""

lexer = Lexer(code)
analyzer = StructuralAnalyzer(lexer)
blocks = analyzer.analyze()
parser = ContextStackParser(analyzer)

print("Blocks found:")
for i, block in enumerate(blocks):
    print(f"\nBlock {i}:")
    print(f"  Type: {block.get('type')}")
    print(f"  Tokens: {[t.literal for t in block.get('tokens', [])][:20]}")

statements = parser._parse_block_statements(blocks)
print(f"\n\nParsed {len(statements)} statements")
for stmt in statements:
    print(f"\nStatement type: {type(stmt).__name__}")
    if hasattr(stmt, 'condition'):
        print(f"  condition: {stmt.condition}")
    if hasattr(stmt, 'message'):
        print(f"  message: {stmt.message}")
    if hasattr(stmt, 'logic_block'):
        print(f"  logic_block: {stmt.logic_block}")
        if stmt.logic_block:
            print(f"    logic_block type: {type(stmt.logic_block).__name__}")
            if hasattr(stmt.logic_block, 'statements'):
                print(f"    statements count: {len(stmt.logic_block.statements)}")
                for s in stmt.logic_block.statements:
                    print(f"      - {type(s).__name__}: {s}")
