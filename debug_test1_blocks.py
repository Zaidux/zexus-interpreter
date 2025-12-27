import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.strategy_structural import StructuralAnalyzer

code = """action test1() -> string {
    let order = Order(1, "pending")
    order.status = "completed"
    persist_set("test1", order)
    return "done"
}"""

lexer = Lexer(code)
tokens = []
while True:
    tok = lexer.next_token()
    tokens.append(tok)
    if tok.type == 'EOF':
        break

analyzer = StructuralAnalyzer()
blocks = analyzer.analyze(tokens[:-1])  # Exclude EOF

print("Blocks from structural analyzer:\n")
for block_id, block_info in sorted(blocks.items()):
    print(f"Block {block_id}:")
    print(f"  Type: {block_info.get('type', 'N/A')}")
    print(f"  Subtype: {block_info.get('subtype', 'N/A')}")
    toks = block_info['tokens']
    print(f"  Tokens ({len(toks)}): {[t.value if hasattr(t, 'value') else t.literal if hasattr(t, 'literal') else str(t.type) for t in toks]}")
    print()
