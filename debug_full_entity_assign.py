#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.strategy_structural import StructuralAnalyzer
from zexus.zexus_token import EOF

code = """
entity Order {
    id: integer,
    status: string
}

let order = Order(999, "pending")
print("Initial status: " + order.status)
order.status = "completed"
print("After assignment: " + order.status)
"""

lexer = Lexer(code)
tokens = []
while True:
    tok = lexer.next_token()
    tokens.append(tok)
    if tok.type == EOF:
        break

analyzer = StructuralAnalyzer()
blocks = analyzer.analyze(tokens)

print("Blocks from analyzer:")
for bid, info in sorted(analyzer.blocks.items()):
    print(f"\nBlock {bid}:")
    print(f"  Subtype: {info.get('subtype')}")
    print(f"  Tokens: {[t.literal for t in info.get('tokens', [])]}")
