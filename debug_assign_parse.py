#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.strategy_structural import StructuralAnalyzer
from zexus.zexus_token import EOF

code = """
order.status = "completed"
"""

lexer = Lexer(code)
tokens = []
while True:
    tok = lexer.next_token()
    tokens.append(tok)
    if tok.type == EOF:
        break

print("Tokens:")
for i, tok in enumerate(tokens):
    print(f"  {i}: {tok.type:15s} '{tok.literal}'")

analyzer = StructuralAnalyzer()
blocks = analyzer.analyze(tokens)

print("\nBlocks from analyzer:")
for bid, info in analyzer.blocks.items():
    print(f"\nBlock {bid}:")
    print(f"  Type: {info.get('type')}")
    print(f"  Subtype: {info.get('subtype')}")
    print(f"  Tokens: {[t.literal for t in info.get('tokens', [])]}")
