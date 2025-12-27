import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer  
from zexus.parser.strategy_structural import StructuralAnalyzer

code = """action process_order(order: Order) -> string {
    order.status = "processing"
    sleep(0.1)
    order.status = "completed"
    return "done"
}"""

lexer = Lexer(code)
tokens = lexer.lex()

analyzer = StructuralAnalyzer(tokens)
blocks = analyzer.analyze()

print("Blocks from analyzer:\n")
for block_id, block_info in sorted(blocks.items()):
    print(f"Block {block_id}:")
    print(f"  Subtype: {block_info.get('subtype', 'N/A')}")
    print(f"  Tokens: {[t.value if hasattr(t, 'value') else str(t.type) for t in block_info['tokens']]}")
    print()
