import sys
from src.zexus.lexer import Lexer
from src.zexus.parser.parser import UltimateParser

code = open('test_simple_async_assign.zx').read()
lexer = Lexer(code)
parser = UltimateParser(lexer)

# Enable debug
import src.zexus.config as config
config.enable_debug_logs = True

program = parser.parse_program()

print(f"\n[TEST] Parsed {len(program.statements)} statements", file=sys.stderr)
for i, stmt in enumerate(program.statements):
    print(f"[TEST]   {i+1}. {type(stmt).__name__}", file=sys.stderr)
