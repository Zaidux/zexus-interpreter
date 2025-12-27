#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer  
from zexus.parser import MainParser

code = """
verify {
    username != "",
    len(username) >= 3
}, "Failed"
"""

lexer = Lexer(code)
parser = MainParser(lexer)
program = parser.parse_program()

print(f"Parsed {len(program.statements)} statements")
for stmt in program.statements:
    print(f"  - {type(stmt).__name__}")
    print(f"    {stmt}")
