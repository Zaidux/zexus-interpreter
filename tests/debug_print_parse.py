#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.strategy_context import ContextStackParser

code = 'print("label:", x)'

lexer = Lexer(code)
parser = ContextStackParser(lexer)
program = parser.parse_program()

if parser.errors:
    print("ERRORS:")
    for err in parser.errors:
        print(f"  {err}")
else:
    print("AST:")
    for stmt in program.statements:
        print(f"  {stmt}")
        if hasattr(stmt, 'value'):
            print(f"    value: {stmt.value}")
            print(f"    value type: {type(stmt.value).__name__}")
