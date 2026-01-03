#!/usr/bin/env python3

from src.zexus.lexer import Lexer
from src.zexus.parser.parser import Parser

code = """
data[key] = {"count": 0};
"""

lexer = Lexer(code)
parser = Parser(lexer)
program = parser.parse_program()

print(f"Errors: {parser.errors}")
print(f"Program statements: {len(program.statements)}")
for i, stmt in enumerate(program.statements):
    print(f"\nStatement {i}: {type(stmt).__name__}")
    print(f"  {stmt}")
    if hasattr(stmt, 'expression'):
        print(f"  Expression: {type(stmt.expression).__name__}")
        print(f"  {stmt.expression}")
        if hasattr(stmt.expression, 'name'):
            print(f"    name: {type(stmt.expression.name).__name__} = {stmt.expression.name}")
        if hasattr(stmt.expression, 'value'):
            print(f"    value: {type(stmt.expression.value).__name__} = {stmt.expression.value}")
