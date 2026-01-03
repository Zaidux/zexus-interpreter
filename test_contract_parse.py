#!/usr/bin/env python3

from src.zexus.lexer import Lexer
from src.zexus.parser.parser import Parser

code = """
contract Store {
    state data = {};
    
    action test(key) {
        data[key] = {"count": 0};
        return data[key];
    }
}
"""

lexer = Lexer(code)
parser = Parser(lexer)
program = parser.parse_program()

print(f"Errors: {parser.errors}")
print(f"\nProgram has {len(program.statements)} statements")

contract_stmt = program.statements[0]
print(f"\nContract: {contract_stmt.name.value}")
print(f"Actions: {[a.name.value for a in contract_stmt.actions]}")

action = contract_stmt.actions[0]
print(f"\nAction: {action.name.value}")
print(f"Body type: {type(action.body).__name__}")
print(f"Body has {len(action.body.statements)} statements")

for i, stmt in enumerate(action.body.statements):
    print(f"\nStatement {i}: {type(stmt).__name__}")
    print(f"  Full statement: {stmt}")
    if hasattr(stmt, 'expression'):
        print(f"  Expression: {type(stmt.expression).__name__}")
        print(f"  Expression value: {stmt.expression}")
        if hasattr(stmt.expression, 'name'):
            print(f"    name: {type(stmt.expression.name).__name__} = {stmt.expression.name}")
        if hasattr(stmt.expression, 'value'):
            print(f"    value type: {type(stmt.expression.value).__name__}")
            print(f"    value: {stmt.expression.value}")
