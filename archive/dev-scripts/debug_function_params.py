#!/usr/bin/env python3
"""Debug test for function parameter binding"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.object import Environment

code = '''function test(x) {
    print("x is: " + x);
    return x;
}

result = test("hello");
print("Result: " + result);'''

print("Testing function parameter binding...\n")
print("Code:")
print(code)
print("\n" + "="*60 + "\n")

try:
    lexer = Lexer(code)
    parser = UltimateParser(lexer, enable_advanced_strategies=False)
    ast = parser.parse_program()
    
    print(f"✅ Parsing successful: {len(ast.statements)} statements\n")
    
    # Debug: Show parsed statements
    for i, stmt in enumerate(ast.statements):
        print(f"Statement {i}: {type(stmt).__name__}")
        if type(stmt).__name__ == 'FunctionStatement':
            print(f"  - Name: {stmt.name.value}")
            print(f"  - Parameters: {[p.value if hasattr(p, 'value') else str(p) for p in stmt.parameters]}")
    
    print("\n" + "="*60 + "\n")
    
    evaluator = Evaluator()
    env = Environment()
    result = evaluator.eval_node(ast, env)
    
    print(f"\n✅ Evaluation successful!")
    print(f"Result: {result}")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
