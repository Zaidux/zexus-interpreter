#!/usr/bin/env python3
"""Test phase10 ecosystem with parameter binding"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.object import Environment

test_file = "src/tests/test_phase10_ecosystem.zx"

if not os.path.exists(test_file):
    print(f"❌ Test file not found: {test_file}")
    sys.exit(1)

with open(test_file, 'r') as f:
    code = f.read()

print("Testing phase10 ecosystem with function parameters...\n")
print(f"File: {test_file}")
print(f"Size: {len(code)} bytes\n")
print("="*70 + "\n")

try:
    lexer = Lexer(code)
    parser = UltimateParser(lexer, enable_advanced_strategies=False)
    ast = parser.parse_program()
    
    print(f"✅ Parsing successful: {len(ast.statements)} statements\n")
    
    # Debug: Show function declarations
    func_count = 0
    for stmt in ast.statements:
        if type(stmt).__name__ == 'FunctionStatement':
            func_count += 1
            params = [p.value if hasattr(p, 'value') else str(p) for p in stmt.parameters]
            print(f"Function {func_count}: {stmt.name.value}({', '.join(params)})")
    
    print("\n" + "="*70 + "\n")
    
    print("Evaluating...\n")
    evaluator = Evaluator()
    env = Environment()
    result = evaluator.eval_node(ast, env)
    
    print(f"\n✅ Evaluation successful!")
    print(f"Result: {result}")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
