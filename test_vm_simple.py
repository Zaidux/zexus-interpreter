#!/usr/bin/env python3
"""Minimal test to debug VM type conversion"""

import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.object import Environment

# Simple code with just assignments and print
code = """
let a = 1;
let b = 2;
let c = 3;
let d = 4;
let e = 5;
let f = 6;
let g = 7;
let h = 8;
let i = 9;
let j = 10;
let k = 11;
print("Done");
"""

print("Parsing code...")
lexer = Lexer(code)
parser = UltimateParser(lexer)
program = parser.parse_program()

if parser.errors:
    print(f"Parser errors: {parser.errors}")
    sys.exit(1)

print(f"Program statements: {len(program.statements)}")

# Create evaluator with VM
evaluator = Evaluator(use_vm=False)  # Start with VM disabled
env = Environment()

print("\n=== Running with direct interpretation ===")
result1 = evaluator.eval_node(program, env)
print(f"Result: {result1}")

# Now try with VM
env2 = Environment()
evaluator2 = Evaluator(use_vm=True)

print("\n=== Running with VM ===")
result2 = evaluator2.eval_with_vm_support(program, env2, debug_mode=True)
print(f"Result: {result2}")

print(f"\n=== VM Stats ===")
print(evaluator2.get_vm_stats())
