#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from both places
from zexus_ast import Boolean as AST_Boolean
import evaluator

# Check what Boolean is in evaluator context
print("Boolean in zexus_ast:", AST_Boolean)

# Check if evaluator has its own Boolean
if hasattr(evaluator, 'Boolean'):
    print("Boolean in evaluator:", evaluator.Boolean)
else:
    print("No Boolean in evaluator module")

# Test instance
test_bool = AST_Boolean(True)
print("Test instance type:", type(test_bool))
print("Type matches AST_Boolean:", type(test_bool) == AST_Boolean)

# Check what's in eval_node
import inspect
source = inspect.getsource(evaluator.eval_node)
boolean_line = [line for line in source.split('\n') if 'Boolean' in line and 'elif' in line]
print("Boolean check in eval_node:")
for line in boolean_line:
    print("  ", line.strip())
