#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zexus_ast import Boolean
from evaluator import eval_node
from object import Environment, TRUE, FALSE

# Test the Boolean class directly
print(f"Boolean class: {Boolean}")
print(f"Boolean class module: {Boolean.__module__}")

# Create a test Boolean node
test_bool = Boolean(True)
print(f"Test Boolean node: {test_bool}")
print(f"Test Boolean node type: {type(test_bool)}")
print(f"Test Boolean value: {test_bool.value}")

# Test the comparison
print(f"type(test_bool) == Boolean: {type(test_bool) == Boolean}")

env = Environment()
result = eval_node(test_bool, env)
print(f"Evaluation result: {result}")
