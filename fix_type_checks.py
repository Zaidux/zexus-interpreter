#!/usr/bin/env python3
"""Fix node_type == checks to use isinstance() instead"""

import re

# Read the file
with open('src/zexus/evaluator/core.py', 'r') as f:
    content = f.read()

# Replace patterns:
# 1. "elif node_type == zexus_ast.ClassName:" -> "elif isinstance(node, zexus_ast.ClassName):"
# 2. "if node_type == zexus_ast.ClassName:" -> "if isinstance(node, zexus_ast.ClassName):"

# Pattern: (if|elif) node_type == (zexus_ast\.\w+):
pattern = r'(if|elif) node_type == (zexus_ast\.\w+):'
replacement = r'\1 isinstance(node, \2):'

content = re.sub(pattern, replacement, content)

# Also fix: "or node_type.__name__ == 'FloatLiteral'"
content = content.replace("or node_type.__name__ == 'FloatLiteral'", "or isinstance(node, zexus_ast.FloatLiteral)")

# Also fix: "if type(k) == zexus_ast.Identifier:"
content = content.replace("if type(k) == zexus_ast.Identifier:", "if isinstance(k, zexus_ast.Identifier):")

# Write back
with open('src/zexus/evaluator/core.py', 'w') as f:
    f.write(content)

print("Fixed all type checks to use isinstance()")
