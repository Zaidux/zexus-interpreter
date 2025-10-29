#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import zexus_ast
import evaluator

print("zexus_ast file:", zexus_ast.__file__)
print("evaluator file:", evaluator.__file__)

# Check if they're the same file
print("Same file?", zexus_ast.__file__ == evaluator.__file__)

# Check imports
print("zexus_ast.Boolean:", zexus_ast.Boolean)
print("evaluator Boolean import:", getattr(evaluator, 'Boolean', 'NOT IMPORTED'))
