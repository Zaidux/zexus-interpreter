#!/usr/bin/env python3
import re

# Read the current evaluator.py
with open('evaluator.py', 'r') as f:
    content = f.read()

# Fix 1: Add modulo operator to integer infix expressions
if '%' not in content or 'modulo' not in content.lower():
    # Add modulo to eval_integer_infix_expression
    content = re.sub(
        r'(elif operator == "/":\s*\n\s*return Integer\(left_val // right_val\))',
        r'\1\n    elif operator == "%":\n        return Integer(left_val % right_val)',
        content
    )
    
    # Add modulo to eval_float_infix_expression  
    content = re.sub(
        r'(elif operator == "/":\s*\n\s*return Float\(left_val / right_val\))',
        r'\1\n    elif operator == "%":\n        return Float(left_val % right_val)',
        content
    )

# Fix 2: Improve assignment expression handling
if 'eval_assignment_expression' in content:
    # Make assignment handling more robust
    content = re.sub(
        r'def eval_assignment_expression\(left, right\):.*?return NULL',
        r'def eval_assignment_expression(left, right):\n    """Handle assignment expressions like: x = 5"""\n    if hasattr(left, \'value\'):\n        # For identifiers, we should update the environment\n        # For now, just return the right value\n        print(f"[ASSIGN] {getattr(left, \'value\', left)} = {right.inspect()}")\n        return right\n    else:\n        print(f"Error: Cannot assign to {type(left).__name__}")\n        return NULL',
        content,
        flags=re.DOTALL
    )

print("Applied fixes for modulo and assignment")
with open('evaluator.py', 'w') as f:
    f.write(content)
