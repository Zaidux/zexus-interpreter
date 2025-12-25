#!/usr/bin/env python3
"""Inspect the bytecode for the closure test to understand function descriptor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zexus.compiler import ZexusCompiler

src = '''
let x = 1
action f ( ) {
    return x
}
let x = 2
let r = f ( )
print ( r )
'''

compiler = ZexusCompiler(src, enable_optimizations=False)
bc = compiler.compile()

if compiler.errors:
    print('Compiler errors:', compiler.errors)
else:
    # Print bytecode constants
    print('=== Bytecode Constants ===')
    for i, const in enumerate(bc.constants):
        print(f'const[{i}]: {repr(const)}')
        if isinstance(const, dict) and 'bytecode' in const:
            print(f'  -> This is a function descriptor')
            print(f'     params: {const.get("params")}')
            print(f'     is_async: {const.get("is_async")}')
            inner_bc = const['bytecode']
            print(f'     Inner bytecode constants: {inner_bc.constants}')
            print(f'     Inner bytecode instructions: ')
            for j, (op, operand) in enumerate(inner_bc.instructions):
                print(f'       {j}: {op} {operand}')

    print('\n=== Bytecode Instructions ===')
    for i, (op, operand) in enumerate(bc.instructions):
        print(f'{i}: {op} {operand}')
