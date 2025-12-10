#!/usr/bin/env python3
"""Debug closure semantics with bytecode execution instrumentation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zexus.compiler import ZexusCompiler
from zexus.vm import vm as vm_module

src = '''
let x = 1
action f ( ) {
    return x
}
let x = 2
let r = f ( )
print ( r )
'''

print("=== COMPILER PATH ===")
compiler = ZexusCompiler(src, enable_optimizations=False)
bc = compiler.compile()

if compiler.errors:
    print('Compiler errors:', compiler.errors)
else:
    vm = vm_module.VM()
    result = vm.execute(bc, debug=True)
    print(f"\nVM Result: {result}")
