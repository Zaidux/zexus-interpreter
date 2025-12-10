#!/usr/bin/env python3
"""Debug closure semantics with VM instrumentation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zexus.compiler import ZexusCompiler
from zexus.vm import vm as vm_module

# Patch the VM to add debug output
original_invoke = vm_module.VM._invoke_callable_or_funcdesc

async def patched_invoke(self, fn, args, is_constant=False):
    print(f"[INVOKE] fn={fn}, args={args}")
    if isinstance(fn, dict) and 'bytecode' in fn:
        print(f"[INVOKE] This is a function descriptor")
        print(f"[INVOKE] parent_vm={fn.get('parent_vm')}")
    result = await original_invoke(self, fn, args, is_constant)
    print(f"[INVOKE] result={result}")
    return result

vm_module.VM._invoke_callable_or_funcdesc = patched_invoke

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
    result = vm.execute(bc, debug=False)
    print(f"\nVM Result: {result}")
