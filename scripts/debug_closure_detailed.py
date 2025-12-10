#!/usr/bin/env python3
"""Debug closure semantics in detail."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from zexus.compiler import ZexusCompiler
from zexus.vm.vm import VM

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
    # Manually walk through the bytecode with detailed tracing
    vm = VM()
    
    # Let's manually execute the first few steps
    print("\n[Manual Step-Through]")
    
    # Step 1: LOAD_CONST 0 -> push 1
    val = bc.constants[0]
    print(f"1. LOAD_CONST 0 -> push {val}")
    stack = [val]
    print(f"   stack={stack}, vm.env={vm.env}")
    
    # Step 2: STORE_NAME 1 -> pop and store in x
    name = bc.constants[1]
    vm.env[name] = stack.pop()
    print(f"2. STORE_NAME 1 ({name}) -> store {vm.env[name]}")
    print(f"   stack={stack}, vm.env={vm.env}")
    
    # Step 3: STORE_FUNC (3, 2) -> define function f
    name_idx, func_idx = 3, 2
    name = bc.constants[name_idx]
    func_desc = bc.constants[func_idx]
    func_desc_copy = dict(func_desc) if isinstance(func_desc, dict) else {"bytecode": func_desc}
    func_desc_copy["parent_vm"] = vm
    vm.env[name] = func_desc_copy
    print(f"3. STORE_FUNC (3, 2) -> define function f with parent_vm={vm}")
    print(f"   vm.env['f'] has parent_vm={vm.env['f'].get('parent_vm')}")
    print(f"   stack={stack}, vm.env={vm.env}")
    
    # Step 4: LOAD_CONST 4 -> push 2
    val = bc.constants[4]
    print(f"4. LOAD_CONST 4 -> push {val}")
    stack.append(val)
    print(f"   stack={stack}, vm.env={vm.env}")
    
    # Step 5: STORE_NAME 5 -> pop and store in x (reassign to 2)
    name = bc.constants[5]
    vm.env[name] = stack.pop()
    print(f"5. STORE_NAME 5 ({name}) -> store {vm.env[name]}")
    print(f"   stack={stack}, vm.env={vm.env}")
    
    print("\n[Now calling function f]")
    # Step 6: CALL_NAME (6, 0) -> call function f with 0 args
    func_name = bc.constants[6]
    fn = vm.env.get(func_name)
    print(f"6. CALL_NAME (6, 0) -> call f")
    print(f"   func_name={func_name}")
    print(f"   fn={fn}")
    if fn and isinstance(fn, dict) and 'bytecode' in fn:
        func_bc = fn['bytecode']
        parent_env = fn.get('parent_vm', vm)
        print(f"   Function descriptor found")
        print(f"   parent_env={parent_env}")
        print(f"   parent_env.env={parent_env.env}")
        
        # Create inner VM
        print(f"\n   Creating inner_vm...")
        inner_vm = VM(builtins=vm.builtins, env={}, parent_env=parent_env)
        print(f"   inner_vm.env={inner_vm.env}")
        print(f"   inner_vm._parent_env={inner_vm._parent_env}")
        print(f"   inner_vm._parent_env.env={inner_vm._parent_env.env}")
        
        # Now the inner VM should execute the function bytecode
        # The function bytecode is:
        #   0: LOAD_NAME 0 (load const[0] which is 'x')
        #   1: RETURN
        print(f"\n   Inner function bytecode:")
        for i, (op, operand) in enumerate(func_bc.instructions):
            print(f"     {i}: {op} {operand}")
        
        print(f"\n   Executing inner function...")
        # Manually simulate LOAD_NAME in inner_vm
        # LOAD_NAME 0 -> load name at const[0] = 'x'
        name = func_bc.constants[0]
        print(f"     LOAD_NAME 0 -> load name '{name}'")
        
        # Check if inner_vm can resolve 'x' from parent_env
        print(f"     inner_vm.env.get('{name}')={inner_vm.env.get(name)}")
        print(f"     inner_vm._parent_env.env.get('{name}')={inner_vm._parent_env.env.get(name)}")
        
        # Now let's actually execute via the VM
        print(f"\n   Actual execution via VM...")
        result = vm.execute(bc, debug=False)
        print(f"   VM Result: {result}")
