#!/usr/bin/env python3
"""
Test VM usage in the Zexus interpreter.
This script verifies that the VM is actually being used for appropriate code patterns.
"""

import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator, VM_AVAILABLE
from zexus.object import Environment

def test_vm_usage():
    print("=" * 60)
    print("VM USAGE VERIFICATION TEST")
    print("=" * 60)
    print(f"\n📊 VM Available: {VM_AVAILABLE}")
    
    if not VM_AVAILABLE:
        print("❌ VM is NOT available - cannot run test")
        print("   The VM module failed to import")
        return False
    
    print("✅ VM module successfully imported\n")
    
    # Read test file
    with open('/workspaces/zexus-interpreter/test_vm_usage.zx', 'r') as f:
        code = f.read()
    
    # Lex and parse
    print("🔍 Lexing and parsing code...")
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    if parser.errors:
        print(f"❌ Parser errors: {parser.errors}")
        return False
    
    print(f"✅ Parsed {len(program.statements)} top-level statements\n")
    
    # Create evaluator with VM enabled
    print("🚀 Creating evaluator with VM enabled...")
    evaluator = Evaluator(use_vm=True)
    
    print(f"   VM_AVAILABLE: {VM_AVAILABLE}")
    print(f"   use_vm param: True")
    print(f"   evaluator.use_vm: {evaluator.use_vm}")
    
    if not evaluator.use_vm:
        print("❌ Evaluator did not enable VM")
        print(f"   Check: use_vm=True AND VM_AVAILABLE={VM_AVAILABLE}")
        return False
    
    print("✅ Evaluator VM enabled\n")
    
    # Check VM instance
    if evaluator.vm_instance:
        print(f"✅ VM instance created: {type(evaluator.vm_instance).__name__}")
        print(f"   JIT enabled: {getattr(evaluator.vm_instance, 'use_jit', False)}")
        print(f"   JIT threshold: {getattr(evaluator.vm_instance, 'jit_threshold', 'N/A')}")
        print(f"   Optimization level: {getattr(evaluator.vm_instance, 'optimization_level', 'N/A')}")
    else:
        print("⚠️  VM instance not yet created (lazy initialization)")
    
    print("\n" + "=" * 60)
    print("EXECUTING CODE")
    print("=" * 60 + "\n")
    
    # Execute
    env = Environment()
    
    print("🔍 Checking should_use_vm for program...")
    from zexus.evaluator.bytecode_compiler import should_use_vm_for_node
    should_vm = should_use_vm_for_node(program)
    print(f"   should_use_vm_for_node(program): {should_vm}")
    print(f"   Program has {len(program.statements)} statements")
    print(f"   Threshold: >10 statements")
    
    result = evaluator.eval_with_vm_support(program, env, debug_mode=False)
    
    # Show results
    print("\n" + "=" * 60)
    print("VM EXECUTION STATISTICS")
    print("=" * 60)
    
    stats = evaluator.get_vm_stats()
    print(f"\n📈 Statistics:")
    print(f"   Bytecode Compilations: {stats['bytecode_compiles']}")
    print(f"   VM Executions:         {stats['vm_executions']}")
    print(f"   VM Fallbacks:          {stats['vm_fallbacks']}")
    print(f"   Direct Evaluations:    {stats['direct_evals']}")
    
    total_executions = stats['vm_executions'] + stats['direct_evals']
    if total_executions > 0:
        vm_percentage = (stats['vm_executions'] / total_executions) * 100
        print(f"\n   VM Usage: {vm_percentage:.1f}%")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60 + "\n")
    
    if stats['vm_executions'] > 0:
        print("✅ VM IS BEING USED!")
        print(f"   The VM executed {stats['vm_executions']} code blocks")
        print(f"   This includes loops, large functions, and complex programs")
        
        if stats['bytecode_compiles'] > 0:
            print(f"\n✅ BYTECODE COMPILATION WORKING")
            print(f"   {stats['bytecode_compiles']} successful compilations")
    else:
        print("❌ VM NOT BEING USED")
        print("   All code was executed via direct interpretation")
        
        if stats['vm_fallbacks'] > 0:
            print(f"\n⚠️  {stats['vm_fallbacks']} VM compilation failures")
            print("   Check bytecode compiler implementation")
    
    # Conditions check
    print("\n" + "=" * 60)
    print("VM USAGE CONDITIONS")
    print("=" * 60 + "\n")
    
    print("The VM is used when:")
    print("  ✓ use_vm=True in Evaluator (currently: {})".format(evaluator.use_vm))
    print("  ✓ VM module available (currently: {})".format(VM_AVAILABLE))
    print("  ✓ Node meets heuristics:")
    print("    - WhileStatement (always)")
    print("    - ForEachStatement (always)")
    print("    - ActionStatement with >5 body statements")
    print("    - Program with >10 statements")
    print("  ✓ Bytecode compiler can handle the node")
    
    print("\n" + "=" * 60)
    
    return stats['vm_executions'] > 0

if __name__ == "__main__":
    success = test_vm_usage()
    sys.exit(0 if success else 1)
