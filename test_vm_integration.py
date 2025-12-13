#!/usr/bin/env python3
"""
Test VM Integration with Evaluator

This script tests the new VM integration in the evaluator.
"""
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser import UltimateParser
from zexus.evaluator import evaluate
from zexus.object import Environment

def test_basic_arithmetic():
    """Test basic arithmetic operations via VM"""
    print("=" * 60)
    print("Test 1: Basic Arithmetic")
    print("=" * 60)
    
    code = """
    let x = 10 + 5;
    let y = x * 2;
    let z = y - 3;
    z
    """
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    env = Environment()
    
    result = evaluate(program, env, use_vm=True)
    print(f"Result: {result}")
    print(f"Expected: 27")
    print()

def test_loop_execution():
    """Test while loop execution via VM"""
    print("=" * 60)
    print("Test 2: While Loop")
    print("=" * 60)
    
    code = """
    let count = 0;
    let sum = 0;
    while (count < 5) {
        sum = sum + count;
        count = count + 1;
    }
    sum
    """
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    env = Environment()
    
    result = evaluate(program, env, use_vm=True)
    print(f"Result: {result}")
    print(f"Expected: 10 (0+1+2+3+4)")
    print()

def test_function_definition():
    """Test function definition and execution"""
    print("=" * 60)
    print("Test 3: Function Definition")
    print("=" * 60)
    
    code = """
    action add(a, b) {
        return a + b;
    }
    
    let result = add(5, 3);
    result
    """
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    env = Environment()
    
    result = evaluate(program, env, use_vm=True)
    print(f"Result: {result}")
    print(f"Expected: 8")
    print()

def test_conditionals():
    """Test if-else conditionals"""
    print("=" * 60)
    print("Test 4: Conditionals")
    print("=" * 60)
    
    code = """
    let x = 10;
    let result = 0;
    
    if (x > 5) {
        result = 100;
    } else {
        result = 50;
    }
    
    result
    """
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    env = Environment()
    
    result = evaluate(program, env, use_vm=True)
    print(f"Result: {result}")
    print(f"Expected: 100")
    print()

def test_list_operations():
    """Test list creation and operations"""
    print("=" * 60)
    print("Test 5: List Operations")
    print("=" * 60)
    
    code = """
    let numbers = [1, 2, 3, 4, 5];
    numbers
    """
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    env = Environment()
    
    result = evaluate(program, env, use_vm=True)
    print(f"Result: {result}")
    print(f"Expected: [1, 2, 3, 4, 5]")
    print()

def test_vm_stats():
    """Test VM statistics tracking"""
    print("=" * 60)
    print("Test 6: VM Statistics")
    print("=" * 60)
    
    from zexus.evaluator.core import Evaluator
    
    code = """
    let x = 1 + 2;
    let y = 3 + 4;
    x + y
    """
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    env = Environment()
    
    evaluator = Evaluator(use_vm=True)
    result = evaluator.eval_with_vm_support(program, env)
    
    stats = evaluator.get_vm_stats()
    print(f"Result: {result}")
    print(f"VM Stats: {stats}")
    print()

def test_bytecode_disassembly():
    """Test bytecode generation and disassembly"""
    print("=" * 60)
    print("Test 7: Bytecode Disassembly")
    print("=" * 60)
    
    from zexus.evaluator.bytecode_compiler import EvaluatorBytecodeCompiler
    
    code = """
    let x = 10;
    let y = 20;
    x + y
    """
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    compiler = EvaluatorBytecodeCompiler()
    bytecode = compiler.compile(program, optimize=True)
    
    if bytecode:
        print(bytecode.disassemble())
    else:
        print(f"Compilation errors: {compiler.errors}")
    print()

def test_fallback_behavior():
    """Test fallback to direct evaluation when VM can't handle code"""
    print("=" * 60)
    print("Test 8: Fallback Behavior")
    print("=" * 60)
    
    # Use a complex feature that might not be in bytecode compiler yet
    code = """
    let x = 10;
    x
    """
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    env = Environment()
    
    # Both should work
    result_with_vm = evaluate(program, env, use_vm=True)
    
    # Reset environment
    env = Environment()
    result_without_vm = evaluate(program, env, use_vm=False)
    
    print(f"Result with VM: {result_with_vm}")
    print(f"Result without VM: {result_without_vm}")
    print(f"Results match: {str(result_with_vm) == str(result_without_vm)}")
    print()

def main():
    print("\n" + "=" * 60)
    print("ZEXUS VM INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_basic_arithmetic()
        test_loop_execution()
        test_function_definition()
        test_conditionals()
        test_list_operations()
        test_vm_stats()
        test_bytecode_disassembly()
        test_fallback_behavior()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
