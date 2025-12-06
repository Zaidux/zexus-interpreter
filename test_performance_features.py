#!/usr/bin/env python3
"""
Test Performance Features: NATIVE, GC, INLINE, BUFFER, SIMD
"""

import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.object import Environment

def test_gc_statement():
    """Test garbage collection statement"""
    print("\n=== Testing GC Statement ===")
    
    code = '''
gc "collect";
gc "pause";
gc "resume";
'''
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    if parser.errors:
        print(f"Parser errors: {parser.errors}")
        return False
    
    evaluator = Evaluator()
    env = Environment()
    result = evaluator.eval_node(program, env)
    
    print(f"✓ GC statements evaluated successfully")
    return True

def test_inline_statement():
    """Test inline statement"""
    print("\n=== Testing INLINE Statement ===")
    
    code = '''
action factorial(n) {
  if n <= 1 { return 1; }
  return n * factorial(n - 1);
}

inline factorial;
'''
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    if parser.errors:
        print(f"Parser errors: {parser.errors}")
        return False
    
    evaluator = Evaluator()
    env = Environment()
    result = evaluator.eval_node(program, env)
    
    print(f"✓ INLINE statement evaluated successfully")
    return True

def test_buffer_statement():
    """Test buffer statement"""
    print("\n=== Testing BUFFER Statement ===")
    
    code = '''
buffer my_buf = allocate(100);
buffer my_buf.write(0, [1, 2, 3, 4]);
buffer my_buf.read(0, 4);
'''
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    if parser.errors:
        print(f"Parser errors: {parser.errors}")
        return False
    
    evaluator = Evaluator()
    env = Environment()
    result = evaluator.eval_node(program, env)
    
    print(f"✓ BUFFER statements evaluated successfully")
    return True

def test_simd_statement():
    """Test SIMD statement"""
    print("\n=== Testing SIMD Statement ===")
    
    code = '''
let a = [1, 2, 3, 4];
let b = [5, 6, 7, 8];
simd a + b;
'''
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    if parser.errors:
        print(f"Parser errors: {parser.errors}")
        return False
    
    evaluator = Evaluator()
    env = Environment()
    result = evaluator.eval_node(program, env)
    
    print(f"✓ SIMD statement evaluated successfully")
    return True

def test_native_statement():
    """Test native statement (basic parsing test, requires actual .so file for full execution)"""
    print("\n=== Testing NATIVE Statement (Parse Only) ===")
    
    code = '''
native "libmath.so", "pow"(2, 3);
'''
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    if parser.errors:
        print(f"Parser errors: {parser.errors}")
        return False
    
    print(f"✓ NATIVE statement parsed successfully")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Performance Features Implementation")
    print("=" * 60)
    
    tests = [
        test_gc_statement,
        test_inline_statement,
        test_buffer_statement,
        test_simd_statement,
        test_native_statement,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    sys.exit(0 if all(results) else 1)
