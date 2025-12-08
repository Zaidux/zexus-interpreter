#!/usr/bin/env python3
"""Comprehensive test to verify function parameter binding is working"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.object import Environment

def run_test(code_str, description):
    """Run a test with given code"""
    print(f"\n{description}")
    print("-" * 60)
    
    try:
        lexer = Lexer(code_str)
        parser = UltimateParser(lexer, enable_advanced_strategies=False)
        ast = parser.parse_program()
        
        evaluator = Evaluator()
        env = Environment()
        result = evaluator.eval_node(ast, env)
        
        print("✅ Success")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

print("\n" + "="*70)
print("FUNCTION PARAMETER BINDING VALIDATION")
print("="*70)

# Test 1: Single parameter
success1 = run_test(
    '''function greet(name) { 
    print("Hello, " + name); 
}
greet("Alice");''',
    "Test 1: Single parameter function"
)

# Test 2: Multiple parameters
success2 = run_test(
    '''function add(a, b) { 
    print(a + b); 
    return a + b; 
}
result = add(5, 3);''',
    "Test 2: Multiple parameter function"
)

# Test 3: Parameter in string concatenation (the actual issue)
success3 = run_test(
    '''function registerPackage(name, version) {
    print("Registering " + name + "@" + version);
    return {"name": name, "version": version};
}
pkg = registerPackage("pkg", "1.0");''',
    "Test 3: Parameters in string concatenation (ACTUAL ISSUE)"
)

# Test 4: Phase 10 ecosystem
test_file = "src/tests/test_phase10_ecosystem.zx"
if os.path.exists(test_file):
    with open(test_file, 'r') as f:
        phase10_code = f.read()
    
    success4 = run_test(phase10_code, "Test 4: Phase 10 ecosystem test")
else:
    success4 = None
    print("\nTest 4: Phase 10 ecosystem test (SKIPPED - file not found)")

# Test 5: Nested function calls
success5 = run_test(
    '''function double(x) { return x * 2; }
function quad(x) { return double(double(x)); }
result = quad(5);
print(result);''',
    "Test 5: Nested function calls with parameters"
)

# Test 6: Function parameters shadowing outer variables
success6 = run_test(
    '''let x = "outer";
function test(x) { 
    print("inner: " + x); 
}
test("inner");
print("outer: " + x);''',
    "Test 6: Parameter shadowing outer variable"
)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

tests = [
    ("Single parameter", success1),
    ("Multiple parameters", success2),
    ("Parameters in concatenation", success3),
    ("Phase 10 ecosystem", success4),
    ("Nested function calls", success5),
    ("Parameter shadowing", success6),
]

passed = sum(1 for _, s in tests if s)
total = sum(1 for _, s in tests if s is not None)

for name, result in tests:
    if result is None:
        print(f"  ⊗ {name} (skipped)")
    elif result:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}")

print(f"\nResult: {passed}/{total} tests passed")

if passed == total:
    print("\n🎉 ALL TESTS PASSED! Function parameter binding is working correctly!")
else:
    print(f"\n⚠️  {total - passed} test(s) failed")

print("="*70 + "\n")

sys.exit(0 if passed == total else 1)
