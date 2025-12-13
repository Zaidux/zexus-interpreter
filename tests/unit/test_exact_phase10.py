#!/usr/bin/env python3
"""Test that exactly mimics the phase10 test execution"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.object import Environment

# Exact code from phase10 test
code = '''
// Test 1: Package registry
function registerPackage(name, version, metadata) {
    print("Registering package: " + name + "@" + version);
    return {
        "name": name,
        "version": version,
        "installed": true,
        "metadata": metadata
    };
}

// Test execution
print("Phase 10: Ecosystem Test Suite");
print("===============================");

// Test package registration
print("\\nTesting package registration:");
let pkg1 = registerPackage("json-utils", "1.0.0", {"author": "user", "license": "MIT"});
print("Package registered: " + pkg1["name"]);

let pkg2 = registerPackage("crypto-lib", "2.1.0", {"author": "dev", "license": "Apache"});
print("Package registered: " + pkg2["name"]);
'''

print("Testing exact phase10 structure...\n")
print("="*70)

try:
    lexer = Lexer(code)
    parser = UltimateParser(lexer, enable_advanced_strategies=False)
    ast = parser.parse_program()
    
    print(f"✅ Parsing successful: {len(ast.statements)} statements\n")
    
    evaluator = Evaluator()
    env = Environment()
    result = evaluator.eval_node(ast, env)
    
    print(f"\n✅ Evaluation successful!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    print("="*70)
    import traceback
    traceback.print_exc()
    sys.exit(1)
