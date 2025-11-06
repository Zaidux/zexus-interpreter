# test_hybrid.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.hybrid_orchestrator import orchestrator
from zexus.object import Environment

def test_hybrid_system():
    """Test the hybrid interpreter/compiler system"""
    print("=== Testing Hybrid System ===")
    
    env = Environment()
    
    # Test cases for different execution modes
    test_cases = [
        ('interpreter', 'let x = 5; print(x)'),
        ('compiler', 'let y = 10; print(y + 2)'),
        ('auto', 'let z = 3 * 4; print(z)'),
    ]
    
    for mode, code in test_cases:
        print(f"\n🔧 Testing {mode} mode:")
        print(f"  Code: {code}")
        
        try:
            result = orchestrator.execute(code, environment=env, mode=mode)
            print(f"  ✅ Execution successful")
            if result and hasattr(result, 'inspect') and result.inspect() != 'null':
                print(f"  Result: {result.inspect()}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n📊 Final Statistics:")
    print(f"  Interpreter uses: {orchestrator.interpreter_used}")
    print(f"  Compiler uses: {orchestrator.compiler_used}")
    print(f"  Fallbacks: {orchestrator.fallbacks}")

if __name__ == "__main__":
    test_hybrid_system()
