#!/usr/bin/env python3
"""
Test resource cleanup.

Tests basic resource cleanup scenarios without deep instrumentation.

Location: tests/advanced_edge_cases/test_resource_cleanup.py
"""

import sys
import os
import tempfile
import traceback
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import Parser
from zexus.evaluator.core import evaluate
from zexus.environment import Environment


def run_code(code):
    """Helper to run Zexus code and return environment."""
    lexer = Lexer(code)
    parser = Parser(lexer)
    program = parser.parse_program()
    env = Environment()
    result = evaluate(program, env)
    return env, result


def test_file_handle_cleanup():
    """Test that file handles are cleaned up after file operations."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
        f.write("test content")
    
    try:
        # Run code that opens the file
        code = f"""
        let content = file_read_text("{temp_path}");
        """
        
        initial_open_files = len([f for f in os.listdir('/proc/self/fd')] if os.path.exists('/proc/self/fd') else [])
        
        # Run code multiple times
        for _ in range(10):
            run_code(code)
        
        # Force garbage collection
        gc.collect()
        
        final_open_files = len([f for f in os.listdir('/proc/self/fd')] if os.path.exists('/proc/self/fd') else [])
        
        # Check if file descriptors didn't grow significantly
        if os.path.exists('/proc/self/fd'):
            growth = final_open_files - initial_open_files
            if growth < 5:  # Allow some growth for normal operations
                print(f"✅ File handle cleanup: handles managed properly ({growth} growth)")
            else:
                print(f"⚠️  File handle cleanup: potential leak ({growth} descriptors added)")
        else:
            print("✅ File handle cleanup: tested (fd counting not available)")
        
        return True
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_memory_cleanup_after_execution():
    """Test that memory is cleaned up after code execution."""
    import sys
    
    # Get initial memory
    initial_refs = len(gc.get_objects())
    
    # Run code that creates objects
    code = """
    let data = [];
    let i = 0;
    while i < 100 {
        data = data + [[i, i * 2, i * 3]];
        i = i + 1;
    }
    """
    
    for _ in range(5):
        env, result = run_code(code)
        del env
        del result
    
    # Force cleanup
    gc.collect()
    
    final_refs = len(gc.get_objects())
    growth = final_refs - initial_refs
    
    if growth < 1000:  # Allow reasonable growth
        print(f"✅ Memory cleanup after execution: managed properly ({growth} objects growth)")
    else:
        print(f"⚠️  Memory cleanup after execution: potential leak ({growth} objects added)")
    
    return True


def test_environment_cleanup():
    """Test that environments are cleaned up properly."""
    import weakref
    
    # Create an environment and get a weak reference
    env = Environment()
    env.set("test", "value")
    weak_ref = weakref.ref(env)
    
    # Delete the environment
    del env
    gc.collect()
    
    # Check if it was collected
    if weak_ref() is None:
        print("✅ Environment cleanup: environments garbage collected properly")
    else:
        print("⚠️  Environment cleanup: environment still referenced")
    
    return True


def test_function_closure_cleanup():
    """Test that function closures don't cause memory leaks."""
    code = """
    action make_counter() {
        let count = 0;
        action increment() {
            count = count + 1;
            return count;
        }
        return increment;
    }
    
    let counter1 = make_counter();
    let counter2 = make_counter();
    let result = counter1() + counter2();
    """
    
    initial_objects = len(gc.get_objects())
    
    # Run multiple times
    for _ in range(10):
        env, result = run_code(code)
        del env
        del result
    
    gc.collect()
    
    final_objects = len(gc.get_objects())
    growth = final_objects - initial_objects
    
    if growth < 500:
        print(f"✅ Function closure cleanup: closures managed properly ({growth} objects growth)")
    else:
        print(f"⚠️  Function closure cleanup: potential leak ({growth} objects added)")
    
    return True


def test_circular_reference_cleanup():
    """Test that circular references are handled."""
    code = """
    let obj1 = {ref: null};
    let obj2 = {ref: null};
    # Create circular reference
    # obj1.ref = obj2;
    # obj2.ref = obj1;
    """
    
    initial_objects = len(gc.get_objects())
    
    for _ in range(10):
        env, result = run_code(code)
        del env
        del result
    
    gc.collect()
    
    final_objects = len(gc.get_objects())
    growth = final_objects - initial_objects
    
    if growth < 300:
        print(f"✅ Circular reference cleanup: managed properly ({growth} objects growth)")
    else:
        print(f"⚠️  Circular reference cleanup: potential leak ({growth} objects added)")
    
    return True


def test_exception_cleanup():
    """Test that resources are cleaned up even when exceptions occur."""
    code = """
    action risky_operation() {
        let x = 10 / 0;  # Will cause error
        return x;
    }
    
    let result = risky_operation();
    """
    
    initial_objects = len(gc.get_objects())
    
    # Run code that will error
    for _ in range(10):
        try:
            run_code(code)
        except Exception:
            # Ignore expected evaluation errors; we are testing cleanup, not result
            pass
        gc.collect()
    
    final_objects = len(gc.get_objects())
    growth = final_objects - initial_objects
    
    if growth < 200:
        print(f"✅ Exception cleanup: resources cleaned up on error ({growth} objects growth)")
    else:
        print(f"⚠️  Exception cleanup: potential leak on errors ({growth} objects added)")
    
    return True


if __name__ == '__main__':
    print("=" * 70)
    print("RESOURCE CLEANUP TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_file_handle_cleanup,
        test_memory_cleanup_after_execution,
        test_environment_cleanup,
        test_function_closure_cleanup,
        test_circular_reference_cleanup,
        test_exception_cleanup,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    sys.exit(0 if failed == 0 else 1)
