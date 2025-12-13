#!/usr/bin/env python3
# investigate_compiler.py

import subprocess
import sys
import os

def run_compiler_test(filename, mode="auto"):
    """Run a test file and capture compiler output"""
    print(f"\n=== Testing {filename} in {mode} mode ===")
    
    cmd = ["zx", "--execution-mode", mode, "run", filename]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        print(f"Return code: {result.returncode}")
        return result
    except Exception as e:
        print(f"Error running command: {e}")
        return None

def test_simple_cases():
    """Test simple cases directly with compiler mode"""
    test_cases = [
        ('print(string(42))', "string function"),
        ('print(42)', "direct number print"),
        ('let x = "hello"; print(x)', "string variable"),
        ('let x = 10 + 5; print(x)', "basic math"),
    ]
    
    for code, description in test_cases:
        print(f"\n--- Testing: {description} ---")
        cmd = ["zx", "--execution-mode", "compiler", "run", "-"]
        result = subprocess.run(cmd, input=code, capture_output=True, text=True)
        
        print(f"Code: {code}")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")

def main():
    # Test simple cases first
    print("="*60)
    print("SIMPLE TEST CASES")
    print("="*60)
    test_simple_cases()
    
    # Create a minimal test file
    test_file = "minimal_test.zx"
    test_content = '''// Minimal test for compiler
print("Testing compiler...")
let x = 42
print("x = " + string(x))
print("Done")
'''
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    print("\n" + "="*60)
    print("FILE TEST CASES") 
    print("="*60)
    
    # Test with different modes
    run_compiler_test(test_file, "compiler")
    run_compiler_test(test_file, "interpreter")
    run_compiler_test(test_file, "auto")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\nCleaned up {test_file}")

if __name__ == "__main__":
    main()