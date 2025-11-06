#!/usr/bin/env python3
# investigate_compiler.py

import subprocess
import sys
import os

def run_compiler_test(filename, mode="compile"):
    """Run a test file and capture compiler output"""
    print(f"\n=== Testing {filename} in {mode} mode ===")
    
    cmd = ["zx", "run", filename]
    if mode == "interpret":
        cmd.extend(["--mode", "interpret"])
    elif mode == "compile":
        cmd.extend(["--mode", "compile"])
    
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

def main():
    # Test with different modes
    test_file = "debug_compiler.zx"
    
    # Create the test file
    test_content = '''// debug_compiler.zx
print("Testing compiler...")
let x = 42
print("x = " + string(x))
print("Done")
'''
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    print("Created test file: debug_compiler.zx")
    
    # Test 1: Try compilation mode
    print("\n" + "="*50)
    print("TEST 1: COMPILATION MODE")
    print("="*50)
    run_compiler_test(test_file, "compile")
    
    # Test 2: Try interpretation mode  
    print("\n" + "="*50)
    print("TEST 2: INTERPRETATION MODE")
    print("="*50)
    run_compiler_test(test_file, "interpret")
    
    # Test 3: Try auto mode
    print("\n" + "="*50)
    print("TEST 3: AUTO MODE")
    print("="*50)
    run_compiler_test(test_file, "auto")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
        print(f"\nCleaned up {test_file}")

if __name__ == "__main__":
    main()
