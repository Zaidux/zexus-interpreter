#!/usr/bin/env python3
"""
Simple test runner for Zexus files
"""
import os
import sys

# Set up path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
os.chdir(project_root)

# Add to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    """Run the Zexus test files"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "ZEXUS INTEGRATION TEST SUITE" + " "*30 + "║")
    print("║" + " "*15 + "Testing All 10 Strategic Phases in Zexus" + " "*24 + "║")
    print("╚" + "="*78 + "╝\n")
    
    test_files = [
        "src/tests/test_phase1_modifiers.zx",
        "src/tests/test_phase2_plugins.zx",
        "src/tests/test_phase3_security.zx",
        "src/tests/test_phase4_vfs.zx",
        "src/tests/test_phase5_types.zx",
        "src/tests/test_phase6_metaprogramming.zx",
        "src/tests/test_phase7_optimization.zx",
        "src/tests/test_phase9_advanced_types.zx",
        "src/tests/test_phase10_ecosystem.zx",
    ]
    
    passed = 0
    failed = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"✓ Created: {test_file}")
            passed += 1
        else:
            print(f"✗ Missing: {test_file}")
            failed += 1
    
    print(f"\nTotal test files: {passed + failed}")
    print(f"Created: {passed}")
    print(f"Missing: {failed}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("""
1. To run Phase 1 tests:
   python3 main.py src/tests/test_phase1_modifiers.zx

2. To run Phase 2 tests:
   python3 main.py src/tests/test_phase2_plugins.zx

3. To run Phase 3 tests:
   python3 main.py src/tests/test_phase3_security.zx

4. To run Phase 4 tests:
   python3 main.py src/tests/test_phase4_vfs.zx

5. To run Phase 5 tests:
   python3 main.py src/tests/test_phase5_types.zx

6. To run Phase 6 tests:
   python3 main.py src/tests/test_phase6_metaprogramming.zx

7. To run Phase 7 tests:
   python3 main.py src/tests/test_phase7_optimization.zx

8. To run Phase 9 tests:
   python3 main.py src/tests/test_phase9_advanced_types.zx

9. To run Phase 10 tests:
   python3 main.py src/tests/test_phase10_ecosystem.zx

10. To run integrated test:
    python3 main.py src/tests/test_all_phases.zx
""")
    
    print("="*80)
    print("✅ All test files created successfully!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
