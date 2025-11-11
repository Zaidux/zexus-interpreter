# Test verify_integration with fixes

# Core test cases that should always pass
CORE_TESTS = [
    {
        'code': '''
action simple() {
    print("inside action")
}
simple()
''',
        'desc': "Basic action execution"
    },
    {
        'code': '''
action sayHello(name) {
    print("Hello " + name)
}
sayHello("World")
''',
        'desc': "Action with parameters"
    },
    {
        'code': '''
action async getData() {
    return await async_identity(42)
}
let result = getData()
print(core.string(result))
''',
        'desc': "Async action execution"
    },
    {
        'code': '''
let obj = {x: 1}
obj.y = 2
print(core.string(obj))
''',
        'desc': "Object property access/assign"
    },
    {
        'code': '''
let arr = [1,2,3]
let doubled = arr.map(x => x * 2)
print(core.string(doubled))
''',
        'desc': "Array methods with arrow functions"
    },
    {
        'code': '''
try {
    throw "oops"
} catch(err) {
    print("Caught: " + core.string(err))
}
''',
        'desc': "Try-catch error handling"
    },
    {
        'code': '''
let nums = [1,2,3,4,5]
let sum = nums.reduce((a,b) => a + b)
print(core.string(sum))
''',
        'desc': "Chained method calls"
    },
    {
        'code': '''
let obj = {
    data: {
        users: [
            { name: "Alice", scores: [10, 20, 30] },
            { name: "Bob", scores: [15, 25, 35] }
        ],
        settings: {
            active: true,
            config: {
                theme: "dark",
                notifications: {
                    email: true,
                    push: false
                }
            }
        }
    }
}
print(core.string(obj.data.users[0].scores[1]))
print(core.string(obj.data.settings.config.notifications.email))
''',
        'desc': "Deep nested structure access"
    }
]

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser import Parser, UltimateParser
from zexus.evaluator import evaluate
from zexus.object import Environment
from zexus.config import config

def run_test(test_case):
    # Add import of core library
    code = '''
use "zexus-core" as core

''' + test_case['code']
    desc = test_case['desc']
    
    print(f"\n=== Testing: {desc} ===")
    print(f"Code:\n{code}")
    
    try:
        # Start timing
        start_time = time.time()
        parse_start = time.time()
        # Parse with both parsers
        lexer = Lexer(code)
        basic_parser = Parser(lexer)
        basic_prog = basic_parser.parse_program()
        
        lexer2 = Lexer(code)
        ultimate_parser = UltimateParser(lexer2)
        ultimate_prog = ultimate_parser.parse_program()
        
        # Check parser outputs match
        if str(basic_prog) != str(ultimate_prog):
            print("⚠️  Parser output mismatch!")
            print("Basic parser:")
            print(basic_prog)
            print("\nUltimate parser:")
            print(ultimate_prog)
            return False
            
        parse_time = time.time() - parse_start
        print(f"⏱️  Parsing time: {parse_time:.4f}s")
            
        # Evaluate program
        eval_start = time.time()
        env = Environment()
        config.enable_debug('minimal')  # Show important logs only
        result = evaluate(ultimate_prog, env)
        config.disable_debug()
        
        eval_time = time.time() - eval_start
        total_time = time.time() - start_time
        
        print(f"⏱️  Evaluation time: {eval_time:.4f}s")
        print(f"⏱️  Total time: {total_time:.4f}s")
        
        print("✅ Test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔬 Running core integration tests...")
    failed = []
    
    for test in CORE_TESTS:
        if not run_test(test):
            failed.append(test['desc'])
            
    print("\n=== Summary ===")
    print(f"Total tests: {len(CORE_TESTS)}")
    print(f"Passed: {len(CORE_TESTS) - len(failed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed tests:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n✨ All tests passed!")

if __name__ == "__main__":
    main()