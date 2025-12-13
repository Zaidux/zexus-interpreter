#!/usr/bin/env python3
"""
Test Convenience & Advanced Features: DEFER, PATTERN, ENUM, STREAM, WATCH
"""

import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.object import Environment

def test_defer_statement():
    """Test defer statement"""
    print("\n=== Testing DEFER Statement ===")
    
    code = '''
defer cleanup();
defer close_file();
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
    
    print(f"✓ DEFER statements evaluated successfully")
    return True

def test_pattern_statement():
    """Test pattern statement"""
    print("\n=== Testing PATTERN Statement ===")
    
    code = '''
let x = 2;
pattern x {
  case 1 => print "one";
  case 2 => print "two";
  default => print "other";
}
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
    
    print(f"✓ PATTERN statement evaluated successfully")
    return True

def test_enum_statement():
    """Test enum statement"""
    print("\n=== Testing ENUM Statement ===")
    
    code = '''
enum Color {
  Red,
  Green,
  Blue
}
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
    
    print(f"✓ ENUM statement evaluated successfully")
    return True

def test_stream_statement():
    """Test stream statement"""
    print("\n=== Testing STREAM Statement ===")
    
    code = '''
stream clicks as event => {
  print "Click event received";
}

stream api_responses as response => {
  print "Response received";
}
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
    
    print(f"✓ STREAM statements evaluated successfully")
    return True

def test_watch_statement():
    """Test watch statement"""
    print("\n=== Testing WATCH Statement ===")
    
    code = '''
let count = 0;
watch count => {
  print "Count changed";
}

watch user_name => print "Name updated";
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
    
    print(f"✓ WATCH statement evaluated successfully")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Convenience & Advanced Features Implementation")
    print("=" * 60)
    
    tests = [
        test_defer_statement,
        test_pattern_statement,
        test_enum_statement,
        test_stream_statement,
        test_watch_statement,
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
