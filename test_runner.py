"""
Test runner for Zexus interpreter
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from zexus.evaluator import evaluate
from zexus.lexer import Lexer
from zexus.parser import Parser
from zexus.object import Environment

def run_test_file(filename):
    """Run a test file and analyze its behavior"""
    with open(filename, 'r') as f:
        code = f.read()

    print("=== Testing Zexus Language Capabilities ===\n")
    print(f"Running file: {filename}\n")

    # Phase 1: Lexical Analysis
    print("Phase 1: Lexical Analysis")
    print("-" * 50)
    lexer = Lexer(code)
    tokens = []
    errors = []
    try:
        while True:
            token = lexer.next_token()
            if token.type == "EOF":
                break
            tokens.append(token)
    except Exception as e:
        errors.append(f"Lexer error: {str(e)}")
    
    print(f"Total tokens: {len(tokens)}")
    if errors:
        print("Lexer errors:", errors)
    print()

    # Phase 2: Parsing
    print("Phase 2: Parsing")
    print("-" * 50)
    parser = Parser(Lexer(code))
    program = parser.parse_program()
    
    if hasattr(parser, 'errors') and parser.errors:
        print("Parser errors:")
        for error in parser.errors:
            print(f"  - {error}")
    else:
        print("Parsing successful")
    print()

    # Phase 3: Evaluation
    print("Phase 3: Evaluation")
    print("-" * 50)
    env = Environment()
    try:
        print("Program output:\n")
        result = evaluate(program, env, debug_mode=True)
        print("\nEvaluation completed successfully")
    except Exception as e:
        print(f"Evaluation error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_runner.py <test_file>")
        sys.exit(1)
    run_test_file(sys.argv[1])