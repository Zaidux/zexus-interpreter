import sys
import os

# Add the src directory to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.evaluator.environment import Environment

def run_file(filename):
    with open(filename, 'r') as f:
        code = f.read()
    
    print(f"Running {filename}...")
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer)
    program = parser.parse_program()
    
    if parser.errors:
        print("Parser Errors:")
        for err in parser.errors:
            print(f"  {err}")
        return

    env = Environment()
    evaluator = Evaluator()
    result = evaluator.eval(program, env)
    
    print(f"Result: {result}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        print("Usage: python3 run_repro.py <filename>")
