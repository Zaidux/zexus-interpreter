import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser import Parser
from zexus.evaluator import eval_node, evaluate, Environment

# Simple test program
code = 'print("Hello from Zexus")\nlet x = 5\nprint("x = " + string(x))\n'

print('--- Running test snippet ---')
lexer = Lexer(code)
parser = Parser(lexer)
program = parser.parse_program()
if getattr(parser, 'errors', None):
    print('Parser errors:', parser.errors)
else:
    env = Environment()
    result = eval_node(program, env)
    print('--- Eval finished ---')
    if result is not None:
        try:
            print('Final result:', result.inspect())
        except Exception:
            print('Final result (raw):', result)
