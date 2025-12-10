import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.evaluator.core import Evaluator
from zexus.object import Environment

code = '''module math_operations {
    function add(a, b) {
        return a + b;
    };
};
'''

lexer = Lexer(code)
parser = UltimateParser(lexer, enable_advanced_strategies=False)
ast = parser.parse_program()

evaluator = Evaluator()
env = Environment()

# Evaluate just the module statement
result = evaluator.eval_node(ast.statements[0], env)
print(f'Module result: {result}')
print(f'Module: {env.store["math_operations"]}')
print(f'Module members: {env.store["math_operations"].members}')
