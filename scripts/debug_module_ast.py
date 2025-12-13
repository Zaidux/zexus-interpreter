import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser

code = '''module math_operations {
    function add(a, b) {
        return a + b;
    };
};
'''

lexer = Lexer(code)
parser = UltimateParser(lexer, enable_advanced_strategies=False)
ast = parser.parse_program()

stmt = ast.statements[0]
print(f'Statement type: {type(stmt).__name__}')
print(f'Statement: {stmt}')
print(f'Module name: {stmt.name}')
print(f'Module body: {stmt.body}')
print(f'Module body type: {type(stmt.body).__name__}')
if hasattr(stmt.body, 'statements'):
    print(f'Module body statements: {stmt.body.statements}')
    for s in stmt.body.statements:
        print(f'  - {type(s).__name__}: {s}')
