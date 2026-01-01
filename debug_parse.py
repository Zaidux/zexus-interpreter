from src.zexus.lexer import Lexer
from src.zexus.parser.parser import Parser

code = '''
let b = Block{index: 42}
'''

lexer = Lexer(code)
parser = Parser(lexer)
program = parser.parse_program()

print(f"Statements: {len(program.statements)}")
for i, stmt in enumerate(program.statements):
    print(f"\nStatement {i}: {type(stmt).__name__}")
    if hasattr(stmt, '__dict__'):
        for key, val in stmt.__dict__.items():
            print(f"  {key}: {type(val).__name__}")
            if key == 'value' and hasattr(val, '__dict__'):
                print(f"    Value details:")
                for vkey, vval in val.__dict__.items():
                    print(f"      {vkey}: {type(vval).__name__} = {repr(vval)[:100]}")
