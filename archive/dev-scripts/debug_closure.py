"""
Debug closure capture in compiler vs. interpreter
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from zexus.lexer import Lexer as IntLexer
from zexus.parser import UltimateParser
from zexus.object import Environment
from zexus.evaluator import evaluate

from zexus.compiler import ZexusCompiler

src = '''
let x = 1
action f() {
    return string(x)
}
let x = 2
let r = f()
print(r)
'''

print('=== INTERPRETER PATH ===')
try:
    lex = IntLexer(src)
    parser = UltimateParser(lex)
    prog = parser.parse_program()
    if parser.errors:
        print('Parser errors:', parser.errors)
    else:
        env = Environment()
        result = evaluate(prog, env, debug_mode=False)
        print('Result:', result)
        print('Environment keys after:', list(env.store.keys()))
except Exception as e:
    print('Interpreter error:', e)
    import traceback
    traceback.print_exc()

print('\n=== COMPILER PATH ===')
try:
    compiler = ZexusCompiler(src, enable_optimizations=False)
    bc = compiler.compile()
    if compiler.errors:
        print('Compiler errors:', compiler.errors)
    else:
        print('Bytecode:', bc)
        
        # Try to execute via VM
        try:
            from zexus.vm.vm import VM
            vm = VM()
            result = vm.execute(bc, debug=True)
            print('VM Result:', result)
        except Exception as ve:
            print('VM execution error:', ve)
            import traceback
            traceback.print_exc()
except Exception as e:
    print('Compiler error:', e)
    import traceback
    traceback.print_exc()
