from zexus.lexer import Lexer
from zexus.parser import UltimateParser

samples = {
    'array': 'let a = [1,2,3,4,5] ;',
    'object': 'let person = { name: "Alice", age: 30 } ;',
    'chain': 'let r = numbers.map(x => x * 2).filter(x => x > 3).reduce((a,b) => a + b) ;',
    'lambda': 'let f = lambda(x): x + 1 ;'
}

for name, code in samples.items():
    print('\n===', name, '===')
    lex = Lexer(code)
    p = UltimateParser(lex, enable_advanced_strategies=False)
    prog = p.parse_program()
    print('Errors:', p.errors)
    for s in prog.statements:
        print(s)
