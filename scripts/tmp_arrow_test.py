from zexus.lexer import Lexer
from zexus.parser import UltimateParser

samples = {
    'arrow_single': 'let f = x => x + 1 ;',
    'arrow_paren': 'let f = (a, b) => a + b ;',
    'map_chain': 'let r = numbers.map(x => x * 2).filter(x => x > 3).reduce((a,b) => a + b) ;'
}

for name, code in samples.items():
    print('\n===', name, '===')
    print(code)
    lex = Lexer(code)
    p = UltimateParser(lex, enable_advanced_strategies=False)
    prog = p.parse_program()
    print('Errors:', p.errors)
    for s in prog.statements:
        print(s)
