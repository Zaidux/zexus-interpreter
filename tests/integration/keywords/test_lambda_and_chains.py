from zexus.lexer import Lexer
from zexus.parser import UltimateParser

def run_case(label, code, advanced=False):
    print(f"\n=== {label} ===\n{code}")
    lex = Lexer(code)
    parser = UltimateParser(lex, enable_advanced_strategies=advanced)
    prog = parser.parse_program()
    print("Errors:", parser.errors)
    for s in prog.statements:
        print(s)

def test_all():
    cases = [
        ("arrow_single", "let f = x => x + 1;"),
        ("arrow_paren", "let f = (a, b) => a + b;"),
        ("lambda_keyword", "let f = lambda(x): x + 1;"),
        ("array", "let a = [1,2,3,4,5];"),
        ("object", 'let person = { name: "Alice", age: 30 };'),
        ("method_chain", 'let r = numbers.map(x => x * 2).filter(x => x > 3).reduce((a,b) => a + b);'),
        ("nested_array", 'let m = [[1,2],[3,4]];'),
        ("chain_with_property", 'let x = foo.bar.baz().qux;'),
    ]
    for label, code in cases:
        run_case(label, code, advanced=False)
    print("\n--- Now with advanced strategies (ContextStackParser) ---")
    for label, code in cases:
        run_case(label + "_adv", code, advanced=True)

if __name__ == "__main__":
    test_all()
