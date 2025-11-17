import pytest
from zexus.lexer import Lexer
from zexus.parser import UltimateParser
from zexus.zexus_ast import (
    LetStatement, LambdaExpression, ListLiteral, MapLiteral,
    MethodCallExpression, PropertyAccessExpression
)

cases = [
    ("arrow_single", "let f = x => x + 1;", LambdaExpression),
    ("arrow_paren", "let f = (a, b) => a + b;", LambdaExpression),
    ("lambda_keyword", "let f = lambda(x): x + 1;", LambdaExpression),
    ("array", "let a = [1,2,3,4,5];", ListLiteral),
    ("object", 'let person = { name: "Alice", age: 30 };', MapLiteral),
    ("method_chain", 'let r = numbers.map(x => x * 2).filter(x => x > 3).reduce((a,b) => a + b);', MethodCallExpression),
    ("nested_array", 'let m = [[1,2],[3,4]];', ListLiteral),
    ("chain_with_property", 'let x = foo.bar.baz().qux;', PropertyAccessExpression),
]


@pytest.mark.parametrize("label,code,expected_type", cases)
@pytest.mark.parametrize("advanced", [False, True])
def test_parsing_cases(label, code, expected_type, advanced):
    lex = Lexer(code)
    parser = UltimateParser(lex, enable_advanced_strategies=advanced)
    prog = parser.parse_program()

    # No parse errors
    assert parser.errors == [], f"Parser errors for {label} (advanced={advanced}): {parser.errors}"

    # We expect at least one statement (let)
    assert len(prog.statements) >= 1, f"No statements parsed for {label} (advanced={advanced})"

    first = prog.statements[0]
    assert isinstance(first, LetStatement), f"First statement not LetStatement for {label}"

    value = first.value
    assert isinstance(value, expected_type), (
        f"For {label} expected value type {expected_type.__name__}, got {type(value).__name__} (advanced={advanced})"
    )
