"""Regression coverage for parser assignment handling."""

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser
from zexus.zexus_ast import (
    AssignmentExpression,
    ExpressionStatement,
    Identifier,
    IntegerLiteral,
)


def test_assignment_expression_round_trip():
    parser = UltimateParser(Lexer("x = 10"), enable_advanced_strategies=False)
    program = parser.parse_program()

    assert parser.errors == []
    assert len(program.statements) == 1

    statement = program.statements[0]
    assert isinstance(statement, ExpressionStatement)

    expression = statement.expression
    assert isinstance(expression, AssignmentExpression)
    assert isinstance(expression.name, Identifier)
    assert expression.name.value == "x"
    assert isinstance(expression.value, IntegerLiteral)
    assert expression.value.value == 10
