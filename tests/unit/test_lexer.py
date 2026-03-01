"""
Unit tests for src/zexus/lexer.py — Lexer class.

Tests tokenisation of all major token types: keywords, operators,
literals (int, float, string), comments, punctuation, edge cases.
"""

import pytest
from src.zexus.lexer import Lexer
from src.zexus.zexus_token import *


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokens(source: str):
    """Return a list of (type, literal) tuples for *source*."""
    lex = Lexer(source)
    toks = []
    while True:
        t = lex.next_token()
        toks.append((t.type, t.literal))
        if t.type == EOF:
            break
    return toks


def _types(source: str):
    """Return just the token types (no EOF)."""
    return [t for t, _ in _tokens(source) if t != EOF]


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------

class TestIntegerLiterals:
    def test_single_digit(self):
        ts = _tokens("5")
        assert ts[0] == (INT, "5")

    def test_multi_digit(self):
        ts = _tokens("12345")
        assert ts[0] == (INT, "12345")

    def test_negative_integer_is_minus_then_int(self):
        types = _types("-7")
        assert MINUS in types and INT in types


class TestFloatLiterals:
    def test_simple_float(self):
        ts = _tokens("3.14")
        assert ts[0] == (FLOAT, "3.14")

    def test_float_leading_zero(self):
        ts = _tokens("0.5")
        assert ts[0] == (FLOAT, "0.5")


class TestStringLiterals:
    def test_double_quoted(self):
        ts = _tokens('"hello"')
        assert ts[0][0] == STRING
        assert ts[0][1] == "hello"

    def test_single_quoted(self):
        ts = _tokens("'world'")
        assert ts[0][0] == STRING
        assert ts[0][1] == "world"

    def test_empty_string(self):
        ts = _tokens('""')
        assert ts[0] == (STRING, "")

    def test_string_with_escape(self):
        ts = _tokens(r'"line\n"')
        assert ts[0][0] == STRING
        assert "\n" in ts[0][1]


class TestBooleanAndNull:
    def test_true(self):
        ts = _tokens("true")
        assert ts[0][0] == TRUE

    def test_false(self):
        ts = _tokens("false")
        assert ts[0][0] == FALSE

    def test_null(self):
        ts = _tokens("null")
        assert ts[0][0] == NULL


# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------

class TestKeywords:
    @pytest.mark.parametrize("kw, expected", [
        ("let", LET),
        ("const", CONST),
        ("if", IF),
        ("else", ELSE),
        ("while", WHILE),
        ("for", FOR),
        ("return", RETURN),
        ("action", ACTION),
        ("function", FUNCTION),
        ("print", PRINT),
        ("try", TRY),
        ("catch", CATCH),
        ("throw", THROW),
        ("break", BREAK),
        ("continue", CONTINUE),
        ("export", EXPORT),
        ("import", IMPORT),
        ("async", ASYNC),
        ("await", AWAIT),
        ("entity", ENTITY),
        ("contract", CONTRACT),
        ("state", STATE),
    ])
    def test_keyword(self, kw, expected):
        ts = _tokens(kw)
        assert ts[0][0] == expected


# ---------------------------------------------------------------------------
# Operators & punctuation
# ---------------------------------------------------------------------------

class TestOperators:
    @pytest.mark.parametrize("src, expected_type", [
        ("+", PLUS),
        ("-", MINUS),
        ("*", ASTERISK),
        ("/", SLASH),
        ("%", MOD),
        ("=", ASSIGN),
        ("==", EQ),
        ("!=", NOT_EQ),
        ("<", LT),
        (">", GT),
        ("<=", LTE),
        (">=", GTE),
        ("!", BANG),
        (".", DOT),
        (",", COMMA),
        (";", SEMICOLON),
        (":", COLON),
        ("(", LPAREN),
        (")", RPAREN),
        ("{", LBRACE),
        ("}", RBRACE),
        ("[", LBRACKET),
        ("]", RBRACKET),
        ("@", AT),
        ("??", NULLISH),
        ("?", QUESTION),
    ])
    def test_operator(self, src, expected_type):
        ts = _tokens(src)
        assert ts[0][0] == expected_type

    def test_arrow_is_lambda(self):
        ts = _tokens("=>")
        assert ts[0][0] == LAMBDA

    def test_double_lt_is_import_op(self):
        ts = _tokens("<<")
        assert ts[0][0] == IMPORT_OP

    def test_double_gt_is_append(self):
        ts = _tokens(">>")
        assert ts[0][0] == APPEND

    def test_and_operator(self):
        ts = _tokens("&&")
        assert ts[0][0] == AND

    def test_or_operator(self):
        ts = _tokens("||")
        assert ts[0][0] == OR

    def test_plus_assign(self):
        ts = _tokens("+=")
        assert ts[0][0] == PLUS_ASSIGN

    def test_minus_assign(self):
        ts = _tokens("-=")
        assert ts[0][0] == MINUS_ASSIGN


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------

class TestComments:
    def test_hash_comment(self):
        ts = _types("# this is a comment\n42")
        assert INT in ts
        # No comment token emitted
        assert len([t for t in ts if t == INT]) == 1

    def test_double_slash_comment(self):
        ts = _types("// comment\n42")
        assert INT in ts

    def test_block_comment(self):
        ts = _types("/* block */42")
        assert INT in ts


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

class TestIdentifiers:
    def test_simple_ident(self):
        ts = _tokens("myVar")
        assert ts[0][0] == IDENT
        assert ts[0][1] == "myVar"

    def test_underscore_ident(self):
        ts = _tokens("_private")
        assert ts[0][0] == IDENT

    def test_ident_with_digits(self):
        ts = _tokens("var123")
        assert ts[0][0] == IDENT


# ---------------------------------------------------------------------------
# Compound expressions
# ---------------------------------------------------------------------------

class TestCompound:
    def test_let_assignment(self):
        types = _types("let x = 10")
        assert types[0] == LET
        assert IDENT in types
        assert ASSIGN in types
        assert INT in types

    def test_function_call(self):
        types = _types("foo(1, 2)")
        assert types[0] == IDENT
        assert LPAREN in types
        assert RPAREN in types
        assert COMMA in types

    def test_if_else(self):
        types = _types("if (x > 0) { 1 } else { 2 }")
        assert IF in types
        assert ELSE in types
        assert GT in types

    def test_dot_access(self):
        types = _types("obj.method()")
        assert DOT in types


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_source(self):
        ts = _tokens("")
        assert ts[-1][0] == EOF

    def test_only_whitespace(self):
        ts = _tokens("   \n\t  ")
        assert ts[-1][0] == EOF

    def test_consecutive_comments(self):
        src = "# line1\n# line2\n# line3\n42"
        ts = _types(src)
        assert INT in ts

    def test_single_ampersand_raises(self):
        with pytest.raises(Exception):
            _tokens("&")

    def test_single_pipe_raises(self):
        with pytest.raises(Exception):
            _tokens("|")

    def test_line_tracking(self):
        lex = Lexer("a\nb")
        t1 = lex.next_token()
        assert t1.line == 1
        t2 = lex.next_token()
        assert t2.line == 2

    def test_multiline_string(self):
        ts = _tokens('"""hello\nworld"""')
        assert ts[0][0] == STRING
        assert "hello" in ts[0][1]
        assert "world" in ts[0][1]
