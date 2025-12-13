#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter/src')

from zexus.lexer import Lexer
from zexus.parser.parser import UltimateParser


def test_modifiers_on_action():
    # Exercise the modifier parser directly to validate collection
    code = 'secure async action fetch() { return 1 }'
    lexer = Lexer(code)
    parser = UltimateParser(lexer, enable_advanced_strategies=False)

    # Advance until we reach the first modifier (parser.next_token already called twice in ctor)
    # Ensure current token is a modifier
    while not (parser.cur_token and parser.cur_token.type in { 'PUBLIC', 'PRIVATE', 'SEALED', 'ASYNC', 'NATIVE', 'INLINE', 'SECURE', 'PURE' }):
        parser.next_token()

    mods = parser._parse_modifiers()
    assert mods is not None and 'secure' in mods and 'async' in mods, f"Unexpected modifiers: {mods}"


if __name__ == '__main__':
    print('Running modifier parsing test')
    test_modifiers_on_action()
    print('OK')
