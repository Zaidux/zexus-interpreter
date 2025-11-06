# check_parser_setup.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.parser import UltimateParser
from zexus.zexus_token import LBRACE

def check_parser_setup():
    """Check if LBRACE is properly registered"""
    print("=== Parser Setup Check ===")
    
    # Create a parser instance
    from zexus.lexer import Lexer
    lexer = Lexer("test")
    parser = UltimateParser(lexer, enable_advanced_strategies=False)
    
    print("Prefix parse functions:")
    for token_type, func in parser.prefix_parse_fns.items():
        if token_type == LBRACE:
            print(f"✅ LBRACE -> {func.__name__}")
        else:
            print(f"  {token_type} -> {func.__name__}")

if __name__ == "__main__":
    check_parser_setup()
