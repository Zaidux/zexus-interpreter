import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from zexus.lexer import Lexer
from zexus.parser.strategy_context import ContextStackParser
from zexus.zexus_token import TokenType, Token

def test_parse_function():
    code = "function() { return 1; }"
    print(f"Testing code: {code}")
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    # Remove EOF
    if tokens and tokens[-1].type == TokenType.EOF:
        tokens.pop()
        
    parser = ContextStackParser()
    # Mock the parser's dependencies if needed, but ContextStackParser seems self-contained enough for this method
    # It uses self._parse_block_statements, so we might need to be careful.
    
    try:
        result = parser._parse_function_literal(tokens)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_parse_function()
