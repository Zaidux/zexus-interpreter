# Let's examine the current parser structure
from parser import Parser
from lexer import Lexer

# Test what happens when we parse "action simplest():"
source = "action simplest(): return \"function_works\""
lexer = Lexer(source)
parser = Parser(lexer)

# Check what token we get first
print("First token:", parser.cur_token.type, "->", parser.cur_token.literal)
print("Peek token:", parser.peek_token.type, "->", parser.peek_token.literal)

# See what parse_statement returns for ACTION token
if parser.cur_token.type == "ACTION":
    print("ACTION token detected! Calling parse_statement...")
    stmt = parser.parse_statement()
    print("Parsed statement type:", type(stmt).__name__ if stmt else "None")
else:
    print("ACTION token NOT detected as first token!")
