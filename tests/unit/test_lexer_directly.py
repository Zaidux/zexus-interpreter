from lexer import Lexer

# Test the lexer with just "action"
source = "action"
lexer = Lexer(source)
token = lexer.next_token()
print(f"Token for 'action': {token.type} -> '{token.literal}'")

# Test with "action simplest"
source2 = "action simplest"
lexer2 = Lexer(source2)
token1 = lexer2.next_token()
token2 = lexer2.next_token()
print(f"Token 1: {token1.type} -> '{token1.literal}'")
print(f"Token 2: {token2.type} -> '{token2.literal}'")
