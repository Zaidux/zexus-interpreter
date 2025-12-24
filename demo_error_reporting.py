"""
Demo: Zexus Error Reporting System

This file demonstrates the new error reporting capabilities.
"""

from src.zexus.lexer import Lexer
from src.zexus.parser import Parser
from src.zexus.error_reporter import print_error, ZexusError
from src.zexus.config import config

# Disable debug logs for cleaner output
config.enable_debug_logs = False

print("=" * 70)
print(" ZEXUS ERROR REPORTING SYSTEM DEMO")
print("=" * 70)
print()

# Demo 1: Unterminated String
print("📝 Demo 1: Unterminated String")
print("-" * 70)
code1 = '''let message = "Hello world'''

try:
    lexer = Lexer(code1, 'demo1.zx')
    tok = lexer.next_token()
    while tok.type != 'EOF':
        tok = lexer.next_token()
except ZexusError as e:
    print_error(e)

print()

# Demo 2: Single & instead of &&
print("📝 Demo 2: Single '&' Instead of '&&'")
print("-" * 70)
code2 = '''if (x & y) {
    print("test")
}'''

try:
    lexer = Lexer(code2, 'demo2.zx')
    tok = lexer.next_token()
    while tok.type != 'EOF':
        tok = lexer.next_token()
except ZexusError as e:
    print_error(e)

print()

# Demo 3: Single | instead of ||
print("📝 Demo 3: Single '|' Instead of '||'")
print("-" * 70)
code3 = '''if (a | b) {
    print("also bad")
}'''

try:
    lexer = Lexer(code3, 'demo3.zx')
    tok = lexer.next_token()
    while tok.type != 'EOF':
        tok = lexer.next_token()
except ZexusError as e:
    print_error(e)

print()

# Demo 4: Unknown character
print("📝 Demo 4: Unknown Character")
print("-" * 70)
code4 = '''let value = 42 @ 10'''

try:
    lexer = Lexer(code4, 'demo4.zx')
    tok = lexer.next_token()
    while tok.type != 'EOF':
        tok = lexer.next_token()
except ZexusError as e:
    print_error(e)

print()

# Demo 5: Incomplete escape sequence
print("📝 Demo 5: Incomplete Escape Sequence")
print("-" * 70)
code5 = '''let text = "Hello\\'''

try:
    lexer = Lexer(code5, 'demo5.zx')
    tok = lexer.next_token()
    while tok.type != 'EOF':
        tok = lexer.next_token()
except ZexusError as e:
    print_error(e)

print()

# Demo 6: Parser - Missing condition
print("📝 Demo 6: Missing Condition After 'if'")
print("-" * 70)
code6 = '''if {
    print("no condition!")
}'''

try:
    lexer = Lexer(code6, 'demo6.zx')
    parser = Parser(lexer, 'universal', enable_advanced_strategies=False)
    program = parser.parse_program()
except ZexusError as e:
    print_error(e)

print()

# Demo 7: Parser - Missing variable name
print("📝 Demo 7: Missing Variable Name After 'let'")
print("-" * 70)
code7 = '''let = 42'''

try:
    lexer = Lexer(code7, 'demo7.zx')
    parser = Parser(lexer, 'universal', enable_advanced_strategies=False)
    program = parser.parse_program()
except ZexusError as e:
    print_error(e)

print()

# Demo 8: Parser - Unclosed map literal
print("📝 Demo 8: Unclosed Map Literal")
print("-" * 70)
code8 = '''let config = {
    name: "test",
    value: 100'''

try:
    lexer = Lexer(code8, 'demo8.zx')
    parser = Parser(lexer, 'universal', enable_advanced_strategies=False)
    program = parser.parse_program()
except ZexusError as e:
    print_error(e)

print()

print("=" * 70)
print(" END OF DEMO")
print("=" * 70)
print()
print("✨ All error messages are:")
print("   • Color-coded for easy reading")
print("   • Show exact line and column")
print("   • Display source code context")
print("   • Provide helpful suggestions")
print("   • Beginner-friendly and clear")
print()
