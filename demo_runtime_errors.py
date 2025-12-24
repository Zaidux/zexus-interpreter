"""
Demo: Runtime Error Reporting

Tests the new runtime error messages with helpful suggestions.
"""

from src.zexus.lexer import Lexer
from src.zexus.parser import Parser
from src.zexus.evaluator import evaluate
from src.zexus.object import Environment, EvaluationError
from src.zexus.config import config

# Disable debug logs
config.enable_debug_logs = False

print("=" * 70)
print(" ZEXUS RUNTIME ERROR REPORTING DEMO")
print("=" * 70)
print()

# Demo 1: Undefined variable
print("📝 Demo 1: Undefined Variable with Suggestion")
print("-" * 70)
code1 = '''
let message = "Hello"
print(messag)  # Typo: should be 'message'
'''

try:
    lexer = Lexer(code1, 'demo1.zx')
    parser = Parser(lexer, 'universal', enable_advanced_strategies=False)
    program = parser.parse_program()
    env = Environment()
    result = evaluate(program, env)
    
    if isinstance(result, EvaluationError):
        print(str(result))
except Exception as e:
    print(f"Error: {e}")

print()

# Demo 2: Division by zero
print("📝 Demo 2: Division by Zero")
print("-" * 70)
code2 = '''
let x = 10
let y = 0
let result = x / y
'''

try:
    lexer = Lexer(code2, 'demo2.zx')
    parser = Parser(lexer, 'universal', enable_advanced_strategies=False)
    program = parser.parse_program()
    env = Environment()
    result = evaluate(program, env)
    
    if isinstance(result, EvaluationError):
        print(str(result))
except Exception as e:
    print(f"Error: {e}")

print()

# Demo 3: Modulo by zero
print("📝 Demo 3: Modulo by Zero")
print("-" * 70)
code3 = '''
let remainder = 42 % 0
'''

try:
    lexer = Lexer(code3, 'demo3.zx')
    parser = Parser(lexer, 'universal', enable_advanced_strategies=False)
    program = parser.parse_program()
    env = Environment()
    result = evaluate(program, env)
    
    if isinstance(result, EvaluationError):
        print(str(result))
except Exception as e:
    print(f"Error: {e}")

print()

# Demo 4: Variable not declared
print("📝 Demo 4: Variable Never Declared")
print("-" * 70)
code4 = '''
print(undefinedVariable)
'''

try:
    lexer = Lexer(code4, 'demo4.zx')
    parser = Parser(lexer, 'universal', enable_advanced_strategies=False)
    program = parser.parse_program()
    env = Environment()
    result = evaluate(program, env)
    
    if isinstance(result, EvaluationError):
        print(str(result))
except Exception as e:
    print(f"Error: {e}")

print()

# Demo 5: Similar variable name suggestion
print("📝 Demo 5: Similar Variable Name (Typo Detection)")
print("-" * 70)
code5 = '''
let userName = "Alice"
let userAge = 30
print(usrName)  # Typo: missing 'e' in 'user'
'''

try:
    lexer = Lexer(code5, 'demo5.zx')
    parser = Parser(lexer, 'universal', enable_advanced_strategies=False)
    program = parser.parse_program()
    env = Environment()
    result = evaluate(program, env)
    
    if isinstance(result, EvaluationError):
        print(str(result))
except Exception as e:
    print(f"Error: {e}")

print()

print("=" * 70)
print(" END OF DEMO")
print("=" * 70)
print()
print("✨ Runtime errors now include:")
print("   • Smart suggestions based on context")
print("   • Similar variable name detection")
print("   • Helpful hints for fixing issues")
print("   • Clear, beginner-friendly messages")
print()
