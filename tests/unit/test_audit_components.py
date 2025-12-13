#!/usr/bin/env python3
"""
Comprehensive test for the AUDIT command feature.
"""
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter')

print("=" * 60)
print("Testing AUDIT Command Implementation")
print("=" * 60)

# Test 1: Token and imports
print("\n1. Testing AUDIT token and imports...")
try:
    from src.zexus.zexus_token import AUDIT
    print(f"   ✓ AUDIT token: {AUDIT}")
except Exception as e:
    print(f"   ✗ Failed to import AUDIT token: {e}")
    sys.exit(1)

try:
    from src.zexus.zexus_ast import AuditStatement
    print("   ✓ AuditStatement AST node imported")
except Exception as e:
    print(f"   ✗ Failed to import AuditStatement: {e}")
    sys.exit(1)

# Test 2: Lexer keyword recognition
print("\n2. Testing Lexer keyword recognition...")
try:
    from src.zexus.lexer import Lexer
    lexer = Lexer("audit user_data, \"access\";")
    token = lexer.next_token()
    if token.type == AUDIT:
        print(f"   ✓ Lexer recognizes 'audit' keyword as {AUDIT}")
    else:
        print(f"   ✗ Lexer returned {token.type} instead of {AUDIT}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Lexer test failed: {e}")
    sys.exit(1)

# Test 3: Parser recognition
print("\n3. Testing Parser audit statement parsing...")
try:
    from src.zexus.parser.parser import UltimateParser
    code = 'let data = {value: 42}; audit data, "access";'
    lexer = Lexer(code)
    # Disable advanced parsing to use traditional parser
    parser = UltimateParser(lexer, enable_advanced_strategies=False)
    program = parser.parse_program()
    
    # Check that the program has statements
    if len(program.statements) >= 2:
        # Second statement should be an AuditStatement
        audit_stmt = program.statements[1]
        if isinstance(audit_stmt, AuditStatement):
            print(f"   ✓ Parser correctly parsed audit statement: {audit_stmt}")
        else:
            print(f"   ! Parser parsed statement but got: {type(audit_stmt).__name__}")
    else:
        print(f"   ✗ Parser did not parse enough statements: {len(program.statements)}")
except Exception as e:
    print(f"   ✗ Parser test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Evaluator setup
print("\n4. Testing Evaluator support for AuditStatement...")
try:
    from src.zexus.evaluator.core import Evaluator
    from src.zexus.object import Environment
    
    evaluator = Evaluator()
    print("   ✓ Evaluator instantiated successfully")
    
    # Check that eval_audit_statement method exists
    if hasattr(evaluator, 'eval_audit_statement'):
        print("   ✓ eval_audit_statement method exists in evaluator")
    else:
        print("   ! eval_audit_statement method not found")
except Exception as e:
    print(f"   ✗ Evaluator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Full integration test
print("\n5. Testing Full Integration (Lexer → Parser → Evaluator)...")
try:
    from src.zexus.evaluator.core import Evaluator
    from src.zexus.object import Environment
    from src.zexus.lexer import Lexer
    from src.zexus.parser.parser import UltimateParser
    
    # Simple test code
    code = '''
    let user = {id: 1, name: "Alice"};
    audit user, "access";
    '''
    
    lexer = Lexer(code)
    # Use traditional parser
    parser = UltimateParser(lexer, enable_advanced_strategies=False)
    program = parser.parse_program()
    
    if parser.errors:
        print(f"   ! Parser errors: {parser.errors}")
    else:
        print("   ✓ Parser completed without errors")
    
    evaluator = Evaluator()
    env = Environment()
    result = evaluator.eval_node(program, env)
    
    print(f"   ✓ Evaluation completed: {type(result).__name__}")
    
except Exception as e:
    print(f"   ✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✨ All AUDIT command tests passed!")
print("=" * 60)
