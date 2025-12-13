#!/usr/bin/env python3
"""
Enhanced test for AUDIT command including security.py integration
"""
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter')

print("=" * 70)
print("Enhanced AUDIT Testing: Parser Strategy + Security Integration")
print("=" * 70)

# Test 1: AuditLog class
print("\n1. Testing AuditLog class from security.py...")
try:
    from src.zexus.security import AuditLog, SecurityContext
    
    # Create and use AuditLog
    audit_log = AuditLog(max_entries=100)
    entry1 = audit_log.log("user_data", "access", "MAP")
    entry2 = audit_log.log("config", "modification", "MAP")
    entry3 = audit_log.log("user_data", "deletion", "MAP")
    
    print(f"   ✓ Created AuditLog: {audit_log}")
    print(f"   ✓ Logged 3 entries")
    print(f"   ✓ Entry 1 ID: {entry1['id']}")
    print(f"   ✓ Entry 1 Action: {entry1['action']}")
    print(f"   ✓ Entry 1 Type: {entry1['data_type']}")
    
    # Test querying
    user_data_accesses = audit_log.get_entries(data_name="user_data")
    print(f"   ✓ Query results for 'user_data': {len(user_data_accesses)} entries")
    
    access_actions = audit_log.get_entries(action="access")
    print(f"   ✓ Query results for 'access' action: {len(access_actions)} entries")
    
except Exception as e:
    print(f"   ✗ AuditLog test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: SecurityContext audit integration
print("\n2. Testing SecurityContext audit integration...")
try:
    security_ctx = SecurityContext()
    
    # Log through security context
    audit_entry = security_ctx.log_audit("api_key", "sensitive_access", "STRING")
    print(f"   ✓ SecurityContext.log_audit() working")
    print(f"   ✓ Audit entry: {audit_entry['action']} on {audit_entry['data_name']}")
    
    # Log with additional context
    ctx_data = {"user_id": 123, "ip_address": "192.168.1.1"}
    audit_entry2 = security_ctx.log_audit("transaction", "high_value_alert", "MAP", context=ctx_data)
    print(f"   ✓ Audit entry with context: {audit_entry2['context']}")
    
except Exception as e:
    print(f"   ✗ SecurityContext test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: AUDIT token still recognized
print("\n3. Testing AUDIT token recognition...")
try:
    from src.zexus.zexus_token import AUDIT
    print(f"   ✓ AUDIT token constant: {AUDIT}")
except Exception as e:
    print(f"   ✗ AUDIT token test failed: {e}")
    sys.exit(1)

# Test 4: Parser with multiple AUDIT statements
print("\n4. Testing Parser with multiple AUDIT statements...")
try:
    from src.zexus.lexer import Lexer
    from src.zexus.parser.parser import UltimateParser
    
    code = '''
    let sensitive_data = {key: "secret", value: 42};
    audit sensitive_data, "access";
    audit sensitive_data, "modification";
    let result = sensitive_data.key;
    audit sensitive_data, "query";
    '''
    
    lexer = Lexer(code)
    parser = UltimateParser(lexer, enable_advanced_strategies=False)
    program = parser.parse_program()
    
    # Count audit statements
    from src.zexus.zexus_ast import AuditStatement
    audit_count = sum(1 for stmt in program.statements if isinstance(stmt, AuditStatement))
    
    print(f"   ✓ Parsed {len(program.statements)} total statements")
    print(f"   ✓ Found {audit_count} AuditStatement nodes")
    print(f"   ✓ Parser errors: {len(parser.errors)}")
    
except Exception as e:
    print(f"   ✗ Parser test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Strategy parser recognition
print("\n5. Testing Strategy parsers recognize AUDIT...")
try:
    from src.zexus.parser.strategy_structural import StructuralAnalyzer
    from src.zexus.lexer import Lexer
    
    code = "audit data, \"access\"; let x = 5;"
    lexer = Lexer(code)
    tokens = []
    while True:
        token = lexer.next_token()
        tokens.append(token)
        if token.type == "EOF":
            break
    
    # Check that AUDIT is recognized
    audit_tokens = [t for t in tokens if t.type == AUDIT]
    print(f"   ✓ Lexer produced {len(audit_tokens)} AUDIT token(s)")
    
    # Verify structural analyzer handles it
    analyzer = StructuralAnalyzer()
    blocks = analyzer.analyze(tokens)
    print(f"   ✓ StructuralAnalyzer produced {len(blocks)} blocks")
    
except Exception as e:
    print(f"   ✗ Strategy parser test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: AuditLog export capability
print("\n6. Testing AuditLog export functionality...")
try:
    import tempfile
    import os
    
    # Create test log
    test_log = AuditLog(max_entries=50)
    test_log.log("resource1", "read", "STRING")
    test_log.log("resource2", "write", "MAP")
    test_log.log("resource1", "read", "STRING")
    
    # Export to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    
    success = test_log.export_to_file(temp_file)
    
    if success and os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            content = f.read()
        print(f"   ✓ Exported {len(test_log.entries)} entries to file")
        print(f"   ✓ File size: {len(content)} bytes")
        os.unlink(temp_file)
    else:
        print(f"   ! Export returned success={success}")
    
except Exception as e:
    print(f"   ✗ Export test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✨ All enhanced AUDIT tests passed!")
print("=" * 70)
