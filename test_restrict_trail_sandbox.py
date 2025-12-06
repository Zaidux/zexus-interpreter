#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/zexus-interpreter')

print("=" * 60)
print("Testing RESTRICT / TRAIL / SANDBOX Integration")
print("=" * 60)

try:
    from src.zexus.security import get_security_context
    from src.zexus.evaluator.core import Evaluator
    from src.zexus.object import Environment, Map, String, EvaluationError
    from src.zexus.zexus_ast import Identifier, PropertyAccessExpression, StringLiteral, AssignmentExpression, PrintStatement
    print("  ✓ Imports OK")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    raise

def reset_ctx(ctx):
    ctx.restrictions.clear()
    ctx._restrictions_index.clear()
    ctx.trails.clear()
    ctx.sandbox_runs.clear()
    ctx.audit_log.clear()

def test_register_and_lookup():
    ctx = get_security_context()
    reset_ctx(ctx)
    entry = ctx.register_restriction('user.email', 'email', 'read-only')
    assert entry['restriction'] == 'read-only'
    found = ctx.get_restriction('user.email')
    assert found and found['restriction'] == 'read-only'
    print('  ✓ register_restriction / get_restriction')

def test_read_enforcement_redact():
    ctx = get_security_context()
    reset_ctx(ctx)
    ctx.register_restriction('user.email', 'email', 'redact')
    env = Environment()
    env.set('user', Map({'email': String('secret@x.com')}))
    evaluator = Evaluator()

    node = PropertyAccessExpression(Identifier('user'), Identifier('email'))
    res = evaluator.eval_node(node, env)
    assert isinstance(res, String) and res.value == '***REDACTED***'
    print('  ✓ read enforcement (redact)')

def test_write_enforcement_read_only():
    ctx = get_security_context()
    reset_ctx(ctx)
    ctx.register_restriction('user.email', 'email', 'read-only')
    env = Environment()
    env.set('user', Map({'email': String('secret@x.com')}))
    evaluator = Evaluator()

    assign = AssignmentExpression(name=PropertyAccessExpression(Identifier('user'), Identifier('email')),
                                   value=StringLiteral('new@x.com'))
    res = evaluator.eval_node(assign, env)
    assert isinstance(res, EvaluationError)
    print('  ✓ write enforcement (read-only)')

def test_trail_registration_and_emit():
    ctx = get_security_context()
    reset_ctx(ctx)
    ctx.register_trail('print', None)
    env = Environment()
    evaluator = Evaluator()

    # emit a print via evaluator
    node = PrintStatement(StringLiteral('hello'))
    _ = evaluator.eval_node(node, env)

    # Check that audit log got a trail event
    entries = ctx.audit_log.get_entries()
    assert any('hello' in (e.get('payload') or '') or 'hello' in str(e) for e in entries)
    print('  ✓ trail registration and emit_event')

if __name__ == '__main__':
    test_register_and_lookup()
    test_read_enforcement_redact()
    test_write_enforcement_read_only()
    test_trail_registration_and_emit()
    print('\nAll tests passed for RESTRICT/TRAIl/SANDBOX basics')
