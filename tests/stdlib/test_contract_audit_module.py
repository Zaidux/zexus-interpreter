"""Tests for stdlib ContractAuditModule."""

import pytest
from src.zexus.stdlib.contract_audit import ContractAuditModule


class TestReentrancy:
    def test_detects_reentrancy(self):
        source = '''
action withdraw(amount) {
    let success = transfer(TX.caller, amount)
    state.balance = state.balance - amount
}
'''
        findings = ContractAuditModule.audit(source)
        assert any("reentrancy" in f["rule"].lower() for f in findings)


class TestIntegerOverflow:
    def test_detects_overflow(self):
        source = '''
action add_balance(amount) {
    state.balance = state.balance + amount
}
'''
        findings = ContractAuditModule.audit(source)
        assert any("overflow" in f["rule"].lower() for f in findings)


class TestAccessControl:
    def test_detects_missing_access_control(self):
        source = '''
action set_admin(new_admin) {
    state.admin = new_admin
}
'''
        findings = ContractAuditModule.audit(source)
        assert any("access" in f["rule"].lower() for f in findings)


class TestGasLimits:
    def test_detects_unbounded_loop(self):
        source = '''
action process_all() {
    for item in state.items {
        process(item)
    }
}
'''
        findings = ContractAuditModule.audit(source)
        assert any("unbounded" in f["rule"].lower() or "loop" in f["rule"].lower() for f in findings)


class TestTimestampDependence:
    def test_detects_timestamp_usage(self):
        source = '''
action lottery() {
    if TX.timestamp % 2 == 0 {
        send_prize(TX.caller)
    }
}
'''
        findings = ContractAuditModule.audit(source)
        assert any("timestamp" in f["rule"].lower() for f in findings)


class TestAuditFile:
    def test_audit_file(self, tmp_path):
        contract = tmp_path / "token.zx"
        contract.write_text('''
contract Token {
    action transfer(to, amount) {
        let success = transfer(to, amount)
        state.balances[TX.caller] = state.balances[TX.caller] - amount
    }
}
''')
        findings = ContractAuditModule.audit_file(str(contract))
        assert len(findings) > 0


class TestAuditDirectory:
    def test_audit_directory(self, tmp_path):
        (tmp_path / "a.zx").write_text('action set_owner(o) { state.owner = o }')
        (tmp_path / "b.zx").write_text('action safe() { require_owner(); state.x = 1 }')
        findings = ContractAuditModule.audit_directory(str(tmp_path))
        assert isinstance(findings, list)


class TestSummary:
    def test_summary(self):
        findings = [
            {"severity": "critical", "rule": "a"},
            {"severity": "high", "rule": "b"},
            {"severity": "medium", "rule": "c"},
        ]
        summary = ContractAuditModule.summary(findings)
        assert summary["total"] == 3
        assert summary["critical"] == 1


class TestFormatReport:
    def test_format_text(self):
        findings = [
            {"rule": "reentrancy", "severity": "critical", "line": 3,
             "column": 0, "message": "Reentrancy detected", 
             "evidence": "transfer() before state update", "file": "test.zx"}
        ]
        report = ContractAuditModule.format_report(findings, format="text")
        assert "reentrancy" in report.lower()
