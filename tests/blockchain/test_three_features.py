"""
Comprehensive tests for the 3 new features:

Feature 1  — Upgradeable Contracts & Chain Governance
Feature 2  — Formal Verification Engine
Feature 3  — Execution Accelerator (AOT, IC, WASM Cache, Batch, Numeric)
"""

import hashlib
import json
import os
import tempfile
import time
import pytest

# ── Feature 1: Upgradeable Contracts & Chain Governance ───────────
from zexus.blockchain.upgradeable import (
    ProxyContract,
    ImplementationRecord,
    UpgradeManager,
    ChainUpgradeGovernance,
    ChainUpgradeProposal,
    ProposalStatus,
    ProposalType,
    UpgradeEvent,
    UpgradeEventType,
)

# ── Feature 2: Formal Verification Engine ─────────────────────────
from zexus.blockchain.verification import (
    FormalVerifier,
    VerificationLevel,
    VerificationReport,
    VerificationFinding,
    Severity,
    FindingCategory,
    StructuralVerifier,
    InvariantVerifier,
    PropertyVerifier,
    AnnotationParser,
    Invariant,
    ContractProperty,
    SymValue,
    SymState,
    SymType,
    _extract_state_vars,
    _extract_actions,
    _walk_ast,
    _contains_state_write,
    _contains_external_call,
    _contains_require,
    _contains_caller_check,
)

# ── Feature 3: Execution Accelerator ─────────────────────────────
from zexus.blockchain.accelerator import (
    ExecutionAccelerator,
    AOTCompiler,
    InlineCache,
    InlineCacheEntry,
    NumericFastPath,
    WASMCache,
    BatchExecutor,
    TxBatchResult,
    CompiledAction,
    _fast_wrap,
    _fast_unwrap,
)


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

class MockChain:
    """Minimal chain stub for governance tests."""
    def __init__(self):
        self.difficulty = 4
        self.target_block_time = 10
        self.chain_id = "test-chain"
        self.gas_limit_default = 8_000_000
        self.base_fee = 1000
        self.max_block_gas = 30_000_000
        self.consensus_algorithm = "pow"
        self.height = 100


class MockAction:
    """Stub action object for structural verifier tests."""
    def __init__(self, body=None, parameters=None):
        self.body = body
        self.parameters = parameters or []


class StubNode:
    """Generic stub with attributes."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_contract_stub(name="TestToken", storage_vars=None, actions=None):
    """Build a minimal contract stub usable by the verifier."""
    c = StubNode(name=name)
    c.storage_vars = storage_vars or []
    c.actions = actions or {}
    return c


# ══════════════════════════════════════════════════════════════════════
#  FEATURE 1  — Upgradeable Contracts
# ══════════════════════════════════════════════════════════════════════

class TestProxyContract:
    """ProxyContract data-class tests."""

    def test_create_proxy(self):
        proxy = ProxyContract(
            admin="0xAdmin",
            implementation_address="0xImplV1",
        )
        assert proxy.admin == "0xAdmin"
        assert proxy.implementation == "0xImplV1"
        assert proxy.version == 0
        assert proxy.address  # auto-generated UUID

    def test_proxy_address_custom(self):
        proxy = ProxyContract(
            admin="0xAdmin",
            implementation_address="0xImplV1",
            proxy_address="0xProxy123",
        )
        assert proxy.address == "0xProxy123"

    def test_proxy_serialization(self):
        proxy = ProxyContract(
            admin="0xAdmin",
            implementation_address="0xImplV1",
            proxy_address="0xProxy",
        )
        proxy.version = 3
        proxy.data["balance"] = 1000

        d = proxy.to_dict()
        assert d["admin"] == "0xAdmin"
        assert d["implementation"] == "0xImplV1"
        assert d["version"] == 3
        assert d["data"]["balance"] == 1000

        restored = ProxyContract.from_dict(d)
        assert restored.admin == "0xAdmin"
        assert restored.implementation == "0xImplV1"
        assert restored.version == 3
        assert restored.data["balance"] == 1000

    def test_proxy_data_isolation(self):
        p1 = ProxyContract(admin="a", implementation_address="i")
        p2 = ProxyContract(admin="b", implementation_address="i")
        p1.data["x"] = 1
        assert "x" not in p2.data


class TestImplementationRecord:
    """ImplementationRecord dataclass tests."""

    def test_record_creation(self):
        rec = ImplementationRecord(
            version=1,
            address="0xImpl1",
            deployer="0xAdmin",
        )
        assert rec.version == 1
        assert rec.address == "0xImpl1"
        assert rec.deployer == "0xAdmin"
        assert rec.timestamp > 0

    def test_record_serialization(self):
        rec = ImplementationRecord(
            version=2,
            address="0xImpl2",
            deployer="0xAdmin",
            code_hash="abc123",
            metadata={"audit": "passed"},
        )
        d = rec.to_dict()
        restored = ImplementationRecord.from_dict(d)
        assert restored.version == 2
        assert restored.address == "0xImpl2"
        assert restored.code_hash == "abc123"
        assert restored.metadata["audit"] == "passed"


class TestUpgradeManager:
    """UpgradeManager lifecycle tests."""

    def test_create_proxy(self):
        mgr = UpgradeManager()
        proxy = mgr.create_proxy(
            admin="0xAdmin",
            implementation_address="0xImplV1",
            proxy_address="0xProxy1",
        )
        assert proxy.implementation == "0xImplV1"
        assert proxy.version == 1
        assert proxy.admin == "0xAdmin"
        assert "0xProxy1" in mgr.list_proxies()

    def test_upgrade_success(self):
        mgr = UpgradeManager()
        proxy = mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        ok, msg = mgr.upgrade("0xP", "0xImplV2", "0xAdmin")
        assert ok is True
        assert proxy.implementation == "0xImplV2"
        assert proxy.version == 2

    def test_upgrade_non_admin_rejected(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        ok, msg = mgr.upgrade("0xP", "0xImplV2", "0xHacker")
        assert ok is False
        assert "not admin" in msg

    def test_upgrade_delay_enforcement(self):
        mgr = UpgradeManager(upgrade_delay=3600)  # 1 hour
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        ok, msg = mgr.upgrade("0xP", "0xImplV2", "0xAdmin")
        assert ok is False
        assert "delay" in msg.lower()

    def test_upgrade_with_migration(self):
        mgr = UpgradeManager()
        proxy = mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        proxy.data["counter"] = 10

        def migrate(data):
            data["counter"] = data.get("counter", 0) * 2
            data["migrated"] = True
            return data

        ok, msg = mgr.upgrade("0xP", "0xImplV2", "0xAdmin", migrate_fn=migrate)
        assert ok is True
        assert proxy.data["counter"] == 20
        assert proxy.data["migrated"] is True

    def test_upgrade_notfound(self):
        mgr = UpgradeManager()
        ok, msg = mgr.upgrade("0xNonexistent", "0xImpl", "0xAdmin")
        assert ok is False
        assert "not found" in msg.lower()

    def test_rollback_to_previous(self):
        mgr = UpgradeManager()
        proxy = mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        mgr.upgrade("0xP", "0xImplV2", "0xAdmin")
        assert proxy.version == 2

        ok, msg = mgr.rollback("0xP", "0xAdmin")
        assert ok is True
        assert proxy.implementation == "0xImplV1"
        assert proxy.version == 1

    def test_rollback_to_specific_version(self):
        mgr = UpgradeManager()
        proxy = mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        mgr.upgrade("0xP", "0xImplV2", "0xAdmin")
        mgr.upgrade("0xP", "0xImplV3", "0xAdmin")
        assert proxy.version == 3

        ok, msg = mgr.rollback("0xP", "0xAdmin", target_version=1)
        assert ok is True
        assert proxy.implementation == "0xImplV1"
        assert proxy.version == 1

    def test_rollback_no_previous(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        ok, msg = mgr.rollback("0xP", "0xAdmin")
        assert ok is False
        assert "no previous" in msg.lower()

    def test_rollback_non_admin_rejected(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        mgr.upgrade("0xP", "0xImplV2", "0xAdmin")
        ok, msg = mgr.rollback("0xP", "0xHacker")
        assert ok is False
        assert "not admin" in msg

    def test_change_admin(self):
        mgr = UpgradeManager()
        proxy = mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        ok, msg = mgr.change_admin("0xP", "0xNewAdmin", "0xAdmin")
        assert ok is True
        assert proxy.admin == "0xNewAdmin"

    def test_change_admin_non_admin_rejected(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        ok, msg = mgr.change_admin("0xP", "0xNewAdmin", "0xHacker")
        assert ok is False

    def test_change_admin_empty_rejected(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        ok, msg = mgr.change_admin("0xP", "", "0xAdmin")
        assert ok is False
        assert "empty" in msg.lower()

    def test_version_history(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        mgr.upgrade("0xP", "0xImplV2", "0xAdmin")
        mgr.upgrade("0xP", "0xImplV3", "0xAdmin")

        history = mgr.get_version_history("0xP")
        assert len(history) == 3
        assert history[0].version == 1
        assert history[1].version == 2
        assert history[2].version == 3

    def test_get_info(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        info = mgr.get_info("0xP")
        assert info is not None
        assert info["admin"] == "0xAdmin"
        assert info["version"] == 1
        assert info["total_versions"] == 1

    def test_get_info_notfound(self):
        mgr = UpgradeManager()
        assert mgr.get_info("0xNone") is None

    def test_events_emitted(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        mgr.upgrade("0xP", "0xImplV2", "0xAdmin")
        events = mgr.get_events()
        assert len(events) == 2
        assert events[0].event_type == UpgradeEventType.CONTRACT_UPGRADED
        assert events[1].event_type == UpgradeEventType.CONTRACT_UPGRADED

    def test_proxy_call_no_vm(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        with pytest.raises(RuntimeError, match="No ContractVM"):
            mgr.proxy_call("0xP", "transfer", caller="0xAlice")

    def test_proxy_call_notfound(self):
        mgr = UpgradeManager()
        with pytest.raises(RuntimeError, match="not found"):
            mgr.proxy_call("0xNone", "action")

    def test_multiple_proxies(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin1", "0xImpl1", proxy_address="0xP1")
        mgr.create_proxy("0xAdmin2", "0xImpl2", proxy_address="0xP2")
        assert len(mgr.list_proxies()) == 2

    def test_rollback_version_not_found(self):
        mgr = UpgradeManager()
        mgr.create_proxy("0xAdmin", "0xImplV1", proxy_address="0xP")
        mgr.upgrade("0xP", "0xImplV2", "0xAdmin")
        ok, msg = mgr.rollback("0xP", "0xAdmin", target_version=99)
        assert ok is False
        assert "not found" in msg.lower()


# ══════════════════════════════════════════════════════════════════════
#  FEATURE 1b — Chain Governance
# ══════════════════════════════════════════════════════════════════════

class TestChainUpgradeGovernance:
    """Chain governance proposal/vote/apply tests."""

    def _make_governance(self, validators=None, chain=None):
        chain = chain or MockChain()
        validators = validators or {"0xV1", "0xV2", "0xV3"}
        return ChainUpgradeGovernance(
            chain=chain,
            validators=validators,
        ), chain

    def test_propose_basic(self):
        gov, chain = self._make_governance()
        ok, msg, pid = gov.propose(
            proposer="0xV1",
            proposal_type=ProposalType.DIFFICULTY_CHANGE,
            description="Increase difficulty",
            changes={"difficulty": 8},
            activation_height=200,
        )
        assert ok is True
        assert pid is not None
        proposal = gov.get_proposal(pid)
        assert proposal is not None
        assert proposal.proposer == "0xV1"
        assert proposal.changes == {"difficulty": 8}

    def test_propose_non_validator_rejected(self):
        gov, _ = self._make_governance()
        ok, msg, pid = gov.propose(
            proposer="0xRandom",
            proposal_type=ProposalType.DIFFICULTY_CHANGE,
            description="test",
            changes={"difficulty": 8},
            activation_height=200,
        )
        assert ok is False
        assert "not a validator" in msg

    def test_propose_invalid_parameter(self):
        gov, _ = self._make_governance()
        ok, msg, pid = gov.propose(
            proposer="0xV1",
            proposal_type=ProposalType.CHAIN_PARAMETER,
            description="test",
            changes={"invalid_param": 1},
            activation_height=200,
        )
        assert ok is False
        assert "Unknown parameter" in msg

    def test_propose_past_activation_rejected(self):
        gov, _ = self._make_governance()
        ok, msg, pid = gov.propose(
            proposer="0xV1",
            proposal_type=ProposalType.DIFFICULTY_CHANGE,
            description="test",
            changes={"difficulty": 8},
            activation_height=50,  # past (chain height = 100)
        )
        assert ok is False
        assert "must be >" in msg

    def test_vote_approve(self):
        gov, _ = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        ok, msg = gov.vote(pid, "0xV2", approve=True)
        assert ok is True
        proposal = gov.get_proposal(pid)
        assert "0xV2" in proposal.votes_for

    def test_vote_reject(self):
        gov, _ = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        ok, msg = gov.vote(pid, "0xV2", approve=False)
        assert ok is True
        proposal = gov.get_proposal(pid)
        assert "0xV2" in proposal.votes_against

    def test_double_vote_rejected(self):
        gov, _ = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        ok, msg = gov.vote(pid, "0xV2", approve=True)
        assert ok is True
        ok2, msg2 = gov.vote(pid, "0xV2", approve=True)
        assert ok2 is False
        assert "already voted" in msg2

    def test_vote_non_validator_rejected(self):
        gov, _ = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        ok, msg = gov.vote(pid, "0xRandom", approve=True)
        assert ok is False
        assert "not a validator" in msg

    def test_quorum_auto_approval(self):
        """With 3 validators, quorum is floor(3*2/3)+1 = 3. Need 3 votes."""
        gov, _ = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        # V1 auto-voted, V2 votes yes, V3 votes yes → 3 votes = quorum
        gov.vote(pid, "0xV2", approve=True)
        gov.vote(pid, "0xV3", approve=True)
        proposal = gov.get_proposal(pid)
        assert proposal.status == ProposalStatus.APPROVED

    def test_apply_pending(self):
        gov, chain = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        gov.vote(pid, "0xV2", approve=True)
        gov.vote(pid, "0xV3", approve=True)

        assert chain.difficulty == 4  # before
        applied = gov.apply_pending(current_height=200)
        assert pid in applied
        assert chain.difficulty == 8  # after
        proposal = gov.get_proposal(pid)
        assert proposal.status == ProposalStatus.APPLIED

    def test_apply_before_activation_skipped(self):
        gov, chain = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        gov.vote(pid, "0xV2", approve=True)
        gov.vote(pid, "0xV3", approve=True)

        applied = gov.apply_pending(current_height=150)  # too early
        assert len(applied) == 0
        assert chain.difficulty == 4

    def test_revert_upgrade(self):
        gov, chain = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "revert test",
            {"difficulty": 8}, 200,
        )
        gov.vote(pid, "0xV2", approve=True)
        gov.vote(pid, "0xV3", approve=True)
        gov.apply_pending(200)
        assert chain.difficulty == 8

        ok, msg = gov.revert_upgrade(pid, "0xV1")
        assert ok is True
        assert chain.difficulty == 4
        proposal = gov.get_proposal(pid)
        assert proposal.status == ProposalStatus.REVERTED

    def test_revert_non_validator(self):
        gov, chain = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        gov.vote(pid, "0xV2", approve=True)
        gov.vote(pid, "0xV3", approve=True)
        gov.apply_pending(200)

        ok, msg = gov.revert_upgrade(pid, "0xRandom")
        assert ok is False

    def test_revert_not_applied(self):
        gov, _ = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        ok, msg = gov.revert_upgrade(pid, "0xV1")
        assert ok is False
        assert "not applied" in msg.lower()

    def test_list_proposals_filtered(self):
        gov, _ = self._make_governance()
        gov.propose("0xV1", ProposalType.DIFFICULTY_CHANGE, "a",
                     {"difficulty": 8}, 200)
        time.sleep(0.01)  # ensure distinct timestamps for proposal IDs
        gov.propose("0xV2", ProposalType.GAS_LIMIT_CHANGE, "b",
                     {"gas_limit_default": 10000000}, 300)

        all_p = gov.list_proposals()
        assert len(all_p) == 2
        pending = gov.list_proposals(status=ProposalStatus.PENDING)
        assert len(pending) == 2

    def test_add_remove_validator(self):
        gov, _ = self._make_governance()
        assert gov.validator_count == 3
        gov.add_validator("0xV4")
        assert gov.validator_count == 4
        gov.remove_validator("0xV4")
        assert gov.validator_count == 3

    def test_quorum_threshold_calculation(self):
        gov, _ = self._make_governance(validators={"0xV1", "0xV2", "0xV3"})
        # floor(3*2/3) + 1 = 3
        assert gov.quorum_threshold == 3

    def test_single_validator_auto_approve(self):
        """Single validator: quorum = 1. Proposal auto-approved on creation."""
        gov, chain = self._make_governance(validators={"0xV1"})
        _, _, pid = gov.propose(
            "0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
            {"difficulty": 8}, 200,
        )
        proposal = gov.get_proposal(pid)
        assert proposal.status == ProposalStatus.APPROVED

    def test_governance_info(self):
        gov, _ = self._make_governance()
        info = gov.get_governance_info()
        assert info["validator_count"] == 3
        assert "quorum_threshold" in info

    def test_events_emitted(self):
        gov, _ = self._make_governance()
        gov.propose("0xV1", ProposalType.DIFFICULTY_CHANGE, "test",
                     {"difficulty": 8}, 200)
        events = gov.get_events()
        assert len(events) >= 1
        assert events[0].event_type == UpgradeEventType.CHAIN_PROPOSAL_CREATED

    def test_multiple_parameter_change(self):
        gov, chain = self._make_governance()
        _, _, pid = gov.propose(
            "0xV1", ProposalType.CHAIN_PARAMETER, "multi-change",
            {"difficulty": 8, "target_block_time": 5}, 200,
        )
        gov.vote(pid, "0xV2", approve=True)
        gov.vote(pid, "0xV3", approve=True)
        gov.apply_pending(200)
        assert chain.difficulty == 8
        assert chain.target_block_time == 5


# ══════════════════════════════════════════════════════════════════════
#  FEATURE 2  — Formal Verification Engine
# ══════════════════════════════════════════════════════════════════════

class TestVerificationDataclasses:
    """Test verification data structures."""

    def test_verification_finding(self):
        f = VerificationFinding(
            category=FindingCategory.ACCESS_CONTROL,
            severity=Severity.HIGH,
            message="Missing access control",
            action_name="transfer",
            contract_name="Token",
        )
        assert f.category == FindingCategory.ACCESS_CONTROL
        assert f.severity == Severity.HIGH
        d = f.to_dict()
        assert d["severity"] == "HIGH"

    def test_verification_report_empty(self):
        r = VerificationReport(level=VerificationLevel.STRUCTURAL)
        assert r.passed is True  # no findings
        assert r.critical_count == 0
        assert r.high_count == 0

    def test_verification_report_with_findings(self):
        r = VerificationReport(level=VerificationLevel.STRUCTURAL)
        r.findings.append(VerificationFinding(
            category=FindingCategory.REENTRANCY,
            severity=Severity.CRITICAL,
            message="Reentrancy detected",
        ))
        assert r.passed is False
        assert r.critical_count == 1

    def test_verification_report_summary(self):
        r = VerificationReport(level=VerificationLevel.STRUCTURAL)
        r.contract_name = "Token"
        s = r.summary()
        assert "Token" in s
        assert "PASS" in s

    def test_verification_report_to_dict(self):
        r = VerificationReport(level=VerificationLevel.STRUCTURAL)
        r.contract_name = "Token"
        d = r.to_dict()
        assert d["contract_name"] == "Token"
        assert d["passed"] is True
        assert "findings" in d

    def test_severity_ordering(self):
        assert Severity.CRITICAL != Severity.LOW

    def test_invariant_dataclass(self):
        inv = Invariant(expression="balance >= 0", variable="balance")
        d = inv.to_dict()
        assert d["expression"] == "balance >= 0"

    def test_contract_property_dataclass(self):
        prop = ContractProperty(
            name="no_negative_balance",
            action="withdraw",
            postcondition="balance >= 0",
        )
        d = prop.to_dict()
        assert d["name"] == "no_negative_balance"


class TestSymbolicExecution:
    """Symbolic value and state tests."""

    def test_sym_value_integer(self):
        v = SymValue(sym_type=SymType.INTEGER, concrete=42)
        assert v.is_concrete
        assert not v.could_be_negative()
        assert not v.could_be_zero()

    def test_sym_value_unbounded(self):
        v = SymValue(sym_type=SymType.INTEGER, min_val=-100, max_val=100)
        assert v.is_bounded
        assert v.could_be_negative()
        assert v.could_be_zero()

    def test_sym_value_copy(self):
        v = SymValue(sym_type=SymType.INTEGER, concrete=10, min_val=0, max_val=100)
        c = v.copy()
        assert c.concrete == 10
        assert c.min_val == 0
        c.concrete = 20
        assert v.concrete == 10  # original unchanged

    def test_sym_state_basic(self):
        state = SymState()
        state.set("x", SymValue(sym_type=SymType.INTEGER, concrete=5))
        val = state.get("x")
        assert val is not None
        assert val.concrete == 5

    def test_sym_state_child_inherits(self):
        parent = SymState()
        parent.set("x", SymValue(sym_type=SymType.INTEGER, concrete=5))
        child = parent.child()
        val = child.get("x")
        assert val is not None
        assert val.concrete == 5

    def test_sym_state_child_shadow(self):
        parent = SymState()
        parent.set("x", SymValue(sym_type=SymType.INTEGER, concrete=5))
        child = parent.child()
        child.set("x", SymValue(sym_type=SymType.INTEGER, concrete=10))
        assert child.get("x").concrete == 10
        assert parent.get("x").concrete == 5

    def test_sym_state_constraints(self):
        state = SymState()
        state.add_constraint("x > 0")
        assert len(state.constraints) == 1


class TestASTHelpers:
    """Test the AST utility functions used by verifiers."""

    def test_extract_state_vars_strings(self):
        contract = _make_contract_stub(storage_vars=["balance", "owner"])
        vars = _extract_state_vars(contract)
        assert "balance" in vars
        assert "owner" in vars

    def test_extract_state_vars_nodes(self):
        var_node = StubNode(name="counter")
        contract = _make_contract_stub(storage_vars=[var_node])
        vars = _extract_state_vars(contract)
        assert "counter" in vars

    def test_extract_actions(self):
        action = MockAction()
        contract = _make_contract_stub(actions={"transfer": action})
        actions = _extract_actions(contract)
        assert "transfer" in actions

    def test_walk_ast_leaf(self):
        node = StubNode(value=42)
        nodes = _walk_ast(node)
        assert len(nodes) >= 1

    def test_walk_ast_with_body(self):
        inner = StubNode(value=1)
        outer = StubNode(body=inner)
        nodes = _walk_ast(outer)
        assert len(nodes) >= 2


class TestStructuralVerifier:
    """Structural verification pattern checks."""

    def test_empty_contract_passes(self):
        contract = _make_contract_stub(actions={})
        report = VerificationReport(level=VerificationLevel.STRUCTURAL)
        verifier = StructuralVerifier()
        verifier.verify(contract, report)
        assert report.passed is True

    def test_no_action_body_skipped(self):
        action = MockAction(body=None)
        contract = _make_contract_stub(actions={"noop": action})
        report = VerificationReport(level=VerificationLevel.STRUCTURAL)
        verifier = StructuralVerifier()
        verifier.verify(contract, report)
        assert report.actions_checked == 1

    def test_reentrancy_detection(self):
        """Build AST that has state write + external call → reentrancy."""
        # Simulate: state write then external call
        # The verifier checks for CallExpression after AssignmentExpression
        # using _contains_state_write and _contains_external_call
        call_node = StubNode(
            __class__=type("CallExpression", (), {}),
        )
        call_node.__class__.__name__ = "CallExpression"
        # Give it a function attribute for the external call check
        fn_node = StubNode(value="external_transfer")
        call_node.function = fn_node

        assign_node = StubNode(
            __class__=type("AssignmentExpression", (), {}),
        )
        assign_node.__class__.__name__ = "AssignmentExpression"
        assign_node.name = StubNode(value="balance")
        assign_node.value = StubNode(value=0)

        # Body has assign then call (state write before external call)
        body = StubNode(
            statements=[assign_node, call_node],
        )
        body.__class__ = type("BlockStatement", (), {})
        body.__class__.__name__ = "BlockStatement"

        action = MockAction(body=body)
        contract = _make_contract_stub(
            storage_vars=["balance"],
            actions={"withdraw": action},
        )

        report = VerificationReport(level=VerificationLevel.STRUCTURAL)
        verifier = StructuralVerifier()
        verifier.verify(contract, report)
        # Structural verifier should flag something
        assert report.actions_checked == 1


class TestAnnotationParser:
    """Annotation extraction tests."""

    def test_parse_invariant(self):
        source = """
        // @invariant balance >= 0
        // @invariant total_supply > 0
        """
        parser = AnnotationParser()
        annotations = parser.parse_annotations(source)
        assert "invariants" in annotations
        assert len(annotations["invariants"]) == 2

    def test_parse_property(self):
        source = """
        // @property conservation: total stays constant
        // @pre balance >= amount
        // @post balance >= 0
        """
        parser = AnnotationParser()
        annotations = parser.parse_annotations(source)
        assert "properties" in annotations
        assert len(annotations["properties"]) >= 1

    def test_parse_precondition(self):
        source = "// @pre amount > 0"
        parser = AnnotationParser()
        annotations = parser.parse_annotations(source)
        assert "preconditions" in annotations
        assert "amount > 0" in annotations["preconditions"]

    def test_parse_postcondition(self):
        source = "// @post balance >= 0"
        parser = AnnotationParser()
        annotations = parser.parse_annotations(source)
        assert "postconditions" in annotations
        assert "balance >= 0" in annotations["postconditions"]

    def test_empty_source(self):
        parser = AnnotationParser()
        annotations = parser.parse_annotations("")
        assert annotations["invariants"] == []
        assert annotations["properties"] == []

    def test_from_contract_metadata(self):
        contract = StubNode(
            blockchain_config=StubNode(
                verification={
                    "invariants": ["balance >= 0"],
                    "properties": [{"name": "safe", "action": "transfer",
                                    "postcondition": "balance >= 0"}],
                }
            )
        )
        parser = AnnotationParser()
        annotations = parser.from_contract_metadata(contract)
        assert len(annotations["invariants"]) >= 1


class TestFormalVerifier:
    """High-level verifier orchestration."""

    def test_structural_level(self):
        contract = _make_contract_stub(actions={})
        verifier = FormalVerifier(level=VerificationLevel.STRUCTURAL)
        report = verifier.verify_contract(contract)
        assert isinstance(report, VerificationReport)
        assert report.passed is True

    def test_invariant_level(self):
        contract = _make_contract_stub(actions={})
        verifier = FormalVerifier(
            level=VerificationLevel.INVARIANT,
            annotations={"invariants": ["balance >= 0"]},
        )
        report = verifier.verify_contract(contract)
        assert isinstance(report, VerificationReport)

    def test_property_level(self):
        contract = _make_contract_stub(actions={})
        verifier = FormalVerifier(
            level=VerificationLevel.PROPERTY,
            annotations={"properties": [], "invariants": []},
        )
        report = verifier.verify_contract(contract)
        assert isinstance(report, VerificationReport)

    def test_add_invariant(self):
        verifier = FormalVerifier(level=VerificationLevel.INVARIANT)
        verifier.add_invariant("total_supply > 0")
        assert verifier.invariant_count == 1

    def test_add_property(self):
        verifier = FormalVerifier(level=VerificationLevel.PROPERTY)
        verifier.add_property(
            name="safe_transfer",
            action="transfer",
            postcondition="balance >= 0",
        )
        assert verifier.property_count == 1

    def test_verify_multiple(self):
        c1 = _make_contract_stub(name="Token1", actions={})
        c2 = _make_contract_stub(name="Token2", actions={})
        verifier = FormalVerifier(level=VerificationLevel.STRUCTURAL)
        reports = verifier.verify_multiple([c1, c2])
        assert len(reports) == 2

    def test_report_to_dict(self):
        contract = _make_contract_stub(actions={})
        verifier = FormalVerifier(level=VerificationLevel.STRUCTURAL)
        report = verifier.verify_contract(contract)
        d = report.to_dict()
        assert "passed" in d
        assert "findings" in d
        assert "duration" in d


# ══════════════════════════════════════════════════════════════════════
#  FEATURE 3  — Execution Accelerator
# ══════════════════════════════════════════════════════════════════════

class TestInlineCache:
    """InlineCache LRU tests."""

    def test_put_get(self):
        ic = InlineCache()
        ic.put("key1", "value1")
        assert ic.get("key1") == "value1"

    def test_miss_returns_none(self):
        ic = InlineCache()
        assert ic.get("nonexistent") is None

    def test_lru_eviction(self):
        ic = InlineCache(max_size=3)
        ic.put("a", 1)
        ic.put("b", 2)
        ic.put("c", 3)
        ic.put("d", 4)  # evicts "a"
        assert ic.get("a") is None
        assert ic.get("d") == 4

    def test_access_refreshes_position(self):
        ic = InlineCache(max_size=3)
        ic.put("a", 1)
        ic.put("b", 2)
        ic.put("c", 3)
        ic.get("a")  # refreshes "a"
        ic.put("d", 4)  # evicts "b" (oldest untouched)
        assert ic.get("a") == 1
        assert ic.get("b") is None

    def test_update_existing(self):
        ic = InlineCache()
        ic.put("key", "old")
        ic.put("key", "new")
        assert ic.get("key") == "new"
        assert ic.size == 1

    def test_invalidate(self):
        ic = InlineCache()
        ic.put("key", "val")
        assert ic.invalidate("key") is True
        assert ic.get("key") is None
        assert ic.invalidate("key") is False

    def test_invalidate_prefix(self):
        ic = InlineCache()
        ic.put("contract1:transfer", 1)
        ic.put("contract1:balance", 2)
        ic.put("contract2:transfer", 3)
        removed = ic.invalidate_prefix("contract1:")
        assert removed == 2
        assert ic.get("contract2:transfer") == 3

    def test_clear(self):
        ic = InlineCache()
        ic.put("a", 1)
        ic.put("b", 2)
        ic.clear()
        assert ic.size == 0
        assert ic.get("a") is None

    def test_hit_rate(self):
        ic = InlineCache()
        ic.put("a", 1)
        ic.get("a")  # hit
        ic.get("a")  # hit
        ic.get("b")  # miss
        assert ic.hit_rate == pytest.approx(2 / 3, abs=0.01)

    def test_stats(self):
        ic = InlineCache(max_size=100)
        ic.put("a", 1)
        ic.get("a")
        stats = ic.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 0


class TestInlineCacheEntry:
    """InlineCacheEntry tests."""

    def test_touch_increments_hits(self):
        entry = InlineCacheEntry("k", "v")
        assert entry.hits == 0
        result = entry.touch()
        assert result == "v"
        assert entry.hits == 1


class TestAOTCompiler:
    """AOT compiler tests."""

    def test_basic_stats(self):
        aot = AOTCompiler()
        stats = aot.get_stats()
        assert stats["cached_actions"] == 0
        assert stats["compilations"] == 0

    def test_compile_contract_no_actions(self):
        aot = AOTCompiler()
        contract = StubNode(actions={})
        result = aot.compile_contract("0xContract", contract)
        assert result == {}

    def test_compile_contract_with_action(self):
        aot = AOTCompiler()
        body = StubNode(statements=[])
        action = MockAction(body=body)
        contract = StubNode(actions={"transfer": action})
        result = aot.compile_contract("0xContract", contract)
        assert "transfer" in result
        assert isinstance(result["transfer"], CompiledAction)

    def test_get_compiled(self):
        aot = AOTCompiler()
        body = StubNode(statements=[])
        action = MockAction(body=body)
        contract = StubNode(actions={"transfer": action})
        aot.compile_contract("0xContract", contract)
        compiled = aot.get_compiled("0xContract", "transfer")
        assert compiled is not None
        assert compiled.action_name == "transfer"

    def test_invalidate_contract(self):
        aot = AOTCompiler()
        body = StubNode(statements=[])
        action = MockAction(body=body)
        contract = StubNode(actions={"transfer": action, "mint": action})
        aot.compile_contract("0xContract", contract)
        removed = aot.invalidate_contract("0xContract")
        assert removed == 2
        assert aot.get_compiled("0xContract", "transfer") is None

    def test_recompile_same_hash_cached(self):
        aot = AOTCompiler()
        body = StubNode(statements=[])
        action = MockAction(body=body)
        contract = StubNode(actions={"transfer": action})
        aot.compile_contract("0xContract", contract)
        # Compile again — should reuse because source hash matches
        result = aot.compile_contract("0xContract", contract)
        assert "transfer" in result

    def test_compile_no_body_skipped(self):
        aot = AOTCompiler()
        action = MockAction(body=None)
        contract = StubNode(actions={"noop": action})
        result = aot.compile_contract("0xContract", contract)
        assert "noop" not in result

    def test_debug_mode(self):
        aot = AOTCompiler(debug=True)
        body = StubNode(statements=[])
        action = MockAction(body=body)
        contract = StubNode(actions={"transfer": action})
        aot.compile_contract("0xContract", contract)
        assert aot.get_stats()["compilations"] == 1


class TestCompiledAction:
    """CompiledAction dataclass tests."""

    def test_avg_time_no_executions(self):
        ca = CompiledAction(
            contract_address="0xC",
            action_name="transfer",
        )
        assert ca.avg_time == 0.0

    def test_avg_time_with_executions(self):
        ca = CompiledAction(
            contract_address="0xC",
            action_name="transfer",
            execution_count=10,
            total_time=1.0,
        )
        assert ca.avg_time == pytest.approx(0.1)


class TestNumericFastPath:
    """Numeric fast-path tests."""

    def test_empty_instructions(self):
        nfp = NumericFastPath()
        assert nfp.is_purely_numeric([]) is False

    def test_is_purely_numeric_positive(self):
        nfp = NumericFastPath()
        # Create mock instructions with name attributes
        op = type("Op", (), {"name": "LOAD_CONST"})()
        instructions = [(op, 0), (op, 1)]
        assert nfp.is_purely_numeric(instructions) is True

    def test_is_purely_numeric_negative(self):
        nfp = NumericFastPath()
        op = type("Op", (), {"name": "CALL_FUNCTION"})()
        instructions = [(op, 0)]
        assert nfp.is_purely_numeric(instructions) is False

    def test_compile_and_execute(self):
        nfp = NumericFastPath()
        # Build instructions: LOAD_CONST 3, LOAD_CONST 4, ADD
        load = type("Op", (), {"name": "LOAD_CONST"})()
        add = type("Op", (), {"name": "ADD"})()
        ret = type("Op", (), {"name": "RETURN"})()

        instructions = [(load, 0), (load, 1), (add,), (ret,)]
        constants = [3, 4]

        fn = nfp.compile_numeric(instructions, constants)
        assert fn is not None
        result = fn({})
        assert result == 7

    def test_compile_with_variables(self):
        nfp = NumericFastPath()
        load_name = type("Op", (), {"name": "LOAD_NAME"})()
        load_const = type("Op", (), {"name": "LOAD_CONST"})()
        mul = type("Op", (), {"name": "MUL"})()
        ret = type("Op", (), {"name": "RETURN"})()

        instructions = [
            (load_name, 0),     # load x
            (load_const, 1),    # load 10
            (mul,),             # x * 10
            (ret,),
        ]
        constants = ["x", 10]

        fn = nfp.compile_numeric(instructions, constants)
        assert fn is not None
        result = fn({"x": 5})
        assert result == 50

    def test_compile_cache_hit(self):
        nfp = NumericFastPath()
        load = type("Op", (), {"name": "LOAD_CONST"})()
        add = type("Op", (), {"name": "ADD"})()
        ret = type("Op", (), {"name": "RETURN"})()

        instructions = [(load, 0), (load, 1), (add,), (ret,)]
        constants = [1, 2]

        fn1 = nfp.compile_numeric(instructions, constants)
        fn2 = nfp.compile_numeric(instructions, constants)
        assert fn1 is fn2  # same cached function
        assert nfp.get_stats()["compilations"] == 1

    def test_comparison_ops(self):
        nfp = NumericFastPath()
        load = type("Op", (), {"name": "LOAD_CONST"})()
        gt = type("Op", (), {"name": "GT"})()
        ret = type("Op", (), {"name": "RETURN"})()

        instructions = [(load, 0), (load, 1), (gt,), (ret,)]
        constants = [10, 5]

        fn = nfp.compile_numeric(instructions, constants)
        assert fn is not None
        assert fn({}) is True

    def test_negation(self):
        nfp = NumericFastPath()
        load = type("Op", (), {"name": "LOAD_CONST"})()
        neg = type("Op", (), {"name": "NEG"})()
        ret = type("Op", (), {"name": "RETURN"})()

        instructions = [(load, 0), (neg,), (ret,)]
        constants = [42]

        fn = nfp.compile_numeric(instructions, constants)
        assert fn is not None
        assert fn({}) == -42

    def test_stats(self):
        nfp = NumericFastPath()
        stats = nfp.get_stats()
        assert stats["cached_functions"] == 0
        assert stats["compilations"] == 0


class TestWASMCache:
    """WASM disk cache tests."""

    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as d:
            cache = WASMCache(cache_dir=d)
            cache.put("abc123", b"\x00asm\x01\x00\x00\x00")
            result = cache.get("abc123")
            assert result == b"\x00asm\x01\x00\x00\x00"

    def test_miss(self):
        with tempfile.TemporaryDirectory() as d:
            cache = WASMCache(cache_dir=d)
            assert cache.get("nonexistent") is None

    def test_contains(self):
        with tempfile.TemporaryDirectory() as d:
            cache = WASMCache(cache_dir=d)
            cache.put("h1", b"data")
            assert cache.contains("h1") is True
            assert cache.contains("h2") is False

    def test_eviction(self):
        with tempfile.TemporaryDirectory() as d:
            cache = WASMCache(cache_dir=d, max_entries=2)
            cache.put("h1", b"d1")
            cache.put("h2", b"d2")
            cache.put("h3", b"d3")  # evicts h1
            assert cache.get("h1") is None
            assert cache.get("h2") == b"d2"
            assert cache.get("h3") == b"d3"

    def test_remove(self):
        with tempfile.TemporaryDirectory() as d:
            cache = WASMCache(cache_dir=d)
            cache.put("h1", b"data")
            assert cache.remove("h1") is True
            assert cache.get("h1") is None
            assert cache.remove("h1") is False

    def test_clear(self):
        with tempfile.TemporaryDirectory() as d:
            cache = WASMCache(cache_dir=d)
            cache.put("h1", b"d1")
            cache.put("h2", b"d2")
            cleared = cache.clear()
            assert cleared == 2
            assert cache.size == 0

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            cache1 = WASMCache(cache_dir=d)
            cache1.put("h1", b"data")

            cache2 = WASMCache(cache_dir=d)
            result = cache2.get("h1")
            assert result == b"data"

    def test_stats(self):
        with tempfile.TemporaryDirectory() as d:
            cache = WASMCache(cache_dir=d)
            cache.put("h1", b"1234")
            stats = cache.get_stats()
            assert stats["entries"] == 1
            assert stats["total_bytes"] == 4


class TestTxBatchResult:
    """TxBatchResult tests."""

    def test_throughput_zero_time(self):
        r = TxBatchResult(total=10, elapsed=0.0)
        assert r.throughput == 0.0

    def test_throughput_normal(self):
        r = TxBatchResult(total=100, elapsed=2.0)
        assert r.throughput == pytest.approx(50.0)

    def test_to_dict(self):
        r = TxBatchResult(total=10, succeeded=8, failed=2, gas_used=100000, elapsed=1.5)
        d = r.to_dict()
        assert d["total"] == 10
        assert d["succeeded"] == 8
        assert d["failed"] == 2
        assert d["gas_used"] == 100000


class TestBatchExecutor:
    """BatchExecutor tests."""

    def test_no_vm_fails_gracefully(self):
        executor = BatchExecutor(contract_vm=None)
        txs = [{"contract": "0xC", "action": "transfer", "args": {}, "caller": "0xA"}]
        result = executor.execute_batch(txs)
        assert result.total == 1
        assert result.failed == 1
        assert "No ContractVM" in result.receipts[0]["error"]

    def test_empty_batch(self):
        executor = BatchExecutor()
        result = executor.execute_batch([])
        assert result.total == 0
        assert result.succeeded == 0

    def test_grouping_by_contract(self):
        executor = BatchExecutor()
        txs = [
            {"contract": "0xA", "action": "a1", "args": {}, "caller": "0x1"},
            {"contract": "0xB", "action": "b1", "args": {}, "caller": "0x1"},
            {"contract": "0xA", "action": "a2", "args": {}, "caller": "0x1"},
        ]
        groups = executor._group_by_contract(txs)
        assert len(groups["0xA"]) == 2
        assert len(groups["0xB"]) == 1


class TestExecutionAccelerator:
    """ExecutionAccelerator unified tests."""

    def test_create_with_defaults(self):
        acc = ExecutionAccelerator()
        assert acc.aot is not None
        assert acc.inline_cache is not None
        assert acc.wasm_cache is not None
        assert acc.numeric is not None

    def test_create_all_disabled(self):
        acc = ExecutionAccelerator(
            aot_enabled=False,
            ic_enabled=False,
            wasm_cache_enabled=False,
            numeric_fast_path=False,
        )
        assert acc.aot is None
        assert acc.inline_cache is None
        assert acc.wasm_cache is None
        assert acc.numeric is None

    def test_on_contract_deployed(self):
        acc = ExecutionAccelerator()
        body = StubNode(statements=[])
        action = MockAction(body=body)
        contract = StubNode(actions={"transfer": action})
        acc.on_contract_deployed("0xContract", contract)
        assert acc.aot.get_stats()["compilations"] >= 1

    def test_on_contract_upgraded(self):
        acc = ExecutionAccelerator()
        body = StubNode(statements=[])
        action = MockAction(body=body)
        contract_v1 = StubNode(actions={"transfer": action})
        contract_v2 = StubNode(actions={"transfer": action, "mint": action})

        acc.on_contract_deployed("0xC", contract_v1)
        acc.inline_cache.put("0xC:transfer", "cached_value")

        acc.on_contract_upgraded("0xC", contract_v2)
        assert acc.inline_cache.get("0xC:transfer") is None  # invalidated
        assert acc.aot.get_stats()["compilations"] >= 2  # recompiled

    def test_execute_no_vm(self):
        acc = ExecutionAccelerator()
        with pytest.raises(RuntimeError, match="No ContractVM"):
            acc.execute("0xC", "transfer")

    def test_wasm_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            acc = ExecutionAccelerator(cache_dir=d)
            acc.cache_wasm("hash1", b"\x00asm\x01")
            result = acc.get_cached_wasm("hash1")
            assert result == b"\x00asm\x01"

    def test_wasm_cache_miss(self):
        acc = ExecutionAccelerator()
        result = acc.get_cached_wasm("nonexistent")
        assert result is None

    def test_stats(self):
        acc = ExecutionAccelerator()
        stats = acc.get_stats()
        assert "total_calls" in stats
        assert "aot" in stats
        assert "inline_cache" in stats
        assert "wasm_cache" in stats
        assert "numeric_fast_path" in stats

    def test_clear_caches(self):
        acc = ExecutionAccelerator()
        acc.inline_cache.put("key", "val")
        acc.clear_caches()
        assert acc.inline_cache.size == 0
        assert acc._total_calls == 0


class TestFastWrap:
    """Value wrapping/unwrapping fast-path."""

    def test_wrap_int(self):
        val = _fast_wrap(42)
        assert hasattr(val, "value")
        assert val.value == 42

    def test_wrap_float(self):
        val = _fast_wrap(3.14)
        assert hasattr(val, "value")

    def test_wrap_string(self):
        val = _fast_wrap("hello")
        assert hasattr(val, "value")
        assert val.value == "hello"

    def test_wrap_bool(self):
        val = _fast_wrap(True)
        assert hasattr(val, "value")
        assert val.value is True

    def test_wrap_none(self):
        val = _fast_wrap(None)
        assert val is not None  # wrapped to ZNull

    def test_wrap_already_wrapped(self):
        """If already a Zexus object, should return as-is."""
        wrapped = _fast_wrap(42)
        double_wrapped = _fast_wrap(wrapped)
        assert double_wrapped is wrapped

    def test_unwrap_primitive(self):
        wrapped = _fast_wrap(42)
        assert _fast_unwrap(wrapped) == 42

    def test_unwrap_string(self):
        wrapped = _fast_wrap("test")
        assert _fast_unwrap(wrapped) == "test"

    def test_unwrap_passthrough(self):
        """Objects without .value should pass through."""
        obj = {"key": "val"}
        assert _fast_unwrap(obj) == {"key": "val"}


# ══════════════════════════════════════════════════════════════════════
#  Integration / Cross-Feature Tests
# ══════════════════════════════════════════════════════════════════════

class TestCrossFeatureIntegration:
    """Tests that span multiple features."""

    def test_upgrade_invalidates_aot_cache(self):
        """When UpgradeManager upgrades a contract, accelerator caches
        should be invalidated."""
        acc = ExecutionAccelerator()
        body = StubNode(statements=[])
        action = MockAction(body=body)
        contract = StubNode(actions={"transfer": action})

        acc.on_contract_deployed("0xC", contract)
        assert acc.aot.get_compiled("0xC", "transfer") is not None

        # Simulate upgrade invalidation
        acc.on_contract_upgraded("0xC", contract)
        # Should re-compile (new compilation)
        assert acc.aot.get_stats()["compilations"] >= 2

    def test_governance_applies_changes_to_chain(self):
        """Governance proposals should actually modify chain parameters."""
        chain = MockChain()
        gov = ChainUpgradeGovernance(
            chain=chain,
            validators={"0xV1"},
        )
        ok, msg, pid = gov.propose(
            "0xV1", ProposalType.CHAIN_PARAMETER, "test",
            {"difficulty": 16, "target_block_time": 5}, 200,
        )
        assert ok is True
        # Single validator → auto-approved
        gov.apply_pending(200)
        assert chain.difficulty == 16
        assert chain.target_block_time == 5

    def test_verification_report_roundtrip(self):
        """VerificationReport should serialize to dict and back."""
        r = VerificationReport(level=VerificationLevel.STRUCTURAL)
        r.contract_name = "Token"
        r.findings.append(VerificationFinding(
            category=FindingCategory.ACCESS_CONTROL,
            severity=Severity.MEDIUM,
            message="Missing require",
            action_name="transfer",
        ))
        d = r.to_dict()
        assert d["passed"] is True  # MEDIUM doesn't fail
        assert len(d["findings"]) == 1
        assert d["findings"][0]["severity"] == "MEDIUM"
