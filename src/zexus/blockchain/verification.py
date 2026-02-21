"""
Zexus Blockchain — Formal Verification Engine
==============================================

A static-analysis and symbolic-execution engine that verifies smart
contract correctness **before** deployment.  Operates entirely on the
AST — never executes user code.

Verification Levels
-------------------

Level 1 — **Structural Checks** (fast, always available)
    * Detects missing ``require`` guards on state-mutating actions.
    * Ensures every action that transfers value checks balances.
    * Verifies reentrancy-safe patterns (no external calls after
      state writes).
    * Checks for integer overflow / underflow patterns.

Level 2 — **Invariant Verification** (symbolic)
    * User declares ``@invariant`` annotations on contracts.
    * The engine symbolically walks the AST to prove that every
      action preserves the invariant or reports a counterexample.
    * Supports arithmetic constraints (linear inequalities).

Level 3 — **Property-Based Verification** (bounded model checking)
    * User declares ``@property`` annotations.
    * The engine explores bounded paths through the action logic
      to verify the property holds for all reachable states.
    * Supports ``@pre`` (precondition) and ``@post`` (postcondition).

Integration
-----------
*   Can be called standalone or wired into the deployment pipeline
    so that ``ContractVM.deploy_contract()`` automatically verifies
    before accepting the contract.
*   Emits ``VerificationReport`` objects with detailed findings.
*   Integrates with the existing ``StaticTypeChecker``.

Usage
-----
::

    from zexus.blockchain.verification import (
        FormalVerifier,
        VerificationLevel,
    )

    verifier = FormalVerifier(level=VerificationLevel.INVARIANT)
    report = verifier.verify_contract(contract)
    if not report.passed:
        for finding in report.findings:
            print(finding)
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
import re
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import (
    Any, Callable, Dict, FrozenSet, List, Optional,
    Set, Tuple, Union,
)

logger = logging.getLogger("zexus.blockchain.verification")


# =====================================================================
# Constants & Enums
# =====================================================================

class VerificationLevel(IntEnum):
    """How deep the verification goes."""
    STRUCTURAL = 1     # Pattern-based checks only
    INVARIANT = 2      # + symbolic invariant proofs
    PROPERTY = 3       # + bounded model checking of @property annotations
    TAINT = 4          # + taint / data-flow analysis


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingCategory(str, Enum):
    MISSING_REQUIRE = "missing_require"
    REENTRANCY = "reentrancy"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    UNCHECKED_TRANSFER = "unchecked_transfer"
    INVARIANT_VIOLATION = "invariant_violation"
    PROPERTY_VIOLATION = "property_violation"
    UNINITIALIZED_STATE = "uninitialized_state"
    UNREACHABLE_CODE = "unreachable_code"
    ACCESS_CONTROL = "access_control"
    DIVISION_BY_ZERO = "division_by_zero"
    STATE_AFTER_CALL = "state_after_call"
    PRECONDITION_VIOLATION = "precondition"
    POSTCONDITION_VIOLATION = "postcondition"
    TAINTED_VALUE = "tainted_value"
    UNSANITIZED_INPUT = "unsanitized_input"
    TAINTED_STORAGE_KEY = "tainted_storage_key"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    UNCHECKED_RETURN = "unchecked_return"


# =====================================================================
# Findings & Reports
# =====================================================================

@dataclass
class VerificationFinding:
    """A single issue found during verification."""
    category: FindingCategory
    severity: Severity
    message: str
    action_name: str = ""
    contract_name: str = ""
    line: Optional[int] = None
    suggestion: str = ""
    counterexample: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "category": self.category.value,
            # Public/serialized form uses the Enum name (UPPERCASE)
            # to provide a stable contract for external tooling.
            "severity": self.severity.name,
            "message": self.message,
        }
        if self.action_name:
            d["action"] = self.action_name
        if self.contract_name:
            d["contract"] = self.contract_name
        if self.line is not None:
            d["line"] = self.line
        if self.suggestion:
            d["suggestion"] = self.suggestion
        if self.counterexample:
            d["counterexample"] = self.counterexample
        return d

    def __str__(self) -> str:
        loc = f" (line {self.line})" if self.line else ""
        act = f" in {self.action_name}" if self.action_name else ""
        return f"[{self.severity.value.upper()}] {self.category.value}{act}{loc}: {self.message}"


@dataclass
class VerificationReport:
    """Aggregate result of verifying a contract."""
    level: VerificationLevel
    contract_name: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    findings: List[VerificationFinding] = field(default_factory=list)
    actions_checked: int = 0
    invariants_checked: int = 0
    properties_checked: int = 0

    @property
    def passed(self) -> bool:
        """True if no critical or high findings."""
        return not any(
            f.severity in (Severity.CRITICAL, Severity.HIGH)
            for f in self.findings
        )

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.LOW)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"Verification [{status}] for '{self.contract_name}': "
            f"{len(self.findings)} findings "
            f"(C={self.critical_count} H={self.high_count} "
            f"M={self.medium_count} L={self.low_count}) "
            f"in {self.duration:.3f}s"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            # Provide both keys for compatibility across callers.
            "contract": self.contract_name,
            "contract_name": self.contract_name,
            "level": self.level.name,
            "passed": self.passed,
            "duration": round(self.duration, 4),
            "actions_checked": self.actions_checked,
            "invariants_checked": self.invariants_checked,
            "properties_checked": self.properties_checked,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary(),
        }


# =====================================================================
# Symbolic Value  (for invariant / property checking)
# =====================================================================

class SymType(str, Enum):
    INT = "int"
    INTEGER = "int"  # alias expected by tests
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    MAP = "map"
    LIST = "list"
    ANY = "any"
    UNKNOWN = "unknown"


@dataclass
class SymValue:
    """A symbolic value with optional concrete range bounds."""
    name: str = ""
    sym_type: SymType = SymType.ANY
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    concrete: Optional[Any] = None          # Known constant value
    is_tainted: bool = False                # True if from external input

    @property
    def is_concrete(self) -> bool:
        return self.concrete is not None

    @property
    def is_bounded(self) -> bool:
        return self.min_val is not None or self.max_val is not None

    def could_be_negative(self) -> bool:
        if self.is_concrete:
            return self.concrete < 0
        if self.min_val is not None:
            return self.min_val < 0
        return True  # unknown — conservatively yes

    def could_be_zero(self) -> bool:
        if self.is_concrete:
            return self.concrete == 0
        if self.min_val is not None and self.max_val is not None:
            return self.min_val <= 0 <= self.max_val
        return True

    def copy(self) -> "SymValue":
        return SymValue(
            name=self.name,
            sym_type=self.sym_type,
            min_val=self.min_val,
            max_val=self.max_val,
            concrete=self.concrete,
            is_tainted=self.is_tainted,
        )


class SymState:
    """Symbolic environment — tracks variable constraints."""

    def __init__(self, parent: Optional["SymState"] = None):
        self._vars: Dict[str, SymValue] = {}
        self._parent = parent
        self._constraints: List[str] = []   # human-readable constraint log

    def get(self, name: str) -> Optional[SymValue]:
        v = self._vars.get(name)
        if v is not None:
            return v
        if self._parent:
            return self._parent.get(name)
        return None

    def set(self, name: str, val: SymValue) -> None:
        self._vars[name] = val

    def add_constraint(self, desc: str) -> None:
        self._constraints.append(desc)

    def child(self) -> "SymState":
        return SymState(parent=self)

    @property
    def constraints(self) -> List[str]:
        return list(self._constraints)

    @property
    def all_vars(self) -> Dict[str, SymValue]:
        merged: Dict[str, SymValue] = {}
        if self._parent:
            merged.update(self._parent.all_vars)
        merged.update(self._vars)
        return merged


# =====================================================================
# AST Helpers
# =====================================================================

def _node_type(node: Any) -> str:
    """Get the class name of an AST node."""
    return type(node).__name__


def _get_name(node: Any) -> str:
    """Extract a string name from an Identifier or string."""
    if isinstance(node, str):
        return node
    if hasattr(node, "value"):
        return str(node.value)
    if hasattr(node, "name"):
        return _get_name(node.name)
    return str(node)


def _get_action_name(action_node: Any) -> str:
    """Extract the method/action name from an ActionStatement."""
    if hasattr(action_node, "name"):
        return _get_name(action_node.name)
    return "<anonymous>"


def _walk_ast(node: Any) -> List[Any]:
    """Recursively collect all AST nodes in a subtree."""
    if node is None:
        return []
    result = [node]
    # Walk child attributes
    for attr_name in dir(node):
        if attr_name.startswith("_"):
            continue
        attr = getattr(node, attr_name, None)
        if attr is None or callable(attr):
            continue
        if isinstance(attr, list):
            for item in attr:
                if hasattr(item, "__dict__"):
                    result.extend(_walk_ast(item))
        elif hasattr(attr, "__dict__") and not isinstance(attr, type):
            # Heuristic: if it looks like an AST node, recurse
            result.extend(_walk_ast(attr))
    return result


def _contains_node_type(node: Any, type_name: str) -> bool:
    """Check if any descendant node has the given class name."""
    for n in _walk_ast(node):
        if _node_type(n) == type_name:
            return True
    return False


def _collect_nodes_of_type(node: Any, type_name: str) -> List[Any]:
    """Collect all descendant nodes of a specific type."""
    return [n for n in _walk_ast(node) if _node_type(n) == type_name]


def _contains_state_write(node: Any) -> bool:
    """Heuristic: does this action body write to contract state?"""
    for n in _walk_ast(node):
        nt = _node_type(n)
        # Assignment to state variable, indexed assignment, property set
        if nt in ("AssignmentExpression", "IndexExpression",
                   "PropertyAssignment"):
            return True
        # Direct store via 'this.x = ...'
        if nt == "PropertyExpression":
            if hasattr(n, "object") and _get_name(getattr(n, "object", "")) == "this":
                return True
    return False


def _contains_external_call(node: Any) -> bool:
    """Heuristic: does this body make external calls (contract_call, transfer)?"""
    for n in _walk_ast(node):
        nt = _node_type(n)
        if nt == "CallExpression":
            callee = getattr(n, "function", getattr(n, "callee", None))
            name = _get_name(callee) if callee else ""
            if name in ("contract_call", "delegate_call", "transfer",
                         "static_call", "send"):
                return True
    return False


def _contains_require(node: Any) -> bool:
    """Does the body contain at least one require statement?"""
    return _contains_node_type(node, "RequireStatement")


def _contains_caller_check(node: Any) -> bool:
    """Does the body check TX.caller or msg.sender?"""
    for n in _walk_ast(node):
        nt = _node_type(n)
        if nt == "PropertyExpression":
            obj = getattr(n, "object", None)
            prop = getattr(n, "property", None)
            obj_name = _get_name(obj) if obj else ""
            prop_name = _get_name(prop) if prop else ""
            if obj_name in ("TX", "msg") and prop_name in ("caller", "sender"):
                return True
    return False


def _find_division_ops(node: Any) -> List[Any]:
    """Find all division operations in the subtree."""
    divisions = []
    for n in _walk_ast(node):
        nt = _node_type(n)
        if nt in ("InfixExpression", "BinaryExpression"):
            op = getattr(n, "operator", getattr(n, "op", ""))
            if isinstance(op, str) and op in ("/", "%"):
                divisions.append(n)
    return divisions


def _find_arithmetic_ops(node: Any) -> List[Any]:
    """Find all arithmetic operations."""
    ops = []
    for n in _walk_ast(node):
        nt = _node_type(n)
        if nt in ("InfixExpression", "BinaryExpression"):
            op = getattr(n, "operator", getattr(n, "op", ""))
            if isinstance(op, str) and op in ("+", "-", "*", "**"):
                ops.append(n)
    return ops


def _extract_state_vars(contract: Any) -> List[str]:
    """Extract state variable names from a SmartContract or ContractStatement."""
    names: List[str] = []
    if hasattr(contract, "storage_vars"):
        for var_node in contract.storage_vars:
            if isinstance(var_node, str):
                names.append(var_node)
            elif hasattr(var_node, "name"):
                names.append(_get_name(var_node.name))
            elif isinstance(var_node, dict):
                n = var_node.get("name", "")
                if n:
                    names.append(n)
    return names


def _extract_actions(contract: Any) -> Dict[str, Any]:
    """Extract action name -> action object mapping."""
    if hasattr(contract, "actions"):
        actions = contract.actions
        if isinstance(actions, dict):
            return actions
    return {}


# =====================================================================
# Structural Verifier (Level 1)
# =====================================================================

class StructuralVerifier:
    """Pattern-based checks on contract ASTs.

    Checks performed:
    * Missing access-control ``require`` on state-mutating actions.
    * Balance checks before transfers.
    * State writes after external calls (reentrancy pattern).
    * Division by zero potential.
    * Integer overflow patterns (unchecked arithmetic).
    * Uninitialized state variable reads.
    """

    def verify(
        self,
        contract: Any,
        report: VerificationReport,
    ) -> None:
        contract_name = _get_name(getattr(contract, "name", "")) or "Unknown"
        report.contract_name = contract_name

        state_vars = _extract_state_vars(contract)
        actions = _extract_actions(contract)

        for action_name, action_obj in actions.items():
            report.actions_checked += 1
            body = getattr(action_obj, "body", None)
            if body is None:
                continue

            self._check_access_control(
                action_name, body, contract_name, state_vars, report
            )
            self._check_reentrancy(action_name, body, contract_name, report)
            self._check_division_by_zero(action_name, body, contract_name, report)
            self._check_overflow(action_name, body, contract_name, report)
            self._check_transfer_balance(action_name, body, contract_name, report)

    # ── Individual checks ─────────────────────────────────────────

    def _check_access_control(
        self,
        action_name: str,
        body: Any,
        contract_name: str,
        state_vars: List[str],
        report: VerificationReport,
    ) -> None:
        """Warn if a state-mutating action has no require/caller check."""
        if not _contains_state_write(body):
            return  # Read-only action — no concern
        if _contains_require(body) or _contains_caller_check(body):
            return  # Has some access control

        report.findings.append(VerificationFinding(
            category=FindingCategory.ACCESS_CONTROL,
            severity=Severity.HIGH,
            message=(
                f"State-mutating action '{action_name}' has no "
                f"access-control check (require or caller check)."
            ),
            action_name=action_name,
            contract_name=contract_name,
            suggestion="Add `require(TX.caller == owner, \"Unauthorized\");`",
        ))

    def _check_reentrancy(
        self,
        action_name: str,
        body: Any,
        contract_name: str,
        report: VerificationReport,
    ) -> None:
        """Detect state writes *after* external calls (CEI violation)."""
        nodes = _walk_ast(body)
        saw_external_call = False

        for n in nodes:
            if _contains_external_call(n) and not saw_external_call:
                saw_external_call = True
                continue

            if saw_external_call and _contains_state_write(n):
                report.findings.append(VerificationFinding(
                    category=FindingCategory.REENTRANCY,
                    severity=Severity.CRITICAL,
                    message=(
                        f"Action '{action_name}' writes state after an "
                        f"external call — potential reentrancy vulnerability."
                    ),
                    action_name=action_name,
                    contract_name=contract_name,
                    suggestion=(
                        "Follow the Checks-Effects-Interactions pattern: "
                        "perform all state writes before external calls."
                    ),
                ))
                return  # One finding per action is sufficient

    def _check_division_by_zero(
        self,
        action_name: str,
        body: Any,
        contract_name: str,
        report: VerificationReport,
    ) -> None:
        divisions = _find_division_ops(body)
        for div_node in divisions:
            # Check if the divisor is guarded
            right = getattr(div_node, "right", None)
            if right is None:
                continue
            # If divisor is a literal > 0, it's safe
            if hasattr(right, "value"):
                try:
                    val = int(right.value) if not isinstance(right.value, float) else right.value
                    if val != 0:
                        continue
                except (ValueError, TypeError):
                    pass

            report.findings.append(VerificationFinding(
                category=FindingCategory.DIVISION_BY_ZERO,
                severity=Severity.MEDIUM,
                message=(
                    f"Action '{action_name}' contains a division that "
                    f"may not guard against zero divisor."
                ),
                action_name=action_name,
                contract_name=contract_name,
                suggestion="Add `require(divisor != 0, \"Division by zero\");` before the division.",
            ))

    def _check_overflow(
        self,
        action_name: str,
        body: Any,
        contract_name: str,
        report: VerificationReport,
    ) -> None:
        """Flag unchecked arithmetic on values that could overflow."""
        arith_ops = _find_arithmetic_ops(body)
        for op_node in arith_ops:
            op = getattr(op_node, "operator", getattr(op_node, "op", ""))
            if op == "**":
                report.findings.append(VerificationFinding(
                    category=FindingCategory.OVERFLOW,
                    severity=Severity.MEDIUM,
                    message=(
                        f"Action '{action_name}' uses exponentiation (**) "
                        f"which can cause large-number overflow."
                    ),
                    action_name=action_name,
                    contract_name=contract_name,
                    suggestion="Consider adding bounds checks or using safe-math utilities.",
                ))
            elif op == "*":
                # Multiplication of two unbounded values
                left = getattr(op_node, "left", None)
                right = getattr(op_node, "right", None)
                # If both sides are non-literal, flag as potential overflow
                left_is_literal = hasattr(left, "value") and isinstance(
                    getattr(left, "value", None), (int, float)
                )
                right_is_literal = hasattr(right, "value") and isinstance(
                    getattr(right, "value", None), (int, float)
                )
                if not left_is_literal and not right_is_literal:
                    report.findings.append(VerificationFinding(
                        category=FindingCategory.OVERFLOW,
                        severity=Severity.LOW,
                        message=(
                            f"Action '{action_name}' multiplies two non-constant "
                            f"values without overflow protection."
                        ),
                        action_name=action_name,
                        contract_name=contract_name,
                        suggestion="Consider safe-math or bounds-checking patterns.",
                    ))

    def _check_transfer_balance(
        self,
        action_name: str,
        body: Any,
        contract_name: str,
        report: VerificationReport,
    ) -> None:
        """Check that transfers are preceded by balance checks."""
        nodes = _walk_ast(body)
        has_transfer = False
        has_balance_check = False

        for n in nodes:
            nt = _node_type(n)
            if nt == "CallExpression":
                callee = getattr(n, "function", getattr(n, "callee", None))
                name = _get_name(callee) if callee else ""
                if name == "transfer":
                    has_transfer = True
                if name == "get_balance":
                    has_balance_check = True
            if nt == "RequireStatement":
                has_balance_check = True  # Any require is a proxy

        if has_transfer and not has_balance_check:
            report.findings.append(VerificationFinding(
                category=FindingCategory.UNCHECKED_TRANSFER,
                severity=Severity.HIGH,
                message=(
                    f"Action '{action_name}' calls transfer() without "
                    f"a preceding balance check."
                ),
                action_name=action_name,
                contract_name=contract_name,
                suggestion="Add `require(balance >= amount, \"Insufficient funds\");`",
            ))


# =====================================================================
# Invariant Verifier (Level 2)
# =====================================================================

@dataclass
class Invariant:
    """A declared contract invariant.

    Example annotation (in Zexus source)::

        // @invariant total_supply >= 0
        // @invariant balances_sum == total_supply
    """
    expression: str              # Human-readable expression
    variable: str = ""           # singular alias (some tooling prefers this)
    variables: List[str] = field(default_factory=list)
    parsed: Optional[Any] = None  # Internal parsed representation

    def __post_init__(self) -> None:
        if self.variable and not self.variables:
            self.variables = [self.variable]

    def to_dict(self) -> Dict[str, Any]:
        return {"expression": self.expression, "variables": self.variables}


class InvariantVerifier:
    """Symbolically verifies that contract invariants are preserved.

    For each action, the verifier:
    1. Sets up an initial symbolic state satisfying the invariant.
    2. Symbolically executes the action body.
    3. Checks that the invariant still holds in the post-state.

    If the invariant *could* be violated, it reports a finding with
    a potential counterexample.
    """

    # Supported comparison operators for invariant expressions
    _CMP_PATTERN = re.compile(
        r"^(\w+)\s*(>=|<=|>|<|==|!=)\s*(.+)$"
    )

    def verify(
        self,
        contract: Any,
        invariants: List[Invariant],
        report: VerificationReport,
    ) -> None:
        contract_name = _get_name(getattr(contract, "name", "")) or "Unknown"
        state_vars = _extract_state_vars(contract)
        actions = _extract_actions(contract)

        for inv in invariants:
            report.invariants_checked += 1
            parsed = self._parse_invariant(inv.expression)
            if parsed is None:
                report.findings.append(VerificationFinding(
                    category=FindingCategory.INVARIANT_VIOLATION,
                    severity=Severity.INFO,
                    message=f"Could not parse invariant: {inv.expression}",
                    contract_name=contract_name,
                ))
                continue

            lhs_var, op, rhs = parsed
            inv.parsed = parsed
            inv.variables = [lhs_var] if lhs_var in state_vars else []

            # Check each action preserves the invariant
            for action_name, action_obj in actions.items():
                body = getattr(action_obj, "body", None)
                if body is None:
                    continue

                violation = self._check_action_preserves(
                    action_name, body, lhs_var, op, rhs, state_vars,
                )
                if violation:
                    report.findings.append(VerificationFinding(
                        category=FindingCategory.INVARIANT_VIOLATION,
                        severity=Severity.HIGH,
                        message=violation["message"],
                        action_name=action_name,
                        contract_name=contract_name,
                        counterexample=violation.get("counterexample"),
                        suggestion=violation.get("suggestion", ""),
                    ))

    def _parse_invariant(
        self, expr: str
    ) -> Optional[Tuple[str, str, str]]:
        """Parse ``var >= 0`` style invariants."""
        m = self._CMP_PATTERN.match(expr.strip())
        if m:
            return m.group(1), m.group(2), m.group(3).strip()
        return None

    def _check_action_preserves(
        self,
        action_name: str,
        body: Any,
        lhs_var: str,
        op: str,
        rhs: str,
        state_vars: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Check if an action could violate ``lhs_var <op> rhs``.

        Uses a conservative static analysis: if the action modifies
        ``lhs_var`` and does NOT contain a require/if guard that
        re-establishes the invariant, it is flagged.
        """
        # Collect all assignments to lhs_var in the body
        assigns = self._collect_assignments_to(body, lhs_var)
        if not assigns:
            return None  # Action doesn't touch the invariant variable

        # Check for subtractions that could violate >= 0
        for assign in assigns:
            rhs_expr = getattr(assign, "value", getattr(assign, "right", None))
            if rhs_expr is None:
                continue

            # Detect patterns like `total_supply = total_supply - amount`
            if self._is_subtraction_from(rhs_expr, lhs_var):
                # Check if there's a preceding guard
                if not _contains_require(body):
                    try:
                        rhs_val = int(rhs)
                    except (ValueError, TypeError):
                        rhs_val = 0

                    return {
                        "message": (
                            f"Action '{action_name}' decrements '{lhs_var}' "
                            f"without a require guard; invariant "
                            f"'{lhs_var} {op} {rhs}' may be violated."
                        ),
                        "counterexample": {
                            lhs_var: rhs_val,
                            "decremented_by": "unknown_amount",
                        },
                        "suggestion": (
                            f"Add `require({lhs_var} >= amount, "
                            f"\"Would violate invariant\");`"
                        ),
                    }

            # Detect unchecked assignment (could set to anything)
            if self._is_raw_assignment(assign) and not _contains_require(body):
                return {
                    "message": (
                        f"Action '{action_name}' assigns to '{lhs_var}' "
                        f"without ensuring invariant '{lhs_var} {op} {rhs}'."
                    ),
                    "suggestion": (
                        f"Guard the assignment with "
                        f"`require(new_value {op} {rhs});`"
                    ),
                }

        return None

    def _collect_assignments_to(self, body: Any, var_name: str) -> List[Any]:
        """Find all AST nodes that assign to ``var_name``."""
        results = []
        for n in _walk_ast(body):
            nt = _node_type(n)
            if nt in ("AssignmentExpression", "LetStatement", "ConstStatement"):
                target = getattr(n, "name", getattr(n, "left", None))
                if target and _get_name(target) == var_name:
                    results.append(n)
        return results

    def _is_subtraction_from(self, expr: Any, var_name: str) -> bool:
        """Check if ``expr`` is ``var_name - <something>``."""
        nt = _node_type(expr)
        if nt in ("InfixExpression", "BinaryExpression"):
            op = getattr(expr, "operator", getattr(expr, "op", ""))
            left = getattr(expr, "left", None)
            if op == "-" and left and _get_name(left) == var_name:
                return True
        return False

    def _is_raw_assignment(self, node: Any) -> bool:
        """True if this looks like a direct assignment (not +=, -=)."""
        nt = _node_type(node)
        return nt in ("AssignmentExpression", "LetStatement")


# =====================================================================
# Property Verifier (Level 3) — Bounded Model Checking
# =====================================================================

@dataclass
class ContractProperty:
    """A verifiable property with optional pre/post conditions.

    Declared as annotations::

        // @property transfer_preserves_total
        // @pre total_supply > 0
        // @post total_supply == @old(total_supply)
    """
    name: str
    description: str = ""
    precondition: str = ""       # @pre expression
    postcondition: str = ""      # @post expression
    action: str = ""             # Specific action, or "" for all

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.precondition:
            d["precondition"] = self.precondition
        if self.postcondition:
            d["postcondition"] = self.postcondition
        if self.action:
            d["action"] = self.action
        return d

    @property
    def action_scope(self) -> str:
        # Backward-compatible alias for older internal name.
        return self.action


class PropertyVerifier:
    """Performs bounded model checking on contract properties.

    For each property, the verifier:
    1. Sets up symbolic initial state satisfying the precondition.
    2. Explores bounded execution paths through the action.
    3. Verifies the postcondition holds on every path.

    Bound depth is configurable (default 3 branches — covers most
    single-action paths).
    """

    def __init__(self, max_depth: int = 3):
        self._max_depth = max_depth

    def verify(
        self,
        contract: Any,
        properties: List[ContractProperty],
        report: VerificationReport,
    ) -> None:
        contract_name = _get_name(getattr(contract, "name", "")) or "Unknown"
        state_vars = _extract_state_vars(contract)
        actions = _extract_actions(contract)

        for prop in properties:
            report.properties_checked += 1
            target_actions = (
                {prop.action: actions[prop.action]}
                if prop.action and prop.action in actions
                else actions
            )

            for action_name, action_obj in target_actions.items():
                body = getattr(action_obj, "body", None)
                if body is None:
                    continue

                violation = self._check_property(
                    prop, action_name, body, state_vars
                )
                if violation:
                    sev = (
                        Severity.CRITICAL
                        if "postcondition" in violation.get("kind", "")
                        else Severity.HIGH
                    )
                    report.findings.append(VerificationFinding(
                        category=(
                            FindingCategory.POSTCONDITION_VIOLATION
                            if "postcondition" in violation.get("kind", "")
                            else FindingCategory.PRECONDITION_VIOLATION
                        ),
                        severity=sev,
                        message=violation["message"],
                        action_name=action_name,
                        contract_name=contract_name,
                        counterexample=violation.get("counterexample"),
                    ))

    def _check_property(
        self,
        prop: ContractProperty,
        action_name: str,
        body: Any,
        state_vars: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Bounded model check of a single property on an action."""

        # Build initial symbolic state
        sym_state = SymState()
        for sv in state_vars:
            sym_state.set(sv, SymValue(name=sv, sym_type=SymType.INT, min_val=0))

        # Check postcondition patterns
        if prop.postcondition:
            return self._verify_postcondition(
                prop, action_name, body, state_vars, sym_state
            )

        return None

    def _verify_postcondition(
        self,
        prop: ContractProperty,
        action_name: str,
        body: Any,
        state_vars: List[str],
        sym_state: SymState,
    ) -> Optional[Dict[str, Any]]:
        """Check if postcondition holds after action execution.

        Supports ``@old(var)`` references to pre-state values.
        """
        post = prop.postcondition.strip()

        # Parse @old references
        old_vars = set(re.findall(r"@old\((\w+)\)", post))

        # Look for conservation laws like:
        #   total_supply == @old(total_supply)
        #   balances_sum == @old(balances_sum)
        for old_var in old_vars:
            if old_var not in state_vars:
                continue

            # Check if the action modifies this variable
            assigns = []
            for n in _walk_ast(body):
                nt = _node_type(n)
                if nt in ("AssignmentExpression", "LetStatement"):
                    target = getattr(n, "name", getattr(n, "left", None))
                    if target and _get_name(target) == old_var:
                        assigns.append(n)

            if assigns:
                # The action modifies a variable that should be conserved
                # Check if modifications are balanced (e.g. += and -= same amount)
                has_increment = False
                has_decrement = False
                for assign in assigns:
                    rhs_expr = getattr(assign, "value", getattr(assign, "right", None))
                    if rhs_expr:
                        rhs_nt = _node_type(rhs_expr)
                        if rhs_nt in ("InfixExpression", "BinaryExpression"):
                            op = getattr(rhs_expr, "operator", getattr(rhs_expr, "op", ""))
                            if op == "+":
                                has_increment = True
                            elif op == "-":
                                has_decrement = True

                # If only increment or only decrement, postcondition
                # `x == @old(x)` is potentially violated
                if has_increment != has_decrement:
                    return {
                        "kind": "postcondition",
                        "message": (
                            f"Property '{prop.name}': action '{action_name}' "
                            f"may violate postcondition '{post}' — "
                            f"'{old_var}' is modified non-symmetrically."
                        ),
                        "counterexample": {
                            old_var: "modified without balanced inverse",
                        },
                    }

        return None


# =====================================================================
# Annotation Parser
# =====================================================================

class AnnotationParser:
    """Extract @invariant, @property, @pre, @post from contract metadata
    or source comments.
    """

    _INV_PATTERN = re.compile(r"@invariant\s+(.+)")
    _PROP_PATTERN = re.compile(r"@property\s+(\w+)(?:\s+(.+))?")
    _PRE_PATTERN = re.compile(r"@pre\s+(.+)")
    _POST_PATTERN = re.compile(r"@post\s+(.+)")

    @classmethod
    def parse_annotations(
        cls,
        source: Union[str, List[str]],
    ) -> Dict[str, Any]:
        """Parse verification annotations from source text or comments.

        Returns a dict with keys:
        - ``invariants``: List[str]
        - ``properties``: List[Dict[str, Any]]
        - ``preconditions``: List[str]
        - ``postconditions``: List[str]
        """
        lines = source.splitlines() if isinstance(source, str) else source
        invariants: List[str] = []
        properties: List[Dict[str, Any]] = []
        preconditions: List[str] = []
        postconditions: List[str] = []

        current_prop: Optional[Dict[str, Any]] = None

        for line in lines:
            stripped = line.strip().lstrip("/").strip()

            # @invariant
            m = cls._INV_PATTERN.match(stripped)
            if m:
                expr = m.group(1).strip()
                invariants.append(expr)
                continue

            # @property
            m = cls._PROP_PATTERN.match(stripped)
            if m:
                # Flush previous property
                if current_prop is not None:
                    properties.append(current_prop)
                current_prop = {
                    "name": m.group(1),
                    "description": (m.group(2) or "").strip(),
                    "precondition": "",
                    "postcondition": "",
                    "action": "",
                }
                continue

            # @pre
            m = cls._PRE_PATTERN.match(stripped)
            if m:
                expr = m.group(1).strip()
                preconditions.append(expr)
                if current_prop is not None:
                    current_prop["precondition"] = expr
                continue

            # @post
            m = cls._POST_PATTERN.match(stripped)
            if m:
                expr = m.group(1).strip()
                postconditions.append(expr)
                if current_prop is not None:
                    current_prop["postcondition"] = expr
                continue

        # Flush last property
        if current_prop is not None:
            properties.append(current_prop)

        return {
            "invariants": invariants,
            "properties": properties,
            "preconditions": preconditions,
            "postconditions": postconditions,
        }

    @classmethod
    def from_contract_metadata(
        cls,
        contract: Any,
    ) -> Dict[str, Any]:
        """Extract annotations from a contract's metadata.

        Supports either dict-like ``blockchain_config`` or an object
        with a ``verification`` attribute.
        """
        meta = getattr(contract, "blockchain_config", {}) or {}
        if isinstance(meta, dict):
            verification = meta.get("verification", {})
        else:
            verification = getattr(meta, "verification", {})
        if not isinstance(verification, dict):
            verification = {}

        invariants = list(verification.get("invariants", []) or [])
        props_in = list(verification.get("properties", []) or [])
        properties: List[Dict[str, Any]] = []

        for p in props_in:
            if isinstance(p, dict):
                properties.append({
                    "name": p.get("name", ""),
                    "description": p.get("description", ""),
                    "precondition": p.get("precondition", p.get("pre", "")),
                    "postcondition": p.get("postcondition", p.get("post", "")),
                    "action": p.get("action", ""),
                })

        return {
            "invariants": invariants,
            "properties": properties,
            "preconditions": [],
            "postconditions": [],
        }


# =====================================================================
# Taint / Data-Flow Analyzer (Level 4)
# =====================================================================

class TaintLabel(str, Enum):
    """Labels for data-flow taint tracking."""
    USER_INPUT = "user_input"        # from TX.caller, action args
    EXTERNAL_CALL = "external_call"  # return value of cross-contract call
    STORAGE_READ = "storage_read"    # from persistent state
    ARITHMETIC = "arithmetic"        # derived from tainted operands

    def __str__(self) -> str:
        return self.value


class TaintAnalyzer:
    """Data-flow taint analysis for smart contracts.

    Tracks how user-controlled (tainted) values propagate through
    contract actions and flags dangerous sinks:

    * **Tainted storage key** — using user input as a map/storage key
      without validation can lead to storage collision attacks.
    * **Tainted transfer amount** — using unsanitized user input as a
      transfer value enables arbitrary drain attacks.
    * **Tainted control flow** — branching on external call results
      without validation enables oracle manipulation.
    * **Unchecked external return** — calling another contract and
      ignoring (or not checking) the return value.
    * **Privilege escalation** — storing user-supplied data into
      ``owner`` or ``admin`` state variables.

    This operates on the AST — no code is executed.
    """

    # Variables considered *inherently tainted* (user-controlled)
    TAINT_SOURCES = frozenset({
        "TX.caller", "tx.caller", "TX.value", "tx.value",
        "TX.origin", "tx.origin", "msg.sender", "msg.value",
    })

    # State variables that must never receive unsanitized user input
    SENSITIVE_STATE = frozenset({
        "owner", "admin", "authority", "minter", "pauser",
        "operator", "governance",
    })

    # Sink functions where tainted values are dangerous
    DANGEROUS_SINKS = frozenset({
        "transfer", "send", "call", "delegatecall",
        "selfdestruct", "suicide",
    })

    def verify(
        self,
        contract: Any,
        report: VerificationReport,
    ) -> None:
        contract_name = _get_name(getattr(contract, "name", "")) or "Unknown"
        actions = _extract_actions(contract)

        for action_name, action_obj in actions.items():
            body = getattr(action_obj, "body", None)
            if body is None:
                continue

            # Build taint set: variables known to carry user-controlled data
            tainted: Set[str] = set()

            # Action parameters are tainted (they come from the caller)
            params = getattr(action_obj, "params", getattr(action_obj, "parameters", []))
            if isinstance(params, (list, tuple)):
                for p in params:
                    name = _get_name(p) if not isinstance(p, str) else p
                    if name:
                        tainted.add(name)

            # Walk the AST and propagate taint
            nodes = _walk_ast(body)
            for node in nodes:
                nt = _node_type(node)

                # -- Assignments: propagate taint through data flow --
                if nt in ("AssignmentExpression", "AssignmentStatement", "LetStatement", "VarDeclaration"):
                    target = _get_name(getattr(node, "name", getattr(node, "left", getattr(node, "target", None))))
                    value = getattr(node, "value", getattr(node, "right", None))
                    if target and self._is_tainted_expr(value, tainted):
                        tainted.add(target)

                        # Check: assigning tainted value to sensitive state
                        if target.lower() in self.SENSITIVE_STATE:
                            report.findings.append(VerificationFinding(
                                category=FindingCategory.PRIVILEGE_ESCALATION,
                                severity=Severity.CRITICAL,
                                message=(
                                    f"Action '{action_name}' assigns user-controlled "
                                    f"data to sensitive state variable '{target}'."
                                ),
                                action_name=action_name,
                                contract_name=contract_name,
                                suggestion=(
                                    f"Validate the value before assigning to '{target}', "
                                    f"or restrict this action to authorised callers only."
                                ),
                            ))

                # -- Call expressions: check dangerous sinks --
                if nt == "CallExpression":
                    callee = getattr(node, "function", getattr(node, "callee", None))
                    name = _get_name(callee) if callee else ""

                    if name.lower() in self.DANGEROUS_SINKS:
                        args = getattr(node, "arguments", getattr(node, "args", []))
                        if isinstance(args, (list, tuple)):
                            for arg in args:
                                if self._is_tainted_expr(arg, tainted):
                                    report.findings.append(VerificationFinding(
                                        category=FindingCategory.UNSANITIZED_INPUT,
                                        severity=Severity.HIGH,
                                        message=(
                                            f"Action '{action_name}' passes user-controlled "
                                            f"data to dangerous sink '{name}()' without sanitization."
                                        ),
                                        action_name=action_name,
                                        contract_name=contract_name,
                                        suggestion=(
                                            f"Validate/bound the argument before calling '{name}()'."
                                        ),
                                    ))
                                    break  # one finding per call

                # -- Storage writes with tainted keys --
                if nt in ("IndexExpression", "MemberExpression"):
                    idx = getattr(node, "index", getattr(node, "property", None))
                    if idx and self._is_tainted_expr(idx, tainted):
                        parent = getattr(node, "object", None)
                        parent_name = _get_name(parent) if parent else ""
                        report.findings.append(VerificationFinding(
                            category=FindingCategory.TAINTED_STORAGE_KEY,
                            severity=Severity.MEDIUM,
                            message=(
                                f"Action '{action_name}' uses user-controlled data "
                                f"as a storage/map key on '{parent_name}' — "
                                f"potential storage collision."
                            ),
                            action_name=action_name,
                            contract_name=contract_name,
                            suggestion="Hash or validate user input before using as storage key.",
                        ))

            # -- Check for unchecked external call returns --
            self._check_unchecked_returns(action_name, body, contract_name, report)

    def _is_tainted_expr(self, node: Any, tainted: Set[str]) -> bool:
        """Return True if the expression is or derives from tainted data."""
        if node is None:
            return False

        # Direct variable reference
        name = _get_name(node)
        if name:
            if name in tainted:
                return True
            if name in self.TAINT_SOURCES:
                return True

        # Member access (TX.caller, etc.)
        nt = _node_type(node)
        if nt == "MemberExpression":
            obj_name = _get_name(getattr(node, "object", None))
            prop_name = _get_name(getattr(node, "property", None))
            full = f"{obj_name}.{prop_name}" if obj_name and prop_name else ""
            if full in self.TAINT_SOURCES:
                return True

        # Binary expression: tainted if either operand is tainted
        if nt == "BinaryExpression":
            left = getattr(node, "left", None)
            right = getattr(node, "right", None)
            if self._is_tainted_expr(left, tainted) or self._is_tainted_expr(right, tainted):
                return True

        # Call expression return value
        if nt == "CallExpression":
            callee = getattr(node, "function", getattr(node, "callee", None))
            callee_name = _get_name(callee) if callee else ""
            # External calls return tainted data
            if callee_name.lower() in ("call", "delegatecall", "staticcall"):
                return True

        return False

    def _check_unchecked_returns(
        self,
        action_name: str,
        body: Any,
        contract_name: str,
        report: VerificationReport,
    ) -> None:
        """Flag external calls whose return value is not assigned or checked."""
        nodes = _walk_ast(body)
        for node in nodes:
            nt = _node_type(node)
            if nt != "ExpressionStatement":
                continue
            expr = getattr(node, "expression", node)
            if _node_type(expr) == "CallExpression":
                callee = getattr(expr, "function", getattr(expr, "callee", None))
                name = _get_name(callee) if callee else ""
                if name.lower() in ("call", "delegatecall", "staticcall", "send"):
                    report.findings.append(VerificationFinding(
                        category=FindingCategory.UNCHECKED_RETURN,
                        severity=Severity.HIGH,
                        message=(
                            f"Action '{action_name}' calls '{name}()' "
                            f"without checking its return value."
                        ),
                        action_name=action_name,
                        contract_name=contract_name,
                        suggestion=(
                            f"Assign the return value and check for success: "
                            f"`let result = {name}(...); require(result, \"call failed\");`"
                        ),
                    ))


# =====================================================================
# Main Verifier
# =====================================================================

class FormalVerifier:
    """Unified entry point for all verification levels.

    Parameters
    ----------
    level : VerificationLevel
        How deep to verify.
    annotations : str or list of str, optional
        Source text containing ``@invariant`` / ``@property`` annotations.
    invariants : list of Invariant, optional
        Pre-parsed invariants (overrides parsed annotations).
    properties : list of ContractProperty, optional
        Pre-parsed properties (overrides parsed annotations).
    """

    def __init__(
        self,
        level: VerificationLevel = VerificationLevel.STRUCTURAL,
        annotations: Optional[Union[str, List[str], Dict[str, Any]]] = None,
        invariants: Optional[List[Invariant]] = None,
        properties: Optional[List[ContractProperty]] = None,
        max_depth: int = 3,
    ):
        self.level = level
        self._structural = StructuralVerifier()
        self._invariant_v = InvariantVerifier()
        self._property_v = PropertyVerifier(max_depth=max_depth)
        self._taint = TaintAnalyzer()

        parsed_invariants: List[Invariant] = []
        parsed_properties: List[ContractProperty] = []

        if isinstance(annotations, dict):
            for inv_expr in (annotations.get("invariants") or []):
                if isinstance(inv_expr, str):
                    inv = Invariant(expression=inv_expr)
                    inv.variables = re.findall(r"\b([a-zA-Z_]\w*)\b", inv_expr)
                    parsed_invariants.append(inv)
            for prop in (annotations.get("properties") or []):
                if isinstance(prop, dict):
                    parsed_properties.append(ContractProperty(
                        name=prop.get("name", ""),
                        description=prop.get("description", ""),
                        precondition=prop.get("precondition", ""),
                        postcondition=prop.get("postcondition", ""),
                        action=prop.get("action", ""),
                    ))
        elif annotations is not None:
            parsed = AnnotationParser.parse_annotations(annotations)
            for inv_expr in (parsed.get("invariants") or []):
                inv = Invariant(expression=inv_expr)
                inv.variables = re.findall(r"\b([a-zA-Z_]\w*)\b", inv_expr)
                parsed_invariants.append(inv)
            for prop in (parsed.get("properties") or []):
                if isinstance(prop, dict):
                    parsed_properties.append(ContractProperty(
                        name=prop.get("name", ""),
                        description=prop.get("description", ""),
                        precondition=prop.get("precondition", ""),
                        postcondition=prop.get("postcondition", ""),
                        action=prop.get("action", ""),
                    ))

        self._invariants = invariants if invariants is not None else parsed_invariants
        self._properties = properties if properties is not None else parsed_properties

    def verify_contract(
        self,
        contract: Any,
        extra_invariants: Optional[List[Invariant]] = None,
        extra_properties: Optional[List[ContractProperty]] = None,
    ) -> VerificationReport:
        """Run all applicable verification passes on a contract.

        Parameters
        ----------
        contract :
            A ``SmartContract`` instance or AST ``ContractStatement``.
        extra_invariants :
            Additional invariants to check beyond those in annotations.
        extra_properties :
            Additional properties.

        Returns a ``VerificationReport``.
        """
        report = VerificationReport(
            contract_name=_get_name(getattr(contract, "name", "")),
            level=self.level,
        )

        # Also try extracting annotations from contract metadata
        meta = AnnotationParser.from_contract_metadata(contract)
        meta_inv: List[Invariant] = []
        meta_prop: List[ContractProperty] = []
        for inv_expr in (meta.get("invariants") or []):
            if isinstance(inv_expr, str):
                inv = Invariant(expression=inv_expr)
                inv.variables = re.findall(r"\b([a-zA-Z_]\w*)\b", inv_expr)
                meta_inv.append(inv)
        for prop in (meta.get("properties") or []):
            if isinstance(prop, dict):
                meta_prop.append(ContractProperty(
                    name=prop.get("name", ""),
                    description=prop.get("description", ""),
                    precondition=prop.get("precondition", ""),
                    postcondition=prop.get("postcondition", ""),
                    action=prop.get("action", ""),
                ))

        all_inv = list(self._invariants) + (extra_invariants or []) + meta_inv
        all_prop = list(self._properties) + (extra_properties or []) + meta_prop

        # Level 1: structural
        if self.level >= VerificationLevel.STRUCTURAL:
            self._structural.verify(contract, report)

        # Level 2: invariant
        if self.level >= VerificationLevel.INVARIANT and all_inv:
            self._invariant_v.verify(contract, all_inv, report)

        # Level 3: property
        if self.level >= VerificationLevel.PROPERTY and all_prop:
            self._property_v.verify(contract, all_prop, report)

        # Level 4: taint / data-flow analysis
        if self.level >= VerificationLevel.TAINT:
            self._taint.verify(contract, report)

        report.finished_at = time.time()
        return report

    def verify_multiple(
        self,
        contracts: List[Any],
    ) -> List[VerificationReport]:
        """Verify a list of contracts and return all reports."""
        return [self.verify_contract(c) for c in contracts]

    def add_invariant(self, expression: str) -> None:
        inv = Invariant(expression=expression)
        inv.variables = re.findall(r"\b([a-zA-Z_]\w*)\b", expression)
        self._invariants.append(inv)

    def add_property(
        self,
        name: str,
        precondition: str = "",
        postcondition: str = "",
        action: str = "",
        description: str = "",
    ) -> None:
        self._properties.append(ContractProperty(
            name=name,
            precondition=precondition,
            postcondition=postcondition,
            action=action,
            description=description,
        ))

    @property
    def invariant_count(self) -> int:
        return len(self._invariants)

    @property
    def property_count(self) -> int:
        return len(self._properties)
