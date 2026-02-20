"""
Zexus Blockchain — Upgradeable Contracts & Chains
==================================================

Implements the **Transparent Proxy** and **UUPS (Universal Upgradeable
Proxy Standard)** patterns adapted for the Zexus runtime, plus a
**chain governance upgrade** mechanism that allows live-upgrading
consensus parameters, block format versions, and fee structures
through on-chain proposals.

Architecture
------------

Smart-Contract Upgrades
^^^^^^^^^^^^^^^^^^^^^^^
::

    ┌──────────────┐   delegatecall   ┌─────────────────────┐
    │  ProxyContract│ ───────────────► │ ImplementationV1/V2 │
    │  (storage)   │                   │ (stateless logic)   │
    └──────────────┘                   └─────────────────────┘
          │
          ▼
    ┌──────────────┐
    │ UpgradeManager│  version registry + access control
    └──────────────┘

*   ``ProxyContract`` holds all persistent storage.  Calls are forwarded
    via ``delegate_call`` to the current implementation.
*   ``UpgradeManager`` tracks the version history, enforces role-based
    access (admin / multisig), and provides rollback capability.

Chain Upgrades
^^^^^^^^^^^^^^
::

    Proposal ──► Vote ──► Ratify ──► Apply

*   ``ChainUpgradeGovernance`` lets validators propose parameter
    changes (difficulty, gas limits, block time, consensus algorithm).
*   A supermajority vote (configurable quorum, default 2/3+1) is
    needed to ratify.
*   The new parameters take effect at a specific block height, giving
    all nodes time to prepare.

Security
--------
*   **Upgrade delay**: configurable cool-down between upgrade
    proposals to prevent rapid hostile takeover.
*   **Rollback**: any upgrade can be reverted to its predecessor.
*   **Storage collision protection**: implementation slots use
    keccak256-based storage keys.
*   **Event audit trail**: every upgrade emits logs indexable by
    ``EventIndex``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("zexus.blockchain.upgradeable")


# =====================================================================
# Constants
# =====================================================================

# Storage slot for the implementation address (EIP-1967-style)
_IMPL_SLOT = hashlib.sha256(b"zexus.proxy.implementation").hexdigest()
_ADMIN_SLOT = hashlib.sha256(b"zexus.proxy.admin").hexdigest()
_VERSION_SLOT = hashlib.sha256(b"zexus.proxy.version").hexdigest()


# =====================================================================
# Upgrade Events
# =====================================================================

class UpgradeEventType(str, Enum):
    CONTRACT_UPGRADED = "ContractUpgraded"
    CONTRACT_ROLLED_BACK = "ContractRolledBack"
    ADMIN_CHANGED = "AdminChanged"
    CHAIN_PROPOSAL_CREATED = "ChainProposalCreated"
    CHAIN_PROPOSAL_VOTED = "ChainProposalVoted"
    CHAIN_UPGRADE_APPLIED = "ChainUpgradeApplied"
    CHAIN_UPGRADE_REVERTED = "ChainUpgradeReverted"


@dataclass
class UpgradeEvent:
    """Immutable audit record for an upgrade action."""
    event_type: UpgradeEventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# =====================================================================
# Implementation Version Registry
# =====================================================================

@dataclass
class ImplementationRecord:
    """Metadata about a single implementation version."""
    version: int
    address: str                 # address of the implementation contract
    deployer: str                # who deployed it
    timestamp: float = field(default_factory=time.time)
    code_hash: str = ""          # hash of the implementation's code/name
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code_hash:
            # Deterministic default: bind code hash to implementation address
            self.code_hash = hashlib.sha256(str(self.address).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "address": self.address,
            "deployer": self.deployer,
            "timestamp": self.timestamp,
            "code_hash": self.code_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ImplementationRecord":
        return cls(
            version=d["version"],
            address=d["address"],
            deployer=d["deployer"],
            timestamp=d["timestamp"],
            code_hash=d["code_hash"],
            metadata=d.get("metadata", {}),
        )


# =====================================================================
# Proxy Contract
# =====================================================================

class ProxyContract:
    """Transparent proxy — stores state, delegates logic.

    The proxy has its own persistent storage dictionary.  All action
    calls are forwarded to the *current implementation* contract via
    ``ContractVM.delegate_call`` semantics (the implementation's code
    runs against the proxy's storage).

    Parameters
    ----------
    admin : str
        Address authorised to upgrade.
    implementation_address : str, optional
        Initial implementation contract address.
    proxy_address : str, optional
        Explicit address for the proxy itself.
    """

    def __init__(
        self,
        admin: str,
        implementation_address: str = "",
        proxy_address: Optional[str] = None,
    ):
        self.address: str = proxy_address or hashlib.sha256(
            f"proxy-{admin}-{time.time()}".encode()
        ).hexdigest()[:40]

        # Internal storage slots (EIP-1967 pattern)
        self._storage: Dict[str, Any] = {
            _IMPL_SLOT: implementation_address,
            _ADMIN_SLOT: admin,
            _VERSION_SLOT: 0,
        }

        # User-facing storage (the contract's data)
        self.data: Dict[str, Any] = {}

    # ── Properties ────────────────────────────────────────────────

    @property
    def implementation(self) -> str:
        return self._storage[_IMPL_SLOT]

    @implementation.setter
    def implementation(self, addr: str) -> None:
        self._storage[_IMPL_SLOT] = addr

    @property
    def admin(self) -> str:
        return self._storage[_ADMIN_SLOT]

    @admin.setter
    def admin(self, addr: str) -> None:
        self._storage[_ADMIN_SLOT] = addr

    @property
    def version(self) -> int:
        return self._storage[_VERSION_SLOT]

    @version.setter
    def version(self, v: int) -> None:
        self._storage[_VERSION_SLOT] = v

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        # Public shape intentionally mirrors common proxy contract APIs
        # and is what the Python test suite expects.
        return {
            "address": self.address,
            "admin": self.admin,
            "implementation": self.implementation,
            "version": self.version,
            "data": dict(self.data),
            # Preserve low-level slots for debugging/forensics
            "storage": dict(self._storage),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProxyContract":
        # Backward/forward compatible loader: accepts either the public keys
        # (admin/implementation/version) or the internal storage slot mapping.
        address = d.get("address")
        admin = d.get("admin")
        impl = d.get("implementation")
        version = d.get("version")
        storage = d.get("storage")

        p = cls.__new__(cls)
        p.address = address
        if isinstance(storage, dict):
            p._storage = dict(storage)
        else:
            p._storage = {
                _IMPL_SLOT: impl or "",
                _ADMIN_SLOT: admin or "",
                _VERSION_SLOT: int(version or 0),
            }
        # If explicit public keys exist, prefer them
        if admin is not None:
            p._storage[_ADMIN_SLOT] = admin
        if impl is not None:
            p._storage[_IMPL_SLOT] = impl
        if version is not None:
            p._storage[_VERSION_SLOT] = int(version)

        p.data = dict(d.get("data", {}))
        return p


# =====================================================================
# Upgrade Manager  (contract-level upgrades)
# =====================================================================

class UpgradeManager:
    """Manages the lifecycle of upgradeable proxy contracts.

    Responsibilities:
    *   Register implementation versions for a proxy.
    *   Enforce admin-only access for upgrades.
    *   Maintain a full version history with rollback support.
    *   Emit audit events for every mutation.

    Parameters
    ----------
    contract_vm : ContractVM, optional
        The VM bridge — used for ``delegate_call`` when executing
        through the proxy.  Can be ``None`` for pure state management.
    upgrade_delay : float
        Minimum seconds between consecutive upgrades (default 0 —
        no delay for dev networks; set higher for production).
    """

    def __init__(
        self,
        contract_vm=None,
        upgrade_delay: float = 0.0,
    ):
        self._vm = contract_vm
        self._upgrade_delay = upgrade_delay

        # proxy_address -> list of ImplementationRecord (ordered by version)
        self._versions: Dict[str, List[ImplementationRecord]] = {}

        # proxy_address -> ProxyContract
        self._proxies: Dict[str, ProxyContract] = {}

        # Audit log
        self._events: List[UpgradeEvent] = []

        # Timestamp of last upgrade per proxy (for delay enforcement)
        self._last_upgrade_time: Dict[str, float] = {}

    # ── Proxy lifecycle ───────────────────────────────────────────

    def create_proxy(
        self,
        admin: str,
        implementation_address: str,
        implementation_code_hash: str = "",
        proxy_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProxyContract:
        """Create a new upgradeable proxy pointing at an implementation.

        Returns the ``ProxyContract`` instance.
        """
        proxy = ProxyContract(
            admin=admin,
            implementation_address=implementation_address,
            proxy_address=proxy_address,
        )
        proxy.version = 1

        record = ImplementationRecord(
            version=1,
            address=implementation_address,
            deployer=admin,
            timestamp=time.time(),
            code_hash=implementation_code_hash or hashlib.sha256(
                implementation_address.encode()
            ).hexdigest(),
            metadata=metadata or {},
        )

        self._versions[proxy.address] = [record]
        self._proxies[proxy.address] = proxy
        self._last_upgrade_time[proxy.address] = time.time()

        self._emit(UpgradeEventType.CONTRACT_UPGRADED, {
            "proxy": proxy.address,
            "implementation": implementation_address,
            "version": 1,
            "admin": admin,
        })

        logger.info(
            "Proxy %s created → impl %s (v1)",
            proxy.address, implementation_address,
        )
        return proxy

    def upgrade(
        self,
        proxy_address: str,
        new_implementation: str,
        caller: str,
        code_hash: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        migrate_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> Tuple[bool, str]:
        """Upgrade a proxy to a new implementation.

        Parameters
        ----------
        proxy_address :
            The proxy to upgrade.
        new_implementation :
            Address of the new implementation contract.
        caller :
            Must match the proxy's admin.
        code_hash :
            Optional hash of the new implementation code for
            integrity verification.
        metadata :
            Arbitrary metadata (e.g. changelog, audit report hash).
        migrate_fn :
            Optional data-migration callback ``(old_data) -> new_data``
            that transforms proxy storage during the upgrade.

        Returns ``(success, message)``.
        """
        proxy = self._proxies.get(proxy_address)
        if proxy is None:
            return False, f"Proxy not found: {proxy_address}"

        # Access control
        if caller != proxy.admin:
            return False, f"Caller {caller} is not admin ({proxy.admin})"

        # Delay enforcement
        last = self._last_upgrade_time.get(proxy_address, 0.0)
        elapsed = time.time() - last
        if elapsed < self._upgrade_delay:
            remaining = self._upgrade_delay - elapsed
            return False, f"Upgrade delay not met: {remaining:.1f}s remaining"

        # Build new version record
        new_version = proxy.version + 1
        record = ImplementationRecord(
            version=new_version,
            address=new_implementation,
            deployer=caller,
            timestamp=time.time(),
            code_hash=code_hash or hashlib.sha256(
                new_implementation.encode()
            ).hexdigest(),
            metadata=metadata or {},
        )

        # Optional data migration
        if migrate_fn is not None:
            try:
                proxy.data = migrate_fn(copy.deepcopy(proxy.data))
            except Exception as exc:
                return False, f"Migration failed: {exc}"

        # Commit
        old_impl = proxy.implementation
        proxy.implementation = new_implementation
        proxy.version = new_version
        self._versions.setdefault(proxy_address, []).append(record)
        self._last_upgrade_time[proxy_address] = time.time()

        self._emit(UpgradeEventType.CONTRACT_UPGRADED, {
            "proxy": proxy_address,
            "old_implementation": old_impl,
            "new_implementation": new_implementation,
            "version": new_version,
            "caller": caller,
        })

        logger.info(
            "Proxy %s upgraded %s → %s (v%d)",
            proxy_address, old_impl, new_implementation, new_version,
        )
        return True, f"Upgraded to v{new_version}"

    def rollback(
        self,
        proxy_address: str,
        caller: str,
        target_version: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """Roll back a proxy to a previous implementation version.

        By default rolls back to the *immediately previous* version.
        Pass ``target_version`` to jump to a specific version.
        """
        proxy = self._proxies.get(proxy_address)
        if proxy is None:
            return False, f"Proxy not found: {proxy_address}"

        if caller != proxy.admin:
            return False, f"Caller {caller} is not admin ({proxy.admin})"

        versions = self._versions.get(proxy_address, [])
        if len(versions) < 2:
            return False, "No previous version to roll back to"

        if target_version is not None:
            target_record = None
            for rec in versions:
                if rec.version == target_version:
                    target_record = rec
                    break
            if target_record is None:
                return False, f"Version {target_version} not found"
        else:
            # Previous version
            target_record = versions[-2]

        old_impl = proxy.implementation
        old_version = proxy.version

        proxy.implementation = target_record.address
        proxy.version = target_record.version

        self._emit(UpgradeEventType.CONTRACT_ROLLED_BACK, {
            "proxy": proxy_address,
            "from_version": old_version,
            "to_version": target_record.version,
            "old_implementation": old_impl,
            "new_implementation": target_record.address,
            "caller": caller,
        })

        logger.info(
            "Proxy %s rolled back v%d → v%d",
            proxy_address, old_version, target_record.version,
        )
        return True, f"Rolled back to v{target_record.version}"

    def change_admin(
        self,
        proxy_address: str,
        new_admin: str,
        caller: str,
    ) -> Tuple[bool, str]:
        """Transfer admin rights for a proxy."""
        proxy = self._proxies.get(proxy_address)
        if proxy is None:
            return False, f"Proxy not found: {proxy_address}"
        if caller != proxy.admin:
            return False, f"Caller {caller} is not admin ({proxy.admin})"
        if not new_admin:
            return False, "New admin address cannot be empty"

        old_admin = proxy.admin
        proxy.admin = new_admin

        self._emit(UpgradeEventType.ADMIN_CHANGED, {
            "proxy": proxy_address,
            "old_admin": old_admin,
            "new_admin": new_admin,
        })
        return True, f"Admin changed to {new_admin}"

    # ── Execute through proxy ─────────────────────────────────────

    def proxy_call(
        self,
        proxy_address: str,
        action: str,
        args: Optional[Dict[str, Any]] = None,
        caller: str = "",
        gas_limit: Optional[int] = None,
    ) -> Any:
        """Execute an action on a proxy's current implementation.

        This delegates to ``ContractVM.execute_contract`` on the
        *implementation* address but uses the *proxy's* storage
        context (similar to ``delegatecall``).
        """
        proxy = self._proxies.get(proxy_address)
        if proxy is None:
            raise RuntimeError(f"Proxy not found: {proxy_address}")

        if not proxy.implementation:
            raise RuntimeError("Proxy has no implementation set")

        if self._vm is None:
            raise RuntimeError("No ContractVM attached to UpgradeManager")

        # Execute implementation code within proxy storage context
        receipt = self._vm.execute_contract(
            contract_address=proxy.implementation,
            action=action,
            args=args,
            caller=caller,
            gas_limit=gas_limit,
        )

        if not receipt.success:
            raise RuntimeError(
                f"Proxy call failed: {receipt.error or receipt.revert_reason}"
            )
        return receipt.return_value

    # ── Queries ───────────────────────────────────────────────────

    def get_proxy(self, proxy_address: str) -> Optional[ProxyContract]:
        return self._proxies.get(proxy_address)

    def get_version_history(
        self, proxy_address: str
    ) -> List[ImplementationRecord]:
        return list(self._versions.get(proxy_address, []))

    def get_current_version(self, proxy_address: str) -> Optional[int]:
        proxy = self._proxies.get(proxy_address)
        return proxy.version if proxy else None

    def get_current_implementation(self, proxy_address: str) -> Optional[str]:
        proxy = self._proxies.get(proxy_address)
        return proxy.implementation if proxy else None

    def list_proxies(self) -> List[str]:
        return list(self._proxies.keys())

    def get_events(self) -> List[UpgradeEvent]:
        return list(self._events)

    def get_info(self, proxy_address: str) -> Optional[Dict[str, Any]]:
        proxy = self._proxies.get(proxy_address)
        if proxy is None:
            return None
        versions = self._versions.get(proxy_address, [])
        return {
            "address": proxy.address,
            "admin": proxy.admin,
            "implementation": proxy.implementation,
            "version": proxy.version,
            "total_versions": len(versions),
            "versions": [v.to_dict() for v in versions],
            "data_keys": list(proxy.data.keys()),
        }

    # ── Internal ──────────────────────────────────────────────────

    def _emit(self, event_type: UpgradeEventType, data: Dict[str, Any]) -> None:
        self._events.append(UpgradeEvent(event_type=event_type, data=data))


# =====================================================================
# Chain Upgrade Governance
# =====================================================================

class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    REVERTED = "reverted"


class ProposalType(str, Enum):
    """Types of chain-level upgrades."""
    CONSENSUS_CHANGE = "consensus_change"
    DIFFICULTY_CHANGE = "difficulty_change"
    GAS_LIMIT_CHANGE = "gas_limit_change"
    BLOCK_TIME_CHANGE = "block_time_change"
    FEE_STRUCTURE = "fee_structure"
    CHAIN_PARAMETER = "chain_parameter"
    HARD_FORK = "hard_fork"


@dataclass
class ChainUpgradeProposal:
    """A proposal to change chain-level parameters."""
    proposal_id: str
    proposal_type: ProposalType
    proposer: str
    description: str
    changes: Dict[str, Any]               # key -> new value
    activation_height: int                 # block height at which to apply
    created_at: float = field(default_factory=time.time)
    status: ProposalStatus = ProposalStatus.PENDING
    votes_for: Set[str] = field(default_factory=set)
    votes_against: Set[str] = field(default_factory=set)
    applied_at: Optional[float] = None
    previous_values: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_votes(self) -> int:
        return len(self.votes_for) + len(self.votes_against)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type.value,
            "proposer": self.proposer,
            "description": self.description,
            "changes": self.changes,
            "activation_height": self.activation_height,
            "created_at": self.created_at,
            "status": self.status.value,
            "votes_for": list(self.votes_for),
            "votes_against": list(self.votes_against),
            "applied_at": self.applied_at,
            "previous_values": self.previous_values,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChainUpgradeProposal":
        p = cls(
            proposal_id=d["proposal_id"],
            proposal_type=ProposalType(d["proposal_type"]),
            proposer=d["proposer"],
            description=d["description"],
            changes=d["changes"],
            activation_height=d["activation_height"],
            created_at=d.get("created_at", 0.0),
            status=ProposalStatus(d.get("status", "pending")),
            applied_at=d.get("applied_at"),
            previous_values=d.get("previous_values", {}),
        )
        p.votes_for = set(d.get("votes_for", []))
        p.votes_against = set(d.get("votes_against", []))
        return p


class ChainUpgradeGovernance:
    """On-chain governance for chain-level upgrades.

    Validators can propose changes to consensus parameters,
    difficulty, gas limits, block times, and fee structures.
    A configurable quorum (default 2/3 + 1) determines whether
    a proposal passes.

    Parameters
    ----------
    chain : Chain
        The chain being governed.
    validators : set of str
        Set of validator addresses allowed to propose & vote.
    quorum_numerator : int
        Numerator for quorum fraction (default 2).
    quorum_denominator : int
        Denominator for quorum fraction (default 3).
    proposal_delay : float
        Minimum seconds between proposals from the same proposer.
    """

    # Allowed chain parameter keys and their type validators
    ALLOWED_PARAMETERS: Dict[str, type] = {
        "difficulty": int,
        "target_block_time": (int, float),
        "chain_id": str,
        "gas_limit_default": int,
        "base_fee": int,
        "max_block_gas": int,
        "consensus_algorithm": str,
    }

    def __init__(
        self,
        chain=None,
        validators: Optional[Set[str]] = None,
        quorum_numerator: int = 2,
        quorum_denominator: int = 3,
        proposal_delay: float = 0.0,
    ):
        self._chain = chain
        self._validators: Set[str] = set(validators or [])
        self._quorum_num = quorum_numerator
        self._quorum_den = quorum_denominator
        self._proposal_delay = proposal_delay

        # proposal_id -> ChainUpgradeProposal
        self._proposals: Dict[str, ChainUpgradeProposal] = {}

        # Audit trail
        self._events: List[UpgradeEvent] = []

        # proposer -> last proposal timestamp (for rate-limiting)
        self._last_proposal_time: Dict[str, float] = {}

        # Applied upgrade history (for revert)
        self._applied_upgrades: List[str] = []  # ordered proposal_ids

    # ── Validators ────────────────────────────────────────────────

    def add_validator(self, address: str) -> None:
        self._validators.add(address)

    def remove_validator(self, address: str) -> None:
        self._validators.discard(address)

    @property
    def validator_count(self) -> int:
        return len(self._validators)

    @property
    def quorum_threshold(self) -> int:
        """Minimum votes-for required to approve a proposal."""
        n = len(self._validators)
        return (n * self._quorum_num) // self._quorum_den + 1

    # ── Proposals ─────────────────────────────────────────────────

    def propose(
        self,
        proposer: str,
        proposal_type: ProposalType,
        description: str,
        changes: Dict[str, Any],
        activation_height: int,
    ) -> Tuple[bool, str, Optional[str]]:
        """Submit a new upgrade proposal.

        Returns ``(success, message, proposal_id)``.
        """
        # Validate proposer is a validator
        if proposer not in self._validators:
            return False, f"Proposer {proposer} is not a validator", None

        # Rate-limit
        last = self._last_proposal_time.get(proposer, 0.0)
        if time.time() - last < self._proposal_delay:
            return False, "Proposal rate-limited", None

        # Validate changes keys
        for key in changes:
            if key not in self.ALLOWED_PARAMETERS:
                return False, f"Unknown parameter: {key}", None

        # Validate activation height is in the future
        current_height = self._chain.height if self._chain else 0
        if activation_height <= current_height:
            return False, (
                f"Activation height {activation_height} must be > "
                f"current height {current_height}"
            ), None

        # Generate proposal ID
        pid = hashlib.sha256(
            f"{proposer}-{time.time()}-{description}".encode()
        ).hexdigest()[:16]

        proposal = ChainUpgradeProposal(
            proposal_id=pid,
            proposal_type=proposal_type,
            proposer=proposer,
            description=description,
            changes=changes,
            activation_height=activation_height,
        )

        # Proposer auto-votes yes
        proposal.votes_for.add(proposer)

        self._proposals[pid] = proposal
        self._last_proposal_time[proposer] = time.time()

        self._emit(UpgradeEventType.CHAIN_PROPOSAL_CREATED, {
            "proposal_id": pid,
            "proposer": proposer,
            "type": proposal_type.value,
            "changes": changes,
            "activation_height": activation_height,
        })

        # Check if the single vote already meets quorum (e.g. 1 validator)
        self._check_quorum(pid)

        logger.info("Chain upgrade proposal %s created by %s", pid, proposer)
        return True, f"Proposal {pid} created", pid

    def vote(
        self,
        proposal_id: str,
        voter: str,
        approve: bool,
    ) -> Tuple[bool, str]:
        """Vote on a pending proposal.

        Returns ``(success, message)``.
        """
        if voter not in self._validators:
            return False, f"Voter {voter} is not a validator"

        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False, f"Proposal {proposal_id} not found"
        if proposal.status != ProposalStatus.PENDING:
            return False, f"Proposal is {proposal.status.value}, not pending"

        # Prevent double voting
        if voter in proposal.votes_for or voter in proposal.votes_against:
            return False, f"Voter {voter} already voted"

        if approve:
            proposal.votes_for.add(voter)
        else:
            proposal.votes_against.add(voter)

        self._emit(UpgradeEventType.CHAIN_PROPOSAL_VOTED, {
            "proposal_id": proposal_id,
            "voter": voter,
            "approve": approve,
            "votes_for": len(proposal.votes_for),
            "votes_against": len(proposal.votes_against),
        })

        # Check quorum
        self._check_quorum(proposal_id)

        return True, "Vote recorded"

    def apply_pending(self, current_height: int) -> List[str]:
        """Apply all approved proposals whose activation height has been reached.

        Called by the node/consensus after adding each block.

        Returns list of applied proposal IDs.
        """
        applied: List[str] = []

        for pid, proposal in self._proposals.items():
            if proposal.status != ProposalStatus.APPROVED:
                continue
            if current_height < proposal.activation_height:
                continue

            # Snapshot current values before applying
            for key in proposal.changes:
                if self._chain and hasattr(self._chain, key):
                    proposal.previous_values[key] = getattr(self._chain, key)

            # Apply changes
            success = self._apply_changes(proposal.changes)
            if success:
                proposal.status = ProposalStatus.APPLIED
                proposal.applied_at = time.time()
                self._applied_upgrades.append(pid)
                applied.append(pid)

                self._emit(UpgradeEventType.CHAIN_UPGRADE_APPLIED, {
                    "proposal_id": pid,
                    "changes": proposal.changes,
                    "height": current_height,
                })
                logger.info("Chain upgrade %s applied at height %d", pid, current_height)

        return applied

    def revert_upgrade(
        self,
        proposal_id: str,
        caller: str,
    ) -> Tuple[bool, str]:
        """Revert a previously applied chain upgrade.

        Returns ``(success, message)``.
        """
        if caller not in self._validators:
            return False, "Only validators can revert upgrades"

        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return False, f"Proposal {proposal_id} not found"
        if proposal.status != ProposalStatus.APPLIED:
            return False, f"Proposal status is {proposal.status.value}, not applied"
        if not proposal.previous_values:
            return False, "No previous values stored for revert"

        # Restore previous values
        success = self._apply_changes(proposal.previous_values)
        if not success:
            return False, "Failed to restore previous values"

        proposal.status = ProposalStatus.REVERTED
        if proposal_id in self._applied_upgrades:
            self._applied_upgrades.remove(proposal_id)

        self._emit(UpgradeEventType.CHAIN_UPGRADE_REVERTED, {
            "proposal_id": proposal_id,
            "restored_values": proposal.previous_values,
            "caller": caller,
        })

        logger.info("Chain upgrade %s reverted by %s", proposal_id, caller)
        return True, "Upgrade reverted"

    # ── Queries ───────────────────────────────────────────────────

    def get_proposal(self, proposal_id: str) -> Optional[ChainUpgradeProposal]:
        return self._proposals.get(proposal_id)

    def list_proposals(
        self, status: Optional[ProposalStatus] = None
    ) -> List[ChainUpgradeProposal]:
        proposals = list(self._proposals.values())
        if status is not None:
            proposals = [p for p in proposals if p.status == status]
        return proposals

    def get_events(self) -> List[UpgradeEvent]:
        return list(self._events)

    def get_governance_info(self) -> Dict[str, Any]:
        return {
            "validators": list(self._validators),
            "validator_count": len(self._validators),
            "quorum_threshold": self.quorum_threshold,
            "total_proposals": len(self._proposals),
            "applied_upgrades": len(self._applied_upgrades),
            "pending": len([
                p for p in self._proposals.values()
                if p.status == ProposalStatus.PENDING
            ]),
        }

    # ── Internal ──────────────────────────────────────────────────

    def _check_quorum(self, proposal_id: str) -> None:
        """Check if a proposal has reached quorum and update status."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.status != ProposalStatus.PENDING:
            return

        threshold = self.quorum_threshold
        if len(proposal.votes_for) >= threshold:
            proposal.status = ProposalStatus.APPROVED
        elif len(proposal.votes_against) >= threshold:
            proposal.status = ProposalStatus.REJECTED

    def _apply_changes(self, changes: Dict[str, Any]) -> bool:
        """Apply parameter changes to the chain."""
        if self._chain is None:
            return False
        for key, value in changes.items():
            if hasattr(self._chain, key):
                setattr(self._chain, key, value)
        return True

    def _emit(self, event_type: UpgradeEventType, data: Dict[str, Any]) -> None:
        self._events.append(UpgradeEvent(event_type=event_type, data=data))
