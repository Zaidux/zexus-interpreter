"""
Token Standard Interfaces for the Zexus Blockchain.

Defines three token standards analogous to Ethereum's ERC ecosystem:

  - **ZX20** — Fungible token (like ERC-20)
  - **ZX721** — Non-fungible token / NFT (like ERC-721)
  - **ZX1155** — Multi-token (like ERC-1155, both fungible and NFT)

Each standard is an abstract base class with a complete reference
implementation, ready to be deployed in the ContractVM.  The standards
define the canonical event names, required methods, and metadata
structures.

Usage::

    token = ZX20Token("MyToken", "MTK", 18, initial_supply=1_000_000)
    token.transfer(sender, recipient, 500)

    nft = ZX721Token("MyNFT", "MNFT")
    nft.mint(owner, token_id=1, token_uri="ipfs://...")

    multi = ZX1155Token("ipfs://metadata/{id}.json")
    multi.mint(owner, token_id=1, amount=100, data=b"")
"""

from __future__ import annotations

import abc
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  Events — canonical event structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TokenEvent:
    """Base token event."""
    event_name: str
    contract_address: str = ""
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event_name,
            "contract": self.contract_address,
            "timestamp": self.timestamp,
            **self.data,
        }


class TransferEvent(TokenEvent):
    """ZX20/ZX721 Transfer event."""
    def __init__(self, from_addr: str, to_addr: str, value: int,
                 contract_address: str = ""):
        super().__init__(
            event_name="Transfer",
            contract_address=contract_address,
            data={"from": from_addr, "to": to_addr, "value": value},
        )


class ApprovalEvent(TokenEvent):
    """ZX20/ZX721 Approval event."""
    def __init__(self, owner: str, spender: str, value: int,
                 contract_address: str = ""):
        super().__init__(
            event_name="Approval",
            contract_address=contract_address,
            data={"owner": owner, "spender": spender, "value": value},
        )


class TransferSingleEvent(TokenEvent):
    """ZX1155 TransferSingle event."""
    def __init__(self, operator: str, from_addr: str, to_addr: str,
                 token_id: int, amount: int, contract_address: str = ""):
        super().__init__(
            event_name="TransferSingle",
            contract_address=contract_address,
            data={
                "operator": operator, "from": from_addr, "to": to_addr,
                "id": token_id, "value": amount,
            },
        )


class TransferBatchEvent(TokenEvent):
    """ZX1155 TransferBatch event."""
    def __init__(self, operator: str, from_addr: str, to_addr: str,
                 ids: List[int], amounts: List[int],
                 contract_address: str = ""):
        super().__init__(
            event_name="TransferBatch",
            contract_address=contract_address,
            data={
                "operator": operator, "from": from_addr, "to": to_addr,
                "ids": ids, "values": amounts,
            },
        )


class ApprovalForAllEvent(TokenEvent):
    """ZX721/ZX1155 ApprovalForAll event."""
    def __init__(self, owner: str, operator: str, approved: bool,
                 contract_address: str = ""):
        super().__init__(
            event_name="ApprovalForAll",
            contract_address=contract_address,
            data={"owner": owner, "operator": operator, "approved": approved},
        )


# ══════════════════════════════════════════════════════════════════════
#  ZX-20 — Fungible Token Standard
# ══════════════════════════════════════════════════════════════════════

ZERO_ADDRESS = "0x" + "0" * 40


class ZX20Interface(abc.ABC):
    """Abstract interface for ZX-20 fungible tokens."""

    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def symbol(self) -> str: ...

    @abc.abstractmethod
    def decimals(self) -> int: ...

    @abc.abstractmethod
    def total_supply(self) -> int: ...

    @abc.abstractmethod
    def balance_of(self, account: str) -> int: ...

    @abc.abstractmethod
    def transfer(self, sender: str, recipient: str, amount: int) -> bool: ...

    @abc.abstractmethod
    def allowance(self, owner: str, spender: str) -> int: ...

    @abc.abstractmethod
    def approve(self, owner: str, spender: str, amount: int) -> bool: ...

    @abc.abstractmethod
    def transfer_from(self, spender: str, from_addr: str,
                      to_addr: str, amount: int) -> bool: ...


class ZX20Token(ZX20Interface):
    """Reference implementation of the ZX-20 fungible token standard.

    All amounts are in the smallest unit (like wei for ETH).
    """

    def __init__(self, token_name: str, token_symbol: str,
                 token_decimals: int = 18,
                 initial_supply: int = 0,
                 owner: str = ""):
        self._name = token_name
        self._symbol = token_symbol
        self._decimals = token_decimals
        self._total_supply: int = 0
        self._balances: Dict[str, int] = {}
        self._allowances: Dict[str, Dict[str, int]] = {}  # owner -> spender -> amount
        self._owner: str = owner

        # Event log
        self.events: List[TokenEvent] = []

        # Contract address (set when deployed)
        self.contract_address: str = ""

        # Mint initial supply to owner
        if initial_supply > 0 and owner:
            self._mint(owner, initial_supply)

    def name(self) -> str:
        return self._name

    def symbol(self) -> str:
        return self._symbol

    def decimals(self) -> int:
        return self._decimals

    def total_supply(self) -> int:
        return self._total_supply

    def balance_of(self, account: str) -> int:
        return self._balances.get(account, 0)

    def transfer(self, sender: str, recipient: str, amount: int) -> bool:
        if amount < 0:
            raise ValueError("Transfer amount must be non-negative")
        if not recipient or recipient == ZERO_ADDRESS:
            raise ValueError("Transfer to zero address")
        if self._balances.get(sender, 0) < amount:
            raise ValueError("Insufficient balance")

        self._balances[sender] = self._balances.get(sender, 0) - amount
        self._balances[recipient] = self._balances.get(recipient, 0) + amount

        self._emit(TransferEvent(sender, recipient, amount, self.contract_address))
        return True

    def allowance(self, owner: str, spender: str) -> int:
        return self._allowances.get(owner, {}).get(spender, 0)

    def approve(self, owner: str, spender: str, amount: int) -> bool:
        if amount < 0:
            raise ValueError("Approval amount must be non-negative")
        if owner not in self._allowances:
            self._allowances[owner] = {}
        self._allowances[owner][spender] = amount
        self._emit(ApprovalEvent(owner, spender, amount, self.contract_address))
        return True

    def transfer_from(self, spender: str, from_addr: str,
                      to_addr: str, amount: int) -> bool:
        allowed = self.allowance(from_addr, spender)
        if allowed < amount:
            raise ValueError("Allowance exceeded")

        self.transfer(from_addr, to_addr, amount)

        # Decrease allowance
        self._allowances[from_addr][spender] = allowed - amount
        return True

    # ── Extensions ────────────────────────────────────────────────

    def _mint(self, to: str, amount: int) -> None:
        """Mint new tokens to an address."""
        if amount < 0:
            raise ValueError("Mint amount must be non-negative")
        self._total_supply += amount
        self._balances[to] = self._balances.get(to, 0) + amount
        self._emit(TransferEvent(ZERO_ADDRESS, to, amount, self.contract_address))

    def mint(self, caller: str, to: str, amount: int) -> None:
        """Public mint (owner-only)."""
        if self._owner and caller != self._owner:
            raise PermissionError("Only owner can mint")
        self._mint(to, amount)

    def burn(self, owner: str, amount: int) -> None:
        """Burn tokens from an address."""
        if self._balances.get(owner, 0) < amount:
            raise ValueError("Burn amount exceeds balance")
        self._balances[owner] -= amount
        self._total_supply -= amount
        self._emit(TransferEvent(owner, ZERO_ADDRESS, amount, self.contract_address))

    def _emit(self, event: TokenEvent) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize token state."""
        return {
            "standard": "ZX-20",
            "name": self._name,
            "symbol": self._symbol,
            "decimals": self._decimals,
            "total_supply": self._total_supply,
            "owner": self._owner,
            "balances": dict(self._balances),
            "allowances": {
                owner: dict(spenders)
                for owner, spenders in self._allowances.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZX20Token":
        token = cls(
            token_name=data["name"],
            token_symbol=data["symbol"],
            token_decimals=data.get("decimals", 18),
            owner=data.get("owner", ""),
        )
        token._total_supply = data.get("total_supply", 0)
        token._balances = data.get("balances", {})
        token._allowances = {
            owner: dict(spenders)
            for owner, spenders in data.get("allowances", {}).items()
        }
        return token


# ══════════════════════════════════════════════════════════════════════
#  ZX-721 — Non-Fungible Token (NFT) Standard
# ══════════════════════════════════════════════════════════════════════

class ZX721Interface(abc.ABC):
    """Abstract interface for ZX-721 non-fungible tokens."""

    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def symbol(self) -> str: ...

    @abc.abstractmethod
    def balance_of(self, owner: str) -> int: ...

    @abc.abstractmethod
    def owner_of(self, token_id: int) -> str: ...

    @abc.abstractmethod
    def transfer_from(self, caller: str, from_addr: str,
                      to_addr: str, token_id: int) -> None: ...

    @abc.abstractmethod
    def approve(self, caller: str, to: str, token_id: int) -> None: ...

    @abc.abstractmethod
    def get_approved(self, token_id: int) -> str: ...

    @abc.abstractmethod
    def set_approval_for_all(self, owner: str, operator: str,
                             approved: bool) -> None: ...

    @abc.abstractmethod
    def is_approved_for_all(self, owner: str, operator: str) -> bool: ...


class ZX721Token(ZX721Interface):
    """Reference implementation of the ZX-721 NFT standard."""

    def __init__(self, token_name: str, token_symbol: str,
                 owner: str = ""):
        self._name = token_name
        self._symbol = token_symbol
        self._owner = owner

        # token_id -> owner address
        self._owners: Dict[int, str] = {}
        # owner -> token count
        self._balances: Dict[str, int] = {}
        # token_id -> approved address
        self._token_approvals: Dict[int, str] = {}
        # owner -> operator -> bool
        self._operator_approvals: Dict[str, Dict[str, bool]] = {}
        # token_id -> URI string
        self._token_uris: Dict[int, str] = {}

        self._total_supply: int = 0

        self.events: List[TokenEvent] = []
        self.contract_address: str = ""

    def name(self) -> str:
        return self._name

    def symbol(self) -> str:
        return self._symbol

    def total_supply(self) -> int:
        return self._total_supply

    def balance_of(self, owner: str) -> int:
        if not owner or owner == ZERO_ADDRESS:
            raise ValueError("Balance query for zero address")
        return self._balances.get(owner, 0)

    def owner_of(self, token_id: int) -> str:
        owner = self._owners.get(token_id, "")
        if not owner:
            raise ValueError(f"Token {token_id} does not exist")
        return owner

    def token_uri(self, token_id: int) -> str:
        if token_id not in self._owners:
            raise ValueError(f"Token {token_id} does not exist")
        return self._token_uris.get(token_id, "")

    def approve(self, caller: str, to: str, token_id: int) -> None:
        owner = self.owner_of(token_id)
        if caller != owner and not self.is_approved_for_all(owner, caller):
            raise PermissionError("Not owner or approved operator")
        self._token_approvals[token_id] = to
        self._emit(ApprovalEvent(owner, to, token_id, self.contract_address))

    def get_approved(self, token_id: int) -> str:
        if token_id not in self._owners:
            raise ValueError(f"Token {token_id} does not exist")
        return self._token_approvals.get(token_id, "")

    def set_approval_for_all(self, owner: str, operator: str,
                             approved: bool) -> None:
        if owner == operator:
            raise ValueError("Cannot approve self")
        if owner not in self._operator_approvals:
            self._operator_approvals[owner] = {}
        self._operator_approvals[owner][operator] = approved
        self._emit(ApprovalForAllEvent(owner, operator, approved, self.contract_address))

    def is_approved_for_all(self, owner: str, operator: str) -> bool:
        return self._operator_approvals.get(owner, {}).get(operator, False)

    def _is_approved_or_owner(self, spender: str, token_id: int) -> bool:
        owner = self.owner_of(token_id)
        return (
            spender == owner
            or self.get_approved(token_id) == spender
            or self.is_approved_for_all(owner, spender)
        )

    def transfer_from(self, caller: str, from_addr: str,
                      to_addr: str, token_id: int) -> None:
        if not self._is_approved_or_owner(caller, token_id):
            raise PermissionError("Not approved or owner")
        owner = self.owner_of(token_id)
        if owner != from_addr:
            raise ValueError("Transfer from incorrect owner")
        if not to_addr or to_addr == ZERO_ADDRESS:
            raise ValueError("Transfer to zero address")

        # Clear approval
        self._token_approvals.pop(token_id, None)

        self._balances[from_addr] = self._balances.get(from_addr, 0) - 1
        self._balances[to_addr] = self._balances.get(to_addr, 0) + 1
        self._owners[token_id] = to_addr

        self._emit(TransferEvent(from_addr, to_addr, token_id, self.contract_address))

    def safe_transfer_from(self, caller: str, from_addr: str,
                           to_addr: str, token_id: int) -> None:
        """Transfer with safety check (same as transfer_from here)."""
        self.transfer_from(caller, from_addr, to_addr, token_id)

    # ── Minting / Burning ─────────────────────────────────────────

    def mint(self, to: str, token_id: int, token_uri: str = "") -> None:
        """Mint a new NFT."""
        if token_id in self._owners:
            raise ValueError(f"Token {token_id} already exists")
        if not to or to == ZERO_ADDRESS:
            raise ValueError("Mint to zero address")

        self._owners[token_id] = to
        self._balances[to] = self._balances.get(to, 0) + 1
        self._total_supply += 1

        if token_uri:
            self._token_uris[token_id] = token_uri

        self._emit(TransferEvent(ZERO_ADDRESS, to, token_id, self.contract_address))

    def burn(self, caller: str, token_id: int) -> None:
        """Burn an NFT."""
        if not self._is_approved_or_owner(caller, token_id):
            raise PermissionError("Not approved or owner")

        owner = self.owner_of(token_id)
        self._token_approvals.pop(token_id, None)
        self._balances[owner] = self._balances.get(owner, 0) - 1
        del self._owners[token_id]
        self._token_uris.pop(token_id, None)
        self._total_supply -= 1

        self._emit(TransferEvent(owner, ZERO_ADDRESS, token_id, self.contract_address))

    def tokens_of_owner(self, owner: str) -> List[int]:
        """Return all token IDs owned by an address."""
        return [tid for tid, o in self._owners.items() if o == owner]

    def _emit(self, event: TokenEvent) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "standard": "ZX-721",
            "name": self._name,
            "symbol": self._symbol,
            "owner": self._owner,
            "total_supply": self._total_supply,
            "owners": {str(k): v for k, v in self._owners.items()},
            "token_uris": {str(k): v for k, v in self._token_uris.items()},
            "balances": dict(self._balances),
            "operator_approvals": {
                o: dict(ops) for o, ops in self._operator_approvals.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZX721Token":
        token = cls(
            token_name=data["name"],
            token_symbol=data["symbol"],
            owner=data.get("owner", ""),
        )
        token._total_supply = data.get("total_supply", 0)
        token._owners = {int(k): v for k, v in data.get("owners", {}).items()}
        token._token_uris = {int(k): v for k, v in data.get("token_uris", {}).items()}
        token._balances = data.get("balances", {})
        token._operator_approvals = {
            o: dict(ops) for o, ops in data.get("operator_approvals", {}).items()
        }
        return token


# ══════════════════════════════════════════════════════════════════════
#  ZX-1155 — Multi-Token Standard
# ══════════════════════════════════════════════════════════════════════

class ZX1155Interface(abc.ABC):
    """Abstract interface for ZX-1155 multi-tokens."""

    @abc.abstractmethod
    def balance_of(self, account: str, token_id: int) -> int: ...

    @abc.abstractmethod
    def balance_of_batch(self, accounts: List[str],
                         ids: List[int]) -> List[int]: ...

    @abc.abstractmethod
    def set_approval_for_all(self, owner: str, operator: str,
                             approved: bool) -> None: ...

    @abc.abstractmethod
    def is_approved_for_all(self, owner: str, operator: str) -> bool: ...

    @abc.abstractmethod
    def safe_transfer_from(self, operator: str, from_addr: str,
                           to_addr: str, token_id: int,
                           amount: int, data: bytes) -> None: ...

    @abc.abstractmethod
    def safe_batch_transfer_from(self, operator: str, from_addr: str,
                                 to_addr: str, ids: List[int],
                                 amounts: List[int],
                                 data: bytes) -> None: ...


class ZX1155Token(ZX1155Interface):
    """Reference implementation of the ZX-1155 multi-token standard.

    Supports both fungible (supply > 1) and non-fungible (supply = 1)
    tokens within a single contract.
    """

    def __init__(self, base_uri: str = "", owner: str = ""):
        self._base_uri = base_uri
        self._owner = owner

        # (token_id, account) -> balance
        self._balances: Dict[int, Dict[str, int]] = {}
        # owner -> operator -> bool
        self._operator_approvals: Dict[str, Dict[str, bool]] = {}
        # token_id -> total supply
        self._supplies: Dict[int, int] = {}
        # token_id -> custom URI (overrides base_uri)
        self._token_uris: Dict[int, str] = {}

        self.events: List[TokenEvent] = []
        self.contract_address: str = ""

    def uri(self, token_id: int) -> str:
        """Return the URI for a token (replaces {id} in base_uri)."""
        custom = self._token_uris.get(token_id)
        if custom:
            return custom
        return self._base_uri.replace("{id}", str(token_id))

    def set_uri(self, token_id: int, token_uri: str) -> None:
        self._token_uris[token_id] = token_uri

    def total_supply(self, token_id: int) -> int:
        return self._supplies.get(token_id, 0)

    def exists(self, token_id: int) -> bool:
        return self._supplies.get(token_id, 0) > 0

    def balance_of(self, account: str, token_id: int) -> int:
        return self._balances.get(token_id, {}).get(account, 0)

    def balance_of_batch(self, accounts: List[str],
                         ids: List[int]) -> List[int]:
        if len(accounts) != len(ids):
            raise ValueError("Accounts and IDs length mismatch")
        return [self.balance_of(a, i) for a, i in zip(accounts, ids)]

    def set_approval_for_all(self, owner: str, operator: str,
                             approved: bool) -> None:
        if owner == operator:
            raise ValueError("Cannot approve self")
        if owner not in self._operator_approvals:
            self._operator_approvals[owner] = {}
        self._operator_approvals[owner][operator] = approved
        self._emit(ApprovalForAllEvent(owner, operator, approved, self.contract_address))

    def is_approved_for_all(self, owner: str, operator: str) -> bool:
        return self._operator_approvals.get(owner, {}).get(operator, False)

    def safe_transfer_from(self, operator: str, from_addr: str,
                           to_addr: str, token_id: int,
                           amount: int, data: bytes = b"") -> None:
        if operator != from_addr and not self.is_approved_for_all(from_addr, operator):
            raise PermissionError("Not owner or approved")
        if not to_addr or to_addr == ZERO_ADDRESS:
            raise ValueError("Transfer to zero address")

        from_bal = self.balance_of(from_addr, token_id)
        if from_bal < amount:
            raise ValueError("Insufficient balance")

        self._balances.setdefault(token_id, {})[from_addr] = from_bal - amount
        to_bal = self.balance_of(to_addr, token_id)
        self._balances.setdefault(token_id, {})[to_addr] = to_bal + amount

        self._emit(TransferSingleEvent(
            operator, from_addr, to_addr, token_id, amount, self.contract_address
        ))

    def safe_batch_transfer_from(self, operator: str, from_addr: str,
                                 to_addr: str, ids: List[int],
                                 amounts: List[int],
                                 data: bytes = b"") -> None:
        if len(ids) != len(amounts):
            raise ValueError("IDs and amounts length mismatch")
        if operator != from_addr and not self.is_approved_for_all(from_addr, operator):
            raise PermissionError("Not owner or approved")
        if not to_addr or to_addr == ZERO_ADDRESS:
            raise ValueError("Transfer to zero address")

        for token_id, amount in zip(ids, amounts):
            from_bal = self.balance_of(from_addr, token_id)
            if from_bal < amount:
                raise ValueError(f"Insufficient balance for token {token_id}")
            self._balances.setdefault(token_id, {})[from_addr] = from_bal - amount
            to_bal = self.balance_of(to_addr, token_id)
            self._balances.setdefault(token_id, {})[to_addr] = to_bal + amount

        self._emit(TransferBatchEvent(
            operator, from_addr, to_addr, ids, amounts, self.contract_address
        ))

    # ── Minting / Burning ─────────────────────────────────────────

    def mint(self, to: str, token_id: int, amount: int,
             data: bytes = b"") -> None:
        """Mint tokens."""
        if not to or to == ZERO_ADDRESS:
            raise ValueError("Mint to zero address")

        bal = self.balance_of(to, token_id)
        self._balances.setdefault(token_id, {})[to] = bal + amount
        self._supplies[token_id] = self._supplies.get(token_id, 0) + amount

        self._emit(TransferSingleEvent(
            to, ZERO_ADDRESS, to, token_id, amount, self.contract_address
        ))

    def mint_batch(self, to: str, ids: List[int], amounts: List[int],
                   data: bytes = b"") -> None:
        """Batch mint."""
        if len(ids) != len(amounts):
            raise ValueError("IDs and amounts length mismatch")
        if not to or to == ZERO_ADDRESS:
            raise ValueError("Mint to zero address")

        for token_id, amount in zip(ids, amounts):
            bal = self.balance_of(to, token_id)
            self._balances.setdefault(token_id, {})[to] = bal + amount
            self._supplies[token_id] = self._supplies.get(token_id, 0) + amount

        self._emit(TransferBatchEvent(
            to, ZERO_ADDRESS, to, ids, amounts, self.contract_address
        ))

    def burn(self, owner: str, token_id: int, amount: int) -> None:
        """Burn tokens."""
        bal = self.balance_of(owner, token_id)
        if bal < amount:
            raise ValueError("Burn amount exceeds balance")

        self._balances.setdefault(token_id, {})[owner] = bal - amount
        self._supplies[token_id] = self._supplies.get(token_id, 0) - amount

        self._emit(TransferSingleEvent(
            owner, owner, ZERO_ADDRESS, token_id, amount, self.contract_address
        ))

    def burn_batch(self, owner: str, ids: List[int],
                   amounts: List[int]) -> None:
        """Batch burn."""
        if len(ids) != len(amounts):
            raise ValueError("IDs and amounts length mismatch")

        for token_id, amount in zip(ids, amounts):
            bal = self.balance_of(owner, token_id)
            if bal < amount:
                raise ValueError(f"Burn amount exceeds balance for token {token_id}")
            self._balances.setdefault(token_id, {})[owner] = bal - amount
            self._supplies[token_id] = self._supplies.get(token_id, 0) - amount

        self._emit(TransferBatchEvent(
            owner, owner, ZERO_ADDRESS, ids, amounts, self.contract_address
        ))

    def _emit(self, event: TokenEvent) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "standard": "ZX-1155",
            "base_uri": self._base_uri,
            "owner": self._owner,
            "supplies": {str(k): v for k, v in self._supplies.items()},
            "balances": {
                str(tid): dict(accts)
                for tid, accts in self._balances.items()
            },
            "token_uris": {str(k): v for k, v in self._token_uris.items()},
            "operator_approvals": {
                o: dict(ops) for o, ops in self._operator_approvals.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZX1155Token":
        token = cls(
            base_uri=data.get("base_uri", ""),
            owner=data.get("owner", ""),
        )
        token._supplies = {int(k): v for k, v in data.get("supplies", {}).items()}
        token._balances = {
            int(tid): dict(accts)
            for tid, accts in data.get("balances", {}).items()
        }
        token._token_uris = {int(k): v for k, v in data.get("token_uris", {}).items()}
        token._operator_approvals = {
            o: dict(ops) for o, ops in data.get("operator_approvals", {}).items()
        }
        return token
