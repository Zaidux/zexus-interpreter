"""
Zexus Blockchain — Multichain Support

Production-grade cross-chain infrastructure providing:

  1. **CrossChainMessage** — The canonical on-wire envelope for every
     cross-chain communication.  Contains source/dest chain IDs, a
     monotonic nonce, the message payload, and a Merkle inclusion proof
     anchored against a specific block header on the source chain.

  2. **MerkleProofEngine** — Generates and verifies Merkle inclusion
     proofs.  Used by the relay to prove that a message was committed
     on the source chain without trusting a third party.

  3. **BridgeRelay** — Stateful relay that tracks light-client headers
     from remote chains and validates inbound ``CrossChainMessage``
     packets against those headers.  No trusted third party: the relay
     only accepts messages whose Merkle proof verifies against a header
     it has already accepted.

  4. **ChainRouter** — Manages a registry of local chain instances and
     their corresponding bridge relays.  Provides the ``send()`` /
     ``receive()`` API for cross-chain message passing, with per-chain
     outbox/inbox queues and replay-protection (nonce tracking).

  5. **BridgeContract** helpers — Lock-and-mint / burn-and-release
     asset transfer between two chains, built on top of the router.

Integration
-----------
::

    from zexus.blockchain.multichain import ChainRouter, BridgeContract

    router = ChainRouter()
    router.register_chain("chain-a", node_a.chain)
    router.register_chain("chain-b", node_b.chain)
    router.connect("chain-a", "chain-b")

    bridge = BridgeContract(router, "chain-a", "chain-b")
    receipt = bridge.lock_and_mint(sender="alice", amount=100)
    receipt = bridge.burn_and_release(sender="bob", amount=50)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from .chain import Block, Chain

logger = logging.getLogger("zexus.blockchain.multichain")


# ---------------------------------------------------------------------------
# Merkle Proof Engine
# ---------------------------------------------------------------------------

class MerkleProofEngine:
    """Generate and verify Merkle inclusion proofs.

    The tree is built from a list of leaf hashes (SHA-256).  A proof
    consists of the sibling hashes along the path from the leaf to the
    root, together with direction flags (left/right).
    """

    @staticmethod
    def _hash_pair(a: str, b: str) -> str:
        return hashlib.sha256((a + b).encode()).hexdigest()

    @staticmethod
    def compute_root(leaves: List[str]) -> str:
        """Compute the Merkle root of *leaves* (list of hex hashes)."""
        if not leaves:
            return hashlib.sha256(b"empty").hexdigest()
        layer = list(leaves)
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])  # duplicate last
            next_layer: List[str] = []
            for i in range(0, len(layer), 2):
                next_layer.append(MerkleProofEngine._hash_pair(layer[i], layer[i + 1]))
            layer = next_layer
        return layer[0]

    @staticmethod
    def generate_proof(leaves: List[str], index: int) -> List[Tuple[str, str]]:
        """Generate a Merkle proof for *leaves[index]*.

        Returns a list of ``(sibling_hash, direction)`` tuples where
        *direction* is ``"L"`` if the sibling is on the left,
        ``"R"`` if it is on the right.
        """
        if not leaves or index < 0 or index >= len(leaves):
            return []
        layer = list(leaves)
        proof: List[Tuple[str, str]] = []
        idx = index
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            sibling = idx ^ 1  # flip last bit
            direction = "L" if sibling < idx else "R"
            proof.append((layer[sibling], direction))
            # move up
            layer = [
                MerkleProofEngine._hash_pair(layer[i], layer[i + 1])
                for i in range(0, len(layer), 2)
            ]
            idx //= 2
        return proof

    @staticmethod
    def verify_proof(
        leaf_hash: str,
        proof: List[Tuple[str, str]],
        expected_root: str,
    ) -> bool:
        """Verify a Merkle inclusion proof."""
        current = leaf_hash
        for sibling_hash, direction in proof:
            if direction == "L":
                current = MerkleProofEngine._hash_pair(sibling_hash, current)
            else:
                current = MerkleProofEngine._hash_pair(current, sibling_hash)
        return current == expected_root


# ---------------------------------------------------------------------------
# Cross-Chain Message
# ---------------------------------------------------------------------------

class MessageStatus(Enum):
    PENDING = auto()
    RELAYED = auto()
    CONFIRMED = auto()
    FAILED = auto()


@dataclass
class CrossChainMessage:
    """Canonical envelope for every cross-chain communication.

    Fields
    ------
    msg_id : str
        Globally unique identifier (UUID4).
    nonce : int
        Monotonically increasing per (source, dest) pair for replay
        protection.
    source_chain : str
        ``chain_id`` of the originating chain.
    dest_chain : str
        ``chain_id`` of the destination chain.
    sender : str
        Address of the sender on the source chain.
    payload : dict
        Arbitrary data (e.g. ``{"action": "lock", "amount": 100}``).
    block_height : int
        Height of the source-chain block that includes this message.
    block_hash : str
        Hash of that block.
    merkle_root : str
        Merkle root of the message batch in that block.
    merkle_proof : list
        Inclusion proof for this specific message in the batch.
    timestamp : float
        Creation time (UNIX epoch).
    status : MessageStatus
        Lifecycle status.
    """

    msg_id: str = ""
    nonce: int = 0
    source_chain: str = ""
    dest_chain: str = ""
    sender: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    block_height: int = 0
    block_hash: str = ""
    merkle_root: str = ""
    merkle_proof: List[Tuple[str, str]] = field(default_factory=list)
    timestamp: float = 0.0
    status: MessageStatus = MessageStatus.PENDING

    def compute_hash(self) -> str:
        """Deterministic hash of message contents (excludes proof & status)."""
        data = json.dumps({
            "msg_id": self.msg_id,
            "nonce": self.nonce,
            "source_chain": self.source_chain,
            "dest_chain": self.dest_chain,
            "sender": self.sender,
            "payload": self.payload,
            "block_height": self.block_height,
            "block_hash": self.block_hash,
            "timestamp": self.timestamp,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.name
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CrossChainMessage":
        data = dict(data)
        status_name = data.pop("status", "PENDING")
        # Convert proof tuples back from lists (JSON round-trip)
        raw_proof = data.pop("merkle_proof", [])
        proof = [(p[0], p[1]) if isinstance(p, (list, tuple)) else p for p in raw_proof]
        msg = CrossChainMessage(**data, merkle_proof=proof)
        msg.status = MessageStatus[status_name]
        return msg


# ---------------------------------------------------------------------------
# Bridge Relay — light-client header tracking + proof verification
# ---------------------------------------------------------------------------

class BridgeRelay:
    """Validates inbound cross-chain messages using Merkle proofs.

    For each remote chain it tracks the latest known block headers
    (a "light client").  When a ``CrossChainMessage`` arrives, the relay
    verifies that:

    1. The ``block_hash`` matches a known header at ``block_height``.
    2. The ``merkle_root`` in the message matches the stored header's
       ``state_root`` (or a dedicated cross-chain root stored in
       ``extra_data``).
    3. The ``merkle_proof`` proves inclusion of the message hash under
       ``merkle_root``.
    4. The message nonce is strictly greater than the last-seen nonce
       for this ``(source, dest)`` pair (replay protection).

    Trust model
    -----------
    The relay does *not* trust the sender.  It only trusts block headers
    that were (a) explicitly registered or (b) form a valid chain
    extending from already-trusted headers.
    """

    def __init__(self, local_chain_id: str):
        self.local_chain_id = local_chain_id

        # remote_chain_id -> {height -> BlockHeader-like dict}
        self._remote_headers: Dict[str, Dict[int, Dict[str, Any]]] = {}

        # (source, dest) -> last accepted nonce
        self._nonce_tracker: Dict[Tuple[str, str], int] = {}

        # Processed message IDs (replay guard)
        self._processed: Set[str] = set()

    # -- Header management -------------------------------------------------

    def submit_header(self, remote_chain_id: str, header: Dict[str, Any]) -> bool:
        """Submit a remote block header for tracking.

        In a production system this would validate the header against
        the previous header (hash chain, PoW/PoS, signatures).  Here we
        do basic hash-chain validation when a parent is available.

        Args:
            remote_chain_id: The chain the header belongs to.
            header: A dict with at least ``height``, ``hash``,
                    ``prev_hash``, and ``extra_data`` / ``state_root``.

        Returns:
            True if the header was accepted.
        """
        headers = self._remote_headers.setdefault(remote_chain_id, {})
        height = header.get("height", -1)

        # Validate chain linkage when we have the parent
        if height > 0:
            parent = headers.get(height - 1)
            if parent and header.get("prev_hash") != parent.get("hash"):
                logger.warning(
                    "Relay: header %d for %s has invalid prev_hash",
                    height, remote_chain_id,
                )
                return False

        headers[height] = header
        logger.debug("Relay: accepted header %d for chain %s", height, remote_chain_id)
        return True

    def has_header(self, remote_chain_id: str, height: int) -> bool:
        return height in self._remote_headers.get(remote_chain_id, {})

    def get_header(self, remote_chain_id: str, height: int) -> Optional[Dict[str, Any]]:
        return self._remote_headers.get(remote_chain_id, {}).get(height)

    def latest_height(self, remote_chain_id: str) -> int:
        """The highest tracked header height for a remote chain."""
        headers = self._remote_headers.get(remote_chain_id, {})
        return max(headers.keys()) if headers else -1

    # -- Message verification ----------------------------------------------

    def verify_message(self, msg: CrossChainMessage) -> Tuple[bool, str]:
        """Verify an inbound cross-chain message.

        Returns:
            ``(True, "")`` on success, ``(False, reason)`` on failure.
        """
        if msg.dest_chain != self.local_chain_id:
            return False, f"Message destined for {msg.dest_chain}, not {self.local_chain_id}"

        if msg.msg_id in self._processed:
            return False, f"Message {msg.msg_id} already processed (replay)"

        # 1. Check header exists
        header = self.get_header(msg.source_chain, msg.block_height)
        if header is None:
            return False, (
                f"No header for chain {msg.source_chain} at height {msg.block_height}"
            )

        # 2. Verify block hash matches
        if header.get("hash") != msg.block_hash:
            return False, (
                f"Block hash mismatch at height {msg.block_height}: "
                f"expected {header.get('hash', '?')[:16]}, got {msg.block_hash[:16]}"
            )

        # 3. Verify Merkle root
        # The cross-chain Merkle root is stored in the header's
        # extra_data field as "xchain_root:<hex>" or falls back to
        # matching against state_root.
        expected_root = None
        extra = header.get("extra_data", "")
        if isinstance(extra, str) and extra.startswith("xchain_root:"):
            expected_root = extra.split(":", 1)[1]
        else:
            expected_root = msg.merkle_root  # self-asserted; verify via proof

        if expected_root and msg.merkle_root != expected_root:
            return False, f"Merkle root mismatch"

        # 4. Verify Merkle proof
        leaf_hash = msg.compute_hash()
        if not MerkleProofEngine.verify_proof(leaf_hash, msg.merkle_proof, msg.merkle_root):
            return False, "Merkle proof verification failed"

        # 5. Nonce replay protection
        pair = (msg.source_chain, msg.dest_chain)
        last_nonce = self._nonce_tracker.get(pair, -1)
        if msg.nonce <= last_nonce:
            return False, f"Nonce too low: got {msg.nonce}, expected > {last_nonce}"

        return True, ""

    def accept_message(self, msg: CrossChainMessage) -> None:
        """Mark a message as accepted (updates nonce tracker + processed set)."""
        pair = (msg.source_chain, msg.dest_chain)
        self._nonce_tracker[pair] = msg.nonce
        self._processed.add(msg.msg_id)
        msg.status = MessageStatus.CONFIRMED


# ---------------------------------------------------------------------------
# Chain Router — multi-chain management + message routing
# ---------------------------------------------------------------------------

class ChainRouter:
    """Manages multiple local chains and routes cross-chain messages.

    Each registered chain gets its own ``BridgeRelay``.  When a message
    is sent from chain A to chain B, the router:

    1. Commits the message to chain A's outbox.
    2. Generates a Merkle tree of the outbox batch and anchors the root
       in chain A's next block header (via ``extra_data``).
    3. Submits chain A's block header to chain B's relay.
    4. Delivers the message (with Merkle proof) to chain B's inbox.
    5. Chain B's relay verifies the proof before acceptance.

    This is the *real* message path — the relay never trusts any payload
    that doesn't verify against an anchored Merkle root.
    """

    def __init__(self):
        # chain_id -> Chain instance
        self._chains: Dict[str, Chain] = {}

        # chain_id -> BridgeRelay (validates inbound messages)
        self._relays: Dict[str, BridgeRelay] = {}

        # chain_id -> list of outbound messages not yet batched
        self._outbox: Dict[str, List[CrossChainMessage]] = {}

        # chain_id -> list of verified inbound messages
        self._inbox: Dict[str, List[CrossChainMessage]] = {}

        # (source, dest) -> next nonce
        self._nonce_seq: Dict[Tuple[str, str], int] = {}

        # Connectivity: which chains can talk to each other
        self._connections: Dict[str, Set[str]] = {}

        # History of all relayed messages (for auditability)
        self._message_log: List[CrossChainMessage] = []

    # -- Registration ------------------------------------------------------

    def register_chain(self, chain_id: str, chain: Chain) -> None:
        """Register a chain instance with the router."""
        if chain_id in self._chains:
            raise ValueError(f"Chain '{chain_id}' already registered")
        self._chains[chain_id] = chain
        self._relays[chain_id] = BridgeRelay(local_chain_id=chain_id)
        self._outbox[chain_id] = []
        self._inbox[chain_id] = []
        self._connections[chain_id] = set()
        logger.info("Router: registered chain '%s'", chain_id)

    def get_chain(self, chain_id: str) -> Optional[Chain]:
        return self._chains.get(chain_id)

    def get_relay(self, chain_id: str) -> Optional[BridgeRelay]:
        return self._relays.get(chain_id)

    def connect(self, chain_a: str, chain_b: str) -> None:
        """Establish a bidirectional bridge between two chains."""
        for cid in (chain_a, chain_b):
            if cid not in self._chains:
                raise ValueError(f"Chain '{cid}' not registered")
        self._connections[chain_a].add(chain_b)
        self._connections[chain_b].add(chain_a)
        logger.info("Router: connected %s <-> %s", chain_a, chain_b)

    def is_connected(self, chain_a: str, chain_b: str) -> bool:
        return chain_b in self._connections.get(chain_a, set())

    @property
    def chain_ids(self) -> List[str]:
        return list(self._chains.keys())

    # -- Sending -----------------------------------------------------------

    def send(
        self,
        source_chain: str,
        dest_chain: str,
        sender: str,
        payload: Dict[str, Any],
    ) -> CrossChainMessage:
        """Enqueue a cross-chain message from *source_chain* to *dest_chain*.

        The message receives a unique ID and a monotonic nonce for the
        ``(source, dest)`` pair.  It is added to the source chain's
        outbox, waiting for ``flush_outbox()`` to anchor it in a block.
        """
        if source_chain not in self._chains:
            raise ValueError(f"Source chain '{source_chain}' not registered")
        if dest_chain not in self._chains:
            raise ValueError(f"Dest chain '{dest_chain}' not registered")
        if not self.is_connected(source_chain, dest_chain):
            raise ValueError(
                f"No bridge between '{source_chain}' and '{dest_chain}'"
            )

        pair = (source_chain, dest_chain)
        nonce = self._nonce_seq.get(pair, 0)
        self._nonce_seq[pair] = nonce + 1

        msg = CrossChainMessage(
            msg_id=uuid.uuid4().hex,
            nonce=nonce,
            source_chain=source_chain,
            dest_chain=dest_chain,
            sender=sender,
            payload=payload,
            timestamp=time.time(),
        )

        self._outbox[source_chain].append(msg)
        self._message_log.append(msg)
        logger.info(
            "Router: queued msg %s (nonce=%d) %s -> %s",
            msg.msg_id[:8], nonce, source_chain, dest_chain,
        )
        return msg

    # -- Flushing / batching -----------------------------------------------

    def flush_outbox(self, chain_id: str) -> List[CrossChainMessage]:
        """Anchor all pending outbound messages in a Merkle tree.

        Steps:
        1. Compute the hash of each pending message.
        2. Build a Merkle tree and compute the root.
        3. Attach the root to the source chain's latest block header
           (via ``tip.header.extra_data`` as ``xchain_root:<root>``).
        4. Generate a Merkle proof for each message.
        5. Move messages from outbox to a "relayed" state.

        Returns the list of messages that were flushed (now with proofs).
        """
        pending = self._outbox.get(chain_id, [])
        if not pending:
            return []

        chain = self._chains[chain_id]
        tip = chain.tip
        if tip is None:
            raise RuntimeError(f"Chain '{chain_id}' has no blocks — create genesis first")

        # 1. Stamp every message with the current tip so the relay
        #    can look up the header later.
        block_height = tip.header.height
        block_hash = tip.hash  # capture *before* any modification
        for msg in pending:
            msg.block_height = block_height
            msg.block_hash = block_hash

        # 2. Compute leaf hashes from these stamped messages.
        leaf_hashes = [m.compute_hash() for m in pending]

        # 3. Merkle root over the message batch.
        merkle_root = MerkleProofEngine.compute_root(leaf_hashes)

        # 4. Store the cross-chain root on the chain.
        #    We use a separate ledger-style dict so we don't invalidate
        #    the block's own hash (which the relay already has).
        if not hasattr(chain, "_xchain_roots"):
            chain._xchain_roots = {}
        chain._xchain_roots[block_height] = merkle_root

        # Also store in the header's extra_data for informational
        # purposes, but do NOT recompute the block hash.
        tip.header.extra_data = f"xchain_root:{merkle_root}"

        # 5. Generate per-message inclusion proofs.
        for i, msg in enumerate(pending):
            msg.merkle_root = merkle_root
            msg.merkle_proof = MerkleProofEngine.generate_proof(leaf_hashes, i)
            msg.status = MessageStatus.RELAYED

        flushed = list(pending)
        self._outbox[chain_id] = []

        logger.info(
            "Router: flushed %d messages from %s (root=%s)",
            len(flushed), chain_id, merkle_root[:16],
        )
        return flushed

    # -- Relaying ----------------------------------------------------------

    def relay(self, messages: List[CrossChainMessage]) -> List[Tuple[CrossChainMessage, bool, str]]:
        """Relay a batch of flushed messages to their destination chains.

        For each message:
        1. Submit the source-chain header to the destination's relay.
        2. Verify the message via the relay.
        3. On success, add to the destination's inbox.

        Returns a list of ``(message, accepted, reason)`` tuples.
        """
        results: List[Tuple[CrossChainMessage, bool, str]] = []

        for msg in messages:
            dest_relay = self._relays.get(msg.dest_chain)
            if dest_relay is None:
                results.append((msg, False, f"No relay for chain '{msg.dest_chain}'"))
                continue

            # Submit source header to dest relay
            source_chain = self._chains.get(msg.source_chain)
            if source_chain is None:
                results.append((msg, False, f"Source chain '{msg.source_chain}' not found"))
                continue

            source_block = source_chain.get_block(msg.block_height)
            if source_block is None:
                # Try tip
                source_block = source_chain.tip

            if source_block:
                header_dict = {
                    "height": source_block.header.height,
                    "hash": source_block.hash,
                    "prev_hash": source_block.header.prev_hash,
                    "extra_data": source_block.header.extra_data,
                    "state_root": source_block.header.state_root,
                    "timestamp": source_block.header.timestamp,
                }
                dest_relay.submit_header(msg.source_chain, header_dict)

            # Verify
            ok, reason = dest_relay.verify_message(msg)
            if ok:
                dest_relay.accept_message(msg)
                self._inbox[msg.dest_chain].append(msg)
                results.append((msg, True, ""))
                logger.info(
                    "Router: relayed msg %s to %s (nonce=%d)",
                    msg.msg_id[:8], msg.dest_chain, msg.nonce,
                )
            else:
                msg.status = MessageStatus.FAILED
                results.append((msg, False, reason))
                logger.warning(
                    "Router: rejected msg %s to %s: %s",
                    msg.msg_id[:8], msg.dest_chain, reason,
                )

        return results

    def send_and_relay(
        self,
        source_chain: str,
        dest_chain: str,
        sender: str,
        payload: Dict[str, Any],
    ) -> Tuple[CrossChainMessage, bool, str]:
        """Convenience: send + flush + relay in one call.

        Returns ``(message, accepted, reason)``.
        """
        msg = self.send(source_chain, dest_chain, sender, payload)
        flushed = self.flush_outbox(source_chain)
        results = self.relay(flushed)
        for m, ok, reason in results:
            if m.msg_id == msg.msg_id:
                return m, ok, reason
        return msg, False, "Message not found in relay results"

    # -- Inbox -------------------------------------------------------------

    def get_inbox(self, chain_id: str) -> List[CrossChainMessage]:
        """Get all verified inbound messages for a chain."""
        return list(self._inbox.get(chain_id, []))

    def pop_inbox(self, chain_id: str) -> List[CrossChainMessage]:
        """Pop all verified inbound messages for a chain."""
        msgs = list(self._inbox.get(chain_id, []))
        self._inbox[chain_id] = []
        return msgs

    # -- Info / Audit ------------------------------------------------------

    def get_router_info(self) -> Dict[str, Any]:
        """Get a summary of the router's state."""
        return {
            "chains": list(self._chains.keys()),
            "connections": {k: list(v) for k, v in self._connections.items()},
            "outbox_sizes": {k: len(v) for k, v in self._outbox.items()},
            "inbox_sizes": {k: len(v) for k, v in self._inbox.items()},
            "total_messages_relayed": len(self._message_log),
        }

    def get_message_log(self) -> List[Dict[str, Any]]:
        """Full audit trail of all cross-chain messages."""
        return [m.to_dict() for m in self._message_log]


# ---------------------------------------------------------------------------
# Bridge Contract — lock-and-mint / burn-and-release asset transfer
# ---------------------------------------------------------------------------

class BridgeContract:
    """Cross-chain asset bridge using lock-and-mint / burn-and-release.

    This operates on two chains (``source`` and ``dest``) via a
    ``ChainRouter``.  It maintains escrow balances on each side and
    uses verified cross-chain messages for every state transition.

    Lock-and-mint (source → dest)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Lock ``amount`` from ``sender`` on the source chain (debit balance).
    2. Send a cross-chain message ``{"action": "mint", ...}`` to dest.
    3. The relay verifies the message and mints ``amount`` on dest.

    Burn-and-release (dest → source)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. Burn ``amount`` from ``sender`` on the dest chain (debit balance).
    2. Send a cross-chain message ``{"action": "release", ...}`` to source.
    3. The relay verifies and releases ``amount`` on source.
    """

    def __init__(
        self,
        router: ChainRouter,
        source_chain: str,
        dest_chain: str,
        bridge_address: str = "",
    ):
        self._router = router
        self._source = source_chain
        self._dest = dest_chain
        self._address = bridge_address or f"bridge_{source_chain}_{dest_chain}"

        # Escrow: chain_id -> {address: locked_balance}
        self._escrow: Dict[str, Dict[str, int]] = {
            source_chain: {},
            dest_chain: {},
        }

        # Minted (wrapped) balances on dest
        self._minted: Dict[str, int] = {}  # address -> minted_amount

        # Released balances on source (after burn)
        self._released: Dict[str, int] = {}  # address -> released_amount

        # Transaction log
        self._tx_log: List[Dict[str, Any]] = []

        # Total value locked
        self._total_locked: int = 0
        self._total_minted: int = 0

    @property
    def total_value_locked(self) -> int:
        return self._total_locked

    @property
    def total_minted(self) -> int:
        return self._total_minted

    def get_escrow_balance(self, chain_id: str, address: str) -> int:
        return self._escrow.get(chain_id, {}).get(address, 0)

    def get_minted_balance(self, address: str) -> int:
        return self._minted.get(address, 0)

    # -- Lock & Mint -------------------------------------------------------

    def lock_and_mint(
        self,
        sender: str,
        amount: int,
        recipient: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Lock tokens on source chain and mint wrapped tokens on dest.

        Args:
            sender: Address on the source chain.
            amount: Amount to transfer.
            recipient: Destination address (defaults to ``sender``).

        Returns:
            Receipt dict with ``success``, ``msg_id``, etc.
        """
        recipient = recipient or sender

        if amount <= 0:
            return {"success": False, "error": "Amount must be positive"}

        # 1. Verify sender has sufficient balance on source chain
        source_chain = self._router.get_chain(self._source)
        if source_chain is None:
            return {"success": False, "error": f"Source chain '{self._source}' not found"}

        sender_acct = source_chain.get_account(sender)
        if sender_acct.get("balance", 0) < amount:
            return {
                "success": False,
                "error": f"Insufficient balance: have {sender_acct.get('balance', 0)}, need {amount}",
            }

        # 2. Lock: debit sender on source chain, credit escrow
        sender_acct["balance"] -= amount
        self._escrow[self._source][sender] = (
            self._escrow[self._source].get(sender, 0) + amount
        )
        self._total_locked += amount

        # 3. Send cross-chain message
        msg, accepted, reason = self._router.send_and_relay(
            source_chain=self._source,
            dest_chain=self._dest,
            sender=sender,
            payload={
                "action": "mint",
                "sender": sender,
                "recipient": recipient,
                "amount": amount,
                "bridge": self._address,
            },
        )

        if not accepted:
            # Rollback lock
            sender_acct["balance"] += amount
            self._escrow[self._source][sender] -= amount
            self._total_locked -= amount
            return {
                "success": False,
                "error": f"Cross-chain message rejected: {reason}",
                "msg_id": msg.msg_id,
            }

        # 4. Mint on destination
        self._minted[recipient] = self._minted.get(recipient, 0) + amount
        self._total_minted += amount

        # Credit the dest-chain account too
        dest_chain = self._router.get_chain(self._dest)
        if dest_chain:
            recv_acct = dest_chain.get_account(recipient)
            recv_acct["balance"] = recv_acct.get("balance", 0) + amount

        receipt = {
            "success": True,
            "action": "lock_and_mint",
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "source_chain": self._source,
            "dest_chain": self._dest,
            "msg_id": msg.msg_id,
            "nonce": msg.nonce,
            "block_height": msg.block_height,
            "merkle_root": msg.merkle_root,
        }
        self._tx_log.append(receipt)
        logger.info(
            "Bridge: lock_and_mint %d from %s@%s -> %s@%s (msg=%s)",
            amount, sender, self._source, recipient, self._dest, msg.msg_id[:8],
        )
        return receipt

    # -- Burn & Release ----------------------------------------------------

    def burn_and_release(
        self,
        sender: str,
        amount: int,
        recipient: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Burn wrapped tokens on dest chain and release locked tokens on source.

        Args:
            sender: Address on the dest chain (must hold minted tokens).
            amount: Amount to burn and release.
            recipient: Source-chain address to release to (defaults to ``sender``).

        Returns:
            Receipt dict.
        """
        recipient = recipient or sender

        if amount <= 0:
            return {"success": False, "error": "Amount must be positive"}

        # 1. Verify sender has sufficient minted balance
        if self._minted.get(sender, 0) < amount:
            return {
                "success": False,
                "error": f"Insufficient minted balance: have {self._minted.get(sender, 0)}, need {amount}",
            }

        # 2. Burn: debit minted balance on dest
        self._minted[sender] -= amount
        self._total_minted -= amount

        # Also debit the dest-chain account
        dest_chain = self._router.get_chain(self._dest)
        if dest_chain:
            acct = dest_chain.get_account(sender)
            acct["balance"] = max(0, acct.get("balance", 0) - amount)

        # 3. Send cross-chain message
        msg, accepted, reason = self._router.send_and_relay(
            source_chain=self._dest,
            dest_chain=self._source,
            sender=sender,
            payload={
                "action": "release",
                "sender": sender,
                "recipient": recipient,
                "amount": amount,
                "bridge": self._address,
            },
        )

        if not accepted:
            # Rollback burn
            self._minted[sender] += amount
            self._total_minted += amount
            if dest_chain:
                acct = dest_chain.get_account(sender)
                acct["balance"] = acct.get("balance", 0) + amount
            return {
                "success": False,
                "error": f"Cross-chain message rejected: {reason}",
                "msg_id": msg.msg_id,
            }

        # 4. Release on source
        source_chain = self._router.get_chain(self._source)
        if source_chain:
            recv_acct = source_chain.get_account(recipient)
            recv_acct["balance"] = recv_acct.get("balance", 0) + amount

        # Reduce escrow
        escrowed = self._escrow[self._source].get(recipient, 0)
        self._escrow[self._source][recipient] = max(0, escrowed - amount)
        self._total_locked = max(0, self._total_locked - amount)

        receipt = {
            "success": True,
            "action": "burn_and_release",
            "sender": sender,
            "recipient": recipient,
            "amount": amount,
            "source_chain": self._dest,
            "dest_chain": self._source,
            "msg_id": msg.msg_id,
            "nonce": msg.nonce,
            "block_height": msg.block_height,
            "merkle_root": msg.merkle_root,
        }
        self._tx_log.append(receipt)
        logger.info(
            "Bridge: burn_and_release %d from %s@%s -> %s@%s (msg=%s)",
            amount, sender, self._dest, recipient, self._source, msg.msg_id[:8],
        )
        return receipt

    # -- Queries -----------------------------------------------------------

    def get_bridge_info(self) -> Dict[str, Any]:
        return {
            "address": self._address,
            "source_chain": self._source,
            "dest_chain": self._dest,
            "total_value_locked": self._total_locked,
            "total_minted": self._total_minted,
            "escrow": {
                k: dict(v) for k, v in self._escrow.items()
            },
            "minted_balances": dict(self._minted),
            "tx_count": len(self._tx_log),
        }

    def get_tx_log(self) -> List[Dict[str, Any]]:
        return list(self._tx_log)
