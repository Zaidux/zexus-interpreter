"""
Zexus Blockchain — Pluggable Consensus Engine

Provides a consensus interface and three concrete implementations:
  - Proof-of-Work (PoW): nonce-grinding on block header hash
  - Proof-of-Authority (PoA): pre-approved validator set with round-robin
  - Proof-of-Stake (PoS): weighted random selection by staked balance

Each engine produces blocks (``seal``) and validates them (``verify``).
The ``ConsensusEngine`` abstract base class defines the contract.
"""

import abc
import hashlib
import json
import random
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .chain import Block, BlockHeader, Transaction, TransactionReceipt, Chain, Mempool

logger = logging.getLogger("zexus.blockchain.consensus")


class ConsensusEngine(abc.ABC):
    """Abstract consensus engine interface."""

    @abc.abstractmethod
    def seal(self, chain: Chain, mempool: Mempool, miner: str) -> Optional[Block]:
        """Produce a new block from pending transactions.
        
        Args:
            chain: Current blockchain state.
            mempool: Pending transaction pool.
            miner: Address of the block producer.
            
        Returns:
            A sealed Block ready to add to chain, or None if unable.
        """
        ...

    @abc.abstractmethod
    def verify(self, block: Block, chain: Chain) -> bool:
        """Verify that a block satisfies the consensus rules.
        
        Args:
            block: Block to verify.
            chain: Chain context (for parent lookup etc.).
            
        Returns:
            True if the block is valid under this consensus.
        """
        ...

    def _prepare_block(self, chain: Chain, mempool: Mempool, miner: str,
                       gas_limit: int = 10_000_000) -> Block:
        """Prepare a block template with pending transactions.
        
        Shared helper for all consensus implementations.
        """
        tip = chain.tip
        block = Block()
        block.header.version = 1
        block.header.height = (tip.header.height + 1) if tip else 0
        block.header.prev_hash = tip.hash if tip else "0" * 64
        block.header.timestamp = time.time()
        block.header.miner = miner
        block.header.gas_limit = gas_limit

        # Select transactions
        txs = mempool.get_pending(gas_limit)
        block.transactions = txs
        total_gas = sum(tx.gas_limit for tx in txs)
        block.header.gas_used = total_gas
        block.header.tx_root = block.compute_tx_root()

        # Execute transactions and produce receipts
        for tx in txs:
            receipt = self._execute_tx(tx, chain, block.header.height)
            block.receipts.append(receipt)

        return block

    def _execute_tx(self, tx: Transaction, chain: Chain, block_height: int) -> TransactionReceipt:
        """Execute a single transaction against the chain state.
        
        Returns a receipt. Keeps it simple — value transfer + gas.
        Contract execution hooks into the evaluator (done at node level).
        """
        sender_acct = chain.get_account(tx.sender)
        cost = tx.value + (tx.gas_limit * tx.gas_price)
        
        receipt = TransactionReceipt(
            tx_hash=tx.tx_hash,
            block_height=block_height,
            gas_used=tx.gas_limit,
        )

        # Validate
        if sender_acct["balance"] < cost:
            receipt.status = 0
            receipt.revert_reason = "insufficient balance"
            return receipt

        if tx.nonce < sender_acct["nonce"]:
            receipt.status = 0
            receipt.revert_reason = "nonce too low"
            return receipt

        # Success
        receipt.status = 1
        
        # If it's a contract deployment (no recipient)
        if not tx.recipient and tx.data:
            contract_addr = hashlib.sha256(
                f"{tx.sender}{tx.nonce}".encode()
            ).hexdigest()[:40]
            receipt.contract_address = contract_addr

        return receipt


class ProofOfWork(ConsensusEngine):
    """Proof-of-Work consensus using SHA-256 nonce grinding.
    
    The block hash must start with ``difficulty`` zero hex chars.
    """

    def __init__(self, difficulty: int = 1, max_iterations: int = 10_000_000):
        self.difficulty = difficulty
        self.max_iterations = max_iterations

    def seal(self, chain: Chain, mempool: Mempool, miner: str) -> Optional[Block]:
        block = self._prepare_block(chain, mempool, miner)
        block.header.difficulty = self.difficulty

        target = "0" * self.difficulty
        for nonce in range(self.max_iterations):
            block.header.nonce = nonce
            h = block.header.compute_hash()
            if h.startswith(target):
                block.hash = h
                logger.info("PoW: mined block %d (nonce=%d, hash=%s)",
                            block.header.height, nonce, h[:16])
                # Remove selected txs from mempool
                for tx in block.transactions:
                    mempool.remove(tx.tx_hash)
                return block

        logger.warning("PoW: failed to find valid nonce in %d iterations", self.max_iterations)
        return None

    def verify(self, block: Block, chain: Chain) -> bool:
        target = "0" * block.header.difficulty
        computed = block.header.compute_hash()
        if computed != block.hash:
            return False
        if not block.hash.startswith(target):
            return False
        return True


class ProofOfAuthority(ConsensusEngine):
    """Proof-of-Authority consensus with a fixed validator set.
    
    Validators take turns producing blocks in round-robin order.
    Only authorized validators can seal blocks.
    """

    def __init__(self, validators: Optional[List[str]] = None,
                 block_interval: float = 5.0):
        self.validators: List[str] = validators or []
        self.block_interval = block_interval

    def add_validator(self, address: str):
        if address not in self.validators:
            self.validators.append(address)

    def remove_validator(self, address: str):
        self.validators = [v for v in self.validators if v != address]

    def _current_validator(self, height: int) -> Optional[str]:
        """Determine which validator should produce the block at ``height``."""
        if not self.validators:
            return None
        idx = height % len(self.validators)
        return self.validators[idx]

    def seal(self, chain: Chain, mempool: Mempool, miner: str) -> Optional[Block]:
        if miner not in self.validators:
            logger.warning("PoA: %s is not an authorized validator", miner[:8])
            return None

        next_height = chain.height + 1
        expected = self._current_validator(next_height)
        if expected != miner:
            logger.debug("PoA: not %s's turn (expected %s)", miner[:8], expected[:8] if expected else "?")
            return None

        block = self._prepare_block(chain, mempool, miner)
        block.header.difficulty = 0  # No PoW
        block.header.extra_data = f"poa:validator={miner[:8]}"
        block.compute_hash()

        for tx in block.transactions:
            mempool.remove(tx.tx_hash)

        logger.info("PoA: validator %s sealed block %d", miner[:8], block.header.height)
        return block

    def verify(self, block: Block, chain: Chain) -> bool:
        # Verify the miner was the correct validator for this height
        expected = self._current_validator(block.header.height)
        if expected != block.header.miner:
            return False
        if block.header.miner not in self.validators:
            return False
        computed = block.header.compute_hash()
        return computed == block.hash


class ProofOfStake(ConsensusEngine):
    """Simple Proof-of-Stake consensus.
    
    Validators are selected with probability proportional to their
    staked balance. Uses a VRF-like (verifiable random function by
    hashing parent hash + height) deterministic selection.
    """

    def __init__(self, min_stake: int = 1000):
        self.min_stake = min_stake
        self.stakes: Dict[str, int] = {}  # validator -> stake amount

    def stake(self, validator: str, amount: int):
        """Register or increase a validator's stake."""
        current = self.stakes.get(validator, 0)
        self.stakes[validator] = current + amount

    def unstake(self, validator: str, amount: int) -> bool:
        """Reduce a validator's stake."""
        current = self.stakes.get(validator, 0)
        if amount > current:
            return False
        self.stakes[validator] = current - amount
        if self.stakes[validator] <= 0:
            del self.stakes[validator]
        return True

    def get_eligible_validators(self) -> List[str]:
        """Return validators with at least min_stake."""
        return [v for v, s in self.stakes.items() if s >= self.min_stake]

    def _select_validator(self, parent_hash: str, height: int) -> Optional[str]:
        """Deterministically select a validator based on parent hash + height."""
        eligible = self.get_eligible_validators()
        if not eligible:
            return None

        # Deterministic seed from chain state
        seed_data = f"{parent_hash}{height}".encode()
        seed_hash = hashlib.sha256(seed_data).hexdigest()
        seed_int = int(seed_hash[:16], 16)

        # Weighted selection
        total_stake = sum(self.stakes[v] for v in eligible)
        target = seed_int % total_stake
        cumulative = 0
        for v in sorted(eligible):  # Sort for deterministic ordering
            cumulative += self.stakes[v]
            if cumulative > target:
                return v
        return eligible[-1]

    def seal(self, chain: Chain, mempool: Mempool, miner: str) -> Optional[Block]:
        if miner not in self.stakes or self.stakes[miner] < self.min_stake:
            logger.warning("PoS: %s has insufficient stake", miner[:8])
            return None

        tip = chain.tip
        parent_hash = tip.hash if tip else "0" * 64
        next_height = chain.height + 1

        selected = self._select_validator(parent_hash, next_height)
        if selected != miner:
            logger.debug("PoS: %s not selected (selected=%s)", miner[:8], selected[:8] if selected else "?")
            return None

        block = self._prepare_block(chain, mempool, miner)
        block.header.difficulty = 0
        block.header.extra_data = f"pos:stake={self.stakes.get(miner, 0)}"
        block.compute_hash()

        for tx in block.transactions:
            mempool.remove(tx.tx_hash)

        logger.info("PoS: validator %s sealed block %d (stake=%d)",
                     miner[:8], block.header.height, self.stakes.get(miner, 0))
        return block

    def verify(self, block: Block, chain: Chain) -> bool:
        parent = chain.get_block(block.header.height - 1)
        parent_hash = parent.hash if parent else "0" * 64
        selected = self._select_validator(parent_hash, block.header.height)
        if selected != block.header.miner:
            return False
        computed = block.header.compute_hash()
        return computed == block.hash


# ══════════════════════════════════════════════════════════════════════
#  BFT Consensus (PBFT / Tendermint-style)
# ══════════════════════════════════════════════════════════════════════

class BFTPhase:
    """PBFT round phases."""
    PRE_PREPARE = "pre-prepare"
    PREPARE = "prepare"
    COMMIT = "commit"
    DECIDED = "decided"


@dataclass
class BFTMessage:
    """A PBFT protocol message."""
    phase: str  # BFTPhase value
    view: int  # Current view number (leader rotation epoch)
    height: int  # Block height this message is about
    block_hash: str  # Hash of the proposed block
    sender: str  # Validator address
    signature: str = ""  # HMAC-SHA256 signature
    timestamp: float = field(default_factory=time.time)

    def signing_payload(self) -> str:
        return f"{self.phase}:{self.view}:{self.height}:{self.block_hash}:{self.sender}"

    def sign(self, key: bytes) -> None:
        import hmac as _hmac
        self.signature = _hmac.new(key, self.signing_payload().encode(), hashlib.sha256).hexdigest()

    def verify_signature(self, key: bytes) -> bool:
        import hmac as _hmac
        expected = _hmac.new(key, self.signing_payload().encode(), hashlib.sha256).hexdigest()
        return self.signature == expected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "view": self.view,
            "height": self.height,
            "block_hash": self.block_hash,
            "sender": self.sender,
            "signature": self.signature,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BFTMessage":
        return cls(
            phase=d["phase"],
            view=d["view"],
            height=d["height"],
            block_hash=d["block_hash"],
            sender=d["sender"],
            signature=d.get("signature", ""),
            timestamp=d.get("timestamp", 0.0),
        )


@dataclass
class BFTRoundState:
    """Tracks the state of a single PBFT round."""
    view: int = 0
    height: int = 0
    phase: str = BFTPhase.PRE_PREPARE
    proposed_block: Optional[Block] = None
    proposed_hash: str = ""

    # Votes collected per phase: sender -> BFTMessage
    pre_prepare: Optional[BFTMessage] = None
    prepares: Dict[str, BFTMessage] = field(default_factory=dict)
    commits: Dict[str, BFTMessage] = field(default_factory=dict)

    decided: bool = False
    locked_block: Optional[Block] = None

    def prepare_count(self) -> int:
        return len(self.prepares)

    def commit_count(self) -> int:
        return len(self.commits)


class BFTConsensus(ConsensusEngine):
    """Byzantine Fault Tolerant consensus engine.

    Implements a PBFT / Tendermint-style 3-phase commit protocol:

    1. **Pre-Prepare** — The leader (proposer) creates a block and
       broadcasts a PRE-PREPARE message.
    2. **Prepare** — Validators validate the proposed block and
       broadcast PREPARE votes.
    3. **Commit** — Once ≥ 2f+1 PREPARE messages are collected,
       validators broadcast COMMIT votes.
    4. **Decide** — When ≥ 2f+1 COMMIT messages are collected the
       block is finalized.

    Where *f = (n-1) // 3* is the maximum tolerable Byzantine faults
    and *n* is the total validator count.

    Features:
    - Round-robin leader rotation (``view`` mod ``len(validators)``)
    - View-change on timeout (leader failure recovery)
    - Instant finality — no probabilistic reorgs
    - Block locking to prevent equivocation
    """

    def __init__(self, validators: Optional[List[str]] = None,
                 block_interval: float = 2.0,
                 view_change_timeout: float = 10.0):
        self.validators: List[str] = list(validators or [])
        self.block_interval = block_interval
        self.view_change_timeout = view_change_timeout

        # Current round state
        self.view: int = 0
        self._rounds: Dict[int, BFTRoundState] = {}  # height -> round

        # View-change tracking
        self._view_change_votes: Dict[int, Set[str]] = {}  # new_view -> set of senders
        self._last_block_time: float = time.time()

        # Finalized blocks awaiting pickup
        self._finalized_blocks: List[Block] = []

        # Validator signing keys (shared secret per validator for simulation)
        self._validator_keys: Dict[str, bytes] = {}

    # ── Properties ────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Total validator count."""
        return len(self.validators)

    @property
    def f(self) -> int:
        """Max tolerable Byzantine faults: (n-1) // 3."""
        return (self.n - 1) // 3

    @property
    def quorum(self) -> int:
        """Quorum size: 2f + 1."""
        return 2 * self.f + 1

    # ── Validator management ──────────────────────────────────────

    def add_validator(self, address: str, key: Optional[bytes] = None) -> None:
        if address not in self.validators:
            self.validators.append(address)
        if key:
            self._validator_keys[address] = key
        else:
            self._validator_keys.setdefault(
                address, hashlib.sha256(address.encode()).digest()
            )

    def remove_validator(self, address: str) -> None:
        self.validators = [v for v in self.validators if v != address]
        self._validator_keys.pop(address, None)

    def _get_leader(self, view: int) -> Optional[str]:
        """Round-robin leader selection based on view number."""
        if not self.validators:
            return None
        return self.validators[view % len(self.validators)]

    def _get_round(self, height: int) -> BFTRoundState:
        if height not in self._rounds:
            self._rounds[height] = BFTRoundState(view=self.view, height=height)
        return self._rounds[height]

    def _cleanup_old_rounds(self, current_height: int) -> None:
        """Remove round state for heights well below current."""
        old = [h for h in self._rounds if h < current_height - 10]
        for h in old:
            del self._rounds[h]

    # ── PBFT Phases ───────────────────────────────────────────────

    def propose(self, chain: Chain, mempool: Mempool, miner: str) -> Optional[BFTMessage]:
        """Phase 1: Leader creates a block and broadcasts PRE-PREPARE.

        Returns the BFTMessage (PRE-PREPARE) to broadcast.
        """
        leader = self._get_leader(self.view)
        if leader != miner:
            return None

        height = chain.height + 1
        block = self._prepare_block(chain, mempool, miner)
        block.header.difficulty = 0
        block.header.extra_data = json.dumps({
            "bft": True,
            "view": self.view,
            "proposer": miner[:16],
        })
        block.compute_hash()

        rnd = self._get_round(height)
        rnd.proposed_block = block
        rnd.proposed_hash = block.hash
        rnd.phase = BFTPhase.PRE_PREPARE

        msg = BFTMessage(
            phase=BFTPhase.PRE_PREPARE,
            view=self.view,
            height=height,
            block_hash=block.hash,
            sender=miner,
        )
        key = self._validator_keys.get(miner, b"")
        if key:
            msg.sign(key)

        rnd.pre_prepare = msg
        logger.info("BFT: PRE-PREPARE (view=%d, height=%d, hash=%s)",
                     self.view, height, block.hash[:16])
        return msg

    def on_pre_prepare(self, msg: BFTMessage, proposed_block: Block,
                       chain: Chain, validator: str) -> Optional[BFTMessage]:
        """Handle a received PRE-PREPARE: validate and send PREPARE."""
        leader = self._get_leader(msg.view)
        if msg.sender != leader:
            logger.warning("BFT: PRE-PREPARE from non-leader %s", msg.sender[:8])
            return None

        if msg.view < self.view:
            return None

        # Validate the proposed block
        if msg.block_hash != proposed_block.hash:
            return None

        rnd = self._get_round(msg.height)
        rnd.proposed_block = proposed_block
        rnd.proposed_hash = msg.block_hash
        rnd.pre_prepare = msg
        rnd.phase = BFTPhase.PREPARE

        # Send PREPARE
        prepare = BFTMessage(
            phase=BFTPhase.PREPARE,
            view=msg.view,
            height=msg.height,
            block_hash=msg.block_hash,
            sender=validator,
        )
        key = self._validator_keys.get(validator, b"")
        if key:
            prepare.sign(key)

        rnd.prepares[validator] = prepare
        logger.debug("BFT: PREPARE from %s (height=%d)", validator[:8], msg.height)
        return prepare

    def on_prepare(self, msg: BFTMessage, validator: str) -> Optional[BFTMessage]:
        """Handle a received PREPARE vote.

        When quorum is reached, emit a COMMIT.
        """
        if msg.sender not in self.validators:
            return None

        rnd = self._get_round(msg.height)

        if msg.block_hash != rnd.proposed_hash:
            return None

        rnd.prepares[msg.sender] = msg

        # Check quorum
        if rnd.prepare_count() >= self.quorum and rnd.phase == BFTPhase.PREPARE:
            rnd.phase = BFTPhase.COMMIT
            rnd.locked_block = rnd.proposed_block

            commit = BFTMessage(
                phase=BFTPhase.COMMIT,
                view=msg.view,
                height=msg.height,
                block_hash=msg.block_hash,
                sender=validator,
            )
            key = self._validator_keys.get(validator, b"")
            if key:
                commit.sign(key)

            rnd.commits[validator] = commit
            logger.info("BFT: COMMIT (quorum reached, height=%d, prepares=%d)",
                        msg.height, rnd.prepare_count())
            return commit

        return None

    def on_commit(self, msg: BFTMessage) -> Optional[Block]:
        """Handle a received COMMIT vote.

        Returns the finalized block if commit quorum is reached.
        """
        if msg.sender not in self.validators:
            return None

        rnd = self._get_round(msg.height)

        if msg.block_hash != rnd.proposed_hash:
            return None

        rnd.commits[msg.sender] = msg

        if rnd.commit_count() >= self.quorum and not rnd.decided:
            rnd.decided = True
            rnd.phase = BFTPhase.DECIDED
            self._last_block_time = time.time()

            logger.info("BFT: DECIDED block %d (commits=%d/%d)",
                        msg.height, rnd.commit_count(), self.n)

            if rnd.proposed_block:
                self._finalized_blocks.append(rnd.proposed_block)
                return rnd.proposed_block

        return None

    # ── View Change ───────────────────────────────────────────────

    def request_view_change(self, validator: str) -> int:
        """Initiate a view change (leader rotation).

        Returns the proposed new view number.
        """
        new_view = self.view + 1
        if new_view not in self._view_change_votes:
            self._view_change_votes[new_view] = set()

        self._view_change_votes[new_view].add(validator)

        if len(self._view_change_votes[new_view]) >= self.quorum:
            old_view = self.view
            self.view = new_view
            self._view_change_votes.pop(new_view, None)
            logger.info("BFT: VIEW CHANGE %d -> %d (new leader: %s)",
                        old_view, new_view,
                        self._get_leader(new_view)[:8] if self._get_leader(new_view) else "?")
        return new_view

    def check_timeout(self, validator: str) -> bool:
        """Check if the current round has timed out, triggering view change."""
        elapsed = time.time() - self._last_block_time
        if elapsed > self.view_change_timeout:
            self.request_view_change(validator)
            return True
        return False

    # ── ConsensusEngine interface ─────────────────────────────────

    def seal(self, chain: Chain, mempool: Mempool, miner: str) -> Optional[Block]:
        """Full BFT round: propose → prepare → commit → decide.

        For local/single-process simulation, runs all phases
        synchronously with all validators. In production, each
        phase would be triggered by network messages.
        """
        if not self.validators:
            logger.warning("BFT: no validators configured")
            return None

        if miner not in self.validators:
            logger.warning("BFT: %s is not a validator", miner[:8])
            return None

        leader = self._get_leader(self.view)
        if leader != miner:
            return None

        height = chain.height + 1
        self._cleanup_old_rounds(height)

        # Phase 1: PRE-PREPARE
        pre_prepare = self.propose(chain, mempool, miner)
        if not pre_prepare:
            return None

        rnd = self._get_round(height)

        # Phase 2: PREPARE — simulate all validators responding
        for v in self.validators:
            if v == miner:
                # Leader also prepares
                rnd.prepares[v] = BFTMessage(
                    phase=BFTPhase.PREPARE,
                    view=self.view,
                    height=height,
                    block_hash=rnd.proposed_hash,
                    sender=v,
                )
                continue
            prepare = self.on_pre_prepare(pre_prepare, rnd.proposed_block, chain, v)
            if prepare:
                rnd.prepares[v] = prepare

        if rnd.prepare_count() < self.quorum:
            logger.warning("BFT: not enough prepares (%d/%d)",
                           rnd.prepare_count(), self.quorum)
            return None

        rnd.phase = BFTPhase.COMMIT

        # Phase 3: COMMIT — simulate all validators committing
        for v in self.validators:
            commit = BFTMessage(
                phase=BFTPhase.COMMIT,
                view=self.view,
                height=height,
                block_hash=rnd.proposed_hash,
                sender=v,
            )
            rnd.commits[v] = commit

        if rnd.commit_count() < self.quorum:
            logger.warning("BFT: not enough commits (%d/%d)",
                           rnd.commit_count(), self.quorum)
            return None

        # Phase 4: DECIDE
        rnd.decided = True
        rnd.phase = BFTPhase.DECIDED
        self._last_block_time = time.time()

        block = rnd.proposed_block
        if block:
            for tx in block.transactions:
                mempool.remove(tx.tx_hash)
            logger.info("BFT: sealed block %d (view=%d, validators=%d, quorum=%d)",
                        height, self.view, self.n, self.quorum)
            return block

        return None

    def verify(self, block: Block, chain: Chain) -> bool:
        """Verify a BFT-sealed block.

        Checks:
        - Block hash matches computed hash
        - Extra data contains BFT metadata
        - Proposer was the correct leader for the view
        """
        computed = block.header.compute_hash()
        if computed != block.hash:
            return False

        # Parse BFT metadata from extra_data
        try:
            meta = json.loads(block.header.extra_data)
        except (json.JSONDecodeError, TypeError):
            return False

        if not meta.get("bft"):
            return False

        view = meta.get("view", 0)
        leader = self._get_leader(view)
        if leader != block.header.miner:
            return False

        return True

    def get_round_state(self, height: int) -> Optional[BFTRoundState]:
        """Get the round state for a given height (for debugging/RPC)."""
        return self._rounds.get(height)

    def get_status(self) -> Dict[str, Any]:
        """Return current BFT consensus status."""
        leader = self._get_leader(self.view)
        return {
            "algorithm": "bft",
            "view": self.view,
            "validators": list(self.validators),
            "validator_count": self.n,
            "max_faults": self.f,
            "quorum": self.quorum,
            "current_leader": leader or "",
            "block_interval": self.block_interval,
            "view_change_timeout": self.view_change_timeout,
            "active_rounds": list(self._rounds.keys()),
        }


# ── Factory helper ─────────────────────────────────────────────────────

def create_consensus(algorithm: str = "pow", **kwargs) -> ConsensusEngine:
    """Create a consensus engine by name.
    
    Args:
        algorithm: One of 'pow', 'poa', 'pos', 'bft'.
        **kwargs: Passed to the constructor.
        
    Returns:
        ConsensusEngine instance.
    """
    engines = {
        "pow": ProofOfWork,
        "poa": ProofOfAuthority,
        "pos": ProofOfStake,
        "bft": BFTConsensus,
    }
    cls = engines.get(algorithm.lower())
    if not cls:
        raise ValueError(f"Unknown consensus algorithm: {algorithm}. Choose from: {list(engines.keys())}")
    return cls(**kwargs)
