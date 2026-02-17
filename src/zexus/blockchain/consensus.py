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


# ── Factory helper ─────────────────────────────────────────────────────

def create_consensus(algorithm: str = "pow", **kwargs) -> ConsensusEngine:
    """Create a consensus engine by name.
    
    Args:
        algorithm: One of 'pow', 'poa', 'pos'.
        **kwargs: Passed to the constructor.
        
    Returns:
        ConsensusEngine instance.
    """
    engines = {
        "pow": ProofOfWork,
        "poa": ProofOfAuthority,
        "pos": ProofOfStake,
    }
    cls = engines.get(algorithm.lower())
    if not cls:
        raise ValueError(f"Unknown consensus algorithm: {algorithm}. Choose from: {list(engines.keys())}")
    return cls(**kwargs)
