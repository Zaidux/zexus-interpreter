"""
Zexus Blockchain Transaction Context

Implements the TX object and gas tracking for smart contracts.
"""

import time
import hashlib
import secrets
from typing import Optional, Dict, Any


class TransactionContext:
    """
    Immutable transaction context (TX object)
    
    Provides mandatory information about the current execution environment:
    - TX.caller: Address/ID of the entity executing the code
    - TX.timestamp: Canonical, un-tamperable time of execution
    - TX.block_hash: Cryptographic reference to the preceding state
    - TX.gas_remaining: Remaining gas for execution
    - TX.gas_used: Gas consumed so far
    """
    
    def __init__(self, caller: str, timestamp: Optional[float] = None, 
                 block_hash: Optional[str] = None, gas_limit: Optional[int] = None):
        # Immutable properties
        self._caller = caller
        self._timestamp = timestamp if timestamp is not None else time.time()
        self._block_hash = block_hash if block_hash else self._generate_block_hash()
        self._gas_limit = gas_limit if gas_limit is not None else 1_000_000
        self._gas_used = 0
        self._reverted = False
        self._revert_reason = None
        
    @property
    def caller(self) -> str:
        """The address/ID of the entity executing the code"""
        return self._caller
    
    @property
    def timestamp(self) -> float:
        """The canonical, un-tamperable time of execution"""
        return self._timestamp
    
    @property
    def block_hash(self) -> str:
        """Cryptographic reference to the preceding state"""
        return self._block_hash
    
    @property
    def gas_limit(self) -> int:
        """Maximum gas allowed for this transaction"""
        return self._gas_limit
    
    @gas_limit.setter
    def gas_limit(self, value: int):
        """Set gas limit (can be changed during execution via LIMIT statement)"""
        if value < 0:
            raise ValueError("Gas limit cannot be negative")
        self._gas_limit = value
    
    @property
    def gas_used(self) -> int:
        """Gas consumed so far"""
        return self._gas_used
    
    @property
    def gas_remaining(self) -> int:
        """Remaining gas for execution"""
        return max(0, self._gas_limit - self._gas_used)
    
    @property
    def reverted(self) -> bool:
        """Whether this transaction has been reverted"""
        return self._reverted
    
    @property
    def revert_reason(self) -> Optional[str]:
        """Reason for revert (if reverted)"""
        return self._revert_reason
    
    def consume_gas(self, amount: int) -> bool:
        """
        Consume gas for an operation
        
        Returns:
            True if enough gas available, False otherwise
        """
        if self._gas_used + amount > self._gas_limit:
            return False
        self._gas_used += amount
        return True
    
    def revert(self, reason: Optional[str] = None):
        """Mark transaction as reverted"""
        self._reverted = True
        self._revert_reason = reason
    
    def _generate_block_hash(self) -> str:
        """Generate a pseudo-random block hash"""
        data = f"{self._caller}{self._timestamp}{secrets.token_hex(16)}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TX context to dictionary"""
        return {
            'caller': self.caller,
            'timestamp': self.timestamp,
            'block_hash': self.block_hash,
            'gas_limit': self.gas_limit,
            'gas_used': self.gas_used,
            'gas_remaining': self.gas_remaining,
            'reverted': self.reverted,
            'revert_reason': self.revert_reason
        }
    
    def __repr__(self):
        return f"TX(caller={self.caller}, timestamp={self.timestamp}, gas={self.gas_remaining}/{self.gas_limit})"


class GasTracker:
    """
    Gas consumption tracker for operations
    
    Defines gas costs for different operations to prevent DoS attacks.
    """
    
    # Base gas costs (inspired by EVM gas model)
    BASE_COSTS = {
        # Storage operations
        'ledger_write': 20_000,
        'ledger_read': 200,
        'state_write': 5_000,
        'state_read': 200,
        
        # Computation
        'add': 3,
        'sub': 3,
        'mul': 5,
        'div': 5,
        'mod': 5,
        'compare': 3,
        
        # Control flow
        'if': 10,
        'loop': 10,
        'function_call': 100,
        'return': 10,
        
        # Cryptography
        'hash_sha256': 60,
        'hash_keccak256': 30,
        'signature_create': 3_000,
        'signature_verify': 3_000,
        
        # Memory
        'memory_read': 3,
        'memory_write': 3,
        'memory_allocate': 100,
        
        # Base operation
        'base': 21_000,
    }
    
    @classmethod
    def get_cost(cls, operation: str, **kwargs) -> int:
        """
        Get gas cost for an operation
        
        Args:
            operation: Operation name
            **kwargs: Additional parameters (e.g., data_size for hashing)
            
        Returns:
            Gas cost
        """
        base_cost = cls.BASE_COSTS.get(operation, 1)
        
        # Adjust for data size if applicable
        if 'data_size' in kwargs:
            # Additional cost per 32 bytes
            size = kwargs['data_size']
            word_cost = (size + 31) // 32
            base_cost += word_cost * 3
        
        return base_cost
    
    @classmethod
    def estimate_limit(cls, action_name: str) -> int:
        """
        Estimate reasonable gas limit for an action
        
        Returns:
            Suggested gas limit
        """
        # Default limits for common action types
        limits = {
            'transfer': 50_000,
            'mint': 100_000,
            'burn': 50_000,
            'approve': 50_000,
            'swap': 150_000,
            'stake': 100_000,
            'unstake': 100_000,
        }
        
        return limits.get(action_name, 1_000_000)


# Transaction context stack (for nested calls)
_tx_stack = []


def create_tx_context(caller: str, gas_limit: Optional[int] = None,
                      timestamp: Optional[float] = None, 
                      block_hash: Optional[str] = None) -> TransactionContext:
    """Create a new transaction context"""
    tx = TransactionContext(
        caller=caller,
        timestamp=timestamp,
        block_hash=block_hash,
        gas_limit=gas_limit
    )
    _tx_stack.append(tx)
    return tx


def get_current_tx() -> Optional[TransactionContext]:
    """Get the current transaction context"""
    return _tx_stack[-1] if _tx_stack else None


def end_tx_context():
    """End the current transaction context"""
    if _tx_stack:
        _tx_stack.pop()


def consume_gas(amount: int, operation: str = "unknown") -> bool:
    """
    Consume gas from current transaction
    
    Returns:
        True if enough gas available, False otherwise
    """
    tx = get_current_tx()
    if not tx:
        return True  # No transaction context = no gas tracking
    
    if not tx.consume_gas(amount):
        # Out of gas!
        tx.revert(f"Out of gas during operation: {operation}")
        return False
    
    return True


def check_gas_and_consume(operation: str, **kwargs) -> bool:
    """
    Check if enough gas and consume it
    
    Returns:
        True if successful, False if out of gas
    """
    cost = GasTracker.get_cost(operation, **kwargs)
    return consume_gas(cost, operation)
