"""
Gas Metering System for Zexus VM

Provides gas cost tracking and enforcement for all VM operations.
Prevents DoS attacks through computational exhaustion.
"""

from typing import Dict, Optional
from enum import IntEnum


class GasCost(IntEnum):
    """Gas costs for different operation types"""
    # Basic operations
    NOP = 0
    LOAD_CONST = 1
    LOAD_NAME = 2
    STORE_NAME = 3
    POP = 1
    DUP = 1
    
    # Arithmetic operations
    ADD = 3
    SUB = 3
    MUL = 5
    DIV = 10
    MOD = 10
    POW = 20  # Expensive due to large numbers
    NEG = 2
    
    # Comparison operations
    EQ = 2
    NEQ = 2
    LT = 2
    GT = 2
    LTE = 2
    GTE = 2
    NOT = 2
    
    # Logical operations
    AND = 2
    OR = 2
    
    # Control flow
    JUMP = 2
    JUMP_IF_TRUE = 3
    JUMP_IF_FALSE = 3
    RETURN = 2
    
    # Collections
    BUILD_LIST = 5  # Base cost
    BUILD_MAP = 5   # Base cost
    INDEX = 3
    GET_LENGTH = 2
    GET_ATTR = 3
    
    # Function operations
    CALL_NAME = 10  # Base cost for function call
    CALL_TOP = 10
    STORE_FUNC = 5
    BUILD_LAMBDA = 5
    
    # Async operations
    SPAWN = 15
    AWAIT = 10
    
    # Blockchain operations (expensive)
    HASH_BLOCK = 50
    VERIFY_SIGNATURE = 100
    MERKLE_ROOT = 30  # Base cost, + per leaf
    STATE_READ = 20
    STATE_WRITE = 50
    TX_BEGIN = 20
    TX_COMMIT = 30
    TX_REVERT = 20
    LEDGER_APPEND = 40
    
    # I/O operations
    PRINT = 10


class GasMetering:
    """Gas metering for VM execution"""
    
    def __init__(self, gas_limit: Optional[int] = None, enable_timeout: bool = True):
        """
        Initialize gas metering
        
        Args:
            gas_limit: Maximum gas allowed (None = unlimited)
            enable_timeout: Enable execution timeout mechanism
        """
        self.gas_limit = gas_limit if gas_limit is not None else 1_000_000
        self.gas_used = 0
        self.enable_timeout = enable_timeout
        self.operation_count = 0
        self.max_operations = 100_000  # Safety limit for operations
        
        # Track gas usage by operation type for profiling
        self.gas_by_operation: Dict[str, int] = {}
        self.operation_counts: Dict[str, int] = {}
        
    def consume(self, operation: str, amount: Optional[int] = None, **kwargs) -> bool:
        """
        Consume gas for an operation
        
        Args:
            operation: Operation name (e.g., 'ADD', 'HASH_BLOCK')
            amount: Custom gas amount (overrides default cost)
            **kwargs: Additional parameters for dynamic cost calculation
        
        Returns:
            True if enough gas available, False if out of gas
        """
        # Calculate gas cost
        if amount is not None:
            cost = amount
        else:
            cost = self._get_operation_cost(operation, **kwargs)
        
        # Check if we have enough gas
        if self.gas_limit is not None and self.gas_used + cost > self.gas_limit:
            return False
        
        # Consume gas
        self.gas_used += cost
        self.operation_count += 1
        
        # Track for profiling
        self.gas_by_operation[operation] = self.gas_by_operation.get(operation, 0) + cost
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1
        
        # Check operation count limit (prevents infinite loops even with high gas)
        if self.operation_count > self.max_operations:
            return False
        
        return True
    
    def _get_operation_cost(self, operation: str, **kwargs) -> int:
        """
        Get gas cost for an operation
        
        Args:
            operation: Operation name
            **kwargs: Additional parameters for dynamic costs
        
        Returns:
            Gas cost for the operation
        """
        # Try to get cost from GasCost enum
        try:
            base_cost = GasCost[operation].value
        except KeyError:
            # Unknown operation - assign default cost
            base_cost = 5
        
        # Dynamic costs based on parameters
        if operation == "BUILD_LIST":
            # Cost scales with list size
            count = kwargs.get('count', 0)
            return base_cost + (count * 1)
        
        elif operation == "BUILD_MAP":
            # Cost scales with map size
            count = kwargs.get('count', 0)
            return base_cost + (count * 2)
        
        elif operation == "MERKLE_ROOT":
            # Cost scales with number of leaves
            leaf_count = kwargs.get('leaf_count', 0)
            return base_cost + (leaf_count * 5)
        
        elif operation in ("CALL_NAME", "CALL_TOP"):
            # Cost scales with number of arguments
            arg_count = kwargs.get('arg_count', 0)
            return base_cost + (arg_count * 2)
        
        return base_cost
    
    def check_limit(self) -> bool:
        """Check if gas limit has been exceeded"""
        if self.gas_limit is None:
            return True
        return self.gas_used <= self.gas_limit
    
    def remaining(self) -> int:
        """Get remaining gas"""
        if self.gas_limit is None:
            return float('inf')
        return max(0, self.gas_limit - self.gas_used)
    
    def reset(self):
        """Reset gas metering"""
        self.gas_used = 0
        self.operation_count = 0
        self.gas_by_operation.clear()
        self.operation_counts.clear()
    
    def get_stats(self) -> Dict:
        """Get gas usage statistics"""
        return {
            'gas_limit': self.gas_limit,
            'gas_used': self.gas_used,
            'gas_remaining': self.remaining(),
            'operation_count': self.operation_count,
            'max_operations': self.max_operations,
            'gas_by_operation': dict(self.gas_by_operation),
            'operation_counts': dict(self.operation_counts),
            'utilization_percent': (self.gas_used / self.gas_limit * 100) if self.gas_limit else 0
        }
    
    def set_limit(self, new_limit: int):
        """Update gas limit"""
        self.gas_limit = new_limit


class OutOfGasError(Exception):
    """Raised when gas limit is exceeded"""
    def __init__(self, gas_used: int, gas_limit: int, operation: str = "unknown"):
        self.gas_used = gas_used
        self.gas_limit = gas_limit
        self.operation = operation
        super().__init__(
            f"Out of gas: used {gas_used}/{gas_limit} during operation '{operation}'"
        )


class OperationLimitExceededError(Exception):
    """Raised when operation count limit is exceeded"""
    def __init__(self, operation_count: int, max_operations: int):
        self.operation_count = operation_count
        self.max_operations = max_operations
        super().__init__(
            f"Operation limit exceeded: {operation_count}/{max_operations}"
        )
