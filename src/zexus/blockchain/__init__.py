"""
Zexus Blockchain Module

Complete blockchain and smart contract support for Zexus.

Features:
- Immutable ledger with versioning
- Transaction context (TX object)
- Gas tracking and execution limits
- Cryptographic primitives (hashing, signatures)
- Smart contract execution environment
"""

from .ledger import Ledger, LedgerManager, get_ledger_manager
from .transaction import (
    TransactionContext, GasTracker,
    create_tx_context, get_current_tx, end_tx_context,
    consume_gas, check_gas_and_consume
)
from .crypto import CryptoPlugin, register_crypto_builtins

__all__ = [
    # Ledger
    'Ledger',
    'LedgerManager',
    'get_ledger_manager',
    
    # Transaction
    'TransactionContext',
    'GasTracker',
    'create_tx_context',
    'get_current_tx',
    'end_tx_context',
    'consume_gas',
    'check_gas_and_consume',
    
    # Crypto
    'CryptoPlugin',
    'register_crypto_builtins',
]
