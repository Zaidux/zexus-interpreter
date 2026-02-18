"""
Zexus Blockchain Module

Complete blockchain and smart contract support for Zexus.

Features:
- Immutable ledger with versioning
- Transaction context (TX object)
- Gas tracking and execution limits
- Cryptographic primitives (hashing, signatures)
- Smart contract execution environment
- Real blockchain: chain, blocks, mempool
- P2P networking with peer discovery
- Pluggable consensus (PoW, PoA, PoS)
- Full node with mining and sync
"""

from .ledger import Ledger, LedgerManager, get_ledger_manager
from .transaction import (
    TransactionContext, GasTracker,
    create_tx_context, get_current_tx, end_tx_context,
    consume_gas, check_gas_and_consume
)
from .crypto import CryptoPlugin, register_crypto_builtins

# Real blockchain infrastructure
from .chain import (
    Block, BlockHeader, Transaction as ChainTransaction,
    TransactionReceipt, Chain, Mempool
)
from .network import P2PNetwork, Message, MessageType, PeerInfo, PeerReputationManager
from .consensus import (
    ConsensusEngine, ProofOfWork, ProofOfAuthority, ProofOfStake,
    create_consensus
)
from .node import BlockchainNode, NodeConfig

# Contract VM bridge (connects VM opcodes to real chain state)
try:
    from .contract_vm import ContractVM, ContractExecutionReceipt, ContractStateAdapter
    _CONTRACT_VM_AVAILABLE = True
except ImportError:
    _CONTRACT_VM_AVAILABLE = False

# Multichain / cross-chain bridge infrastructure
try:
    from .multichain import (
        ChainRouter,
        BridgeRelay,
        BridgeContract,
        CrossChainMessage,
        MerkleProofEngine,
        MessageStatus,
    )
    _MULTICHAIN_AVAILABLE = True
except ImportError:
    _MULTICHAIN_AVAILABLE = False

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

    # Chain & Blocks
    'Block',
    'BlockHeader',
    'ChainTransaction',
    'TransactionReceipt',
    'Chain',
    'Mempool',

    # P2P Network
    'P2PNetwork',
    'Message',
    'MessageType',
    'PeerInfo',

    # Consensus
    'ConsensusEngine',
    'ProofOfWork',
    'ProofOfAuthority',
    'ProofOfStake',
    'create_consensus',

    # Node
    'BlockchainNode',
    'NodeConfig',

    # Contract VM
    'ContractVM',
    'ContractExecutionReceipt',
    'ContractStateAdapter',

    # Multichain / Bridge
    'ChainRouter',
    'BridgeRelay',
    'BridgeContract',
    'CrossChainMessage',
    'MerkleProofEngine',
    'MessageStatus',
]
