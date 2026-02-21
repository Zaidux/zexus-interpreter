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
from .storage import (
    StorageBackend, SQLiteBackend, LevelDBBackend, RocksDBBackend,
    MemoryBackend, get_storage_backend, register_backend,
)
from .network import P2PNetwork, Message, MessageType, PeerInfo, PeerReputationManager
from .consensus import (
    ConsensusEngine, ProofOfWork, ProofOfAuthority, ProofOfStake,
    BFTConsensus, BFTMessage, BFTPhase, BFTRoundState,
    create_consensus
)
from .node import BlockchainNode, NodeConfig

# JSON-RPC server (optional — requires aiohttp)
try:
    from .rpc import RPCServer, RPCError, RPCErrorCode, RPCMethodRegistry
    _RPC_AVAILABLE = True
except ImportError:
    _RPC_AVAILABLE = False

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

# Event indexing & log filtering
try:
    from .events import EventIndex, EventLog, LogFilter, BloomFilter
    _EVENTS_AVAILABLE = True
except ImportError:
    _EVENTS_AVAILABLE = False

# Merkle Patricia Trie
try:
    from .mpt import MerklePatriciaTrie, StateTrie, TrieNode, NodeType
    _MPT_AVAILABLE = True
except ImportError:
    _MPT_AVAILABLE = False

# HD Wallet & Keystore
try:
    from .wallet import (
        HDWallet, Account, Keystore, ExtendedKey,
        generate_mnemonic, validate_mnemonic, mnemonic_to_seed,
    )
    _WALLET_AVAILABLE = True
except ImportError:
    _WALLET_AVAILABLE = False

# Token Standards (ZX-20, ZX-721, ZX-1155)
try:
    from .tokens import (
        ZX20Token, ZX20Interface,
        ZX721Token, ZX721Interface,
        ZX1155Token, ZX1155Interface,
        TokenEvent, TransferEvent, ApprovalEvent,
        TransferSingleEvent, TransferBatchEvent, ApprovalForAllEvent,
    )
    _TOKENS_AVAILABLE = True
except ImportError:
    _TOKENS_AVAILABLE = False

# Upgradeable Contracts & Chain Governance
try:
    from .upgradeable import (
        ProxyContract,
        ImplementationRecord,
        UpgradeManager,
        ChainUpgradeGovernance,
        ChainUpgradeProposal,
        ProposalStatus,
        ProposalType,
        UpgradeEvent,
        UpgradeEventType,
    )
    _UPGRADEABLE_AVAILABLE = True
except ImportError:
    _UPGRADEABLE_AVAILABLE = False

# Formal Verification Engine
try:
    from .verification import (
        FormalVerifier,
        VerificationLevel,
        VerificationReport,
        VerificationFinding,
        Severity,
        FindingCategory,
        StructuralVerifier,
        InvariantVerifier,
        PropertyVerifier,
        TaintAnalyzer,
        AnnotationParser,
        Invariant,
        ContractProperty,
    )
    _VERIFICATION_AVAILABLE = True
except ImportError:
    _VERIFICATION_AVAILABLE = False

# Execution Accelerator
try:
    from .accelerator import (
        ExecutionAccelerator,
        AOTCompiler,
        InlineCache,
        NumericFastPath,
        WASMCache,
        BatchExecutor,
        TxBatchResult,
        CompiledAction,
    )
    _ACCELERATOR_AVAILABLE = True
except ImportError:
    _ACCELERATOR_AVAILABLE = False

# Production monitoring
try:
    from .monitoring import NodeMetrics, MetricsServer
    _MONITORING_AVAILABLE = True
except ImportError:
    _MONITORING_AVAILABLE = False

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

    # Storage Backends
    'StorageBackend',
    'SQLiteBackend',
    'LevelDBBackend',
    'RocksDBBackend',
    'MemoryBackend',
    'get_storage_backend',
    'register_backend',

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
    'BFTConsensus',
    'BFTMessage',
    'BFTPhase',
    'BFTRoundState',
    'create_consensus',

    # Node
    'BlockchainNode',
    'NodeConfig',

    # RPC Server
    'RPCServer',
    'RPCError',
    'RPCErrorCode',
    'RPCMethodRegistry',

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

    # Event Indexing
    'EventIndex',
    'EventLog',
    'LogFilter',
    'BloomFilter',

    # Merkle Patricia Trie
    'MerklePatriciaTrie',
    'StateTrie',
    'TrieNode',
    'NodeType',

    # HD Wallet & Keystore
    'HDWallet',
    'Account',
    'Keystore',
    'ExtendedKey',
    'generate_mnemonic',
    'validate_mnemonic',
    'mnemonic_to_seed',

    # Token Standards
    'ZX20Token',
    'ZX20Interface',
    'ZX721Token',
    'ZX721Interface',
    'ZX1155Token',
    'ZX1155Interface',
    'TokenEvent',
    'TransferEvent',
    'ApprovalEvent',
    'TransferSingleEvent',
    'TransferBatchEvent',
    'ApprovalForAllEvent',

    # Upgradeable Contracts & Chain Governance
    'ProxyContract',
    'ImplementationRecord',
    'UpgradeManager',
    'ChainUpgradeGovernance',
    'ChainUpgradeProposal',
    'ProposalStatus',
    'ProposalType',
    'UpgradeEvent',
    'UpgradeEventType',

    # Formal Verification
    'FormalVerifier',
    'VerificationLevel',
    'VerificationReport',
    'VerificationFinding',
    'Severity',
    'FindingCategory',
    'StructuralVerifier',
    'InvariantVerifier',
    'PropertyVerifier',
    'TaintAnalyzer',
    'AnnotationParser',
    'Invariant',
    'ContractProperty',

    # Execution Accelerator
    'ExecutionAccelerator',
    'AOTCompiler',
    'InlineCache',
    'NumericFastPath',
    'WASMCache',
    'BatchExecutor',
    'TxBatchResult',
    'CompiledAction',

    # Production Monitoring
    'NodeMetrics',
    'MetricsServer',
]
