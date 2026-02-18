"""
Zexus Blockchain — Node

The Node is the top-level integration point that ties together:
  - Chain (block storage + state)
  - Mempool (pending transactions)
  - P2P Network (peer connectivity)
  - Consensus Engine (block production + validation)

It provides the public API used by the Zexus evaluator, CLI, and RPC.
"""

import asyncio
import json
import hashlib
import time
import threading
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from .chain import Block, BlockHeader, Chain, Mempool, Transaction, TransactionReceipt
from .network import P2PNetwork, Message, MessageType, PeerConnection, PeerReputationManager
from .consensus import ConsensusEngine, ProofOfWork, create_consensus

# Multichain bridge (optional)
try:
    from .multichain import ChainRouter, BridgeContract, CrossChainMessage
    _MULTICHAIN_AVAILABLE = True
except ImportError:
    _MULTICHAIN_AVAILABLE = False
    ChainRouter = None  # type: ignore
    BridgeContract = None  # type: ignore

# Contract VM bridge (optional — only if the VM module is present)
try:
    from .contract_vm import ContractVM, ContractExecutionReceipt
    _CONTRACT_VM_AVAILABLE = True
except ImportError:
    _CONTRACT_VM_AVAILABLE = False
    ContractVM = None  # type: ignore
    ContractExecutionReceipt = None  # type: ignore

logger = logging.getLogger("zexus.blockchain.node")


class NodeConfig:
    """Configuration for a blockchain node."""

    def __init__(self, **kwargs):
        self.chain_id: str = kwargs.get("chain_id", "zexus-mainnet")
        self.host: str = kwargs.get("host", "0.0.0.0")
        self.port: int = kwargs.get("port", 30303)
        self.data_dir: Optional[str] = kwargs.get("data_dir", None)
        self.consensus: str = kwargs.get("consensus", "pow")
        self.consensus_params: Dict[str, Any] = kwargs.get("consensus_params", {})
        self.miner_address: str = kwargs.get("miner_address", "")
        self.mining_enabled: bool = kwargs.get("mining_enabled", False)
        self.mining_interval: float = kwargs.get("mining_interval", 5.0)
        self.max_peers: int = kwargs.get("max_peers", 25)
        self.bootstrap_nodes: List[Dict[str, Any]] = kwargs.get("bootstrap_nodes", [])
        self.initial_balances: Dict[str, int] = kwargs.get("initial_balances", {})
        self.rpc_enabled: bool = kwargs.get("rpc_enabled", False)
        self.rpc_port: int = kwargs.get("rpc_port", 8545)


class BlockchainNode:
    """Full blockchain node.
    
    Coordinates chain storage, transaction processing, P2P networking,
    consensus, and mining into a single cohesive system.
    """

    def __init__(self, config: Optional[NodeConfig] = None):
        self.config = config or NodeConfig()
        
        # Core components
        self.chain = Chain(
            chain_id=self.config.chain_id,
            data_dir=self.config.data_dir,
        )
        self.mempool = Mempool()
        self.consensus_engine: ConsensusEngine = create_consensus(
            self.config.consensus,
            **self.config.consensus_params,
        )
        self.network = P2PNetwork(
            host=self.config.host,
            port=self.config.port,
            chain_id=self.config.chain_id,
            max_peers=self.config.max_peers,
        )

        # Mining state
        self._mining = False
        self._mining_task: Optional[asyncio.Task] = None

        # Contract VM bridge — enables smart-contract execution via
        # the Zexus VM with real chain-backed state.
        self.contract_vm: Optional["ContractVM"] = None
        if _CONTRACT_VM_AVAILABLE:
            self.contract_vm = ContractVM(
                chain=self.chain,
                gas_limit=self.config.consensus_params.get("gas_limit", 10_000_000),
            )

        # Event listeners
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Node lifecycle
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bg_thread: Optional[threading.Thread] = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self):
        """Start the node: initialize chain, start networking, begin mining."""
        if self._running:
            return

        self._loop = asyncio.get_event_loop()
        self._running = True

        # Initialize chain with genesis if empty
        if not self.chain.blocks:
            self.chain.create_genesis(
                miner=self.config.miner_address or "0x" + "0" * 40,
                initial_balances=self.config.initial_balances or None,
            )
            logger.info("Genesis block created for chain '%s'", self.config.chain_id)

        # Set up network handlers
        self._register_network_handlers()

        # Start P2P network
        for bn in self.config.bootstrap_nodes:
            self.network.add_bootstrap_node(bn.get("host", ""), bn.get("port", 0))
        await self.network.start()

        # Start mining if enabled
        if self.config.mining_enabled and self.config.miner_address:
            self.start_mining()

        logger.info("Node started: chain=%s height=%d peers=%d",
                     self.config.chain_id, self.chain.height, self.network.peer_count)

    async def stop(self):
        """Stop the node gracefully."""
        self._running = False
        self.stop_mining()
        await self.network.stop()
        self.chain.close()
        logger.info("Node stopped")

    def start_sync(self):
        """Start the node in a background thread (for non-async callers)."""
        if self._bg_thread and self._bg_thread.is_alive():
            return

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_until_complete(self.start())
            loop.run_forever()

        self._bg_thread = threading.Thread(target=_run, daemon=True)
        self._bg_thread.start()

    def stop_sync(self):
        """Stop the node from the synchronous world."""
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(self.stop(), self._loop)

    # ── Events ─────────────────────────────────────────────────────────

    def on(self, event: str, handler: Callable):
        """Register an event handler. Events: new_block, new_tx, mined, sync."""
        self._event_handlers.setdefault(event, []).append(handler)

    def _emit(self, event: str, data: Any = None):
        for handler in self._event_handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                logger.error("Event handler error (%s): %s", event, e)

    # ── Transaction API ────────────────────────────────────────────────

    def submit_transaction(self, tx: Transaction) -> Dict[str, Any]:
        """Submit a transaction to the node.
        
        Validates, adds to mempool, and broadcasts to peers.
        
        Returns:
            {"success": bool, "tx_hash": str, "error": str}
        """
        # Ensure hash
        if not tx.tx_hash:
            tx.compute_hash()

        # Basic validation
        if not tx.sender:
            return {"success": False, "tx_hash": "", "error": "missing sender"}

        sender_acct = self.chain.get_account(tx.sender)
        cost = tx.value + (tx.gas_limit * tx.gas_price)
        if sender_acct["balance"] < cost:
            return {"success": False, "tx_hash": tx.tx_hash,
                    "error": f"insufficient balance: need {cost}, have {sender_acct['balance']}"}

        if tx.nonce < sender_acct["nonce"]:
            return {"success": False, "tx_hash": tx.tx_hash,
                    "error": f"nonce too low: expected >= {sender_acct['nonce']}, got {tx.nonce}"}

        # Add to mempool
        if not self.mempool.add(tx):
            return {"success": False, "tx_hash": tx.tx_hash, "error": "mempool rejected"}

        # Broadcast to peers
        if self._loop and self.network.is_running:
            asyncio.run_coroutine_threadsafe(
                self.network.gossip(Message(
                    type=MessageType.NEW_TX,
                    payload=tx.to_dict(),
                )),
                self._loop,
            )

        self._emit("new_tx", tx)
        return {"success": True, "tx_hash": tx.tx_hash, "error": ""}

    def create_transaction(self, sender: str, recipient: str, value: int,
                           data: str = "", gas_limit: int = 21_000,
                           gas_price: int = 1) -> Transaction:
        """Convenience: create and submit a transaction."""
        acct = self.chain.get_account(sender)
        tx = Transaction(
            sender=sender,
            recipient=recipient,
            value=value,
            data=data,
            nonce=acct["nonce"],
            gas_limit=gas_limit,
            gas_price=gas_price,
            timestamp=time.time(),
        )
        tx.compute_hash()
        return tx

    # ── Block API ──────────────────────────────────────────────────────

    def get_block(self, hash_or_height) -> Optional[Dict[str, Any]]:
        """Get a block by hash or height."""
        block = self.chain.get_block(hash_or_height)
        return block.to_dict() if block else None

    def get_latest_block(self) -> Optional[Dict[str, Any]]:
        """Get the latest block."""
        tip = self.chain.tip
        return tip.to_dict() if tip else None

    def get_chain_info(self) -> Dict[str, Any]:
        """Get combined chain + network info."""
        info = self.chain.get_chain_info()
        info["network"] = self.network.get_network_info()
        info["mempool_size"] = self.mempool.size
        info["mining"] = self._mining
        info["consensus"] = self.config.consensus
        return info

    # ── Account API ────────────────────────────────────────────────────

    def get_balance(self, address: str) -> int:
        return self.chain.get_account(address)["balance"]

    def get_nonce(self, address: str) -> int:
        return self.chain.get_account(address)["nonce"]

    def get_account(self, address: str) -> Dict[str, Any]:
        return self.chain.get_account(address)

    def fund_account(self, address: str, amount: int):
        """Fund an account (for testnets / development)."""
        acct = self.chain.get_account(address)
        acct["balance"] += amount

    # ── Smart Contract API ─────────────────────────────────────────────

    def deploy_contract(self, contract, deployer: str,
                        gas_limit: int = 10_000_000,
                        initial_value: int = 0) -> Dict[str, Any]:
        """Deploy a SmartContract onto the chain via the ContractVM.

        Returns:
            {"success": bool, "address": str, "error": str}
        """
        if self.contract_vm is None:
            return {"success": False, "address": "", "error": "ContractVM not available"}

        receipt = self.contract_vm.deploy_contract(
            contract=contract,
            deployer=deployer,
            gas_limit=gas_limit,
            initial_value=initial_value,
        )
        return {
            "success": receipt.success,
            "address": contract.address,
            "error": receipt.error,
            "gas_used": receipt.gas_used,
        }

    def call_contract(self, contract_address: str, action: str,
                      args: Optional[Dict[str, Any]] = None,
                      caller: str = "",
                      gas_limit: int = 500_000,
                      value: int = 0) -> Dict[str, Any]:
        """Execute a contract action (state-mutating).

        Returns:
            {"success": bool, "return_value": Any, "gas_used": int, ...}
        """
        if self.contract_vm is None:
            return {"success": False, "error": "ContractVM not available"}

        receipt = self.contract_vm.execute_contract(
            contract_address=contract_address,
            action=action,
            args=args,
            caller=caller,
            gas_limit=gas_limit,
            value=value,
        )
        return receipt.to_dict()

    def static_call_contract(self, contract_address: str, action: str,
                             args: Optional[Dict[str, Any]] = None,
                             caller: str = "") -> Dict[str, Any]:
        """Execute a read-only contract call (no state changes committed).

        Returns:
            {"success": bool, "return_value": Any, "gas_used": int, ...}
        """
        if self.contract_vm is None:
            return {"success": False, "error": "ContractVM not available"}

        receipt = self.contract_vm.static_call(
            contract_address=contract_address,
            action=action,
            args=args,
            caller=caller,
        )
        return receipt.to_dict()

    # ── Mining ─────────────────────────────────────────────────────────

    def start_mining(self):
        """Start the mining loop."""
        if self._mining:
            return
        if not self.config.miner_address:
            raise RuntimeError("Cannot mine without a miner_address")
        self._mining = True
        if self._loop:
            self._mining_task = asyncio.run_coroutine_threadsafe(
                self._mining_loop(), self._loop
            )
        logger.info("Mining started (miner=%s)", self.config.miner_address[:8])

    def stop_mining(self):
        """Stop the mining loop."""
        self._mining = False
        if self._mining_task:
            self._mining_task.cancel()
            self._mining_task = None
        logger.info("Mining stopped")

    async def _mining_loop(self):
        """Continuous mining loop."""
        while self._mining and self._running:
            try:
                block = await asyncio.get_event_loop().run_in_executor(
                    None, self._mine_one_block
                )
                if block:
                    success, err = self.chain.add_block(block)
                    if success:
                        logger.info("Mined block %d: %s", block.header.height, block.hash[:16])
                        self._emit("mined", block)
                        # Broadcast new block
                        await self.network.gossip(Message(
                            type=MessageType.NEW_BLOCK,
                            payload=block.to_dict(),
                        ))
                    else:
                        logger.warning("Mined block rejected: %s", err)
                else:
                    await asyncio.sleep(self.config.mining_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Mining error: %s", e)
                await asyncio.sleep(1)

    def _mine_one_block(self) -> Optional[Block]:
        """Attempt to mine a single block."""
        if self.mempool.size == 0:
            return None
        return self.consensus_engine.seal(
            self.chain, self.mempool, self.config.miner_address
        )

    def mine_block_sync(self) -> Optional[Block]:
        """Mine a single block synchronously (for testing/scripting)."""
        block = self.consensus_engine.seal(
            self.chain, self.mempool, self.config.miner_address
        )
        if block:
            success, err = self.chain.add_block(block)
            if success:
                self._emit("mined", block)
                return block
            else:
                logger.warning("Block rejected: %s", err)
        return None

    # ── Chain Sync (Network) ───────────────────────────────────────────

    def _register_network_handlers(self):
        """Set up P2P message handlers for blockchain protocol."""
        self.network.on(MessageType.NEW_BLOCK, self._on_new_block)
        self.network.on(MessageType.NEW_TX, self._on_new_tx)
        self.network.on(MessageType.GET_BLOCKS, self._on_get_blocks)
        self.network.on(MessageType.BLOCKS, self._on_blocks)
        self.network.on(MessageType.GET_HEADERS, self._on_get_headers)
        self.network.on(MessageType.HEADERS, self._on_headers)

    async def _on_new_block(self, msg: Message, conn: PeerConnection):
        """Handle a NEW_BLOCK message from a peer."""
        try:
            block = Block.from_dict(msg.payload)
        except Exception as e:
            logger.debug("Invalid block from %s: %s", msg.sender[:8], e)
            return

        # Skip if we already have this block
        if block.hash in self.chain.block_index:
            return

        # Validate via consensus
        if not self.consensus_engine.verify(block, self.chain):
            logger.debug("Block %d from %s failed consensus", block.header.height, msg.sender[:8])
            # Penalize peer for invalid block
            if conn.peer_info.peer_id:
                self.network.reputation.update(
                    conn.peer_info, PeerReputationManager.INVALID_BLOCK,
                    reason="failed consensus verification")
            return

        # Try to add to chain
        success, err = self.chain.add_block(block)
        if success:
            logger.info("Accepted block %d from peer %s", block.header.height, msg.sender[:8])
            self._emit("new_block", block)
            # Reward peer for valid block
            if conn.peer_info.peer_id:
                self.network.reputation.update(
                    conn.peer_info, PeerReputationManager.VALID_BLOCK,
                    reason="valid block accepted")
            # Remove block's txs from our mempool
            for tx in block.transactions:
                self.mempool.remove(tx.tx_hash)
            # Relay to other peers
            await self.network.gossip(msg, exclude={msg.sender})
        else:
            logger.debug("Block %d rejected: %s", block.header.height, err)
            # If block is ahead of us, we might need to sync
            if block.header.height > self.chain.height + 1:
                await self._request_sync(conn)

    async def _on_new_tx(self, msg: Message, conn: PeerConnection):
        """Handle a NEW_TX message from a peer."""
        try:
            tx = Transaction(**msg.payload)
        except Exception as e:
            logger.debug("Invalid TX from %s: %s", msg.sender[:8], e)
            return

        if self.mempool.add(tx):
            self._emit("new_tx", tx)
            # Reward peer for valid transaction
            if conn.peer_info.peer_id:
                self.network.reputation.update(
                    conn.peer_info, PeerReputationManager.VALID_TX,
                    reason="valid transaction")
            # Relay
            await self.network.gossip(msg, exclude={msg.sender})

    async def _on_get_blocks(self, msg: Message, conn: PeerConnection):
        """Respond to a GET_BLOCKS request."""
        start_height = msg.payload.get("start", 0)
        count = min(msg.payload.get("count", 50), 100)  # Max 100 blocks per request

        blocks_data = []
        for h in range(start_height, min(start_height + count, self.chain.height + 1)):
            block = self.chain.get_block(h)
            if block:
                blocks_data.append(block.to_dict())

        await conn.send(Message(
            type=MessageType.BLOCKS,
            payload={"blocks": blocks_data},
        ))

    async def _on_blocks(self, msg: Message, conn: PeerConnection):
        """Handle received blocks (sync response)."""
        blocks_data = msg.payload.get("blocks", [])
        added = 0
        for bd in blocks_data:
            try:
                block = Block.from_dict(bd)
                if block.hash not in self.chain.block_index:
                    if self.consensus_engine.verify(block, self.chain):
                        success, _ = self.chain.add_block(block)
                        if success:
                            added += 1
            except Exception as e:
                logger.debug("Error processing sync block: %s", e)
                continue

        if added > 0:
            logger.info("Synced %d blocks from peer %s (height now %d)",
                        added, msg.sender[:8], self.chain.height)

    async def _on_get_headers(self, msg: Message, conn: PeerConnection):
        """Respond to header requests."""
        start = msg.payload.get("start", 0)
        count = min(msg.payload.get("count", 100), 500)
        
        headers = []
        for h in range(start, min(start + count, self.chain.height + 1)):
            block = self.chain.get_block(h)
            if block:
                from dataclasses import asdict
                headers.append(asdict(block.header))
        
        await conn.send(Message(
            type=MessageType.HEADERS,
            payload={"headers": headers},
        ))

    async def _on_headers(self, msg: Message, conn: PeerConnection):
        """Handle received headers — decide if we need full blocks."""
        headers = msg.payload.get("headers", [])
        if headers:
            last = headers[-1]
            remote_height = last.get("height", 0)
            if remote_height > self.chain.height:
                # Request missing blocks
                await conn.send(Message(
                    type=MessageType.GET_BLOCKS,
                    payload={"start": self.chain.height + 1, "count": 50},
                ))

    async def _request_sync(self, conn: PeerConnection):
        """Request blocks from a peer to catch up."""
        await conn.send(Message(
            type=MessageType.GET_BLOCKS,
            payload={"start": self.chain.height + 1, "count": 50},
        ))

    # ── Utility ────────────────────────────────────────────────────────

    def validate_chain(self) -> Dict[str, Any]:
        """Full chain integrity check."""
        valid, err = self.chain.validate_chain()
        return {"valid": valid, "error": err, "height": self.chain.height}

    def export_chain(self) -> List[Dict[str, Any]]:
        """Export the full chain as a list of block dicts."""
        return [b.to_dict() for b in self.chain.blocks]

    def __repr__(self):
        return (f"BlockchainNode(chain_id={self.config.chain_id!r}, "
                f"height={self.chain.height}, peers={self.network.peer_count})")

    # ── Multichain / Cross-chain Bridge ────────────────────────────────

    def join_router(self, router: "ChainRouter") -> None:
        """Register this node's chain with an external ChainRouter."""
        if not _MULTICHAIN_AVAILABLE:
            raise RuntimeError("Multichain module not available")
        router.register_chain(self.config.chain_id, self.chain)
        self._router = router
        logger.info("Node %s joined ChainRouter", self.config.chain_id)

    def bridge_to(
        self,
        other_node: "BlockchainNode",
        router: Optional["ChainRouter"] = None,
    ) -> "BridgeContract":
        """Create a bridge contract between this node and *other_node*.

        If no *router* is provided, a new one is created.

        Returns a ``BridgeContract`` for lock-and-mint / burn-and-release.
        """
        if not _MULTICHAIN_AVAILABLE:
            raise RuntimeError("Multichain module not available")
        if router is None:
            router = ChainRouter()
        if self.config.chain_id not in router.chain_ids:
            self.join_router(router)
        if other_node.config.chain_id not in router.chain_ids:
            other_node.join_router(router)
        router.connect(self.config.chain_id, other_node.config.chain_id)
        bridge = BridgeContract(
            router=router,
            source_chain=self.config.chain_id,
            dest_chain=other_node.config.chain_id,
        )
        logger.info(
            "Bridge created: %s <-> %s",
            self.config.chain_id, other_node.config.chain_id,
        )
        return bridge
