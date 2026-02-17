"""
Zexus Blockchain — Peer-to-Peer Networking Layer

Provides peer discovery, connection management, and message propagation
for the Zexus blockchain network.  Built on top of the existing asyncio
TCP/WebSocket infrastructure in ``stdlib.sockets``.

Protocol messages are JSON-encoded with a type field and optional payload.
"""

import asyncio
import hashlib
import json
import random
import threading
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("zexus.blockchain.network")


# ── Message types ──────────────────────────────────────────────────────────

class MessageType:
    """Protocol message types."""
    # Discovery
    PING = "ping"
    PONG = "pong"
    FIND_PEERS = "find_peers"
    PEERS = "peers"
    # Chain sync
    GET_BLOCKS = "get_blocks"
    BLOCKS = "blocks"
    GET_HEADERS = "get_headers"
    HEADERS = "headers"
    NEW_BLOCK = "new_block"
    # Transactions
    NEW_TX = "new_tx"
    GET_TX = "get_tx"
    TX = "tx"
    # Consensus
    VOTE = "vote"
    PROPOSE = "propose"
    # General
    HANDSHAKE = "handshake"
    DISCONNECT = "disconnect"
    ERROR = "error"


@dataclass
class Message:
    """Network protocol message."""
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    sender: str = ""
    nonce: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        self.timestamp = self.timestamp or time.time()
        self.nonce = self.nonce or hashlib.sha256(
            f"{self.type}{self.timestamp}{random.random()}".encode()
        ).hexdigest()[:16]

    def encode(self) -> bytes:
        """Serialize to JSON bytes."""
        return json.dumps(asdict(self)).encode("utf-8")

    @staticmethod
    def decode(data: bytes) -> 'Message':
        """Deserialize from JSON bytes."""
        d = json.loads(data.decode("utf-8"))
        return Message(**d)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PeerInfo:
    """Information about a network peer."""
    peer_id: str = ""
    host: str = ""
    port: int = 0
    chain_id: str = ""
    height: int = 0
    version: str = "1.0.0"
    last_seen: float = 0.0
    latency_ms: float = 0.0
    reputation: int = 100  # 0-100 score
    connected: bool = False

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class PeerConnection:
    """Manages a single peer connection with send/receive capabilities."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 peer_info: PeerInfo, is_inbound: bool = False):
        self.reader = reader
        self.writer = writer
        self.peer_info = peer_info
        self.is_inbound = is_inbound
        self._closed = False
        self._recv_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        return not self._closed and self.writer is not None

    async def send(self, msg: Message) -> bool:
        """Send a message to this peer."""
        if self._closed:
            return False
        try:
            data = msg.encode()
            length = len(data)
            self.writer.write(length.to_bytes(4, "big") + data)
            await self.writer.drain()
            return True
        except (ConnectionError, OSError, asyncio.CancelledError):
            self._closed = True
            return False

    async def receive(self) -> Optional[Message]:
        """Receive a single message from this peer."""
        if self._closed:
            return None
        try:
            length_bytes = await self.reader.readexactly(4)
            length = int.from_bytes(length_bytes, "big")
            if length > 10 * 1024 * 1024:  # 10 MB max message size
                logger.warning("Message too large from %s: %d bytes", self.peer_info.address, length)
                self._closed = True
                return None
            data = await self.reader.readexactly(length)
            return Message.decode(data)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.CancelledError):
            self._closed = True
            return None

    async def close(self):
        """Close the connection."""
        self._closed = True
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass


class P2PNetwork:
    """Peer-to-peer network manager for the Zexus blockchain.
    
    Handles:
    - Listening for inbound connections
    - Connecting to peers
    - Peer discovery and management
    - Message routing and broadcasting
    - Seen-message deduplication
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 30303,
                 chain_id: str = "zexus-mainnet", node_id: str = "",
                 max_peers: int = 25, min_peers: int = 3):
        self.host = host
        self.port = port
        self.chain_id = chain_id
        self.node_id = node_id or hashlib.sha256(
            f"{host}:{port}:{time.time()}:{random.random()}".encode()
        ).hexdigest()[:40]
        self.max_peers = max_peers
        self.min_peers = min_peers

        # Connection state
        self.peers: Dict[str, PeerConnection] = {}  # peer_id -> connection
        self.known_peers: Dict[str, PeerInfo] = {}  # peer_id -> info (includes disconnected)
        self.bootstrap_nodes: List[Tuple[str, int]] = []

        # Message handling
        self._handlers: Dict[str, List[Callable]] = {}
        self._seen_messages: Set[str] = set()
        self._seen_max = 10_000

        # Server state
        self._server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._recv_tasks: Dict[str, asyncio.Task] = {}

    # ── Event handler registration ─────────────────────────────────────

    def on(self, msg_type: str, handler: Callable):
        """Register a handler for a message type."""
        self._handlers.setdefault(msg_type, []).append(handler)

    def off(self, msg_type: str, handler: Optional[Callable] = None):
        """Remove handler(s) for a message type."""
        if handler is None:
            self._handlers.pop(msg_type, None)
        elif msg_type in self._handlers:
            self._handlers[msg_type] = [h for h in self._handlers[msg_type] if h != handler]

    async def _dispatch(self, msg: Message, conn: PeerConnection):
        """Dispatch a received message to registered handlers."""
        handlers = self._handlers.get(msg.type, [])
        for handler in handlers:
            try:
                result = handler(msg, conn)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Handler error for %s: %s", msg.type, e)

    # ── Server lifecycle ───────────────────────────────────────────────

    async def start(self):
        """Start listening for inbound connections."""
        if self._running:
            return
        self._loop = asyncio.get_event_loop()
        self._server = await asyncio.start_server(
            self._handle_inbound, self.host, self.port
        )
        self._running = True
        logger.info("P2P listening on %s:%d (node_id=%s)", self.host, self.port, self.node_id[:8])

        # Register built-in handlers
        self.on(MessageType.PING, self._handle_ping)
        self.on(MessageType.FIND_PEERS, self._handle_find_peers)
        self.on(MessageType.HANDSHAKE, self._handle_handshake_msg)

        # Bootstrap connections
        asyncio.ensure_future(self._bootstrap())

    async def stop(self):
        """Stop the P2P network."""
        self._running = False
        # Close all peer connections
        for peer_id, conn in list(self.peers.items()):
            await conn.send(Message(type=MessageType.DISCONNECT, sender=self.node_id))
            await conn.close()
        self.peers.clear()

        # Cancel receive tasks
        for task in self._recv_tasks.values():
            task.cancel()
        self._recv_tasks.clear()

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("P2P network stopped")

    @property
    def peer_count(self) -> int:
        return len(self.peers)

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Connections ────────────────────────────────────────────────────

    async def _handle_inbound(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a new inbound TCP connection."""
        addr = writer.get_extra_info("peername")
        logger.debug("Inbound connection from %s", addr)

        if len(self.peers) >= self.max_peers:
            writer.close()
            return

        # Create temporary peer info, will be updated on handshake
        temp_info = PeerInfo(host=addr[0] if addr else "unknown", port=addr[1] if addr else 0)
        conn = PeerConnection(reader, writer, temp_info, is_inbound=True)

        # Send our handshake
        await conn.send(Message(
            type=MessageType.HANDSHAKE,
            sender=self.node_id,
            payload={
                "chain_id": self.chain_id,
                "port": self.port,
                "version": "1.0.0",
            }
        ))

        # Wait for handshake response with timeout
        try:
            msg = await asyncio.wait_for(conn.receive(), timeout=10.0)
        except asyncio.TimeoutError:
            await conn.close()
            return

        if not msg or msg.type != MessageType.HANDSHAKE:
            await conn.close()
            return

        peer_id = msg.sender
        if peer_id == self.node_id or peer_id in self.peers:
            await conn.close()
            return

        # Accept the peer
        conn.peer_info.peer_id = peer_id
        conn.peer_info.chain_id = msg.payload.get("chain_id", "")
        conn.peer_info.connected = True
        conn.peer_info.last_seen = time.time()
        self.peers[peer_id] = conn
        self.known_peers[peer_id] = conn.peer_info
        logger.info("Peer connected (inbound): %s", peer_id[:8])

        # Start receive loop
        self._recv_tasks[peer_id] = asyncio.ensure_future(self._peer_recv_loop(peer_id))

    async def connect_to(self, host: str, port: int) -> Optional[PeerConnection]:
        """Connect to a remote peer."""
        if len(self.peers) >= self.max_peers:
            return None

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=10.0
            )
        except (ConnectionError, asyncio.TimeoutError, OSError) as e:
            logger.debug("Failed to connect to %s:%d: %s", host, port, e)
            return None

        temp_info = PeerInfo(host=host, port=port)
        conn = PeerConnection(reader, writer, temp_info, is_inbound=False)

        # Send handshake
        await conn.send(Message(
            type=MessageType.HANDSHAKE,
            sender=self.node_id,
            payload={
                "chain_id": self.chain_id,
                "port": self.port,
                "version": "1.0.0",
            }
        ))

        # Wait for remote handshake
        try:
            msg = await asyncio.wait_for(conn.receive(), timeout=10.0)
        except asyncio.TimeoutError:
            await conn.close()
            return None

        if not msg or msg.type != MessageType.HANDSHAKE:
            await conn.close()
            return None

        peer_id = msg.sender
        if peer_id == self.node_id or peer_id in self.peers:
            await conn.close()
            return None

        conn.peer_info.peer_id = peer_id
        conn.peer_info.chain_id = msg.payload.get("chain_id", "")
        conn.peer_info.connected = True
        conn.peer_info.last_seen = time.time()
        self.peers[peer_id] = conn
        self.known_peers[peer_id] = conn.peer_info
        logger.info("Peer connected (outbound): %s @ %s:%d", peer_id[:8], host, port)

        self._recv_tasks[peer_id] = asyncio.ensure_future(self._peer_recv_loop(peer_id))
        return conn

    async def disconnect_peer(self, peer_id: str):
        """Disconnect a specific peer."""
        conn = self.peers.pop(peer_id, None)
        if conn:
            await conn.send(Message(type=MessageType.DISCONNECT, sender=self.node_id))
            await conn.close()
            if peer_id in self.known_peers:
                self.known_peers[peer_id].connected = False
        task = self._recv_tasks.pop(peer_id, None)
        if task:
            task.cancel()

    # ── Message sending ────────────────────────────────────────────────

    async def send(self, peer_id: str, msg: Message) -> bool:
        """Send a message to a specific peer."""
        conn = self.peers.get(peer_id)
        if not conn:
            return False
        msg.sender = self.node_id
        return await conn.send(msg)

    async def broadcast(self, msg: Message, exclude: Optional[Set[str]] = None):
        """Broadcast a message to all connected peers."""
        msg.sender = self.node_id
        exclude = exclude or set()
        for peer_id, conn in list(self.peers.items()):
            if peer_id not in exclude:
                success = await conn.send(msg)
                if not success:
                    await self.disconnect_peer(peer_id)

    async def gossip(self, msg: Message, fanout: int = 0, exclude: Optional[Set[str]] = None):
        """Gossip a message to a random subset of peers.
        
        If fanout is 0, broadcasts to all peers. Otherwise, selects
        ``fanout`` random peers from the connected set.
        """
        # Deduplication
        if msg.nonce in self._seen_messages:
            return
        self._seen_messages.add(msg.nonce)
        if len(self._seen_messages) > self._seen_max:
            # Trim oldest (convert to list, drop first half)
            self._seen_messages = set(list(self._seen_messages)[self._seen_max // 2:])

        msg.sender = self.node_id
        exclude = exclude or set()
        candidates = [pid for pid in self.peers if pid not in exclude]

        if fanout > 0 and len(candidates) > fanout:
            candidates = random.sample(candidates, fanout)

        for peer_id in candidates:
            conn = self.peers.get(peer_id)
            if conn:
                await conn.send(msg)

    # ── Receive loop ───────────────────────────────────────────────────

    async def _peer_recv_loop(self, peer_id: str):
        """Continuous receive loop for a peer."""
        conn = self.peers.get(peer_id)
        if not conn:
            return
        while self._running and conn.is_connected:
            msg = await conn.receive()
            if msg is None:
                break
            conn.peer_info.last_seen = time.time()

            if msg.type == MessageType.DISCONNECT:
                break

            await self._dispatch(msg, conn)

        # Cleanup
        self.peers.pop(peer_id, None)
        if peer_id in self.known_peers:
            self.known_peers[peer_id].connected = False
        logger.debug("Peer disconnected: %s", peer_id[:8])

    # ── Built-in handlers ──────────────────────────────────────────────

    async def _handle_ping(self, msg: Message, conn: PeerConnection):
        await conn.send(Message(
            type=MessageType.PONG,
            sender=self.node_id,
            payload={"echo": msg.nonce}
        ))

    async def _handle_find_peers(self, msg: Message, conn: PeerConnection):
        """Return known peers to the requester."""
        peers_list = []
        for pid, info in self.known_peers.items():
            if pid != msg.sender and info.connected:
                peers_list.append({
                    "peer_id": info.peer_id,
                    "host": info.host,
                    "port": info.port,
                })
        await conn.send(Message(
            type=MessageType.PEERS,
            sender=self.node_id,
            payload={"peers": peers_list[:20]}
        ))

    async def _handle_handshake_msg(self, msg: Message, conn: PeerConnection):
        """Handle late handshake messages (re-announcements)."""
        pass  # Already handled during connection setup

    # ── Peer discovery ─────────────────────────────────────────────────

    def add_bootstrap_node(self, host: str, port: int):
        """Add a bootstrap node for initial peer discovery."""
        self.bootstrap_nodes.append((host, port))

    async def _bootstrap(self):
        """Connect to bootstrap nodes and discover more peers."""
        for host, port in self.bootstrap_nodes:
            if len(self.peers) >= self.max_peers:
                break
            await self.connect_to(host, port)

        # Discover more peers from connected ones
        if self.peers:
            await self.broadcast(Message(
                type=MessageType.FIND_PEERS,
                sender=self.node_id,
            ))

    async def discover_peers(self):
        """Actively discover new peers by querying existing connections."""
        if not self.peers:
            await self._bootstrap()
            return
        await self.broadcast(Message(
            type=MessageType.FIND_PEERS,
            sender=self.node_id,
        ))

    # ── Utility ────────────────────────────────────────────────────────

    def get_network_info(self) -> Dict[str, Any]:
        """Get network status information."""
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "chain_id": self.chain_id,
            "connected_peers": len(self.peers),
            "known_peers": len(self.known_peers),
            "running": self._running,
            "peers": [
                {
                    "peer_id": pid[:8],
                    "address": conn.peer_info.address,
                    "inbound": conn.is_inbound,
                    "last_seen": conn.peer_info.last_seen,
                }
                for pid, conn in self.peers.items()
            ]
        }
