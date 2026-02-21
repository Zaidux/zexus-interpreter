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
import ssl
import threading
import time
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("zexus.blockchain.network")


# ── TLS helpers ────────────────────────────────────────────────────────

def _generate_self_signed_cert(cert_path: str, key_path: str):
    """Generate a self-signed TLS certificate for node identity.

    Uses the ``cryptography`` library (already a dependency) to produce
    an ECDSA P-256 keypair and an X.509 certificate valid for 10 years.
    The certificate's Subject is set to the key's SHA-256 fingerprint so
    it doubles as a verifiable node identity.
    """
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.backends import default_backend
    import datetime

    key = ec.generate_private_key(ec.SECP256R1(), default_backend())

    # Derive a human-readable CN from the public key fingerprint
    pub_bytes = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    fingerprint = hashlib.sha256(pub_bytes).hexdigest()[:16]
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, f"zexus-node-{fingerprint}"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .sign(key, hashes.SHA256(), default_backend())
    )

    os.makedirs(os.path.dirname(cert_path) or ".", exist_ok=True)

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path


def _make_server_ssl_context(
    cert_path: str,
    key_path: str,
    ca_cert_path: Optional[str] = None,
    require_client_cert: bool = False,
) -> ssl.SSLContext:
    """Create a TLS server context with optional mutual-TLS (mTLS).

    Parameters
    ----------
    cert_path : str
        Path to the server certificate (PEM).
    key_path : str
        Path to the server private key (PEM).
    ca_cert_path : str, optional
        Path to a CA bundle.  When provided, the server verifies peer
        certificates against this CA — enabling proper CA-signed
        certificate chains instead of self-signed only.
    require_client_cert : bool
        When *True*, the server demands a valid client certificate
        (mutual TLS).  Only effective when ``ca_cert_path`` is set.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(cert_path, key_path)

    if ca_cert_path and os.path.isfile(ca_cert_path):
        ctx.load_verify_locations(ca_cert_path)
        if require_client_cert:
            ctx.verify_mode = ssl.CERT_REQUIRED
        else:
            ctx.verify_mode = ssl.CERT_OPTIONAL
        logger.info("Server TLS: CA-signed verification enabled (mTLS=%s)", require_client_cert)
    else:
        ctx.verify_mode = ssl.CERT_NONE

    ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5:!DSS")
    return ctx


def _make_client_ssl_context(
    cert_path: Optional[str] = None,
    key_path: Optional[str] = None,
    ca_cert_path: Optional[str] = None,
) -> ssl.SSLContext:
    """Create a TLS client context with optional CA verification.

    Parameters
    ----------
    cert_path : str, optional
        Client certificate for mutual TLS.
    key_path : str, optional
        Client private key for mutual TLS.
    ca_cert_path : str, optional
        CA bundle.  When provided the client verifies the server's
        certificate against known CA roots — enabling production-grade
        certificate validation instead of trusting all self-signed certs.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    if ca_cert_path and os.path.isfile(ca_cert_path):
        ctx.load_verify_locations(ca_cert_path)
        ctx.check_hostname = False   # P2P nodes don't use DNS hostnames
        ctx.verify_mode = ssl.CERT_REQUIRED
        logger.info("Client TLS: CA-signed server verification enabled")
    else:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE  # fallback: accept self-signed

    if cert_path and key_path:
        ctx.load_cert_chain(cert_path, key_path)

    ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5:!DSS")
    return ctx


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


class PeerReputationManager:
    """Track and enforce peer reputation to resist Sybil and DoS attacks.

    Every peer starts at score 100 (maximum).  Good behaviour (valid blocks,
    valid transactions) earns points; bad behaviour (invalid messages, spam,
    protocol violations) costs points.  Peers whose score drops to 0 are
    banned for ``ban_duration`` seconds.

    The manager also enforces per-peer message rate limiting.
    """

    # ── Reputation deltas ──────────────────────────────────────────
    VALID_BLOCK = 5
    VALID_TX = 1
    INVALID_BLOCK = -20
    INVALID_TX = -10
    PROTOCOL_VIOLATION = -25
    TIMEOUT = -5
    SPAM = -15
    SUCCESSFUL_SYNC = 10

    def __init__(self, ban_duration: float = 3600.0,
                 rate_limit: int = 100,
                 rate_window: float = 60.0):
        self.ban_duration = ban_duration          # seconds
        self.rate_limit = rate_limit              # msgs per window
        self.rate_window = rate_window            # seconds
        self._bans: Dict[str, float] = {}         # peer_id -> unban timestamp
        self._msg_counts: Dict[str, List[float]] = {}  # peer_id -> [timestamps]

    def update(self, peer: PeerInfo, delta: int, reason: str = "") -> int:
        """Adjust a peer's reputation score.

        Returns the new score. If it drops to 0, the peer is banned.
        """
        old = peer.reputation
        peer.reputation = max(0, min(100, peer.reputation + delta))
        if reason:
            logger.debug("Reputation %s: %d -> %d (%s)", peer.peer_id[:8], old, peer.reputation, reason)
        if peer.reputation == 0:
            self.ban(peer.peer_id)
        return peer.reputation

    def ban(self, peer_id: str):
        """Ban a peer for ``ban_duration`` seconds."""
        self._bans[peer_id] = time.time() + self.ban_duration
        logger.warning("Peer %s BANNED for %ds", peer_id[:8], int(self.ban_duration))

    def unban(self, peer_id: str):
        """Manually unban a peer."""
        self._bans.pop(peer_id, None)

    def is_banned(self, peer_id: str) -> bool:
        """Check if a peer is currently banned."""
        if peer_id not in self._bans:
            return False
        if time.time() >= self._bans[peer_id]:
            del self._bans[peer_id]
            return False
        return True

    def check_rate_limit(self, peer_id: str) -> bool:
        """Check if a peer has exceeded the message rate limit.

        Returns True if the message should be allowed, False if rate-limited.
        """
        now = time.time()
        timestamps = self._msg_counts.setdefault(peer_id, [])
        # Prune old entries
        cutoff = now - self.rate_window
        self._msg_counts[peer_id] = [t for t in timestamps if t > cutoff]
        timestamps = self._msg_counts[peer_id]

        if len(timestamps) >= self.rate_limit:
            return False  # Rate limited
        timestamps.append(now)
        return True

    def get_banned_peers(self) -> List[str]:
        """Return list of currently banned peer IDs."""
        now = time.time()
        # Prune expired bans
        self._bans = {pid: ts for pid, ts in self._bans.items() if ts > now}
        return list(self._bans.keys())


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
                 max_peers: int = 25, min_peers: int = 3,
                 tls_cert: Optional[str] = None, tls_key: Optional[str] = None,
                 tls_ca: Optional[str] = None,
                 tls_enabled: bool = True,
                 tls_mutual: bool = False,
                 tls_pinned_certs: Optional[List[str]] = None,
                 data_dir: Optional[str] = None):
        self.host = host
        self.port = port
        self.chain_id = chain_id
        self.node_id = node_id or hashlib.sha256(
            f"{host}:{port}:{time.time()}:{random.random()}".encode()
        ).hexdigest()[:40]
        self.max_peers = max_peers
        self.min_peers = min_peers

        # ── TLS configuration ────────────────────────────────────────
        self.tls_enabled = tls_enabled
        self.tls_mutual = tls_mutual
        self._server_ssl: Optional[ssl.SSLContext] = None
        self._client_ssl: Optional[ssl.SSLContext] = None

        # Certificate pinning: set of SHA-256 fingerprints of trusted
        # peer certificates.  If non-empty, only peers whose TLS cert
        # fingerprint is in this set are accepted.
        self._pinned_certs: Set[str] = set(tls_pinned_certs or [])

        if tls_enabled:
            # Auto-generate certs if none provided
            _data = data_dir or os.path.join(os.path.expanduser("~"), ".zexus", "tls")
            self._cert_path = tls_cert or os.path.join(_data, "node.crt")
            self._key_path = tls_key or os.path.join(_data, "node.key")
            self._ca_path = tls_ca  # None = self-signed mode

            if not (os.path.exists(self._cert_path) and os.path.exists(self._key_path)):
                logger.info("Generating TLS certificate for node identity...")
                _generate_self_signed_cert(self._cert_path, self._key_path)

            self._server_ssl = _make_server_ssl_context(
                self._cert_path, self._key_path,
                ca_cert_path=self._ca_path,
                require_client_cert=self.tls_mutual,
            )
            self._client_ssl = _make_client_ssl_context(
                self._cert_path, self._key_path,
                ca_cert_path=self._ca_path,
            )
            mode = "mTLS" if tls_mutual else ("CA-verified" if tls_ca else "self-signed")
            logger.info("TLS enabled (%s) — all P2P traffic is encrypted", mode)

        # Connection state
        self.peers: Dict[str, PeerConnection] = {}  # peer_id -> connection
        self.known_peers: Dict[str, PeerInfo] = {}  # peer_id -> info (includes disconnected)
        self.bootstrap_nodes: List[Tuple[str, int]] = []

        # Message handling
        self._handlers: Dict[str, List[Callable]] = {}
        self._seen_messages: Set[str] = set()
        self._seen_max = 10_000

        # Sybil / DoS resistance
        self.reputation = PeerReputationManager()

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
        """Dispatch a received message to registered handlers.

        Enforces rate-limiting and reputation checks before dispatch.
        """
        peer_id = conn.peer_info.peer_id

        # Block banned peers
        if self.reputation.is_banned(peer_id):
            logger.debug("Dropping message from banned peer %s", peer_id[:8])
            return

        # Rate-limit check
        if not self.reputation.check_rate_limit(peer_id):
            self.reputation.update(conn.peer_info, PeerReputationManager.SPAM,
                                   reason="rate limit exceeded")
            logger.warning("Rate-limited peer %s", peer_id[:8])
            if self.reputation.is_banned(peer_id):
                await self.disconnect_peer(peer_id)
            return

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
            self._handle_inbound, self.host, self.port,
            ssl=self._server_ssl,  # None when TLS disabled → plain TCP
        )
        self._running = True
        tls_status = "TLS" if self.tls_enabled else "plaintext"
        logger.info("P2P listening on %s:%d (%s, node_id=%s)",
                     self.host, self.port, tls_status, self.node_id[:8])

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

    # ── Certificate pinning ──────────────────────────────────────────

    def _check_cert_pin(self, writer: asyncio.StreamWriter) -> bool:
        """Verify the peer's TLS certificate fingerprint against the
        pinned set.  Returns *True* if pinning is disabled or if the
        cert is in the pinned set; *False* to reject the connection."""
        if not self._pinned_certs:
            return True  # pinning not configured — accept all
        ssl_obj = writer.get_extra_info("ssl_object")
        if ssl_obj is None:
            return True  # non-TLS — nothing to pin
        try:
            der = ssl_obj.getpeercert(binary_form=True)
            if der is None:
                return False  # no cert presented
            fp = hashlib.sha256(der).hexdigest()
            if fp in self._pinned_certs:
                return True
            logger.warning("Certificate pin mismatch: %s", fp)
            return False
        except Exception:
            return False

    # ── Connections ────────────────────────────────────────────────────

    async def _handle_inbound(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a new inbound TCP connection."""
        addr = writer.get_extra_info("peername")
        logger.debug("Inbound connection from %s", addr)

        if len(self.peers) >= self.max_peers:
            writer.close()
            return

        # Certificate pinning check
        if not self._check_cert_pin(writer):
            logger.warning("Rejected inbound connection (cert pin mismatch) from %s", addr)
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

        # Reject banned peers
        if self.reputation.is_banned(peer_id):
            logger.info("Rejected banned peer %s", peer_id[:8])
            await conn.close()
            return
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
                asyncio.open_connection(host, port, ssl=self._client_ssl),
                timeout=10.0,
            )
        except (ConnectionError, asyncio.TimeoutError, OSError) as e:
            logger.debug("Failed to connect to %s:%d: %s", host, port, e)
            return None

        # Certificate pinning check
        if not self._check_cert_pin(writer):
            logger.warning("Rejected outbound connection (cert pin mismatch) to %s:%d", host, port)
            writer.close()
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

        # Reject banned peers
        if self.reputation.is_banned(peer_id):
            logger.info("Refused connection to banned peer %s", peer_id[:8])
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
            "tls_enabled": self.tls_enabled,
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
