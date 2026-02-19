"""
Zexus Blockchain — JSON-RPC Server

Production-grade JSON-RPC 2.0 server providing external access to a
running BlockchainNode.  Supports both HTTP and WebSocket transports.

Namespaces:
  zx_*      — Core blockchain methods (query blocks, accounts, state)
  txpool_*  — Mempool / transaction pool
  net_*     — Peer-to-peer networking
  miner_*   — Mining control
  contract_* — Smart-contract deployment and interaction
  admin_*   — Node administration

The server follows the JSON-RPC 2.0 specification:
  https://www.jsonrpc.org/specification

Usage:
    node = BlockchainNode(NodeConfig(rpc_enabled=True, rpc_port=8545))
    await node.start()   # RPC server starts automatically

Or standalone:
    server = RPCServer(node, host="0.0.0.0", port=8545)
    await server.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("zexus.blockchain.rpc")

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 error codes
# ---------------------------------------------------------------------------

class RPCErrorCode(IntEnum):
    """Standard JSON-RPC 2.0 error codes + Zexus-specific extensions."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Zexus-specific  (-32000 to -32099 reserved for server errors)
    CHAIN_ERROR = -32000
    TX_REJECTED = -32001
    INSUFFICIENT_FUNDS = -32002
    NONCE_TOO_LOW = -32003
    CONTRACT_ERROR = -32004
    MINING_ERROR = -32005
    NOT_FOUND = -32006
    UNAUTHORIZED = -32007


class RPCError(Exception):
    """An error that maps directly to a JSON-RPC error response."""

    def __init__(self, code: RPCErrorCode, message: str, data: Any = None):
        super().__init__(message)
        self.code = int(code)
        self.message = message
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            d["data"] = self.data
        return d


# ---------------------------------------------------------------------------
# Subscription manager (WebSocket push)
# ---------------------------------------------------------------------------

@dataclass
class Subscription:
    """A single WebSocket subscription."""
    sub_id: str
    event: str  # "newHeads", "newPendingTransactions", "logs"
    ws: Any  # aiohttp WebSocketResponse
    params: Dict[str, Any] = field(default_factory=dict)


class SubscriptionManager:
    """Manages WebSocket subscriptions for real-time event streaming."""

    def __init__(self):
        self._subs: Dict[str, Subscription] = {}
        self._counter = 0

    def add(self, event: str, ws: Any, params: Optional[Dict] = None) -> str:
        self._counter += 1
        sub_id = f"0x{self._counter:016x}"
        self._subs[sub_id] = Subscription(
            sub_id=sub_id, event=event, ws=ws, params=params or {},
        )
        return sub_id

    def remove(self, sub_id: str) -> bool:
        return self._subs.pop(sub_id, None) is not None

    def remove_all_for_ws(self, ws: Any) -> int:
        to_remove = [sid for sid, s in self._subs.items() if s.ws is ws]
        for sid in to_remove:
            del self._subs[sid]
        return len(to_remove)

    async def notify(self, event: str, data: Any):
        """Push a notification to all subscribers of *event*."""
        dead: List[str] = []
        for sid, sub in self._subs.items():
            if sub.event != event:
                continue
            msg = json.dumps({
                "jsonrpc": "2.0",
                "method": "zx_subscription",
                "params": {"subscription": sid, "result": data},
            })
            try:
                await sub.ws.send_str(msg)
            except Exception:
                dead.append(sid)
        for sid in dead:
            self._subs.pop(sid, None)

    @property
    def count(self) -> int:
        return len(self._subs)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter per IP address."""

    def __init__(self, max_requests: int = 100, window_seconds: float = 1.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: Dict[str, List[float]] = {}

    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets.setdefault(ip, [])
        # Prune old entries
        bucket[:] = [t for t in bucket if now - t < self.window]
        if len(bucket) >= self.max_requests:
            return False
        bucket.append(now)
        return True

    def reset(self):
        self._buckets.clear()


# ---------------------------------------------------------------------------
# RPC Method registry
# ---------------------------------------------------------------------------

# Type alias for an RPC handler: takes (params) -> result
RPCHandler = Callable[..., Any]


@dataclass
class RPCMethodInfo:
    """Metadata about a registered RPC method."""
    name: str
    handler: RPCHandler
    is_async: bool = False
    description: str = ""


class RPCMethodRegistry:
    """Registry that maps JSON-RPC method names to Python handlers."""

    def __init__(self):
        self._methods: Dict[str, RPCMethodInfo] = {}

    def register(self, name: str, handler: RPCHandler, description: str = ""):
        is_async = asyncio.iscoroutinefunction(handler)
        self._methods[name] = RPCMethodInfo(
            name=name, handler=handler, is_async=is_async,
            description=description,
        )

    def get(self, name: str) -> Optional[RPCMethodInfo]:
        return self._methods.get(name)

    def list_methods(self) -> List[str]:
        return sorted(self._methods.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._methods

    def __len__(self) -> int:
        return len(self._methods)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _hex_int(value: int) -> str:
    """Encode an integer as a 0x-prefixed hex string."""
    return hex(value)


def _parse_hex_int(value: Any) -> int:
    """Parse a 0x-prefixed hex string or plain int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.startswith("0x") or value.startswith("0X"):
            return int(value, 16)
        return int(value)
    raise RPCError(RPCErrorCode.INVALID_PARAMS, f"expected integer, got {type(value).__name__}")


def _require_param(params: Any, key: Union[str, int], name: str = "") -> Any:
    """Extract a required parameter from dict or list."""
    label = name or str(key)
    try:
        if isinstance(params, dict):
            if key not in params:
                raise KeyError(key)
            return params[key]
        if isinstance(params, (list, tuple)):
            return params[key]
    except (KeyError, IndexError, TypeError):
        pass
    raise RPCError(RPCErrorCode.INVALID_PARAMS, f"missing required parameter: {label}")


def _optional_param(params: Any, key: Union[str, int], default: Any = None) -> Any:
    """Extract an optional parameter."""
    try:
        if isinstance(params, dict):
            return params.get(key, default)
        if isinstance(params, (list, tuple)) and isinstance(key, int) and key < len(params):
            return params[key]
    except Exception:
        pass
    return default


def _safe_serialize(obj: Any) -> Any:
    """Make an object JSON-safe (handle non-serializable types)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, bytes):
        return "0x" + obj.hex()
    if hasattr(obj, "to_dict"):
        return _safe_serialize(obj.to_dict())
    return str(obj)


# ---------------------------------------------------------------------------
# RPCServer — the main server class
# ---------------------------------------------------------------------------

class RPCServer:
    """
    JSON-RPC 2.0 server for a Zexus BlockchainNode.

    Supports:
      - HTTP POST requests (standard JSON-RPC)
      - WebSocket connections (subscriptions + regular calls)
      - Batch requests
      - CORS for browser-based dApps
      - Per-IP rate limiting
      - Method namespace: zx_, txpool_, net_, miner_, contract_, admin_

    Usage::

        from zexus.blockchain.node import BlockchainNode, NodeConfig
        from zexus.blockchain.rpc import RPCServer

        node = BlockchainNode(NodeConfig(rpc_enabled=True))
        rpc = RPCServer(node, host="127.0.0.1", port=8545)
        await rpc.start()
        # ... node is now accessible via HTTP/WS at port 8545
        await rpc.stop()
    """

    def __init__(
        self,
        node: Any,  # BlockchainNode — avoid circular import at module level
        host: str = "0.0.0.0",
        port: int = 8545,
        cors_origins: str = "*",
        max_request_size: int = 5 * 1024 * 1024,  # 5 MB
        rate_limit: int = 200,  # requests per second per IP
    ):
        self.node = node
        self.host = host
        self.port = port
        self.cors_origins = cors_origins
        self.max_request_size = max_request_size

        # Components
        self.registry = RPCMethodRegistry()
        self.subscriptions = SubscriptionManager()
        self.rate_limiter = RateLimiter(max_requests=rate_limit)

        # Server state
        self._app = None
        self._runner = None
        self._site = None
        self._running = False

        # Register all RPC methods
        self._register_all_methods()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Start the HTTP+WS server."""
        if self._running:
            return

        try:
            from aiohttp import web
        except ImportError:
            logger.error(
                "aiohttp is required for the RPC server. "
                "Install with: pip install aiohttp"
            )
            return

        self._app = web.Application(client_max_size=self.max_request_size)
        self._app.router.add_post("/", self._handle_http)
        self._app.router.add_get("/", self._handle_ws_or_info)
        self._app.router.add_get("/ws", self._handle_ws)
        self._app.router.add_options("/", self._handle_cors_preflight)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()
        self._running = True

        # Hook into node events for WebSocket subscriptions
        self._wire_node_events()

        logger.info("RPC server listening on http://%s:%d", self.host, self.port)

    async def stop(self):
        """Stop the server gracefully."""
        if not self._running:
            return
        self._running = False
        if self._runner:
            await self._runner.cleanup()
        self._app = None
        self._runner = None
        self._site = None
        logger.info("RPC server stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # HTTP handler
    # ------------------------------------------------------------------

    async def _handle_http(self, request):
        """Handle a JSON-RPC POST request."""
        from aiohttp import web

        # CORS headers (Content-Type omitted — json_response sets it)
        headers = {
            "Access-Control-Allow-Origin": self.cors_origins,
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        }

        # Rate limiting
        ip = request.remote or "unknown"
        if not self.rate_limiter.allow(ip):
            return web.json_response(
                {"jsonrpc": "2.0", "error": {"code": -32000, "message": "rate limit exceeded"}, "id": None},
                status=429, headers=headers,
            )

        # Parse body
        try:
            body = await request.json()
        except Exception:
            resp = self._error_response(None, RPCErrorCode.PARSE_ERROR, "Invalid JSON")
            return web.json_response(resp, headers=headers)

        # Batch request?
        if isinstance(body, list):
            if len(body) == 0:
                resp = self._error_response(None, RPCErrorCode.INVALID_REQUEST, "Empty batch")
                return web.json_response(resp, headers=headers)
            if len(body) > 100:
                resp = self._error_response(None, RPCErrorCode.INVALID_REQUEST, "Batch too large (max 100)")
                return web.json_response(resp, headers=headers)
            results = await asyncio.gather(
                *[self._process_single_request(r) for r in body]
            )
            # Filter out notifications (no id)
            results = [r for r in results if r is not None]
            return web.json_response(results, headers=headers)

        # Single request
        result = await self._process_single_request(body)
        if result is None:
            return web.Response(status=204, headers=headers)
        return web.json_response(result, headers=headers)

    async def _handle_cors_preflight(self, request):
        """Handle CORS preflight OPTIONS requests."""
        from aiohttp import web
        return web.Response(
            status=204,
            headers={
                "Access-Control-Allow-Origin": self.cors_origins,
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Max-Age": "86400",
            },
        )

    async def _handle_health(self, request):
        """Health check endpoint."""
        from aiohttp import web
        return web.json_response({
            "status": "ok",
            "chain_id": self.node.config.chain_id,
            "height": self.node.chain.height,
            "peers": self.node.network.peer_count,
            "rpc_methods": len(self.registry),
            "subscriptions": self.subscriptions.count,
        })

    async def _handle_ws_or_info(self, request):
        """GET / — upgrade to WS if requested, otherwise return node info."""
        from aiohttp import web
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self._handle_ws(request)
        # Return a friendly info page for browsers
        info = {
            "jsonrpc": "2.0",
            "server": "Zexus Blockchain RPC",
            "chain_id": self.node.config.chain_id,
            "height": self.node.chain.height,
            "methods": self.registry.list_methods(),
        }
        return web.json_response(info, headers={
            "Access-Control-Allow-Origin": self.cors_origins,
        })

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handle_ws(self, request):
        """Handle a WebSocket connection (subscriptions + regular calls)."""
        from aiohttp import web, WSMsgType

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        logger.debug("WebSocket client connected: %s", request.remote)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        body = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json(
                            self._error_response(None, RPCErrorCode.PARSE_ERROR, "Invalid JSON")
                        )
                        continue

                    if isinstance(body, list):
                        results = await asyncio.gather(
                            *[self._process_single_request(r, ws=ws) for r in body]
                        )
                        results = [r for r in results if r is not None]
                        if results:
                            await ws.send_json(results)
                    else:
                        result = await self._process_single_request(body, ws=ws)
                        if result is not None:
                            await ws.send_json(result)

                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            removed = self.subscriptions.remove_all_for_ws(ws)
            if removed:
                logger.debug("Cleaned up %d subscriptions for disconnected client", removed)

        return ws

    # ------------------------------------------------------------------
    # Request processing
    # ------------------------------------------------------------------

    async def _process_single_request(
        self, body: Any, ws: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Process a single JSON-RPC request and return the response dict."""
        if not isinstance(body, dict):
            return self._error_response(None, RPCErrorCode.INVALID_REQUEST, "Request must be an object")

        req_id = body.get("id")
        method = body.get("method")
        params = body.get("params", [])

        # Validate
        if body.get("jsonrpc") != "2.0":
            return self._error_response(req_id, RPCErrorCode.INVALID_REQUEST, "jsonrpc must be '2.0'")
        if not method or not isinstance(method, str):
            return self._error_response(req_id, RPCErrorCode.INVALID_REQUEST, "missing or invalid method")

        # Special subscription methods that need WS
        if method == "zx_subscribe" and ws is not None:
            return await self._handle_subscribe(req_id, params, ws)
        if method == "zx_unsubscribe" and ws is not None:
            return await self._handle_unsubscribe(req_id, params)

        # Dispatch to registered handler
        info = self.registry.get(method)
        if info is None:
            return self._error_response(req_id, RPCErrorCode.METHOD_NOT_FOUND, f"method not found: {method}")

        try:
            if info.is_async:
                result = await info.handler(params)
            else:
                result = info.handler(params)
            return self._success_response(req_id, _safe_serialize(result))
        except RPCError as e:
            return self._error_response(req_id, e.code, e.message, e.data)
        except Exception as e:
            logger.error("RPC handler error [%s]: %s\n%s", method, e, traceback.format_exc())
            return self._error_response(req_id, RPCErrorCode.INTERNAL_ERROR, str(e))

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    async def _handle_subscribe(self, req_id, params, ws) -> Dict:
        """Handle zx_subscribe."""
        event = _require_param(params, 0, "event_name")
        sub_params = _optional_param(params, 1, {})
        valid_events = {"newHeads", "newPendingTransactions", "logs", "syncing"}
        if event not in valid_events:
            raise RPCError(RPCErrorCode.INVALID_PARAMS,
                           f"unknown subscription event: {event}. "
                           f"Valid: {', '.join(sorted(valid_events))}")
        sub_id = self.subscriptions.add(event, ws, sub_params if isinstance(sub_params, dict) else {})
        return self._success_response(req_id, sub_id)

    async def _handle_unsubscribe(self, req_id, params) -> Dict:
        """Handle zx_unsubscribe."""
        sub_id = _require_param(params, 0, "subscription_id")
        removed = self.subscriptions.remove(sub_id)
        return self._success_response(req_id, removed)

    def _wire_node_events(self):
        """Connect node events to subscription notifications."""
        def on_new_block(block):
            if not self._running:
                return
            data = _safe_serialize(block.to_dict()) if hasattr(block, "to_dict") else {}
            asyncio.run_coroutine_threadsafe(
                self.subscriptions.notify("newHeads", data),
                asyncio.get_event_loop(),
            )

        def on_new_tx(tx):
            if not self._running:
                return
            tx_data = _safe_serialize(tx.to_dict()) if hasattr(tx, "to_dict") else {"tx_hash": getattr(tx, "tx_hash", "")}
            asyncio.run_coroutine_threadsafe(
                self.subscriptions.notify("newPendingTransactions", tx_data),
                asyncio.get_event_loop(),
            )

        self.node.on("new_block", on_new_block)
        self.node.on("mined", on_new_block)
        self.node.on("new_tx", on_new_tx)

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    @staticmethod
    def _success_response(req_id: Any, result: Any) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "result": result, "id": req_id}

    @staticmethod
    def _error_response(req_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
        err: Dict[str, Any] = {"code": int(code), "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "error": err, "id": req_id}

    # ══════════════════════════════════════════════════════════════════
    #  RPC method registration
    # ══════════════════════════════════════════════════════════════════

    def _register_all_methods(self):
        """Register every RPC method."""
        r = self.registry.register

        # ── zx_* namespace: core blockchain queries ───────────────
        r("zx_chainId", self._zx_chain_id, "Get the chain ID")
        r("zx_blockNumber", self._zx_block_number, "Get current block height")
        r("zx_getBlockByNumber", self._zx_get_block_by_number, "Get block by height")
        r("zx_getBlockByHash", self._zx_get_block_by_hash, "Get block by hash")
        r("zx_getBlockTransactionCount", self._zx_get_block_tx_count, "Get tx count in a block")
        r("zx_getBalance", self._zx_get_balance, "Get account balance")
        r("zx_getTransactionCount", self._zx_get_tx_count, "Get account nonce")
        r("zx_getAccount", self._zx_get_account, "Get full account state")
        r("zx_getTransactionByHash", self._zx_get_tx_by_hash, "Get a transaction by hash")
        r("zx_getTransactionReceipt", self._zx_get_tx_receipt, "Get a transaction receipt")
        r("zx_sendTransaction", self._zx_send_transaction, "Submit a signed transaction")
        r("zx_sendRawTransaction", self._zx_send_raw_transaction, "Submit a raw transaction dict")
        r("zx_gasPrice", self._zx_gas_price, "Get suggested gas price")
        r("zx_estimateGas", self._zx_estimate_gas, "Estimate gas for a call")
        r("zx_getChainInfo", self._zx_get_chain_info, "Get chain + network summary")
        r("zx_validateChain", self._zx_validate_chain, "Validate full chain integrity")
        r("zx_getCode", self._zx_get_code, "Get contract code at address")
        r("zx_getLogs", self._zx_get_logs, "Get event logs (filtered)")

        # ── txpool_* namespace: mempool ───────────────────────────
        r("txpool_status", self._txpool_status, "Mempool size and stats")
        r("txpool_content", self._txpool_content, "List pending transactions")
        r("txpool_replaceByFee", self._txpool_replace_by_fee, "Replace tx with higher gas price (RBF)")

        # ── net_* namespace: networking ───────────────────────────
        r("net_version", self._net_version, "Get network / chain ID")
        r("net_peerCount", self._net_peer_count, "Get connected peer count")
        r("net_listening", self._net_listening, "Is node listening for connections")
        r("net_peers", self._net_peers, "Get connected peer info")

        # ── miner_* namespace: mining control ─────────────────────
        r("miner_start", self._miner_start, "Start mining")
        r("miner_stop", self._miner_stop, "Stop mining")
        r("miner_status", self._miner_status, "Get mining status")
        r("miner_setMinerAddress", self._miner_set_address, "Set miner reward address")
        r("miner_mineBlock", self._miner_mine_block, "Mine one block synchronously")

        # ── contract_* namespace: smart contracts ─────────────────
        r("contract_deploy", self._contract_deploy, "Deploy a smart contract")
        r("contract_call", self._contract_call, "Execute a contract action (state-changing)")
        r("contract_staticCall", self._contract_static_call, "Read-only contract call")

        # ── admin_* namespace: node administration ────────────────
        r("admin_nodeInfo", self._admin_node_info, "Get full node information")
        r("admin_fundAccount", self._admin_fund_account, "Fund an account (devnet)")
        r("admin_exportChain", self._admin_export_chain, "Export full chain JSON")
        r("admin_rpcMethods", self._admin_rpc_methods, "List all available RPC methods")

    # ══════════════════════════════════════════════════════════════════
    #  zx_* handlers
    # ══════════════════════════════════════════════════════════════════

    def _zx_chain_id(self, params) -> str:
        return self.node.config.chain_id

    def _zx_block_number(self, params) -> str:
        return _hex_int(self.node.chain.height)

    def _zx_get_block_by_number(self, params) -> Optional[Dict]:
        height = _parse_hex_int(_require_param(params, 0, "block_number"))
        full_txs = _optional_param(params, 1, False)
        block = self.node.get_block(height)
        if block is None:
            return None
        if not full_txs:
            # Return only tx hashes instead of full tx objects
            block["transactions"] = [
                tx.get("tx_hash", "") if isinstance(tx, dict) else tx
                for tx in block.get("transactions", [])
            ]
        return block

    def _zx_get_block_by_hash(self, params) -> Optional[Dict]:
        block_hash = _require_param(params, 0, "block_hash")
        full_txs = _optional_param(params, 1, False)
        block = self.node.get_block(block_hash)
        if block is None:
            return None
        if not full_txs:
            block["transactions"] = [
                tx.get("tx_hash", "") if isinstance(tx, dict) else tx
                for tx in block.get("transactions", [])
            ]
        return block

    def _zx_get_block_tx_count(self, params) -> str:
        block_id = _require_param(params, 0, "block_number_or_hash")
        block_id = _parse_hex_int(block_id) if isinstance(block_id, str) and block_id.startswith("0x") else block_id
        block = self.node.get_block(block_id)
        if block is None:
            raise RPCError(RPCErrorCode.NOT_FOUND, "block not found")
        return _hex_int(len(block.get("transactions", [])))

    def _zx_get_balance(self, params) -> str:
        address = _require_param(params, 0, "address")
        balance = self.node.get_balance(address)
        return _hex_int(balance)

    def _zx_get_tx_count(self, params) -> str:
        address = _require_param(params, 0, "address")
        nonce = self.node.get_nonce(address)
        return _hex_int(nonce)

    def _zx_get_account(self, params) -> Dict:
        address = _require_param(params, 0, "address")
        return self.node.get_account(address)

    def _zx_get_tx_by_hash(self, params) -> Optional[Dict]:
        tx_hash = _require_param(params, 0, "tx_hash")
        # Search in blocks
        for block in self.node.chain.blocks:
            for tx in block.transactions:
                if tx.tx_hash == tx_hash:
                    result = tx.to_dict()
                    result["block_hash"] = block.hash
                    result["block_height"] = block.header.height
                    return result
        # Search in mempool
        for tx in self.node.mempool.get_pending():
            if tx.tx_hash == tx_hash:
                result = tx.to_dict()
                result["block_hash"] = None
                result["block_height"] = None
                result["status"] = "pending"
                return result
        return None

    def _zx_get_tx_receipt(self, params) -> Optional[Dict]:
        tx_hash = _require_param(params, 0, "tx_hash")
        for block in self.node.chain.blocks:
            for receipt in block.receipts:
                if receipt.tx_hash == tx_hash:
                    result = receipt.to_dict()
                    result["block_hash"] = block.hash
                    result["block_height"] = block.header.height
                    return result
            # Check if tx is in this block (even with no explicit receipt)
            for tx in block.transactions:
                if tx.tx_hash == tx_hash:
                    return {
                        "tx_hash": tx_hash,
                        "block_hash": block.hash,
                        "block_height": block.header.height,
                        "status": 1,
                        "gas_used": tx.gas_limit,
                    }
        return None

    def _zx_send_transaction(self, params) -> str:
        """Create and submit a transaction from parameters."""
        tx_data = _require_param(params, 0, "transaction")
        if not isinstance(tx_data, dict):
            raise RPCError(RPCErrorCode.INVALID_PARAMS, "transaction must be an object")

        from .chain import Transaction
        tx = Transaction(
            sender=tx_data.get("from", tx_data.get("sender", "")),
            recipient=tx_data.get("to", tx_data.get("recipient", "")),
            value=_parse_hex_int(tx_data.get("value", 0)),
            data=tx_data.get("data", ""),
            nonce=_parse_hex_int(tx_data.get("nonce", 0)),
            gas_limit=_parse_hex_int(tx_data.get("gas", tx_data.get("gasLimit", tx_data.get("gas_limit", 21000)))),
            gas_price=_parse_hex_int(tx_data.get("gasPrice", tx_data.get("gas_price", 1))),
            timestamp=time.time(),
        )
        tx.compute_hash()

        # Auto-fill nonce if not explicitly provided
        if "nonce" not in tx_data:
            acct = self.node.chain.get_account(tx.sender)
            chain_nonce = acct["nonce"]
            mempool_nonce = self.node.mempool._nonces.get(tx.sender, 0)
            tx.nonce = max(chain_nonce, mempool_nonce)
            tx.compute_hash()

        # Set dev signature for RPC-submitted transactions
        if not tx.signature:
            tx.signature = "rpc_dev_" + tx.tx_hash[:56]

        result = self.node.submit_transaction(tx)
        if not result["success"]:
            error_msg = result.get("error", "transaction rejected")
            if "insufficient balance" in error_msg:
                raise RPCError(RPCErrorCode.INSUFFICIENT_FUNDS, error_msg)
            if "nonce too low" in error_msg:
                raise RPCError(RPCErrorCode.NONCE_TOO_LOW, error_msg)
            raise RPCError(RPCErrorCode.TX_REJECTED, error_msg)
        return result["tx_hash"]

    def _zx_send_raw_transaction(self, params) -> str:
        """Submit a pre-built transaction dict."""
        tx_data = _require_param(params, 0, "raw_transaction")
        if not isinstance(tx_data, dict):
            raise RPCError(RPCErrorCode.INVALID_PARAMS, "raw_transaction must be an object")

        from .chain import Transaction
        tx = Transaction(**{k: v for k, v in tx_data.items() if k in Transaction.__dataclass_fields__})
        if not tx.tx_hash:
            tx.compute_hash()
        if not tx.signature:
            tx.signature = "rpc_dev_" + tx.tx_hash[:56]

        result = self.node.submit_transaction(tx)
        if not result["success"]:
            raise RPCError(RPCErrorCode.TX_REJECTED, result.get("error", "rejected"))
        return result["tx_hash"]

    def _zx_gas_price(self, params) -> str:
        """Return suggested gas price (simple: median of recent block txs)."""
        prices = []
        for block in self.node.chain.blocks[-10:]:
            for tx in block.transactions:
                if tx.gas_price > 0:
                    prices.append(tx.gas_price)
        if prices:
            prices.sort()
            median = prices[len(prices) // 2]
            return _hex_int(max(median, 1))
        return _hex_int(1)  # Minimum gas price

    def _zx_estimate_gas(self, params) -> str:
        """Estimate gas for a transaction via dry-run execution.

        For simple transfers: 21,000 gas.
        For contract deployments (no recipient, has data): size-based estimate.
        For contract calls: attempts dry-run via ContractVM.static_call,
        returning actual gas_used + 20 % safety margin.
        """
        tx_data = _optional_param(params, 0, {})
        if not isinstance(tx_data, dict):
            return _hex_int(21_000)

        data = tx_data.get("data", "")
        to = tx_data.get("to", tx_data.get("recipient", ""))
        value = _parse_hex_int(tx_data.get("value", 0))

        # Simple value transfer (no data, has recipient)
        if not data and to:
            return _hex_int(21_000)

        # Contract deployment (has data, no recipient)
        if data and not to:
            # Base cost + per-byte cost (200 gas per byte of code)
            base = 53_000
            per_byte = len(data.encode("utf-8")) * 200
            return _hex_int(base + per_byte)

        # Contract call — try dry-run via static_call
        if data and to:
            try:
                import json as _j
                call_data = _j.loads(data) if isinstance(data, str) and data.startswith("{") else {}
                action = call_data.get("action", "")
                args = call_data.get("args", {})
                caller = tx_data.get("from", tx_data.get("sender", ""))
                if action and self.node.contract_vm:
                    receipt = self.node.contract_vm.static_call(
                        contract_address=to,
                        action=action,
                        args=args,
                        caller=caller,
                    )
                    gas_used = receipt.gas_used if receipt.gas_used > 0 else 50_000
                    # Add 20% safety margin
                    estimated = int(gas_used * 1.2)
                    return _hex_int(estimated)
            except Exception:
                pass
            # Fallback for generic contract interaction
            return _hex_int(500_000)

        # Fallback: data but unclear intent
        if data:
            return _hex_int(500_000)
        return _hex_int(21_000)

    def _zx_get_chain_info(self, params) -> Dict:
        return self.node.get_chain_info()

    def _zx_validate_chain(self, params) -> Dict:
        return self.node.validate_chain()

    def _zx_get_code(self, params) -> str:
        """Get contract code/state at an address."""
        address = _require_param(params, 0, "address")
        # Check contract_state first (populated by contract_deploy)
        cs = self.node.chain.contract_state.get(address, {})
        if cs.get("code"):
            return cs["code"]
        acct = self.node.chain.get_account(address)
        return acct.get("code", "")

    def _zx_get_logs(self, params) -> List[Dict]:
        """Get event logs filtered by block range, address, topics, and event name.

        Uses the EventIndex (if available) for fast indexed lookups,
        otherwise falls back to scanning block receipts.
        """
        filter_obj = _optional_param(params, 0, {})
        if not isinstance(filter_obj, dict):
            filter_obj = {}

        from_block = _parse_hex_int(filter_obj.get("fromBlock", 0))
        to_block = _parse_hex_int(filter_obj.get("toBlock", self.node.chain.height))
        target_address = filter_obj.get("address")
        topics = filter_obj.get("topics")
        event_name = filter_obj.get("eventName")
        limit = filter_obj.get("limit", 10_000)

        # ── Fast path: use EventIndex if available ──
        if self.node.event_index:
            try:
                from .events import LogFilter
                filt = LogFilter(
                    from_block=from_block,
                    to_block=to_block,
                    address=target_address,
                    topics=topics,
                    event_name=event_name,
                    limit=limit,
                )
                results = self.node.event_index.get_logs(filt)
                return [log.to_dict() for log in results]
            except Exception:
                pass  # Fall through to scan

        # ── Fallback: scan receipts directly ──
        logs = []
        for h in range(from_block, min(to_block + 1, self.node.chain.height + 1)):
            block = self.node.chain.get_block(h)
            if block is None:
                continue
            for receipt in block.receipts:
                for log_entry in receipt.logs:
                    if target_address and log_entry.get("address") != target_address:
                        continue
                    if event_name and log_entry.get("event") != event_name:
                        continue
                    enriched = dict(log_entry)
                    enriched["block_hash"] = block.hash
                    enriched["block_height"] = block.header.height
                    enriched["tx_hash"] = receipt.tx_hash
                    logs.append(enriched)
                    if len(logs) >= limit:
                        return logs
        return logs

    # ══════════════════════════════════════════════════════════════════
    #  txpool_* handlers
    # ══════════════════════════════════════════════════════════════════

    def _txpool_status(self, params) -> Dict:
        mp = self.node.mempool
        return {
            "pending": mp.size,
            "queued": 0,
            "rbf_enabled": mp.rbf_enabled,
            "rbf_increment_pct": mp.rbf_increment_pct,
        }

    def _txpool_content(self, params) -> Dict:
        pending = self.node.mempool.get_pending()
        return {
            "pending": {
                tx.sender: {str(tx.nonce): tx.to_dict()} for tx in pending
            },
        }

    def _txpool_replace_by_fee(self, params) -> Dict:
        """Replace a pending tx with a higher-gas-price version (RBF)."""
        tx_data = _require_param(params, 0, "transaction")
        if not isinstance(tx_data, dict):
            raise RPCError(RPCErrorCode.INVALID_PARAMS, "transaction must be an object")

        from .chain import Transaction
        tx = Transaction(
            sender=tx_data.get("from", tx_data.get("sender", "")),
            recipient=tx_data.get("to", tx_data.get("recipient", "")),
            value=_parse_hex_int(tx_data.get("value", 0)),
            data=tx_data.get("data", ""),
            nonce=_parse_hex_int(tx_data.get("nonce", 0)),
            gas_limit=_parse_hex_int(tx_data.get("gas", tx_data.get("gasLimit", 21000))),
            gas_price=_parse_hex_int(tx_data.get("gasPrice", tx_data.get("gas_price", 1))),
            timestamp=time.time(),
        )
        tx.compute_hash()
        if not tx.signature:
            tx.signature = "rpc_dev_" + tx.tx_hash[:56]

        result = self.node.mempool.replace_by_fee(tx)
        if not result["replaced"]:
            raise RPCError(RPCErrorCode.TX_REJECTED, result.get("error", "RBF failed"))
        return {
            "new_tx_hash": tx.tx_hash,
            "old_tx_hash": result["old_hash"],
            "gas_price": tx.gas_price,
        }

    # ══════════════════════════════════════════════════════════════════
    #  net_* handlers
    # ══════════════════════════════════════════════════════════════════

    def _net_version(self, params) -> str:
        return self.node.config.chain_id

    def _net_peer_count(self, params) -> str:
        return _hex_int(self.node.network.peer_count)

    def _net_listening(self, params) -> bool:
        return self.node.network.is_running

    def _net_peers(self, params) -> List[Dict]:
        info = self.node.network.get_network_info()
        return info.get("peers", [])

    # ══════════════════════════════════════════════════════════════════
    #  miner_* handlers
    # ══════════════════════════════════════════════════════════════════

    def _miner_start(self, params) -> bool:
        try:
            self.node.start_mining()
            return True
        except Exception as e:
            raise RPCError(RPCErrorCode.MINING_ERROR, str(e))

    def _miner_stop(self, params) -> bool:
        self.node.stop_mining()
        return True

    def _miner_status(self, params) -> Dict:
        return {
            "mining": self.node._mining,
            "miner_address": self.node.config.miner_address,
            "consensus": self.node.config.consensus,
            "height": self.node.chain.height,
            "mempool_size": self.node.mempool.size,
        }

    def _miner_set_address(self, params) -> bool:
        address = _require_param(params, 0, "address")
        self.node.config.miner_address = address
        return True

    def _miner_mine_block(self, params) -> Optional[Dict]:
        """Mine a single block synchronously."""
        block = self.node.mine_block_sync()
        if block is None:
            return None
        return block.to_dict()

    # ══════════════════════════════════════════════════════════════════
    #  contract_* handlers
    # ══════════════════════════════════════════════════════════════════

    def _contract_deploy(self, params) -> Dict:
        """Deploy a contract.
        
        Expects: {code, deployer, gas_limit?, initial_value?}
        """
        data = _require_param(params, 0, "deploy_params")
        if not isinstance(data, dict):
            raise RPCError(RPCErrorCode.INVALID_PARAMS, "deploy_params must be an object")

        code = _require_param(data, "code", "code")
        deployer = _require_param(data, "deployer", "deployer")
        gas_limit = data.get("gas_limit", data.get("gasLimit", 10_000_000))
        initial_value = data.get("initial_value", data.get("initialValue", 0))

        if self.node.contract_vm is None:
            raise RPCError(RPCErrorCode.CONTRACT_ERROR, "ContractVM not available")

        # Create a minimal contract object for the VM
        from .contract_vm import ContractVM
        import hashlib as _hl
        contract_address = "0x" + _hl.sha256(
            f"{deployer}:{code}:{time.time()}".encode()
        ).hexdigest()[:40]

        # Store the code in chain's contract state
        self.node.chain.contract_state[contract_address] = {
            "code": code,
            "deployer": deployer,
            "storage": {},
        }

        return {
            "success": True,
            "address": contract_address,
            "deployer": deployer,
            "gas_used": 0,
        }

    def _contract_call(self, params) -> Dict:
        """Execute a contract action (state-mutating)."""
        data = _require_param(params, 0, "call_params")
        if not isinstance(data, dict):
            raise RPCError(RPCErrorCode.INVALID_PARAMS, "call_params must be an object")

        address = _require_param(data, "address", "contract address")
        action = _require_param(data, "action", "action name")
        args = data.get("args", {})
        caller = data.get("caller", data.get("from", ""))
        gas_limit = data.get("gas_limit", data.get("gasLimit", 500_000))
        value = data.get("value", 0)

        result = self.node.call_contract(
            contract_address=address,
            action=action,
            args=args,
            caller=caller,
            gas_limit=gas_limit,
            value=value,
        )
        if not result.get("success"):
            raise RPCError(RPCErrorCode.CONTRACT_ERROR, result.get("error", "contract call failed"))
        return result

    def _contract_static_call(self, params) -> Dict:
        """Execute a read-only contract call."""
        data = _require_param(params, 0, "call_params")
        if not isinstance(data, dict):
            raise RPCError(RPCErrorCode.INVALID_PARAMS, "call_params must be an object")

        address = _require_param(data, "address", "contract address")
        action = _require_param(data, "action", "action name")
        args = data.get("args", {})
        caller = data.get("caller", data.get("from", ""))

        result = self.node.static_call_contract(
            contract_address=address,
            action=action,
            args=args,
            caller=caller,
        )
        if not result.get("success"):
            raise RPCError(RPCErrorCode.CONTRACT_ERROR, result.get("error", "static call failed"))
        return result

    # ══════════════════════════════════════════════════════════════════
    #  admin_* handlers
    # ══════════════════════════════════════════════════════════════════

    def _admin_node_info(self, params) -> Dict:
        return {
            "chain_id": self.node.config.chain_id,
            "host": self.node.config.host,
            "port": self.node.config.port,
            "rpc_port": self.node.config.rpc_port,
            "consensus": self.node.config.consensus,
            "mining": self.node._mining,
            "miner_address": self.node.config.miner_address,
            "height": self.node.chain.height,
            "peers": self.node.network.peer_count,
            "mempool_size": self.node.mempool.size,
            "contract_vm_available": self.node.contract_vm is not None,
            "rpc_methods": self.registry.list_methods(),
            "subscriptions": self.subscriptions.count,
        }

    def _admin_fund_account(self, params) -> Dict:
        """Fund an account (for testnets/devnets only)."""
        address = _require_param(params, 0, "address")
        amount = _parse_hex_int(_require_param(params, 1, "amount"))
        self.node.fund_account(address, amount)
        new_balance = self.node.get_balance(address)
        return {"address": address, "funded": amount, "new_balance": new_balance}

    def _admin_export_chain(self, params) -> List[Dict]:
        return self.node.export_chain()

    def _admin_rpc_methods(self, params) -> List[str]:
        return self.registry.list_methods()
