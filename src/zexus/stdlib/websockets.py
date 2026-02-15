"""WebSocket client & server module for Zexus standard library.

Provides ``WebSocketServer`` and ``WebSocketClient`` backed by the
``websockets`` library running on the shared asyncio background loop
(same one used by the async TCP sockets module).
"""

import asyncio
import threading
from typing import Callable, Optional, Dict, Any, List

# Re-use the background event loop from sockets module
from .sockets import _get_bg_loop, _run_async


# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import websockets
    import websockets.server
    import websockets.client
    _WS_AVAILABLE = True
except ImportError:
    _WS_AVAILABLE = False


def _require_ws():
    if not _WS_AVAILABLE:
        raise RuntimeError(
            "WebSocket support requires the 'websockets' package. "
            "Install with: pip install websockets"
        )


# ---------------------------------------------------------------------------
# WebSocket Module (factory)
# ---------------------------------------------------------------------------

class WebSocketModule:
    """Factory for WebSocket servers and clients."""

    @staticmethod
    def create_server(host: str, port: int, handler: Callable,
                      path: Optional[str] = None) -> 'WebSocketServer':
        """Create a WebSocket server.

        Args:
            host: Host address to bind (e.g. '0.0.0.0').
            port: Port number.
            handler: Callback ``action(ws)`` invoked for each connection.
                     ``ws`` is a *WebSocketConnection* with ``send`` /
                     ``receive`` / ``close`` methods.
            path: Optional URL path filter (not enforced; for documentation).

        Returns:
            WebSocketServer instance (call ``.start()`` to begin listening).
        """
        _require_ws()
        return WebSocketServer(host, port, handler, path)

    @staticmethod
    def connect(url: str, timeout: float = 10.0) -> 'WebSocketClient':
        """Open a WebSocket client connection.

        Args:
            url: WebSocket URL, e.g. ``ws://localhost:8080/path``.
            timeout: Connection timeout in seconds.

        Returns:
            WebSocketClient with ``send`` / ``receive`` / ``close``.
        """
        _require_ws()
        return WebSocketClient(url, timeout)


# ---------------------------------------------------------------------------
# WebSocket Server
# ---------------------------------------------------------------------------

class WebSocketServer:
    """WebSocket server backed by ``websockets.serve``."""

    def __init__(self, host: str, port: int, handler: Callable,
                 path: Optional[str] = None):
        _require_ws()
        self.host = host
        self.port = port
        self.handler = handler
        self.path = path
        self.running = False
        self._server = None
        self._loop = _get_bg_loop()

    def start(self) -> None:
        if self.running:
            raise RuntimeError("WebSocket server is already running")
        _run_async(self._async_start())

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self._server:
            asyncio.run_coroutine_threadsafe(
                self._async_stop(), self._loop
            ).result(timeout=5)

    def is_running(self) -> bool:
        return self.running

    def get_address(self) -> Dict[str, Any]:
        return {
            'host': self.host,
            'port': self.port,
            'running': self.running,
            'path': self.path or '/',
        }

    # -- async internals ---------------------------------------------------

    async def _async_start(self):
        self._server = await websockets.server.serve(
            self._async_handle,
            self.host,
            self.port,
        )
        self.running = True

    async def _async_stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self.running = False

    async def _async_handle(self, ws):
        """Per-connection handler coroutine."""
        conn = WebSocketConnection(ws)
        loop = asyncio.get_running_loop()
        try:
            # Run sync handler in thread executor to avoid deadlock
            await loop.run_in_executor(None, self.handler, conn)
        except Exception as e:
            print(f"WebSocket handler error: {e}")
        finally:
            conn._closed = True


# ---------------------------------------------------------------------------
# WebSocket Client
# ---------------------------------------------------------------------------

class WebSocketClient:
    """WebSocket client connection."""

    def __init__(self, url: str, timeout: float = 10.0):
        _require_ws()
        self.url = url
        self._ws = None
        self._closed = False

        async def _connect():
            return await asyncio.wait_for(
                websockets.client.connect(url), timeout=timeout
            )

        self._ws = _run_async(_connect())

    def send(self, message: str) -> None:
        """Send a text message."""
        if self._closed:
            raise RuntimeError("WebSocket is closed")

        async def _send():
            await self._ws.send(message)

        _run_async(_send())

    def send_bytes(self, data: bytes) -> None:
        """Send binary data."""
        if self._closed:
            raise RuntimeError("WebSocket is closed")

        async def _send():
            await self._ws.send(data)

        _run_async(_send())

    def receive(self, timeout: float = 30.0) -> str:
        """Receive a text message."""
        if self._closed:
            raise RuntimeError("WebSocket is closed")

        async def _recv():
            return await asyncio.wait_for(self._ws.recv(), timeout=timeout)

        return _run_async(_recv())

    def receive_bytes(self, timeout: float = 30.0) -> bytes:
        """Receive binary data."""
        return self.receive(timeout)  # websockets handles both

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            async def _close():
                await self._ws.close()
            _run_async(_close(), timeout=5)
        except Exception:
            pass

    def is_connected(self) -> bool:
        return not self._closed and self._ws is not None

    def get_address(self) -> Dict[str, Any]:
        return {'url': self.url, 'connected': self.is_connected()}


# ---------------------------------------------------------------------------
# WebSocket Connection (server-side wrapper)
# ---------------------------------------------------------------------------

class WebSocketConnection:
    """Wraps a server-side websocket for use in synchronous Zexus handlers."""

    def __init__(self, ws):
        self._ws = ws
        self._closed = False

    def send(self, message: str) -> None:
        """Send a text message to the client."""
        if self._closed:
            raise RuntimeError("WebSocket is closed")

        async def _send():
            await self._ws.send(message)

        _run_async(_send())

    def send_bytes(self, data: bytes) -> None:
        """Send binary data to the client."""
        if self._closed:
            raise RuntimeError("WebSocket is closed")

        async def _send():
            await self._ws.send(data)

        _run_async(_send())

    def receive(self, timeout: float = 30.0) -> str:
        """Receive a message from the client."""
        if self._closed:
            raise RuntimeError("WebSocket is closed")

        async def _recv():
            return await asyncio.wait_for(self._ws.recv(), timeout=timeout)

        return _run_async(_recv())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            async def _close():
                await self._ws.close()
            _run_async(_close(), timeout=5)
        except Exception:
            pass

    def is_connected(self) -> bool:
        return not self._closed

    def get_address(self) -> Dict[str, Any]:
        remote = getattr(self._ws, 'remote_address', None) or ('unknown', 0)
        return {
            'host': remote[0] if isinstance(remote, tuple) else str(remote),
            'port': remote[1] if isinstance(remote, tuple) and len(remote) > 1 else 0,
            'connected': self.is_connected(),
        }
