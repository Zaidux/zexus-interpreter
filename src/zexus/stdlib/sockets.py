"""Socket/TCP primitives module for Zexus standard library.

Uses ``asyncio`` for non-blocking I/O instead of one-thread-per-connection.
A background event-loop thread is shared across all sockets so callers that
aren't themselves running inside an asyncio loop get synchronous-looking
wrappers automatically.
"""

import asyncio
import socket
import threading
import time
from typing import Callable, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Shared background event loop (lazily created, one per interpreter)
# ---------------------------------------------------------------------------
_BG_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BG_LOOP_LOCK = threading.Lock()


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background asyncio event loop, starting it if needed."""
    global _BG_LOOP
    if _BG_LOOP is not None and _BG_LOOP.is_running():
        return _BG_LOOP
    with _BG_LOOP_LOCK:
        if _BG_LOOP is not None and _BG_LOOP.is_running():
            return _BG_LOOP
        loop = asyncio.new_event_loop()

        def _run(l: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(l)
            l.run_forever()

        t = threading.Thread(target=_run, args=(loop,), daemon=True)
        t.start()
        _BG_LOOP = loop
        return loop


def _run_async(coro, timeout=10):
    """Submit *coro* to the background loop and block until it finishes."""
    loop = _get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


class SocketModule:
    """Provides socket and TCP operations."""
    
    @staticmethod
    def create_server(host: str, port: int, handler: Callable, backlog: int = 5) -> 'TCPServer':
        """Create a TCP server that listens for connections.
        
        Args:
            host: Host address to bind to (e.g., '0.0.0.0', 'localhost')
            port: Port number to listen on
            handler: Callback function called for each connection
            backlog: Maximum number of queued connections
            
        Returns:
            TCPServer instance
        """
        return TCPServer(host, port, handler, backlog)
    
    @staticmethod
    def create_connection(host: str, port: int, timeout: float = 5.0) -> 'TCPConnection':
        """Create a TCP client connection.
        
        Args:
            host: Remote host to connect to
            port: Remote port to connect to
            timeout: Connection timeout in seconds
            
        Returns:
            TCPConnection instance
        """
        return TCPConnection(host, port, timeout)


class TCPServer:
    """TCP server backed by ``asyncio.start_server``."""

    def __init__(self, host: str, port: int, handler: Callable, backlog: int = 5):
        self.host = host
        self.port = port
        self.handler = handler
        self.backlog = backlog
        self.running = False
        self._server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self) -> None:
        """Start the server on the background event loop."""
        if self.running:
            raise RuntimeError("Server is already running")
        self._loop = _get_bg_loop()
        _run_async(self._async_start())

    def stop(self) -> None:
        """Gracefully stop the server."""
        if not self.running:
            return
        self.running = False
        if self._server and self._loop:
            asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop).result(timeout=5)

    def is_running(self) -> bool:
        return self.running

    def get_address(self) -> Dict[str, Any]:
        return {'host': self.host, 'port': self.port, 'running': self.running}

    # -- async internals ----------------------------------------------------

    async def _async_start(self):
        server = await asyncio.start_server(
            self._async_handle,
            self.host,
            self.port,
            backlog=self.backlog,
            reuse_address=True,
        )
        self._server = server
        self.running = True

    async def _async_stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self.running = False

    async def _async_handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle a single incoming connection.
        
        The user-supplied handler is synchronous (Zexus Action), so we run it
        in a thread executor.  This avoids the send/receive → _run_async
        deadlock because the handler thread is NOT the event-loop thread.
        """
        addr = writer.get_extra_info('peername') or ('unknown', 0)
        conn = TCPConnection._from_streams(reader, writer, addr[0], addr[1])
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self.handler, conn)
        except Exception as e:
            print(f"Connection handler error: {e}")
        finally:
            # Ensure connection is marked closed (handler may have already closed it)
            conn.connected = False
            try:
                if not writer.is_closing():
                    writer.close()
            except Exception:
                pass


class TCPConnection:
    """TCP connection using asyncio streams.

    All public methods are **synchronous** (for Zexus evaluator compat)
    but internally schedule asyncio coroutines on the background loop.
    """

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        """Create a new client connection (blocking)."""
        self.host = host
        self.port = port
        self.connected = False
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._loop = _get_bg_loop()

        async def _open():
            return await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=timeout)

        self._reader, self._writer = _run_async(_open())
        self.connected = True

    @classmethod
    def _from_streams(cls, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                      host: str, port: int) -> 'TCPConnection':
        """Wrap existing asyncio streams (used by TCPServer for accepted conns)."""
        conn = cls.__new__(cls)
        conn.host = host
        conn.port = port
        conn._reader = reader
        conn._writer = writer
        conn._loop = _get_bg_loop()
        conn.connected = True
        return conn

    @classmethod
    def from_socket(cls, sock: socket.socket, address: tuple) -> 'TCPConnection':
        """Create from a raw socket (backward-compat shim)."""
        conn = cls.__new__(cls)
        conn.host = address[0]
        conn.port = address[1]
        conn._reader = None
        conn._writer = None
        conn._loop = _get_bg_loop()
        conn.connected = True
        conn._raw_sock = sock

        async def _wrap():
            return await asyncio.open_connection(sock=sock)

        try:
            conn._reader, conn._writer = _run_async(_wrap())
        except Exception:
            pass
        return conn

    # -- send ---------------------------------------------------------------

    def send(self, data: bytes) -> int:
        if not self.connected:
            raise RuntimeError("Connection is closed")

        async def _send():
            self._writer.write(data)
            await self._writer.drain()

        _run_async(_send())
        return len(data)

    def send_string(self, text: str, encoding: str = 'utf-8') -> int:
        return self.send(text.encode(encoding))

    # -- receive ------------------------------------------------------------

    def receive(self, buffer_size: int = 4096) -> bytes:
        if not self.connected:
            raise RuntimeError("Connection is closed")

        async def _recv():
            try:
                return await asyncio.wait_for(self._reader.read(buffer_size), timeout=5.0)
            except asyncio.TimeoutError:
                return b''

        data = _run_async(_recv())
        if not data:
            self.connected = False
        return data

    def receive_string(self, buffer_size: int = 4096, encoding: str = 'utf-8') -> str:
        data = self.receive(buffer_size)
        return data.decode(encoding) if data else ''

    def receive_all(self, timeout: float = 5.0) -> bytes:
        async def _recv_all():
            chunks = []
            try:
                while True:
                    chunk = await asyncio.wait_for(self._reader.read(4096), timeout=0.1)
                    if not chunk:
                        break
                    chunks.append(chunk)
            except asyncio.TimeoutError:
                pass
            return b''.join(chunks)

        return _run_async(_recv_all())

    # -- close --------------------------------------------------------------

    def close(self) -> None:
        if not self.connected:
            return
        self.connected = False
        if self._writer:
            try:
                async def _close():
                    if not self._writer.is_closing():
                        self._writer.close()
                        try:
                            await asyncio.wait_for(self._writer.wait_closed(), timeout=2.0)
                        except (asyncio.TimeoutError, Exception):
                            pass
                _run_async(_close(), timeout=3)
            except Exception:
                pass
        raw = getattr(self, '_raw_sock', None)
        if raw:
            try:
                raw.close()
            except Exception:
                pass

    def is_connected(self) -> bool:
        return self.connected

    def get_address(self) -> Dict[str, Any]:
        return {'host': self.host, 'port': self.port, 'connected': self.connected}
