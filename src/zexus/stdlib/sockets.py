"""Socket/TCP primitives module for Zexus standard library."""

import socket
import threading
import time
from typing import Callable, Optional, Dict, Any


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
    """TCP server that accepts connections and handles them with a callback."""
    
    def __init__(self, host: str, port: int, handler: Callable, backlog: int = 5):
        self.host = host
        self.port = port
        self.handler = handler
        self.backlog = backlog
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the server in a background thread."""
        if self.running:
            raise RuntimeError("Server is already running")
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(self.backlog)
        self.running = True
        
        # Start accept loop in background thread
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
    
    def _accept_loop(self):
        """Accept connections and spawn handler threads."""
        while self.running:
            try:
                self.socket.settimeout(1.0)  # Allow checking self.running
                client_socket, address = self.socket.accept()
                
                # Spawn handler in new thread
                handler_thread = threading.Thread(
                    target=self._handle_connection,
                    args=(client_socket, address),
                    daemon=True
                )
                handler_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:  # Only log if we're not shutting down
                    print(f"Server accept error: {e}")
                break
    
    def _handle_connection(self, client_socket: socket.socket, address: tuple):
        """Handle a single client connection."""
        try:
            connection = TCPConnection.from_socket(client_socket, address)
            self.handler(connection)
        except Exception as e:
            print(f"Connection handler error: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def stop(self) -> None:
        """Stop the server."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def is_running(self) -> bool:
        """Check if server is running."""
        return self.running
    
    def get_address(self) -> Dict[str, Any]:
        """Get server address info."""
        return {
            'host': self.host,
            'port': self.port,
            'running': self.running
        }


class TCPConnection:
    """Represents a TCP connection (client or server-side)."""
    
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        """Create a new client connection."""
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.socket.connect((host, port))
        self.connected = True
    
    @classmethod
    def from_socket(cls, sock: socket.socket, address: tuple):
        """Create TCPConnection from existing socket (for server-side)."""
        conn = cls.__new__(cls)
        conn.socket = sock
        conn.host = address[0]
        conn.port = address[1]
        conn.connected = True
        return conn
    
    def send(self, data: bytes) -> int:
        """Send data over the connection.
        
        Args:
            data: Bytes to send
            
        Returns:
            Number of bytes sent
        """
        if not self.connected:
            raise RuntimeError("Connection is closed")
        return self.socket.sendall(data) or len(data)
    
    def send_string(self, text: str, encoding: str = 'utf-8') -> int:
        """Send string over the connection.
        
        Args:
            text: String to send
            encoding: Text encoding
            
        Returns:
            Number of bytes sent
        """
        return self.send(text.encode(encoding))
    
    def receive(self, buffer_size: int = 4096) -> bytes:
        """Receive data from the connection.
        
        Args:
            buffer_size: Maximum bytes to receive
            
        Returns:
            Received bytes (empty if connection closed)
        """
        if not self.connected:
            raise RuntimeError("Connection is closed")
        
        try:
            data = self.socket.recv(buffer_size)
            if not data:
                self.connected = False
            return data
        except socket.timeout:
            return b''
    
    def receive_string(self, buffer_size: int = 4096, encoding: str = 'utf-8') -> str:
        """Receive string from the connection.
        
        Args:
            buffer_size: Maximum bytes to receive
            encoding: Text encoding
            
        Returns:
            Received string
        """
        data = self.receive(buffer_size)
        return data.decode(encoding) if data else ''
    
    def receive_all(self, timeout: float = 5.0) -> bytes:
        """Receive all available data until connection closes or timeout.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            All received bytes
        """
        chunks = []
        start_time = time.time()
        self.socket.settimeout(0.1)  # Small timeout for checking
        
        while time.time() - start_time < timeout:
            try:
                chunk = self.socket.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            except socket.timeout:
                if chunks:  # If we got some data, we're done
                    break
                continue
        
        return b''.join(chunks)
    
    def close(self) -> None:
        """Close the connection."""
        if self.connected:
            try:
                self.socket.close()
            except:
                pass
            self.connected = False
    
    def is_connected(self) -> bool:
        """Check if connection is still open."""
        return self.connected
    
    def get_address(self) -> Dict[str, Any]:
        """Get connection address info."""
        return {
            'host': self.host,
            'port': self.port,
            'connected': self.connected
        }
