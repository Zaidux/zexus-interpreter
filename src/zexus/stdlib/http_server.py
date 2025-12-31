"""HTTP Server module for Zexus standard library.
Built on top of the socket primitives."""

import socket
import threading
import re
from typing import Dict, List, Callable, Optional, Any, Tuple
from urllib.parse import parse_qs, urlparse


class HTTPRequest:
    """Represents an HTTP request."""
    
    def __init__(self, method: str, path: str, headers: Dict[str, str], 
                 body: str, query: Dict[str, List[str]]):
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body
        self.query = query
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Zexus."""
        return {
            'method': self.method,
            'path': self.path,
            'headers': self.headers,
            'body': self.body,
            'query': self.query
        }


class HTTPResponse:
    """Represents an HTTP response."""
    
    def __init__(self):
        self.status = 200
        self.headers: Dict[str, str] = {'Content-Type': 'text/plain'}
        self.body = ''
    
    def set_status(self, code: int) -> 'HTTPResponse':
        """Set response status code."""
        self.status = code
        return self
    
    def set_header(self, name: str, value: str) -> 'HTTPResponse':
        """Set response header."""
        self.headers[name] = value
        return self
    
    def send(self, body: str) -> 'HTTPResponse':
        """Set response body."""
        self.body = body
        return self
    
    def json(self, data: Any) -> 'HTTPResponse':
        """Send JSON response."""
        import json
        self.headers['Content-Type'] = 'application/json'
        self.body = json.dumps(data)
        return self
    
    def build(self) -> str:
        """Build HTTP response string."""
        status_messages = {
            200: 'OK',
            201: 'Created',
            400: 'Bad Request',
            404: 'Not Found',
            500: 'Internal Server Error'
        }
        
        status_msg = status_messages.get(self.status, 'OK')
        response = f"HTTP/1.1 {self.status} {status_msg}\r\n"
        
        # Add Content-Length
        self.headers['Content-Length'] = str(len(self.body.encode('utf-8')))
        
        # Add headers
        for name, value in self.headers.items():
            response += f"{name}: {value}\r\n"
        
        response += "\r\n"
        response += self.body
        
        return response


class HTTPServer:
    """Simple HTTP server with routing."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port
        self.routes: Dict[Tuple[str, str], Callable] = {}  # (method, path) -> handler
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
    
    def route(self, method: str, path: str, handler: Callable):
        """Register a route handler."""
        self.routes[(method.upper(), path)] = handler
    
    def get(self, path: str, handler: Callable):
        """Register GET route."""
        self.route('GET', path, handler)
    
    def post(self, path: str, handler: Callable):
        """Register POST route."""
        self.route('POST', path, handler)
    
    def put(self, path: str, handler: Callable):
        """Register PUT route."""
        self.route('PUT', path, handler)
    
    def delete(self, path: str, handler: Callable):
        """Register DELETE route."""
        self.route('DELETE', path, handler)
    
    def _parse_request(self, raw_request: str) -> Optional[HTTPRequest]:
        """Parse raw HTTP request."""
        try:
            lines = raw_request.split('\r\n')
            if not lines:
                return None
            
            # Parse request line
            request_line = lines[0]
            parts = request_line.split(' ')
            if len(parts) < 3:
                return None
            
            method, full_path, _ = parts
            
            # Parse path and query string
            parsed = urlparse(full_path)
            path = parsed.path
            query = parse_qs(parsed.query)
            
            # Parse headers
            headers = {}
            body_start = 0
            for i, line in enumerate(lines[1:], 1):
                if line == '':
                    body_start = i + 1
                    break
                if ': ' in line:
                    name, value = line.split(': ', 1)
                    headers[name] = value
            
            # Parse body
            body = '\r\n'.join(lines[body_start:]) if body_start > 0 else ''
            
            return HTTPRequest(method, path, headers, body, query)
        
        except Exception as e:
            print(f"Request parse error: {e}")
            return None
    
    def _handle_connection(self, client_socket: socket.socket, address: tuple):
        """Handle a client connection."""
        try:
            # Receive request
            raw_request = client_socket.recv(8192).decode('utf-8')
            
            if not raw_request:
                return
            
            # Parse request
            request = self._parse_request(raw_request)
            if not request:
                response = HTTPResponse().set_status(400).send("Bad Request")
                client_socket.sendall(response.build().encode('utf-8'))
                return
            
            # Find matching route
            route_key = (request.method, request.path)
            handler = self.routes.get(route_key)
            
            if not handler:
                response = HTTPResponse().set_status(404).send("Not Found")
                client_socket.sendall(response.build().encode('utf-8'))
                return
            
            # Create response object
            response = HTTPResponse()
            
            # Call handler
            try:
                handler(request, response)
            except Exception as e:
                print(f"Handler error: {e}")
                response = HTTPResponse().set_status(500).send(f"Internal Server Error: {str(e)}")
            
            # Send response
            client_socket.sendall(response.build().encode('utf-8'))
        
        except Exception as e:
            print(f"Connection handler error: {e}")
        
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def listen(self):
        """Start the HTTP server."""
        if self.running:
            raise RuntimeError("Server is already running")
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(10)
        self.running = True
        
        print(f"HTTP Server listening on {self.host}:{self.port}")
        
        # Accept connections
        while self.running:
            try:
                self.socket.settimeout(1.0)
                client_socket, address = self.socket.accept()
                
                # Handle in new thread
                handler_thread = threading.Thread(
                    target=self._handle_connection,
                    args=(client_socket, address),
                    daemon=True
                )
                handler_thread.start()
            
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Accept error: {e}")
                break
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
