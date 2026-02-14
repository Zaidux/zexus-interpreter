"""HTTP module for Zexus standard library.

Provides sync + async HTTP client operations with connection pooling.
"""

import urllib.request
import urllib.parse
import urllib.error
import json as json_lib
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, Future
import http.client
import threading


# ---------------------------------------------------------------------------
# Connection pool for keep-alive reuse (reduces TCP handshake overhead)
# ---------------------------------------------------------------------------

class _ConnectionPool:
    """Simple per-host HTTP/HTTPS keep-alive connection pool."""

    _MAX_PER_HOST = 4

    def __init__(self):
        self._lock = threading.Lock()
        # key: (scheme, host, port) -> list[http.client.HTTP(S)Connection]
        self._pool: Dict[tuple, list] = {}

    def _key(self, url: str):
        from urllib.parse import urlparse
        p = urlparse(url)
        scheme = p.scheme or "http"
        host = p.hostname or "localhost"
        port = p.port or (443 if scheme == "https" else 80)
        path = p.path or "/"
        if p.query:
            path += "?" + p.query
        return (scheme, host, port), path

    def get_connection(self, url: str):
        """Return (conn, path, key) — reuses pooled conn or creates new."""
        key, path = self._key(url)
        scheme, host, port = key
        with self._lock:
            conns = self._pool.get(key, [])
            if conns:
                conn = conns.pop()
                return conn, path, key
        # Create new connection
        if scheme == "https":
            import ssl
            ctx = ssl.create_default_context()
            conn = http.client.HTTPSConnection(host, port, timeout=30, context=ctx)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=30)
        return conn, path, key

    def return_connection(self, key, conn):
        """Return a connection to the pool for reuse."""
        with self._lock:
            conns = self._pool.setdefault(key, [])
            if len(conns) < self._MAX_PER_HOST:
                conns.append(conn)
            else:
                try:
                    conn.close()
                except Exception:
                    pass

    def close_all(self):
        """Drain and close every pooled connection."""
        with self._lock:
            for conns in self._pool.values():
                for c in conns:
                    try:
                        c.close()
                    except Exception:
                        pass
            self._pool.clear()


# Module-level singleton pool
_pool = _ConnectionPool()

# Shared thread pool for async HTTP requests
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="zexus-http")


def _do_pooled_request(method: str, url: str, data: bytes = None,
                       headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
    """Execute an HTTP request using the connection pool."""
    conn, path, key = _pool.get_connection(url)
    reuse = True
    try:
        conn.timeout = timeout
        conn.request(method, path, body=data, headers=headers or {})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        result = {
            "status": resp.status,
            "headers": dict(resp.getheaders()),
            "body": body,
        }
        return result
    except Exception:
        reuse = False
        # Fall back to urllib for robustness
        return _do_urllib_request(method, url, data, headers, timeout)
    finally:
        if reuse:
            _pool.return_connection(key, conn)
        else:
            try:
                conn.close()
            except Exception:
                pass


def _do_urllib_request(method: str, url: str, data: bytes = None,
                      headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
    """Fallback: single-use urllib request."""
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "status": response.status,
                "headers": dict(response.headers),
                "body": body,
            }
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        return {
            "status": e.code,
            "headers": dict(e.headers),
            "body": error_body,
            "error": str(e),
        }
    except Exception as e:
        return {"status": 0, "headers": {}, "body": "", "error": str(e)}


class HttpModule:
    """Provides HTTP client operations with connection pooling and async support."""

    @staticmethod
    def _prepare_body(data: Any, headers: Dict[str, str], json: bool = False) -> bytes:
        """Encode request body and set Content-Type when necessary."""
        if data is None:
            return None
        if json:
            headers.setdefault("Content-Type", "application/json")
            return json_lib.dumps(data).encode("utf-8")
        if isinstance(data, str):
            return data.encode("utf-8")
        if isinstance(data, dict):
            return urllib.parse.urlencode(data).encode("utf-8")
        if isinstance(data, bytes):
            return data
        return str(data).encode("utf-8")

    # ------------------------------------------------------------------
    # Synchronous (pooled) API
    # ------------------------------------------------------------------

    @staticmethod
    def get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP GET request (connection-pooled)."""
        return _do_pooled_request("GET", url, headers=headers or {}, timeout=timeout)

    @staticmethod
    def post(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None, 
             json: bool = False, timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP POST request (connection-pooled)."""
        hdrs = dict(headers or {})
        body = HttpModule._prepare_body(data, hdrs, json)
        return _do_pooled_request("POST", url, data=body, headers=hdrs, timeout=timeout)

    @staticmethod
    def put(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
            json: bool = False, timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP PUT request (connection-pooled)."""
        hdrs = dict(headers or {})
        body = HttpModule._prepare_body(data, hdrs, json)
        return _do_pooled_request("PUT", url, data=body, headers=hdrs, timeout=timeout)

    @staticmethod
    def delete(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP DELETE request (connection-pooled)."""
        return _do_pooled_request("DELETE", url, headers=headers or {}, timeout=timeout)

    @staticmethod
    def request(method: str, url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
                timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP request with custom method (connection-pooled)."""
        hdrs = dict(headers or {})
        body = HttpModule._prepare_body(data, hdrs)
        return _do_pooled_request(method.upper(), url, data=body, headers=hdrs, timeout=timeout)

    # ------------------------------------------------------------------
    # Async (non-blocking) API — returns Future objects
    # ------------------------------------------------------------------

    @staticmethod
    def async_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Future:
        """Non-blocking HTTP GET — returns a concurrent.futures.Future."""
        return _executor.submit(HttpModule.get, url, headers, timeout)

    @staticmethod
    def async_post(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
                   json: bool = False, timeout: int = 30) -> Future:
        """Non-blocking HTTP POST — returns a concurrent.futures.Future."""
        return _executor.submit(HttpModule.post, url, data, headers, json, timeout)

    @staticmethod
    def async_put(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
                  json: bool = False, timeout: int = 30) -> Future:
        """Non-blocking HTTP PUT — returns a concurrent.futures.Future."""
        return _executor.submit(HttpModule.put, url, data, headers, json, timeout)

    @staticmethod
    def async_delete(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Future:
        """Non-blocking HTTP DELETE — returns a concurrent.futures.Future."""
        return _executor.submit(HttpModule.delete, url, headers, timeout)

    @staticmethod
    def async_request(method: str, url: str, data: Any = None,
                      headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Future:
        """Non-blocking HTTP request — returns a concurrent.futures.Future."""
        return _executor.submit(HttpModule.request, method, url, data, headers, timeout)

    @staticmethod
    def parallel_get(urls: List[str], headers: Optional[Dict[str, str]] = None,
                     timeout: int = 30) -> List[Dict[str, Any]]:
        """Execute multiple GET requests in parallel and return results in order."""
        futures = [_executor.submit(HttpModule.get, u, headers, timeout) for u in urls]
        return [f.result() for f in futures]

    @staticmethod
    def close_pool():
        """Close all pooled connections (call at shutdown)."""
        _pool.close_all()


# Export functions for easy access
get = HttpModule.get
post = HttpModule.post
put = HttpModule.put
delete = HttpModule.delete
request = HttpModule.request
