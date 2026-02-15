"""HTTP module for Zexus standard library.

Provides sync + async HTTP client operations.

Primary backend:  **httpx** (true async, connection pooling, HTTP/2 ready)
Fallback backend: ``urllib.request`` (when httpx is not installed)
"""

import json as json_lib
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, Future

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

import urllib.request
import urllib.parse
import urllib.error


# ---------------------------------------------------------------------------
# Shared clients (lazy singletons)
# ---------------------------------------------------------------------------
_sync_client: Optional[Any] = None
_async_client: Optional[Any] = None
_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="zexus-http")


def _get_sync_client():
    """Return (or create) the module-level httpx.Client with pooling."""
    global _sync_client
    if _sync_client is None and _HTTPX_AVAILABLE:
        _sync_client = _httpx.Client(
            follow_redirects=True,
            timeout=_httpx.Timeout(30.0, connect=10.0),
            limits=_httpx.Limits(max_connections=20, max_keepalive_connections=8),
        )
    return _sync_client


def _get_async_client():
    """Return (or create) the module-level httpx.AsyncClient."""
    global _async_client
    if _async_client is None and _HTTPX_AVAILABLE:
        _async_client = _httpx.AsyncClient(
            follow_redirects=True,
            timeout=_httpx.Timeout(30.0, connect=10.0),
            limits=_httpx.Limits(max_connections=20, max_keepalive_connections=8),
        )
    return _async_client


# ---------------------------------------------------------------------------
# Core request functions
# ---------------------------------------------------------------------------

def _httpx_request(method: str, url: str, data: bytes = None,
                   headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
    """Execute an HTTP request via httpx (pooled, keep-alive)."""
    client = _get_sync_client()
    try:
        resp = client.request(
            method, url,
            content=data,
            headers=headers or {},
            timeout=timeout,
        )
        return {
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp.text,
        }
    except Exception as exc:
        return {"status": 0, "headers": {}, "body": "", "error": str(exc)}


def _urllib_request(method: str, url: str, data: bytes = None,
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


def _do_request(method: str, url: str, data: bytes = None,
                headers: Dict[str, str] = None, timeout: int = 30) -> Dict[str, Any]:
    """Primary dispatcher — httpx if available, else urllib."""
    if _HTTPX_AVAILABLE:
        return _httpx_request(method, url, data, headers, timeout)
    return _urllib_request(method, url, data, headers, timeout)


# ---------------------------------------------------------------------------
# Async request via background event loop (true async when httpx available)
# ---------------------------------------------------------------------------

async def _httpx_async_request(method: str, url: str, data: bytes = None,
                               headers: Dict[str, str] = None,
                               timeout: int = 30) -> Dict[str, Any]:
    """True async HTTP request via httpx.AsyncClient."""
    client = _get_async_client()
    try:
        resp = await client.request(
            method, url,
            content=data,
            headers=headers or {},
            timeout=timeout,
        )
        return {
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp.text,
        }
    except Exception as exc:
        return {"status": 0, "headers": {}, "body": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# HttpModule (public API — unchanged signatures)
# ---------------------------------------------------------------------------

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
    # Synchronous API
    # ------------------------------------------------------------------

    @staticmethod
    def get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict[str, Any]:
        return _do_request("GET", url, headers=headers or {}, timeout=timeout)

    @staticmethod
    def post(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
             json: bool = False, timeout: int = 30) -> Dict[str, Any]:
        hdrs = dict(headers or {})
        body = HttpModule._prepare_body(data, hdrs, json)
        return _do_request("POST", url, data=body, headers=hdrs, timeout=timeout)

    @staticmethod
    def put(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
            json: bool = False, timeout: int = 30) -> Dict[str, Any]:
        hdrs = dict(headers or {})
        body = HttpModule._prepare_body(data, hdrs, json)
        return _do_request("PUT", url, data=body, headers=hdrs, timeout=timeout)

    @staticmethod
    def delete(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict[str, Any]:
        return _do_request("DELETE", url, headers=headers or {}, timeout=timeout)

    @staticmethod
    def request(method: str, url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
                timeout: int = 30) -> Dict[str, Any]:
        hdrs = dict(headers or {})
        body = HttpModule._prepare_body(data, hdrs)
        return _do_request(method.upper(), url, data=body, headers=hdrs, timeout=timeout)

    # ------------------------------------------------------------------
    # Async API — returns Future objects (compatible with existing callers)
    # ------------------------------------------------------------------

    @staticmethod
    def async_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Future:
        if _HTTPX_AVAILABLE:
            from .sockets import _get_bg_loop, _run_async
            loop = _get_bg_loop()
            import asyncio
            return asyncio.run_coroutine_threadsafe(
                _httpx_async_request("GET", url, headers=headers or {}, timeout=timeout), loop
            )
        return _executor.submit(HttpModule.get, url, headers, timeout)

    @staticmethod
    def async_post(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
                   json: bool = False, timeout: int = 30) -> Future:
        if _HTTPX_AVAILABLE:
            from .sockets import _get_bg_loop
            import asyncio
            hdrs = dict(headers or {})
            body = HttpModule._prepare_body(data, hdrs, json)
            return asyncio.run_coroutine_threadsafe(
                _httpx_async_request("POST", url, data=body, headers=hdrs, timeout=timeout),
                _get_bg_loop()
            )
        return _executor.submit(HttpModule.post, url, data, headers, json, timeout)

    @staticmethod
    def async_put(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
                  json: bool = False, timeout: int = 30) -> Future:
        if _HTTPX_AVAILABLE:
            from .sockets import _get_bg_loop
            import asyncio
            hdrs = dict(headers or {})
            body = HttpModule._prepare_body(data, hdrs, json)
            return asyncio.run_coroutine_threadsafe(
                _httpx_async_request("PUT", url, data=body, headers=hdrs, timeout=timeout),
                _get_bg_loop()
            )
        return _executor.submit(HttpModule.put, url, data, headers, json, timeout)

    @staticmethod
    def async_delete(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Future:
        if _HTTPX_AVAILABLE:
            from .sockets import _get_bg_loop
            import asyncio
            return asyncio.run_coroutine_threadsafe(
                _httpx_async_request("DELETE", url, headers=headers or {}, timeout=timeout),
                _get_bg_loop()
            )
        return _executor.submit(HttpModule.delete, url, headers, timeout)

    @staticmethod
    def async_request(method: str, url: str, data: Any = None,
                      headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Future:
        if _HTTPX_AVAILABLE:
            from .sockets import _get_bg_loop
            import asyncio
            hdrs = dict(headers or {})
            body = HttpModule._prepare_body(data, hdrs)
            return asyncio.run_coroutine_threadsafe(
                _httpx_async_request(method.upper(), url, data=body, headers=hdrs, timeout=timeout),
                _get_bg_loop()
            )
        return _executor.submit(HttpModule.request, method, url, data, headers, timeout)

    @staticmethod
    def parallel_get(urls: List[str], headers: Optional[Dict[str, str]] = None,
                     timeout: int = 30) -> List[Dict[str, Any]]:
        """Execute multiple GET requests in parallel."""
        if _HTTPX_AVAILABLE:
            import asyncio
            from .sockets import _get_bg_loop

            async def _batch():
                cl = _get_async_client()
                tasks = [
                    cl.request("GET", u, headers=headers or {}, timeout=timeout)
                    for u in urls
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                out = []
                for r in results:
                    if isinstance(r, Exception):
                        out.append({"status": 0, "headers": {}, "body": "", "error": str(r)})
                    else:
                        out.append({"status": r.status_code, "headers": dict(r.headers), "body": r.text})
                return out

            return asyncio.run_coroutine_threadsafe(_batch(), _get_bg_loop()).result()

        futures = [_executor.submit(HttpModule.get, u, headers, timeout) for u in urls]
        return [f.result() for f in futures]

    @staticmethod
    def close_pool():
        """Close pooled connections (call at shutdown)."""
        global _sync_client, _async_client
        if _sync_client:
            try:
                _sync_client.close()
            except Exception:
                pass
            _sync_client = None
        if _async_client:
            try:
                import asyncio
                from .sockets import _get_bg_loop
                async def _aclose():
                    await _async_client.aclose()
                asyncio.run_coroutine_threadsafe(_aclose(), _get_bg_loop()).result(timeout=5)
            except Exception:
                pass
            _async_client = None


# Export functions for easy access
get = HttpModule.get
post = HttpModule.post
put = HttpModule.put
delete = HttpModule.delete
request = HttpModule.request
