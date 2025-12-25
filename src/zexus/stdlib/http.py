"""HTTP module for Zexus standard library."""

import urllib.request
import urllib.parse
import urllib.error
import json as json_lib
from typing import Dict, Any, Optional


class HttpModule:
    """Provides HTTP client operations."""

    @staticmethod
    def get(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP GET request."""
        if headers is None:
            headers = {}
        
        req = urllib.request.Request(url, headers=headers, method='GET')
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode('utf-8')
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'body': body
                }
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode('utf-8')
            except UnicodeDecodeError:
                error_body = e.read().decode('utf-8', errors='replace')
            return {
                'status': e.code,
                'headers': dict(e.headers),
                'body': error_body,
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 0,
                'headers': {},
                'body': '',
                'error': str(e)
            }

    @staticmethod
    def post(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None, 
             json: bool = False, timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP POST request."""
        if headers is None:
            headers = {}
        
        if json and data is not None:
            data = json_lib.dumps(data).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        elif isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, dict):
            data = urllib.parse.urlencode(data).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode('utf-8')
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'body': body
                }
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode('utf-8')
            except UnicodeDecodeError:
                error_body = e.read().decode('utf-8', errors='replace')
            return {
                'status': e.code,
                'headers': dict(e.headers),
                'body': error_body,
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 0,
                'headers': {},
                'body': '',
                'error': str(e)
            }

    @staticmethod
    def put(url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
            json: bool = False, timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP PUT request."""
        if headers is None:
            headers = {}
        
        if json and data is not None:
            data = json_lib.dumps(data).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        elif isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, dict):
            data = urllib.parse.urlencode(data).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers=headers, method='PUT')
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode('utf-8')
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'body': body
                }
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode('utf-8')
            except UnicodeDecodeError:
                error_body = e.read().decode('utf-8', errors='replace')
            return {
                'status': e.code,
                'headers': dict(e.headers),
                'body': error_body,
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 0,
                'headers': {},
                'body': '',
                'error': str(e)
            }

    @staticmethod
    def delete(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP DELETE request."""
        if headers is None:
            headers = {}
        
        req = urllib.request.Request(url, headers=headers, method='DELETE')
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode('utf-8')
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'body': body
                }
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode('utf-8')
            except UnicodeDecodeError:
                error_body = e.read().decode('utf-8', errors='replace')
            return {
                'status': e.code,
                'headers': dict(e.headers),
                'body': error_body,
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 0,
                'headers': {},
                'body': '',
                'error': str(e)
            }

    @staticmethod
    def request(method: str, url: str, data: Any = None, headers: Optional[Dict[str, str]] = None,
                timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP request with custom method."""
        if headers is None:
            headers = {}
        
        if data is not None:
            if isinstance(data, str):
                data = data.encode('utf-8')
            elif isinstance(data, dict):
                data = urllib.parse.urlencode(data).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode('utf-8')
                return {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'body': body
                }
        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode('utf-8')
            except UnicodeDecodeError:
                error_body = e.read().decode('utf-8', errors='replace')
            return {
                'status': e.code,
                'headers': dict(e.headers),
                'body': error_body,
                'error': str(e)
            }
        except Exception as e:
            return {
                'status': 0,
                'headers': {},
                'body': '',
                'error': str(e)
            }


# Export functions for easy access
get = HttpModule.get
post = HttpModule.post
put = HttpModule.put
delete = HttpModule.delete
request = HttpModule.request
