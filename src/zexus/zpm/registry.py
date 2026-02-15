"""
Package Registry - Manages package discovery and metadata

Supports both a local cache and a remote HTTP registry.
The registry URL defaults to ``https://registry.zexus.dev`` and can be
overridden via the ``ZPM_REGISTRY`` environment variable.

Auth tokens are read from ``~/.zpm/auth_token`` or the ``ZPM_AUTH_TOKEN``
environment variable.

REST API contract
-----------------
The following endpoints are expected on the registry server:

    GET  /packages/<name>                 → package metadata (latest)
    GET  /packages/<name>/<version>       → package metadata for version
    GET  /packages/<name>/versions        → {"versions": ["0.1.0", …]}
    GET  /search?q=<query>               → {"results": [{…}, …]}
    POST /packages                        → publish (multipart: meta + tarball)
    GET  /packages/<name>/<version>/download → tarball stream
"""
import os
import json
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional


class RegistryError(Exception):
    """Raised when a registry operation fails."""


class PackageRegistry:
    """Package registry for discovering and managing packages"""
    
    def __init__(self, registry_url: str = None):
        self.registry_url = (
            registry_url
            or os.environ.get("ZPM_REGISTRY", "https://registry.zexus.dev")
        ).rstrip("/")
        
        # Local cache directory
        self.cache_dir = Path.home() / ".zpm" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Auth token for publish operations
        self._auth_token: Optional[str] = None
        
        # Built-in packages
        self.builtin_packages = self._load_builtin_packages()

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    @property
    def auth_token(self) -> Optional[str]:
        """Lazily resolve the auth token from env or disk."""
        if self._auth_token is not None:
            return self._auth_token
        # 1. Environment variable
        token = os.environ.get("ZPM_AUTH_TOKEN")
        if token:
            self._auth_token = token
            return token
        # 2. Token file
        token_path = Path.home() / ".zpm" / "auth_token"
        if token_path.exists():
            token = token_path.read_text().strip()
            if token:
                self._auth_token = token
                return token
        return None

    def login(self, token: str) -> None:
        """Store an auth token to ``~/.zpm/auth_token``."""
        self._auth_token = token
        token_path = Path.home() / ".zpm" / "auth_token"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(token + "\n")
        # Restrict permissions (owner-only read/write)
        try:
            token_path.chmod(0o600)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(self, path: str, *, method: str = "GET",
                 data: bytes = None, headers: dict = None,
                 timeout: int = 30) -> Optional[bytes]:
        """Perform an HTTP request against the registry.

        Returns the response body on success, or ``None`` when the
        resource is not found (HTTP 404).  Raises ``RegistryError`` for
        other HTTP errors or connection failures.
        """
        url = f"{self.registry_url}{path}"
        hdrs = {"User-Agent": "zpm/1.0"}
        if self.auth_token:
            hdrs["Authorization"] = f"Bearer {self.auth_token}"
        if headers:
            hdrs.update(headers)

        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RegistryError(
                f"Registry HTTP {exc.code} for {method} {path}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RegistryError(
                f"Cannot reach registry at {self.registry_url}: {exc.reason}"
            ) from exc

    def _get_json(self, path: str) -> Optional[Dict]:
        """GET ``path`` and parse the JSON body.  Returns None on 404."""
        body = self._request(path)
        if body is None:
            return None
        return json.loads(body)

    # ------------------------------------------------------------------
    # Built-in packages
    # ------------------------------------------------------------------
    
    def _load_builtin_packages(self) -> Dict:
        """Load built-in package definitions"""
        return {
            "std": {
                "name": "std",
                "version": "0.1.0",
                "description": "Zexus standard library",
                "type": "builtin",
                "files": []
            },
            "crypto": {
                "name": "crypto",
                "version": "0.1.0",
                "description": "Cryptography utilities",
                "type": "builtin",
                "files": []
            },
            "web": {
                "name": "web",
                "version": "0.1.0",
                "description": "Web framework for Zexus",
                "type": "builtin",
                "files": []
            },
            "blockchain": {
                "name": "blockchain",
                "version": "0.1.0",
                "description": "Blockchain utilities and helpers",
                "type": "builtin",
                "files": []
            }
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def get_package(self, name: str, version: str = "latest") -> Optional[Dict]:
        """Get package metadata from registry.

        Resolution order:
        1. Built-in packages
        2. Local cache (``~/.zpm/cache/<name>-<version>.json``)
        3. Remote registry (``GET /packages/<name>[/<version>]``)
        """
        # 1. Check built-in packages
        if name in self.builtin_packages:
            return self.builtin_packages[name]
        
        # 2. Check local cache
        cache_file = self.cache_dir / f"{name}-{version}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        
        # 3. Fetch from remote registry
        try:
            path = f"/packages/{urllib.parse.quote(name)}"
            if version and version != "latest":
                path += f"/{urllib.parse.quote(version)}"
            meta = self._get_json(path)
            if meta:
                # Cache the result locally
                self._cache_metadata(name, meta.get("version", version), meta)
                return meta
        except RegistryError as e:
            # Log but don't crash — fall through to None
            print(f"⚠️  Registry lookup failed for {name}: {e}")

        return None
    
    def search(self, query: str) -> List[Dict]:
        """Search for packages.

        Searches built-in packages locally *and* queries the remote
        registry via ``GET /search?q=<query>``.
        """
        results = []
        
        # Search built-in packages
        q = query.lower()
        for name, pkg in self.builtin_packages.items():
            if q in name.lower() or q in pkg.get("description", "").lower():
                results.append(pkg)
        
        # Search remote registry
        try:
            encoded = urllib.parse.urlencode({"q": query})
            data = self._get_json(f"/search?{encoded}")
            if data and isinstance(data.get("results"), list):
                for pkg in data["results"]:
                    # Deduplicate against builtins
                    if pkg.get("name") not in self.builtin_packages:
                        results.append(pkg)
        except RegistryError as e:
            print(f"⚠️  Remote search failed: {e}")
        
        return results
    
    def publish_package(self, package_data: Dict, files: List[str]) -> bool:
        """Publish a package to the registry.

        Sends a multipart POST to ``/packages`` with the package metadata
        (JSON) and the tarball (binary).  Requires an auth token.
        """
        name = package_data.get("name", "")
        version = package_data.get("version", "")

        # Always cache locally first
        self._cache_metadata(name, version, package_data)

        # Attempt remote publish
        if not self.auth_token:
            print(f"⚠️  No auth token — package cached locally only.")
            print(f"   Run 'zpm login <token>' to enable remote publishing.")
            return True

        tarball_path = package_data.get("tarball")
        try:
            import mimetypes
            boundary = "----ZPMPublishBoundary"
            body_parts = []

            # Part 1: metadata JSON
            body_parts.append(f"--{boundary}".encode())
            body_parts.append(b'Content-Disposition: form-data; name="metadata"')
            body_parts.append(b"Content-Type: application/json")
            body_parts.append(b"")
            body_parts.append(json.dumps(package_data).encode())

            # Part 2: tarball file (if present)
            if tarball_path and os.path.isfile(tarball_path):
                body_parts.append(f"--{boundary}".encode())
                body_parts.append(
                    f'Content-Disposition: form-data; name="tarball"; '
                    f'filename="{os.path.basename(tarball_path)}"'.encode()
                )
                body_parts.append(b"Content-Type: application/gzip")
                body_parts.append(b"")
                with open(tarball_path, "rb") as f:
                    body_parts.append(f.read())

            body_parts.append(f"--{boundary}--".encode())
            body = b"\r\n".join(body_parts)

            self._request(
                "/packages",
                method="POST",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            print(f"✅ Published {name}@{version} to {self.registry_url}")
            return True
        except RegistryError as e:
            print(f"❌ Remote publish failed: {e}")
            print(f"   Package was cached locally at ~/.zpm/cache/")
            return False
    
    def get_versions(self, package: str) -> List[str]:
        """Get all available versions of a package."""
        if package in self.builtin_packages:
            return [self.builtin_packages[package]["version"]]
        
        # Check remote registry
        try:
            encoded = urllib.parse.quote(package)
            data = self._get_json(f"/packages/{encoded}/versions")
            if data and isinstance(data.get("versions"), list):
                return data["versions"]
        except RegistryError as e:
            print(f"⚠️  Version lookup failed for {package}: {e}")
        
        return []

    def download_tarball(self, name: str, version: str) -> Optional[Path]:
        """Download a package tarball from the registry.

        Returns the local path to the downloaded ``.tar.gz`` file,
        or ``None`` on failure.
        """
        tarball_cache = self.cache_dir / f"{name}-{version}.tar.gz"
        if tarball_cache.exists():
            return tarball_cache

        try:
            encoded_name = urllib.parse.quote(name)
            encoded_ver = urllib.parse.quote(version)
            body = self._request(f"/packages/{encoded_name}/{encoded_ver}/download")
            if body is None:
                return None
            tarball_cache.write_bytes(body)
            return tarball_cache
        except RegistryError as e:
            print(f"⚠️  Tarball download failed for {name}@{version}: {e}")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_metadata(self, name: str, version: str, meta: Dict) -> None:
        """Write package metadata to the local cache."""
        cache_file = self.cache_dir / f"{name}-{version}.json"
        with open(cache_file, "w") as f:
            json.dump(meta, f, indent=2)
