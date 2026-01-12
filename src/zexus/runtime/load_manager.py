"""Resource loading utilities for the `load` keyword."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


class LoadManager:
    """Coordinates value lookups for the `load` keyword."""

    def __init__(self, project_root: Optional[Path] = None) -> None:
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self._providers: Dict[str, Callable[..., Any]] = {}
        self._dotenv_cache: Dict[Path, Dict[str, Any]] = {}
        self._structured_cache: Dict[tuple, Dict[str, Any]] = {}
        self._register_defaults()

    # ------------------------------------------------------------------
    # Provider registration
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        self.register_provider("env", self._load_from_env)
        self.register_provider("dotenv", self._load_from_dotenv)
        self.register_provider("file", self._load_from_file)

    def register_provider(self, name: str, handler: Callable[..., Any]) -> None:
        self._providers[name.lower()] = handler

    def unregister_provider(self, name: str) -> None:
        self._providers.pop(name.lower(), None)

    def available_providers(self) -> Sequence[str]:
        return tuple(self._providers.keys())

    def is_provider_registered(self, name: Optional[str]) -> bool:
        return bool(name) and name.lower() in self._providers

    # ------------------------------------------------------------------
    # Public load API
    # ------------------------------------------------------------------

    def load(
        self,
        key: str,
        *,
        provider: Optional[str] = None,
        source: Optional[str] = None,
        current_dir: Optional[str] = None,
    ) -> Any:
        """Resolve a value using registered providers and fallbacks."""

        provider_name = provider.lower() if provider else None
        if provider_name:
            handler = self._providers.get(provider_name)
            if not handler:
                raise KeyError(f"Unknown provider '{provider}'")
            value = handler(
                key,
                source=source,
                current_dir=current_dir,
                explicit=True,
            )
            if value is None:
                raise KeyError(key)
            return value

        # Default lookup chain
        value = self._load_from_env(key)
        if value is not None:
            return value

        value = self._load_from_dotenv(
            key,
            source=source,
            current_dir=current_dir,
        )
        if value is not None:
            return value

        if source:
            value = self._load_from_file(
                key,
                source,
                current_dir=current_dir,
                explicit=True,
            )
            if value is not None:
                return value

        raise KeyError(key)

    def clear_caches(self) -> None:
        self._dotenv_cache.clear()
        self._structured_cache.clear()

    # ------------------------------------------------------------------
    # Provider helpers
    # ------------------------------------------------------------------

    def _load_from_env(
        self,
        key: str,
        *,
        source: Optional[str] = None,
        current_dir: Optional[str] = None,
        explicit: bool | None = None,
    ) -> Optional[str]:
        if not key:
            return None

        for candidate in self._key_variants(key):
            value = os.environ.get(candidate)
            if value is not None:
                return value
        return None

    def _load_from_dotenv(
        self,
        key: str,
        *,
        source: Optional[str] = None,
        current_dir: Optional[str] = None,
        explicit: bool | None = None,
    ) -> Optional[str]:
        candidates: List[Path] = []

        if source:
            resolved = self._resolve_source(source, current_dir)
            if resolved and resolved.is_file():
                candidates.append(resolved)

        if not explicit:
            # Current directory .env
            if current_dir:
                candidates.append(Path(current_dir) / ".env")
            # Project root .env
            candidates.append(self.project_root / ".env")

        for path in candidates:
            mapping = self._load_dotenv_file(path)
            if not mapping:
                continue
            value = self._lookup_mapping(mapping, key)
            if value is not None:
                return value
        return None

    def _load_from_file(
        self,
        key: str,
        source: str,
        *,
        current_dir: Optional[str] = None,
        explicit: bool | None = None,
    ) -> Any:
        resolved = self._resolve_source(source, current_dir)
        if not resolved:
            raise FileNotFoundError(source)

        if resolved.is_dir():
            # Allow pointing at directory containing .env
            env_candidate = resolved / ".env"
            return self._load_from_dotenv(
                key,
                source=str(env_candidate),
                current_dir=current_dir,
                explicit=True,
            )

        suffix = resolved.suffix.lower()
        if suffix in (".env", ""):
            return self._load_from_dotenv(
                key,
                source=str(resolved),
                current_dir=current_dir,
                explicit=True,
            )

        if suffix in (".json", ".jsn"):
            data = self._load_json_file(resolved)
            return self._extract_from_data(data, key)

        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise FileNotFoundError(
                    f"YAML support requires PyYAML to be installed: {resolved}"
                ) from exc
            data = self._load_yaml_file(resolved, yaml)
            return self._extract_from_data(data, key)

        if not key:
            return resolved.read_text(encoding="utf-8")
        return None

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _resolve_source(
        self,
        source: str,
        current_dir: Optional[str],
    ) -> Optional[Path]:
        candidate = Path(source)
        search_order: Iterable[Path]

        if candidate.is_absolute():
            search_order = (candidate,)
        else:
            search_order = filter(
                None,
                (
                    Path(current_dir) / candidate if current_dir else None,
                    self.project_root / candidate,
                ),
            )

        for root in search_order:
            try:
                resolved = root.resolve()
            except (OSError, RuntimeError):
                continue
            if resolved.exists():
                return resolved
        return candidate.resolve() if candidate.exists() else None

    def _load_dotenv_file(self, path: Path) -> Optional[Dict[str, str]]:
        if not path or not path.exists() or not path.is_file():
            return None

        try:
            mtime = path.stat().st_mtime_ns
        except (OSError, AttributeError):
            mtime = None

        cache = self._dotenv_cache.get(path)
        if cache and cache.get("mtime") == mtime:
            return cache.get("data")

        data: Dict[str, str] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if "=" not in stripped:
                        continue
                    key, raw_value = stripped.split("=", 1)
                    value = raw_value.strip().strip('"').strip("'")
                    data[key.strip()] = value
        except OSError:
            return None

        self._dotenv_cache[path] = {"mtime": mtime, "data": data}
        return data

    def _load_json_file(self, path: Path) -> Any:
        try:
            mtime = path.stat().st_mtime_ns
        except (OSError, AttributeError):
            mtime = None

        cache = self._structured_cache.get((path, "json"))
        if cache and cache.get("mtime") == mtime:
            return cache.get("data")

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        self._structured_cache[(path, "json")] = {"mtime": mtime, "data": data}
        return data

    def _load_yaml_file(self, path: Path, yaml_module) -> Any:  # pragma: no cover - optional dependency
        try:
            mtime = path.stat().st_mtime_ns
        except (OSError, AttributeError):
            mtime = None

        cache = self._structured_cache.get((path, "yaml"))
        if cache and cache.get("mtime") == mtime:
            return cache.get("data")

        with path.open("r", encoding="utf-8") as handle:
            data = yaml_module.safe_load(handle)

        self._structured_cache[(path, "yaml")] = {"mtime": mtime, "data": data}
        return data

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _lookup_mapping(self, mapping: Dict[str, Any], key: str) -> Optional[str]:
        for candidate in self._key_variants(key):
            value = mapping.get(candidate)
            if value is not None:
                return value
        return None

    def _extract_from_data(self, data: Any, key: str) -> Any:
        if key is None:
            return data
        if not key:
            return data

        parts = key.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict):
                if part in current:
                    current = current[part]
                    continue
                alt = part.replace("-", "_")
                if alt in current:
                    current = current[alt]
                    continue
                alt_upper = alt.upper()
                if alt_upper in current:
                    current = current[alt_upper]
                    continue
                return None
            if isinstance(current, list):
                try:
                    index = int(part)
                except ValueError:
                    return None
                if 0 <= index < len(current):
                    current = current[index]
                    continue
                return None
            return None
        return current

    def _key_variants(self, key: str) -> List[str]:
        variants = {key, key.lower(), key.upper()}
        if "." in key:
            collapsed = key.replace(".", "_")
            variants.add(collapsed)
            variants.add(collapsed.lower())
            variants.add(collapsed.upper())
        if "-" in key:
            hyphenless = key.replace("-", "_")
            variants.add(hyphenless)
            variants.add(hyphenless.lower())
            variants.add(hyphenless.upper())
        return [variant for variant in variants if variant]


_manager = LoadManager()


def get_load_manager() -> LoadManager:
    return _manager


def register_provider(name: str, handler: Callable[..., Any]) -> None:
    _manager.register_provider(name, handler)


def clear_load_caches() -> None:
    _manager.clear_caches()
