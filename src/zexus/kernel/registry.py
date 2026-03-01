"""
Domain Registry — Central registration point for Zexus domains.

Each domain (blockchain, web, ui, system, …) registers itself here at
import-time.  The kernel uses the registry to resolve domain-specific
opcodes, security policies, and runtime services without directly
importing any domain package.

Usage
-----
>>> from zexus.kernel import get_registry
>>> reg = get_registry()
>>> reg.register_domain(
...     name="blockchain",
...     version="1.0.0",
...     opcodes={0x1000: "HASH_BLOCK", 0x1001: "VERIFY_SIG"},
... )
>>> reg.get_domain("blockchain").version
'1.0.0'
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Domain descriptor
# ---------------------------------------------------------------------------

@dataclass
class DomainDescriptor:
    """Immutable description of a registered domain."""

    name: str
    version: str = "0.0.0"
    opcodes: Dict[int, str] = field(default_factory=dict)
    security_policy: Optional[Any] = None
    runtime: Optional[Any] = None
    validate_zir: Optional[Callable] = None
    # Additional metadata
    description: str = ""
    dependencies: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"DomainDescriptor(name={self.name!r}, version={self.version!r}, opcodes={len(self.opcodes)})"


# ---------------------------------------------------------------------------
# Domain Registry
# ---------------------------------------------------------------------------

class DomainRegistry:
    """Thread-safe singleton registry for Zexus domains.

    The registry enforces two key constraints:
    1. Domain names are unique.
    2. Opcode ranges must not overlap between domains.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._domains: Dict[str, DomainDescriptor] = {}
        self._opcode_owners: Dict[int, str] = {}  # opcode → domain name
        self._listeners: List[Callable[[DomainDescriptor], None]] = []

    # -- Registration -------------------------------------------------------

    def register_domain(
        self,
        name: str,
        *,
        version: str = "0.0.0",
        opcodes: Optional[Dict[int, str]] = None,
        security_policy: Optional[Any] = None,
        runtime: Optional[Any] = None,
        validate_zir: Optional[Callable] = None,
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> DomainDescriptor:
        """Register a new domain.

        Raises
        ------
        ValueError
            If *name* is already registered or any opcode collides with
            another domain's opcode range.
        """
        opcodes = opcodes or {}
        dependencies = dependencies or []

        with self._lock:
            if name in self._domains:
                raise ValueError(f"Domain {name!r} is already registered")

            # Check for opcode collisions
            for op, op_name in opcodes.items():
                if op in self._opcode_owners:
                    owner = self._opcode_owners[op]
                    raise ValueError(
                        f"Opcode 0x{op:04X} ({op_name}) collides with "
                        f"domain {owner!r}"
                    )

            desc = DomainDescriptor(
                name=name,
                version=version,
                opcodes=opcodes,
                security_policy=security_policy,
                runtime=runtime,
                validate_zir=validate_zir,
                description=description,
                dependencies=dependencies,
            )

            self._domains[name] = desc
            for op in opcodes:
                self._opcode_owners[op] = name

            # Notify listeners
            for listener in self._listeners:
                try:
                    listener(desc)
                except Exception:
                    pass

        return desc

    def unregister_domain(self, name: str) -> None:
        """Remove a previously registered domain."""
        with self._lock:
            desc = self._domains.pop(name, None)
            if desc:
                for op in desc.opcodes:
                    self._opcode_owners.pop(op, None)

    # -- Query --------------------------------------------------------------

    def get_domain(self, name: str) -> Optional[DomainDescriptor]:
        """Return the descriptor for *name*, or ``None``."""
        return self._domains.get(name)

    def list_domains(self) -> List[DomainDescriptor]:
        """Return all registered domains (snapshot)."""
        with self._lock:
            return list(self._domains.values())

    @property
    def domain_names(self) -> Set[str]:
        """Set of registered domain names."""
        return set(self._domains.keys())

    def resolve_opcode(self, opcode: int) -> Optional[str]:
        """Return the domain name that owns *opcode*, or ``None``."""
        return self._opcode_owners.get(opcode)

    # -- Listeners ----------------------------------------------------------

    def on_domain_registered(self, callback: Callable[[DomainDescriptor], None]) -> None:
        """Register a callback that fires whenever a new domain is registered."""
        self._listeners.append(callback)

    # -- Utilities ----------------------------------------------------------

    def check_dependencies(self, name: str) -> List[str]:
        """Return list of missing dependencies for domain *name*.

        Returns an empty list if all dependencies are satisfied.
        """
        desc = self._domains.get(name)
        if desc is None:
            return [name]
        return [dep for dep in desc.dependencies if dep not in self._domains]

    def reset(self) -> None:
        """Remove all registered domains (useful for testing)."""
        with self._lock:
            self._domains.clear()
            self._opcode_owners.clear()

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._domains.keys()))
        return f"DomainRegistry(domains=[{names}])"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_registry: Optional[DomainRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> DomainRegistry:
    """Return the process-wide :class:`DomainRegistry` singleton."""
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = DomainRegistry()
    return _global_registry
