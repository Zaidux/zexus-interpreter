"""
Kernel Hooks — Integration points between the kernel and the interpreter.

The :class:`Kernel` object is the single entry-point that the evaluator,
VM, or CLI can *optionally* use to tap into the domain/extension system.

It does NOT wrap or replace any existing class.  It provides *additional*
services that sit alongside the evaluator and VM:

* Resolve domain-specific opcodes to handler functions.
* Compose security policies across multiple domains.
* Broadcast lifecycle events (program start, module load, …).
* Provide a unified introspection API for tooling (LSP, debugger).

Example — wiring the kernel into an existing run:

    from zexus.evaluator import Evaluator
    from zexus.kernel import get_kernel

    kernel = get_kernel()
    kernel.boot()              # auto-discovers built-in domains

    evaluator = Evaluator()
    # The evaluator still works exactly as before.
    # But now you can also ask the kernel:
    info = kernel.registry.get_domain("blockchain")
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .registry import DomainRegistry, DomainDescriptor, get_registry


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

class KernelEvent:
    """Simple event descriptor."""

    __slots__ = ("name", "timestamp", "data")

    def __init__(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.timestamp = time.time()
        self.data = data or {}

    def __repr__(self) -> str:
        return f"KernelEvent({self.name!r})"


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

class Kernel:
    """Extension layer that sits alongside the interpreter and VM.

    The kernel is **purely additive** — the interpreter works with or
    without it.  When present it provides domain discovery, opcode
    resolution, security composition, and lifecycle events.
    """

    def __init__(self, registry: Optional[DomainRegistry] = None) -> None:
        self.registry: DomainRegistry = registry or get_registry()
        self._booted = False
        self._event_listeners: Dict[str, List[Callable[[KernelEvent], None]]] = {}
        self._opcode_handlers: Dict[int, Callable] = {}
        self._middleware: List[Callable] = []

    # -- Boot ---------------------------------------------------------------

    def boot(self) -> "Kernel":
        """Discover and register all built-in domains.

        Safe to call multiple times — subsequent calls are no-ops.
        Returns *self* for chaining.
        """
        if self._booted:
            return self

        self._register_builtin_domains()
        self._booted = True
        self._emit("kernel.booted", {"domains": list(self.registry.domain_names)})
        return self

    @property
    def is_booted(self) -> bool:
        return self._booted

    # -- Opcode resolution --------------------------------------------------

    def register_opcode_handler(self, opcode: int, handler: Callable) -> None:
        """Map a domain opcode to a Python callable.

        This lets domain authors provide executable implementations for
        their custom opcodes.  The VM can query these at dispatch time.
        """
        self._opcode_handlers[opcode] = handler

    def get_opcode_handler(self, opcode: int) -> Optional[Callable]:
        """Return the handler for *opcode*, or ``None``."""
        return self._opcode_handlers.get(opcode)

    def resolve_opcode_domain(self, opcode: int) -> Optional[str]:
        """Return the domain name that owns *opcode*."""
        return self.registry.resolve_opcode(opcode)

    # -- Security composition -----------------------------------------------

    def check_security(self, operation: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Ask every registered domain's security policy to approve *operation*.

        Returns ``True`` only if **all** domains approve (or have no policy).
        """
        context = context or {}
        for desc in self.registry.list_domains():
            policy = desc.security_policy
            if policy is not None and hasattr(policy, "check"):
                try:
                    if not policy.check(operation, context):
                        return False
                except Exception:
                    return False
        return True

    # -- Middleware ----------------------------------------------------------

    def use(self, middleware_fn: Callable) -> None:
        """Register a middleware function.

        Middleware is called in order for certain pipeline stages
        (compile, execute) and can inspect or transform the data.
        """
        self._middleware.append(middleware_fn)

    def run_middleware(self, stage: str, data: Any) -> Any:
        """Run all middleware for *stage*, threading *data* through."""
        for mw in self._middleware:
            try:
                result = mw(stage, data)
                if result is not None:
                    data = result
            except Exception:
                pass
        return data

    # -- Events -------------------------------------------------------------

    def on(self, event_name: str, callback: Callable[[KernelEvent], None]) -> None:
        """Subscribe to a kernel lifecycle event."""
        self._event_listeners.setdefault(event_name, []).append(callback)

    def _emit(self, event_name: str, data: Optional[Dict[str, Any]] = None) -> None:
        event = KernelEvent(event_name, data)
        for cb in self._event_listeners.get(event_name, []):
            try:
                cb(event)
            except Exception:
                pass
        # Also fire wildcard listeners
        for cb in self._event_listeners.get("*", []):
            try:
                cb(event)
            except Exception:
                pass

    # -- Introspection ------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a snapshot of the kernel state (useful for tooling/debug)."""
        domains = self.registry.list_domains()
        return {
            "booted": self._booted,
            "domain_count": len(domains),
            "domains": {d.name: d.version for d in domains},
            "opcode_handlers": len(self._opcode_handlers),
            "middleware": len(self._middleware),
            "event_subscriptions": {
                k: len(v) for k, v in self._event_listeners.items()
            },
        }

    # -- Built-in domain registration (private) -----------------------------

    def _register_builtin_domains(self) -> None:
        """Register the built-in domains that ship with the interpreter.

        Each registration is idempotent — already-registered domains are
        silently skipped.
        """
        builtins = [
            {
                "name": "blockchain",
                "version": "1.8.3",
                "description": "Ledger, smart-contract execution, crypto, gas metering",
                "opcodes": {
                    0x1000: "HASH_BLOCK",
                    0x1001: "VERIFY_SIGNATURE",
                    0x1002: "STATE_READ",
                    0x1003: "STATE_WRITE",
                    0x1004: "GAS_CHARGE",
                    0x1005: "MERKLE_ROOT",
                    0x1006: "CREATE_TX",
                    0x1007: "EMIT_EVENT",
                    0x1008: "REQUIRE",
                    0x1009: "BALANCE_OF",
                    0x100A: "CREATE_CHAIN",
                    0x100B: "ADD_BLOCK",
                },
            },
            {
                "name": "web",
                "version": "1.8.3",
                "description": "HTTP server/client, WebSocket, middleware",
                "opcodes": {
                    0x1100: "HTTP_ROUTE",
                    0x1101: "HTTP_METHOD",
                    0x1102: "HTTP_RESPONSE",
                    0x1103: "WS_CONNECT",
                    0x1104: "WS_SEND",
                    0x1105: "WS_RECEIVE",
                },
            },
            {
                "name": "system",
                "version": "1.8.3",
                "description": "File I/O, process management, environment variables",
                "opcodes": {
                    0x1200: "FS_READ",
                    0x1201: "FS_WRITE",
                    0x1202: "FS_DELETE",
                    0x1203: "FS_LIST_DIR",
                    0x1204: "PROCESS_SPAWN",
                    0x1205: "ENV_GET",
                    0x1206: "ENV_SET",
                },
            },
            {
                "name": "ui",
                "version": "1.8.3",
                "description": "Terminal graphics, web rendering, native widgets",
                "opcodes": {},
            },
        ]

        for domain in builtins:
            if self.registry.get_domain(domain["name"]) is None:
                self.registry.register_domain(**domain)

    def __repr__(self) -> str:
        state = "booted" if self._booted else "idle"
        n = len(self.registry.list_domains())
        return f"Kernel({state}, {n} domains)"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_kernel: Optional[Kernel] = None
_kernel_lock = threading.Lock()


def get_kernel() -> Kernel:
    """Return the process-wide :class:`Kernel` singleton."""
    global _global_kernel
    if _global_kernel is None:
        with _kernel_lock:
            if _global_kernel is None:
                _global_kernel = Kernel()
    return _global_kernel
