"""
Zexus Kernel — Extension layer for the interpreter and VM.

The kernel sits *alongside* the existing evaluator and VM.  It does NOT
replace them.  Instead it provides:

* **DomainRegistry** — a place where feature domains (blockchain, web,
  system, …) register their capabilities so the interpreter can discover
  them at runtime.
* **ZIR** — a formal opcode catalogue that documents (and validates) the
  bytecode the VM already uses, plus domain-specific opcode ranges.
* **Hooks** — optional integration points the evaluator/VM can call into
  (e.g. ``kernel.resolve_opcode()``, ``kernel.check_security()``).

The kernel is entirely opt-in.  Everything works without it — but with
it, third-party domains can plug into the Zexus runtime cleanly.

Quick start
-----------
>>> from zexus.kernel import get_kernel
>>> k = get_kernel()
>>> k.registry.list_domains()         # see what's loaded
>>> k.registry.get_domain("blockchain")  # query a specific domain
"""

from .registry import DomainRegistry, get_registry
from .hooks import Kernel, get_kernel

__all__ = [
    "DomainRegistry",
    "get_registry",
    "Kernel",
    "get_kernel",
]
