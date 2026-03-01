"""
Zexus Intermediate Representation (ZIR) — Opcode catalogue and validation.

This module formalises the opcodes the VM already uses, putting them into
an enum so that tooling (LSP, debugger, profiler) and domain authors can
reference them by name.  It does NOT replace the existing Opcode class in
``vm/compiler.py`` — it sits alongside it as a typed reference.

Opcode ranges
-------------
0x0001–0x00FF  Core operations (math, stack, variables)
0x0100–0x01FF  Control flow (jump, call, return, try)
0x0200–0x02FF  Memory / field access
0x1000–0x1FFF  Domain-specific (registered at runtime via DomainRegistry)
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional


class CoreOpcode(IntEnum):
    """Core opcodes understood by every execution backend.

    These mirror the opcodes already defined in the VM — they provide
    a stable, importable reference for tooling and domain code.
    """

    # -- Constants / stack --------------------------------------------------
    LOAD_CONST    = 0x0001
    LOAD_NAME     = 0x0002
    STORE_NAME    = 0x0003
    POP_TOP       = 0x0004
    DUP_TOP       = 0x0005

    # -- Arithmetic ---------------------------------------------------------
    ADD           = 0x0010
    SUB           = 0x0011
    MUL           = 0x0012
    DIV           = 0x0013
    MOD           = 0x0014
    POW           = 0x0015
    NEG           = 0x0016

    # -- Comparison ---------------------------------------------------------
    EQ            = 0x0020
    NEQ           = 0x0021
    LT            = 0x0022
    GT            = 0x0023
    LTE           = 0x0024
    GTE           = 0x0025

    # -- Logic --------------------------------------------------------------
    AND           = 0x0030
    OR            = 0x0031
    NOT           = 0x0032

    # -- Collections --------------------------------------------------------
    BUILD_LIST    = 0x0040
    BUILD_MAP     = 0x0041
    INDEX         = 0x0042
    STORE_INDEX   = 0x0043

    # -- Control flow -------------------------------------------------------
    JUMP          = 0x0100
    JUMP_IF_FALSE = 0x0101
    CALL          = 0x0102
    RETURN        = 0x0103
    TRY           = 0x0104
    THROW         = 0x0105

    # -- Memory / fields ----------------------------------------------------
    LOAD_FIELD    = 0x0200
    STORE_FIELD   = 0x0201
    ALLOC         = 0x0202


# All valid core opcode values for fast membership testing
_CORE_OPCODES = frozenset(int(op) for op in CoreOpcode)


def is_core_opcode(opcode: int) -> bool:
    """Return True if *opcode* is in the core range."""
    return opcode in _CORE_OPCODES


def resolve_opcode_name(opcode: int) -> Optional[str]:
    """Human-readable name for any opcode (core or domain).

    Returns ``None`` for completely unknown opcodes.
    """
    # Try core first
    try:
        return CoreOpcode(opcode).name
    except ValueError:
        pass

    # Try domain registry
    from ..registry import get_registry
    owner = get_registry().resolve_opcode(opcode)
    if owner is not None:
        domain = get_registry().get_domain(owner)
        if domain:
            return domain.opcodes.get(opcode)
    return None


def validate_zir(
    instructions: list,
    *,
    allowed_domains: Optional[set[str]] = None,
) -> list[str]:
    """Validate a sequence of ZIR instructions.

    Returns a list of error strings (empty == valid).

    Parameters
    ----------
    instructions
        ``[(opcode, *operands), ...]``
    allowed_domains
        If given, only opcodes belonging to these domains (plus core)
        are permitted.
    """
    from ..registry import get_registry

    errors: list[str] = []
    registry = get_registry()

    for idx, instr in enumerate(instructions):
        opcode = instr[0] if isinstance(instr, (list, tuple)) else instr

        if opcode in _CORE_OPCODES:
            continue

        owner = registry.resolve_opcode(opcode)
        if owner is None:
            errors.append(f"Instruction {idx}: unknown opcode 0x{opcode:04X}")
        elif allowed_domains is not None and owner not in allowed_domains:
            errors.append(
                f"Instruction {idx}: opcode 0x{opcode:04X} belongs to "
                f"domain {owner!r} which is not in allowed set"
            )
    return errors
