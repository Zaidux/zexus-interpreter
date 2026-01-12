"""Utility helpers for higher-level graphics composition.

The original module exposed a number of free-form helpers.  For now we provide
lightweight primitives that are sufficient for interpreter driven demos while
keeping the door open for future expansion.
"""
from __future__ import annotations

from typing import Iterable, List


def merge_layers(layers: Iterable[List[str]]) -> List[str]:
    """Merge multiple ASCII layers, preferring non-space characters."""
    merged: List[str] = []
    for layer in layers:
        if not merged:
            merged = list(layer)
            continue
        merged = [
            "".join(c2 if c2 != " " else c1 for c1, c2 in zip(row1, row2))
            for row1, row2 in zip(merged, layer)
        ]
    return merged


def frame(text: str, *, padding: int = 1) -> str:
    lines = text.splitlines() or [""]
    width = max(len(line) for line in lines)
    padded = [" " * padding + line.ljust(width) + " " * padding for line in lines]
    horizontal = "─" * (width + padding * 2)
    return "\n".join([f"┌{horizontal}┐", *[f"│{line}│" for line in padded], f"└{horizontal}┘"])
