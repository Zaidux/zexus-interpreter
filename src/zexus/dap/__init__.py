"""
Debug Adapter Protocol (DAP) support for Zexus.

Provides step-through debugging with breakpoints, variable inspection,
call stack display, and expression evaluation.
"""

from .debug_engine import DebugEngine, StopReason

__all__ = ["DebugEngine", "StopReason"]
