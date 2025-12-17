# src/zexus/evaluator/__init__.py
from typing import Any, Dict

from .core import Evaluator, evaluate
from .utils import EVAL_SUMMARY

# Module-level builtins registry. Tests and other code may inject into this
# dict (e.g. `from zexus.evaluator import builtins as evaluator_builtins`).
# At runtime, `evaluate()` will copy these into each Evaluator instance.
builtins: Dict[str, Any] = {}

__all__ = ['Evaluator', 'evaluate', 'EVAL_SUMMARY', 'builtins']
