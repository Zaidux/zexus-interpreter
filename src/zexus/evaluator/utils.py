# src/zexus/evaluator/utils.py
import asyncio
import os
import sys
from ..object import EvaluationError, Null, Boolean as BooleanObj, String
from ..config import config as zexus_config

NULL = Null()
TRUE = BooleanObj(True)
FALSE = BooleanObj(False)

EVAL_SUMMARY = {
    'parsed_statements': 0,
    'evaluated_statements': 0,
    'errors': 0,
    'async_tasks_run': 0,
    'max_statements_in_block': 0
}

def is_error(obj):
    return isinstance(obj, EvaluationError)

def debug_log(message, data=None, level='debug'):
    try:
        if not zexus_config.should_log(level):
            return
    except Exception:
        return
    
    if data is not None:
        print(f"🔍 [EVAL DEBUG] {message}: {data}")
    else:
        print(f"🔍 [EVAL DEBUG] {message}")

def _is_awaitable(obj):
    try:
        return asyncio.iscoroutine(obj) or isinstance(obj, asyncio.Future)
    except Exception:
        return False

def _resolve_awaitable(obj):
    if _is_awaitable(obj):
        try:
            EVAL_SUMMARY['async_tasks_run'] += 1
            return asyncio.run(obj)
        except RuntimeError:
            return obj
    return obj

def is_truthy(obj):
    """Check if an object is truthy (for if statements, while loops, etc.)"""
    from ..object import Boolean as BooleanObj
    
    if is_error(obj):
        return False
    if obj == NULL or obj is NULL:
        return False
    if obj == FALSE or obj is FALSE:
        return False
    # Check Boolean objects by their value
    if isinstance(obj, BooleanObj):
        return obj.value
    return True

# JSON conversion functions (moved from original)
def _zexus_to_python(value):
    """Convert Zexus objects to Python native types for JSON serialization"""
    from ..object import Map, List, String, Integer, Float, Boolean as BooleanObj
    
    if isinstance(value, Map):
        python_dict = {}
        for key, val in value.pairs.items():
            python_key = key.inspect() if hasattr(key, 'inspect') else str(key)
            python_dict[python_key] = _zexus_to_python(val)
        return python_dict
    elif isinstance(value, List):
        python_list = [_zexus_to_python(item) for item in value.elements]
        return python_list
    elif isinstance(value, String):
        return value.value
    elif isinstance(value, Integer):
        return value.value
    elif isinstance(value, Float):
        return value.value
    elif isinstance(value, BooleanObj):
        return value.value
    elif value == NULL:
        return None
    else:
        return str(value)

def _python_to_zexus(value):
    """Convert Python native types to Zexus objects"""
    from ..object import Map, List, String, Integer, Float, Boolean as BooleanObj
    
    if isinstance(value, dict):
        pairs = {}
        for k, v in value.items():
            pairs[k] = _python_to_zexus(v)
        return Map(pairs)
    elif isinstance(value, list):
        zexus_list = List([_python_to_zexus(item) for item in value])
        return zexus_list
    elif isinstance(value, str):
        return String(value)
    elif isinstance(value, int):
        return Integer(value)
    elif isinstance(value, float):
        return Float(value)
    elif isinstance(value, bool):
        return BooleanObj(value)
    elif value is None:
        return NULL
    else:
        return String(str(value))

def _to_str(obj):
    """Helper to convert Zexus object to string"""
    from ..object import String, Integer, Float, EntityInstance
    if isinstance(obj, String):
        return obj.value
    if isinstance(obj, (Integer, Float)):
        return str(obj.value)
    if isinstance(obj, EntityInstance):
        return obj.inspect()
    return getattr(obj, 'inspect', lambda: str(obj))()
