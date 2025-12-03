"""Compatibility shim: expose a small set of runtime utilities that the
interpreter provides. This is a best-effort adapter: when the interpreter
is importable we re-export real implementations; otherwise we provide
safe, minimal fallbacks to avoid hard import errors in tooling.

This file intentionally keeps fallbacks minimal and side-effect free.
"""
from typing import Any

try:
    # UPDATED: Import from new evaluator structure
    from ..evaluator.utils import (
        EVAL_SUMMARY, NULL, TRUE, FALSE,
        _is_awaitable, _resolve_awaitable, _python_to_zexus, _zexus_to_python,
        _to_python, _to_str, is_error, is_truthy, debug_log
    )
    
    from ..evaluator import evaluate
    from ..evaluator.functions import FunctionEvaluatorMixin
    from ..object import EvaluationError
    
    # Get builtins from FunctionEvaluatorMixin
    fe = FunctionEvaluatorMixin()
    fe.__init__()
    BUILTINS = fe.builtins
    
    # Create an Evaluator instance to access methods
    from ..evaluator.core import Evaluator
    evaluator = Evaluator()
    
    # Alias evaluator methods
    eval_program = evaluator.eval_program
    eval_block_statement = evaluator.eval_block_statement
    eval_expressions = evaluator.eval_expressions
    eval_identifier = evaluator.eval_identifier
    eval_if_expression = evaluator.eval_if_expression
    eval_infix_expression = evaluator.eval_infix_expression
    eval_prefix_expression = evaluator.eval_prefix_expression
    eval_assignment_expression = evaluator.eval_assignment_expression
    eval_let_statement_fixed = evaluator.eval_let_statement
    eval_try_catch_statement_fixed = evaluator.eval_try_catch_statement
    apply_function = evaluator.apply_function
    array_map = evaluator._array_map
    array_filter = evaluator._array_filter
    array_reduce = evaluator._array_reduce
    extend_function_env = evaluator.extend_function_env
    unwrap_return_value = evaluator.unwrap_return_value
    
    # Builtin functions are already in BUILTINS, but we can create aliases
    # Get them from the evaluator's builtins
    builtin_datetime_now = BUILTINS.get("now", lambda *a, **k: None)
    builtin_timestamp = BUILTINS.get("timestamp", lambda *a, **k: None)
    builtin_math_random = BUILTINS.get("random", lambda *a, **k: None)
    builtin_to_hex = BUILTINS.get("to_hex", lambda *a, **k: None)
    builtin_from_hex = BUILTINS.get("from_hex", lambda *a, **k: None)
    builtin_sqrt = BUILTINS.get("sqrt", lambda *a, **k: None)
    builtin_file_read_text = BUILTINS.get("file_read_text", lambda *a, **k: None)
    builtin_file_write_text = BUILTINS.get("file_write_text", lambda *a, **k: None)
    builtin_file_exists = BUILTINS.get("file_exists", lambda *a, **k: None)
    builtin_file_read_json = BUILTINS.get("file_read_json", lambda *a, **k: None)
    builtin_file_write_json = BUILTINS.get("file_write_json", lambda *a, **k: None)
    builtin_file_append = BUILTINS.get("file_append", lambda *a, **k: None)
    builtin_file_list_dir = BUILTINS.get("file_list_dir", lambda *a, **k: None)
    builtin_debug_log = BUILTINS.get("debug_log", lambda *a, **k: None)
    builtin_debug_trace = BUILTINS.get("debug_trace", lambda *a, **k: None)
    builtin_string = BUILTINS.get("string", lambda *a, **k: None)
    builtin_len = BUILTINS.get("len", lambda *a, **k: None)
    builtin_first = BUILTINS.get("first", lambda *a, **k: None)
    builtin_rest = BUILTINS.get("rest", lambda *a, **k: None)
    builtin_push = BUILTINS.get("push", lambda *a, **k: None)
    builtin_reduce = BUILTINS.get("reduce", lambda *a, **k: None)
    builtin_map = BUILTINS.get("map", lambda *a, **k: None)
    builtin_filter = BUILTINS.get("filter", lambda *a, **k: None)
    
    # Render registry is now instance-specific in FunctionEvaluatorMixin
    RENDER_REGISTRY = evaluator.render_registry if hasattr(evaluator, 'render_registry') else {
        'screens': {}, 'components': {}, 'themes': {}, 'canvases': {}, 'current_theme': None
    }
    
    # Import permission check (if available)
    try:
        from ..evaluator.statements import StatementEvaluatorMixin
        stmt_evaluator = StatementEvaluatorMixin()
        check_import_permission = stmt_evaluator._check_import_permission
    except Exception:
        def check_import_permission(exported_value, importer_file, env):
            return True
    
except Exception as e:
    print(f"⚠️  Could not import from new evaluator structure: {e}")
    # Fallbacks (keep existing fallback code)
    class EvaluationError(Exception):
        pass

    EVAL_SUMMARY = {}
    NULL = None
    TRUE = True
    FALSE = False

    def _is_awaitable(obj: Any) -> bool:
        return False

    def _resolve_awaitable(obj: Any) -> Any:
        return obj

    # Try to reuse conversion helpers from object.py if available
    try:
        from ..object import File as _File

        def _python_to_zexus(value: Any) -> Any:
            try:
                return _File._python_to_zexus(value)
            except Exception:
                return value

        def _zexus_to_python(value: Any) -> Any:
            try:
                return _File._zexus_to_python(value)
            except Exception:
                return value

    except Exception:
        def _python_to_zexus(value: Any) -> Any:
            return value

        def _zexus_to_python(value: Any) -> Any:
            return value

    def _to_python(value: Any) -> Any:
        return value

    def _to_str(obj: Any) -> str:
        return str(obj)

    # Minimal evaluation helper fallbacks
    def is_error(obj: Any) -> bool:
        return isinstance(obj, EvaluationError)

    def is_truthy(obj: Any) -> bool:
        return bool(obj) and obj is not NULL

    # UPDATED: eval_node now points to evaluate
    def eval_node(node, env, *a, **kw):
        raise EvaluationError("eval_node: interpreter runtime not available")

    def evaluate(program, env, debug_mode=False):
        raise EvaluationError("evaluate: interpreter runtime not available")

    def eval_program(statements, env):
        raise EvaluationError("eval_program: interpreter runtime not available")

    def eval_block_statement(block, env):
        raise EvaluationError("eval_block_statement: interpreter runtime not available")

    def eval_expressions(expressions, env):
        raise EvaluationError("eval_expressions: interpreter runtime not available")

    def eval_identifier(node, env):
        raise EvaluationError("eval_identifier: interpreter runtime not available")

    def eval_if_expression(node, env):
        raise EvaluationError("eval_if_expression: interpreter runtime not available")

    def eval_infix_expression(op, left, right):
        raise EvaluationError("eval_infix_expression: interpreter runtime not available")

    def eval_prefix_expression(op, right):
        raise EvaluationError("eval_prefix_expression: interpreter runtime not available")

    def eval_assignment_expression(node, env):
        raise EvaluationError("eval_assignment_expression: interpreter runtime not available")

    def eval_let_statement_fixed(node, env, *a, **kw):
        raise EvaluationError("eval_let_statement_fixed: interpreter runtime not available")

    def eval_try_catch_statement_fixed(node, env, *a, **kw):
        raise EvaluationError("eval_try_catch_statement_fixed: interpreter runtime not available")

    def extend_function_env(fn, args):
        raise EvaluationError("extend_function_env: interpreter runtime not available")

    def unwrap_return_value(obj):
        return obj

    # Minimal renderer fallback
    RENDER_REGISTRY = {'screens': {}, 'components': {}, 'themes': {}, 'canvases': {}, 'current_theme': None}

    # Try to create small wrappers for builtin functions by reading from object.File when present
    try:
        from ..object import File as _File

        def _wrap_file_read_text(path):
            return _File.read_text(path)

        def _wrap_file_write_text(path, content):
            return _File.write_text(path, content)

        def _wrap_file_exists(path):
            return _File.exists(path)

        def _wrap_file_read_json(path):
            return _File.read_json(path)

        def _wrap_file_write_json(path, data):
            return _File.write_json(path, data)

        def _wrap_file_append(path, content):
            return _File.append_text(path, content)

        def _wrap_file_list_dir(path):
            return _File.list_directory(path)

    except Exception:
        _wrap_file_read_text = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('file read not available'))
        _wrap_file_write_text = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('file write not available'))
        _wrap_file_exists = lambda *a, **k: False
        _wrap_file_read_json = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('file read json not available'))
        _wrap_file_write_json = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('file write json not available'))
        _wrap_file_append = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('file append not available'))
        _wrap_file_list_dir = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('file list not available'))

    # Expose builtin_* names as top-level names (fallback wrappers)
    builtin_file_read_text = _wrap_file_read_text
    builtin_file_write_text = _wrap_file_write_text
    builtin_file_exists = _wrap_file_exists
    builtin_file_read_json = _wrap_file_read_json
    builtin_file_write_json = _wrap_file_write_json
    builtin_file_append = _wrap_file_append
    builtin_file_list_dir = _wrap_file_list_dir

    # Other builtin placeholders
    builtin_datetime_now = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('datetime not available'))
    builtin_timestamp = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('timestamp not available'))
    builtin_math_random = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('math random not available'))
    builtin_to_hex = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('to_hex not available'))
    builtin_from_hex = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('from_hex not available'))
    builtin_sqrt = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('sqrt not available'))
    builtin_debug_log = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('debug_log not available'))
    builtin_debug_trace = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('debug_trace not available'))
    builtin_string = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('string not available'))
    builtin_len = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('len not available'))
    builtin_first = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('first not available'))
    builtin_rest = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('rest not available'))
    builtin_push = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('push not available'))
    builtin_reduce = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('reduce not available'))
    builtin_map = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('map not available'))
    builtin_filter = lambda *a, **k: (_ for _ in ()).throw(EvaluationError('filter not available'))

    def apply_function(fn, args):
        raise EvaluationError("apply_function: runtime not available")

    def array_map(array_obj, lambda_fn, env=None):
        raise EvaluationError("array_map: runtime not available")

    def array_filter(array_obj, lambda_fn, env=None):
        raise EvaluationError("array_filter: runtime not available")

    def array_reduce(array_obj, lambda_fn, initial_value=None, env=None):
        raise EvaluationError("array_reduce: runtime not available")

    BUILTINS = {}

    def check_import_permission(exported_value, importer_file, env):
        return True

# Note: eval_node is now an alias for evaluate in the new structure
# For compatibility, we'll keep eval_node as a function that calls evaluate
def eval_node(node, env, *args, **kwargs):
    """Compatibility wrapper: eval_node now calls evaluate"""
    return evaluate(node, env, *args, **kwargs)

__all__ = [
    'EVAL_SUMMARY', 'EvaluationError', 'NULL', 'TRUE', 'FALSE',
    '_is_awaitable', '_resolve_awaitable', '_python_to_zexus', '_zexus_to_python',
    '_to_python', '_to_str', 'apply_function', 'array_map', 'array_filter', 'array_reduce',
    'BUILTINS', 'check_import_permission', 'evaluate', 'eval_node'  # UPDATED
]