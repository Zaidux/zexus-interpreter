# evaluator.py (FIXED VERSION)
import sys
import traceback
import json
import os
import asyncio
from . import zexus_ast
from .zexus_ast import (
    Program, ExpressionStatement, BlockStatement, ReturnStatement, LetStatement,
    ActionStatement, IfStatement, WhileStatement, ForEachStatement, MethodCallExpression,
    EmbeddedLiteral, PrintStatement, ScreenStatement, EmbeddedCodeStatement, UseStatement,
    ExactlyStatement, TryCatchStatement, IntegerLiteral, StringLiteral, ListLiteral, MapLiteral, Identifier,
    ActionLiteral, CallExpression, PrefixExpression, InfixExpression, IfExpression,
    Boolean as AST_Boolean, AssignmentExpression, PropertyAccessExpression,
    ExportStatement, LambdaExpression, FromStatement, ComponentStatement, ThemeStatement,
    DebugStatement, ExternalDeclaration, EntityStatement, SealStatement # Add all missing types
)

from .object import (
    Environment, Integer, Float, String, List, Map, Null, Boolean as BooleanObj,
    Builtin, Action, EmbeddedCode, ReturnValue, LambdaFunction, DateTime, Math, File, Debug,
    EvaluationError as ObjectEvaluationError
)

NULL, TRUE, FALSE = Null(), BooleanObj(True), BooleanObj(False)

# Registry for builtin functions (populated later)
builtins = {}

# Use the unified EvaluationError from object.py
EvaluationError = ObjectEvaluationError

# Helper to centralize error checks (includes the new FixedEvaluationError)
def is_error(obj):
    return isinstance(obj, (EvaluationError, ObjectEvaluationError))

# Summary counters for lightweight, 5-line summary logging when debug is off
EVAL_SUMMARY = {
    'parsed_statements': 0,
    'evaluated_statements': 0,
    'errors': 0,
    'async_tasks_run': 0,
    'max_statements_in_block': 0
}

def _is_awaitable(obj):
    try:
        return asyncio.iscoroutine(obj) or isinstance(obj, asyncio.Future)
    except Exception:
        return False

def _resolve_awaitable(obj):
    """If obj is a coroutine/future, run it to completion and return the result.
    If already in an event loop, return the coroutine (caller may handle it).
    """
    if _is_awaitable(obj):
        try:
            EVAL_SUMMARY['async_tasks_run'] += 1
            return asyncio.run(obj)
        except RuntimeError:
            # Already running event loop (e.g., invoked from async VM). Return as-is.
            return obj
    return obj

# === DEBUG FLAGS (controlled by user config) ===
from .config import config as zexus_config

def debug_log(message, data=None, level='debug'):
    """Conditional debug logging that respects the user's persistent config.
    level: 'debug' (very verbose), 'info', 'warn', 'error'
    """
    try:
        if not zexus_config.should_log(level):
            return
    except Exception:
        # If config fails for any reason, default to not logging
        return

    if data is not None:
        print(f"🔍 [EVAL DEBUG] {message}: {data}")
    else:
        print(f"🔍 [EVAL DEBUG] {message}")

# === FIXED HELPER FUNCTIONS ===

def eval_program(statements, env):
    debug_log("eval_program", f"Processing {len(statements)} statements")
    try:
        EVAL_SUMMARY['parsed_statements'] = max(EVAL_SUMMARY.get('parsed_statements', 0), len(statements))
    except Exception:
        pass

    result = NULL
    for i, stmt in enumerate(statements):
        debug_log(f"  Statement {i+1}", type(stmt).__name__)
        res = eval_node(stmt, env)
        res = _resolve_awaitable(res)
        EVAL_SUMMARY['evaluated_statements'] += 1
        if isinstance(res, ReturnValue):
            debug_log("  ReturnValue encountered", res.value)
            return res.value
        if is_error(res):
            debug_log("  Error encountered", res)
            try:
                EVAL_SUMMARY['errors'] += 1
            except Exception:
                pass
            return res
        result = res
    debug_log("eval_program completed", result)
    return result

def eval_assignment_expression(node, env):
    """Handle assignment expressions like: x = 5"""
    debug_log("eval_assignment_expression", f"Assigning to {node.name.value}")

    # CRITICAL FIX: Add sealed object check before assignment
    from .security import SealedObject
    target_obj = env.get(node.name.value)
    if isinstance(target_obj, SealedObject):
        return EvaluationError(f"Cannot assign to sealed object: {node.name.value}")

    value = eval_node(node.value, env)
    # Check using helper
    if is_error(value):
        debug_log("  Assignment error", value)
        return value
    env.set(node.name.value, value)
    debug_log("  Assignment successful", f"{node.name.value} = {value}")
    return value

def eval_block_statement(block, env):
    debug_log("eval_block_statement", f"Processing {len(block.statements)} statements in block")
    try:
        EVAL_SUMMARY['max_statements_in_block'] = max(EVAL_SUMMARY.get('max_statements_in_block', 0), len(block.statements))
    except Exception:
        pass

    result = NULL
    for stmt in block.statements:
        res = eval_node(stmt, env)
        res = _resolve_awaitable(res)
        EVAL_SUMMARY['evaluated_statements'] += 1
        if isinstance(res, (ReturnValue, EvaluationError, ObjectEvaluationError)):
            debug_log("  Block interrupted", res)
            if is_error(res):
                try:
                    EVAL_SUMMARY['errors'] += 1
                except Exception:
                    pass
            return res
        result = res
    debug_log("  Block completed", result)
    return result

def eval_expressions(expressions, env):
    debug_log("eval_expressions", f"Evaluating {len(expressions)} expressions")
    results = []
    for i, expr in enumerate(expressions):
        debug_log(f"  Expression {i+1}", type(expr).__name__)
        res = eval_node(expr, env)
        res = _resolve_awaitable(res)
        if is_error(res):
            debug_log("  Expression evaluation interrupted", res)
            try:
                EVAL_SUMMARY['errors'] += 1
            except Exception:
                pass
            return res
        results.append(res)
        EVAL_SUMMARY['evaluated_statements'] += 1
        debug_log(f"  Expression {i+1} result", res)
    debug_log("  All expressions evaluated", results)
    return results

def eval_identifier(node, env):
    debug_log("eval_identifier", f"Looking up: {node.value}")
    val = env.get(node.value)
    if val:
        debug_log("  Found in environment", f"{node.value} = {val}")
        return val
    # Check builtins
    builtin = builtins.get(node.value)
    if builtin:
        debug_log("  Found builtin", f"{node.value} = {builtin}")
        return builtin

    debug_log("  Identifier not found", node.value)
    # FIXED: Return the new FixedEvaluationError so downstream code won't crash if len() is used
    return EvaluationError(f"Identifier '{node.value}' not found")

def is_truthy(obj):
    # FIXED: Handle all error types
    if is_error(obj):
        return False
    result = not (obj == NULL or obj == FALSE)
    debug_log("is_truthy", f"{obj} -> {result}")
    return result

def eval_prefix_expression(operator, right):
    debug_log("eval_prefix_expression", f"{operator} {right}")
    if is_error(right): # Use is_error helper
        return right

    if operator == "!":
        result = eval_bang_operator_expression(right)
    elif operator == "-":
        result = eval_minus_prefix_operator_expression(right)
    else:
        result = EvaluationError(f"Unknown operator: {operator}{right.type()}")

    debug_log("  Prefix result", result)
    return result

def eval_bang_operator_expression(right):
    if right == TRUE:
        return FALSE
    elif right == FALSE:
        return TRUE
    elif right == NULL:
        return TRUE
    return FALSE

def eval_minus_prefix_operator_expression(right):
    if isinstance(right, Integer):
        return Integer(-right.value)
    elif isinstance(right, Float):
        return Float(-right.value)
    return EvaluationError(f"Unknown operator: -{right.type()}")

def eval_infix_expression(operator, left, right):
    debug_log("eval_infix_expression", f"{left} {operator} {right}")
    # Handle errors first
    if is_error(left): # Use is_error helper
        return left
    if is_error(right): # Use is_error helper
        return right

    # Logical operators
    if operator == "&&":
        result = TRUE if is_truthy(left) and is_truthy(right) else FALSE
    elif operator == "||":
        result = TRUE if is_truthy(left) or is_truthy(right) else FALSE
    elif operator == "==":
        # FIXED: Handle different object types properly
        if hasattr(left, 'value') and hasattr(right, 'value'):
            result = TRUE if left.value == right.value else FALSE
        else:
            result = TRUE if left == right else FALSE
    elif operator == "!=":
        if hasattr(left, 'value') and hasattr(right, 'value'):
            result = TRUE if left.value != right.value else FALSE
        else:
            result = TRUE if left != right else FALSE
    elif operator == "<=":
        if hasattr(left, 'value') and hasattr(right, 'value'):
            result = TRUE if left.value <= right.value else FALSE
        else:
            result = EvaluationError(f"Cannot compare: {left.type()} <= {right.type()}")
    elif operator == ">=":
        if hasattr(left, 'value') and hasattr(right, 'value'):
            result = TRUE if left.value >= right.value else FALSE
        else:
            result = EvaluationError(f"Cannot compare: {left.type()} >= {right.type()}")
    # Type-specific operations
    elif isinstance(left, Integer) and isinstance(right, Integer):
        result = eval_integer_infix_expression(operator, left, right)
    elif isinstance(left, Float) and isinstance(right, Float):
        result = eval_float_infix_expression(operator, left, right)
    elif isinstance(left, String) and isinstance(right, String):
        result = eval_string_infix_expression(operator, left, right)
    # NEW: Handle string concatenation with different types
    elif operator == "+":
        if isinstance(left, String):
            # Convert right to string and concatenate
            right_str = right.inspect() if not isinstance(right, String) else right.value
            result = String(left.value + str(right_str))
        elif isinstance(right, String):
            # Convert left to string and concatenate
            left_str = left.inspect() if not isinstance(left, String) else left.value
            result = String(str(left_str) + right.value)
        elif isinstance(left, Integer) and isinstance(right, Integer):
            result = Integer(left.value + right.value)
        elif isinstance(left, Float) and isinstance(right, Float):
            result = Float(left.value + right.value)
        elif isinstance(left, (Integer, Float)) and isinstance(right, (Integer, Float)):
            # Mixed numeric types
            left_val = left.value if isinstance(left, (Integer, Float)) else float(left.value) if hasattr(left, 'value') else 0
            right_val = right.value if isinstance(right, (Integer, Float)) else float(right.value) if hasattr(right, 'value') else 0
            result = Float(left_val + right_val)
        else:
            result = EvaluationError(f"Type mismatch: {left.type()} {operator} {right.type()}")
    else:
        result = EvaluationError(f"Type mismatch: {left.type()} {operator} {right.type()}")

    debug_log("  Infix result", result)
    return result

def eval_integer_infix_expression(operator, left, right):
    left_val = left.value
    right_val = right.value

    if operator == "+": return Integer(left_val + right_val)
    elif operator == "-": return Integer(left_val - right_val)
    elif operator == "*": return Integer(left_val * right_val)
    elif operator == "/":
        if right_val == 0:
            return EvaluationError("Division by zero")
        return Integer(left_val // right_val)
    elif operator == "%":
        if right_val == 0:
            return EvaluationError("Modulo by zero")
        return Integer(left_val % right_val)
    elif operator == "<": return TRUE if left_val < right_val else FALSE
    elif operator == ">": return TRUE if left_val > right_val else FALSE
    elif operator == "<=": return TRUE if left_val <= right_val else FALSE
    elif operator == ">=": return TRUE if left_val >= right_val else FALSE
    elif operator == "==": return TRUE if left_val == right_val else FALSE
    elif operator == "!=": return TRUE if left_val != right_val else FALSE
    return EvaluationError(f"Unknown integer operator: {operator}")

def eval_float_infix_expression(operator, left, right):
    left_val = left.value
    right_val = right.value

    if operator == "+": return Float(left_val + right_val)
    elif operator == "-": return Float(left_val - right_val)
    elif operator == "*": return Float(left_val * right_val)
    elif operator == "/":
        if right_val == 0:
            return EvaluationError("Division by zero")
        return Float(left_val / right_val)
    elif operator == "%":
        if right_val == 0:
            return EvaluationError("Modulo by zero")
        return Float(left_val % right_val)
    elif operator == "<": return TRUE if left_val < right_val else FALSE
    elif operator == ">": return TRUE if left_val > right_val else FALSE
    elif operator == "<=": return TRUE if left_val <= right_val else FALSE
    elif operator == ">=": return TRUE if left_val >= right_val else FALSE
    elif operator == "==": return TRUE if left_val == right_val else FALSE
    elif operator == "!=": return TRUE if left_val != right_val else FALSE
    return EvaluationError(f"Unknown float operator: {operator}")

def eval_string_infix_expression(operator, left, right):
    if operator == "+": return String(left.value + right.value)
    elif operator == "==": return TRUE if left.value == right.value else FALSE
    elif operator == "!=": return TRUE if left.value != right.value else FALSE
    return EvaluationError(f"Unknown string operator: {operator}")

def eval_if_expression(ie, env):
    debug_log("eval_if_expression", "Evaluating condition")
    condition = eval_node(ie.condition, env)
    if is_error(condition): # Use is_error helper
        return condition

    if is_truthy(condition):
        debug_log("  Condition true, evaluating consequence")
        return eval_node(ie.consequence, env)
    elif ie.alternative:
        debug_log("  Condition false, evaluating alternative")
        return eval_node(ie.alternative, env)
    debug_log("  Condition false, no alternative")
    return NULL

def apply_function(fn, args, call_site=None):
    # SAFE debug: avoid len() on errors
    arg_count = len(args) if isinstance(args, (list, tuple)) else ("err" if is_error(args) else "unknown")
    debug_log("apply_function", f"Calling {fn} with {arg_count} arguments: {args}")

    if isinstance(fn, (Action, LambdaFunction)):
        debug_log("  Calling user-defined function")
        extended_env = extend_function_env(fn, args)
        evaluated = eval_node(fn.body, extended_env)
        evaluated = _resolve_awaitable(evaluated)
        return unwrap_return_value(evaluated)
    elif isinstance(fn, Builtin):
        debug_log("  Calling builtin function", f"{fn.name} with args: {args}")
        try:
            # Builtin functions expect Zexus objects as args; ensure args is a list
            if not isinstance(args, (list, tuple)):
                return EvaluationError("Invalid arguments to builtin")
            result = fn.fn(*args)
            # If builtin returned a coroutine/future, resolve it when possible
            if _is_awaitable(result):
                result = _resolve_awaitable(result)
            debug_log("  Builtin result", result)
            return result
        except Exception as e:
            error = EvaluationError(f"Builtin function error: {str(e)}")
            debug_log("  Builtin error", error)
            return error
    error = EvaluationError(f"Not a function: {fn.type()}")
    debug_log("  Not a function error", error)
    return error

def extend_function_env(fn, args):
    env = Environment(outer=fn.env)
    for param, arg in zip(fn.parameters, args):
        env.set(param.value, arg)
    return env

def unwrap_return_value(obj):
    if isinstance(obj, ReturnValue):
        return obj.value
    return obj

# NEW: Lambda function evaluation
def eval_lambda_expression(node, env):
    debug_log("eval_lambda_expression", f"Creating lambda with {len(node.parameters)} parameters")
    return LambdaFunction(node.parameters, node.body, env)

# NEW: Array method implementations
def array_reduce(array_obj, lambda_fn, initial_value=None, env=None):
    """Implement array.reduce(lambda, initial_value)"""
    if not isinstance(array_obj, List):
        return EvaluationError("reduce() called on non-array object")
    if not isinstance(lambda_fn, (LambdaFunction, Action)):
        return EvaluationError("reduce() requires a lambda function as first argument")

    accumulator = initial_value if initial_value is not None else array_obj.elements[0] if array_obj.elements else NULL
    start_index = 0 if initial_value is not None else 1

    for i in range(start_index, len(array_obj.elements)):
        element = array_obj.elements[i]
        result = apply_function(lambda_fn, [accumulator, element])
        if is_error(result): # Use is_error helper
            return result
        accumulator = result

    return accumulator

def array_map(array_obj, lambda_fn, env=None):
    """Implement array.map(lambda)"""
    if not isinstance(array_obj, List):
        return EvaluationError("map() called on non-array object")
    if not isinstance(lambda_fn, (LambdaFunction, Action)):
        return EvaluationError("map() requires a lambda function")

    mapped_elements = []
    for element in array_obj.elements:
        result = apply_function(lambda_fn, [element])
        if is_error(result): # Use is_error helper
            return result
        mapped_elements.append(result)

    return List(mapped_elements)

def array_filter(array_obj, lambda_fn, env=None):
    """Implement array.filter(lambda)"""
    if not isinstance(array_obj, List):
        return EvaluationError("filter() called on non-array object")
    if not isinstance(lambda_fn, (LambdaFunction, Action)):
        return EvaluationError("filter() requires a lambda function")

    filtered_elements = []
    for element in array_obj.elements:
        result = apply_function(lambda_fn, [element])
        if is_error(result): # Use is_error helper
            return result
        if is_truthy(result):
            filtered_elements.append(element)

    return List(filtered_elements)

# NEW: Export system
def eval_export_statement(node, env):
    """Handle export statements - FIXED VERSION"""
    # Support single-name and multi-name ExportStatement
    names = []
    if hasattr(node, 'names') and node.names:
        names = [n.value if hasattr(n, 'value') else str(n) for n in node.names]
    elif hasattr(node, 'name') and node.name is not None:
        names = [node.name.value if hasattr(node.name, 'value') else str(node.name)]

    if not names:
        return EvaluationError("export: no identifiers provided to export")

    debug_log("eval_export_statement", f"Exporting {len(names)} names: {names}")

    for nm in names:
        value = env.get(nm)
        if not value:
            return EvaluationError(f"Cannot export undefined identifier: {nm}")

        debug_log(f"  Exporting '{nm}'", f"value: {value}")

        # CRITICAL FIX: Use the Environment's export method
        try:
            env.export(nm, value)
            debug_log(f"    Successfully exported via env.export()", "success")
        except Exception as e:
            debug_log(f"    env.export() failed", str(e))
            return EvaluationError(f"Failed to export '{nm}': {str(e)}")

    debug_log("eval_export_statement", f"All exports completed. Total exports: {len(env.exports)}")
    return NULL

def check_import_permission(exported_value, importer_file, env):
    """Check if importer has permission to access exported value"""
    # For now, implement basic file-based permission checking
    allowed_files = getattr(exported_value, '_allowed_files', [])
    permission = getattr(exported_value, '_export_permission', 'read_only')

    # If no restrictions, allow
    if not allowed_files or allowed_files == []:
        return True

    # Normalize paths for comparison
    importer_normalized = os.path.normpath(os.path.abspath(importer_file)) if importer_file else None

    for allowed in allowed_files:
        allowed_normalized = os.path.normpath(os.path.abspath(allowed)) if allowed else None
        if importer_normalized and allowed_normalized and importer_normalized == allowed_normalized:
            return True
        # Also allow if the allowed file is a substring (module path match)
        if importer_file and allowed in importer_file:
            return True

    return EvaluationError(f"File '{importer_file}' not authorized to import restricted export (permission: {permission})")

# === FIXED: JSON CONVERSION FUNCTIONS ===
def _zexus_to_python(value):
    """Convert Zexus objects to Python native types for JSON serialization"""
    debug_log("_zexus_to_python", f"Converting {type(value).__name__}: {value}")

    if isinstance(value, Map):
        python_dict = {}
        for key, val in value.pairs.items():
            python_key = key.inspect() if hasattr(key, 'inspect') else str(key)
            python_dict[python_key] = _zexus_to_python(val)
        debug_log("  Converted Map to dict", python_dict)
        return python_dict
    elif isinstance(value, List):
        python_list = [_zexus_to_python(item) for item in value.elements]
        debug_log("  Converted List to list", python_list)
        return python_list
    elif isinstance(value, String):
        debug_log("  Converted String to str", value.value)
        return value.value
    elif isinstance(value, Integer):
        debug_log("  Converted Integer to int", value.value)
        return value.value
    elif isinstance(value, Float):
        debug_log("  Converted Float to float", value.value)
        return value.value
    elif isinstance(value, BooleanObj):
        debug_log("  Converted Boolean to bool", value.value)
        return value.value
    elif value == NULL:
        debug_log("  Converted NULL to None")
        return None
    elif isinstance(value, Builtin):
        debug_log("  Converted Builtin to string")
        return f"<builtin: {value.name}>"
    elif isinstance(value, DateTime):
        debug_log("  Converted DateTime to float", value.timestamp)
        return value.timestamp
    else:
        debug_log("  Converted unknown to string", str(value))
        return str(value)

def _python_to_zexus(value):
    """Convert Python native types to Zexus objects"""
    debug_log("_python_to_zexus", f"Converting Python type: {type(value)}: {value}")

    if isinstance(value, dict):
        pairs = {}
        for k, v in value.items():
            pairs[k] = _python_to_zexus(v)
        debug_log("  Converted dict to Map", pairs)
        return Map(pairs)
    elif isinstance(value, list):
        zexus_list = List([_python_to_zexus(item) for item in value])
        debug_log("  Converted list to List", zexus_list)
        return zexus_list
    elif isinstance(value, str):
        debug_log("  Converted str to String", value)
        return String(value)
    elif isinstance(value, int):
        debug_log("  Converted int to Integer", value)
        return Integer(value)
    elif isinstance(value, float):
        debug_log("  Converted float to Float", value)
        return Float(value)
    elif isinstance(value, bool):
        debug_log("  Converted bool to Boolean", value)
        return BooleanObj(value)
    elif value is None:
        debug_log("  Converted None to NULL")
        return NULL
    else:
        debug_log("  Converted unknown to String", str(value))
        return String(str(value))

# === FIXED BUILTIN FUNCTIONS FOR PHASE 1 ===

def builtin_datetime_now(*args):
    debug_log("builtin_datetime_now", "called")
    return DateTime.now()

def builtin_timestamp(*args):
    debug_log("builtin_timestamp", f"called with {len(args)} args")
    if len(args) == 0:
        return DateTime.now().to_timestamp()
    elif len(args) == 1 and isinstance(args[0], DateTime):
        return args[0].to_timestamp()
    return EvaluationError("timestamp() takes 0 or 1 DateTime argument")

def builtin_math_random(*args):
    debug_log("builtin_math_random", f"called with {len(args)} args")
    if len(args) == 0:
        return Math.random_int(0, 100)
    elif len(args) == 1 and isinstance(args[0], Integer):
        return Math.random_int(0, args[0].value)
    elif len(args) == 2 and all(isinstance(a, Integer) for a in args):
        return Math.random_int(args[0].value, args[1].value)
    return EvaluationError("random() takes 0, 1, or 2 integer arguments")

def builtin_to_hex(*args):
    debug_log("builtin_to_hex", f"called with {args}")
    if len(args) != 1:
        return EvaluationError("to_hex() takes exactly 1 argument")
    return Math.to_hex_string(args[0])

def builtin_from_hex(*args):
    debug_log("builtin_from_hex", f"called with {args}")
    if len(args) != 1 or not isinstance(args[0], String):
        return EvaluationError("from_hex() takes exactly 1 string argument")
    return Math.hex_to_int(args[0])

def builtin_sqrt(*args):
    debug_log("builtin_sqrt", f"called with {args}")
    if len(args) != 1:
        return EvaluationError("sqrt() takes exactly 1 argument")
    return Math.sqrt(args[0])

# File I/O builtins - FIXED VERSIONS
def builtin_file_read_text(*args):
    debug_log("builtin_file_read_text", f"called with {args}")
    if len(args) != 1 or not isinstance(args[0], String):
        return EvaluationError("file_read_text() takes exactly 1 string argument")
    return File.read_text(args[0])

def builtin_file_write_text(*args):
    debug_log("builtin_file_write_text", f"called with {args}")
    if len(args) != 2 or not all(isinstance(a, String) for a in args):
        return EvaluationError("file_write_text() takes exactly 2 string arguments")
    return File.write_text(args[0], args[1])

def builtin_file_exists(*args):
    debug_log("builtin_file_exists", f"called with {args}")
    if len(args) != 1 or not isinstance(args[0], String):
        return EvaluationError("file_exists() takes exactly 1 string argument")
    return File.exists(args[0])

def builtin_file_read_json(*args):
    debug_log("builtin_file_read_json", f"called with {args}")
    if len(args) != 1 or not isinstance(args[0], String):
        return EvaluationError("file_read_json() takes exactly 1 string argument")
    return File.read_json(args[0])

# FIXED: JSON write function - CRITICAL FIX
def builtin_file_write_json(*args):
    debug_log("builtin_file_write_json", f"called with {args}")
    if len(args) != 2 or not isinstance(args[0], String):
        return EvaluationError("file_write_json() takes path string and data")

    path = args[0]
    data = args[1]

    debug_log("  JSON write - path", path.value if isinstance(path, String) else path)
    debug_log("  JSON write - data type", type(data).__name__)
    debug_log("  JSON write - data value", data)

    try:
        # FIX: Use the File.write_json method which properly handles conversion
        return File.write_json(path, data)
    except Exception as e:
        return EvaluationError(f"JSON write error: {str(e)}")

def builtin_file_append(*args):
    debug_log("builtin_file_append", f"called with {args}")
    if len(args) != 2 or not all(isinstance(a, String) for a in args):
        return EvaluationError("file_append() takes exactly 2 string arguments")
    return File.append_text(args[0], args[1])

def builtin_file_list_dir(*args):
    debug_log("builtin_file_list_dir", f"called with {args}")
    if len(args) != 1 or not isinstance(args[0], String):
        return EvaluationError("file_list_dir() takes exactly 1 string argument")
    return File.list_directory(args[0])

# Debug builtins
def builtin_debug_log(*args):
    debug_log("builtin_debug_log", f"called with {len(args)} args")
    if len(args) == 0:
        return EvaluationError("debug_log() requires at least a message")
    message = args[0]
    value = args[1] if len(args) > 1 else None
    return Debug.log(message, value)

def builtin_debug_trace(*args):
    debug_log("builtin_debug_trace", f"called with {args}")
    if len(args) != 1 or not isinstance(args[0], String):
        return EvaluationError("debug_trace() takes exactly 1 string argument")
    return Debug.trace(args[0])

# FIXED: String function to handle all Zexus types
def builtin_string(*args):
    debug_log("builtin_string", f"called with {args}")
    if len(args) != 1:
        return EvaluationError(f"string() takes exactly 1 argument ({len(args)} given)")
    arg = args[0]

    if isinstance(arg, Integer):
        result = String(str(arg.value))
    elif isinstance(arg, Float):
        result = String(str(arg.value))
    elif isinstance(arg, String):
        result = arg
    elif isinstance(arg, BooleanObj):
        result = String("true" if arg.value else "false")
    elif isinstance(arg, (List, Map)):
        result = String(arg.inspect())
    elif isinstance(arg, Builtin):
        result = String(f"<built-in function: {arg.name}>")
    elif isinstance(arg, DateTime):
        result = String(f"<DateTime: {arg.timestamp}>")
    elif is_error(arg): # Use is_error helper
        result = String(str(arg))
    elif arg == NULL:
        result = String("null")
    else:
        result = String("unknown")

    debug_log("  builtin_string result", result)
    return result

# Other existing builtin functions
def builtin_len(*args):
    debug_log("builtin_len", f"called with {args}")
    if len(args) != 1:
        return EvaluationError(f"len() takes exactly 1 argument ({len(args)} given)")
    arg = args[0]
    if isinstance(arg, String):
        return Integer(len(arg.value))
    elif isinstance(arg, List):
        return Integer(len(arg.elements))
    return EvaluationError(f"len() not supported for type {arg.type()}")

def builtin_first(*args):
    debug_log("builtin_first", f"called with {args}")
    if len(args) != 1:
        return EvaluationError(f"first() takes exactly 1 argument ({len(args)} given)")
    if not isinstance(args[0], List):
        return EvaluationError("first() expects a list")
    list_obj = args[0]
    return list_obj.elements[0] if list_obj.elements else NULL

def builtin_rest(*args):
    debug_log("builtin_rest", f"called with {args}")
    if len(args) != 1:
        return EvaluationError(f"rest() takes exactly 1 argument ({len(args)} given)")
    if not isinstance(args[0], List):
        return EvaluationError("rest() expects a list")
    list_obj = args[0]
    return List(list_obj.elements[1:]) if len(list_obj.elements) > 0 else List([])

def builtin_push(*args):
    debug_log("builtin_push", f"called with {args}")
    if len(args) != 2:
        return EvaluationError(f"push() takes exactly 2 arguments ({len(args)} given)")
    if not isinstance(args[0], List):
        return EvaluationError("push() expects a list as first argument")
    list_obj = args[0]
    new_elements = list_obj.elements + [args[1]]
    return List(new_elements)

def builtin_reduce(*args):
    """Built-in reduce function for arrays"""
    debug_log("builtin_reduce", f"called with {args}")
    if len(args) < 2 or len(args) > 3:
        return EvaluationError("reduce() takes 2 or 3 arguments (array, lambda[, initial])")
    array_obj, lambda_fn = args[0], args[1]
    initial = args[2] if len(args) == 3 else None
    return array_reduce(array_obj, lambda_fn, initial)

def builtin_map(*args):
    """Built-in map function for arrays"""
    debug_log("builtin_map", f"called with {args}")
    if len(args) != 2:
        return EvaluationError("map() takes 2 arguments (array, lambda)")
    return array_map(args[0], args[1])

def builtin_filter(*args):
    """Built-in filter function for arrays"""
    debug_log("builtin_filter", f"called with {args}")
    if len(args) != 2:
        return EvaluationError("filter() takes 2 arguments (array, lambda)")
    return array_filter(args[0], args[1])

# Register core builtins
try:
    builtins.update({
        "now": Builtin(builtin_datetime_now, "now"),
        "timestamp": Builtin(builtin_timestamp, "timestamp"),
        "random": Builtin(builtin_math_random, "random"),
        "to_hex": Builtin(builtin_to_hex, "to_hex"),
        "from_hex": Builtin(builtin_from_hex, "from_hex"),
        "sqrt": Builtin(builtin_sqrt, "sqrt"),

        "file_read_text": Builtin(builtin_file_read_text, "file_read_text"),
        "file_write_text": Builtin(builtin_file_write_text, "file_write_text"),
        "file_exists": Builtin(builtin_file_exists, "file_exists"),
        "file_read_json": Builtin(builtin_file_read_json, "file_read_json"),
        "file_write_json": Builtin(builtin_file_write_json, "file_write_json"),
        "file_append": Builtin(builtin_file_append, "file_append"),
        "file_list_dir": Builtin(builtin_file_list_dir, "file_list_dir"),

        "debug_log": Builtin(builtin_debug_log, "debug_log"),
        "debug_trace": Builtin(builtin_debug_trace, "debug_trace"),

        "string": Builtin(builtin_string, "string"),
        "len": Builtin(builtin_len, "len"),
        "first": Builtin(builtin_first, "first"),
        "rest": Builtin(builtin_rest, "rest"),
        "push": Builtin(builtin_push, "push"),
        "reduce": Builtin(builtin_reduce, "reduce"),
        "map": Builtin(builtin_map, "map"),
        "filter": Builtin(builtin_filter, "filter"),
    })
except NameError:
    # If Builtin class is not available at import time, keep pending mapping
    try:
        __CORE_BUILTINS_PENDING = {
            "now": builtin_datetime_now,
            "timestamp": builtin_timestamp,
            "random": builtin_math_random,
            "to_hex": builtin_to_hex,
            "from_hex": builtin_from_hex,
            "sqrt": builtin_sqrt,
            "file_read_text": builtin_file_read_text,
            "file_write_text": builtin_file_write_text,
            "file_exists": builtin_file_exists,
            "file_read_json": builtin_file_read_json,
            "file_write_json": builtin_file_write_json,
            "file_append": builtin_file_append,
            "file_list_dir": builtin_file_list_dir,
            "debug_log": builtin_debug_log,
            "debug_trace": builtin_debug_trace,
            "string": builtin_string,
            "len": builtin_len,
            "first": builtin_first,
            "rest": builtin_rest,
            "push": builtin_push,
            "reduce": builtin_reduce,
            "map": builtin_map,
            "filter": builtin_filter,
        }
    except Exception:
        pass

# --- RENDERER REGISTRY & HELPERS ---------------------------------------
# Try to use the real renderer backend if available, otherwise keep a safe registry.
try:
        from renderer import backend as _BACKEND
        _BACKEND_AVAILABLE = True
except Exception:
        _BACKEND_AVAILABLE = False
        _BACKEND = None

# Local fallback registry and palette (used only if backend unavailable)
RENDER_REGISTRY = {
        'screens': {},
        'components': {},
        'themes': {},
        'canvases': {},
        'current_theme': None
}

# Helper converters: Zexus object -> Python native/simple printable
def _to_str(obj):
        if isinstance(obj, String):
                return obj.value
        if isinstance(obj, (Integer, Float)):
                return str(obj.value)
        return getattr(obj, 'inspect', lambda: str(obj))()

def _to_python(obj):
        """Convert Map/List/Zexus primitives into Python primitives for registry storage."""
        if obj is None:
                return None
        if isinstance(obj, Map):
                py = {}
                for k, v in obj.pairs.items():
                        key = k.inspect() if hasattr(k, 'inspect') else str(k)
                        py[key] = _to_python(v)
                return py
        if isinstance(obj, List):
                return [_to_python(e) for e in obj.elements]
        if isinstance(obj, String):
                return obj.value
        if isinstance(obj, Integer):
                return obj.value
        if isinstance(obj, Float):
                return obj.value
        if obj == NULL:
                return None
        return getattr(obj, 'inspect', lambda: str(obj))()

# --- RENDERER BUILTIN IMPLEMENTATIONS (delegating to backend if present) ---

def builtin_mix(*args):
        """mix(colorA, colorB, ratio) -> String"""
        if len(args) != 3:
                return EvaluationError("mix() expects 3 arguments (colorA, colorB, ratio)")
        a, b, ratio = args
        a_name = _to_str(a); b_name = _to_str(b)
        try:
                ratio_val = float(ratio.value) if isinstance(ratio, (Integer, Float)) else float(str(ratio))
        except Exception:
                ratio_val = 0.5

        if _BACKEND_AVAILABLE:
                try:
                        res = _BACKEND.mix(a_name, b_name, ratio_val)
                        return String(str(res))
                except Exception as e:
                        return String(f"mix({a_name},{b_name},{ratio_val})")
        # fallback: store mix representation locally
        return String(f"mix({a_name},{b_name},{ratio_val})")

def builtin_define_screen(*args):
        if len(args) < 1:
                return EvaluationError("define_screen() requires at least a name")
        name = _to_str(args[0])
        props = _to_python(args[1]) if len(args) > 1 else {}
        if _BACKEND_AVAILABLE:
                try:
                        _BACKEND.define_screen(name, props)
                        return NULL
                except Exception as e:
                        return EvaluationError(str(e))
        # fallback
        RENDER_REGISTRY['screens'].setdefault(name, {'properties': props, 'components': [], 'theme': None})
        return NULL

def builtin_define_component(*args):
        if len(args) < 1:
                return EvaluationError("define_component() requires at least a name")
        name = _to_str(args[0]); props = _to_python(args[1]) if len(args) > 1 else {}
        if _BACKEND_AVAILABLE:
                try:
                        _BACKEND.define_component(name, props)
                        return NULL
                except Exception as e:
                        return EvaluationError(str(e))
        RENDER_REGISTRY['components'][name] = props
        return NULL

def builtin_add_to_screen(*args):
        if len(args) != 2:
                return EvaluationError("add_to_screen() requires (screen_name, component_name)")
        screen = _to_str(args[0]); comp = _to_str(args[1])
        if _BACKEND_AVAILABLE:
                try:
                        _BACKEND.add_to_screen(screen, comp)
                        return NULL
                except Exception as e:
                        return EvaluationError(str(e))
        if screen not in RENDER_REGISTRY['screens']:
                return EvaluationError(f"Screen '{screen}' not found")
        RENDER_REGISTRY['screens'][screen]['components'].append(comp)
        return NULL

def builtin_render_screen(*args):
        if len(args) != 1:
                return EvaluationError("render_screen() requires exactly 1 argument")
        name = _to_str(args[0])
        if _BACKEND_AVAILABLE:
                try:
                        out = _BACKEND.render_screen(name)
                        return String(str(out))
                except Exception as e:
                        return String(f"<render error: {str(e)}>")
        screen = RENDER_REGISTRY['screens'].get(name)
        if not screen:
                return String(f"<missing screen: {name}>")
        props = screen.get('properties', {}); comps = screen.get('components', [])
        theme = screen.get('theme') or RENDER_REGISTRY.get('current_theme')
        return String(f"Screen:{name} props={props} components={comps} theme={theme}")

def builtin_set_theme(*args):
        if len(args) == 1:
                theme_name = _to_str(args[0])
                if _BACKEND_AVAILABLE:
                        try:
                                _BACKEND.set_theme(theme_name)
                                return NULL
                        except Exception as e:
                                return EvaluationError(str(e))
                RENDER_REGISTRY['current_theme'] = theme_name
                return NULL
        if len(args) == 2:
                target = _to_str(args[0]); theme_name = _to_str(args[1])
                if _BACKEND_AVAILABLE:
                        try:
                                _BACKEND.set_theme(target, theme_name)
                                return NULL
                        except Exception as e:
                                return EvaluationError(str(e))
                if target in RENDER_REGISTRY['screens']:
                        RENDER_REGISTRY['screens'][target]['theme'] = theme_name
                        return NULL
                RENDER_REGISTRY['themes'].setdefault(theme_name, {})
                return NULL
        return EvaluationError("set_theme() requires 1 (theme) or 2 (target,theme) args")

def builtin_create_canvas(*args):
        if len(args) != 2:
                return EvaluationError("create_canvas(width, height)")
        try:
                wid = int(args[0].value) if isinstance(args[0], Integer) else int(str(args[0]))
                hei = int(args[1].value) if isinstance(args[1], Integer) else int(str(args[1]))
        except Exception:
                return EvaluationError("Invalid numeric arguments to create_canvas()")
        if _BACKEND_AVAILABLE:
                try:
                        cid = _BACKEND.create_canvas(wid, hei)
                        return String(str(cid))
                except Exception as e:
                        return EvaluationError(str(e))
        cid = f"canvas_{len(RENDER_REGISTRY['canvases'])+1}"
        RENDER_REGISTRY['canvases'][cid] = {'width': wid, 'height': hei, 'draw_ops': []}
        return String(cid)

def builtin_draw_line(*args):
        if len(args) != 5:
                return EvaluationError("draw_line(canvas_id,x1,y1,x2,y2)")
        cid = _to_str(args[0])
        try:
                coords = [int(a.value) if isinstance(a, Integer) else int(str(a)) for a in args[1:]]
        except Exception:
                return EvaluationError("Invalid coordinates in draw_line()")
        if _BACKEND_AVAILABLE:
                try:
                        _BACKEND.draw_line(cid, *coords)
                        return NULL
                except Exception as e:
                        return EvaluationError(str(e))
        canvas = RENDER_REGISTRY['canvases'].get(cid)
        if not canvas:
                return EvaluationError(f"Canvas {cid} not found")
        canvas['draw_ops'].append(('line', coords))
        return NULL

def builtin_draw_text(*args):
        if len(args) != 4:
                return EvaluationError("draw_text(canvas_id,x,y,text)")
        cid = _to_str(args[0])
        try:
                x = int(args[1].value) if isinstance(args[1], Integer) else int(str(args[1]))
                y = int(args[2].value) if isinstance(args[2], Integer) else int(str(args[2]))
        except Exception:
                return EvaluationError("Invalid coordinates in draw_text()")
        text = _to_str(args[3])
        if _BACKEND_AVAILABLE:
                try:
                        _BACKEND.draw_text(cid, x, y, text)
                        return NULL
                except Exception as e:
                        return EvaluationError(str(e))
        canvas = RENDER_REGISTRY['canvases'].get(cid)
        if not canvas:
                return EvaluationError(f"Canvas {cid} not found")
        canvas['draw_ops'].append(('text', (x, y, text)))
        return NULL

# --- REGISTER RENDERER BUILTINS INTO builtins DICTIONARY ---------------------
# (leave the existing update logic in place; this code will add entries if `builtins` exists)
try:
        builtins.update({
                "mix": Builtin(builtin_mix, "mix"),
                "define_screen": Builtin(builtin_define_screen, "define_screen"),
                "define_component": Builtin(builtin_define_component, "define_component"),
                "add_to_screen": Builtin(builtin_add_to_screen, "add_to_screen"),
                "render_screen": Builtin(builtin_render_screen, "render_screen"),
                "set_theme": Builtin(builtin_set_theme, "set_theme"),
                "create_canvas": Builtin(builtin_create_canvas, "create_canvas"),
                "draw_line": Builtin(builtin_draw_line, "draw_line"),
                "draw_text": Builtin(builtin_draw_text, "draw_text"),
        })
except NameError:
        # keep the pending dict as before; other code will merge later
        try:
                __RENDERER_BUILTINS_PENDING = {
                        "mix": ("builtin", builtin_mix),
                        "define_screen": ("builtin", builtin_define_screen),
                        "define_component": ("builtin", builtin_define_component),
                        "add_to_screen": ("builtin", builtin_add_to_screen),
                        "render_screen": ("builtin", builtin_render_screen),
                        "set_theme": ("builtin", builtin_set_theme),
                        "create_canvas": ("builtin", builtin_create_canvas),
                        "draw_line": ("builtin", builtin_draw_line),
                        "draw_text": ("builtin", builtin_draw_text),
                }
        except Exception:
                pass

# === CRITICAL FIX: Enhanced LetStatement Evaluation ===
def eval_let_statement_fixed(node, env, stack_trace):
    """FIXED: Evaluate let statement without circular dependencies"""
    debug_log("eval_let_statement", f"let {node.name.value}")

    # CRITICAL FIX: Evaluate the value FIRST, before setting the variable
    value = eval_node(node.value, env, stack_trace)
    if is_error(value): # Use is_error helper
        debug_log("  Let statement value evaluation error", value)
        return value

    # THEN set the variable in the environment
    env.set(node.name.value, value)
    debug_log("  Let statement successful", f"{node.name.value} = {value}")
    return NULL

# === CRITICAL FIX: Enhanced Try-Catch Evaluation ===
def eval_try_catch_statement_fixed(node, env, stack_trace):
    """FIXED: Evaluate try-catch statement with proper error handling"""
    debug_log("eval_try_catch_statement", f"error_var: {node.error_variable.value if node.error_variable else 'error'}")
    try:
        debug_log("    Executing try block")
        result = eval_node(node.try_block, env, stack_trace)
        if is_error(result): # Use is_error helper
            debug_log("    Try block returned error", result)
            catch_env = Environment(outer=env)
            error_var_name = node.error_variable.value if node.error_variable else "error"
            error_value = String(str(result))
            catch_env.set(error_var_name, error_value)
            debug_log(f"    Set error variable '{error_var_name}' to: {error_value}")
            debug_log("    Executing catch block")
            return eval_node(node.catch_block, catch_env, stack_trace)
        else:
            debug_log("    Try block completed successfully")
            return result
    except Exception as e:
        debug_log(f"    Exception caught in try block: {e}")
        catch_env = Environment(outer=env)
        error_var_name = node.error_variable.value if node.error_variable else "error"
        error_value = String(str(e))
        catch_env.set(error_var_name, error_value)
        debug_log(f"    Set error variable '{error_var_name}' to: {error_value}")
        debug_log("    Executing catch block")
        return eval_node(node.catch_block, catch_env, stack_trace)

# === ENHANCED MAIN EVAL_NODE FUNCTION WITH CRITICAL FIXES ===
def eval_node(node, env, stack_trace=None):
    if node is None:
        debug_log("eval_node", "Node is None, returning NULL")
        return NULL

    node_type = type(node)
    stack_trace = stack_trace or []

    # Add to stack trace for better error reporting
    current_frame = f"  at {node_type.__name__}"
    if hasattr(node, 'token') and node.token:
        current_frame += f" (line {node.token.line})"
    stack_trace.append(current_frame)

    debug_log("eval_node", f"Processing {node_type.__name__}")

    try:
        # Statements
        if node_type == Program:
            debug_log("  Program node", f"{len(node.statements)} statements")
            return eval_program(node.statements, env)

        elif node_type == ExpressionStatement:
            debug_log("  ExpressionStatement node")
            return eval_node(node.expression, env, stack_trace)

        elif node_type == BlockStatement:
            debug_log("  BlockStatement node", f"{len(node.statements)} statements")
            return eval_block_statement(node, env)

        elif node_type == ReturnStatement:
            debug_log("  ReturnStatement node")
            val = eval_node(node.return_value, env, stack_trace)
            if is_error(val): # Use is_error helper
                return val
            return ReturnValue(val)

        # CRITICAL FIX: Use the fixed let statement evaluation
        elif node_type == LetStatement:
            return eval_let_statement_fixed(node, env, stack_trace)

        elif node_type == ActionStatement:
            debug_log("  ActionStatement node", f"action {node.name.value}")
            action_obj = Action(node.parameters, node.body, env)
            env.set(node.name.value, action_obj)
            return NULL

        # NEW: Export statement
        elif node_type == ExportStatement:
            # safe logging for single/multi-name export statements
            try:
                if hasattr(node, 'names') and node.names:
                    names_text = ','.join([n.value if hasattr(n, 'value') else str(n) for n in node.names])
                elif hasattr(node, 'name') and node.name is not None:
                    names_text = getattr(node.name, 'value', str(node.name))
                else:
                    names_text = '<no-names>'
            except Exception:
                names_text = '<unknown>'
            debug_log("  ExportStatement node", f"export {names_text}")
            return eval_export_statement(node, env)

        elif node_type == IfStatement:
            debug_log("  IfStatement node")
            condition = eval_node(node.condition, env, stack_trace)
            if is_error(condition): # Use is_error helper
                return condition
            if is_truthy(condition):
                debug_log("    If condition true")
                return eval_node(node.consequence, env, stack_trace)
            elif node.alternative is not None:
                debug_log("    If condition false, has alternative")
                return eval_node(node.alternative, env, stack_trace)
            debug_log("    If condition false, no alternative")
            return NULL

        elif node_type == WhileStatement:
            debug_log("  WhileStatement node")
            result = NULL
            while True:
                condition = eval_node(node.condition, env, stack_trace)
                if is_error(condition): # Use is_error helper
                    return condition
                if not is_truthy(condition):
                    break
                result = eval_node(node.body, env, stack_trace)
                if isinstance(result, (ReturnValue, EvaluationError, ObjectEvaluationError)):
                    break
            return result

        elif node_type == ForEachStatement:
            debug_log("  ForEachStatement node", f"for each {node.item.value}")
            iterable = eval_node(node.iterable, env, stack_trace)
            if is_error(iterable): # Use is_error helper
                return iterable
            if not isinstance(iterable, List):
                return EvaluationError("for-each loop expected list")

            result = NULL
            for element in iterable.elements:
                env.set(node.item.value, element)
                result = eval_node(node.body, env, stack_trace)
                if isinstance(result, (ReturnValue, EvaluationError, ObjectEvaluationError)):
                    break

            return result

        # CRITICAL FIX: Use the fixed try-catch evaluation
        elif node_type == TryCatchStatement:
            return eval_try_catch_statement_fixed(node, env, stack_trace)

        elif node_type == AssignmentExpression:
            debug_log("  AssignmentExpression node")
            return eval_assignment_expression(node, env)

        elif node_type == PropertyAccessExpression:
            debug_log("  PropertyAccessExpression node", f"{node.object}.{node.property}")
            obj = eval_node(node.object, env, stack_trace)
            if is_error(obj): # Use is_error helper
                return obj
            property_name = node.property.value

            if isinstance(obj, EmbeddedCode):
                if property_name == "code":
                    return String(obj.code)
                elif property_name == "language":
                    return String(obj.language)
            # Default behavior for property access: return NULL if not found
            # (eval_identifier would return an error, but property access
            # might just mean a missing property in dynamic objects like Maps)
            # However, for entity instances, we would expect a proper getter.
            if isinstance(obj, Map):
                return obj.pairs.get(property_name, NULL)
            # You might have a specific `EntityInstance` or similar object
            # that implements a `get` method for properties.
            elif hasattr(obj, 'get') and callable(obj.get):
                return obj.get(property_name)
            
            return NULL # Or raise an error if strict property access is desired

        elif node_type == AST_Boolean:
            debug_log("  Boolean node", f"value: {node.value}")
            return TRUE if node.value else FALSE

        # NEW: Lambda expression
        elif node_type == LambdaExpression:
            debug_log("  LambdaExpression node")
            return eval_lambda_expression(node, env)

        elif node_type == MethodCallExpression:
            debug_log("  MethodCallExpression node", f"{node.object}.{node.method}")
            obj = eval_node(node.object, env, stack_trace)
            if is_error(obj): # Use is_error helper
                return obj
            method_name = node.method.value

            # Handle array methods with lambdas
            if isinstance(obj, List):
                args = eval_expressions(node.arguments, env)
                if is_error(args): # Use is_error helper
                    return args

                if method_name == "reduce":
                    if len(args) < 1:
                        return EvaluationError("reduce() requires at least a lambda function")
                    lambda_fn = args[0]
                    initial = args[1] if len(args) > 1 else None
                    return array_reduce(obj, lambda_fn, initial, env)
                elif method_name == "map":
                    if len(args) != 1:
                        return EvaluationError("map() requires exactly one lambda function")
                    return array_map(obj, args[0], env)
                elif method_name == "filter":
                    if len(args) != 1:
                        return EvaluationError("filter() requires exactly one lambda function")
                    return array_filter(obj, args[0], env)

            # Handle embedded code method calls
            if isinstance(obj, EmbeddedCode):
                args = eval_expressions(node.arguments, env)
                if is_error(args): # Use is_error helper
                    return args
                # Simplified embedded execution
                print(f"[EMBED] Executing {obj.language}.{method_name}")
                return Integer(42) # Placeholder result

            return EvaluationError(f"Method '{method_name}' not supported for {obj.type()}")

        elif node_type == EmbeddedLiteral:
            debug_log("  EmbeddedLiteral node")
            return EmbeddedCode("embedded_block", node.language, node.code)

        elif node_type == PrintStatement:
            debug_log("  PrintStatement node")
            val = eval_node(node.value, env, stack_trace)
            if is_error(val): # Use is_error helper
                # Print errors to stderr but don't stop execution
                print(f"❌ Error: {val}", file=sys.stderr)
                return NULL
            debug_log("    Printing value", val)
            print(val.inspect())
            return NULL

        elif node_type == ScreenStatement:
            debug_log("  ScreenStatement node", node.name.value)
            print(f"[RENDER] Screen: {node.name.value}")
            return NULL

        elif node_type == EmbeddedCodeStatement:
            debug_log("  EmbeddedCodeStatement node", node.name.value)
            embedded_obj = EmbeddedCode(node.name.value, node.language, node.code)
            env.set(node.name.value, embedded_obj)
            return NULL

        elif node_type == UseStatement:
            debug_log("  UseStatement node", node.file_path)
            from .module_cache import get_cached_module, cache_module, get_module_candidates, normalize_path

            # Extract file path from node
            file_path_attr = getattr(node, 'file_path', None) or getattr(node, 'embedded_ref', None)
            if isinstance(file_path_attr, StringLiteral):
                file_path = file_path_attr.value
            else:
                file_path = file_path_attr

            if not file_path:
                return EvaluationError("use: missing file path")

            debug_log("  UseStatement loading", file_path)

            # Try to get normalized path and candidates
            normalized_path = normalize_path(file_path)
            candidates = get_module_candidates(file_path)

            # Check cache first
            cached_env = get_cached_module(normalized_path)
            if cached_env:
                debug_log("  Found module in cache", normalized_path)
                module_env = cached_env
                loaded = True
            else:
                # Not in cache, load from filesystem
                debug_log("  Module not in cache, loading from disk")
                module_env = Environment()
                loaded = False
                parse_errors = []

                # CRITICAL: Cache a placeholder before evaluation to break circular dependencies.
                # This allows module A -> module B -> module A to resolve to the same Environment
                # instance rather than recursing indefinitely.
                try:
                    cache_module(normalized_path, module_env)
                except Exception:
                    # Best-effort: if caching fails, continue without placeholder
                    pass

                for candidate in candidates:
                    try:
                        print(f"[MOD-DEBUG] trying candidate: {candidate}")
                        if not os.path.exists(candidate):
                            print(f"[MOD-DEBUG] candidate does not exist: {candidate}")
                            continue
                        debug_log("  Found module file", candidate)
                        with open(candidate, 'r', encoding='utf-8') as f:
                            code = f.read()

                        # Import parser/lexer here to avoid top-level circular imports
                        from .lexer import Lexer
                        from .parser import Parser
                        lexer = Lexer(code)
                        parser = Parser(lexer)
                        program = parser.parse_program()

                        if getattr(parser, 'errors', None):
                            print(f"[MOD-DEBUG] parser errors for {candidate}: {parser.errors}")
                            parse_errors.append((candidate, parser.errors))
                            continue

                        # Evaluate module into its own environment
                        eval_node(program, module_env)
                        # Cache the successfully loaded module (overwrite placeholder)
                        cache_module(normalized_path, module_env)
                        loaded = True
                        debug_log("  Module loaded and cached", normalized_path)
                        print(f"[MOD-DEBUG] module evaluated: {candidate}")
                        break
                    except Exception as e:
                        print(f"[MOD-DEBUG] exception loading candidate {candidate}: {e}")
                        parse_errors.append((candidate, str(e)))

            if not loaded:
                # If we previously placed a placeholder in cache, remove it to avoid stale entries
                try:
                    invalidate = None
                    try:
                        # import locally to avoid top-level import cycles
                        from .module_cache import invalidate_module
                        invalidate = invalidate_module
                    except Exception:
                        invalidate = None
                    if invalidate:
                        invalidate(normalized_path)
                except Exception:
                    pass

                debug_log("  UseStatement failed to load candidates", parse_errors)
                return EvaluationError(f"Module not found or failed to load: {file_path}")

            # Set alias or import exported names
            alias = getattr(node, 'alias', None)

            # Debug: show exports discovered in module_env
            try:
                exports_debug = module_env.get_exports() if hasattr(module_env, 'get_exports') else {}
                print(f"[MOD-DEBUG] module_env exports for {file_path}: {exports_debug}")
            except Exception as e:
                print(f"[MOD-DEBUG] error reading exports: {e}")
            if alias:
                debug_log("  Setting module alias", alias)
                env.set(alias, module_env)
            else:
                try:
                    exports = module_env.get_exports()
                    # Get importer file path (if available from stack trace or environment context)
                    importer_file = env.get("__file__") if hasattr(env, 'get') else None
                    if importer_file and hasattr(importer_file, 'value'):
                        importer_file = importer_file.value

                    for name, value in exports.items():
                        debug_log("  Importing export", name)
                        # Check permission if importer_file is available
                        if importer_file:
                            perm_check = check_import_permission(value, importer_file, env)
                            if is_error(perm_check): # Use is_error helper
                                debug_log("  Permission denied for export", name)
                                return perm_check
                        env.set(name, value)
                except Exception:
                    # If module has no exports, make its env available as a module object
                    module_name = os.path.basename(file_path)
                    debug_log("  Setting module object", module_name)
                    env.set(module_name, module_env)

            return NULL

        # FROM statement: import specific names from a module
        elif node_type == FromStatement:
            debug_log("  FromStatement node", node.file_path)
            # Reuse the UseStatement logic to obtain module env
            use_node = UseStatement(node.file_path)
            res = eval_node(use_node, env, stack_trace)
            if is_error(res): # Use is_error helper
                return res

            # module should now be available in env (either alias or exports)
            try:
                module_env = env.get(os.path.basename(node.file_path))
                if not module_env or not hasattr(module_env, 'get_exports'):
                    # fallback: look up by normalized path
                    module_env = None
            except Exception:
                module_env = None

            # Import requested names
            importer_file = env.get("__file__") if hasattr(env, 'get') else None
            if importer_file and hasattr(importer_file, 'value'):
                importer_file = importer_file.value

            for name_pair in node.imports:
                src_name = name_pair[0].value if hasattr(name_pair[0], 'value') else str(name_pair[0])
                as_name = name_pair[1].value if len(name_pair) > 1 and name_pair[1] is not None else src_name
                try:
                    value = module_env.get_exports().get(src_name)
                    if value is None:
                        return EvaluationError(f"From import: '{src_name}' not found in module")
                    # Check permission if importer_file is available
                    if importer_file:
                        perm_check = check_import_permission(value, importer_file, env)
                        if is_error(perm_check): # Use is_error helper
                            debug_log("  Permission denied for from-import", src_name)
                            return perm_check
                    env.set(as_name, value)
                except Exception:
                    return EvaluationError(f"From import failed for '{src_name}'")

            return NULL

        elif node_type == ComponentStatement:
            debug_log("  ComponentStatement node", node.name.value)
            # Evaluate properties (map or block)
            props = None
            if hasattr(node, 'properties') and node.properties is not None:
                props_val = eval_node(node.properties, env, stack_trace)
                if is_error(props_val): # Use is_error helper
                    return props_val
                props = _to_python(props_val) if isinstance(props_val, (Map, List, String)) else None
            # Register via runtime builtin if available
            if 'define_component' in builtins:
                return builtins['define_component'].fn(String(node.name.value), Map(props) if isinstance(props, dict) else NULL)
            env.set(node.name.value, String(f"<component {node.name.value}>") )
            return NULL

        elif node_type == ThemeStatement:
            debug_log("  ThemeStatement node", node.name.value)
            props_val = eval_node(node.properties, env, stack_trace) if hasattr(node, 'properties') else NULL
            if is_error(props_val): # Use is_error helper
                return props_val
            # Set theme locally
            env.set(node.name.value, props_val)
            return NULL

        elif node_type == DebugStatement:
            debug_log("  DebugStatement node")
            val = eval_node(node.value, env, stack_trace)
            if is_error(val): # Use is_error helper
                return val
            Debug.log(String(str(val)))
            return NULL

        elif node_type == ExternalDeclaration:
            debug_log("  ExternalDeclaration node", node.name.value)
            # Register a placeholder builtin that raises when called until linked
            def _external_placeholder(*a):
                return EvaluationError(f"External function '{node.name.value}' not linked")
            env.set(node.name.value, Builtin(_external_placeholder, node.name.value))
            return NULL

        elif node_type == ExactlyStatement:
            debug_log("  ExactlyStatement node")
            return eval_node(node.body, env, stack_trace)

        # NEW: EntityStatement - Call the helper for entity definition
        elif node_type == EntityStatement:
            debug_log("  EntityStatement node", node.name.value)
            return eval_entity_statement(node, env)

        # NEW: SealStatement - Call the helper for sealing
        elif node_type == SealStatement:
            debug_log("  SealStatement node", node.target)
            return eval_seal_statement(node, env, stack_trace)

        # Expressions
        elif node_type == IntegerLiteral:
            debug_log("  IntegerLiteral node", node.value)
            return Integer(node.value)

        elif node_type == zexus_ast.FloatLiteral or node_type.__name__ == 'FloatLiteral':
            # FloatLiteral support
            try:
                val = getattr(node, 'value', None)
                return Float(val)
            except Exception:
                return EvaluationError(f"Invalid float literal: {getattr(node, 'value', None)}")

        elif node_type == StringLiteral:
            debug_log("  StringLiteral node", node.value)
            return String(node.value)

        elif node_type == ListLiteral:
            debug_log("  ListLiteral node", f"{len(node.elements)} elements")
            elements = eval_expressions(node.elements, env)
            # FIXED: use is_error helper
            if is_error(elements):
                return elements
            return List(elements)

        elif node_type == MapLiteral:
            debug_log("  MapLiteral node", f"{len(node.pairs)} pairs")
            pairs = {}
            for key_expr, value_expr in node.pairs:
                key = eval_node(key_expr, env, stack_trace)
                # FIXED: use is_error helper
                if is_error(key):
                    return key
                value = eval_node(value_expr, env, stack_trace)
                if is_error(value):
                    return value
                key_str = key.inspect()
                pairs[key_str] = value
            return Map(pairs)

        elif node_type == Identifier:
            debug_log("  Identifier node", node.value)
            return eval_identifier(node, env)

        elif node_type == ActionLiteral:
            debug_log("  ActionLiteral node")
            return Action(node.parameters, node.body, env)

        # FIXED: CallExpression - Properly handle builtin function calls
        elif node_type == CallExpression:
            debug_log("🚀 CallExpression node", f"Calling {node.function}")
            function = eval_node(node.function, env, stack_trace)
            if is_error(function): # Use is_error helper
                debug_log("  Function evaluation error", function)
                return function

            args = eval_expressions(node.arguments, env)
            # FIXED: detect error results using is_error() BEFORE attempting to len()/unpack
            if is_error(args):
                debug_log("  Arguments evaluation error", args)
                return args

            arg_count = len(args) if isinstance(args, (list, tuple)) else "unknown"
            debug_log("  Arguments evaluated", f"{args} (count: {arg_count})")

            # CRITICAL FIX: Ensure builtin functions are called properly
            debug_log("  Calling apply_function", f"function: {function}, args: {args}")
            result = apply_function(function, args)
            debug_log("  CallExpression result", result)
            return result

        elif node_type == PrefixExpression:
    # Use is_error helper to check `right`
            debug_log("  PrefixExpression node", f"{node.operator} {node.right}")
            right = eval_node(node.right, env, stack_trace)
            if is_error(right):
                return right
            return eval_prefix_expression(node.operator, right)

        elif node_type == InfixExpression:
            debug_log("  InfixExpression node", f"{node.left} {node.operator} {node.right}")
            left = eval_node(node.left, env, stack_trace)
            if is_error(left): # Use is_error helper
                return left
            right = eval_node(node.right, env, stack_trace)
            if is_error(right): # Use is_error helper
                return right
            return eval_infix_expression(node.operator, left, right)

        elif node_type == IfExpression:
            debug_log("  IfExpression node")
            return eval_if_expression(node, env)

        debug_log("  Unknown node type", node_type)
        return EvaluationError(f"Unknown node type: {node_type}", stack_trace=stack_trace)

    except Exception as e:
        # Enhanced error with stack trace
        error_msg = f"Internal error: {str(e)}"
        debug_log("  Exception in eval_node", error_msg)
        return EvaluationError(error_msg, stack_trace=stack_trace[-5:])  # Last 5 frames


# =====================================================
# NEW STATEMENT HANDLERS - ENTITY, VERIFY, CONTRACT, PROTECT, SEAL
# =====================================================

def eval_entity_statement(node, env):
    """Evaluate entity statement - create entity definition"""
    from .object import EntityDefinition # Ensure EntityDefinition is imported from object.py

    properties = {}
    for prop in node.properties:
        prop_name = prop.name.value if hasattr(prop.name, 'value') else str(prop.name)
        prop_type = prop.type.value if hasattr(prop.type, 'value') else str(prop.type)
        default_value = eval_node(prop.default_value, env) if hasattr(prop, 'default_value') and prop.default_value else NULL
        if is_error(default_value):
            return default_value
        properties[prop_name] = {
            "type": prop_type,
            "default_value": default_value # Store Zexus object for defaults
        }

    entity_def = EntityDefinition(node.name.value, properties)
    env.set(node.name.value, entity_def)
    return NULL


def eval_verify_statement(node, env, stack_trace=None):
    """Evaluate verify statement - register verification checks"""
    from .security import VerifyWrapper, VerificationCheck, get_security_context

    # Evaluate target function
    target_value = eval_node(node.target, env, stack_trace)
    if is_error(target_value): # Use is_error helper
        return target_value

    # Evaluate conditions
    checks = []
    for condition_node in node.conditions:
        condition_value = eval_node(condition_node, env, stack_trace)
        if is_error(condition_value): # Use is_error helper
            return condition_value
        # Assuming condition_value is a Zexus object that can be evaluated to truthy/falsy
        # Or if it's an Action/Lambda, it should be wrapped.
        if callable(condition_value) or isinstance(condition_value, Action):
            checks.append(VerificationCheck(str(condition_node), lambda ctx: condition_value))
        else:
             # For simpler truthy/falsy conditions directly from eval_node
            checks.append(VerificationCheck(str(condition_node), lambda ctx, cond=condition_value: cond))


    # Wrap function with verification
    wrapped = VerifyWrapper(target_value, checks, node.error_handler)

    # Register in security context
    ctx = get_security_context()
    ctx.register_verify_check(str(node.target), wrapped) # Assuming target has a string representation for key

    return wrapped


def eval_contract_statement(node, env, stack_trace=None):
    """Evaluate contract statement - create smart contract"""
    from .security import SmartContract

    storage_vars = {}
    for storage_var_node in node.storage_vars:
        storage_var_name = storage_var_node.name.value
        # Initial value might be present
        initial_value = eval_node(storage_var_node.initial_value, env) if hasattr(storage_var_node, 'initial_value') and storage_var_node.initial_value else NULL
        if is_error(initial_value):
            return initial_value
        storage_vars[storage_var_name] = initial_value

    actions = {}
    for action_node in node.actions:
        # Assuming action_node is an ActionStatement or ActionLiteral
        action_obj = eval_node(action_node, env, stack_trace) # Evaluate action literal/statement
        if is_error(action_obj):
            return action_obj
        actions[action_node.name.value] = action_obj

    contract = SmartContract(node.name.value, storage_vars, actions)
    contract.deploy() # This should probably be a method call from Zexus or an explicit statement

    env.set(node.name.value, contract)
    return NULL


def eval_protect_statement(node, env, stack_trace=None):
    """Evaluate protect statement - register protection rules"""
    from .security import ProtectionPolicy, get_security_context

    # Evaluate target
    target_value = eval_node(node.target, env, stack_trace)
    if is_error(target_value): # Use is_error helper
        return target_value

    # Evaluate rules
    rules_value = eval_node(node.rules, env, stack_trace)
    if is_error(rules_value): # Use is_error helper
        return rules_value

    # Convert rules to dict
    rules_dict = {}
    if isinstance(rules_value, Map):
        for key, value in rules_value.pairs.items():
            key_str = key.value if isinstance(key, String) else str(key)
            rules_dict[key_str] = value

    # Create and register protection policy
    policy = ProtectionPolicy(str(node.target), rules_dict, node.enforcement_level) # Assuming target has a string representation
    ctx = get_security_context()
    ctx.register_protection(str(node.target), policy)

    return policy


def eval_middleware_statement(node, env):
    """Evaluate middleware statement - register middleware handler"""
    from .security import Middleware, get_security_context

    # Evaluate handler
    handler = eval_node(node.handler, env)
    if is_error(handler): # Use is_error helper
        return handler

    # Create middleware
    middleware = Middleware(node.name.value, handler)

    # Register in security context
    ctx = get_security_context()
    ctx.middlewares[node.name.value] = middleware

    return NULL


def eval_auth_statement(node, env):
    """Evaluate auth statement - set authentication configuration"""
    from .security import AuthConfig, get_security_context

    # Evaluate config
    config_value = eval_node(node.config, env)
    if is_error(config_value): # Use is_error helper
        return config_value

    # Convert config to dict
    config_dict = {}
    if isinstance(config_value, Map):
        for key, value in config_value.pairs.items():
            key_str = key.value if isinstance(key, String) else str(key)
            config_dict[key_str] = value

    # Create auth config
    auth_config = AuthConfig(config_dict)

    # Register in security context
    ctx = get_security_context()
    ctx.auth_config = auth_config

    return NULL


def eval_throttle_statement(node, env):
    """Evaluate throttle statement - register rate limiting"""
    from .security import RateLimiter, get_security_context

    # Evaluate target and limits
    target_value = eval_node(node.target, env)
    if is_error(target_value): # Use is_error helper
        return target_value

    limits_value = eval_node(node.limits, env)
    if is_error(limits_value): # Use is_error helper
        return limits_value

    # Extract limits from map
    rpm = 100  # Default requests per minute
    burst = 10  # Default burst size
    per_user = FALSE # Default is BooleanObj(False)

    if isinstance(limits_value, Map):
        for key, value in limits_value.pairs.items():
            key_str = key.value if isinstance(key, String) else str(key)
            if key_str == "requests_per_minute" and isinstance(value, Integer):
                rpm = value.value
            elif key_str == "burst_size" and isinstance(value, Integer):
                burst = value.value
            elif key_str == "per_user" and isinstance(value, BooleanObj):
                per_user = value # Keep as BooleanObj for consistency
            elif key_str == "per_user" and isinstance(value, Boolean): # If AST Boolean, convert to Zexus BooleanObj
                per_user = TRUE if value.value else FALSE

    # Create rate limiter
    limiter = RateLimiter(rpm, burst, per_user.value) # Pass Python bool to RateLimiter

    # Register in security context
    ctx = get_security_context()
    ctx.rate_limiters = getattr(ctx, 'rate_limiters', {})
    ctx.rate_limiters[str(node.target)] = limiter

    return NULL


def eval_cache_statement(node, env):
    """Evaluate cache statement - register caching policy"""
    from .security import CachePolicy, get_security_context

    # Evaluate target and policy
    target_value = eval_node(node.target, env)
    if is_error(target_value): # Use is_error helper
        return target_value

    policy_value = eval_node(node.policy, env)
    if is_error(policy_value): # Use is_error helper
        return policy_value

    # Extract policy settings from map
    ttl = 3600  # Default 1 hour
    invalidate_on = []

    if isinstance(policy_value, Map):
        for key, value in policy_value.pairs.items():
            key_str = key.value if isinstance(key, String) else str(key)
            if key_str == "ttl" and isinstance(value, Integer):
                ttl = value.value
            elif key_str == "invalidate_on" and isinstance(value, List):
                invalidate_on = [item.value if hasattr(item, 'value') else str(item) for item in value.elements]

    # Create cache policy
    cache_policy = CachePolicy(ttl=ttl, invalidate_on=invalidate_on)

    # Register in security context
    ctx = get_security_context()
    ctx.cache_policies = getattr(ctx, 'cache_policies', {})
    ctx.cache_policies[str(node.target)] = cache_policy

    return NULL


def eval_seal_statement(node, env, stack_trace=None):
    """Evaluate seal statement - mark a variable or property as sealed (immutable)

    Supports sealing identifiers (`seal myVar`) and property access (`seal myMap.key`).
    """
    from .security import SealedObject

    target_node = node.target
    if target_node is None:
        return EvaluationError("seal: missing target to seal")

    # Case 1: Sealing an Identifier (e.g., `seal x`)
    if isinstance(target_node, Identifier):
        name = target_node.value
        current_value = env.get(name)
        if current_value is None:
            return EvaluationError(f"seal: identifier '{name}' not found")
        sealed_object = SealedObject(current_value)
        env.set(name, sealed_object)
        debug_log("  Sealed identifier", name)
        return sealed_object

    # Case 2: Sealing a PropertyAccessExpression (e.g., `seal obj.prop` or `seal map[key]`)
    # Note: For `map[key]`, the parser usually creates a `PropertyAccessExpression`
    # or `IndexExpression` depending on language design. Assuming PropertyAccessExpression for now.
    elif isinstance(target_node, PropertyAccessExpression):
        # Evaluate the object part (e.g., `obj` in `obj.prop`)
        obj = eval_node(target_node.object, env, stack_trace)
        if is_error(obj):
            return obj

        # Determine the property name/key
        # Assuming node.property is an Identifier for 'obj.prop' like access
        # If it could be an arbitrary expression (like `map[expression]`),
        # node.property would need to be evaluated further.
        prop_key_node = target_node.property
        if isinstance(prop_key_node, Identifier):
            prop_key = prop_key_node.value
        else:
            # Fallback for dynamic keys like map[expr] if PropertyAccessExpression supports it
            prop_key_evaluated = eval_node(prop_key_node, env, stack_trace)
            if is_error(prop_key_evaluated):
                return prop_key_evaluated
            prop_key = prop_key_evaluated.inspect() # Use inspect to get a string key for maps

        try:
            # Handle Map objects: seal a specific key's value
            if isinstance(obj, Map):
                if prop_key not in obj.pairs:
                    return EvaluationError(f"seal: map key '{prop_key}' not found on map")
                # Create a NEW Map with the sealed value, or modify in place if Map allows it.
                # If Map is mutable, just update:
                obj.pairs[prop_key] = SealedObject(obj.pairs[prop_key])
                debug_log(f"  Sealed map key '{prop_key}' for map {obj.inspect()}", "")
                return obj.pairs[prop_key]
            # Handle EntityInstance-like objects with get/set methods
            elif hasattr(obj, 'get') and callable(obj.get) and \
                 hasattr(obj, 'set') and callable(obj.set):
                current_prop_value = obj.get(prop_key)
                if current_prop_value is None:
                    return EvaluationError(f"seal: property '{prop_key}' not found on object {obj.type()}")
                sealed_prop_value = SealedObject(current_prop_value)
                obj.set(prop_key, sealed_prop_value) # This set should trigger immutability check in EntityInstance.set
                debug_log(f"  Sealed property '{prop_key}' on object {obj.inspect()}", "")
                return sealed_prop_value
            else:
                return EvaluationError(f"seal: cannot seal property '{prop_key}' on object of type {obj.type()}")
        except Exception as e:
            return EvaluationError(f"seal error on property '{prop_key}': {str(e)}")

    return EvaluationError("seal: unsupported target type for sealing. Expected Identifier or PropertyAccessExpression.")


# Production evaluation with enhanced debugging
def evaluate(program, env, debug_mode=False):
    """Production evaluation with enhanced error handling and debugging"""
    if debug_mode:
        env.enable_debug()
        print("🔧 Debug mode enabled")

    result = eval_node(program, env)
    # Resolve awaitables at the top level when possible
    result = _resolve_awaitable(result)

    if debug_mode:
        env.disable_debug()

    # When debug mode is off, print a concise 5-line summary only
    if not debug_mode:
        try:
            print(f"Summary: statements parsed={EVAL_SUMMARY.get('parsed_statements',0)}")
            print(f"Summary: statements evaluated={EVAL_SUMMARY.get('evaluated_statements',0)}")
            print(f"Summary: errors={EVAL_SUMMARY.get('errors',0)}")
            print(f"Summary: async_tasks_run={EVAL_SUMMARY.get('async_tasks_run',0)}")
            print(f"Summary: max_statements_in_block={EVAL_SUMMARY.get('max_statements_in_block',0)}")
        except Exception:
            # If summary printing fails, ignore and continue
            pass

    if is_error(result): # Use is_error helper
        return str(result)

    return result