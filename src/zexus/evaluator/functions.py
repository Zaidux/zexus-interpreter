# src/zexus/evaluator/functions.py
import sys
import os

from .. import zexus_ast
from ..zexus_ast import CallExpression, MethodCallExpression
from ..object import (
    Environment, Integer, Float, String, List, Map, Boolean as BooleanObj,
    Null, Builtin, Action, LambdaFunction, ReturnValue, DateTime, Math, File, Debug,
    EvaluationError
)
from .utils import is_error, debug_log, NULL, TRUE, FALSE, _resolve_awaitable, _zexus_to_python, _python_to_zexus, _to_str

# Try to import backend, handle failure gracefully (as per your original code)
try:
    from renderer import backend as _BACKEND
    _BACKEND_AVAILABLE = True
except Exception:
    _BACKEND_AVAILABLE = False
    _BACKEND = None

class FunctionEvaluatorMixin:
    """Handles function application, method calls, and defines all builtins."""
    
    def __init__(self):
        # Initialize registries
        self.builtins = {}
        
        # Renderer Registry (moved from global scope to instance scope)
        self.render_registry = {
            'screens': {},
            'components': {},
            'themes': {},
            'canvases': {},
            'current_theme': None
        }
        
        # Register all functions
        self._register_core_builtins()
        self._register_renderer_builtins()
    
    def eval_call_expression(self, node, env, stack_trace):
        debug_log("🚀 CallExpression node", f"Calling {node.function}")
        
        fn = self.eval_node(node.function, env, stack_trace)
        if is_error(fn): 
            return fn
        
        args = self.eval_expressions(node.arguments, env)
        if is_error(args): 
            return args
        
        arg_count = len(args) if isinstance(args, (list, tuple)) else "unknown"
        debug_log("  Arguments evaluated", f"{args} (count: {arg_count})")
        
        # Contract instantiation check
        from ..security import SmartContract
        if isinstance(fn, SmartContract):
            return fn.instantiate(args)
        
        return self.apply_function(fn, args, env)
    
    def apply_function(self, fn, args, env=None):
        debug_log("apply_function", f"Calling {fn}")
        
        # Phase 2 & 3: Trigger plugin hooks and check capabilities
        if hasattr(self, 'integration_context'):
            if isinstance(fn, (Action, LambdaFunction)):
                func_name = fn.name if hasattr(fn, 'name') else str(fn)
                # Trigger before-call hook
                self.integration_context.plugins.before_action_call(func_name, {})
                
                # Check required capabilities
                try:
                    self.integration_context.capabilities.require_capability("core.language")
                except PermissionError:
                    return EvaluationError(f"Permission denied: insufficient capabilities for {func_name}")
        
        if isinstance(fn, (Action, LambdaFunction)):
            debug_log("  Calling user-defined function")
            
            # Check if this is an async action
            is_async = getattr(fn, 'is_async', False)
            
            if is_async:
                # Create a coroutine that lazily executes the async action
                from ..object import Coroutine
                
                def async_generator():
                    """Generator that executes the async action body"""
                    new_env = Environment(outer=fn.env)
                    
                    # Bind parameters
                    for i, param in enumerate(fn.parameters):
                        if i < len(args):
                            param_name = param.value if hasattr(param, 'value') else str(param)
                            new_env.set(param_name, args[i])
                    
                    # Yield control first (makes it a true generator)
                    yield None
                    
                    try:
                        # Evaluate the function body
                        res = self.eval_node(fn.body, new_env)
                        
                        # Unwrap ReturnValue if needed
                        if isinstance(res, ReturnValue):
                            result = res.value
                        else:
                            result = res
                        
                        # Execute deferred cleanup
                        if hasattr(self, '_execute_deferred_cleanup'):
                            self._execute_deferred_cleanup(new_env, [])
                        
                        # Return the result (will be caught by StopIteration)
                        return result
                    except Exception as e:
                        # Re-raise exception to be caught by coroutine
                        raise e
                
                # Create and return coroutine
                gen = async_generator()
                coroutine = Coroutine(gen, fn)
                return coroutine
            
            # Synchronous function execution
            new_env = Environment(outer=fn.env)
            
            param_names = []
            for i, param in enumerate(fn.parameters):
                if i < len(args):
                    # Handle both Identifier objects and strings
                    param_name = param.value if hasattr(param, 'value') else str(param)
                    param_names.append(param_name)
                    new_env.set(param_name, args[i])
                    # Lightweight debug: show what is being bound
                    try:
                        debug_log("    Set parameter", f"{param_name} = {type(args[i]).__name__}")
                    except Exception:
                        pass

            try:
                if param_names:
                    debug_log("  Function parameters bound", f"{param_names}")
            except Exception:
                pass
            
            try:
                res = self.eval_node(fn.body, new_env)
                res = _resolve_awaitable(res)
                
                # Unwrap ReturnValue if needed
                if isinstance(res, ReturnValue):
                    result = res.value
                else:
                    result = res
                
                return result
            finally:
                # CRITICAL: Execute deferred cleanup when function exits
                # This happens in finally block to ensure cleanup runs even on errors
                if hasattr(self, '_execute_deferred_cleanup'):
                    self._execute_deferred_cleanup(new_env, [])
                
                # Phase 2: Trigger after-call hook
                if hasattr(self, 'integration_context'):
                    func_name = fn.name if hasattr(fn, 'name') else str(fn)
                    self.integration_context.plugins.after_action_call(func_name, result)
        
        elif isinstance(fn, Builtin):
            debug_log("  Calling builtin function", f"{fn.name}")
            # Sandbox enforcement: if current env is sandboxed, consult policy
            try:
                in_sandbox = False
                policy_name = None
                if env is not None:
                    try:
                        in_sandbox = bool(env.get('__in_sandbox__'))
                        policy_name = env.get('__sandbox_policy__')
                    except Exception:
                        in_sandbox = False

                if in_sandbox:
                    from ..security import get_security_context
                    ctx = get_security_context()
                    policy = ctx.get_sandbox_policy(policy_name or 'default')
                    allowed = None if policy is None else policy.get('allowed_builtins')
                    # If allowed set exists and builtin not in it -> block
                    if allowed is not None and fn.name not in allowed:
                        return EvaluationError(f"Builtin '{fn.name}' not allowed inside sandbox policy '{policy_name or 'default'}'")
            except Exception:
                # If enforcement fails unexpectedly, proceed to call but log nothing
                pass

            try:
                res = fn.fn(*args)
                return _resolve_awaitable(res)
            except Exception as e:
                return EvaluationError(f"Builtin error: {str(e)}")
        
        return EvaluationError(f"Not a function: {fn}")
    
    def eval_method_call_expression(self, node, env, stack_trace):
        debug_log("  MethodCallExpression node", f"{node.object}.{node.method}")
        
        obj = self.eval_node(node.object, env, stack_trace)
        if is_error(obj): 
            return obj
        
        method_name = node.method.value
        
        # === List Methods ===
        if isinstance(obj, List):
            # For map/filter/reduce, we need to evaluate arguments first
            if method_name in ["map", "filter", "reduce"]:
                args = self.eval_expressions(node.arguments, env)
                if is_error(args): 
                    return args
                
                if method_name == "reduce":
                    if len(args) < 1: 
                        return EvaluationError("reduce() requires at least a lambda function")
                    lambda_fn = args[0]
                    initial = args[1] if len(args) > 1 else None
                    return self._array_reduce(obj, lambda_fn, initial)
                
                elif method_name == "map":
                    if len(args) != 1: 
                        return EvaluationError("map() requires exactly one lambda function")
                    return self._array_map(obj, args[0])
                
                elif method_name == "filter":
                    if len(args) != 1: 
                        return EvaluationError("filter() requires exactly one lambda function")
                    return self._array_filter(obj, args[0])
            
            # Other list methods
            args = self.eval_expressions(node.arguments, env)
            if is_error(args): 
                return args
            
            if method_name == "push":
                obj.elements.append(args[0])
                return obj
            elif method_name == "count":
                return Integer(len(obj.elements))
            elif method_name == "contains":
                target = args[0]
                found = any(elem.value == target.value for elem in obj.elements 
                          if hasattr(elem, 'value') and hasattr(target, 'value'))
                return TRUE if found else FALSE
        
        # === Coroutine Methods ===
        from ..object import Coroutine
        if isinstance(obj, Coroutine):
            if method_name == "inspect":
                # Return string representation of coroutine state
                return String(obj.inspect())
        
        # === Map Methods ===
        if isinstance(obj, Map):
            args = self.eval_expressions(node.arguments, env)
            if is_error(args): 
                return args
            
            if method_name == "has":
                key = args[0].value if hasattr(args[0], 'value') else str(args[0])
                return TRUE if key in obj.pairs else FALSE
            elif method_name == "get":
                key = args[0].value if hasattr(args[0], 'value') else str(args[0])
                default = args[1] if len(args) > 1 else NULL
                return obj.pairs.get(key, default)
        
        # === Module Methods ===
        from ..complexity_system import Module, Package
        if isinstance(obj, Module):
            debug_log("  MethodCallExpression", f"Calling method '{method_name}' on module '{obj.name}'")
            debug_log("  MethodCallExpression", f"Module members: {list(obj.members.keys())}")
            
            # For module methods, get the member and call it if it's a function
            member_value = obj.get(method_name)
            if member_value is None:
                return EvaluationError(f"Method '{method_name}' not found in module '{obj.name}'")
            
            debug_log("  MethodCallExpression", f"Found member value: {member_value}")
            
            # Evaluate arguments
            args = self.eval_expressions(node.arguments, env)
            if is_error(args):
                return args
            
            # Call the function/action using apply_function
            return self.apply_function(member_value, args, env)
        
        # === Package Methods ===
        if isinstance(obj, Package):
            debug_log("  MethodCallExpression", f"Calling method '{method_name}' on package '{obj.name}'")
            debug_log("  MethodCallExpression", f"Package modules: {list(obj.modules.keys())}")
            
            # For package methods, get the module/function and call it
            member_value = obj.get(method_name)
            if member_value is None:
                return EvaluationError(f"Method '{method_name}' not found in package '{obj.name}'")
            
            debug_log("  MethodCallExpression", f"Found member value: {member_value}")
            
            # Evaluate arguments
            args = self.eval_expressions(node.arguments, env)
            if is_error(args):
                return args
            
            # Call the function/action using apply_function
            return self.apply_function(member_value, args, env)
        
        # === Contract Instance Methods ===
        if hasattr(obj, 'call_method'):
            args = self.eval_expressions(node.arguments, env)
            if is_error(args): 
                return args
            return obj.call_method(method_name, args)
        
        # === Embedded Code Methods ===
        from ..object import EmbeddedCode
        if isinstance(obj, EmbeddedCode):
            print(f"[EMBED] Executing {obj.language}.{method_name}")
            return Integer(42)  # Placeholder
        
        # === Environment (Module) Methods ===
        # Support for module.function() syntax (e.g., crypto.keccak256())
        from ..object import Environment
        if isinstance(obj, Environment):
            # Look up the method in the environment's store
            method_value = obj.get(method_name)
            if method_value is None or method_value == NULL:
                return EvaluationError(f"Module has no method '{method_name}'")
            
            # Evaluate arguments
            args = self.eval_expressions(node.arguments, env)
            if is_error(args):
                return args
            
            # Call the function using apply_function
            return self.apply_function(method_value, args, env)
        
        obj_type = obj.type() if hasattr(obj, 'type') and callable(obj.type) else type(obj).__name__
        return EvaluationError(f"Method '{method_name}' not supported for {obj_type}")
    
    # --- Array Helpers (Internal) ---
    
    def _array_reduce(self, array_obj, lambda_fn, initial_value=None):
        if not isinstance(array_obj, List): 
            return EvaluationError("reduce() called on non-array")
        if not isinstance(lambda_fn, (LambdaFunction, Action)): 
            return EvaluationError("reduce() requires lambda")
        
        accumulator = initial_value if initial_value is not None else (
            array_obj.elements[0] if array_obj.elements else NULL
        )
        start_index = 0 if initial_value is not None else 1
        
        for i in range(start_index, len(array_obj.elements)):
            element = array_obj.elements[i]
            result = self.apply_function(lambda_fn, [accumulator, element])
            if is_error(result): 
                return result
            accumulator = result
        
        return accumulator
    
    def _array_map(self, array_obj, lambda_fn):
        if not isinstance(array_obj, List): 
            return EvaluationError("map() called on non-array")
        if not isinstance(lambda_fn, (LambdaFunction, Action)): 
            return EvaluationError("map() requires lambda")
        
        mapped = []
        for element in array_obj.elements:
            result = self.apply_function(lambda_fn, [element])
            if is_error(result): 
                return result
            mapped.append(result)
        
        return List(mapped)
    
    def _array_filter(self, array_obj, lambda_fn):
        if not isinstance(array_obj, List): 
            return EvaluationError("filter() called on non-array")
        if not isinstance(lambda_fn, (LambdaFunction, Action)): 
            return EvaluationError("filter() requires lambda")
        
        filtered = []
        for element in array_obj.elements:
            result = self.apply_function(lambda_fn, [element])
            if is_error(result): 
                return result
            
            # Use is_truthy from utils
            from .utils import is_truthy
            if is_truthy(result):
                filtered.append(element)
        
        return List(filtered)
    
    # --- BUILTIN IMPLEMENTATIONS ---
    
    def _register_core_builtins(self):
        # Date & Time
        def _now(*a): 
            return DateTime.now()
        
        def _timestamp(*a):
            if len(a) == 0: 
                return DateTime.now().to_timestamp()
            if len(a) == 1 and isinstance(a[0], DateTime): 
                return a[0].to_timestamp()
            return EvaluationError("timestamp() takes 0 or 1 DateTime")
        
        # Math
        def _random(*a):
            if len(a) == 0: 
                return Math.random_int(0, 100)
            if len(a) == 1 and isinstance(a[0], Integer): 
                return Math.random_int(0, a[0].value)
            if len(a) == 2 and all(isinstance(x, Integer) for x in a): 
                return Math.random_int(a[0].value, a[1].value)
            return EvaluationError("random() takes 0, 1, or 2 integer arguments")
        
        def _to_hex(*a): 
            if len(a) != 1: 
                return EvaluationError("to_hex() takes exactly 1 argument")
            return Math.to_hex_string(a[0])
        
        def _from_hex(*a): 
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("from_hex() takes exactly 1 string argument")
            return Math.hex_to_int(a[0])
        
        def _sqrt(*a): 
            if len(a) != 1: 
                return EvaluationError("sqrt() takes exactly 1 argument")
            return Math.sqrt(a[0])
        
        # File I/O
        def _read_text(*a): 
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("file_read_text() takes exactly 1 string argument")
            return File.read_text(a[0])
        
        def _write_text(*a): 
            if len(a) != 2 or not all(isinstance(x, String) for x in a): 
                return EvaluationError("file_write_text() takes exactly 2 string arguments")
            return File.write_text(a[0], a[1])
        
        def _exists(*a): 
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("file_exists() takes exactly 1 string argument")
            return File.exists(a[0])
        
        def _read_json(*a): 
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("file_read_json() takes exactly 1 string argument")
            return File.read_json(a[0])
        
        def _write_json(*a):
            if len(a) != 2 or not isinstance(a[0], String): 
                return EvaluationError("file_write_json() takes path string and data")
            return File.write_json(a[0], a[1])
        
        def _append(*a): 
            if len(a) != 2 or not all(isinstance(x, String) for x in a): 
                return EvaluationError("file_append() takes exactly 2 string arguments")
            return File.append_text(a[0], a[1])
        
        def _list_dir(*a): 
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("file_list_dir() takes exactly 1 string argument")
            return File.list_directory(a[0])
        
        # Debug
        def _debug(*a):
            """Simple debug function that works like print"""
            if len(a) == 0:
                return EvaluationError("debug() requires at least 1 argument")
            msg = a[0]
            # Convert to string representation
            if isinstance(msg, String):
                output = msg.value
            elif isinstance(msg, (Integer, Float)):
                output = str(msg.value)
            elif isinstance(msg, BooleanObj):
                output = "true" if msg.value else "false"
            elif msg == NULL:
                output = "null"
            elif isinstance(msg, (List, Map)):
                output = msg.inspect()
            else:
                output = str(msg)
            # Output with DEBUG prefix
            print(f"[DEBUG] {output}", flush=True)
            return msg  # Return the original value for use in expressions
        
        def _debug_log(*a):
            if len(a) == 0: 
                return EvaluationError("debug_log() requires at least a message")
            msg = a[0]
            val = a[1] if len(a) > 1 else None
            return Debug.log(msg, val)
        
        def _debug_trace(*a): 
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("debug_trace() takes exactly 1 string argument")
            return Debug.trace(a[0])
        
        # String & Utility
        def _string(*a):
            if len(a) != 1: 
                return EvaluationError(f"string() takes 1 arg ({len(a)} given)")
            arg = a[0]
            if isinstance(arg, Integer) or isinstance(arg, Float): 
                return String(str(arg.value))
            if isinstance(arg, String): 
                return arg
            if isinstance(arg, BooleanObj): 
                return String("true" if arg.value else "false")
            if isinstance(arg, (List, Map)): 
                return String(arg.inspect())
            if arg == NULL: 
                return String("null")
            return String(str(arg))
        
        def _len(*a):
            if len(a) != 1: 
                return EvaluationError("len() takes 1 arg")
            arg = a[0]
            if isinstance(arg, String): 
                return Integer(len(arg.value))
            if isinstance(arg, List): 
                return Integer(len(arg.elements))
            return EvaluationError(f"len() not supported for {arg.type()}")
        
        # List Utils (Builtin versions of methods)
        def _first(*a): 
            if not isinstance(a[0], List): 
                return EvaluationError("first() expects a list")
            return a[0].elements[0] if a[0].elements else NULL
        
        def _rest(*a): 
            if not isinstance(a[0], List): 
                return EvaluationError("rest() expects a list")
            return List(a[0].elements[1:]) if len(a[0].elements) > 0 else List([])
        
        def _push(*a):
            if len(a) != 2 or not isinstance(a[0], List): 
                return EvaluationError("push(list, item)")
            return List(a[0].elements + [a[1]])
        
        def _reduce(*a):
            if len(a) < 2: 
                return EvaluationError("reduce(arr, fn, [init])")
            return self._array_reduce(a[0], a[1], a[2] if len(a) > 2 else None)
        
        def _map(*a):
            if len(a) != 2: 
                return EvaluationError("map(arr, fn)")
            return self._array_map(a[0], a[1])
        
        def _filter(*a):
            if len(a) != 2: 
                return EvaluationError("filter(arr, fn)")
            return self._array_filter(a[0], a[1])
        
        # File object creation (for RAII using statements)
        def _file(*a):
            if len(a) == 0 or len(a) > 2:
                return EvaluationError("file() takes 1 or 2 arguments: file(path) or file(path, mode)")
            if not isinstance(a[0], String):
                return EvaluationError("file() path must be a string")
            
            from ..object import File as FileObject
            path = a[0].value
            mode = a[1].value if len(a) > 1 and isinstance(a[1], String) else 'r'
            
            try:
                file_obj = FileObject(path, mode)
                file_obj.open()
                return file_obj
            except Exception as e:
                return EvaluationError(f"file() error: {str(e)}")
        
        # Register mappings
        self.builtins.update({
            "now": Builtin(_now, "now"),
            "timestamp": Builtin(_timestamp, "timestamp"),
            "random": Builtin(_random, "random"),
            "to_hex": Builtin(_to_hex, "to_hex"),
            "from_hex": Builtin(_from_hex, "from_hex"),
            "sqrt": Builtin(_sqrt, "sqrt"),
            "file": Builtin(_file, "file"),
            "file_read_text": Builtin(_read_text, "file_read_text"),
            "file_write_text": Builtin(_write_text, "file_write_text"),
            "file_exists": Builtin(_exists, "file_exists"),
            "file_read_json": Builtin(_read_json, "file_read_json"),
            "file_write_json": Builtin(_write_json, "file_write_json"),
            "file_append": Builtin(_append, "file_append"),
            "file_list_dir": Builtin(_list_dir, "file_list_dir"),
            "debug": Builtin(_debug, "debug"),
            "debug_log": Builtin(_debug_log, "debug_log"),
            "debug_trace": Builtin(_debug_trace, "debug_trace"),
            "string": Builtin(_string, "string"),
            "len": Builtin(_len, "len"),
            "first": Builtin(_first, "first"),
            "rest": Builtin(_rest, "rest"),
            "push": Builtin(_push, "push"),
            "reduce": Builtin(_reduce, "reduce"),
            "map": Builtin(_map, "map"),
            "filter": Builtin(_filter, "filter"),
        })
        
        # Register concurrency builtins
        self._register_concurrency_builtins()
        
        # Register blockchain builtins
        self._register_blockchain_builtins()
        
        # Register verification helper builtins
        self._register_verification_builtins()
    
    def _register_concurrency_builtins(self):
        """Register concurrency operations as builtin functions"""
        
        def _send(*a):
            """Send value to channel: send(channel, value)"""
            if len(a) != 2:
                return EvaluationError("send() requires 2 arguments: channel, value")
            
            channel = a[0]
            value = a[1]
            
            # Check if it's a valid channel object
            if not hasattr(channel, 'send'):
                return EvaluationError(f"send() first argument must be a channel, got {type(channel).__name__}")
            
            try:
                channel.send(value, timeout=5.0)
                return NULL  # send returns nothing on success
            except Exception as e:
                return EvaluationError(f"send() error: {str(e)}")
        
        def _receive(*a):
            """Receive value from channel: value = receive(channel)"""
            if len(a) != 1:
                return EvaluationError("receive() requires 1 argument: channel")
            
            channel = a[0]
            
            # Check if it's a valid channel object
            if not hasattr(channel, 'receive'):
                return EvaluationError(f"receive() first argument must be a channel, got {type(channel).__name__}")
            
            try:
                value = channel.receive(timeout=5.0)
                return value if value is not None else NULL
            except Exception as e:
                return EvaluationError(f"receive() error: {str(e)}")
        
        def _close_channel(*a):
            """Close a channel: close_channel(channel)"""
            if len(a) != 1:
                return EvaluationError("close_channel() requires 1 argument: channel")
            
            channel = a[0]
            
            if not hasattr(channel, 'close'):
                return EvaluationError(f"close_channel() argument must be a channel, got {type(channel).__name__}")
            
            try:
                channel.close()
                return NULL
            except Exception as e:
                return EvaluationError(f"close_channel() error: {str(e)}")
        
        # Register concurrency builtins
        self.builtins.update({
            "send": Builtin(_send, "send"),
            "receive": Builtin(_receive, "receive"),
            "close_channel": Builtin(_close_channel, "close_channel"),
        })
    
    def _register_blockchain_builtins(self):
        """Register blockchain cryptographic and utility functions"""
        from ..blockchain.crypto import CryptoPlugin
        from ..blockchain.transaction import get_current_tx, create_tx_context
        
        # hash(data, algorithm?)
        def _hash(*a):
            if len(a) < 1:
                return EvaluationError("hash() requires at least 1 argument: data, [algorithm]")
            
            data = a[0].value if hasattr(a[0], 'value') else str(a[0])
            algorithm = a[1].value if len(a) > 1 and hasattr(a[1], 'value') else 'SHA256'
            
            try:
                result = CryptoPlugin.hash_data(data, algorithm)
                return String(result)
            except Exception as e:
                return EvaluationError(f"Hash error: {str(e)}")
        
        # keccak256(data)
        def _keccak256(*a):
            if len(a) != 1:
                return EvaluationError("keccak256() expects 1 argument: data")
            
            data = a[0].value if hasattr(a[0], 'value') else str(a[0])
            
            try:
                result = CryptoPlugin.keccak256(data)
                return String(result)
            except Exception as e:
                return EvaluationError(f"Keccak256 error: {str(e)}")
        
        # signature(data, private_key, algorithm?)
        def _signature(*a):
            if len(a) < 2:
                return EvaluationError("signature() requires at least 2 arguments: data, private_key, [algorithm]")
            
            data = a[0].value if hasattr(a[0], 'value') else str(a[0])
            private_key = a[1].value if hasattr(a[1], 'value') else str(a[1])
            algorithm = a[2].value if len(a) > 2 and hasattr(a[2], 'value') else 'ECDSA'
            
            try:
                result = CryptoPlugin.sign_data(data, private_key, algorithm)
                return String(result)
            except Exception as e:
                return EvaluationError(f"Signature error: {str(e)}")
        
        # verify_sig(data, signature, public_key, algorithm?)
        def _verify_sig(*a):
            if len(a) < 3:
                return EvaluationError("verify_sig() requires at least 3 arguments: data, signature, public_key, [algorithm]")
            
            data = a[0].value if hasattr(a[0], 'value') else str(a[0])
            signature = a[1].value if hasattr(a[1], 'value') else str(a[1])
            public_key = a[2].value if hasattr(a[2], 'value') else str(a[2])
            algorithm = a[3].value if len(a) > 3 and hasattr(a[3], 'value') else 'ECDSA'
            
            try:
                result = CryptoPlugin.verify_signature(data, signature, public_key, algorithm)
                return TRUE if result else FALSE
            except Exception as e:
                return EvaluationError(f"Verification error: {str(e)}")
        
        # tx object - returns transaction context
        def _tx(*a):
            # Get or create TX context
            tx = get_current_tx()
            if tx is None:
                tx = create_tx_context(caller="system", gas_limit=1000000)
            
            # Return as Map object
            return Map({
                String("caller"): String(tx.caller),
                String("timestamp"): Integer(int(tx.timestamp)),
                String("block_hash"): String(tx.block_hash),
                String("gas_used"): Integer(tx.gas_used),
                String("gas_remaining"): Integer(tx.gas_remaining),
                String("gas_limit"): Integer(tx.gas_limit)
            })
        
        # gas object - returns gas tracking info
        def _gas(*a):
            # Get or create TX context
            tx = get_current_tx()
            if tx is None:
                tx = create_tx_context(caller="system", gas_limit=1000000)
            
            # Return as Map object
            return Map({
                String("used"): Integer(tx.gas_used),
                String("remaining"): Integer(tx.gas_remaining),
                String("limit"): Integer(tx.gas_limit)
            })
        
        self.builtins.update({
            "hash": Builtin(_hash, "hash"),
            "keccak256": Builtin(_keccak256, "keccak256"),
            "signature": Builtin(_signature, "signature"),
            "verify_sig": Builtin(_verify_sig, "verify_sig"),
            "tx": Builtin(_tx, "tx"),
            "gas": Builtin(_gas, "gas"),
        })
        
        # Register advanced feature builtins
        self._register_advanced_feature_builtins()
    
    def _register_advanced_feature_builtins(self):
        """Register builtins for persistence, policy, and dependency injection"""
        
        # === PERSISTENCE & MEMORY BUILTINS ===
        
        def _persistent_set(*a):
            """Set a persistent variable: persistent_set(name, value)"""
            if len(a) != 2:
                return EvaluationError("persistent_set() takes 2 arguments: name, value")
            if not isinstance(a[0], String):
                return EvaluationError("persistent_set() name must be a string")
            
            # Get current environment from evaluator context
            env = getattr(self, '_current_env', None)
            if env and hasattr(env, 'set_persistent'):
                name = a[0].value
                value = a[1]
                env.set_persistent(name, value)
                return String(f"Persistent variable '{name}' set")
            return EvaluationError("Persistence not enabled in this environment")
        
        def _persistent_get(*a):
            """Get a persistent variable: persistent_get(name, [default])"""
            if len(a) < 1 or len(a) > 2:
                return EvaluationError("persistent_get() takes 1 or 2 arguments: name, [default]")
            if not isinstance(a[0], String):
                return EvaluationError("persistent_get() name must be a string")
            
            env = getattr(self, '_current_env', None)
            if env and hasattr(env, 'get_persistent'):
                name = a[0].value
                default = a[1] if len(a) > 1 else NULL
                value = env.get_persistent(name, default)
                return value if value is not None else default
            return NULL
        
        def _persistent_delete(*a):
            """Delete a persistent variable: persistent_delete(name)"""
            if len(a) != 1:
                return EvaluationError("persistent_delete() takes 1 argument: name")
            if not isinstance(a[0], String):
                return EvaluationError("persistent_delete() name must be a string")
            
            env = getattr(self, '_current_env', None)
            if env and hasattr(env, 'delete_persistent'):
                name = a[0].value
                env.delete_persistent(name)
                return String(f"Persistent variable '{name}' deleted")
            return NULL
        
        def _memory_stats(*a):
            """Get memory tracking statistics: memory_stats()"""
            env = getattr(self, '_current_env', None)
            if env and hasattr(env, 'get_memory_stats'):
                stats = env.get_memory_stats()
                return Map({
                    String("tracked_objects"): Integer(stats.get("tracked_objects", 0)),
                    String("message"): String(stats.get("message", ""))
                })
            return Map({String("message"): String("Memory tracking not enabled")})
        
        # === POLICY & PROTECTION BUILTINS ===
        
        def _create_policy(*a):
            """Create a protection policy: create_policy(name, rules_map)"""
            if len(a) != 2:
                return EvaluationError("create_policy() takes 2 arguments: name, rules")
            if not isinstance(a[0], String):
                return EvaluationError("create_policy() name must be a string")
            if not isinstance(a[1], Map):
                return EvaluationError("create_policy() rules must be a Map")
            
            from ..policy_engine import get_policy_registry, PolicyBuilder, EnforcementLevel
            
            name = a[0].value
            rules = a[1].pairs
            
            builder = PolicyBuilder(name)
            builder.set_enforcement(EnforcementLevel.STRICT)
            
            # Parse rules from Map
            for key, value in rules.items():
                key_str = key.value if hasattr(key, 'value') else str(key)
                if key_str == "verify" and isinstance(value, List):
                    for cond in value.elements:
                        cond_str = cond.value if hasattr(cond, 'value') else str(cond)
                        builder.add_verify_rule(cond_str)
                elif key_str == "restrict" and isinstance(value, Map):
                    for field, constraints in value.pairs.items():
                        field_str = field.value if hasattr(field, 'value') else str(field)
                        constraint_list = []
                        if isinstance(constraints, List):
                            for c in constraints.elements:
                                constraint_list.append(c.value if hasattr(c, 'value') else str(c))
                        builder.add_restrict_rule(field_str, constraint_list)
            
            policy = builder.build()
            registry = get_policy_registry()
            registry.register(name, policy)
            
            return String(f"Policy '{name}' created and registered")
        
        def _check_policy(*a):
            """Check policy enforcement: check_policy(target, context_map)"""
            if len(a) != 2:
                return EvaluationError("check_policy() takes 2 arguments: target, context")
            if not isinstance(a[0], String):
                return EvaluationError("check_policy() target must be a string")
            if not isinstance(a[1], Map):
                return EvaluationError("check_policy() context must be a Map")
            
            from ..policy_engine import get_policy_registry
            
            target = a[0].value
            context = {}
            for k, v in a[1].pairs.items():
                key_str = k.value if hasattr(k, 'value') else str(k)
                val = v.value if hasattr(v, 'value') else v
                context[key_str] = val
            
            registry = get_policy_registry()
            policy = registry.get(target)
            
            if policy is None:
                return String(f"No policy found for '{target}'")
            
            result = policy.enforce(context)
            if result["success"]:
                return TRUE
            else:
                return String(f"Policy violation: {result['message']}")
        
        # === DEPENDENCY INJECTION BUILTINS ===
        
        def _register_dependency(*a):
            """Register a dependency: register_dependency(name, value, [module])"""
            if len(a) < 2 or len(a) > 3:
                return EvaluationError("register_dependency() takes 2 or 3 arguments: name, value, [module]")
            if not isinstance(a[0], String):
                return EvaluationError("register_dependency() name must be a string")
            
            from ..dependency_injection import get_di_registry
            
            name = a[0].value
            value = a[1]
            module = a[2].value if len(a) > 2 and isinstance(a[2], String) else "__main__"
            
            registry = get_di_registry()
            container = registry.get_container(module)
            if not container:
                # Create container if it doesn't exist
                registry.register_module(module)
                container = registry.get_container(module)
            # Declare and provide the dependency
            container.declare_dependency(name, "any", False)
            container.provide(name, value)
            
            return String(f"Dependency '{name}' registered in module '{module}'")
        
        def _mock_dependency(*a):
            """Create a mock for dependency: mock_dependency(name, mock_value, [module])"""
            if len(a) < 2 or len(a) > 3:
                return EvaluationError("mock_dependency() takes 2 or 3 arguments: name, mock, [module]")
            if not isinstance(a[0], String):
                return EvaluationError("mock_dependency() name must be a string")
            
            from ..dependency_injection import get_di_registry, ExecutionMode
            
            name = a[0].value
            mock = a[1]
            module = a[2].value if len(a) > 2 and isinstance(a[2], String) else "__main__"
            
            registry = get_di_registry()
            container = registry.get_container(module)
            if not container:
                # Create container if it doesn't exist
                registry.register_module(module)
                container = registry.get_container(module)
            # Declare and mock the dependency
            if name not in container.contracts:
                container.declare_dependency(name, "any", False)
            container.mock(name, mock)
            
            return String(f"Mock for '{name}' registered in module '{module}'")
        
        def _clear_mocks(*a):
            """Clear all mocks: clear_mocks([module])"""
            from ..dependency_injection import get_di_registry
            
            module = a[0].value if len(a) > 0 and isinstance(a[0], String) else "__main__"
            
            registry = get_di_registry()
            container = registry.get_container(module)
            if container:
                container.clear_mocks()
                return String(f"All mocks cleared in module '{module}'")
            return String(f"Module '{module}' not registered")
        
        def _set_execution_mode(*a):
            """Set execution mode: set_execution_mode(mode_string)"""
            if len(a) != 1:
                return EvaluationError("set_execution_mode() takes 1 argument: mode")
            if not isinstance(a[0], String):
                return EvaluationError("set_execution_mode() mode must be a string")
            
            from ..dependency_injection import ExecutionMode
            
            mode_str = a[0].value.upper()
            try:
                mode = ExecutionMode[mode_str]
                # Store in current environment
                env = getattr(self, '_current_env', None)
                if env:
                    env.set("__execution_mode__", String(mode_str))
                return String(f"Execution mode set to {mode.name}")
            except KeyError:
                return EvaluationError(f"Invalid execution mode: {mode_str}. Valid: PRODUCTION, DEBUG, TEST, SANDBOX")
        
        # Register all advanced feature builtins
        self.builtins.update({
            # Persistence
            "persistent_set": Builtin(_persistent_set, "persistent_set"),
            "persistent_get": Builtin(_persistent_get, "persistent_get"),
            "persistent_delete": Builtin(_persistent_delete, "persistent_delete"),
            "memory_stats": Builtin(_memory_stats, "memory_stats"),
            # Policy
            "create_policy": Builtin(_create_policy, "create_policy"),
            "check_policy": Builtin(_check_policy, "check_policy"),
            # Dependency Injection
            "register_dependency": Builtin(_register_dependency, "register_dependency"),
            "mock_dependency": Builtin(_mock_dependency, "mock_dependency"),
            "clear_mocks": Builtin(_clear_mocks, "clear_mocks"),
            "set_execution_mode": Builtin(_set_execution_mode, "set_execution_mode"),
        })
    
    def _register_renderer_builtins(self):
        """Logic extracted from the original RENDER_REGISTRY and helper functions."""
        
        # Mix
        def builtin_mix(*args):
            if len(args) != 3: 
                return EvaluationError("mix(colorA, colorB, ratio)")
            a, b, ratio = args
            a_name = _to_str(a)
            b_name = _to_str(b)
            
            try:
                ratio_val = float(ratio.value) if isinstance(ratio, (Integer, Float)) else float(str(ratio))
            except Exception:
                ratio_val = 0.5
            
            if _BACKEND_AVAILABLE:
                try:
                    res = _BACKEND.mix(a_name, b_name, ratio_val)
                    return String(str(res))
                except Exception:
                    pass
            
            return String(f"mix({a_name},{b_name},{ratio_val})")
        
        # Define Screen
        def builtin_define_screen(*args):
            if len(args) < 1: 
                return EvaluationError("define_screen() requires at least a name")
            
            name = _to_str(args[0])
            props = _zexus_to_python(args[1]) if len(args) > 1 else {}
            
            if _BACKEND_AVAILABLE:
                try:
                    _BACKEND.define_screen(name, props)
                    return NULL
                except Exception as e:
                    return EvaluationError(str(e))
            
            self.render_registry['screens'].setdefault(name, {
                'properties': props, 
                'components': [], 
                'theme': None
            })
            return NULL
        
        # Define Component
        def builtin_define_component(*args):
            if len(args) < 1: 
                return EvaluationError("define_component() requires at least a name")
            
            name = _to_str(args[0])
            props = _zexus_to_python(args[1]) if len(args) > 1 else {}
            
            if _BACKEND_AVAILABLE:
                try:
                    _BACKEND.define_component(name, props)
                    return NULL
                except Exception as e:
                    return EvaluationError(str(e))
            
            self.render_registry['components'][name] = props
            return NULL
        
        # Add to Screen
        def builtin_add_to_screen(*args):
            if len(args) != 2: 
                return EvaluationError("add_to_screen() requires (screen_name, component_name)")
            
            screen = _to_str(args[0])
            comp = _to_str(args[1])
            
            if _BACKEND_AVAILABLE:
                try:
                    _BACKEND.add_to_screen(screen, comp)
                    return NULL
                except Exception as e:
                    return EvaluationError(str(e))
            
            if screen not in self.render_registry['screens']:
                return EvaluationError(f"Screen '{screen}' not found")
            
            self.render_registry['screens'][screen]['components'].append(comp)
            return NULL
        
        # Render Screen
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
            
            screen = self.render_registry['screens'].get(name)
            if not screen: 
                return String(f"<missing screen: {name}>")
            
            return String(f"Screen:{name} props={screen.get('properties')} components={screen.get('components')}")
        
        # Set Theme
        def builtin_set_theme(*args):
            if len(args) == 1:
                theme_name = _to_str(args[0])
                if _BACKEND_AVAILABLE:
                    try:
                        _BACKEND.set_theme(theme_name)
                        return NULL
                    except Exception as e:
                        return EvaluationError(str(e))
                
                self.render_registry['current_theme'] = theme_name
                return NULL
            
            if len(args) == 2:
                target = _to_str(args[0])
                theme_name = _to_str(args[1])
                
                if _BACKEND_AVAILABLE:
                    try:
                        _BACKEND.set_theme(target, theme_name)
                        return NULL
                    except Exception as e:
                        return EvaluationError(str(e))
                
                if target in self.render_registry['screens']:
                    self.render_registry['screens'][target]['theme'] = theme_name
                else:
                    self.render_registry['themes'].setdefault(theme_name, {})
                
                return NULL
            
            return EvaluationError("set_theme() requires 1 (theme) or 2 (target,theme) args")
        
        # Canvas Ops
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
            
            cid = f"canvas_{len(self.render_registry['canvases'])+1}"
            self.render_registry['canvases'][cid] = {
                'width': wid, 
                'height': hei, 
                'draw_ops': []
            }
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
            
            canvas = self.render_registry['canvases'].get(cid)
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
            
            canvas = self.render_registry['canvases'].get(cid)
            if not canvas:
                return EvaluationError(f"Canvas {cid} not found")
            
            canvas['draw_ops'].append(('text', (x, y, text)))
            return NULL
        
        # Register renderer builtins
        self.builtins.update({
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
    
    def _register_verification_builtins(self):
        """Register verification helper functions for VERIFY keyword"""
        import re
        import os
        
        def _is_email(*a):
            """Check if string is valid email format"""
            if len(a) != 1:
                return EvaluationError("is_email() takes 1 argument")
            
            val = a[0]
            email_str = val.value if isinstance(val, String) else str(val)
            
            # Simple email validation pattern
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            is_valid = bool(re.match(pattern, email_str))
            return TRUE if is_valid else FALSE
        
        def _is_url(*a):
            """Check if string is valid URL format"""
            if len(a) != 1:
                return EvaluationError("is_url() takes 1 argument")
            
            val = a[0]
            url_str = val.value if isinstance(val, String) else str(val)
            
            # Simple URL validation pattern
            pattern = r'^https?://[^\s/$.?#].[^\s]*$'
            is_valid = bool(re.match(pattern, url_str))
            return TRUE if is_valid else FALSE
        
        def _is_phone(*a):
            """Check if string is valid phone number format"""
            if len(a) != 1:
                return EvaluationError("is_phone() takes 1 argument")
            
            val = a[0]
            phone_str = val.value if isinstance(val, String) else str(val)
            
            # Remove common separators
            clean = re.sub(r'[\s\-\(\)\.]', '', phone_str)
            
            # Check if it's digits and reasonable length
            is_valid = clean.isdigit() and 10 <= len(clean) <= 15
            return TRUE if is_valid else FALSE
        
        def _is_numeric(*a):
            """Check if string contains only numbers"""
            if len(a) != 1:
                return EvaluationError("is_numeric() takes 1 argument")
            
            val = a[0]
            if isinstance(val, (Integer, Float)):
                return TRUE
            
            str_val = val.value if isinstance(val, String) else str(val)
            
            try:
                float(str_val)
                return TRUE
            except ValueError:
                return FALSE
        
        def _is_alpha(*a):
            """Check if string contains only alphabetic characters"""
            if len(a) != 1:
                return EvaluationError("is_alpha() takes 1 argument")
            
            val = a[0]
            str_val = val.value if isinstance(val, String) else str(val)
            
            is_valid = str_val.isalpha()
            return TRUE if is_valid else FALSE
        
        def _is_alphanumeric(*a):
            """Check if string contains only alphanumeric characters"""
            if len(a) != 1:
                return EvaluationError("is_alphanumeric() takes 1 argument")
            
            val = a[0]
            str_val = val.value if isinstance(val, String) else str(val)
            
            is_valid = str_val.isalnum()
            return TRUE if is_valid else FALSE
        
        def _matches_pattern(*a):
            """Check if string matches regex pattern: matches_pattern(value, pattern)"""
            if len(a) != 2:
                return EvaluationError("matches_pattern() takes 2 arguments: value, pattern")
            
            val = a[0]
            pattern_obj = a[1]
            
            str_val = val.value if isinstance(val, String) else str(val)
            pattern = pattern_obj.value if isinstance(pattern_obj, String) else str(pattern_obj)
            
            try:
                is_valid = bool(re.match(pattern, str_val))
                return TRUE if is_valid else FALSE
            except Exception as e:
                return EvaluationError(f"Pattern matching error: {str(e)}")
        
        def _env_get(*a):
            """Get environment variable: env_get("VAR_NAME") or env_get("VAR_NAME", "default")"""
            if len(a) < 1 or len(a) > 2:
                return EvaluationError("env_get() takes 1 or 2 arguments: var_name, [default]")
            
            var_name_obj = a[0]
            var_name = var_name_obj.value if isinstance(var_name_obj, String) else str(var_name_obj)
            
            default = a[1] if len(a) == 2 else None
            
            value = os.environ.get(var_name)
            
            if value is None:
                return default if default is not None else NULL
            
            return String(value)
        
        def _env_set(*a):
            """Set environment variable: env_set("VAR_NAME", "value")"""
            if len(a) != 2:
                return EvaluationError("env_set() takes 2 arguments: var_name, value")
            
            var_name_obj = a[0]
            value_obj = a[1]
            
            var_name = var_name_obj.value if isinstance(var_name_obj, String) else str(var_name_obj)
            value = value_obj.value if isinstance(value_obj, String) else str(value_obj)
            
            os.environ[var_name] = value
            return TRUE
        
        def _env_exists(*a):
            """Check if environment variable exists: env_exists("VAR_NAME")"""
            if len(a) != 1:
                return EvaluationError("env_exists() takes 1 argument: var_name")
            
            var_name_obj = a[0]
            var_name = var_name_obj.value if isinstance(var_name_obj, String) else str(var_name_obj)
            
            exists = var_name in os.environ
            return TRUE if exists else FALSE
        
        def _password_strength(*a):
            """Check password strength: password_strength(password) -> "weak"/"medium"/"strong" """
            if len(a) != 1:
                return EvaluationError("password_strength() takes 1 argument")
            
            val = a[0]
            password = val.value if isinstance(val, String) else str(val)
            
            score = 0
            length = len(password)
            
            # Length check
            if length >= 8:
                score += 1
            if length >= 12:
                score += 1
            
            # Complexity checks
            if re.search(r'[a-z]', password):
                score += 1
            if re.search(r'[A-Z]', password):
                score += 1
            if re.search(r'[0-9]', password):
                score += 1
            if re.search(r'[^a-zA-Z0-9]', password):
                score += 1
            
            if score <= 2:
                return String("weak")
            elif score <= 4:
                return String("medium")
            else:
                return String("strong")
        
        def _sanitize_input(*a):
            """Sanitize user input by removing dangerous characters"""
            if len(a) != 1:
                return EvaluationError("sanitize_input() takes 1 argument")
            
            val = a[0]
            input_str = val.value if isinstance(val, String) else str(val)
            
            # Remove potentially dangerous characters
            # Remove HTML tags
            sanitized = re.sub(r'<[^>]+>', '', input_str)
            # Remove script tags
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE)
            # Remove SQL injection patterns
            sanitized = re.sub(r'(;|--|\'|\"|\bOR\b|\bAND\b)', '', sanitized, flags=re.IGNORECASE)
            
            return String(sanitized)
        
        def _validate_length(*a):
            """Validate string length: validate_length(value, min, max)"""
            if len(a) != 3:
                return EvaluationError("validate_length() takes 3 arguments: value, min, max")
            
            val = a[0]
            min_len_obj = a[1]
            max_len_obj = a[2]
            
            str_val = val.value if isinstance(val, String) else str(val)
            min_len = min_len_obj.value if isinstance(min_len_obj, Integer) else int(min_len_obj)
            max_len = max_len_obj.value if isinstance(max_len_obj, Integer) else int(max_len_obj)
            
            length = len(str_val)
            is_valid = min_len <= length <= max_len
            
            return TRUE if is_valid else FALSE
        
        # Register verification builtins
        self.builtins.update({
            "is_email": Builtin(_is_email, "is_email"),
            "is_url": Builtin(_is_url, "is_url"),
            "is_phone": Builtin(_is_phone, "is_phone"),
            "is_numeric": Builtin(_is_numeric, "is_numeric"),
            "is_alpha": Builtin(_is_alpha, "is_alpha"),
            "is_alphanumeric": Builtin(_is_alphanumeric, "is_alphanumeric"),
            "matches_pattern": Builtin(_matches_pattern, "matches_pattern"),
            "env_get": Builtin(_env_get, "env_get"),
            "env_set": Builtin(_env_set, "env_set"),
            "env_exists": Builtin(_env_exists, "env_exists"),
            "password_strength": Builtin(_password_strength, "password_strength"),
            "sanitize_input": Builtin(_sanitize_input, "sanitize_input"),
            "validate_length": Builtin(_validate_length, "validate_length"),
        })
