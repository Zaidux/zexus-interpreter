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
            
            res = self.eval_node(fn.body, new_env)
            res = _resolve_awaitable(res)
            
            # Unwrap ReturnValue if needed
            if isinstance(res, ReturnValue):
                result = res.value
            else:
                result = res
            
            # Phase 2: Trigger after-call hook
            if hasattr(self, 'integration_context'):
                func_name = fn.name if hasattr(fn, 'name') else str(fn)
                self.integration_context.plugins.after_action_call(func_name, result)
            
            return result
        
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
        from ..complexity_system import Module
        if isinstance(obj, Module):
            # For module methods, get the member and call it if it's a function
            member_value = obj.get(method_name)
            if member_value is None:
                return EvaluationError(f"Method '{method_name}' not found in module '{obj.name}'")
            
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
        
        return EvaluationError(f"Method '{method_name}' not supported for {obj.type()}")
    
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
        
        # Register mappings
        self.builtins.update({
            "now": Builtin(_now, "now"),
            "timestamp": Builtin(_timestamp, "timestamp"),
            "random": Builtin(_random, "random"),
            "to_hex": Builtin(_to_hex, "to_hex"),
            "from_hex": Builtin(_from_hex, "from_hex"),
            "sqrt": Builtin(_sqrt, "sqrt"),
            "file_read_text": Builtin(_read_text, "file_read_text"),
            "file_write_text": Builtin(_write_text, "file_write_text"),
            "file_exists": Builtin(_exists, "file_exists"),
            "file_read_json": Builtin(_read_json, "file_read_json"),
            "file_write_json": Builtin(_write_json, "file_write_json"),
            "file_append": Builtin(_append, "file_append"),
            "file_list_dir": Builtin(_list_dir, "file_list_dir"),
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
