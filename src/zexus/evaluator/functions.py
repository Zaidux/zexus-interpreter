# src/zexus/evaluator/functions.py
import sys
import os

from .. import zexus_ast
from ..zexus_ast import CallExpression, MethodCallExpression
from ..object import (
    Environment, Integer, Float, String, List, Map, Boolean as BooleanObj,
    Null, Builtin, Action, LambdaFunction, ReturnValue, DateTime, Math, File, Debug,
    EvaluationError, EntityDefinition
)
from .utils import is_error, debug_log, NULL, TRUE, FALSE, _resolve_awaitable, _zexus_to_python, _python_to_zexus, _to_str

# Try to import backend, handle failure gracefully (as per your original code)
try:
    from ..renderer import backend as _BACKEND
    _BACKEND_AVAILABLE = True
except Exception:
    _BACKEND_AVAILABLE = False
    _BACKEND = None

class FunctionEvaluatorMixin:
    """Handles function application, method calls, and defines all builtins."""
    
    def __init__(self):
        # Initialize registries
        self.builtins = {}
        self._allow_coroutine_result = False
        self._pending_revert_signature = None
        self._tolerant_skip_counts = {}
        
        # Renderer Registry (moved from global scope to instance scope)
        self.render_registry = {
            'screens': {},
            'components': {},
            'themes': {},
            'canvases': {},
            'canvas_aliases': {},
            'colours': {},
            'graphics': {},
            'animations': {},
            'clocks': {},
            'current_theme': None
        }
        
        # Register all functions
        self._register_core_builtins()
        self._register_main_entry_point_builtins()
        self._register_renderer_builtins()
    
    def _execute_coroutine_to_completion(self, coroutine):
        from ..object import EvaluationError

        max_steps = 10000
        pending_value = None

        for _ in range(max_steps):
            done, value = coroutine.resume(pending_value)
            if done:
                if isinstance(value, EvaluationError):
                    return value
                return value if value is not None else coroutine.result
            pending_value = None

        return EvaluationError("Coroutine did not complete within step limit")

    def _compute_revert_signature(self, reason_node):
        if reason_node is None:
            return ("none",)

        try:
            from .. import zexus_ast
            if isinstance(reason_node, zexus_ast.StringLiteral):
                return ("string", getattr(reason_node, "value", None))
            if isinstance(reason_node, zexus_ast.Identifier):
                return ("identifier", getattr(reason_node, "value", None))
        except Exception:
            pass

        return ("expr", repr(reason_node))

    def eval_call_expression(self, node, env, stack_trace):
        debug_log("🚀 CallExpression node", f"Calling {node.function}")
        
        if (
            isinstance(node.function, zexus_ast.Identifier)
            and getattr(node.function, "value", None) == "revert"
        ):
            if len(node.arguments) > 1:
                return EvaluationError(
                    "revert() accepts at most one argument",
                    stack_trace=stack_trace,
                )

            debug_log(
                "eval_call_expression",
                "Treating call expression as revert statement",
            )

            reason_expr = node.arguments[0] if node.arguments else None
            result = self._perform_revert(reason_expr, env, stack_trace)
            self._pending_revert_signature = self._compute_revert_signature(reason_expr)
            return result

        fn = self.eval_node(node.function, env, stack_trace)
        if is_error(fn):
            return fn
        
        # Check if this is a generic type instantiation: Box<number>(42)
        type_args = getattr(node, 'type_args', [])
        if type_args and hasattr(fn, 'is_generic') and fn.is_generic:
            debug_log("  Generic type instantiation", f"Type args: {type_args}")
            
            # Create specialized constructor for this type combination
            template = fn.generic_template
            specialized_constructor = self._create_specialized_generic_constructor(
                template, type_args, env, stack_trace
            )
            
            if is_error(specialized_constructor):
                return specialized_constructor
            
            # Now call the specialized constructor with the actual arguments
            fn = specialized_constructor
        
        # Check if arguments contain keyword arguments (AssignmentExpression nodes)
        # This handles syntax like: Person(name="Bob", age=25)
        has_keyword_args = any(isinstance(arg, zexus_ast.AssignmentExpression) for arg in node.arguments)
        
        if has_keyword_args:
            # Process keyword arguments - build a dict of name->value
            kwargs = {}
            for arg in node.arguments:
                if isinstance(arg, zexus_ast.AssignmentExpression):
                    # This is a keyword argument: name=value
                    param_name = arg.name.value if hasattr(arg.name, 'value') else str(arg.name)
                    param_value = self.eval_node(arg.value, env, stack_trace)
                    if is_error(param_value):
                        return param_value
                    kwargs[param_name] = param_value
                else:
                    # Mixed positional and keyword args not supported yet
                    return EvaluationError("Cannot mix positional and keyword arguments")
            
            # For entity constructors, pass kwargs as-is
            from ..security import EntityDefinition as SecurityEntityDef
            if isinstance(fn, SecurityEntityDef):
                return fn.create_instance(kwargs)
            
            # For other functions, would need to map kwargs to positional args
            # Not implemented yet
            return EvaluationError("Keyword arguments only supported for entity constructors currently")
        
        # Regular positional arguments
        args = self.eval_expressions(node.arguments, env)
        if is_error(args): 
            return args
        
        arg_count = len(args) if isinstance(args, (list, tuple)) else "unknown"
        debug_log("  Arguments evaluated", f"{args} (count: {arg_count})")
        
        # Contract instantiation check
        from ..security import SmartContract
        if isinstance(fn, SmartContract):
            return fn.instantiate(args)
        
        should_allow_coroutine = (
            isinstance(fn, (Action, LambdaFunction))
            and getattr(fn, "is_async", False)
        )

        previous_allow = getattr(self, "_allow_coroutine_result", False)
        if should_allow_coroutine:
            self._allow_coroutine_result = True

        try:
            return self.apply_function(fn, args, env)
        finally:
            if should_allow_coroutine:
                self._allow_coroutine_result = previous_allow
    
    def _create_specialized_generic_constructor(self, template, type_args, env, stack_trace):
        """Create a specialized constructor for a generic type with concrete type arguments
        
        Example: Box<number> creates a specialized Box constructor with T = number
        """
        from ..object import EvaluationError, String, Map
        
        debug_log("_create_specialized_generic_constructor", f"Specializing with types: {type_args}")
        
        type_params = template['type_params']
        
        if len(type_args) != len(type_params):
            return EvaluationError(
                f"Generic type requires {len(type_params)} type argument(s), got {len(type_args)}"
            )
        
        # Create specialized type name
        specialized_type_name = f"{template['type_name']}<{', '.join(type_args)}>"
        
        # Check if we've already created this specialization (cache it)
        existing = template['env'].get(specialized_type_name)
        if existing and not is_error(existing):
            debug_log("  Using cached specialization", specialized_type_name)
            return existing
        
        # Create type substitution map: T -> number, U -> string, etc.
        type_subst = dict(zip(type_params, type_args))
        debug_log("  Type substitution map", str(type_subst))
        
        # Create a specialized version of the fields with type substitution
        specialized_fields = []
        for field in template['fields']:
            # Create a copy of the field with substituted type
            field_type = field.field_type
            
            # If field type is a type parameter, substitute it
            if field_type in type_subst:
                field_type = type_subst[field_type]
                debug_log(f"  Substituted field type", f"{field.name}: {field_type}")
            
            # Create new field with substituted type
            specialized_field = zexus_ast.DataField(
                name=field.name,
                field_type=field_type,
                default_value=field.default_value,
                constraint=field.constraint,
                computed=field.computed,
                method_body=field.method_body,
                method_params=field.method_params,
                operator=field.operator,
                decorators=field.decorators
            )
            specialized_fields.append(specialized_field)
        
        # Create a specialized DataStatement node
        specialized_node = zexus_ast.DataStatement(
            name=zexus_ast.Identifier(specialized_type_name),
            fields=specialized_fields,
            modifiers=template['modifiers'],
            parent=template['parent_type'],
            decorators=template['decorators'],
            type_params=[]  # No longer generic after specialization
        )
        
        # Evaluate the specialized data statement to create the constructor
        # This will register it in the environment
        evaluator = template['evaluator']
        result = evaluator.eval_data_statement(
            specialized_node,
            template['env'],
            template['stack_trace']
        )
        
        if is_error(result):
            return result
        
        # Get the constructor from the environment (it was registered by eval_data_statement)
        constructor = template['env'].get(specialized_type_name)
        
        if constructor is None:
            return EvaluationError(f"Failed to create specialized constructor for {specialized_type_name}")
        
        return constructor
    
    def apply_function(self, fn, args, env=None):
        debug_log("apply_function", f"Calling {fn}")
        
        # Resource limit: Track call depth (Security Fix #7)
        func_name = None
        if isinstance(fn, (Action, LambdaFunction)):
            func_name = fn.name if hasattr(fn, 'name') else str(fn)
            try:
                self.resource_limiter.enter_call(func_name)
            except Exception as e:
                # Convert ResourceError to EvaluationError
                from .resource_limiter import ResourceError, TimeoutError
                if isinstance(e, (ResourceError, TimeoutError)):
                    return EvaluationError(str(e))
                raise  # Re-raise if not a resource error
        
        try:
            # Phase 2 & 3: Trigger plugin hooks and check capabilities
            if hasattr(self, 'integration_context'):
                if isinstance(fn, (Action, LambdaFunction)):
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
                    import sys
                    
                    # print(f"[ASYNC CREATE] Creating coroutine for async action, fn.env has keys: {list(fn.env.store.keys()) if hasattr(fn.env, 'store') else 'N/A'}", file=sys.stderr)
                    
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
                    
                    # Create coroutine and decide how to surface result
                    gen = async_generator()
                    coroutine = Coroutine(gen, fn)
                    if getattr(self, "_allow_coroutine_result", False):
                        return coroutine
                    return self._execute_coroutine_to_completion(coroutine)
                
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
                
                result = None
                try:
                    res = self.eval_node(fn.body, new_env)
                    res = _resolve_awaitable(res)
                    
                    # Unwrap ReturnValue if needed
                    if isinstance(res, ReturnValue):
                        result = res.value
                    else:
                        result = res
                    
                    return result
                except Exception as e:
                    # Store result for after-call hook
                    result = EvaluationError(str(e))
                    raise
                except BaseException as e:
                    # Ensure result is set even for interrupts/SystemExit
                    result = EvaluationError(str(e))
                    raise
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
            
            elif isinstance(fn, EntityDefinition):
                debug_log("  Creating entity instance (old format)")
                # Entity constructor: Person("Alice", 30) or Person{name: "Alice", age: 30}
                # Create instance with positional arguments mapped to properties
                from ..object import EntityInstance, String, Integer
                
                values = {}
                
                # Special case: If single argument is a Map, use it as field values
                # This handles Entity{field: value} syntax which becomes Entity(Map{...})
                if len(args) == 1 and isinstance(args[0], Map):
                    debug_log("  Single Map argument detected - using as field values")
                    map_arg = args[0]
                    # Extract key-value pairs from the Map
                    for key, value in map_arg.pairs.items():
                        # Convert key to string if it's a String object
                        key_str = key.value if isinstance(key, String) else str(key)
                        values[key_str] = value
                else:
                    # Map positional arguments to property names
                    if isinstance(fn.properties, dict):
                        prop_names = list(fn.properties.keys())
                    else:
                        prop_names = [prop['name'] for prop in fn.properties]
                    
                    for i, arg in enumerate(args):
                        if i < len(prop_names):
                            values[prop_names[i]] = arg
                
                return EntityInstance(fn, values)
            
            # Handle SecurityEntityDefinition (from security.py with methods support)
            from ..security import EntityDefinition as SecurityEntityDef, EntityInstance as SecurityEntityInstance
            from ..object import String
            if isinstance(fn, SecurityEntityDef):
                debug_log("  Creating entity instance (with methods)")
                
                values = {}
                
                # Special case: If single argument is a Map, use it as field values
                # This handles Entity{field: value} syntax which becomes Entity(Map{...})
                if len(args) == 1 and isinstance(args[0], Map):
                    debug_log("  Single Map argument detected - using as field values")
                    map_arg = args[0]
                    # Extract key-value pairs from the Map
                    for key, value in map_arg.pairs.items():
                        # Convert key to string if it's a String object
                        key_str = key.value if isinstance(key, String) else str(key)
                        values[key_str] = value
                else:
                    # Map positional arguments to property names, INCLUDING INHERITED PROPERTIES
                    # Use get_all_properties() to get the full property list in correct order
                    if hasattr(fn, 'get_all_properties'):
                        # Get all properties (parent + child) in correct order
                        all_props = fn.get_all_properties()
                        prop_names = list(all_props.keys())
                    else:
                        # Fallback for old-style properties
                        prop_names = list(fn.properties.keys()) if isinstance(fn.properties, dict) else [prop['name'] for prop in fn.properties]
                    
                    for i, arg in enumerate(args):
                        if i < len(prop_names):
                            values[prop_names[i]] = arg
                
                debug_log(f"  Entity instance created with {len(values)} properties: {list(values.keys())}")
                # Use create_instance to handle dependency injection
                return fn.create_instance(values)
        finally:
            # Resource limit: Exit call depth tracking (Security Fix #7)
            if isinstance(fn, (Action, LambdaFunction)):
                self.resource_limiter.exit_call()
        
        return EvaluationError(f"Not a function: {fn}")
    
    def eval_method_call_expression(self, node, env, stack_trace):
        debug_log("  MethodCallExpression node", f"{node.object}.{node.method}")
        
        debug_log("DEBUG evaluating method call object", node.object)
        obj = self.eval_node(node.object, env, stack_trace)
        debug_log("DEBUG object evaluated to", obj)
        if is_error(obj): 
            return obj
        
        method_name = node.method.value
        
        # === Builtin Static Methods ===
        # Handle static methods on dataclass constructors (e.g., User.default(), User.fromJSON())
        if isinstance(obj, Builtin):
            if hasattr(obj, 'static_methods') and method_name in obj.static_methods:
                static_method = obj.static_methods[method_name]
                args = self.eval_expressions(node.arguments, env)
                if is_error(args):
                    return args
                return self.apply_function(static_method, args, env)
            # If no static method found, fall through to error
        
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
            # First check if the method is a callable stored in the Map (for DATA dataclasses)
            method_key = String(method_name)
            if method_key in obj.pairs:
                method_value = obj.pairs[method_key]
                if isinstance(method_value, Builtin):
                    # This is a dataclass method - evaluate args and call it
                    args = self.eval_expressions(node.arguments, env)
                    if is_error(args):
                        return args
                    return self.apply_function(method_value, args, env)
            
            # Otherwise handle built-in Map methods
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
            elif method_name == "keys":
                # Return array of all keys
                return List([String(k) if isinstance(k, str) else k for k in obj.pairs.keys()])
            elif method_name == "values":
                # Return array of all values
                return List([v for v in obj.pairs.values()])
            elif method_name == "entries":
                # Return array of [key, value] pairs
                return List([List([String(k) if isinstance(k, str) else k, v]) for k, v in obj.pairs.items()])
            elif method_name == "size":
                # Return number of entries
                return Integer(len(obj.pairs))

        
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
        
        # === Entity Instance Methods ===
        from ..security import EntityInstance as SecurityEntityInstance
        if isinstance(obj, SecurityEntityInstance):
            args = self.eval_expressions(node.arguments, env)
            if is_error(args):
                return args
            return obj.call_method(method_name, args)
        
        # === Contract Instance Methods ===
        if hasattr(obj, 'call_method'):
            args = self.eval_expressions(node.arguments, env)
            if is_error(args): 
                return args
            result = obj.call_method(method_name, args)
            
            # CRITICAL: Sync state variables back to caller's environment after internal call
            # This ensures that when action A calls this.actionB(), any state changes made
            # by actionB are visible to actionA's local environment
            # OPTIMIZATION: Only sync for 'this' calls (internal calls within same contract)
            from ..security import SmartContract
            if isinstance(obj, SmartContract):
                # Check if this is a 'this.method()' call by checking the caller's env
                this_ref = env.get('this')
                if this_ref is obj and hasattr(obj, 'storage_vars'):
                    # This is an internal call - sync state variables
                    for var_node in obj.storage_vars:
                        var_name = None
                        if hasattr(var_node, 'name'):
                            var_name = var_node.name.value if hasattr(var_node.name, 'value') else var_node.name
                        elif isinstance(var_node, dict):
                            var_name = var_node.get("name")
                        elif isinstance(var_node, str):
                            var_name = var_node
                        
                        if var_name:
                            # Read the updated value from storage and refresh the caller's env
                            updated_value = obj.storage.get(var_name)
                            if updated_value is not None:
                                env.set(var_name, updated_value)
            
            # Unwrap ReturnValue if needed
            from ..object import ReturnValue
            if isinstance(result, ReturnValue):
                return result.value
            return result
        
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
        debug_log("DEBUG method dispatch failure", f"method={method_name}, obj={obj}, obj_type={obj_type}")
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
        
        def _time(*a):
            """Get current time in milliseconds since epoch (for performance measurement)"""
            import time
            return Integer(int(time.time() * 1000))
        
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
        
        # User Input (SECURITY: Returns untrusted strings)
        def _input(*a):
            """Read user input from stdin - automatically marked as untrusted"""
            prompt = ""
            if len(a) == 1:
                if isinstance(a[0], String):
                    prompt = a[0].value
                else:
                    prompt = str(a[0].inspect() if hasattr(a[0], 'inspect') else a[0])
            elif len(a) > 1:
                return EvaluationError("input() takes 0 or 1 argument (optional prompt)")
            
            try:
                user_input = input(prompt)
                # SECURITY: User input is ALWAYS untrusted - external data source
                return String(user_input, is_trusted=False)
            except Exception as e:
                return EvaluationError(f"input() error: {str(e)}")
        
        # Cryptographic Functions (SECURITY FIX #5)
        def _hash_password(*a):
            """
            Hash password using bcrypt (secure, industry-standard)
            Usage: hash_password(password) -> hashed_string
            """
            if len(a) != 1:
                return EvaluationError("hash_password() takes exactly 1 argument")
            
            password = a[0].value if isinstance(a[0], String) else str(a[0])
            
            try:
                import bcrypt
                # Generate salt and hash password
                salt = bcrypt.gensalt()
                hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
                # Return as trusted string (hash, not user input)
                return String(hashed.decode('utf-8'), is_trusted=True)
            except ImportError:
                return EvaluationError("hash_password() requires bcrypt library. Install: pip install bcrypt")
            except Exception as e:
                return EvaluationError(f"hash_password() error: {str(e)}")
        
        def _verify_password(*a):
            """
            Verify password against bcrypt hash (constant-time comparison)
            Usage: verify_password(password, hash) -> boolean
            """
            if len(a) != 2:
                return EvaluationError("verify_password() takes exactly 2 arguments: password, hash")
            
            password = a[0].value if isinstance(a[0], String) else str(a[0])
            password_hash = a[1].value if isinstance(a[1], String) else str(a[1])
            
            try:
                import bcrypt
                # Constant-time comparison via bcrypt
                result = bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
                return BooleanObj(result)
            except ImportError:
                return EvaluationError("verify_password() requires bcrypt library. Install: pip install bcrypt")
            except Exception as e:
                return EvaluationError(f"verify_password() error: {str(e)}")
        
        def _crypto_random(*a):
            """
            Generate cryptographically secure random string
            Usage: crypto_random(length?) -> hex_string (default 32 bytes = 64 hex chars)
            """
            length = 32  # Default: 32 bytes
            if len(a) >= 1:
                if isinstance(a[0], Integer):
                    length = a[0].value
                else:
                    return EvaluationError("crypto_random() length must be an integer")
            
            if len(a) > 1:
                return EvaluationError("crypto_random() takes 0 or 1 argument (optional length)")
            
            try:
                import secrets
                # Generate cryptographically secure random hex string
                random_hex = secrets.token_hex(length)
                # Return as trusted string (generated, not user input)
                return String(random_hex, is_trusted=True)
            except Exception as e:
                return EvaluationError(f"crypto_random() error: {str(e)}")
        
        def _constant_time_compare(*a):
            """
            Constant-time string comparison (timing-attack resistant)
            Usage: constant_time_compare(a, b) -> boolean
            """
            if len(a) != 2:
                return EvaluationError("constant_time_compare() takes exactly 2 arguments")
            
            str_a = a[0].value if isinstance(a[0], String) else str(a[0])
            str_b = a[1].value if isinstance(a[1], String) else str(a[1])
            
            try:
                import secrets
                # Use secrets.compare_digest for constant-time comparison
                result = secrets.compare_digest(str_a, str_b)
                return BooleanObj(result)
            except Exception as e:
                return EvaluationError(f"constant_time_compare() error: {str(e)}")
        
        # File I/O  — VFS-aware helpers
        # The VFS manager (with its file cache) is obtained lazily so the
        # evaluator can function even when the integration layer is not loaded.
        _vfs_mgr = None
        def _get_vfs_manager():
            nonlocal _vfs_mgr
            if _vfs_mgr is None:
                try:
                    from .integration import get_integration
                    _vfs_mgr = get_integration().vfs_manager
                except Exception:
                    pass
            return _vfs_mgr

        def _resolve_read_path(path_str):
            """Resolve a path for reading – uses VFS cache when available."""
            import os
            if not os.path.isabs(path_str):
                path_str = os.path.join(os.getcwd(), path_str)
            return os.path.normpath(path_str)

        def _cached_read(real_path):
            """Read via VFS cache (skips disk if file unchanged)."""
            mgr = _get_vfs_manager()
            if mgr is not None:
                return mgr.cached_read(real_path)
            with open(real_path, 'r') as f:
                return f.read()

        def _invalidate_file_cache(real_path):
            """Invalidate VFS cache after a write."""
            mgr = _get_vfs_manager()
            if mgr is not None:
                mgr.invalidate_cache(real_path)

        def _read_text(*a): 
            cap_err = _check_io_read_capability()
            if cap_err: return cap_err
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("file_read_text() takes exactly 1 string argument")
            return File.read_text(a[0])
        
        def _write_text(*a): 
            cap_err = _check_io_write_capability()
            if cap_err: return cap_err
            if len(a) != 2 or not all(isinstance(x, String) for x in a): 
                return EvaluationError("file_write_text() takes exactly 2 string arguments")
            result = File.write_text(a[0], a[1])
            _invalidate_file_cache(_resolve_read_path(a[0].value))
            return result
        
        def _exists(*a): 
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("file_exists() takes exactly 1 string argument")
            return File.exists(a[0])
        
        def _read_json(*a): 
            cap_err = _check_io_read_capability()
            if cap_err: return cap_err
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("file_read_json() takes exactly 1 string argument")
            return File.read_json(a[0])
        
        def _write_json(*a):
            cap_err = _check_io_write_capability()
            if cap_err: return cap_err
            if len(a) != 2 or not isinstance(a[0], String): 
                return EvaluationError("file_write_json() takes path string and data")
            result = File.write_json(a[0], a[1])
            _invalidate_file_cache(_resolve_read_path(a[0].value))
            return result
        
        def _file_append(*a): 
            cap_err = _check_io_write_capability()
            if cap_err: return cap_err
            if len(a) != 2 or not all(isinstance(x, String) for x in a): 
                return EvaluationError("file_append() takes exactly 2 string arguments")
            result = File.append_text(a[0], a[1])
            _invalidate_file_cache(_resolve_read_path(a[0].value))
            return result
        
        def _list_dir(*a): 
            cap_err = _check_io_read_capability()
            if cap_err: return cap_err
            if len(a) != 1 or not isinstance(a[0], String): 
                return EvaluationError("file_list_dir() takes exactly 1 string argument")
            return File.list_directory(a[0])
        
        # Extended File System Operations (with capability checks)
        def _fs_is_file(*a):
            if len(a) != 1 or not isinstance(a[0], String):
                return EvaluationError("fs_is_file() takes exactly 1 string argument")
            import os
            return BooleanObj(os.path.isfile(a[0].value))
        
        def _fs_is_dir(*a):
            if len(a) != 1 or not isinstance(a[0], String):
                return EvaluationError("fs_is_dir() takes exactly 1 string argument")
            import os
            return BooleanObj(os.path.isdir(a[0].value))
        
        def _fs_mkdir(*a):
            cap_err = _check_io_write_capability()
            if cap_err: return cap_err
            if len(a) < 1 or not isinstance(a[0], String):
                return EvaluationError("fs_mkdir() takes path string and optional parents boolean")
            from pathlib import Path
            parents = True  # Default to creating parent directories
            if len(a) >= 2 and isinstance(a[1], BooleanObj):
                parents = a[1].value
            try:
                Path(a[0].value).mkdir(parents=parents, exist_ok=True)
                return BooleanObj(True)
            except Exception as e:
                return EvaluationError(f"fs_mkdir() error: {str(e)}")
        
        def _fs_remove(*a):
            cap_err = _check_io_write_capability()
            if cap_err: return cap_err
            if len(a) != 1 or not isinstance(a[0], String):
                return EvaluationError("fs_remove() takes exactly 1 string argument")
            import os
            try:
                os.remove(a[0].value)
                _invalidate_file_cache(_resolve_read_path(a[0].value))
                return BooleanObj(True)
            except Exception as e:
                return EvaluationError(f"fs_remove() error: {str(e)}")
        
        def _fs_rmdir(*a):
            cap_err = _check_io_write_capability()
            if cap_err: return cap_err
            if len(a) < 1 or not isinstance(a[0], String):
                return EvaluationError("fs_rmdir() takes path string and optional recursive boolean")
            import os
            import shutil
            recursive = False
            if len(a) >= 2 and isinstance(a[1], BooleanObj):
                recursive = a[1].value
            try:
                if recursive:
                    shutil.rmtree(a[0].value)
                else:
                    os.rmdir(a[0].value)
                return BooleanObj(True)
            except Exception as e:
                return EvaluationError(f"fs_rmdir() error: {str(e)}")
        
        def _fs_rename(*a):
            cap_err = _check_io_write_capability()
            if cap_err: return cap_err
            if len(a) != 2 or not all(isinstance(x, String) for x in a):
                return EvaluationError("fs_rename() takes exactly 2 string arguments: old_path, new_path")
            import os
            try:
                os.rename(a[0].value, a[1].value)
                return BooleanObj(True)
            except Exception as e:
                return EvaluationError(f"fs_rename() error: {str(e)}")
        
        def _fs_copy(*a):
            cap_err = _check_io_write_capability()
            if cap_err: return cap_err
            if len(a) != 2 or not all(isinstance(x, String) for x in a):
                return EvaluationError("fs_copy() takes exactly 2 string arguments: src, dst")
            import shutil
            import os
            try:
                src = a[0].value
                dst = a[1].value
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    return EvaluationError(f"fs_copy() source does not exist: {src}")
                return BooleanObj(True)
            except Exception as e:
                return EvaluationError(f"fs_copy() error: {str(e)}")
        
        # Socket/TCP Primitives
        def _socket_create_server(*a):
            """Create TCP server: socket_create_server(host, port, handler, backlog?)"""
            cap_err = _check_network_capability()
            if cap_err: return cap_err
            if len(a) < 3:
                return EvaluationError("socket_create_server() requires at least 3 arguments: host, port, handler")
            
            if not isinstance(a[0], String):
                return EvaluationError("socket_create_server() host must be a string")
            if not isinstance(a[1], Integer):
                return EvaluationError("socket_create_server() port must be an integer")
            if not isinstance(a[2], Action):
                return EvaluationError("socket_create_server() handler must be an action")
            
            host = a[0].value
            port = a[1].value
            handler = a[2]
            backlog = 5
            
            if len(a) >= 4 and isinstance(a[3], Integer):
                backlog = a[3].value
            
            try:
                from ..stdlib.sockets import SocketModule
                
                # Wrap the Zexus handler
                def python_handler(conn):
                    # Create Zexus builtin wrappers for connection methods
                    def _conn_send(*args):
                        if len(args) != 1:
                            return EvaluationError("send() takes 1 argument")
                        data = args[0].value if isinstance(args[0], String) else str(args[0])
                        try:
                            conn.send_string(data)
                            return NULL
                        except Exception as e:
                            return EvaluationError(f"send() error: {str(e)}")
                    
                    def _conn_receive(*args):
                        size = 4096
                        if len(args) >= 1 and isinstance(args[0], Integer):
                            size = args[0].value
                        try:
                            data = conn.receive_string(size)
                            return String(data)
                        except Exception as e:
                            return EvaluationError(f"receive() error: {str(e)}")
                    
                    def _conn_close(*args):
                        try:
                            conn.close()
                            return NULL
                        except Exception as e:
                            return EvaluationError(f"close() error: {str(e)}")
                    
                    # Convert Python connection to Zexus Map
                    conn_obj = Map({
                        String("send"): Builtin(_conn_send, "send"),
                        String("receive"): Builtin(_conn_receive, "receive"),
                        String("close"): Builtin(_conn_close, "close"),
                        String("host"): String(conn.host),
                        String("port"): Integer(conn.port)
                    })
                    
                    # Call Zexus handler
                    self.apply_function(handler, [conn_obj])
                
                server = SocketModule.create_server(host, port, python_handler, backlog)
                server.start()
                
                # Create builtins for server methods
                def _server_stop(*args):
                    try:
                        server.stop()
                        return NULL
                    except Exception as e:
                        return EvaluationError(f"stop() error: {str(e)}")
                
                def _server_is_running(*args):
                    return BooleanObj(server.is_running())
                
                # Return server object as Map
                return Map({
                    String("stop"): Builtin(_server_stop, "stop"),
                    String("is_running"): Builtin(_server_is_running, "is_running"),
                    String("host"): String(server.host),
                    String("port"): Integer(server.port)
                })
            except Exception as e:
                return EvaluationError(f"socket_create_server() error: {str(e)}")
        
        def _socket_create_connection(*a):
            """Create TCP client: socket_create_connection(host, port, timeout?)"""
            cap_err = _check_network_capability()
            if cap_err: return cap_err
            if len(a) < 2:
                return EvaluationError("socket_create_connection() requires at least 2 arguments: host, port")
            
            if not isinstance(a[0], String):
                return EvaluationError("socket_create_connection() host must be a string")
            if not isinstance(a[1], Integer):
                return EvaluationError("socket_create_connection() port must be an integer")
            
            host = a[0].value
            port = a[1].value
            timeout = 5.0
            
            if len(a) >= 3 and isinstance(a[2], (Integer, Float)):
                timeout = float(a[2].value)
            
            try:
                from ..stdlib.sockets import SocketModule
                conn = SocketModule.create_connection(host, port, timeout)
                
                # Create Zexus builtin wrappers for connection methods
                def _conn_send(*args):
                    if len(args) != 1:
                        return EvaluationError("send() takes 1 argument")
                    data = args[0].value if isinstance(args[0], String) else str(args[0])
                    try:
                        conn.send_string(data)
                        return NULL
                    except Exception as e:
                        return EvaluationError(f"send() error: {str(e)}")
                
                def _conn_receive(*args):
                    size = 4096
                    if len(args) >= 1 and isinstance(args[0], Integer):
                        size = args[0].value
                    try:
                        data = conn.receive_string(size)
                        return String(data)
                    except Exception as e:
                        return EvaluationError(f"receive() error: {str(e)}")
                
                def _conn_close(*args):
                    try:
                        conn.close()
                        return NULL
                    except Exception as e:
                        return EvaluationError(f"close() error: {str(e)}")
                
                def _conn_is_connected(*args):
                    return BooleanObj(conn.is_connected())
                
                # Return connection object as Map with Builtin functions
                return Map({
                    String("send"): Builtin(_conn_send, "send"),
                    String("receive"): Builtin(_conn_receive, "receive"),
                    String("close"): Builtin(_conn_close, "close"),
                    String("is_connected"): Builtin(_conn_is_connected, "is_connected"),
                    String("host"): String(conn.host),
                    String("port"): Integer(conn.port)
                })
            except Exception as e:
                return EvaluationError(f"socket_create_connection() error: {str(e)}")
        
        # HTTP Server
        def _http_server(*a):
            """Create HTTP server: http_server(port, host?)"""
            if len(a) < 1:
                return EvaluationError("http_server() requires at least 1 argument: port")
            
            if not isinstance(a[0], Integer):
                return EvaluationError("http_server() port must be an integer")
            
            port = a[0].value
            host = "0.0.0.0"
            
            if len(a) >= 2 and isinstance(a[1], String):
                host = a[1].value
            
            try:
                from ..stdlib.http_server import HTTPServer
                server = HTTPServer(host, port)
                
                # Create builtins for server methods
                def _server_get(*args):
                    if len(args) != 2 or not isinstance(args[0], String) or not isinstance(args[1], Action):
                        return EvaluationError("get() takes 2 arguments: path, handler")
                    
                    path = args[0].value
                    handler = args[1]
                    
                    # Wrap Zexus handler for Python
                    def python_handler(req, res):
                        # Convert request to Zexus Map
                        req_map = Map({
                            String("method"): String(req.method),
                            String("path"): String(req.path),
                            String("headers"): _python_to_zexus(req.headers),
                            String("body"): String(req.body),
                            String("query"): _python_to_zexus(req.query)
                        })
                        
                        # Create response builtins
                        def _res_status(*a):
                            if len(a) != 1 or not isinstance(a[0], Integer):
                                return EvaluationError("status() takes 1 integer argument")
                            res.set_status(a[0].value)
                            return NULL
                        
                        def _res_send(*a):
                            if len(a) != 1:
                                return EvaluationError("send() takes 1 argument")
                            data = a[0].value if isinstance(a[0], String) else str(a[0])
                            res.send(data)
                            return NULL
                        
                        def _res_json(*a):
                            if len(a) != 1:
                                return EvaluationError("json() takes 1 argument")
                            data = _zexus_to_python(a[0])
                            res.json(data)
                            return NULL
                        
                        res_map = Map({
                            String("status"): Builtin(_res_status, "status"),
                            String("send"): Builtin(_res_send, "send"),
                            String("json"): Builtin(_res_json, "json")
                        })
                        
                        # Call Zexus handler
                        self.apply_function(handler, [req_map, res_map])
                    
                    server.get(path, python_handler)
                    return NULL
                
                def _server_post(*args):
                    if len(args) != 2 or not isinstance(args[0], String) or not isinstance(args[1], Action):
                        return EvaluationError("post() takes 2 arguments: path, handler")
                    path = args[0].value
                    handler = args[1]
                    
                    def python_handler(req, res):
                        req_map = Map({
                            String("method"): String(req.method),
                            String("path"): String(req.path),
                            String("headers"): _python_to_zexus(req.headers),
                            String("body"): String(req.body),
                            String("query"): _python_to_zexus(req.query)
                        })
                        
                        def _res_status(*a):
                            if len(a) == 1 and isinstance(a[0], Integer):
                                res.set_status(a[0].value)
                            return NULL
                        
                        def _res_send(*a):
                            if len(a) == 1:
                                data = a[0].value if isinstance(a[0], String) else str(a[0])
                                res.send(data)
                            return NULL
                        
                        def _res_json(*a):
                            if len(a) == 1:
                                data = _zexus_to_python(a[0])
                                res.json(data)
                            return NULL
                        
                        res_map = Map({
                            String("status"): Builtin(_res_status, "status"),
                            String("send"): Builtin(_res_send, "send"),
                            String("json"): Builtin(_res_json, "json")
                        })
                        
                        self.apply_function(handler, [req_map, res_map])
                    
                    server.post(path, python_handler)
                    return NULL
                
                def _server_listen(*args):
                    try:
                        # Start server in background thread
                        import threading
                        thread = threading.Thread(target=server.listen, daemon=True)
                        thread.start()
                        return NULL
                    except Exception as e:
                        return EvaluationError(f"listen() error: {str(e)}")
                
                def _server_stop(*args):
                    try:
                        server.stop()
                        return NULL
                    except Exception as e:
                        return EvaluationError(f"stop() error: {str(e)}")
                
                # Return server object as Map
                return Map({
                    String("get"): Builtin(_server_get, "get"),
                    String("post"): Builtin(_server_post, "post"),
                    String("listen"): Builtin(_server_listen, "listen"),
                    String("stop"): Builtin(_server_stop, "stop"),
                    String("host"): String(host),
                    String("port"): Integer(port)
                })
            except Exception as e:
                return EvaluationError(f"http_server() error: {str(e)}")
        
        # Database - SQLite
        def _sqlite_connect(*a):
            """Connect to SQLite database: sqlite_connect(database_path)"""
            if len(a) != 1 or not isinstance(a[0], String):
                return EvaluationError("sqlite_connect() takes 1 string argument: database path")
            
            try:
                from ..stdlib.db_sqlite import SQLiteConnection
                db = SQLiteConnection(a[0].value)
                
                if not db.connect():
                    return EvaluationError("Failed to connect to SQLite database")
                
                # Store db in a list to keep it alive (prevent garbage collection)
                db_ref = [db]
                
                # Create database methods as Zexus builtins
                def _db_execute(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("execute() takes query string and optional params")
                    
                    query = args[0].value
                    params = None
                    
                    if len(args) >= 2:
                        # Convert Zexus List to Python tuple for params
                        if isinstance(args[1], List):
                            params = tuple(_zexus_to_python(args[1]))
                        else:
                            params = (_zexus_to_python(args[1]),)
                    
                    result = db_ref[0].execute(query, params)
                    return BooleanObj(result)
                
                def _db_query(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("query() takes query string and optional params")
                    
                    query = args[0].value
                    params = None
                    
                    if len(args) >= 2:
                        if isinstance(args[1], List):
                            params = tuple(_zexus_to_python(args[1]))
                        else:
                            params = (_zexus_to_python(args[1]),)
                    
                    results = db_ref[0].query(query, params)
                    # Convert Python list of dicts to Zexus List of Maps
                    return _python_to_zexus(results)
                
                def _db_query_one(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("query_one() takes query string and optional params")
                    
                    query = args[0].value
                    params = None
                    
                    if len(args) >= 2:
                        if isinstance(args[1], List):
                            params = tuple(_zexus_to_python(args[1]))
                        else:
                            params = (_zexus_to_python(args[1]),)
                    
                    result = db_ref[0].query_one(query, params)
                    return _python_to_zexus(result) if result else NULL
                
                def _db_last_insert_id(*args):
                    return Integer(db_ref[0].last_insert_id())
                
                def _db_affected_rows(*args):
                    return Integer(db_ref[0].affected_rows())
                
                def _db_begin(*args):
                    return BooleanObj(db_ref[0].begin_transaction())
                
                def _db_commit(*args):
                    return BooleanObj(db_ref[0].commit())
                
                def _db_rollback(*args):
                    return BooleanObj(db_ref[0].rollback())
                
                def _db_close(*args):
                    return BooleanObj(db_ref[0].close())
                
                # Return database connection as Map
                return Map({
                    String("execute"): Builtin(_db_execute, "execute"),
                    String("query"): Builtin(_db_query, "query"),
                    String("query_one"): Builtin(_db_query_one, "query_one"),
                    String("last_insert_id"): Builtin(_db_last_insert_id, "last_insert_id"),
                    String("affected_rows"): Builtin(_db_affected_rows, "affected_rows"),
                    String("begin"): Builtin(_db_begin, "begin"),
                    String("commit"): Builtin(_db_commit, "commit"),
                    String("rollback"): Builtin(_db_rollback, "rollback"),
                    String("close"): Builtin(_db_close, "close"),
                    String("database"): String(a[0].value)
                })
            
            except Exception as e:
                return EvaluationError(f"sqlite_connect() error: {str(e)}")
        
        # Database - PostgreSQL
        def _postgres_connect(*a):
            """Connect to PostgreSQL: postgres_connect(host, port, database, user, password)"""
            if len(a) < 1:
                return EvaluationError("postgres_connect() requires at least database name")
            
            # Parse parameters
            host = "localhost"
            port = 5432
            database = a[0].value if isinstance(a[0], String) else "postgres"
            user = "postgres"
            password = ""
            
            if len(a) >= 2 and isinstance(a[1], String):
                user = a[1].value
            if len(a) >= 3 and isinstance(a[2], String):
                password = a[2].value
            if len(a) >= 4 and isinstance(a[3], String):
                host = a[3].value
            if len(a) >= 5 and isinstance(a[4], Integer):
                port = a[4].value
            
            try:
                from ..stdlib.db_postgres import PostgreSQLConnection
                db = PostgreSQLConnection(host, port, database, user, password)
                
                if not db.connect():
                    return EvaluationError("Failed to connect to PostgreSQL database")
                
                db_ref = [db]
                
                # Same methods as SQLite
                def _db_execute(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("execute() takes query string and optional params")
                    query = args[0].value
                    params = None
                    if len(args) >= 2:
                        if isinstance(args[1], List):
                            params = tuple(_zexus_to_python(args[1]))
                        else:
                            params = (_zexus_to_python(args[1]),)
                    result = db_ref[0].execute(query, params)
                    return BooleanObj(result)
                
                def _db_query(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("query() takes query string and optional params")
                    query = args[0].value
                    params = None
                    if len(args) >= 2:
                        if isinstance(args[1], List):
                            params = tuple(_zexus_to_python(args[1]))
                        else:
                            params = (_zexus_to_python(args[1]),)
                    results = db_ref[0].query(query, params)
                    return _python_to_zexus(results)
                
                def _db_query_one(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("query_one() takes query string and optional params")
                    query = args[0].value
                    params = None
                    if len(args) >= 2:
                        if isinstance(args[1], List):
                            params = tuple(_zexus_to_python(args[1]))
                        else:
                            params = (_zexus_to_python(args[1]),)
                    result = db_ref[0].query_one(query, params)
                    return _python_to_zexus(result) if result else NULL
                
                def _db_last_insert_id(*args):
                    return Integer(db_ref[0].last_insert_id())
                
                def _db_affected_rows(*args):
                    return Integer(db_ref[0].affected_rows())
                
                def _db_begin(*args):
                    return BooleanObj(db_ref[0].begin_transaction())
                
                def _db_commit(*args):
                    return BooleanObj(db_ref[0].commit())
                
                def _db_rollback(*args):
                    return BooleanObj(db_ref[0].rollback())
                
                def _db_close(*args):
                    return BooleanObj(db_ref[0].close())
                
                return Map({
                    String("execute"): Builtin(_db_execute, "execute"),
                    String("query"): Builtin(_db_query, "query"),
                    String("query_one"): Builtin(_db_query_one, "query_one"),
                    String("last_insert_id"): Builtin(_db_last_insert_id, "last_insert_id"),
                    String("affected_rows"): Builtin(_db_affected_rows, "affected_rows"),
                    String("begin"): Builtin(_db_begin, "begin"),
                    String("commit"): Builtin(_db_commit, "commit"),
                    String("rollback"): Builtin(_db_rollback, "rollback"),
                    String("close"): Builtin(_db_close, "close"),
                    String("database"): String(database),
                    String("type"): String("postgresql")
                })
            
            except Exception as e:
                return EvaluationError(f"postgres_connect() error: {str(e)}")
        
        # Database - MySQL
        def _mysql_connect(*a):
            """Connect to MySQL: mysql_connect(database, user, password, host?, port?)"""
            if len(a) < 1:
                return EvaluationError("mysql_connect() requires at least database name")
            
            host = "localhost"
            port = 3306
            database = a[0].value if isinstance(a[0], String) else "mysql"
            user = "root"
            password = ""
            
            if len(a) >= 2 and isinstance(a[1], String):
                user = a[1].value
            if len(a) >= 3 and isinstance(a[2], String):
                password = a[2].value
            if len(a) >= 4 and isinstance(a[3], String):
                host = a[3].value
            if len(a) >= 5 and isinstance(a[4], Integer):
                port = a[4].value
            
            try:
                from ..stdlib.db_mysql import MySQLConnection
                db = MySQLConnection(host, port, database, user, password)
                
                if not db.connect():
                    return EvaluationError("Failed to connect to MySQL database")
                
                db_ref = [db]
                
                # Same interface as SQLite/PostgreSQL
                def _db_execute(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("execute() takes query string and optional params")
                    query = args[0].value
                    params = None
                    if len(args) >= 2:
                        if isinstance(args[1], List):
                            params = tuple(_zexus_to_python(args[1]))
                        else:
                            params = (_zexus_to_python(args[1]),)
                    result = db_ref[0].execute(query, params)
                    return BooleanObj(result)
                
                def _db_query(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("query() takes query string and optional params")
                    query = args[0].value
                    params = None
                    if len(args) >= 2:
                        if isinstance(args[1], List):
                            params = tuple(_zexus_to_python(args[1]))
                        else:
                            params = (_zexus_to_python(args[1]),)
                    results = db_ref[0].query(query, params)
                    return _python_to_zexus(results)
                
                def _db_query_one(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("query_one() takes query string and optional params")
                    query = args[0].value
                    params = None
                    if len(args) >= 2:
                        if isinstance(args[1], List):
                            params = tuple(_zexus_to_python(args[1]))
                        else:
                            params = (_zexus_to_python(args[1]),)
                    result = db_ref[0].query_one(query, params)
                    return _python_to_zexus(result) if result else NULL
                
                def _db_last_insert_id(*args):
                    return Integer(db_ref[0].last_insert_id())
                
                def _db_affected_rows(*args):
                    return Integer(db_ref[0].affected_rows())
                
                def _db_begin(*args):
                    return BooleanObj(db_ref[0].begin_transaction())
                
                def _db_commit(*args):
                    return BooleanObj(db_ref[0].commit())
                
                def _db_rollback(*args):
                    return BooleanObj(db_ref[0].rollback())
                
                def _db_close(*args):
                    return BooleanObj(db_ref[0].close())
                
                return Map({
                    String("execute"): Builtin(_db_execute, "execute"),
                    String("query"): Builtin(_db_query, "query"),
                    String("query_one"): Builtin(_db_query_one, "query_one"),
                    String("last_insert_id"): Builtin(_db_last_insert_id, "last_insert_id"),
                    String("affected_rows"): Builtin(_db_affected_rows, "affected_rows"),
                    String("begin"): Builtin(_db_begin, "begin"),
                    String("commit"): Builtin(_db_commit, "commit"),
                    String("rollback"): Builtin(_db_rollback, "rollback"),
                    String("close"): Builtin(_db_close, "close"),
                    String("database"): String(database),
                    String("type"): String("mysql")
                })
            
            except Exception as e:
                return EvaluationError(f"mysql_connect() error: {str(e)}")
        
        # Database - MongoDB
        def _mongo_connect(*a):
            """Connect to MongoDB: mongo_connect(database, host?, port?, username?, password?)"""
            if len(a) < 1:
                return EvaluationError("mongo_connect() requires at least database name")
            
            database = a[0].value if isinstance(a[0], String) else "test"
            host = "localhost"
            port = 27017
            username = None
            password = None
            
            if len(a) >= 2 and isinstance(a[1], String):
                host = a[1].value
            if len(a) >= 3 and isinstance(a[2], Integer):
                port = a[2].value
            if len(a) >= 4 and isinstance(a[3], String):
                username = a[3].value
            if len(a) >= 5 and isinstance(a[4], String):
                password = a[4].value
            
            try:
                from ..stdlib.db_mongo import MongoDBConnection
                db = MongoDBConnection(host, port, database, username, password)
                
                if not db.connect():
                    return EvaluationError("Failed to connect to MongoDB database")
                
                db_ref = [db]
                
                # MongoDB-specific operations
                def _db_insert_one(*args):
                    if len(args) < 2 or not isinstance(args[0], String) or not isinstance(args[1], Map):
                        return EvaluationError("insert_one() takes collection name and document")
                    collection = args[0].value
                    document = _zexus_to_python(args[1])
                    result = db_ref[0].insert_one(collection, document)
                    return String(result) if result else NULL
                
                def _db_insert_many(*args):
                    if len(args) < 2 or not isinstance(args[0], String) or not isinstance(args[1], List):
                        return EvaluationError("insert_many() takes collection name and list of documents")
                    collection = args[0].value
                    documents = _zexus_to_python(args[1])
                    result = db_ref[0].insert_many(collection, documents)
                    return _python_to_zexus(result) if result else NULL
                
                def _db_find(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("find() takes collection name and optional query")
                    collection = args[0].value
                    query = None
                    if len(args) >= 2 and isinstance(args[1], Map):
                        query = _zexus_to_python(args[1])
                    results = db_ref[0].find(collection, query)
                    return _python_to_zexus(results)
                
                def _db_find_one(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("find_one() takes collection name and optional query")
                    collection = args[0].value
                    query = None
                    if len(args) >= 2 and isinstance(args[1], Map):
                        query = _zexus_to_python(args[1])
                    result = db_ref[0].find_one(collection, query)
                    return _python_to_zexus(result) if result else NULL
                
                def _db_update_one(*args):
                    if len(args) < 3 or not isinstance(args[0], String) or not isinstance(args[1], Map) or not isinstance(args[2], Map):
                        return EvaluationError("update_one() takes collection, query, and update")
                    collection = args[0].value
                    query = _zexus_to_python(args[1])
                    update = _zexus_to_python(args[2])
                    result = db_ref[0].update_one(collection, query, update)
                    return Integer(result)
                
                def _db_update_many(*args):
                    if len(args) < 3 or not isinstance(args[0], String) or not isinstance(args[1], Map) or not isinstance(args[2], Map):
                        return EvaluationError("update_many() takes collection, query, and update")
                    collection = args[0].value
                    query = _zexus_to_python(args[1])
                    update = _zexus_to_python(args[2])
                    result = db_ref[0].update_many(collection, query, update)
                    return Integer(result)
                
                def _db_delete_one(*args):
                    if len(args) < 2 or not isinstance(args[0], String) or not isinstance(args[1], Map):
                        return EvaluationError("delete_one() takes collection and query")
                    collection = args[0].value
                    query = _zexus_to_python(args[1])
                    result = db_ref[0].delete_one(collection, query)
                    return Integer(result)
                
                def _db_delete_many(*args):
                    if len(args) < 2 or not isinstance(args[0], String) or not isinstance(args[1], Map):
                        return EvaluationError("delete_many() takes collection and query")
                    collection = args[0].value
                    query = _zexus_to_python(args[1])
                    result = db_ref[0].delete_many(collection, query)
                    return Integer(result)
                
                def _db_count(*args):
                    if len(args) < 1 or not isinstance(args[0], String):
                        return EvaluationError("count() takes collection name and optional query")
                    collection = args[0].value
                    query = None
                    if len(args) >= 2 and isinstance(args[1], Map):
                        query = _zexus_to_python(args[1])
                    result = db_ref[0].count(collection, query)
                    return Integer(result)
                
                def _db_close(*args):
                    return BooleanObj(db_ref[0].close())
                
                return Map({
                    String("insert_one"): Builtin(_db_insert_one, "insert_one"),
                    String("insert_many"): Builtin(_db_insert_many, "insert_many"),
                    String("find"): Builtin(_db_find, "find"),
                    String("find_one"): Builtin(_db_find_one, "find_one"),
                    String("update_one"): Builtin(_db_update_one, "update_one"),
                    String("update_many"): Builtin(_db_update_many, "update_many"),
                    String("delete_one"): Builtin(_db_delete_one, "delete_one"),
                    String("delete_many"): Builtin(_db_delete_many, "delete_many"),
                    String("count"): Builtin(_db_count, "count"),
                    String("close"): Builtin(_db_close, "close"),
                    String("database"): String(database),
                    String("type"): String("mongodb")
                })
            
            except Exception as e:
                return EvaluationError(f"mongo_connect() error: {str(e)}")
        
        # Capability-checked helpers
        def _check_network_capability():
            """Check if network operations are allowed in current context."""
            try:
                from .integration import CapabilityChecker
                if not CapabilityChecker.check_network():
                    return EvaluationError("Network access denied: missing 'network.tcp' capability")
            except Exception:
                pass  # If capability system isn't loaded, allow by default
            return None

        def _check_io_read_capability():
            """Check if file read operations are allowed."""
            try:
                from .integration import CapabilityChecker
                if not CapabilityChecker.check_io_read():
                    return EvaluationError("File read access denied: missing 'io.read' capability")
            except Exception:
                pass
            return None

        def _check_io_write_capability():
            """Check if file write operations are allowed."""
            try:
                from .integration import CapabilityChecker
                if not CapabilityChecker.check_io_write():
                    return EvaluationError("File write access denied: missing 'io.write' capability")
            except Exception:
                pass
            return None

        # HTTP Client
        def _http_get(*a):
            """HTTP GET request: http_get(url, headers?, timeout?)"""
            cap_err = _check_network_capability()
            if cap_err: return cap_err
            if len(a) < 1:
                return EvaluationError("http_get() requires at least 1 argument: url")
            
            url = a[0].value if isinstance(a[0], String) else str(a[0])
            headers = None
            timeout = 30
            
            # Parse optional headers (Map)
            if len(a) >= 2 and isinstance(a[1], Map):
                headers = _zexus_to_python(a[1])
            
            # Parse optional timeout (Integer)
            if len(a) >= 3 and isinstance(a[2], Integer):
                timeout = a[2].value
            
            try:
                from ..stdlib.http import HttpModule
                result = HttpModule.get(url, headers, timeout)
                # HTTP responses are external data - mark as untrusted
                return _python_to_zexus(result, mark_untrusted=True)
            except Exception as e:
                return EvaluationError(f"HTTP GET error: {str(e)}")
        
        def _http_post(*a):
            """HTTP POST request: http_post(url, data, headers?, timeout?)"""
            cap_err = _check_network_capability()
            if cap_err: return cap_err
            if len(a) < 2:
                return EvaluationError("http_post() requires at least 2 arguments: url, data")
            
            url = a[0].value if isinstance(a[0], String) else str(a[0])
            data = _zexus_to_python(a[1])
            headers = None
            timeout = 30
            
            # Parse optional headers (Map)
            if len(a) >= 3 and isinstance(a[2], Map):
                headers = _zexus_to_python(a[2])
            
            # Parse optional timeout (Integer)
            if len(a) >= 4 and isinstance(a[3], Integer):
                timeout = a[3].value
            
            try:
                from ..stdlib.http import HttpModule
                # Determine if data should be sent as JSON
                json_mode = isinstance(a[1], (Map, List))
                result = HttpModule.post(url, data, headers, json=json_mode, timeout=timeout)
                # HTTP responses are external data - mark as untrusted
                return _python_to_zexus(result, mark_untrusted=True)
            except Exception as e:
                return EvaluationError(f"HTTP POST error: {str(e)}")
        
        def _http_put(*a):
            """HTTP PUT request: http_put(url, data, headers?, timeout?)"""
            cap_err = _check_network_capability()
            if cap_err: return cap_err
            if len(a) < 2:
                return EvaluationError("http_put() requires at least 2 arguments: url, data")
            
            url = a[0].value if isinstance(a[0], String) else str(a[0])
            data = _zexus_to_python(a[1])
            headers = None
            timeout = 30
            
            if len(a) >= 3 and isinstance(a[2], Map):
                headers = _zexus_to_python(a[2])
            
            if len(a) >= 4 and isinstance(a[3], Integer):
                timeout = a[3].value
            
            try:
                from ..stdlib.http import HttpModule
                json_mode = isinstance(a[1], (Map, List))
                result = HttpModule.put(url, data, headers, json=json_mode, timeout=timeout)
                # HTTP responses are external data - mark as untrusted
                return _python_to_zexus(result, mark_untrusted=True)
            except Exception as e:
                return EvaluationError(f"HTTP PUT error: {str(e)}")
        
        def _http_delete(*a):
            """HTTP DELETE request: http_delete(url, headers?, timeout?)"""
            cap_err = _check_network_capability()
            if cap_err: return cap_err
            if len(a) < 1:
                return EvaluationError("http_delete() requires at least 1 argument: url")
            
            url = a[0].value if isinstance(a[0], String) else str(a[0])
            headers = None
            timeout = 30
            
            if len(a) >= 2 and isinstance(a[1], Map):
                headers = _zexus_to_python(a[1])
            
            if len(a) >= 3 and isinstance(a[2], Integer):
                timeout = a[2].value
            
            try:
                from ..stdlib.http import HttpModule
                result = HttpModule.delete(url, headers, timeout)
                # HTTP responses are external data - mark as untrusted
                return _python_to_zexus(result, mark_untrusted=True)
            except Exception as e:
                return EvaluationError(f"HTTP DELETE error: {str(e)}")

        def _http_request(*a):
            """Generic HTTP request: http_request(method, url, data?, headers?, timeout?)"""
            cap_err = _check_network_capability()
            if cap_err: return cap_err
            if len(a) < 2:
                return EvaluationError("http_request() requires at least 2 arguments: method, url")
            method_str = a[0].value if isinstance(a[0], String) else str(a[0])
            url = a[1].value if isinstance(a[1], String) else str(a[1])
            data = _zexus_to_python(a[2]) if len(a) >= 3 and a[2] != NULL else None
            headers = _zexus_to_python(a[3]) if len(a) >= 4 and isinstance(a[3], Map) else None
            timeout = a[4].value if len(a) >= 5 and isinstance(a[4], Integer) else 30
            try:
                from ..stdlib.http import HttpModule
                result = HttpModule.request(method_str, url, data, headers, timeout)
                return _python_to_zexus(result, mark_untrusted=True)
            except Exception as e:
                return EvaluationError(f"HTTP {method_str} error: {str(e)}")

        def _http_parallel_get(*a):
            """Parallel GET requests: http_parallel_get([url1, url2, ...], headers?, timeout?)"""
            cap_err = _check_network_capability()
            if cap_err: return cap_err
            if len(a) < 1 or not isinstance(a[0], List):
                return EvaluationError("http_parallel_get() requires a list of URLs as first argument")
            urls = [e.value if isinstance(e, String) else str(e) for e in a[0].elements]
            headers = _zexus_to_python(a[1]) if len(a) >= 2 and isinstance(a[1], Map) else None
            timeout = a[2].value if len(a) >= 3 and isinstance(a[2], Integer) else 30
            try:
                from ..stdlib.http import HttpModule
                results = HttpModule.parallel_get(urls, headers, timeout)
                return List([_python_to_zexus(r, mark_untrusted=True) for r in results])
            except Exception as e:
                return EvaluationError(f"http_parallel_get() error: {str(e)}")

        def _http_async_get(*a):
            """Non-blocking HTTP GET: http_async_get(url, headers?, timeout?) -> Future"""
            cap_err = _check_network_capability()
            if cap_err: return cap_err
            if len(a) < 1:
                return EvaluationError("http_async_get() requires at least 1 argument: url")
            url = a[0].value if isinstance(a[0], String) else str(a[0])
            headers = _zexus_to_python(a[1]) if len(a) >= 2 and isinstance(a[1], Map) else None
            timeout = a[2].value if len(a) >= 3 and isinstance(a[2], Integer) else 30
            try:
                from ..stdlib.http import HttpModule
                future = HttpModule.async_get(url, headers, timeout)
                # Wrap future so await_result() works from Zexus
                from ..object import Map as ZMap
                return ZMap({
                    "type": String("HttpFuture"),
                    "done": Builtin(lambda *_: BooleanObj(future.done()), "done"),
                    "result": Builtin(lambda *_: _python_to_zexus(future.result(), mark_untrusted=True), "result"),
                })
            except Exception as e:
                return EvaluationError(f"http_async_get() error: {str(e)}")
        
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
            # Output the debug information
            print(output, flush=True)
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
            from ..object import EntityInstance
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
            if isinstance(arg, EntityInstance):
                return String(arg.inspect())
            if arg == NULL: 
                return String("null")
            # For any object with an inspect method
            if hasattr(arg, 'inspect') and callable(arg.inspect):
                return String(arg.inspect())
            return String(str(arg))
        
        def _int(*a):
            """Convert value to integer"""
            if len(a) != 1:
                return EvaluationError(f"int() takes 1 arg ({len(a)} given)")
            arg = a[0]
            if isinstance(arg, Integer):
                return arg
            if isinstance(arg, Float):
                return Integer(int(arg.value))
            if isinstance(arg, String):
                try:
                    return Integer(int(arg.value))
                except ValueError:
                    return EvaluationError(f"Cannot convert '{arg.value}' to integer")
            if isinstance(arg, BooleanObj):
                return Integer(1 if arg.value else 0)
            return EvaluationError(f"Cannot convert {type(arg).__name__} to integer")
        
        def _float(*a):
            """Convert value to float"""
            if len(a) != 1:
                return EvaluationError(f"float() takes 1 arg ({len(a)} given)")
            arg = a[0]
            if isinstance(arg, Float):
                return arg
            if isinstance(arg, Integer):
                return Float(float(arg.value))
            if isinstance(arg, String):
                try:
                    return Float(float(arg.value))
                except ValueError:
                    return EvaluationError(f"Cannot convert '{arg.value}' to float")
            if isinstance(arg, BooleanObj):
                return Float(1.0 if arg.value else 0.0)
            return EvaluationError(f"Cannot convert {type(arg).__name__} to float")
        
        def _uppercase(*a):
            """Convert string to uppercase"""
            if len(a) != 1:
                return EvaluationError(f"uppercase() takes 1 arg ({len(a)} given)")
            arg = a[0]
            if isinstance(arg, String):
                return String(arg.value.upper())
            return EvaluationError(f"uppercase() requires a string argument")
        
        def _lowercase(*a):
            """Convert string to lowercase"""
            if len(a) != 1:
                return EvaluationError(f"lowercase() takes 1 arg ({len(a)} given)")
            arg = a[0]
            if isinstance(arg, String):
                return String(arg.value.lower())
            return EvaluationError(f"lowercase() requires a string argument")

        def _split(*a):
            """Split string by delimiter"""
            if len(a) != 2:
                return EvaluationError(f"split() takes 2 args: string, delimiter")
            
            str_obj = a[0]
            delim_obj = a[1]
            
            if not isinstance(str_obj, String):
                return EvaluationError("split() first argument must be a string")
            if not isinstance(delim_obj, String):
                return EvaluationError("split() second argument must be a delimiter string")
                
            parts = str_obj.value.split(delim_obj.value)
            return List([String(p) for p in parts])
        
        def _random(*a):
            """Generate random number. random() -> 0-1, random(max) -> 0 to max-1"""
            import random
            if len(a) == 0:
                return Float(random.random())
            elif len(a) == 1:
                if isinstance(a[0], Integer):
                    return Integer(random.randint(0, a[0].value - 1))
                elif isinstance(a[0], Float):
                    return Float(random.random() * a[0].value)
                return EvaluationError("random() argument must be a number")
            else:
                return EvaluationError(f"random() takes 0 or 1 arg ({len(a)} given)")
        
        def _persist_set(*a):
            """Store a value in persistent storage: persist_set(key, value)"""
            if len(a) != 2:
                return EvaluationError(f"persist_set() takes 2 args (key, value), got {len(a)}")
            if not isinstance(a[0], String):
                return EvaluationError("persist_set() key must be a string")
            
            import json
            import os
            
            key = a[0].value
            value = a[1]
            
            # Create persistence directory if it doesn't exist
            persist_dir = os.path.join(os.getcwd(), '.zexus_persist')
            os.makedirs(persist_dir, exist_ok=True)
            
            # Convert value to JSON-serializable format
            json_value = _to_python_value(value)
            
            # Save to file
            persist_file = os.path.join(persist_dir, f'{key}.json')
            try:
                with open(persist_file, 'w') as f:
                    json.dump(json_value, f)
                return TRUE
            except Exception as e:
                return EvaluationError(f"persist_set() error: {str(e)}")
        
        def _persist_get(*a):
            """Retrieve a value from persistent storage: persist_get(key)"""
            if len(a) != 1:
                return EvaluationError(f"persist_get() takes 1 arg (key), got {len(a)}")
            if not isinstance(a[0], String):
                return EvaluationError("persist_get() key must be a string")
            
            import json
            import os
            
            key = a[0].value
            persist_dir = os.path.join(os.getcwd(), '.zexus_persist')
            persist_file = os.path.join(persist_dir, f'{key}.json')
            
            if not os.path.exists(persist_file):
                return NULL
            
            try:
                with open(persist_file, 'r') as f:
                    json_value = json.load(f)
                
                # Convert back to Zexus object
                return _from_python_value(json_value)
            except Exception as e:
                return EvaluationError(f"persist_get() error: {str(e)}")
        
        def _to_python_value(obj):
            """Helper to convert Zexus object to Python value"""
            from ..security import EntityInstance as SecurityEntityInstance
            
            if isinstance(obj, String):
                return obj.value
            elif isinstance(obj, (Integer, Float)):
                return obj.value
            elif isinstance(obj, BooleanObj):
                return obj.value
            elif isinstance(obj, List):
                return [_to_python_value(v) for v in obj.elements]
            elif isinstance(obj, Map):
                return {str(k): _to_python_value(v) for k, v in obj.pairs.items()}
            elif isinstance(obj, SecurityEntityInstance):
                # Convert entity to a dict with its properties
                result = {}
                for key, value in obj.data.items():
                    result[key] = _to_python_value(value)
                return result
            elif obj == NULL:
                return None
            else:
                return str(obj)
        
        def _from_python_value(val):
            """Helper to convert Python value to Zexus object"""
            if isinstance(val, bool):
                return BooleanObj(val)
            elif isinstance(val, int):
                return Integer(val)
            elif isinstance(val, float):
                return Float(val)
            elif isinstance(val, str):
                return String(val)
            elif isinstance(val, list):
                return List([_from_python_value(v) for v in val])
            elif isinstance(val, dict):
                # Convert dict to Map with String keys
                pairs = {}
                for k, v in val.items():
                    key = String(str(k)) if not isinstance(k, str) else String(k)
                    pairs[key] = _from_python_value(v)
                return Map(pairs)
            elif val is None:
                return NULL
            else:
                return String(str(val))
        
        def _len(*a):
            if len(a) != 1: 
                return EvaluationError("len() takes 1 arg")
            arg = a[0]
            if isinstance(arg, String): 
                return Integer(len(arg.value))
            if isinstance(arg, List): 
                return Integer(len(arg.elements))
            if isinstance(arg, Map):
                return Integer(len(arg.pairs))
            # Handle Python list (shouldn't happen, but defensive)
            if isinstance(arg, list):
                return Integer(len(arg))
            arg_type = arg.type() if hasattr(arg, 'type') else type(arg).__name__
            return EvaluationError(f"len() not supported for {arg_type}")
        
        def _type(*a):
            """Return the type name of the argument"""
            if len(a) != 1:
                return EvaluationError("type() takes exactly 1 argument")
            arg = a[0]
            if isinstance(arg, Integer):
                return String("Integer")
            elif isinstance(arg, Float):
                return String("Float")
            elif isinstance(arg, String):
                return String("String")
            elif isinstance(arg, BooleanObj):
                return String("Boolean")
            elif isinstance(arg, List):
                return String("List")
            elif isinstance(arg, Map):
                return String("Map")
            elif isinstance(arg, Action):
                return String("Action")
            elif isinstance(arg, LambdaFunction):
                return String("Lambda")
            elif isinstance(arg, Builtin):
                return String("Builtin")
            elif isinstance(arg, Null):
                return String("Null")
            else:
                return String(type(arg).__name__)
        
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
        
        def _append(*a):
            """Mutating append: modifies list in-place and returns it"""
            if len(a) != 2: 
                return EvaluationError("append() takes 2 arguments: append(list, item)")
            if not isinstance(a[0], List): 
                return EvaluationError("append() first argument must be a list")
            # Mutate the list in-place
            a[0].append(a[1])
            return a[0]
        
        def _extend(*a):
            """Mutating extend: modifies list in-place by adding elements from another list"""
            if len(a) != 2: 
                return EvaluationError("extend() takes 2 arguments: extend(list, other_list)")
            if not isinstance(a[0], List): 
                return EvaluationError("extend() first argument must be a list")
            if not isinstance(a[1], List): 
                return EvaluationError("extend() second argument must be a list")
            # Mutate the list in-place
            a[0].extend(a[1])
            return a[0]
        
        def _sort(*a):
            """Sort a list: sort(list) or sort(list, key_field) for maps
            
            - sort([3, 1, 2]) -> [1, 2, 3]
            - sort([{fee: 5}, {fee: 3}], "fee") -> [{fee: 3}, {fee: 5}]
            - sort(list, "fee", true) -> descending order
            """
            if len(a) < 1:
                return EvaluationError("sort() takes 1-3 arguments: sort(list, [key], [descending])")
            if not isinstance(a[0], List):
                return EvaluationError("sort() first argument must be a list")
            
            lst = a[0]
            key_field = None
            descending = False
            
            if len(a) >= 2:
                if isinstance(a[1], String):
                    key_field = a[1].value
                elif isinstance(a[1], BooleanObj):
                    descending = a[1].value
                else:
                    return EvaluationError("sort() second argument must be a key string or boolean for descending")
            
            if len(a) >= 3:
                if isinstance(a[2], BooleanObj):
                    descending = a[2].value
                else:
                    return EvaluationError("sort() third argument must be a boolean for descending order")
            
            try:
                # Make a copy of elements for non-destructive sort
                elements = list(lst.elements)
                
                def get_sort_key(elem):
                    if key_field is not None:
                        # Sort by field in map
                        if isinstance(elem, Map):
                            # Try string key first
                            val = elem.pairs.get(String(key_field))
                            if val is None:
                                # Try with key_field as-is (for internal dict keys)
                                val = elem.pairs.get(key_field)
                            if val is None:
                                return 0  # Default for missing key
                            if isinstance(val, (Integer, Float)):
                                return val.value
                            if isinstance(val, String):
                                return val.value
                            return 0
                        elif isinstance(elem, dict):
                            val = elem.get(key_field, 0)
                            if hasattr(val, 'value'):
                                return val.value
                            return val
                        return 0
                    else:
                        # Direct sort
                        if isinstance(elem, (Integer, Float)):
                            return elem.value
                        if isinstance(elem, String):
                            return elem.value
                        return 0
                
                elements.sort(key=get_sort_key, reverse=descending)
                return List(elements)
            except Exception as e:
                return EvaluationError(f"sort() error: {str(e)}")
        
        def _slice(*a):
            """Get a slice of a list: slice(list, start, end?) -> list
            
            - slice([1, 2, 3, 4], 1) -> [2, 3, 4]  (from index 1 to end)
            - slice([1, 2, 3, 4], 1, 3) -> [2, 3]  (from index 1 to 3, exclusive)
            - slice([1, 2, 3, 4], 0, 2) -> [1, 2]  (first 2 elements)
            """
            if len(a) < 2:
                return EvaluationError("slice() takes 2-3 arguments: slice(list, start, [end])")
            if not isinstance(a[0], List):
                return EvaluationError("slice() first argument must be a list")
            if not isinstance(a[1], Integer):
                return EvaluationError("slice() start must be an integer")
            
            lst = a[0]
            start = a[1].value
            end = len(lst.elements)  # Default: to the end
            
            if len(a) >= 3:
                if not isinstance(a[2], Integer):
                    return EvaluationError("slice() end must be an integer")
                end = a[2].value
            
            # Handle negative indices
            if start < 0:
                start = max(0, len(lst.elements) + start)
            if end < 0:
                end = max(0, len(lst.elements) + end)
            
            return List(lst.elements[start:end])
        
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
        
        def _vfs_stats(*a):
            """Return VFS file cache statistics as a Map: vfs_stats()"""
            mgr = _get_vfs_manager()
            if mgr is None:
                return Map({"error": String("VFS not initialized")})
            stats = mgr.file_cache.stats()
            return Map({
                "entries": Integer(stats["entries"]),
                "bytes": Integer(stats["bytes"]),
                "hits": Integer(stats["hits"]),
                "misses": Integer(stats["misses"]),
                "hit_rate": Float(round(stats["hit_rate"], 4)),
            })

        def _vfs_clear_cache(*a):
            """Flush the VFS file content cache: vfs_clear_cache()"""
            mgr = _get_vfs_manager()
            if mgr is None:
                return NULL
            mgr.file_cache.clear()
            return NULL
        
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
        
        def _read_file(*a):
            """Read entire file contents as string (VFS-cached)"""
            cap_err = _check_io_read_capability()
            if cap_err: return cap_err
            if len(a) != 1:
                return EvaluationError("read_file() takes exactly 1 argument: path")
            if not isinstance(a[0], String):
                return EvaluationError("read_file() path must be a string")
            
            path = _resolve_read_path(a[0].value)
            
            try:
                content = _cached_read(path)
                return String(content)
            except FileNotFoundError:
                return EvaluationError(f"File not found: {path}")
            except Exception as e:
                return EvaluationError(f"read_file() error: {str(e)}")
        
        def _eval_file(*a):
            """Execute code from a file based on its extension"""
            if len(a) < 1 or len(a) > 2:
                return EvaluationError("eval_file() takes 1-2 arguments: eval_file(path) or eval_file(path, language)")
            if not isinstance(a[0], String):
                return EvaluationError("eval_file() path must be a string")
            
            import os
            import subprocess
            path = a[0].value
            
            import os
            import subprocess
            path = a[0].value
            
            # Normalize path
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            
            # Determine language from extension or argument
            if len(a) == 2 and isinstance(a[1], String):
                language = a[1].value.lower()
            else:
                _, ext = os.path.splitext(path)
                language = ext[1:].lower() if ext else "unknown"
            
            # Read file content
            try:
                with open(path, 'r') as f:
                    content = f.read()
            except FileNotFoundError:
                return EvaluationError(f"File not found: {path}")
            except Exception as e:
                return EvaluationError(f"eval_file() read error: {str(e)}")
            
            # Execute based on language
            if language == "zx" or language == "zexus":
                # Execute Zexus code
                from ..parser.parser import UltimateParser
                from ..lexer import Lexer
                
                try:
                    lexer = Lexer(content)
                    parser = UltimateParser(lexer)
                    program = parser.parse_program()
                    
                    if parser.errors:
                        return EvaluationError(f"Parse errors: {parser.errors[0]}")
                    
                    # Use the current evaluator instance to execute in a new environment
                    from ..object import Environment
                    new_env = Environment()
                    
                    # Copy builtins to new environment
                    for name, builtin in self.builtins.items():
                        new_env.set(name, builtin)
                    
                    result = NULL
                    for stmt in program.statements:
                        result = self.eval_node(stmt, new_env, [])
                        if is_error(result):
                            return result
                    
                    # Export all defined functions/actions to global builtins
                    # This allows cross-file code reuse
                    for key in new_env.store.keys():
                        if key not in ['__file__', '__FILE__', '__MODULE__', '__DIR__', '__ARGS__', '__ARGV__', '__PACKAGE__']:
                            val = new_env.get(key)
                            if val and not is_error(val):
                                # Add to builtins so it's available globally
                                self.builtins[key] = val
                    
                    return result if result else NULL
                except Exception as e:
                    return EvaluationError(f"eval_file() zexus execution error: {str(e)}")
            
            elif language == "py" or language == "python":
                # Execute Python code
                try:
                    exec_globals = {}
                    exec(content, exec_globals)
                    # Return the result if there's a 'result' variable
                    if 'result' in exec_globals:
                        result_val = exec_globals['result']
                        # Convert Python types to Zexus types
                        if isinstance(result_val, str):
                            return String(result_val)
                        elif isinstance(result_val, int):
                            return Integer(result_val)
                        elif isinstance(result_val, float):
                            return Float(result_val)
                        elif isinstance(result_val, bool):
                            return Boolean(result_val)
                        elif isinstance(result_val, list):
                            return List([Integer(x) if isinstance(x, int) else String(str(x)) for x in result_val])
                    return NULL
                except Exception as e:
                    return EvaluationError(f"eval_file() python execution error: {str(e)}")
            
            elif language in ["cpp", "c++", "c", "rs", "rust", "go"]:
                # For compiled languages, try to compile and run
                return EvaluationError(f"eval_file() for {language} requires compilation - not yet implemented")
            
            elif language == "js" or language == "javascript":
                # Execute JavaScript (if Node.js is available)
                try:
                    result = subprocess.run(['node', '-e', content], 
                                          capture_output=True, 
                                          text=True, 
                                          timeout=5)
                    if result.returncode != 0:
                        return EvaluationError(f"JavaScript error: {result.stderr}")
                    return String(result.stdout.strip())
                except FileNotFoundError:
                    return EvaluationError("Node.js not found - cannot execute JavaScript")
                except Exception as e:
                    return EvaluationError(f"eval_file() js execution error: {str(e)}")
            
            else:
                return EvaluationError(f"Unsupported language: {language}")
        
        # Contract Assertions
        def _require(*a):
            """Assert a condition in smart contracts: require(condition, message)
            
            Throws an error if condition is false. Essential for contract validation.
            
            Example:
                require(balance >= amount, "Insufficient balance")
                require(sender == owner, "Not authorized")
                require(value > 0, "Amount must be positive")
            """
            if len(a) < 1 or len(a) > 2:
                return EvaluationError("require() takes 1-2 arguments: require(condition, [message])")
            
            condition = a[0]
            message = a[1].value if len(a) > 1 and isinstance(a[1], String) else "Requirement failed"
            
            # Check if condition is truthy
            from .utils import is_truthy
            if not is_truthy(condition):
                # Return error with contract-specific formatting
                return EvaluationError(f"Contract requirement failed: {message}")
            
            # Condition passed, return NULL
            return NULL
        
        # Contract Assertions
        def _require(*a):
            """Assert a condition in smart contracts: require(condition, message)
            
            Throws an error if condition is false. Essential for contract validation.
            Note: This is a fallback for contexts where the require statement isn't available.
            
            Example:
                require(balance >= amount, "Insufficient balance")
                require(sender == owner, "Not authorized")
                require(value > 0, "Amount must be positive")
            """
            if len(a) < 1 or len(a) > 2:
                return EvaluationError("require() takes 1-2 arguments: require(condition, [message])")
            
            condition = a[0]
            message = a[1].value if len(a) > 1 and isinstance(a[1], String) else "Requirement failed"
            
            # Check if condition is truthy
            from .utils import is_truthy
            if not is_truthy(condition):
                # Return error with contract-specific formatting
                return EvaluationError(f"Contract requirement failed: {message}")
            
            # Condition passed, return NULL
            return NULL
        
        # Map/Object helper functions
        def _keys(*a):
            """Get all keys from a map: keys(map) -> [key1, key2, ...]"""
            if len(a) != 1:
                return EvaluationError("keys() requires exactly 1 argument: map")
            if not isinstance(a[0], Map):
                return EvaluationError(f"keys() argument must be a Map, got {type(a[0]).__name__}")
            
            return List([String(k) if isinstance(k, str) else k for k in a[0].pairs.keys()])
        
        def _values(*a):
            """Get all values from a map: values(map) -> [val1, val2, ...]"""
            if len(a) != 1:
                return EvaluationError("values() requires exactly 1 argument: map")
            if not isinstance(a[0], Map):
                return EvaluationError(f"values() argument must be a Map, got {type(a[0]).__name__}")
            
            return List([v for v in a[0].pairs.values()])
        
        def _entries(*a):
            """Get all entries from a map: entries(map) -> [[k1,v1], [k2,v2], ...]"""
            if len(a) != 1:
                return EvaluationError("entries() requires exactly 1 argument: map")
            if not isinstance(a[0], Map):
                return EvaluationError(f"entries() argument must be a Map, got {type(a[0]).__name__}")
            
            return List([List([String(k) if isinstance(k, str) else k, v]) for k, v in a[0].pairs.items()])
        
        def _size(*a):
            """Get size of a collection: size(map|list) -> int"""
            if len(a) != 1:
                return EvaluationError("size() requires exactly 1 argument")
            
            obj = a[0]
            if isinstance(obj, Map):
                return Integer(len(obj.pairs))
            elif isinstance(obj, List):
                return Integer(len(obj.elements))
            elif isinstance(obj, String):
                return Integer(len(obj.value))
            else:
                return EvaluationError(f"size() argument must be a Map, List, or String, got {type(obj).__name__}")
        
        # Register mappings
        self.builtins.update({
            "now": Builtin(_now, "now"),
            "timestamp": Builtin(_timestamp, "timestamp"),
            "time": Builtin(_time, "time"),
            "random": Builtin(_random, "random"),
            "to_hex": Builtin(_to_hex, "to_hex"),
            "from_hex": Builtin(_from_hex, "from_hex"),
            "sqrt": Builtin(_sqrt, "sqrt"),
            "require": Builtin(_require, "require"),
            "require": Builtin(_require, "require"),
            "input": Builtin(_input, "input"),
            "hash_password": Builtin(_hash_password, "hash_password"),
            "verify_password": Builtin(_verify_password, "verify_password"),
            "crypto_random": Builtin(_crypto_random, "crypto_random"),
            "constant_time_compare": Builtin(_constant_time_compare, "constant_time_compare"),
            "file": Builtin(_file, "file"),
            "file_read_text": Builtin(_read_text, "file_read_text"),
            "file_write_text": Builtin(_write_text, "file_write_text"),
            "file_exists": Builtin(_exists, "file_exists"),
            "file_read_json": Builtin(_read_json, "file_read_json"),
            "file_write_json": Builtin(_write_json, "file_write_json"),
            "file_append": Builtin(_file_append, "file_append"),
            "file_list_dir": Builtin(_list_dir, "file_list_dir"),
            "fs_is_file": Builtin(_fs_is_file, "fs_is_file"),
            "fs_is_dir": Builtin(_fs_is_dir, "fs_is_dir"),
            "fs_mkdir": Builtin(_fs_mkdir, "fs_mkdir"),
            "fs_remove": Builtin(_fs_remove, "fs_remove"),
            "fs_rmdir": Builtin(_fs_rmdir, "fs_rmdir"),
            "fs_rename": Builtin(_fs_rename, "fs_rename"),
            "fs_copy": Builtin(_fs_copy, "fs_copy"),
            "socket_create_server": Builtin(_socket_create_server, "socket_create_server"),
            "socket_create_connection": Builtin(_socket_create_connection, "socket_create_connection"),
            "http_server": Builtin(_http_server, "http_server"),
            "sqlite_connect": Builtin(_sqlite_connect, "sqlite_connect"),
            "postgres_connect": Builtin(_postgres_connect, "postgres_connect"),
            "mysql_connect": Builtin(_mysql_connect, "mysql_connect"),
            "mongo_connect": Builtin(_mongo_connect, "mongo_connect"),
            "http_get": Builtin(_http_get, "http_get"),
            "http_post": Builtin(_http_post, "http_post"),
            "http_put": Builtin(_http_put, "http_put"),
            "http_delete": Builtin(_http_delete, "http_delete"),
            "http_request": Builtin(_http_request, "http_request"),
            "http_parallel_get": Builtin(_http_parallel_get, "http_parallel_get"),
            "http_async_get": Builtin(_http_async_get, "http_async_get"),
            "read_file": Builtin(_read_file, "read_file"),
            "eval_file": Builtin(_eval_file, "eval_file"),
            "vfs_stats": Builtin(_vfs_stats, "vfs_stats"),
            "vfs_clear_cache": Builtin(_vfs_clear_cache, "vfs_clear_cache"),
            "debug": Builtin(_debug, "debug"),
            "debug_log": Builtin(_debug_log, "debug_log"),
            "debug_trace": Builtin(_debug_trace, "debug_trace"),
            "string": Builtin(_string, "string"),
            "int": Builtin(_int, "int"),
            "float": Builtin(_float, "float"),
            "uppercase": Builtin(_uppercase, "uppercase"),
            "lowercase": Builtin(_lowercase, "lowercase"),
            "split": Builtin(_split, "split"),
            "random": Builtin(_random, "random"),
            "persist_set": Builtin(_persist_set, "persist_set"),
            "persist_get": Builtin(_persist_get, "persist_get"),
            "input": Builtin(_input, "input"),
            "len": Builtin(_len, "len"),
            "type": Builtin(_type, "type"),
            "first": Builtin(_first, "first"),
            "rest": Builtin(_rest, "rest"),
            "push": Builtin(_push, "push"),
            "append": Builtin(_append, "append"),  # Mutating list append
            "extend": Builtin(_extend, "extend"),  # Mutating list extend
            "sort": Builtin(_sort, "sort"),        # Sort list or list of maps by key
            "slice": Builtin(_slice, "slice"),     # Get list slice: slice(list, start, end?)
            "reduce": Builtin(_reduce, "reduce"),
            "map": Builtin(_map, "map"),
            "filter": Builtin(_filter, "filter"),
            "keys": Builtin(_keys, "keys"),
            "values": Builtin(_values, "values"),
            "entries": Builtin(_entries, "entries"),
            "size": Builtin(_size, "size"),
        })
        
        # Register access control builtins
        self._register_access_control_builtins()
        
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
        
        def _async(*a):
            """Execute action asynchronously in background thread: async action_call()
            
            Example: async producer()
            
            Accepts either:
            1. A Coroutine (from calling an async action)
            2. A regular value (from calling a regular action) - will execute in thread
            """
            import threading
            import sys
            
            if len(a) != 1:
                return EvaluationError("async() requires 1 argument: result of action call")
            
            result = a[0]
            
            # If it's already a Coroutine, start it in a thread
            if hasattr(result, '__class__') and result.__class__.__name__ == 'Coroutine':
                
                def run_coroutine():
                    try:
                        # Prime the generator
                        val = next(result.generator)
                        # Execute until completion
                        try:
                            while True:
                                val = next(result.generator)
                        except StopIteration as e:
                            # Coroutine completed successfully
                            pass
                    except Exception as e:
                        # Print error to stderr for visibility
                        print(f"[ASYNC ERROR] Coroutine execution failed: {str(e)}", file=sys.stderr)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                
                thread = threading.Thread(target=run_coroutine, daemon=True)
                thread.start()
                return NULL
            
            # For regular (non-async) actions, the action has already executed!
            # This is because producer() executes immediately and returns its result.
            # So async(producer()) just receives the result.
            # We need a different approach - we can't retroactively make it async.
            
            # The solution: If they want async execution, the action itself must be async.
            # For now, just return NULL to indicate "completed" (it already ran).
            return NULL
        
        def _sleep(*a):
            """Sleep for specified seconds: sleep(seconds)"""
            import time
            
            if len(a) != 1:
                return EvaluationError("sleep() requires 1 argument: seconds")
            
            seconds = a[0]
            if isinstance(seconds, (Integer, Float)):
                time.sleep(float(seconds.value))
                return NULL
            
            return EvaluationError(f"sleep() argument must be a number, got {type(seconds).__name__}")
        
        def _spawn(*a):
            """Spawn an async task and return a coroutine that can be awaited: task = spawn async_func()
            
            Example:
                async function asyncTask(id) {
                    await sleep(1)
                    return id * 10
                }
                let task1 = spawn asyncTask(1)
                let result = await task1
            """
            import threading
            import sys
            
            if len(a) != 1:
                return EvaluationError("spawn() requires 1 argument: coroutine or async function call result")
            
            result = a[0]
            
            # If it's a Coroutine, we need to wrap it for async execution
            from ..object import Coroutine
            if isinstance(result, Coroutine):
                # Create a wrapper coroutine that executes in background
                # The original coroutine will be run in a thread
                task_result = [None]  # Mutable container to store result
                task_error = [None]
                task_complete = [False]
                
                def run_coroutine():
                    try:
                        # Prime the generator
                        next(result.generator)
                        # Execute until completion
                        try:
                            while True:
                                next(result.generator)
                        except StopIteration as e:
                            # Coroutine completed, store result
                            task_result[0] = e.value if hasattr(e, 'value') else NULL
                            task_complete[0] = True
                    except Exception as e:
                        # Store error
                        task_error[0] = e
                        task_complete[0] = True
                        print(f"[SPAWN ERROR] Task execution failed: {str(e)}", file=sys.stderr)
                        import traceback
                        traceback.print_exc(file=sys.stderr)
                
                # Start execution in background thread
                thread = threading.Thread(target=run_coroutine, daemon=False)
                thread.start()
                
                # Create a new coroutine that waits for the result
                def result_generator():
                    yield None  # Make it a generator
                    # Wait for completion
                    thread.join(timeout=30)  # 30 second timeout
                    if task_error[0]:
                        raise task_error[0]
                    return task_result[0] if task_result[0] is not None else NULL
                
                return Coroutine(result_generator(), result.fn if hasattr(result, 'fn') else None)
            
            # If not a coroutine, return error
            return EvaluationError(f"spawn() argument must be a coroutine (async function call), got {type(result).__name__}")
        
        def _wait_group(*a):
            """Create a wait group for synchronizing async operations: wg = wait_group()
            
            Example:
                let wg = wait_group()
                wg.add(2)  # Expecting 2 tasks
                async task1()
                async task2()
                wg.wait()  # Blocks until both tasks call wg.done()
            """
            from ..concurrency_system import WaitGroup
            
            if len(a) != 0:
                return EvaluationError("wait_group() takes no arguments")
            
            return WaitGroup()
        
        def _wg_add(*a):
            """Add delta to wait group counter: wg.add(delta)"""
            if len(a) != 2:
                return EvaluationError("wg_add() requires 2 arguments: wait_group, delta")
            
            wg = a[0]
            delta_obj = a[1]
            
            if not hasattr(wg, 'add'):
                return EvaluationError(f"wg_add() first argument must be a WaitGroup, got {type(wg).__name__}")
            
            if isinstance(delta_obj, Integer):
                try:
                    wg.add(delta_obj.value)
                    return NULL
                except Exception as e:
                    return EvaluationError(f"wg_add() error: {str(e)}")
            elif isinstance(delta_obj, int):
                try:
                    wg.add(delta_obj)
                    return NULL
                except Exception as e:
                    return EvaluationError(f"wg_add() error: {str(e)}")
            
            return EvaluationError(f"wg_add() delta must be an integer, got {type(delta_obj).__name__}")
        
        def _wg_done(*a):
            """Decrement wait group counter: wg.done()"""
            if len(a) != 1:
                return EvaluationError("wg_done() requires 1 argument: wait_group")
            
            wg = a[0]
            
            if not hasattr(wg, 'done'):
                return EvaluationError(f"wg_done() argument must be a WaitGroup, got {type(wg).__name__}")
            
            try:
                wg.done()
                return NULL
            except Exception as e:
                return EvaluationError(f"wg_done() error: {str(e)}")
        
        def _wg_wait(*a):
            """Wait for wait group counter to reach zero: wg.wait()"""
            if len(a) < 1 or len(a) > 2:
                return EvaluationError("wg_wait() requires 1 or 2 arguments: wait_group [, timeout]")
            
            wg = a[0]
            timeout = None
            
            if len(a) == 2:
                timeout_obj = a[1]
                if isinstance(timeout_obj, (Integer, Float)):
                    timeout = float(timeout_obj.value)
                elif isinstance(timeout_obj, (int, float)):
                    timeout = float(timeout_obj)
                else:
                    return EvaluationError(f"wg_wait() timeout must be a number, got {type(timeout_obj).__name__}")
            
            if not hasattr(wg, 'wait'):
                return EvaluationError(f"wg_wait() first argument must be a WaitGroup, got {type(wg).__name__}")
            
            try:
                success = wg.wait(timeout=timeout)
                return TRUE if success else FALSE
            except Exception as e:
                return EvaluationError(f"wg_wait() error: {str(e)}")
        
        def _barrier(*a):
            """Create a barrier for synchronizing N tasks: barrier = barrier(parties)
            
            Example:
                let barrier = barrier(2)  # Wait for 2 tasks
                async task1()  # Will call barrier.wait()
                async task2()  # Will call barrier.wait()
                # Both released once both reach barrier
            """
            from ..concurrency_system import Barrier
            
            if len(a) != 1:
                return EvaluationError("barrier() requires 1 argument: parties")
            
            parties_obj = a[0]
            if isinstance(parties_obj, Integer):
                try:
                    return Barrier(parties=parties_obj.value)
                except Exception as e:
                    return EvaluationError(f"barrier() error: {str(e)}")
            
            return EvaluationError(f"barrier() parties must be an integer, got {type(parties_obj).__name__}")
        
        def _barrier_wait(*a):
            """Wait at barrier until all parties arrive: barrier.wait()"""
            if len(a) < 1 or len(a) > 2:
                return EvaluationError("barrier_wait() requires 1 or 2 arguments: barrier [, timeout]")
            
            barrier = a[0]
            timeout = None
            
            if len(a) == 2:
                timeout_obj = a[1]
                if isinstance(timeout_obj, (Integer, Float)):
                    timeout = float(timeout_obj.value)
                else:
                    return EvaluationError(f"barrier_wait() timeout must be a number, got {type(timeout_obj).__name__}")
            
            if not hasattr(barrier, 'wait'):
                return EvaluationError(f"barrier_wait() first argument must be a Barrier, got {type(barrier).__name__}")
            
            try:
                generation = barrier.wait(timeout=timeout)
                return Integer(generation)
            except Exception as e:
                return EvaluationError(f"barrier_wait() error: {str(e)}")
        
        def _barrier_reset(*a):
            """Reset barrier to initial state: barrier.reset()"""
            if len(a) != 1:
                return EvaluationError("barrier_reset() requires 1 argument: barrier")
            
            barrier = a[0]
            
            if not hasattr(barrier, 'reset'):
                return EvaluationError(f"barrier_reset() argument must be a Barrier, got {type(barrier).__name__}")
            
            try:
                barrier.reset()
                return NULL
            except Exception as e:
                return EvaluationError(f"barrier_reset() error: {str(e)}")
        
        # Register concurrency builtins
        self.builtins.update({
            "send": Builtin(_send, "send"),
            "receive": Builtin(_receive, "receive"),
            "close_channel": Builtin(_close_channel, "close_channel"),
            "async": Builtin(_async, "async"),
            "sleep": Builtin(_sleep, "sleep"),
            "spawn": Builtin(_spawn, "spawn"),
            "wait_group": Builtin(_wait_group, "wait_group"),
            "wg_add": Builtin(_wg_add, "wg_add"),
            "wg_done": Builtin(_wg_done, "wg_done"),
            "wg_wait": Builtin(_wg_wait, "wg_wait"),
            "barrier": Builtin(_barrier, "barrier"),
            "barrier_wait": Builtin(_barrier_wait, "barrier_wait"),
            "barrier_reset": Builtin(_barrier_reset, "barrier_reset"),
        })
    
    def _register_access_control_builtins(self):
        """Register access control functions for contracts"""
        from ..access_control_system import get_access_control
        from ..blockchain.transaction import get_current_tx
        
        def _set_owner(*a):
            """Set owner of current contract: set_owner(contract_id, owner_address)"""
            if len(a) != 2:
                return EvaluationError("set_owner() requires 2 arguments: contract_id, owner_address")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            owner = a[1].value if hasattr(a[1], 'value') else str(a[1])
            
            ac = get_access_control()
            ac.set_owner(contract_id, owner)
            return NULL
        
        def _get_owner(*a):
            """Get owner of contract: get_owner(contract_id)"""
            if len(a) != 1:
                return EvaluationError("get_owner() requires 1 argument: contract_id")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            
            ac = get_access_control()
            owner = ac.get_owner(contract_id)
            return String(owner) if owner else NULL
        
        def _is_owner(*a):
            """Check if address is owner: is_owner(contract_id, address)"""
            if len(a) != 2:
                return EvaluationError("is_owner() requires 2 arguments: contract_id, address")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            address = a[1].value if hasattr(a[1], 'value') else str(a[1])
            
            ac = get_access_control()
            return TRUE if ac.is_owner(contract_id, address) else FALSE
        
        def _grant_role(*a):
            """Grant role to address: grant_role(contract_id, address, role)"""
            if len(a) != 3:
                return EvaluationError("grant_role() requires 3 arguments: contract_id, address, role")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            address = a[1].value if hasattr(a[1], 'value') else str(a[1])
            role = a[2].value if hasattr(a[2], 'value') else str(a[2])
            
            ac = get_access_control()
            ac.grant_role(contract_id, address, role)
            return NULL
        
        def _revoke_role(*a):
            """Revoke role from address: revoke_role(contract_id, address, role)"""
            if len(a) != 3:
                return EvaluationError("revoke_role() requires 3 arguments: contract_id, address, role")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            address = a[1].value if hasattr(a[1], 'value') else str(a[1])
            role = a[2].value if hasattr(a[2], 'value') else str(a[2])
            
            ac = get_access_control()
            ac.revoke_role(contract_id, address, role)
            return NULL
        
        def _has_role(*a):
            """Check if address has role: has_role(contract_id, address, role)"""
            if len(a) != 3:
                return EvaluationError("has_role() requires 3 arguments: contract_id, address, role")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            address = a[1].value if hasattr(a[1], 'value') else str(a[1])
            role = a[2].value if hasattr(a[2], 'value') else str(a[2])
            
            ac = get_access_control()
            return TRUE if ac.has_role(contract_id, address, role) else FALSE
        
        def _get_roles(*a):
            """Get all roles for address: get_roles(contract_id, address)"""
            if len(a) != 2:
                return EvaluationError("get_roles() requires 2 arguments: contract_id, address")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            address = a[1].value if hasattr(a[1], 'value') else str(a[1])
            
            ac = get_access_control()
            roles = ac.get_roles(contract_id, address)
            return List([String(role) for role in roles])
        
        def _grant_permission(*a):
            """Grant permission to address: grant_permission(contract_id, address, permission)"""
            if len(a) != 3:
                return EvaluationError("grant_permission() requires 3 arguments: contract_id, address, permission")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            address = a[1].value if hasattr(a[1], 'value') else str(a[1])
            permission = a[2].value if hasattr(a[2], 'value') else str(a[2])
            
            ac = get_access_control()
            ac.grant_permission(contract_id, address, permission)
            return NULL
        
        def _revoke_permission(*a):
            """Revoke permission from address: revoke_permission(contract_id, address, permission)"""
            if len(a) != 3:
                return EvaluationError("revoke_permission() requires 3 arguments: contract_id, address, permission")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            address = a[1].value if hasattr(a[1], 'value') else str(a[1])
            permission = a[2].value if hasattr(a[2], 'value') else str(a[2])
            
            ac = get_access_control()
            ac.revoke_permission(contract_id, address, permission)
            return NULL
        
        def _has_permission(*a):
            """Check if address has permission: has_permission(contract_id, address, permission)"""
            if len(a) != 3:
                return EvaluationError("has_permission() requires 3 arguments: contract_id, address, permission")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            address = a[1].value if hasattr(a[1], 'value') else str(a[1])
            permission = a[2].value if hasattr(a[2], 'value') else str(a[2])
            
            ac = get_access_control()
            return TRUE if ac.has_permission(contract_id, address, permission) else FALSE
        
        def _require_owner(*a):
            """Require caller is owner: require_owner(contract_id, message?)"""
            if len(a) < 1 or len(a) > 2:
                return EvaluationError("require_owner() requires 1 or 2 arguments: contract_id, [message]")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            message = a[1].value if len(a) > 1 and hasattr(a[1], 'value') else None
            
            # Get current transaction caller
            tx = get_current_tx()
            if not tx:
                return EvaluationError("require_owner() requires transaction context (TX.caller)")
            
            caller = tx.caller
            
            ac = get_access_control()
            try:
                if message:
                    ac.require_owner(contract_id, caller, message)
                else:
                    ac.require_owner(contract_id, caller)
                return NULL
            except Exception as e:
                return EvaluationError(str(e))
        
        def _require_role(*a):
            """Require caller has role: require_role(contract_id, role, message?)"""
            if len(a) < 2 or len(a) > 3:
                return EvaluationError("require_role() requires 2 or 3 arguments: contract_id, role, [message]")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            role = a[1].value if hasattr(a[1], 'value') else str(a[1])
            message = a[2].value if len(a) > 2 and hasattr(a[2], 'value') else None
            
            # Get current transaction caller
            tx = get_current_tx()
            if not tx:
                return EvaluationError("require_role() requires transaction context (TX.caller)")
            
            caller = tx.caller
            
            ac = get_access_control()
            try:
                if message:
                    ac.require_role(contract_id, caller, role, message)
                else:
                    ac.require_role(contract_id, caller, role)
                return NULL
            except Exception as e:
                return EvaluationError(str(e))
        
        def _require_permission(*a):
            """Require caller has permission: require_permission(contract_id, permission, message?)"""
            if len(a) < 2 or len(a) > 3:
                return EvaluationError("require_permission() requires 2 or 3 arguments: contract_id, permission, [message]")
            
            contract_id = a[0].value if hasattr(a[0], 'value') else str(a[0])
            permission = a[1].value if hasattr(a[1], 'value') else str(a[1])
            message = a[2].value if len(a) > 2 and hasattr(a[2], 'value') else None
            
            # Get current transaction caller
            tx = get_current_tx()
            if not tx:
                return EvaluationError("require_permission() requires transaction context (TX.caller)")
            
            caller = tx.caller
            
            ac = get_access_control()
            try:
                if message:
                    ac.require_permission(contract_id, caller, permission, message)
                else:
                    ac.require_permission(contract_id, caller, permission)
                return NULL
            except Exception as e:
                return EvaluationError(str(e))
        
        # Register access control builtins
        self.builtins.update({
            "set_owner": Builtin(_set_owner, "set_owner"),
            "get_owner": Builtin(_get_owner, "get_owner"),
            "is_owner": Builtin(_is_owner, "is_owner"),
            "grant_role": Builtin(_grant_role, "grant_role"),
            "revoke_role": Builtin(_revoke_role, "revoke_role"),
            "has_role": Builtin(_has_role, "has_role"),
            "get_roles": Builtin(_get_roles, "get_roles"),
            "grant_permission": Builtin(_grant_permission, "grant_permission"),
            "revoke_permission": Builtin(_revoke_permission, "revoke_permission"),
            "has_permission": Builtin(_has_permission, "has_permission"),
            "require_owner": Builtin(_require_owner, "require_owner"),
            "require_role": Builtin(_require_role, "require_role"),
            "require_permission": Builtin(_require_permission, "require_permission"),
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
            import sys
            import gc
            
            # Get process memory usage
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                current_bytes = mem_info.rss  # Resident Set Size
                peak_bytes = getattr(mem_info, 'peak_wset', mem_info.rss)  # Windows has peak_wset
            except (ImportError, AttributeError):
                # Fallback: use Python's internal memory tracking
                current_bytes = sys.getsizeof(gc.get_objects())
                peak_bytes = current_bytes
            
            # Get GC statistics
            gc_count = len(gc.get_objects())
            gc_collections = sum(gc.get_count())
            
            # Get environment-specific tracking if available
            env = getattr(self, '_current_env', None)
            tracked_objects = 0
            if env and hasattr(env, 'get_memory_stats'):
                stats = env.get_memory_stats()
                tracked_objects = stats.get("tracked_objects", 0)
            
            return Map({
                String("current"): Integer(current_bytes),
                String("peak"): Integer(peak_bytes),
                String("gc_count"): Integer(gc_collections),
                String("objects"): Integer(gc_count),
                String("tracked_objects"): Integer(tracked_objects)
            })
        
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
    
    def _register_main_entry_point_builtins(self):
        """Register builtins for main entry point pattern and continuous execution"""
        import signal
        import time as time_module
        
        # Storage for lifecycle hooks and signal handlers
        self._lifecycle_hooks = {'on_start': [], 'on_exit': []}
        self._signal_handlers = {}
        
        def _run(*a):
            """
            Keep the program running until interrupted (Ctrl+C).
            Useful for servers, event loops, or long-running programs.
            
            Enhanced version supports:
            - callback with arguments
            - interval timing
            - lifecycle hooks (on_start, on_exit)
            
            Usage:
                if __MODULE__ == "__main__":
                    run()
            
            or with a callback:
                if __MODULE__ == "__main__":
                    run(lambda: print("Still running..."))
            
            or with callback and interval:
                if __MODULE__ == "__main__":
                    run(callback, 0.5)  # Run every 500ms
            
            or with callback and arguments:
                if __MODULE__ == "__main__":
                    run(server.process_requests, 1.0, [port, host])
            """
            callback = None
            interval = 1.0  # Default interval in seconds
            callback_args = []
            
            if len(a) >= 1:
                # First argument is the callback function
                callback = a[0]
                if not isinstance(callback, (Action, LambdaFunction)):
                    return EvaluationError("run() callback must be a function")
            
            if len(a) >= 2:
                # Second argument is the interval
                interval_obj = a[1]
                if isinstance(interval_obj, (Integer, Float)):
                    interval = float(interval_obj.value)
                else:
                    return EvaluationError("run() interval must be a number")
            
            if len(a) >= 3:
                # Third argument is callback arguments
                if isinstance(a[2], List):
                    callback_args = a[2].elements
                else:
                    callback_args = [a[2]]
            
            print("🚀 Program running. Press Ctrl+C to exit.")
            
            # Execute on_start hooks
            for hook in self._lifecycle_hooks.get('on_start', []):
                try:
                    result = self.apply_function(hook, [])
                    if is_error(result):
                        print(f"⚠️  on_start hook error: {result.message}")
                except Exception as e:
                    print(f"⚠️  on_start hook error: {str(e)}")
            
            # Setup signal handler for graceful shutdown
            shutdown_requested = [False]  # Use list for closure mutability
            
            def signal_handler(sig, frame):
                shutdown_requested[0] = True
                print("\n⏹️  Shutdown requested. Cleaning up...")
                
                # Execute custom signal handlers if registered
                sig_name = signal.Signals(sig).name
                if sig_name in self._signal_handlers:
                    for handler in self._signal_handlers[sig_name]:
                        try:
                            self.apply_function(handler, [String(sig_name)])
                        except Exception as e:
                            print(f"⚠️  Signal handler error: {str(e)}")
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            try:
                # Keep running until interrupted
                while not shutdown_requested[0]:
                    if callback:
                        # Execute callback function with arguments
                        result = self.apply_function(callback, callback_args)
                        if is_error(result):
                            print(f"⚠️  Callback error: {result.message}")
                    
                    # Sleep for the interval
                    time_module.sleep(interval)
            
            except KeyboardInterrupt:
                print("\n⏹️  Interrupted. Exiting...")
            
            except Exception as e:
                print(f"❌ Error in run loop: {str(e)}")
                return EvaluationError(f"run() error: {str(e)}")
            
            finally:
                # Execute on_exit hooks
                for hook in self._lifecycle_hooks.get('on_exit', []):
                    try:
                        result = self.apply_function(hook, [])
                        if is_error(result):
                            print(f"⚠️  on_exit hook error: {result.message}")
                    except Exception as e:
                        print(f"⚠️  on_exit hook error: {str(e)}")
                
                print("✅ Program terminated gracefully.")
            
            return NULL
        
        def _execute(*a):
            """
            Alias for run() - keeps the program executing until interrupted.
            
            Usage:
                if __MODULE__ == "__main__":
                    execute()
            """
            return _run(*a)
        
        def _is_main(*a):
            """
            Check if the current module is being run as the main program.
            Returns true if __MODULE__ == "__main__", false otherwise.
            
            Usage:
                if is_main():
                    print("Running as main program")
            """
            env = getattr(self, '_current_env', None)
            if env:
                module_name = env.get('__MODULE__')
                if module_name and isinstance(module_name, String):
                    return TRUE if module_name.value == "__main__" else FALSE
            return FALSE
        
        def _exit_program(*a):
            """
            Exit the program with an optional exit code.
            
            Usage:
                exit_program()      # Exit with code 0
                exit_program(1)     # Exit with code 1
            """
            exit_code = 0
            if len(a) >= 1:
                if isinstance(a[0], Integer):
                    exit_code = a[0].value
                else:
                    return EvaluationError("exit_program() exit code must be an integer")
            
            # Execute on_exit hooks before exiting
            for hook in self._lifecycle_hooks.get('on_exit', []):
                try:
                    result = self.apply_function(hook, [])
                    if is_error(result):
                        print(f"⚠️  on_exit hook error: {result.message}")
                except Exception as e:
                    print(f"⚠️  on_exit hook error: {str(e)}")
            
            print(f"👋 Exiting with code {exit_code}")
            sys.exit(exit_code)
        
        def _on_start(*a):
            """
            Register a callback to run when the program starts (before run loop).
            
            Usage:
                on_start(lambda: print("Starting up..."))
                on_start(initialize_database)
            """
            if len(a) != 1:
                return EvaluationError("on_start() requires exactly one function argument")
            
            callback = a[0]
            if not isinstance(callback, (Action, LambdaFunction)):
                return EvaluationError("on_start() argument must be a function")
            
            self._lifecycle_hooks['on_start'].append(callback)
            return NULL
        
        def _on_exit(*a):
            """
            Register a callback to run when the program exits (after run loop).
            
            Usage:
                on_exit(lambda: print("Cleaning up..."))
                on_exit(close_connections)
            """
            if len(a) != 1:
                return EvaluationError("on_exit() requires exactly one function argument")
            
            callback = a[0]
            if not isinstance(callback, (Action, LambdaFunction)):
                return EvaluationError("on_exit() argument must be a function")
            
            self._lifecycle_hooks['on_exit'].append(callback)
            return NULL
        
        def _signal_handler(*a):
            """
            Register a custom signal handler for specific signals.
            
            Usage:
                signal_handler("SIGINT", lambda sig: print("Caught SIGINT"))
                signal_handler("SIGTERM", cleanup_handler)
            """
            if len(a) != 2:
                return EvaluationError("signal_handler() requires signal name and callback function")
            
            signal_name = _to_str(a[0])
            callback = a[1]
            
            if not isinstance(callback, (Action, LambdaFunction)):
                return EvaluationError("signal_handler() callback must be a function")
            
            if signal_name not in self._signal_handlers:
                self._signal_handlers[signal_name] = []
            
            self._signal_handlers[signal_name].append(callback)
            return NULL
        
        def _schedule(*a):
            """
            Schedule multiple tasks with different intervals to run in parallel.
            
            Usage:
                schedule([
                    {interval: 1, action: check_queue},
                    {interval: 5, action: save_state},
                    {interval: 60, action: cleanup}
                ])
            
            Returns: List of task IDs
            """
            if len(a) != 1:
                return EvaluationError("schedule() requires a list of task definitions")
            
            tasks_arg = a[0]
            if not isinstance(tasks_arg, List):
                return EvaluationError("schedule() argument must be a list")
            
            import threading
            import time as time_module
            
            task_ids = []
            
            for i, task_def in enumerate(tasks_arg.elements):
                if not isinstance(task_def, Map):
                    return EvaluationError(f"Task {i} must be a map with 'interval' and 'action' keys")
                
                # Extract interval and action - map keys can be strings or String objects
                interval_obj = None
                action_obj = None
                
                for key, value in task_def.pairs.items():
                    key_str = key if isinstance(key, str) else (key.value if hasattr(key, 'value') else str(key))
                    if key_str == "interval":
                        interval_obj = value
                    elif key_str == "action":
                        action_obj = value
                
                if not interval_obj or not action_obj:
                    return EvaluationError(f"Task {i} must have 'interval' and 'action' keys")
                
                if isinstance(interval_obj, (Integer, Float)):
                    interval = float(interval_obj.value)
                else:
                    return EvaluationError(f"Task {i} interval must be a number")
                
                if not isinstance(action_obj, (Action, LambdaFunction)):
                    return EvaluationError(f"Task {i} action must be a function")
                
                # Create task thread
                task_id = f"task_{i}_{id(action_obj)}"
                task_ids.append(String(task_id))
                
                def task_worker(action, interval_sec, task_id):
                    """Worker function that runs the task at specified interval"""
                    while True:
                        try:
                            time_module.sleep(interval_sec)
                            result = self.apply_function(action, [])
                            if is_error(result):
                                print(f"⚠️  Task {task_id} error: {result.message}")
                        except Exception as e:
                            print(f"⚠️  Task {task_id} exception: {str(e)}")
                            break
                
                # Start thread in daemon mode so it exits when main program exits
                thread = threading.Thread(
                    target=task_worker,
                    args=(action_obj, interval, task_id),
                    daemon=True
                )
                thread.start()
            
            return List(task_ids)
        
        def _sleep(*args):
            """
            Sleep for specified seconds.
            
            Usage:
                sleep(2)      # Sleep for 2 seconds
                sleep(0.5)    # Sleep for 0.5 seconds
            """
            if len(args) != 1:
                return EvaluationError("sleep() requires exactly 1 argument (seconds)")
            
            seconds_arg = args[0]
            if isinstance(seconds_arg, (Integer, Float)):
                try:
                    time_module.sleep(float(seconds_arg.value))
                    return NULL
                except Exception as e:
                    return EvaluationError(f"sleep() error: {str(e)}")
            else:
                return EvaluationError("sleep() argument must be a number")
        
        def _daemonize(*args):
            """
            Run the current process as a background daemon.
            
            Detaches from terminal and runs in background. On Unix systems, this
            performs a double fork to properly daemonize. On Windows, it's a no-op.
            
            Usage:
                if is_main() {
                    daemonize()
                    # Now running as daemon
                    run(my_server_task)
                }
            
            Optional arguments:
                daemonize()              # Use defaults
                daemonize(working_dir)   # Set working directory
            """
            import os
            import sys
            
            # Check if we're on a Unix-like system
            if not hasattr(os, 'fork'):
                return EvaluationError("daemonize() is only supported on Unix-like systems")
            
            # Get optional working directory
            working_dir = None
            if len(args) > 0:
                if isinstance(args[0], String):
                    working_dir = args[0].value
                else:
                    return EvaluationError("daemonize() working_dir must be a string")
            
            try:
                # First fork
                pid = os.fork()
                if pid > 0:
                    # Parent process - exit
                    sys.exit(0)
            except OSError as e:
                return EvaluationError(f"First fork failed: {str(e)}")
            
            # Decouple from parent environment
            os.chdir(working_dir if working_dir else '/')
            os.setsid()
            os.umask(0)
            
            # Second fork to prevent acquiring a controlling terminal
            try:
                pid = os.fork()
                if pid > 0:
                    # Parent of second fork - exit
                    sys.exit(0)
            except OSError as e:
                return EvaluationError(f"Second fork failed: {str(e)}")
            
            # Redirect standard file descriptors to /dev/null
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Open /dev/null
            dev_null = os.open(os.devnull, os.O_RDWR)
            
            # Redirect stdin, stdout, stderr
            os.dup2(dev_null, sys.stdin.fileno())
            os.dup2(dev_null, sys.stdout.fileno())
            os.dup2(dev_null, sys.stderr.fileno())
            
            # Close the dev_null file descriptor
            if dev_null > 2:
                os.close(dev_null)
            
            return NULL
        
        def _watch_and_reload(*args):
            """
            Watch files for changes and reload modules automatically.
            Useful for development to see code changes without restarting.
            
            Usage:
                watch_and_reload([__file__])                    # Watch current file
                watch_and_reload([file1, file2])                # Watch multiple files
                watch_and_reload([__file__], 1.0)               # Custom check interval
                watch_and_reload([__file__], 1.0, my_callback)  # With callback
            
            Returns: Map with watch info
            """
            import os
            import time as time_module
            import threading
            
            if len(args) < 1:
                return EvaluationError("watch_and_reload() requires at least 1 argument (files)")
            
            # Parse arguments
            files_arg = args[0]
            check_interval = 1.0  # Default: check every second
            reload_callback = None
            
            if not isinstance(files_arg, List):
                return EvaluationError("watch_and_reload() files must be a list")
            
            if len(args) >= 2:
                interval_obj = args[1]
                if isinstance(interval_obj, (Integer, Float)):
                    check_interval = float(interval_obj.value)
                else:
                    return EvaluationError("watch_and_reload() interval must be a number")
            
            if len(args) >= 3:
                callback_obj = args[2]
                if isinstance(callback_obj, (Action, LambdaFunction)):
                    reload_callback = callback_obj
                else:
                    return EvaluationError("watch_and_reload() callback must be a function")
            
            # Extract file paths
            file_paths = []
            for file_obj in files_arg.elements:
                if isinstance(file_obj, String):
                    path = file_obj.value
                    if os.path.exists(path):
                        file_paths.append(path)
                    else:
                        return EvaluationError(f"File not found: {path}")
                else:
                    return EvaluationError("watch_and_reload() file paths must be strings")
            
            if not file_paths:
                return EvaluationError("No valid files to watch")
            
            # Get initial modification times
            file_mtimes = {}
            for path in file_paths:
                try:
                    file_mtimes[path] = os.path.getmtime(path)
                except OSError as e:
                    return EvaluationError(f"Cannot stat {path}: {str(e)}")
            
            reload_count = [0]  # Use list for closure mutability
            
            def watch_worker():
                """Background thread that watches for file changes"""
                while True:
                    time_module.sleep(check_interval)
                    
                    for path in file_paths:
                        try:
                            current_mtime = os.path.getmtime(path)
                            if current_mtime > file_mtimes[path]:
                                # File was modified!
                                print(f"\n🔄 File changed: {path}")
                                file_mtimes[path] = current_mtime
                                reload_count[0] += 1
                                
                                # Execute reload callback if provided
                                if reload_callback:
                                    try:
                                        result = self.apply_function(reload_callback, [String(path)])
                                        if is_error(result):
                                            print(f"⚠️  Reload callback error: {result.message}")
                                    except Exception as e:
                                        print(f"⚠️  Reload callback exception: {str(e)}")
                                else:
                                    print(f"   Reload #{reload_count[0]} - No auto-reload callback set")
                                    print(f"   Tip: Restart the program to see changes")
                        except OSError:
                            # File might have been deleted/renamed
                            pass
            
            # Start watch thread
            watch_thread = threading.Thread(target=watch_worker, daemon=True)
            watch_thread.start()
            
            print(f"👁️  Watching {len(file_paths)} file(s) for changes...")
            for path in file_paths:
                print(f"   - {path}")
            print(f"   Check interval: {check_interval}s")
            
            # Return watch info
            return Map({
                "files": files_arg,
                "interval": Float(check_interval),
                "active": BooleanObj(True)
            })
        
        def _get_module_name(*a):
            """
            Get the current module name (__MODULE__).
            
            Usage:
                name = get_module_name()
                print("Module: " + name)
            """
            env = getattr(self, '_current_env', None)
            if not env:
                return String("")
            
            module = env.get('__MODULE__')
            return module if module else String("")
        
        def _get_module_path(*a):
            """
            Get the current module file path (__file__).
            
            Usage:
                path = get_module_path()
                print("Path: " + path)
            """
            env = getattr(self, '_current_env', None)
            if not env:
                return String("")
            
            file_path = env.get('__file__')
            if not file_path:
                file_path = env.get('__FILE__')
            return file_path if file_path else String("")
        
        def _module_info(*a):
            """
            Get information about the current module.
            Returns a map with module metadata.
            
            Usage:
                info = module_info()
                print(info["name"])     # Module name
                print(info["file"])     # File path
                print(info["dir"])      # Directory
                print(info["package"])  # Package name
            """
            env = getattr(self, '_current_env', None)
            if not env:
                return Map({})
            
            result = {}
            
            # Get module variables
            for var_name in ['__MODULE__', '__file__', '__FILE__', '__DIR__', '__PACKAGE__']:
                val = env.get(var_name)
                if val:
                    key = var_name.strip('_').lower()
                    result[String(key)] = val
            
            return Map(result)
        
        def _list_imports(*a):
            """
            List all imported modules in the current environment.
            
            Usage:
                imports = list_imports()
                print(imports)  # ["math", "json", "./utils"]
            """
            env = getattr(self, '_current_env', None)
            if not env:
                return List([])
            
            # Collect all imported module names (this is a simplified version)
            # In a more complete implementation, we'd track imports explicitly
            imports = []
            
            # Look for common module indicators in the environment
            for name, value in env.store.items():
                # Skip special variables and builtins
                if name.startswith('__') or name in self.builtins:
                    continue
                # Check if it looks like an imported module
                if isinstance(value, Map) and len(value.pairs) > 3:
                    imports.append(String(name))
            
            return List(imports)
        
        def _get_exported_names(*a):
            """
            Get all exported variable names from the current module.
            
            Usage:
                exports = get_exported_names()
                print(exports)  # ["myFunction", "MY_CONSTANT", "MyClass"]
            """
            env = getattr(self, '_current_env', None)
            if not env:
                return List([])
            
            exports = []
            
            # Get all user-defined names (skip special variables and builtins)
            for name in env.store.keys():
                if not name.startswith('__') and name not in self.builtins:
                    exports.append(String(name))
            
            return List(exports)
        
        # Register the builtins
        self.builtins.update({
            "run": Builtin(_run, "run"),
            "execute": Builtin(_execute, "execute"),
            "is_main": Builtin(_is_main, "is_main"),
            "exit_program": Builtin(_exit_program, "exit_program"),
            "on_start": Builtin(_on_start, "on_start"),
            "on_exit": Builtin(_on_exit, "on_exit"),
            "signal_handler": Builtin(_signal_handler, "signal_handler"),
            "schedule": Builtin(_schedule, "schedule"),
            "sleep": Builtin(_sleep, "sleep"),
            "daemonize": Builtin(_daemonize, "daemonize"),
            "watch_and_reload": Builtin(_watch_and_reload, "watch_and_reload"),
            "get_module_name": Builtin(_get_module_name, "get_module_name"),
            "get_module_path": Builtin(_get_module_path, "get_module_path"),
            "module_info": Builtin(_module_info, "module_info"),
            "list_imports": Builtin(_list_imports, "list_imports"),
            "get_exported_names": Builtin(_get_exported_names, "get_exported_names"),
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
        
        def _require(*a):
            """Assert a condition is true, revert transaction if false: require(condition, error_message?)"""
            if len(a) < 1 or len(a) > 2:
                return EvaluationError("require() takes 1 or 2 arguments: condition, error_message?")
            
            condition = a[0]
            
            # Evaluate condition as boolean
            is_true = False
            if isinstance(condition, BooleanObj):
                is_true = condition.value
            elif isinstance(condition, Integer):
                is_true = condition.value != 0
            elif isinstance(condition, String):
                is_true = len(condition.value) > 0
            elif condition == NULL:
                is_true = False
            else:
                is_true = True  # Non-null objects are truthy
            
            if not is_true:
                # Condition failed - revert transaction
                error_msg = "Requirement failed"
                if len(a) >= 2:
                    msg_obj = a[1]
                    error_msg = msg_obj.value if isinstance(msg_obj, String) else str(msg_obj)
                
                # Return an EvaluationError which will stop execution
                return EvaluationError(f"require() failed: {error_msg}")
            
            # Condition passed - return true
            return TRUE
        
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
            "require": Builtin(_require, "require"),
        })
        
        # Register main entry point and event loop builtins
        self._register_main_entry_point_builtins()
