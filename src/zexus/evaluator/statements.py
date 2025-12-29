# src/zexus/evaluator/statements.py
import os
import sys

from ..zexus_ast import (
    Program, ExpressionStatement, BlockStatement, ReturnStatement, ContinueStatement, BreakStatement, ThrowStatement, LetStatement, ConstStatement,
    ActionStatement, FunctionStatement, IfStatement, WhileStatement, ForEachStatement,
    TryCatchStatement, UseStatement, FromStatement, ExportStatement,
    ContractStatement, EntityStatement, VerifyStatement, ProtectStatement,
    SealStatement, MiddlewareStatement, AuthStatement, ThrottleStatement, CacheStatement,
    ComponentStatement, ThemeStatement, DebugStatement, ExternalDeclaration, AssignmentExpression,
    PrintStatement, ScreenStatement, EmbeddedCodeStatement, ExactlyStatement,
    Identifier, PropertyAccessExpression, RestrictStatement, SandboxStatement, TrailStatement,
    NativeStatement, GCStatement, InlineStatement, BufferStatement, SIMDStatement,
    DeferStatement, PatternStatement, PatternCase, EnumStatement, EnumMember, StreamStatement, WatchStatement,
    CapabilityStatement, GrantStatement, RevokeStatement, ValidateStatement, SanitizeStatement, ImmutableStatement,
    InterfaceStatement, TypeAliasStatement, ModuleStatement, PackageStatement, UsingStatement,
    ChannelStatement, SendStatement, ReceiveStatement, AtomicStatement,
    # Blockchain statements and expressions
    LedgerStatement, StateStatement, RequireStatement, RevertStatement, LimitStatement,
    TXExpression, HashExpression, SignatureExpression, VerifySignatureExpression, GasExpression
)
from ..object import (
    Environment, Integer, Float, String, Boolean as Boolean, ReturnValue,
    Action, List, Map, EvaluationError, EntityDefinition, EmbeddedCode, Builtin,
    start_collecting_dependencies, stop_collecting_dependencies
)
from ..security import (
    SealedObject, SmartContract, VerifyWrapper, VerificationCheck, get_security_context,
    ProtectionPolicy, Middleware, AuthConfig, RateLimiter, CachePolicy
)
from .utils import is_error, debug_log, EVAL_SUMMARY, NULL, TRUE, FALSE, _resolve_awaitable, _zexus_to_python, _python_to_zexus, is_truthy

# Break exception for loop control flow
class BreakException:
    """Exception raised when break statement is encountered in a loop."""
    def __repr__(self):
        return "BreakException()"

class StatementEvaluatorMixin:
    """Handles evaluation of statements, flow control, module loading, and security features."""
    
    def ceval_program(self, statements, env):
        debug_log("eval_program", f"Processing {len(statements)} statements")
        
        # Track current environment for builtin functions
        self._current_env = env
        
        try:
            EVAL_SUMMARY['parsed_statements'] = max(EVAL_SUMMARY.get('parsed_statements', 0), len(statements))
        except Exception: 
            pass
        
        result = NULL
        try:
            for i, stmt in enumerate(statements):
                debug_log(f"  Statement {i+1}", type(stmt).__name__)
                res = self.eval_node(stmt, env)
                res = _resolve_awaitable(res)
                EVAL_SUMMARY['evaluated_statements'] += 1
                
                if isinstance(res, ReturnValue): 
                    debug_log("  ReturnValue encountered", res.value)
                    # Execute deferred cleanup before returning
                    self._execute_deferred_cleanup(env, [])
                    return res.value
                if is_error(res):
                    debug_log("  Error encountered", res)
                    try:
                        EVAL_SUMMARY['errors'] += 1
                    except Exception:
                        pass
                    
                    # Check if continue_on_error mode is enabled
                    if self.continue_on_error:
                        # Log the error and continue execution
                        error_msg = str(res)
                        self.error_log.append(error_msg)
                        print(f"[ERROR] {error_msg}")
                        debug_log("  Continuing after error", "continue_on_error=True")
                        result = NULL  # Continue with null result
                        continue
                    else:
                        # Execute deferred cleanup even on error
                        self._execute_deferred_cleanup(env, [])
                        return res
                result = res
            
            debug_log("eval_program completed", result)
            return result
        finally:
            # CRITICAL: Execute all deferred cleanup at program exit
            self._execute_deferred_cleanup(env, [])
    
    def eval_block_statement(self, block, env, stack_trace=None):
        debug_log("eval_block_statement", f"len={len(block.statements)}")
        
        try:
            EVAL_SUMMARY['max_statements_in_block'] = max(EVAL_SUMMARY.get('max_statements_in_block', 0), len(block.statements))
        except Exception:
            pass
        
        if stack_trace is None:
            stack_trace = []
        
        result = NULL
        try:
            for stmt in block.statements:
                res = self.eval_node(stmt, env, stack_trace)
                res = _resolve_awaitable(res)
                EVAL_SUMMARY['evaluated_statements'] += 1
                
                if isinstance(res, (ReturnValue, BreakException, EvaluationError)):
                    debug_log("  Block interrupted", res)
                    if is_error(res):
                        try:
                            EVAL_SUMMARY['errors'] += 1
                        except Exception:
                            pass
                        
                        # Check if continue_on_error mode is enabled
                        if self.continue_on_error:
                            # Log the error and continue execution
                            error_msg = str(res)
                            self.error_log.append(error_msg)
                            print(f"[ERROR] {error_msg}")
                            debug_log("  Continuing after error in block", "continue_on_error=True")
                            result = NULL  # Continue with null result
                            continue
                    
                    # Execute deferred cleanup before returning
                    self._execute_deferred_cleanup(env, stack_trace)
                    # Restore stdout before returning
                    self._restore_stdout(env)
                    return res
                result = res
            
            debug_log("  Block completed", result)
            return result
        finally:
            # Always execute deferred cleanup when block exits (normal or error)
            self._execute_deferred_cleanup(env, stack_trace)
            # Restore stdout to previous state (scope-aware)
            self._restore_stdout(env)
    
    def eval_expression_statement(self, node, env, stack_trace):
        # Debug: Check if expression is being evaluated
        if hasattr(node.expression, 'function') and hasattr(node.expression.function, 'value'):
            func_name = node.expression.function.value
            if func_name in ['persist_set', 'persist_get']:
                print(f"[EVAL_EXPR_STMT] Evaluating {func_name} call", flush=True)
        result = self.eval_node(node.expression, env, stack_trace)
        if hasattr(node.expression, 'function') and hasattr(node.expression.function, 'value'):
            func_name = node.expression.function.value
            if func_name in ['persist_set', 'persist_get']:
                print(f"[EVAL_EXPR_STMT] Result from {func_name}: {result}", flush=True)
        return result
    
    # === VARIABLE & CONTROL FLOW ===
    
    def eval_let_statement(self, node, env, stack_trace):
        debug_log("eval_let_statement", f"let {node.name.value}")
        
        # FIXED: Evaluate value FIRST to prevent recursion issues
        value = self.eval_node(node.value, env, stack_trace)
        if is_error(value): 
            return value
        
        # Type annotation validation
        if node.type_annotation:
            type_name = node.type_annotation.value
            debug_log("eval_let_statement", f"Validating type: {type_name}")
            
            # Resolve type alias
            type_alias = env.get(type_name)
            if type_alias and hasattr(type_alias, '__class__') and type_alias.__class__.__name__ == 'TypeAlias':
                # Get the base type
                base_type = type_alias.base_type
                debug_log("eval_let_statement", f"Resolved type alias: {type_name} -> {base_type}")
                
                # Validate value type matches the base type
                if not self._validate_type(value, base_type):
                    return EvaluationError(f"Type mismatch: cannot assign {type(value).__name__} to {type_name} (expected {base_type})")
            else:
                # Direct type validation (for built-in types)
                if not self._validate_type(value, type_name):
                    return EvaluationError(f"Type mismatch: cannot assign {type(value).__name__} to {type_name}")
        
        env.set(node.name.value, value)
        return NULL
    
    def _validate_type(self, value, expected_type):
        """Validate that a value matches an expected type"""
        # Map Zexus types to Python types
        type_map = {
            'int': ('Integer',),
            'integer': ('Integer',),
            'str': ('String',),
            'string': ('String',),
            'bool': ('Boolean',),
            'boolean': ('Boolean',),
            'float': ('Float', 'Integer'),  # int can be used as float
            'array': ('Array',),
            'list': ('Array',),
            'map': ('Map',),
            'dict': ('Map',),
            'null': ('Null',),
        }
        
        value_type = type(value).__name__
        expected_types = type_map.get(expected_type.lower(), (expected_type,))
        
        return value_type in expected_types
    
    def eval_const_statement(self, node, env, stack_trace):
        debug_log("eval_const_statement", f"const {node.name.value}")
        
        # Evaluate value FIRST
        value = self.eval_node(node.value, env, stack_trace)
        if is_error(value): 
            return value
        
        # Set as const in environment
        env.set_const(node.name.value, value)
        return NULL
    
    def eval_data_statement(self, node, env, stack_trace):
        """Evaluate data statement - creates a production-grade dataclass constructor
        
        data User {
            name: string,
            email: string = "default",
            age: number require age >= 0
        }
        
        data Box<T> {
            value: T
        }
        
        Creates a User() or Box<T>() constructor function with:
        - Type validation
        - Constraint validation
        - Generic type substitution
        - Auto-generated methods: toString(), toJSON(), clone(), hash(), verify()
        - Static methods: fromJSON()
        - Immutability support
        - Verification support
        """
        from ..object import Map, String, Integer, Boolean, List, NULL, EvaluationError, Builtin
        from ..environment import Environment
        import json
        import hashlib
        
        debug_log("eval_data_statement", f"data {node.name.value}")
        
        type_name = node.name.value
        fields = node.fields
        modifiers = node.modifiers or []
        parent_type = node.parent
        decorators = node.decorators or []
        type_params = node.type_params or []
        
        # Check modifiers
        is_immutable = "immutable" in modifiers
        is_verified = "verified" in modifiers
        is_validated = "validated" in decorators
        
        debug_log(f"  Fields: {len(fields)}, Immutable: {is_immutable}, Verified: {is_verified}, Validated: {is_validated}")
        
        if type_params:
            debug_log(f"  Generic type parameters: {type_params}")
        
        # If this is a generic type, we need to create a factory that produces specialized constructors
        if type_params:
            # Store the generic template
            generic_template = {
                'type_name': type_name,
                'fields': fields,
                'modifiers': modifiers,
                'parent_type': parent_type,
                'decorators': decorators,
                'type_params': type_params,
                'env': env,
                'stack_trace': stack_trace,
                'evaluator': self
            }
            
            # Create a Builtin that stores the template
            template_constructor = Builtin(lambda *args: EvaluationError(
                f"Generic type '{type_name}' requires type arguments. Use {type_name}<Type>(...)"
            ))
            template_constructor.is_generic = True
            template_constructor.generic_template = generic_template
            
            # Register the generic template
            env.set(type_name, template_constructor)
            return NULL
        
        # Check modifiers
        is_immutable = "immutable" in modifiers
        is_verified = "verified" in modifiers
        is_validated = "validated" in decorators
        
        debug_log(f"  Fields: {len(fields)}, Immutable: {is_immutable}, Verified: {is_verified}, Validated: {is_validated}")
        
        # If there's a parent, get parent fields
        parent_fields = []
        parent_constructor = None
        if parent_type:
            debug_log(f"  Inheritance: {type_name} extends {parent_type}")
            parent_constructor = env.get(parent_type)
            if parent_constructor is None:
                return EvaluationError(f"Parent type '{parent_type}' not found")
            
            # Extract parent fields from the parent's constructor metadata
            if hasattr(parent_constructor, 'dataclass_fields'):
                parent_fields = parent_constructor.dataclass_fields
                debug_log(f"  Parent has {len(parent_fields)} fields")
        
        # Combine parent fields + child fields
        all_fields = parent_fields + fields
        
        # Store reference to self and env for closures
        evaluator_self = self
        parent_env = env
        
        # Create constructor function
        def dataclass_constructor(*args):
            """Production-grade dataclass constructor with full validation"""
            
            # Create instance as a Map with String keys
            instance = Map({})
            instance.pairs = {}
            
            # Set type metadata
            instance.pairs[String("__type__")] = String(type_name)
            instance.pairs[String("__immutable__")] = Boolean(is_immutable)
            instance.pairs[String("__verified__")] = Boolean(is_verified)
            
            # Process each field with validation (parent fields first, then child fields)
            arg_index = 0
            for field in all_fields:
                field_name = field.name
                
                # Skip computed properties and methods - they'll be added later
                if field.computed or field.method_body is not None:
                    continue
                
                field_value = NULL
                
                # Get value from arguments or default
                if arg_index < len(args):
                    field_value = args[arg_index]
                    arg_index += 1
                elif field.default_value is not None:
                    # Evaluate default value in parent environment
                    field_value = evaluator_self.eval_node(field.default_value, parent_env, stack_trace)
                    if is_error(field_value):
                        return field_value
                
                # Type validation
                if field.field_type and field_value != NULL:
                    expected_type = field.field_type
                    
                    type_map = {
                        "string": String,
                        "number": Integer,
                        "bool": Boolean,
                        "array": List,
                        "map": Map
                    }
                    
                    if expected_type in type_map:
                        expected_class = type_map[expected_type]
                        if not isinstance(field_value, expected_class):
                            actual_type = type(field_value).__name__
                            return EvaluationError(
                                f"Type mismatch for field '{field_name}': expected {expected_type}, got {actual_type}"
                            )
                
                # Constraint validation (require clause)
                if field.constraint and field_value != NULL:
                    # Evaluate constraint with field value in scope
                    temp_env = Environment(outer=parent_env)
                    temp_env.set(field_name, field_value)
                    
                    constraint_result = evaluator_self.eval_node(field.constraint, temp_env, stack_trace)
                    if is_error(constraint_result):
                        return constraint_result
                    
                    # Check if constraint is truthy
                    is_valid = False
                    if hasattr(constraint_result, 'value'):
                        is_valid = bool(constraint_result.value)
                    elif isinstance(constraint_result, Boolean):
                        is_valid = constraint_result.value
                    else:
                        is_valid = bool(constraint_result)
                    
                    if not is_valid:
                        return EvaluationError(
                            f"Validation failed for field '{field_name}': constraint not satisfied"
                        )
                
                # Set field value
                instance.pairs[String(field_name)] = field_value
            
            # Auto-generated methods
            
            # toString() method
            def to_string_method(*args):
                parts = []
                for field in fields:
                    fname = field.name
                    fkey = String(fname)
                    if fkey in instance.pairs:
                        fval = instance.pairs[fkey]
                        if hasattr(fval, 'inspect'):
                            val_str = fval.inspect()
                        elif hasattr(fval, 'value'):
                            val_str = repr(fval.value)
                        else:
                            val_str = str(fval)
                        parts.append(f'{fname}={val_str}')
                return String(f"{type_name}({', '.join(parts)})")
            
            instance.pairs[String("toString")] = Builtin(to_string_method)
            
            # toJSON() method
            def to_json_method(*args):
                obj = {}
                for field in fields:
                    fname = field.name
                    fkey = String(fname)
                    if fkey in instance.pairs:
                        fval = instance.pairs[fkey]
                        if hasattr(fval, 'value'):
                            obj[fname] = fval.value
                        elif fval == NULL:
                            obj[fname] = None
                        else:
                            obj[fname] = str(fval)
                return String(json.dumps(obj))
            
            instance.pairs[String("toJSON")] = Builtin(to_json_method)
            
            # clone() method
            def clone_method(*args):
                clone_args = []
                for field in fields:
                    fname = field.name
                    fkey = String(fname)
                    if fkey in instance.pairs:
                        clone_args.append(instance.pairs[fkey])
                    else:
                        clone_args.append(NULL)
                return dataclass_constructor(*clone_args)
            
            instance.pairs[String("clone")] = Builtin(clone_method)
            
            # equals() method
            def equals_method(*args):
                if len(args) == 0:
                    return Boolean(False)
                other = args[0]
                if not isinstance(other, Map):
                    return Boolean(False)
                
                # Check type match
                other_type = other.pairs.get(String("__type__"))
                if not other_type or other_type.value != type_name:
                    return Boolean(False)
                
                # Compare all fields
                for field in fields:
                    fname = field.name
                    fkey = String(fname)
                    if fkey not in instance.pairs or fkey not in other.pairs:
                        return Boolean(False)
                    
                    val1 = instance.pairs[fkey]
                    val2 = other.pairs[fkey]
                    
                    # Compare values
                    if hasattr(val1, 'value') and hasattr(val2, 'value'):
                        if val1.value != val2.value:
                            return Boolean(False)
                    elif val1 != val2:
                        return Boolean(False)
                
                return Boolean(True)
            
            instance.pairs[String("equals")] = Builtin(equals_method)
            
            # Verified type methods
            if is_verified:
                # hash() method - cryptographic hash of all fields
                def hash_method(*args):
                    json_str = to_json_method()
                    hash_val = hashlib.sha256(json_str.value.encode()).hexdigest()
                    return String(hash_val)
                
                instance.pairs[String("hash")] = Builtin(hash_method)
                
                # verify() method - re-validate all constraints
                def verify_method(*args):
                    for field in fields:
                        if field.constraint:
                            fname = field.name
                            fkey = String(fname)
                            if fkey in instance.pairs:
                                fval = instance.pairs[fkey]
                                temp_env = Environment(outer=parent_env)
                                temp_env.set(fname, fval)
                                result = evaluator_self.eval_node(field.constraint, temp_env, stack_trace)
                                if is_error(result):
                                    return Boolean(False)
                                if hasattr(result, 'value') and not result.value:
                                    return Boolean(False)
                    return Boolean(True)
                
                instance.pairs[String("verify")] = Builtin(verify_method)
            
            # Add custom methods defined in the dataclass (parent + child)
            for field in all_fields:
                if field.method_body is not None:
                    # Check if this is an operator overload
                    is_operator = hasattr(field, 'operator') and field.operator is not None
                    
                    # Create a closure for the method that has access to instance fields
                    def make_method(method_body, method_params, method_name, decorators, is_op=False):
                        def custom_method(*args):
                            # Create method environment with instance fields
                            from ..environment import Environment
                            method_env = Environment(outer=parent_env)
                            
                            # Bind 'this' to the instance
                            method_env.set('this', instance)
                            
                            # Bind parameters to arguments
                            for i, param in enumerate(method_params):
                                if i < len(args):
                                    method_env.set(param, args[i])
                                else:
                                    method_env.set(param, NULL)
                            
                            # Apply @logged decorator (operators don't get logged)
                            if "logged" in decorators and not is_op:
                                arg_str = ", ".join([str(arg.value if hasattr(arg, 'value') else arg) for arg in args])
                                print(f"📝 Calling {method_name}({arg_str})")
                            
                            # Execute method body
                            from ..object import ReturnValue
                            from .. import zexus_ast
                            result = NULL
                            
                            # Handle both BlockStatement and list of statements
                            statements = method_body
                            if isinstance(method_body, zexus_ast.BlockStatement):
                                statements = method_body.statements
                            
                            for stmt in statements:
                                result = evaluator_self.eval_node(stmt, method_env, stack_trace)
                                if is_error(result):
                                    return result
                                # Handle return statements
                                if isinstance(result, ReturnValue):
                                    return result.value
                            
                            # Apply @logged decorator (return value)
                            if "logged" in decorators and not is_op:
                                result_str = str(result.value if hasattr(result, 'value') else result)
                                print(f"📝 {method_name} returned: {result_str}")
                            
                            return result
                        
                        return custom_method
                    
                    # Create the method function
                    method_func = make_method(field.method_body, field.method_params, field.name, field.decorators, is_operator)
                    
                    # Apply @cached decorator if present
                    if "cached" in field.decorators:
                        cache = {}
                        original_func = method_func
                        
                        def cached_method(*args):
                            # Create cache key from arguments
                            cache_key = tuple(arg.value if hasattr(arg, 'value') else str(arg) for arg in args)
                            if cache_key in cache:
                                return cache[cache_key]
                            result = original_func(*args)
                            cache[cache_key] = result
                            return result
                        
                        method_func = cached_method
                    
                    # Store the method/operator with appropriate key
                    if is_operator:
                        # Store with __operator_{symbol}__ key for operator overloading
                        operator_key = f"__operator_{field.operator}__"
                        instance.pairs[String(operator_key)] = Builtin(
                            method_func,
                            name=operator_key
                        )
                    else:
                        # Regular method
                        instance.pairs[String(field.name)] = Builtin(
                            method_func,
                            name=field.name
                        )
            
            # Store computed property definitions for auto-calling on access
            computed_props = {}
            for field in all_fields:
                if field.computed:
                    computed_props[field.name] = field.computed
            
            # Store computed property metadata
            if computed_props:
                instance.pairs[String("__computed__")] = computed_props
            
            return instance
        
        # Create static fromJSON method
        def from_json_static(*args):
            """Static method to deserialize from JSON"""
            if len(args) == 0:
                return EvaluationError("fromJSON requires a JSON string argument")
            
            json_str = args[0]
            if not isinstance(json_str, String):
                return EvaluationError("fromJSON expects a string argument")
            
            try:
                data = json.loads(json_str.value)
                constructor_args = []
                
                for field in fields:
                    fname = field.name
                    if fname in data:
                        val = data[fname]
                        # Convert JSON values to Zexus objects
                        if isinstance(val, str):
                            constructor_args.append(String(val))
                        elif isinstance(val, bool):
                            constructor_args.append(Boolean(val))
                        elif isinstance(val, (int, float)):
                            constructor_args.append(Integer(int(val)))
                        elif isinstance(val, list):
                            # Convert to List
                            zx_list = List()
                            zx_list.elements = [String(str(item)) for item in val]
                            constructor_args.append(zx_list)
                        elif isinstance(val, dict):
                            # Convert to Map
                            zx_map = Map()
                            zx_map.pairs = {String(k): String(str(v)) for k, v in val.items()}
                            constructor_args.append(zx_map)
                        elif val is None:
                            constructor_args.append(NULL)
                        else:
                            constructor_args.append(String(str(val)))
                    else:
                        constructor_args.append(NULL)
                
                return dataclass_constructor(*constructor_args)
            
            except json.JSONDecodeError as e:
                return EvaluationError(f"Invalid JSON: {str(e)}")
            except Exception as e:
                return EvaluationError(f"Error deserializing JSON: {str(e)}")
        
        # Create static default() method
        def default_static(*args):
            """Static method to create instance with all default values"""
            default_args = []
            
            for field in fields:
                if field.default_value is not None:
                    # Evaluate default value in parent environment
                    default_val = evaluator_self.eval_node(field.default_value, parent_env, stack_trace)
                    if is_error(default_val):
                        return default_val
                    default_args.append(default_val)
                else:
                    # No default - use NULL
                    default_args.append(NULL)
            
            return dataclass_constructor(*default_args)
        
        # Register constructor as a Builtin with static methods
        constructor = Builtin(dataclass_constructor)
        
        # Store fields for inheritance (child classes need access to parent fields)
        constructor.dataclass_fields = all_fields
        
        # Add static methods as properties on the constructor
        # (We'll store them in a special way that the evaluator can access)
        constructor.static_methods = {
            "fromJSON": Builtin(from_json_static),
            "default": Builtin(default_static)
        }
        
        # Register constructor in environment as const
        # For specialized generics (e.g., Box<number>), don't fail if already registered
        try:
            env.set_const(type_name, constructor)
        except ValueError as e:
            # If it's a specialized generic that's already registered, just return the existing one
            if '<' in type_name and '>' in type_name:
                debug_log(f"  ℹ️  Specialized generic already registered: {type_name}")
                return NULL
            else:
                # Re-raise for non-generic types
                raise e
        
        debug_log(f"  ✅ Registered production-grade dataclass: {type_name}")
        return NULL
    
    def eval_return_statement(self, node, env, stack_trace):
        val = self.eval_node(node.return_value, env, stack_trace)
        if is_error(val): 
            return val
        return ReturnValue(val)
    
    def eval_continue_statement(self, node, env, stack_trace):
        """Enable continue-on-error mode for the evaluator."""
        debug_log("eval_continue_statement", "Enabling error recovery mode")
        self.continue_on_error = True
        return NULL
    
    def eval_break_statement(self, node, env, stack_trace):
        """Return BreakException to signal loop exit."""
        debug_log("eval_break_statement", "Breaking out of loop")
        return BreakException()
    
    def eval_throw_statement(self, node, env, stack_trace):
        """Throw an error/exception."""
        debug_log("eval_throw_statement", "Throwing error")
        # Evaluate error message
        message = self.eval_node(node.message, env, stack_trace)
        if is_error(message):
            return message
        # Convert to string
        error_msg = str(message) if hasattr(message, '__str__') else "Unknown error"
        return EvaluationError(error_msg, stack_trace=stack_trace)
    
    def eval_assignment_expression(self, node, env, stack_trace):
        # Support assigning to identifiers or property access targets
        from ..object import EvaluationError, NULL

        # If target is a property access expression
        if isinstance(node.name, PropertyAccessExpression):
            # Evaluate object and property
            obj = self.eval_node(node.name.object, env, stack_trace)
            if is_error(obj):
                return obj

            # Safely extract property key
            if hasattr(node.name.property, 'value'):
                prop_key = node.name.property.value
            else:
                # Evaluate property expression
                prop_result = self.eval_node(node.name.property, env, stack_trace)
                if is_error(prop_result):
                    return prop_result
                prop_key = prop_result.value if hasattr(prop_result, 'value') else str(prop_result)

            # Evaluate value first
            value = self.eval_node(node.value, env, stack_trace)
            if is_error(value):
                return value

            # Check for seal on property
            try:
                if isinstance(obj, Map):
                    existing = obj.pairs.get(prop_key)
                    if existing is not None and existing.__class__.__name__ == 'SealedObject':
                        return EvaluationError(f"Cannot modify sealed property: {prop_key}")
                elif hasattr(obj, 'get') and hasattr(obj, 'set'):
                    existing = obj.get(prop_key)
                    if existing is not None and getattr(existing, '__class__', None) and existing.__class__.__name__ == 'SealedObject':
                        return EvaluationError(f"Cannot modify sealed property: {prop_key}")
            except Exception:
                pass

            # Enforcement: consult security restrictions for writes
            try:
                ctx = get_security_context()
                target = f"{getattr(node.name.object, 'value', str(node.name.object))}.{prop_key}"
                restriction = ctx.get_restriction(target)
            except Exception:
                restriction = None

            if restriction:
                rule = restriction.get('restriction')
                if rule == 'read-only':
                    return EvaluationError(f"Write prohibited by restriction: {target}")
                if rule == 'admin-only':
                    is_admin = bool(env.get('__is_admin__')) if env and hasattr(env, 'get') else False
                    if not is_admin:
                        return EvaluationError('Admin privileges required to modify this field')

            # Perform set
            try:
                if isinstance(obj, Map):
                    obj.pairs[prop_key] = value
                    return value
                elif hasattr(obj, 'set'):
                    obj.set(prop_key, value)
                    return value
            except Exception as e:
                return EvaluationError(str(e))

            return EvaluationError('Assignment to property failed')

        # Otherwise it's an identifier assignment
        if isinstance(node.name, Identifier):
            name = node.name.value
            target_obj = env.get(name)
            if isinstance(target_obj, SealedObject):
                return EvaluationError(f"Cannot assign to sealed object: {name}")

            value = self.eval_node(node.value, env, stack_trace)
            if is_error(value):
                return value

            try:
                env.assign(name, value)
            except ValueError as e:
                return EvaluationError(str(e))
            return value

        return EvaluationError('Invalid assignment target')
    
    def eval_try_catch_statement(self, node, env, stack_trace):
        debug_log("eval_try_catch", f"error_var: {node.error_variable.value if node.error_variable else 'error'}")
        
        try:
            result = self.eval_node(node.try_block, env, stack_trace)
            if is_error(result):
                catch_env = Environment(outer=env)
                var_name = node.error_variable.value if node.error_variable else "error"
                catch_env.set(var_name, String(str(result)))
                return self.eval_node(node.catch_block, catch_env, stack_trace)
            return result
        except Exception as e:
            catch_env = Environment(outer=env)
            var_name = node.error_variable.value if node.error_variable else "error"
            catch_env.set(var_name, String(str(e)))
            return self.eval_node(node.catch_block, catch_env, stack_trace)
    
    def eval_if_statement(self, node, env, stack_trace):
        cond = self.eval_node(node.condition, env, stack_trace)
        if is_error(cond): 
            return cond
        
        if is_truthy(cond):
            return self.eval_node(node.consequence, env, stack_trace)
        
        # Check elif conditions
        if hasattr(node, 'elif_parts') and node.elif_parts:
            for elif_condition, elif_consequence in node.elif_parts:
                elif_cond = self.eval_node(elif_condition, env, stack_trace)
                if is_error(elif_cond):
                    return elif_cond
                if is_truthy(elif_cond):
                    return self.eval_node(elif_consequence, env, stack_trace)
        
        # Check else clause
        if node.alternative:
            return self.eval_node(node.alternative, env, stack_trace)
        
        return NULL
    
    def eval_while_statement(self, node, env, stack_trace):
        result = NULL
        while True:
            cond = self.eval_node(node.condition, env, stack_trace)
            if is_error(cond): 
                return cond
            if not is_truthy(cond): 
                break
            
            result = self.eval_node(node.body, env, stack_trace)
            if isinstance(result, ReturnValue):
                return result
            if isinstance(result, BreakException):
                # Break out of loop, return NULL to continue execution in block
                return NULL
            if isinstance(result, EvaluationError):
                return result
        
        return result
    
    def eval_foreach_statement(self, node, env, stack_trace):
        iterable = self.eval_node(node.iterable, env, stack_trace)
        if is_error(iterable): 
            return iterable
        
        if not isinstance(iterable, List):
            return EvaluationError("ForEach expects List")
        
        result = NULL
        for item in iterable.elements:
            env.set(node.item.value, item)
            result = self.eval_node(node.body, env, stack_trace)
            if isinstance(result, ReturnValue):
                return result
            if isinstance(result, BreakException):
                # Break out of loop, return NULL to continue execution in block
                return NULL
            if isinstance(result, EvaluationError):
                return result
        
        return result
    
    def eval_watch_statement(self, node, env, stack_trace):
        # 1. Start collecting dependencies
        start_collecting_dependencies()
        
        # 2. Evaluate the watched expression or block
        if node.watched_expr:
            # Explicit watch: watch expr => block
            # Evaluate expression to capture dependencies
            res = self.eval_node(node.watched_expr, env, stack_trace)
            if is_error(res):
                stop_collecting_dependencies()
                return res
        else:
            # Implicit watch: watch block
            # Evaluate block to capture dependencies AND execute it
            res = self.eval_node(node.reaction, env, stack_trace)
            if is_error(res):
                stop_collecting_dependencies()
                return res
                
        # 3. Stop collecting and get dependencies
        deps = stop_collecting_dependencies()
        
        # 4. Define the reaction callback WITH GUARD against infinite recursion
        executing = [False]  # Mutable flag to track execution state
        def reaction_callback(new_val):
            if executing[0]:
                # Already executing, skip to prevent infinite loop
                return
            executing[0] = True
            try:
                # Re-evaluate the reaction block WITHOUT collecting dependencies
                result = self.eval_node(node.reaction, env, [])
                # Check for errors but don't propagate them (watchers shouldn't crash the program)
                if is_error(result):
                    pass  # Silently ignore watcher errors to prevent cascading failures
            except Exception as e:
                pass  # Silently ignore exceptions in watchers
            finally:
                executing[0] = False
            
        # 5. Register callback for each dependency
        for dep_env, name in deps:
            dep_env.add_watcher(name, reaction_callback)
            
        return NULL

    def eval_log_statement(self, node, env, stack_trace):
        """
        Evaluates a LOG statement: log > filepath
        Redirects subsequent print output to the specified file.
        Output is automatically restored when the current block exits.
        """
        import sys
        import os
        
        # 1. Evaluate the filepath expression
        filepath_obj = self.eval_node(node.filepath, env, stack_trace)
        if is_error(filepath_obj):
            return filepath_obj
        
        # 2. Convert to string
        if hasattr(filepath_obj, 'value'):
            filepath = str(filepath_obj.value)
        else:
            filepath = str(filepath_obj)
        
        # 3. Normalize path (handle relative paths relative to CWD)
        if not os.path.isabs(filepath):
            filepath = os.path.join(os.getcwd(), filepath)
        
        # 4. Open file for writing (append mode)
        try:
            log_file = open(filepath, 'a')
        except Exception as e:
            return new_error(f"Cannot open log file '{filepath}': {e}", stack_trace)
        
        # 5. Save current stdout state for restoration
        if not hasattr(env, '_stdout_stack'):
            env._stdout_stack = []
        env._stdout_stack.append(sys.stdout)
        
        # 6. Redirect stdout to this file
        sys.stdout = log_file
        
        # 7. Store the file handle for cleanup
        if not hasattr(env, '_log_files'):
            env._log_files = []
        env._log_files.append(log_file)
        
        return NULL

    def _restore_stdout(self, env):
        """Restore stdout to previous state and close log file (scope-aware)"""
        import sys
        
        # Restore stdout if we have a saved state
        if hasattr(env, '_stdout_stack') and env._stdout_stack:
            previous_stdout = env._stdout_stack.pop()
            
            # Close current log file if it's a file object
            if hasattr(sys.stdout, 'close') and sys.stdout != sys.__stdout__:
                try:
                    sys.stdout.flush()
                    sys.stdout.close()
                except Exception:
                    pass
            
            # Restore previous stdout
            sys.stdout = previous_stdout
            
            # Remove from log files list
            if hasattr(env, '_log_files') and env._log_files:
                env._log_files.pop()

    # === MODULE LOADING (FULL LOGIC) ===
    
    def _check_import_permission(self, val, importer):
        """Helper to check if a file is allowed to import a specific value."""
        allowed = getattr(val, '_allowed_files', [])
        if not allowed: 
            return True
        
        try:
            importer_norm = os.path.normpath(os.path.abspath(importer))
            for a in allowed:
                a_norm = os.path.normpath(os.path.abspath(a))
                if importer_norm == a_norm: 
                    return True
                if a in importer: 
                    return True
        except Exception:
            return False
        
        return False
    
    def eval_use_statement(self, node, env, stack_trace):
        from ..module_cache import get_cached_module, cache_module, get_module_candidates, normalize_path, invalidate_module
        from ..builtin_modules import is_builtin_module, get_builtin_module
        from ..stdlib_integration import is_stdlib_module, get_stdlib_module
        
        # 1. Determine File Path
        file_path_attr = getattr(node, 'file_path', None) or getattr(node, 'embedded_ref', None)
        file_path = file_path_attr.value if hasattr(file_path_attr, 'value') else file_path_attr
        if not file_path: 
            return EvaluationError("use: missing file path")
        
        debug_log("  UseStatement loading", file_path)
        
        # 1a. Check if this is a stdlib module (fs, http, json, datetime, crypto, blockchain)
        if is_stdlib_module(file_path):
            debug_log(f"  Loading stdlib module: {file_path}")
            try:
                module_env = get_stdlib_module(file_path, self)
                if module_env:
                    # Handle named imports: use {read_file, write_file} from "stdlib/fs"
                    is_named_import = getattr(node, 'is_named_import', False)
                    names = getattr(node, 'names', [])
                    alias = getattr(node, 'alias', None)
                    
                    if is_named_import and names:
                        # Import specific functions
                        for name_node in names:
                            name = name_node.value if hasattr(name_node, 'value') else str(name_node)
                            value = module_env.get(name)
                            if value is None:
                                return EvaluationError(f"'{name}' is not exported from {file_path}")
                            env.set(name, value)
                            debug_log(f"  Imported '{name}' from {file_path}", value)
                    elif alias:
                        # Import as alias: use "stdlib/fs" as fs
                        env.set(alias, module_env)
                    else:
                        # Import all functions into current scope
                        for key in module_env.store.keys():
                            env.set(key, module_env.get(key))
                    return NULL
                else:
                    return EvaluationError(f"Stdlib module '{file_path}' not available")
            except Exception as e:
                return EvaluationError(f"Error loading stdlib module '{file_path}': {str(e)}")
        
        # 1b. Check if this is a builtin module (crypto, datetime, math)
        if is_builtin_module(file_path):
            debug_log(f"  Loading builtin module: {file_path}")
            module_env = get_builtin_module(file_path, self)
            if module_env:
                alias = getattr(node, 'alias', None)
                if alias:
                    env.set(alias, module_env)
                else:
                    # Import all functions into current scope
                    for key in module_env.store.keys():
                        env.set(key, module_env.get(key))
                return NULL
            else:
                return EvaluationError(f"Builtin module '{file_path}' not available")
        
        normalized_path = normalize_path(file_path)
        
        # 2. Check Cache
        module_env = get_cached_module(normalized_path)
        
        # 3. Load if not cached
        if not module_env:
            # Get the importing file's path for relative resolution
            importer_file = None
            __file_obj = env.get("__file__")
            if __file_obj:
                if hasattr(__file_obj, 'value'):
                    importer_file = __file_obj.value
                elif isinstance(__file_obj, str):
                    importer_file = __file_obj
            
            candidates = get_module_candidates(file_path, importer_file)
            module_env = Environment()
            loaded = False
            parse_errors = []
            
            # Circular dependency placeholder
            try: 
                cache_module(normalized_path, module_env)
            except Exception: 
                pass
            
            for candidate in candidates:
                try:
                    if not os.path.exists(candidate): 
                        continue
                    
                    debug_log("  Found module file", candidate)
                    with open(candidate, 'r', encoding='utf-8') as f: 
                        code = f.read()
                    
                    from ..lexer import Lexer
                    from ..parser import Parser
                    
                    lexer = Lexer(code)
                    parser = Parser(lexer)
                    program = parser.parse_program()
                    
                    if getattr(parser, 'errors', None):
                        parse_errors.append((candidate, parser.errors))
                        continue
                    
                    # Set __file__ in module environment so it can do relative imports
                    module_env.set("__file__", String(os.path.abspath(candidate)))
                    # Set __MODULE__ to the module path (not "__main__" since it's imported)
                    module_env.set("__MODULE__", String(file_path))
                    
                    # Recursive evaluation
                    self.eval_node(program, module_env)
                    
                    # Update cache with fully loaded env
                    cache_module(normalized_path, module_env)
                    loaded = True
                    break
                except Exception as e:
                    parse_errors.append((candidate, str(e)))
            
            if not loaded:
                try: 
                    invalidate_module(normalized_path)
                except Exception: 
                    pass
                return EvaluationError(f"Module not found or failed to load: {file_path}")
        
        # 4. Bind to Current Environment
        is_named_import = getattr(node, 'is_named_import', False)
        names = getattr(node, 'names', [])
        alias = getattr(node, 'alias', None)
        
        if is_named_import and names:
            # Handle: use { name1, name2 } from "./file.zx"
            exports = module_env.get_exports()
            __file_obj = env.get("__file__")
            importer_file = None
            if __file_obj:
                importer_file = __file_obj.value if hasattr(__file_obj, 'value') else __file_obj
            
            for name_node in names:
                name = name_node.value if hasattr(name_node, 'value') else str(name_node)
                
                # First check if there's a Module object with exports
                value = None
                for key in module_env.store if hasattr(module_env, 'store') else []:
                    potential_module = module_env.get(key)
                    if potential_module and hasattr(potential_module, 'exports') and hasattr(potential_module, 'get_member'):
                        if name in potential_module.exports:
                            member = potential_module.get_member(name)
                            if member:
                                value = member.value
                                break
                
                # Try to get from exports if not found in Module
                if value is None:
                    value = exports.get(name)
                
                if value is None:
                    # Fallback: try to get from module environment directly
                    value = module_env.get(name)
                
                if value is None:
                    return EvaluationError(f"'{name}' is not exported from {file_path}")
                
                # Security check
                if importer_file and not self._check_import_permission(value, importer_file):
                    return EvaluationError(f"Permission denied: cannot import '{name}' from '{file_path}'")
                
                env.set(name, value)
                debug_log(f"  Imported '{name}' from {file_path}", value)
                
        elif alias:
            # Handle: use "./file.zx" as alias
            env.set(alias, module_env)
        else:
            # Handle: use "./file.zx" (import all exports)
            try:
                exports = module_env.get_exports()
                __file_obj = env.get("__file__")
                importer_file = None
                if __file_obj:
                    importer_file = __file_obj.value if hasattr(__file_obj, 'value') else __file_obj
                
                for name, value in exports.items():
                    if importer_file:
                        if not self._check_import_permission(value, importer_file):
                            return EvaluationError(f"Permission denied for export {name}")
                    env.set(name, value)
            except Exception:
                # Fallback: expose module as filename object
                module_name = os.path.basename(file_path)
                env.set(module_name, module_env)
        
        return NULL
    
    def eval_from_statement(self, node, env, stack_trace):
        """Full implementation of FromStatement."""
        from ..module_cache import get_cached_module, cache_module, get_module_candidates, normalize_path, invalidate_module
        
        # 1. Resolve Path
        file_path = node.file_path
        if not file_path: 
            return EvaluationError("from: missing file path")
        
        normalized_path = normalize_path(file_path)
        module_env = get_cached_module(normalized_path)
        
        # 2. Load Logic (Explicitly repeated to ensure isolation)
        if not module_env:
            # Get the importing file's path for relative resolution
            importer_file = None
            __file_obj = env.get("__file__")
            if __file_obj:
                if hasattr(__file_obj, 'value'):
                    importer_file = __file_obj.value
                elif isinstance(__file_obj, str):
                    importer_file = __file_obj
            
            candidates = get_module_candidates(file_path, importer_file)
            module_env = Environment()
            loaded = False
            
            try: 
                cache_module(normalized_path, module_env)
            except Exception: 
                pass
            
            for candidate in candidates:
                try:
                    if not os.path.exists(candidate): 
                        continue
                    
                    with open(candidate, 'r', encoding='utf-8') as f: 
                        code = f.read()
                    
                    from ..lexer import Lexer
                    from ..parser import Parser
                    
                    lexer = Lexer(code)
                    parser = Parser(lexer)
                    program = parser.parse_program()
                    
                    if getattr(parser, 'errors', None): 
                        continue
                    
                    # Set __file__ in module environment so it can do relative imports
                    module_env.set("__file__", String(os.path.abspath(candidate)))
                    # Set __MODULE__ to the module path (not "__main__" since it's imported)
                    module_env.set("__MODULE__", String(file_path))
                    
                    self.eval_node(program, module_env)
                    cache_module(normalized_path, module_env)
                    loaded = True
                    break
                except Exception:
                    continue
            
            if not loaded:
                try: 
                    invalidate_module(normalized_path)
                except Exception: 
                    pass
                return EvaluationError(f"From import: failed to load module {file_path}")
        
        # 3. Import Specific Names
        __file_obj = env.get("__file__")
        importer_file = None
        if __file_obj:
            importer_file = __file_obj.value if hasattr(__file_obj, 'value') else __file_obj
        
        for name_pair in node.imports:
            # name_pair is [source_name, dest_name] (dest_name optional)
            src = name_pair[0].value if hasattr(name_pair[0], 'value') else str(name_pair[0])
            dest = name_pair[1].value if len(name_pair) > 1 and name_pair[1] else src
            
            # First, check if there's a Module object in the environment
            val = None
            found_in_module = False
            
            # Look for modules in the module_env
            for key in module_env.store if hasattr(module_env, 'store') else []:
                potential_module = module_env.get(key)
                if potential_module and hasattr(potential_module, 'exports') and hasattr(potential_module, 'get_member'):
                    # This is a Module object - check if src is in its exports
                    if src in potential_module.exports:
                        member = potential_module.get_member(src)
                        if member:
                            val = member.value
                            found_in_module = True
                            break
            
            # If not found in module exports, try standard exports
            if not found_in_module:
                # Retrieve from module exports
                exports = module_env.get_exports() if hasattr(module_env, 'get_exports') else {}
                val = exports.get(src)
            
            if val is None:
                # Fallback: check if it's in the environment directly
                val = module_env.get(src)
            
            if val is None:
                return EvaluationError(f"'{src}' is not exported from {file_path}")
            
            # Security Check
            if importer_file and not self._check_import_permission(val, importer_file):
                return EvaluationError(f"Permission denied: cannot import '{src}' into '{importer_file}'")
            
            env.set(dest, val)
        
        return NULL
    
    def eval_export_statement(self, node, env, stack_trace):
        names = []
        if hasattr(node, 'names') and node.names:
            names = [n.value for n in node.names]
        elif hasattr(node, 'name') and node.name:
            names = [node.name.value]
        
        # Check if we're inside a module
        current_module = env.get('__current_module__') if env else None
        
        for nm in names:
            val = env.get(nm)
            if not val: 
                return EvaluationError(f"Cannot export undefined: {nm}")
            
            # If inside a module, add to module's exports list
            if current_module and hasattr(current_module, 'add_export'):
                current_module.add_export(nm)
            
            # Also do standard env export
            try: 
                env.export(nm, val)
            except Exception as e: 
                return EvaluationError(f"Export failed: {str(e)}")
        
        return NULL
    
    # === SECURITY STATEMENTS (Full Logic) ===
    
    def eval_seal_statement(self, node, env, stack_trace):
        target_node = node.target
        if not target_node: 
            return EvaluationError("seal: missing target")
        
        if isinstance(target_node, Identifier):
            name = target_node.value
            val = env.get(name)
            if not val: 
                return EvaluationError(f"seal: identifier '{name}' not found")
            
            sealed = SealedObject(val)
            env.set(name, sealed)
            return sealed
        
        elif isinstance(target_node, PropertyAccessExpression):
            obj = self.eval_node(target_node.object, env, stack_trace)
            if is_error(obj): 
                return obj
            
            # Safely extract property key
            if hasattr(target_node.property, 'value'):
                prop_key = target_node.property.value
            else:
                # Evaluate property expression
                prop_result = self.eval_node(target_node.property, env, stack_trace)
                if is_error(prop_result):
                    return prop_result
                prop_key = prop_result.value if hasattr(prop_result, 'value') else str(prop_result)
            
            if isinstance(obj, Map):
                if prop_key not in obj.pairs: 
                    return EvaluationError(f"seal: key '{prop_key}' missing")
                
                obj.pairs[prop_key] = SealedObject(obj.pairs[prop_key])
                return obj.pairs[prop_key]
            
            if hasattr(obj, 'set') and hasattr(obj, 'get'):
                curr = obj.get(prop_key)
                if not curr: 
                    return EvaluationError(f"seal: prop '{prop_key}' missing")
                
                sealed = SealedObject(curr)
                obj.set(prop_key, sealed)
                return sealed
        
        return EvaluationError("seal: unsupported target")
    
    def eval_audit_statement(self, node, env, stack_trace):
        """Evaluate audit statement for compliance logging.
        
        Syntax: audit data_name, "action_type", [optional_timestamp];
        
        Returns a log entry dictionary with the audited data reference.
        """
        from datetime import datetime
        from ..object import String, Map
        
        # Get the data identifier
        if not isinstance(node.data_name, Identifier):
            return EvaluationError(f"audit: expected identifier, got {type(node.data_name).__name__}")
        
        data_name = node.data_name.value
        
        # Evaluate the action type string
        if isinstance(node.action_type, StringLiteral):
            action_type = node.action_type.value
        else:
            action_type_result = self.eval_node(node.action_type, env, stack_trace)
            if is_error(action_type_result):
                return action_type_result
            action_type = to_string(action_type_result)
        
        # Get optional timestamp
        timestamp = None
        if node.timestamp:
            if isinstance(node.timestamp, Identifier):
                timestamp = env.get(node.timestamp.value)
            else:
                timestamp = self.eval_node(node.timestamp, env, stack_trace)
                if is_error(timestamp):
                    return timestamp
        
        # If no timestamp provided, use current time
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        else:
            timestamp = to_string(timestamp)
        
        # Get reference to the audited data
        audited_data = env.get(data_name)
        if audited_data is None:
            return EvaluationError(f"audit: identifier '{data_name}' not found")
        
        # Create audit log entry as a Map object and record via security context
        audit_log_pairs = {
            "data_name": String(data_name),
            "action": String(action_type),
            "timestamp": String(timestamp),
            "data_type": String(type(audited_data).__name__),
        }

        # Register to AuditLog via SecurityContext for persistence/inspection
        try:
            ctx = get_security_context()
            ctx.log_audit(data_name, action_type, type(audited_data).__name__, timestamp, {'source': 'audit_statement'})
            # Also emit a trail event so live traces can capture it
            ctx.emit_event('audit', {'data_name': data_name, 'action': action_type})
        except Exception:
            pass

        return Map(audit_log_pairs)
    
    def eval_restrict_statement(self, node, env, stack_trace):
        """Evaluate restrict statement for field-level access control.
        
        Syntax: restrict obj.field = "restriction_type";
        
        Returns a restriction entry with the applied rule.
        """
        from datetime import datetime, timezone
        from ..object import String, Map
        
        # Get target field information
        if not isinstance(node.target, PropertyAccessExpression):
            return EvaluationError("restrict: target must be object.field")
        
        obj_name = node.target.object.value if isinstance(node.target.object, Identifier) else str(node.target.object)
        field_name = node.target.property.value if isinstance(node.target.property, Identifier) else str(node.target.property)
        
        # Get restriction type
        if isinstance(node.restriction_type, StringLiteral):
            restriction = node.restriction_type.value
        else:
            restriction = to_string(self.eval_node(node.restriction_type, env, stack_trace))
        
        # Get the object to apply restriction
        obj = env.get(obj_name)
        if obj is None:
            return EvaluationError(f"restrict: object '{obj_name}' not found")

        # Register restriction with security context so enforcement can consult it
        try:
            ctx = get_security_context()
            entry = ctx.register_restriction(f"{obj_name}.{field_name}", field_name, restriction)
        except Exception:
            entry = None

        # Return restriction entry (include id if available)
        result_map = {
            "target": String(f"{obj_name}.{field_name}"),
            "field": String(field_name),
            "restriction": String(restriction),
            "status": String("applied"),
            "timestamp": String(datetime.now(timezone.utc).isoformat())
        }
        if entry and entry.get('id'):
            result_map['id'] = String(entry.get('id'))

        return Map(result_map)
    
    def eval_sandbox_statement(self, node, env, stack_trace):
        """Evaluate sandbox statement for isolated execution environments.
        
        Syntax: sandbox { code }
        
        Creates a new isolated environment and executes code within it.
        """

        # Create isolated environment (child of current)
        sandbox_env = Environment(outer=env)
        # Mark as running inside a sandbox and attach a default policy name
        sandbox_env.set('__in_sandbox__', True)
        # Allow caller to specify a policy on the node (future enhancement)
        sandbox_policy = getattr(node, 'policy', None) or 'default'
        sandbox_env.set('__sandbox_policy__', sandbox_policy)
        # Ensure default sandbox policy exists
        try:
            sec = get_security_context()
            if 'default' not in sec.sandbox_policies:
                # conservative default: disallow file I/O builtins
                sec.register_sandbox_policy('default', allowed_builtins=[
                    'now','timestamp','random','to_hex','from_hex','sqrt',
                    'string','len','first','rest','push','reduce','map','filter',
                    'debug_log','debug_trace'
                ])
        except Exception:
            pass

        # Execute body in sandbox
        if node.body is None:
            return NULL

        result = self.eval_node(node.body, sandbox_env, stack_trace)

        # Register sandbox run for observability
        try:
            ctx = get_security_context()
            # store a minimal summary (stringified result) for now
            result_summary = None
            try:
                result_summary = str(result)
            except Exception:
                result_summary = None
            ctx.register_sandbox_run(parent_context=getattr(env, 'name', None), policy=None, result_summary=result_summary)
        except Exception:
            pass

        # Return result from sandbox execution
        return result if result is not None else NULL
    
    def eval_trail_statement(self, node, env, stack_trace):
        """Evaluate trail statement for real-time audit/debug/print tracking.
        
        Syntax:
            trail audit;           // follow all audit events
            trail print;           // follow all print statements
            trail debug;           // follow all debug output
        
        Sets up event tracking and returns trail configuration.
        """
        from datetime import datetime, timezone
        from ..object import String, Map
        
        trail_type = node.trail_type
        filter_key = None
        
        if isinstance(node.filter_key, StringLiteral):
            filter_key = node.filter_key.value
        elif node.filter_key:
            filter_result = self.eval_node(node.filter_key, env, stack_trace)
            if not is_error(filter_result):
                filter_key = to_string(filter_result)
        
        # Register trail with security context so runtime can wire event sinks
        try:
            ctx = get_security_context()
            entry = ctx.register_trail(trail_type, filter_key)
        except Exception:
            entry = None

        # Create trail configuration entry (include id if available)
        trail_config = {
            "type": String(trail_type),
            "filter": String(filter_key) if filter_key else String("*"),
            "enabled": String("true"),
            "timestamp": String(datetime.now(timezone.utc).isoformat())
        }
        if entry and entry.get('id'):
            trail_config['id'] = String(entry.get('id'))

        return Map(trail_config)
    
    def eval_tx_statement(self, node, env, stack_trace):
        """Evaluate transaction block - executes statements in transactional context.
        
        For now, this simply executes the block body.
        In a full blockchain implementation, this would:
        - Create a transaction context
        - Track state changes
        - Support rollback on failure
        - Emit transaction events
        """
        debug_log("eval_tx_statement", "Executing transaction block")
        
        # Execute the transaction body
        result = self.eval_block_statement(node.body, env, stack_trace)
        
        # Return the result of the last statement in the block
        return result if result is not None else NULL
    
    def eval_contract_statement(self, node, env, stack_trace):
        storage = {}
        for sv in node.storage_vars:
            init = NULL
            if getattr(sv, 'initial_value', None):
                init = self.eval_node(sv.initial_value, env, stack_trace)
                if is_error(init): 
                    return init
            storage[sv.name.value] = init
        
        actions = {}
        for act in node.actions:
            # Evaluate action node to get Action object
            action_obj = Action(act.parameters, act.body, env)
            actions[act.name.value] = action_obj
        
        contract = SmartContract(node.name.value, storage, actions)
        contract.deploy()
        
        # Check if contract has a constructor and execute it
        if 'constructor' in actions:
            constructor = actions['constructor']
            # Create contract environment with storage access
            contract_env = Environment(outer=env)
            
            # Set up TX context
            from ..object import Map, String, Integer
            tx_context = Map({
                String("caller"): String("system"),  # Default caller
                String("timestamp"): Integer(int(__import__('time').time())),
            })
            contract_env.set("TX", tx_context)
            
            # Pre-populate environment with storage variables so assignments update storage
            for storage_var in node.storage_vars:
                var_name = storage_var.name.value
                # Get initial value from storage (which was set during deploy)
                initial_val = contract.storage.get(var_name)
                if initial_val is not None:
                    contract_env.set(var_name, initial_val)
            
            # Execute constructor body
            result = self.eval_node(constructor.body, contract_env, stack_trace)
            if is_error(result):
                return result
            
            # After constructor runs, update contract storage with any modified variables
            for storage_var in node.storage_vars:
                var_name = storage_var.name.value
                # Get the value from constructor environment
                val = contract_env.get(var_name)
                if val is not None:
                    # Update persistent storage
                    contract.storage.set(var_name, val)
        
        env.set(node.name.value, contract)
        return NULL
    
    def eval_entity_statement(self, node, env, stack_trace):
        props = {}
        methods = {}
        injected_deps = []  # Track which properties are injected dependencies
        
        # Handle inheritance - get parent reference but DON'T merge properties yet
        parent_entity = None
        if node.parent:
            parent_entity = env.get(node.parent.value)
            if not parent_entity:
                return EvaluationError(f"Parent entity '{node.parent.value}' not found")
            # Check if it's a SecurityEntityDef (the actual entity class we use)
            from ..security import EntityDefinition as SecurityEntityDef
            if not isinstance(parent_entity, SecurityEntityDef):
                return EvaluationError(f"'{node.parent.value}' exists but is not an entity")
            # Note: We no longer copy parent properties here - they'll be accessed via parent_ref
        
        for prop in node.properties:
            # Check if this is an injected dependency
            is_injected = getattr(prop, 'is_injected', False)
            
            # Handle both dict and object formats
            if isinstance(prop, dict):
                p_name = prop['name']
                p_type = prop['type']
            else:
                p_name = prop.name.value
                p_type = prop.type.value if hasattr(prop.type, 'value') else str(prop.type)
            
            if is_injected:
                # This is an injected dependency - mark for injection during instantiation
                injected_deps.append(p_name)
                props[p_name] = {"type": p_type, "default_value": NULL, "injected": True}
                debug_log("eval_entity_statement", f"Marked {p_name} as injected dependency")
            else:
                def_val = NULL
                
                if isinstance(prop, dict):
                    # For dict format, default_value is in the dict
                    if 'default_value' in prop:
                        def_val = self.eval_node(prop['default_value'], env, stack_trace)
                        if is_error(def_val): 
                            return def_val
                else:
                    # For object format, default_value is an attribute
                    if getattr(prop, 'default_value', None):
                        def_val = self.eval_node(prop.default_value, env, stack_trace)
                        if is_error(def_val): 
                            return def_val
                
                props[p_name] = {"type": p_type, "default_value": def_val}
        
        # Process methods (actions defined inside the entity)
        if hasattr(node, 'methods') and node.methods:
            for method in node.methods:
                # Don't evaluate the action statement - that would just store it in env and return NULL
                # Instead, create the Action object directly
                from ..object import Action
                method_action = Action(method.parameters, method.body, env)
                
                # Store the method by name
                method_name = method.name.value if hasattr(method, 'name') else str(method)
                methods[method_name] = method_action
        
        # Create entity with methods and parent reference
        # Now parent_ref points to the actual parent, and props only contains THIS entity's properties
        # Import SecurityEntityDef first for isinstance check
        from ..security import EntityDefinition as SecurityEntityDef
        
        parent_ref = parent_entity if (node.parent and isinstance(parent_entity, SecurityEntityDef)) else None
        
        # Use the EntityDefinition from security.py which supports methods
        entity = SecurityEntityDef(node.name.value, props, methods, parent_ref)
        
        # Store injected dependencies list for constructor use
        entity.injected_deps = injected_deps
        
        env.set(node.name.value, entity)
        return NULL
    
    def eval_verify_statement(self, node, env, stack_trace):
        """Evaluate VERIFY statement - supports multiple forms including extended modes"""
        from .utils import is_truthy as check_truthy
        import os
        import re
        
        # Handle extended verification modes
        if node.mode:
            return self._eval_verify_mode(node, env, stack_trace)
        
        # Special case: verify { cond1, cond2, ... }, "message"
        # When condition is None but logic_block exists, the block contains the conditions
        if node.condition is None and node.logic_block is not None:
            # The logic_block contains expressions that should all be true
            from ..zexus_ast import BlockStatement, ExpressionStatement
            if isinstance(node.logic_block, BlockStatement):
                all_true = True
                for stmt in node.logic_block.statements:
                    # Each statement should be an expression to evaluate
                    if isinstance(stmt, ExpressionStatement):
                        cond_val = self.eval_node(stmt.expression, env, stack_trace)
                    else:
                        cond_val = self.eval_node(stmt, env, stack_trace)
                    
                    if is_error(cond_val):
                        return cond_val
                    
                    if not check_truthy(cond_val):
                        all_true = False
                        break
                
                if not all_true:
                    error_msg = "Verification failed"
                    if node.message:
                        msg_val = self.eval_node(node.message, env, stack_trace)
                        if not is_error(msg_val):
                            error_msg = str(msg_val.value if hasattr(msg_val, 'value') else msg_val)
                    return EvaluationError(error_msg)
                
                # All conditions passed
                from ..object import Boolean
                return Boolean(True)
        
        # Simple assertion form: verify condition, "message"
        if node.condition is not None:
            condition_val = self.eval_node(node.condition, env, stack_trace)
            if is_error(condition_val):
                return condition_val
            
            # Check if condition is truthy
            if not check_truthy(condition_val):
                error_msg = "Verification failed"
                if node.message:
                    msg_val = self.eval_node(node.message, env, stack_trace)
                    if not is_error(msg_val):
                        error_msg = str(msg_val.value if hasattr(msg_val, 'value') else msg_val)
                
                # Execute logic block if provided
                if node.logic_block:
                    block_result = self.eval_node(node.logic_block, env, stack_trace)
                    if is_error(block_result):
                        return block_result
                
                return EvaluationError(error_msg)
            
            # Verification passed
            from ..object import Boolean
            return Boolean(True)
        
        # Complex wrapper form: verify(target, [conditions...])
        if node.target is not None:
            target = self.eval_node(node.target, env, stack_trace)
            if is_error(target): 
                return target
            
            checks = []
            if node.conditions:
                for cond in node.conditions:
                    val = self.eval_node(cond, env, stack_trace)
                    if is_error(val): 
                        return val
                    
                    if callable(val) or isinstance(val, Action):
                        checks.append(VerificationCheck(str(cond), lambda ctx: val))
                    else:
                        checks.append(VerificationCheck(str(cond), lambda ctx, v=val: v))
            
            wrapped = VerifyWrapper(target, checks, node.error_handler)
            get_security_context().register_verify_check(str(node.target), wrapped)
            return wrapped
        
        # Neither form provided
        return EvaluationError("Invalid VERIFY statement: requires condition or target")
    
    def _eval_verify_mode(self, node, env, stack_trace):
        """Evaluate verify statement with specific mode (data, access, db, env, pattern)"""
        from .utils import is_truthy as check_truthy
        import os
        import re
        from ..object import Boolean, String
        
        mode = node.mode
        
        # verify:data - Data/format verification
        if mode == 'data':
            return self._eval_verify_data(node, env, stack_trace)
        
        # verify:access - Access control with blocking
        elif mode == 'access':
            return self._eval_verify_access(node, env, stack_trace)
        
        # verify:db - Database verification
        elif mode == 'db':
            return self._eval_verify_db(node, env, stack_trace)
        
        # verify:env - Environment variable verification
        elif mode == 'env':
            return self._eval_verify_env(node, env, stack_trace)
        
        # verify:pattern - Pattern matching
        elif mode == 'pattern':
            return self._eval_verify_pattern(node, env, stack_trace)
        
        return EvaluationError(f"Unknown verification mode: {mode}")
    
    def _eval_verify_data(self, node, env, stack_trace):
        """Evaluate verify:data - data/format verification"""
        from .utils import is_truthy as check_truthy
        import re
        from ..object import Boolean, String
        
        # Evaluate the value to verify
        value_val = self.eval_node(node.condition, env, stack_trace)
        if is_error(value_val):
            return value_val
        
        value = value_val.value if hasattr(value_val, 'value') else str(value_val)
        verify_type = node.verify_type
        
        # Evaluate pattern/expected value
        pattern_val = self.eval_node(node.pattern, env, stack_trace) if node.pattern else None
        if pattern_val and is_error(pattern_val):
            return pattern_val
        
        pattern = pattern_val.value if pattern_val and hasattr(pattern_val, 'value') else str(pattern_val) if pattern_val else None
        
        # Perform verification based on type
        is_valid = False
        
        if verify_type == 'matches':
            # Pattern matching
            if pattern:
                try:
                    is_valid = bool(re.match(pattern, str(value)))
                except (re.error, TypeError, ValueError):
                    is_valid = False
        
        elif verify_type == 'is_type' or verify_type == 'is':
            # Type checking
            type_map = {
                'string': str,
                'number': (int, float),
                'integer': int,
                'float': float,
                'boolean': bool,
                'bool': bool,
                'email': lambda v: '@' in str(v) and '.' in str(v),
            }
            if pattern in type_map:
                expected_type = type_map[pattern]
                if callable(expected_type):
                    is_valid = expected_type(value)
                else:
                    is_valid = isinstance(value, expected_type)
        
        elif verify_type == 'equals':
            # Equality check
            is_valid = str(value) == str(pattern)
        
        # Handle verification failure
        if not is_valid:
            error_msg = "Data verification failed"
            if node.message:
                msg_val = self.eval_node(node.message, env, stack_trace)
                if not is_error(msg_val):
                    error_msg = str(msg_val.value if hasattr(msg_val, 'value') else msg_val)
            
            # Execute logic block if provided
            if node.logic_block:
                block_result = self.eval_node(node.logic_block, env, stack_trace)
                if is_error(block_result):
                    return block_result
            
            return EvaluationError(error_msg)
        
        return Boolean(True)
    
    def _eval_verify_access(self, node, env, stack_trace):
        """Evaluate verify:access - access control with blocking actions"""
        from .utils import is_truthy as check_truthy
        from ..object import Boolean
        
        # Evaluate access condition
        condition_val = self.eval_node(node.condition, env, stack_trace)
        if is_error(condition_val):
            return condition_val
        
        # Check if access should be granted
        if not check_truthy(condition_val):
            error_msg = "Access denied"
            if node.message:
                msg_val = self.eval_node(node.message, env, stack_trace)
                if not is_error(msg_val):
                    error_msg = str(msg_val.value if hasattr(msg_val, 'value') else msg_val)
            
            # Execute action block (blocking actions)
            if node.action_block:
                self.eval_node(node.action_block, env, stack_trace)
                # Don't return error from block - it's for logging/actions
                # The access denial itself is the error
            
            # Block access by returning error
            return EvaluationError(error_msg)
        
        return Boolean(True)
    
    def _eval_verify_db(self, node, env, stack_trace):
        """Evaluate verify:db - database verification"""
        from ..object import Boolean, String
        
        # Evaluate value to check
        value_val = self.eval_node(node.condition, env, stack_trace)
        if is_error(value_val):
            return value_val
        
        value = value_val.value if hasattr(value_val, 'value') else str(value_val)
        
        # Evaluate table name
        table_val = self.eval_node(node.db_table, env, stack_trace) if node.db_table else None
        if table_val and is_error(table_val):
            return table_val
        
        table = table_val.value if table_val and hasattr(table_val, 'value') else str(table_val) if table_val else None
        
        # Get database query type
        query_type = node.db_query  # exists_in, unique_in, matches_in
        
        # Try to get database connection from environment
        # This allows users to inject their own database handlers
        db_handler = env.get('__db_handler__') if hasattr(env, 'get') else None
        
        is_valid = False
        
        if db_handler and hasattr(db_handler, query_type):
            # Use custom database handler
            try:
                result = getattr(db_handler, query_type)(table, value)
                is_valid = bool(result)
            except Exception as e:
                return EvaluationError(f"Database verification error: {str(e)}")
        else:
            # Fallback: Check if persistence module is available
            try:
                from ..persistence import get_storage_backend
                storage = get_storage_backend()
                
                if query_type == 'exists_in':
                    # Check if value exists
                    key = f"{table}:{value}"
                    result = storage.get(key)
                    is_valid = result is not None
                
                elif query_type == 'unique_in':
                    # Check if value is unique (doesn't exist)
                    key = f"{table}:{value}"
                    result = storage.get(key)
                    is_valid = result is None
                
                elif query_type == 'matches_in':
                    # Custom query - requires db_handler
                    return EvaluationError(f"Database query '{query_type}' requires custom db_handler")
            
            except Exception as e:
                # No database available - treat as verification failure
                is_valid = False
        
        # Handle verification failure
        if not is_valid:
            error_msg = "Database verification failed"
            if node.message:
                msg_val = self.eval_node(node.message, env, stack_trace)
                if not is_error(msg_val):
                    error_msg = str(msg_val.value if hasattr(msg_val, 'value') else msg_val)
            
            # Execute logic block if provided
            if node.logic_block:
                block_result = self.eval_node(node.logic_block, env, stack_trace)
                if is_error(block_result):
                    return block_result
            
            return EvaluationError(error_msg)
        
        return Boolean(True)
    
    def _eval_verify_env(self, node, env, stack_trace):
        """Evaluate verify:env - environment variable verification"""
        import os
        from ..object import Boolean, String
        
        # Evaluate env var name
        var_val = self.eval_node(node.env_var, env, stack_trace)
        if is_error(var_val):
            return var_val
        
        var_name = var_val.value if hasattr(var_val, 'value') else str(var_val)
        verify_type = node.verify_type or 'is_set'
        
        # Get environment variable value
        env_value = os.environ.get(var_name)
        
        is_valid = False
        
        if verify_type == 'is_set' or verify_type == 'exists':
            # Check if env var is set
            is_valid = env_value is not None
        
        elif verify_type == 'equals':
            # Check if env var equals expected value
            expected_val = self.eval_node(node.expected_value, env, stack_trace) if node.expected_value else None
            if expected_val and is_error(expected_val):
                return expected_val
            
            expected = expected_val.value if expected_val and hasattr(expected_val, 'value') else str(expected_val) if expected_val else None
            is_valid = env_value == expected
        
        elif verify_type == 'matches':
            # Pattern matching on env var value
            import re
            pattern_val = self.eval_node(node.expected_value, env, stack_trace) if node.expected_value else None
            if pattern_val and is_error(pattern_val):
                return pattern_val
            
            pattern = pattern_val.value if pattern_val and hasattr(pattern_val, 'value') else str(pattern_val) if pattern_val else None
            
            if env_value and pattern:
                try:
                    is_valid = bool(re.match(pattern, env_value))
                except (re.error, TypeError, ValueError):
                    is_valid = False
        
        # Handle verification failure
        if not is_valid:
            error_msg = f"Environment variable verification failed: {var_name}"
            if node.message:
                msg_val = self.eval_node(node.message, env, stack_trace)
                if not is_error(msg_val):
                    error_msg = str(msg_val.value if hasattr(msg_val, 'value') else msg_val)
            
            # Execute logic block if provided
            if node.logic_block:
                block_result = self.eval_node(node.logic_block, env, stack_trace)
                if is_error(block_result):
                    return block_result
            
            return EvaluationError(error_msg)
        
        return Boolean(True)
    
    def _eval_verify_pattern(self, node, env, stack_trace):
        """Evaluate verify:pattern - pattern matching verification"""
        import re
        from ..object import Boolean
        
        # Evaluate value to match
        value_val = self.eval_node(node.condition, env, stack_trace)
        if is_error(value_val):
            return value_val
        
        value = value_val.value if hasattr(value_val, 'value') else str(value_val)
        
        # Evaluate pattern
        pattern_val = self.eval_node(node.pattern, env, stack_trace) if node.pattern else None
        if pattern_val and is_error(pattern_val):
            return pattern_val
        
        pattern = pattern_val.value if pattern_val and hasattr(pattern_val, 'value') else str(pattern_val) if pattern_val else None
        
        # Perform pattern matching
        is_valid = False
        if pattern:
            try:
                is_valid = bool(re.match(pattern, str(value)))
            except Exception as e:
                return EvaluationError(f"Pattern matching error: {str(e)}")
        
        # Handle verification failure
        if not is_valid:
            error_msg = "Pattern verification failed"
            if node.message:
                msg_val = self.eval_node(node.message, env, stack_trace)
                if not is_error(msg_val):
                    error_msg = str(msg_val.value if hasattr(msg_val, 'value') else msg_val)
            
            # Execute logic block if provided
            if node.logic_block:
                block_result = self.eval_node(node.logic_block, env, stack_trace)
                if is_error(block_result):
                    return block_result
            
            return EvaluationError(error_msg)
        
        return Boolean(True)
    
    def eval_protect_statement(self, node, env, stack_trace):
        """Evaluate PROTECT statement with full policy engine integration."""
        from ..policy_engine import get_policy_registry, PolicyBuilder
        from ..object import String as StringObj
        
        # Evaluate target expression
        target = self.eval_node(node.target, env, stack_trace)
        if is_error(target): 
            return target
        
        # Get target name (for registration)
        target_name = str(node.target.value) if hasattr(node.target, 'value') else str(target)
        
        # Evaluate rules - could be a Map literal or BlockStatement
        rules_val = self.eval_node(node.rules, env, stack_trace)
        if is_error(rules_val): 
            return rules_val
        
        # Convert rules to dictionary
        rules_dict = {}
        if isinstance(rules_val, Map):
            # Direct map literal: {rate_limit: 10, auth_required: true, ...}
            for k, v in rules_val.pairs.items():
                key = k.value if isinstance(k, String) else str(k)
                # Convert Zexus objects to Python values
                if isinstance(v, Integer):
                    rules_dict[key] = v.value
                elif isinstance(v, String):
                    rules_dict[key] = v.value
                elif isinstance(v, (TRUE.__class__, FALSE.__class__)):
                    rules_dict[key] = v == TRUE
                elif isinstance(v, List):
                    rules_dict[key] = [item.value if hasattr(item, 'value') else item for item in v.elements]
                else:
                    rules_dict[key] = v
        elif hasattr(rules_val, 'statements'):
            # Block statement (old style)
            for stmt in rules_val.statements:
                # Handle statement-based rules
                pass
        
        # Determine enforcement level
        enforcement_level = "strict"  # Default
        if hasattr(node, 'enforcement_level') and node.enforcement_level:
            enforcement_level = node.enforcement_level.lower()
        
        # Build policy using PolicyBuilder
        builder = PolicyBuilder(target_name)
        builder.set_enforcement(enforcement_level)
        
        # Parse rules and add to policy
        for rule_type, rule_config in rules_dict.items():
            if rule_type == "verify":
                # Add verification rules
                if isinstance(rule_config, list):
                    for condition in rule_config:
                        builder.add_verify_rule(str(condition))
            
            elif rule_type == "restrict":
                # Add restriction rules
                if isinstance(rule_config, dict):
                    for field, constraints in rule_config.items():
                        builder.add_restrict_rule(field, constraints if isinstance(constraints, list) else [constraints])
            
            elif rule_type in ("audit", "log_access") and rule_config:
                # Enable audit logging
                builder.enable_audit()
        
        # Build and register policy
        policy = builder.build()
        policy_registry = get_policy_registry()
        policy_registry.register_policy(target_name, policy)
        
        debug_log("eval_protect_statement", f"✓ Policy registered for {target_name} (level: {enforcement_level})")
        
        # Store policy reference in environment for enforcement
        env.set(f"__policy_{target_name}__", policy)
        
        # Also register with legacy security context for backwards compatibility
        try:
            from ..security import ProtectionPolicy, get_security_context
            policy_legacy = ProtectionPolicy(target_name, rules_dict, enforcement_level)
            get_security_context().register_protection(target_name, policy_legacy)
        except (AttributeError, NameError, ImportError):
            pass  # Legacy context may not be available
        
        return StringObj(f"Protection policy activated for '{target_name}' (level: {enforcement_level})")
    
    def eval_middleware_statement(self, node, env, stack_trace):
        handler = self.eval_node(node.handler, env)
        if is_error(handler): 
            return handler
        
        mw = Middleware(node.name.value, handler)
        get_security_context().middlewares[node.name.value] = mw
        return NULL
    
    def eval_auth_statement(self, node, env, stack_trace):
        config = self.eval_node(node.config, env)
        if is_error(config): 
            return config
        
        c_dict = {}
        if isinstance(config, Map):
            for k, v in config.pairs.items():
                c_dict[k.value if isinstance(k, String) else str(k)] = v
        
        get_security_context().auth_config = AuthConfig(c_dict)
        return NULL
    
    def eval_throttle_statement(self, node, env, stack_trace):
        limits = self.eval_node(node.limits, env)
        
        rpm, burst, per_user = 100, 10, False
        if isinstance(limits, Map):
            for k, v in limits.pairs.items():
                ks = k.value if isinstance(k, String) else str(k)
                if ks == "requests_per_minute" and isinstance(v, Integer): 
                    rpm = v.value
                elif ks == "burst_size" and isinstance(v, Integer): 
                    burst = v.value
                elif ks == "per_user": 
                    per_user = True if (isinstance(v, Boolean) and v.value) else False
        
        limiter = RateLimiter(rpm, burst, per_user)
        ctx = get_security_context()
        if not hasattr(ctx, 'rate_limiters'): 
            ctx.rate_limiters = {}
        ctx.rate_limiters[str(node.target)] = limiter
        return NULL
    
    def eval_cache_statement(self, node, env, stack_trace):
        policy = self.eval_node(node.policy, env)
        
        ttl, inv = 3600, []
        if isinstance(policy, Map):
            for k, v in policy.pairs.items():
                ks = k.value if isinstance(k, String) else str(k)
                if ks == "ttl" and isinstance(v, Integer): 
                    ttl = v.value
                elif ks == "invalidate_on" and isinstance(v, List):
                    inv = [x.value if hasattr(x, 'value') else str(x) for x in v.elements]
        
        cp = CachePolicy(ttl, inv)
        ctx = get_security_context()
        if not hasattr(ctx, 'cache_policies'): 
            ctx.cache_policies = {}
        ctx.cache_policies[str(node.target)] = cp
        return NULL
    


    # === MISC STATEMENTS ===
    
    def eval_print_statement(self, node, env, stack_trace):
        # Check if conditional print
        if hasattr(node, 'condition') and node.condition is not None:
            # Evaluate the condition
            condition_val = self.eval_node(node.condition, env, stack_trace)
            if is_error(condition_val):
                print(f"❌ Error in print condition: {condition_val}", file=sys.stderr)
                return NULL
            
            # Check if condition is truthy
            is_truthy = False
            if hasattr(condition_val, 'value'):
                # Boolean, Integer, etc.
                if isinstance(condition_val.value, bool):
                    is_truthy = condition_val.value
                elif isinstance(condition_val.value, (int, float)):
                    is_truthy = condition_val.value != 0
                elif isinstance(condition_val.value, str):
                    is_truthy = len(condition_val.value) > 0
                else:
                    is_truthy = bool(condition_val.value)
            else:
                # For objects without .value, check if it's NULL
                is_truthy = not (hasattr(condition_val, 'type') and condition_val.type == 'NULL')
            
            # If condition is false, don't print
            if not is_truthy:
                return NULL
        
        # Handle both legacy single value and new multiple values
        values_to_print = []
        
        if hasattr(node, 'values') and node.values:
            # New format with multiple values
            for expr in node.values:
                val = self.eval_node(expr, env, stack_trace)
                if is_error(val):
                    print(f"❌ Error: {val}", file=sys.stderr)
                    return NULL
                values_to_print.append(val)
        elif hasattr(node, 'value') and node.value is not None:
            # Legacy single value format
            val = self.eval_node(node.value, env, stack_trace)
            if is_error(val):
                print(f"❌ Error: {val}", file=sys.stderr)
                return NULL
            values_to_print.append(val)
        else:
            return NULL
        
        # Convert all values to strings and join with space
        output_parts = []
        for v in values_to_print:
            part = v.inspect() if hasattr(v, 'inspect') else str(v)
            output_parts.append(part)
        
        output = ' '.join(output_parts)
        print(output, flush=True)  # Flush immediately for async threads
        
        try:
            ctx = get_security_context()
            ctx.emit_event('print', {'value': output})
        except Exception:
            pass
        
        return NULL
    
    def eval_screen_statement(self, node, env, stack_trace):
        print(f"[RENDER] Screen: {node.name.value}")
        return NULL
    
    def eval_embedded_code_statement(self, node, env, stack_trace):
        obj = EmbeddedCode(node.name.value, node.language, node.code)
        env.set(node.name.value, obj)
        return NULL
    
    def eval_component_statement(self, node, env, stack_trace):
        props = None
        if hasattr(node, 'properties') and node.properties:
            val = self.eval_node(node.properties, env, stack_trace)
            if is_error(val): 
                return val
            props = _zexus_to_python(val)
        
        # Check builtin
        if hasattr(self, 'builtins') and 'define_component' in self.builtins:
            self.builtins['define_component'].fn(String(node.name.value), Map(props) if isinstance(props, dict) else NULL)
            return NULL
        
        env.set(node.name.value, String(f"<component {node.name.value}>"))
        return NULL
    
    def eval_theme_statement(self, node, env, stack_trace):
        val = self.eval_node(node.properties, env, stack_trace) if hasattr(node, 'properties') else NULL
        if is_error(val): 
            return val
        env.set(node.name.value, val)
        return NULL
    
    def eval_debug_statement(self, node, env, stack_trace):
        # Check if conditional debug
        if hasattr(node, 'condition') and node.condition is not None:
            # Evaluate the condition
            condition_val = self.eval_node(node.condition, env, stack_trace)
            if is_error(condition_val):
                return condition_val
            
            # Check if condition is truthy
            is_truthy = False
            if hasattr(condition_val, 'value'):
                # Boolean, Integer, etc.
                if isinstance(condition_val.value, bool):
                    is_truthy = condition_val.value
                elif isinstance(condition_val.value, (int, float)):
                    is_truthy = condition_val.value != 0
                elif isinstance(condition_val.value, str):
                    is_truthy = len(condition_val.value) > 0
                else:
                    is_truthy = bool(condition_val.value)
            else:
                # For objects without .value, check if it's NULL
                is_truthy = not (hasattr(condition_val, 'type') and condition_val.type == 'NULL')
            
            # If condition is false, don't debug
            if not is_truthy:
                return NULL
        
        val = self.eval_node(node.value, env, stack_trace)
        if is_error(val): 
            return val
        
        from ..object import Debug, String, Integer, Float, Boolean
        # Convert to human-readable string
        if isinstance(val, String):
            message = val.value
        elif isinstance(val, (Integer, Float)):
            message = str(val.value)
        elif isinstance(val, Boolean):
            message = "true" if val.value else "false"
        else:
            message = val.inspect() if hasattr(val, 'inspect') else str(val)
        
        Debug.log(message)
        try:
            ctx = get_security_context()
            ctx.emit_event('debug', {'value': message})
        except Exception:
            pass
        return NULL
    
    def eval_external_declaration(self, node, env, stack_trace):
        def _placeholder(*a): 
            return EvaluationError(f"External '{node.name.value}' not linked")
        
        env.set(node.name.value, Builtin(_placeholder, node.name.value))
        return NULL
    
    def eval_exactly_statement(self, node, env, stack_trace):
        return self.eval_node(node.body, env, stack_trace)
    
    def eval_action_statement(self, node, env, stack_trace):
        action = Action(node.parameters, node.body, env)
        
        # Check for direct is_async attribute (from UltimateParser)
        if hasattr(node, 'is_async') and node.is_async:
            action.is_async = True
        
        # Apply modifiers if present (from standard parser)
        modifiers = getattr(node, 'modifiers', [])
        if modifiers:
            # Set modifier flags on the action object
            if 'inline' in modifiers:
                action.is_inlined = True
            if 'async' in modifiers:
                action.is_async = True
            if 'secure' in modifiers:
                action.is_secure = True
            if 'pure' in modifiers:
                action.is_pure = True
            if 'native' in modifiers:
                action.is_native = True
            
            # 'public' modifier: automatically export the action
            if 'public' in modifiers:
                try:
                    env.export(node.name.value, action)
                except Exception:
                    pass
        
        env.set(node.name.value, action)
        return NULL
    
    def eval_function_statement(self, node, env, stack_trace):
        """Evaluate function statement - identical to action statement in Zexus"""
        action = Action(node.parameters, node.body, env)
        
        # Apply modifiers if present
        modifiers = getattr(node, 'modifiers', [])
        if modifiers:
            # Set modifier flags on the action object
            if 'inline' in modifiers:
                action.is_inlined = True
            if 'async' in modifiers:
                action.is_async = True
            if 'secure' in modifiers:
                action.is_secure = True
            if 'pure' in modifiers:
                action.is_pure = True
            if 'native' in modifiers:
                action.is_native = True
            
            # 'public' modifier: automatically export the function
            if 'public' in modifiers:
                try:
                    env.export(node.name.value, action)
                except Exception:
                    pass
        
        env.set(node.name.value, action)
        return NULL
    
    # === PERFORMANCE OPTIMIZATION STATEMENTS ===
    
    def eval_native_statement(self, node, env, stack_trace):
        """Evaluate native statement - call C/C++ code directly."""
        try:
            import ctypes
            
            # Load the shared library
            try:
                lib = ctypes.CDLL(node.library_name)
            except (OSError, AttributeError) as e:
                return EvaluationError(f"Failed to load native library '{node.library_name}': {str(e)}")
            
            # Get the function from the library
            try:
                native_func = getattr(lib, node.function_name)
            except AttributeError:
                return EvaluationError(f"Function '{node.function_name}' not found in library '{node.library_name}'")
            
            # Evaluate arguments
            args = []
            for arg in node.args:
                val = self.eval_node(arg, env, stack_trace)
                if is_error(val):
                    return val
                # Convert Zexus objects to Python types for FFI
                args.append(_zexus_to_python(val))
            
            # Call the native function
            try:
                result = native_func(*args)
                # Convert result back to Zexus object
                zexus_result = _python_to_zexus(result)
                
                # Store result if alias provided
                if node.alias:
                    env.set(node.alias, zexus_result)
                
                return zexus_result
            except Exception as e:
                return EvaluationError(f"Error calling native function '{node.function_name}': {str(e)}")
        
        except ImportError:
            return EvaluationError("ctypes module required for native statements")
    
    def eval_gc_statement(self, node, env, stack_trace):
        """Evaluate garbage collection statement."""
        try:
            import gc
            
            action = node.action.lower()
            
            if action == "collect":
                # Force garbage collection
                collected = gc.collect()
                return Integer(collected)
            
            elif action == "pause":
                # Pause garbage collection
                gc.disable()
                return String("GC paused")
            
            elif action == "resume":
                # Resume garbage collection
                gc.enable()
                return String("GC resumed")
            
            elif action == "enable_debug":
                # Enable GC debug output
                gc.set_debug(gc.DEBUG_STATS)
                return String("GC debug enabled")
            
            elif action == "disable_debug":
                # Disable GC debug output
                gc.set_debug(0)
                return String("GC debug disabled")
            
            else:
                return EvaluationError(f"Unknown GC action: {action}")
        
        except Exception as e:
            return EvaluationError(f"Error in GC statement: {str(e)}")
    
    def eval_inline_statement(self, node, env, stack_trace):
        """Evaluate inline statement - mark function for inlining optimization."""
        # Get the function to inline
        func_name = node.function_name
        if isinstance(func_name, Identifier):
            func_name = func_name.value
        
        func = env.get(func_name)
        if func is None:
            return EvaluationError(f"Function '{func_name}' not found for inlining")
        
        # Mark function as inlined by setting a flag
        if hasattr(func, 'is_inlined'):
            func.is_inlined = True
        elif isinstance(func, Action):
            func.is_inlined = True
        elif isinstance(func, Builtin):
            func.is_inlined = True
        else:
            # Try to set the attribute dynamically
            try:
                func.is_inlined = True
            except AttributeError:
                pass  # Function object doesn't support dynamic attributes
        
        return String(f"Function '{func_name}' marked for inlining")
    
    def eval_buffer_statement(self, node, env, stack_trace):
        """Evaluate buffer statement - direct memory access and manipulation."""
        try:
            import array
            
            buffer_name = node.buffer_name
            operation = node.operation
            arguments = node.arguments
            
            if operation == "allocate":
                # allocate(size) - allocate a buffer
                if len(arguments) != 1:
                    return EvaluationError(f"allocate expects 1 argument, got {len(arguments)}")
                
                size_val = self.eval_node(arguments[0], env, stack_trace)
                if is_error(size_val):
                    return size_val
                
                size = _zexus_to_python(size_val)
                try:
                    size = int(size)
                    # Create a byte array as a simple buffer representation
                    buf = bytearray(size)
                    env.set(buffer_name, _python_to_zexus(buf))
                    return String(f"Buffer '{buffer_name}' allocated with size {size}")
                except (ValueError, TypeError):
                    return EvaluationError(f"Invalid size for buffer allocation: {size}")
            
            elif operation == "read":
                # buffer.read(offset, length)
                if len(arguments) != 2:
                    return EvaluationError(f"read expects 2 arguments, got {len(arguments)}")
                
                offset_val = self.eval_node(arguments[0], env, stack_trace)
                length_val = self.eval_node(arguments[1], env, stack_trace)
                
                if is_error(offset_val) or is_error(length_val):
                    return offset_val if is_error(offset_val) else length_val
                
                buf = env.get(buffer_name)
                if buf is None:
                    return EvaluationError(f"Buffer '{buffer_name}' not found")
                
                offset = _zexus_to_python(offset_val)
                length = _zexus_to_python(length_val)
                
                try:
                    offset, length = int(offset), int(length)
                    buf_data = _zexus_to_python(buf)
                    data = buf_data[offset:offset+length]
                    return _python_to_zexus(list(data))
                except Exception as e:
                    return EvaluationError(f"Error reading from buffer: {str(e)}")
            
            elif operation == "write":
                # buffer.write(offset, data)
                if len(arguments) != 2:
                    return EvaluationError(f"write expects 2 arguments, got {len(arguments)}")
                
                offset_val = self.eval_node(arguments[0], env, stack_trace)
                data_val = self.eval_node(arguments[1], env, stack_trace)
                
                if is_error(offset_val) or is_error(data_val):
                    return offset_val if is_error(offset_val) else data_val
                
                buf = env.get(buffer_name)
                if buf is None:
                    return EvaluationError(f"Buffer '{buffer_name}' not found")
                
                offset = _zexus_to_python(offset_val)
                data = _zexus_to_python(data_val)
                
                try:
                    offset = int(offset)
                    buf_data = _zexus_to_python(buf)
                    if isinstance(buf_data, (bytearray, list)):
                        if isinstance(data, list):
                            for i, byte in enumerate(data):
                                buf_data[offset + i] = int(byte)
                        else:
                            buf_data[offset] = int(data)
                    else:
                        return EvaluationError(f"Buffer is not writable")
                    return String(f"Wrote {len(data) if isinstance(data, list) else 1} bytes at offset {offset}")
                except Exception as e:
                    return EvaluationError(f"Error writing to buffer: {str(e)}")
            
            elif operation == "free":
                # free() - deallocate buffer
                buf = env.get(buffer_name)
                if buf is None:
                    return EvaluationError(f"Buffer '{buffer_name}' not found")
                
                env.delete(buffer_name)
                return String(f"Buffer '{buffer_name}' freed")
            
            else:
                return EvaluationError(f"Unknown buffer operation: {operation}")
        
        except Exception as e:
            return EvaluationError(f"Error in buffer statement: {str(e)}")
    
    def eval_simd_statement(self, node, env, stack_trace):
        """Evaluate SIMD statement - vector operations using SIMD instructions."""
        try:
            import numpy as np
            
            # Evaluate the SIMD operation expression
            result = self.eval_node(node.operation, env, stack_trace)
            
            if is_error(result):
                return result
            
            # Convert result to Zexus object
            zexus_result = result
            
            return zexus_result
        
        except ImportError:
            # Fallback to pure Python implementation if numpy not available
            result = self.eval_node(node.operation, env, stack_trace)
            return result if not is_error(result) else EvaluationError("SIMD operations require numpy or fallback implementation")
        
        except Exception as e:
            return EvaluationError(f"Error in SIMD statement: {str(e)}")
    
    def _execute_deferred_cleanup(self, env, stack_trace):
        """Execute all deferred cleanup code in LIFO order (Last In, First Out)."""
        if not hasattr(env, '_deferred') or not env._deferred:
            return
        
        # Execute in reverse order (LIFO - like a stack)
        while env._deferred:
            deferred_block = env._deferred.pop()  # Remove and get last item
            try:
                # Execute the deferred cleanup code
                self.eval_node(deferred_block, env, stack_trace)
            except Exception as e:
                # Deferred cleanup should not crash the program
                # But we can log it for debugging
                debug_log(f"Error in deferred cleanup: {e}")
    
    def eval_defer_statement(self, node, env, stack_trace):
        """Evaluate defer statement - cleanup code execution."""
        # Store the deferred code for later execution (at end of scope/function)
        if not hasattr(env, '_deferred'):
            env._deferred = []
        
        env._deferred.append(node.code_block)
        return NULL  # Don't return message, just silently register
    
    def eval_pattern_statement(self, node, env, stack_trace):
        """Evaluate pattern statement - pattern matching."""
        debug_log("eval_pattern_statement", f"Matching against {len(node.cases)} cases")
        
        # Evaluate the expression to match
        value = self.eval_node(node.expression, env, stack_trace)
        if is_error(value):
            return value
        
        debug_log("  Match value", f"{value.inspect() if hasattr(value, 'inspect') else value}")
        
        # Try each pattern case
        for i, case in enumerate(node.cases):
            debug_log(f"  Trying case {i}", f"pattern={case.pattern}")
            
            # Check if this is the default case
            if isinstance(case.pattern, str) and case.pattern == "default":
                debug_log("  ✅ Default case matched", "")
                action_result = self.eval_node(case.action, env, stack_trace)
                return action_result
            
            # Evaluate the pattern expression
            pattern_value = self.eval_node(case.pattern, env, stack_trace)
            if is_error(pattern_value):
                debug_log(f"  ❌ Pattern evaluation error", str(pattern_value))
                continue  # Skip invalid patterns
            
            debug_log("  Pattern value", f"{pattern_value.inspect() if hasattr(pattern_value, 'inspect') else pattern_value}")
            
            # Compare values
            matched = False
            if isinstance(value, Integer) and isinstance(pattern_value, Integer):
                matched = value.value == pattern_value.value
                debug_log("  Integer comparison", f"{value.value} == {pattern_value.value} = {matched}")
            elif isinstance(value, Float) and isinstance(pattern_value, Float):
                matched = value.value == pattern_value.value
            elif isinstance(value, String) and isinstance(pattern_value, String):
                matched = value.value == pattern_value.value
            elif isinstance(value, Boolean) and isinstance(pattern_value, Boolean):
                matched = value.value == pattern_value.value
            elif value == pattern_value:
                matched = True
            
            if matched:
                debug_log("  ✅ Pattern matched!", f"Executing action")
                # Execute action
                action_result = self.eval_node(case.action, env, stack_trace)
                debug_log("  Action result", f"{action_result}")
                return action_result
        
        debug_log("  ❌ No pattern matched", "")
        # No match found
        return NULL
    
    def eval_enum_statement(self, node, env, stack_trace):
        """Evaluate enum statement - type-safe enumerations."""
        # Create an enum object
        enum_obj = Map({})
        
        for i, member in enumerate(node.members):
            # Use provided value or auto-increment
            if member.value is not None:
                value = member.value
            else:
                value = i
            
            enum_obj.set(member.name, Integer(value) if isinstance(value, int) else String(value))
        
        # Store enum in environment
        env.set(node.name, enum_obj)
        return String(f"Enum '{node.name}' defined with {len(node.members)} members")
    
    def eval_stream_statement(self, node, env, stack_trace):
        """Evaluate stream statement - event streaming."""
        # Register stream handler
        if not hasattr(env, '_streams'):
            env._streams = {}
        
        # Store handler for stream
        env._streams[node.stream_name] = {
            'event_var': node.event_var,
            'handler': node.handler
        }
        
        return String(f"Stream '{node.stream_name}' handler registered")
    
    


    # === NEW SECURITY STATEMENT HANDLERS ===

    def eval_capability_statement(self, node, env, stack_trace):
        """Evaluate capability definition statement."""
        from ..capability_system import Capability, CapabilityLevel
        
        # Get capability name
        cap_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Extract definition details
        scope = ""
        level = CapabilityLevel.ALLOWED
        
        if node.definition and isinstance(node.definition, Map):
            # Extract from map
            for key, val in node.definition.pairs:
                if hasattr(key, 'value'):
                    if key.value == "scope" and hasattr(val, 'value'):
                        scope = val.value
        
        # Create capability object
        cap = Capability(
            name=cap_name,
            level=level,
            reason=f"Defined with scope: {scope}"
        )
        
        # Store in environment both as identifier and in _capabilities
        if not hasattr(env, '_capabilities'):
            env._capabilities = {}
        env._capabilities[cap_name] = cap
        env.set(cap_name, cap)  # Also store as identifier so it can be referenced
        
        debug_log("eval_capability_statement", f"Defined capability: {cap_name} ({scope})")
        return cap  # Return the capability object instead of just a string

    def eval_grant_statement(self, node, env, stack_trace):
        """Evaluate grant statement - grant capabilities to entity."""
        from ..capability_system import get_capability_manager
        
        manager = get_capability_manager()
        
        # Get entity name
        entity_name = node.entity_name.value if hasattr(node.entity_name, 'value') else str(node.entity_name)
        
        # Extract capability names
        capability_names = []
        for cap in node.capabilities:
            if hasattr(cap, 'value'):
                capability_names.append(cap.value)
            elif hasattr(cap, 'function') and hasattr(cap.function, 'value'):
                # Function call style
                capability_names.append(cap.function.value)
            else:
                capability_names.append(str(cap))
        
        # Grant capabilities
        try:
            manager.grant_capabilities(entity_name, capability_names)
            debug_log("eval_grant_statement", f"Granted {len(capability_names)} capabilities to {entity_name}")
            return String(f"Granted {len(capability_names)} capabilities to '{entity_name}'")
        except Exception as e:
            return String(f"Error granting capabilities: {e}")

    def eval_revoke_statement(self, node, env, stack_trace):
        """Evaluate revoke statement - revoke capabilities from entity."""
        from ..capability_system import get_capability_manager
        
        manager = get_capability_manager()
        
        # Get entity name
        entity_name = node.entity_name.value if hasattr(node.entity_name, 'value') else str(node.entity_name)
        
        # Extract capability names
        capability_names = []
        for cap in node.capabilities:
            if hasattr(cap, 'value'):
                capability_names.append(cap.value)
            elif hasattr(cap, 'function') and hasattr(cap.function, 'value'):
                capability_names.append(cap.function.value)
            else:
                capability_names.append(str(cap))
        
        # Revoke by removing from granted set (simple implementation)
        # In production, this would use a proper revocation mechanism
        try:
            # Access the manager's granted_capabilities
            if entity_name in manager.granted_capabilities:
                for cap_name in capability_names:
                    manager.granted_capabilities[entity_name].discard(cap_name)
            
            debug_log("eval_revoke_statement", f"Revoked {len(capability_names)} capabilities from {entity_name}")
            return String(f"Revoked {len(capability_names)} capabilities from '{entity_name}'")
        except Exception as e:
            return String(f"Error revoking capabilities: {e}")

    def eval_validate_statement(self, node, env, stack_trace):
        """Evaluate validate statement - validate data against schema."""
        from ..validation_system import (
            get_validation_manager, ValidationError, StandardValidators
        )
        
        manager = get_validation_manager()
        
        # Evaluate data expression
        data = self.eval_node(node.data, env, stack_trace)
        
        # Evaluate schema
        schema = None
        if node.schema:
            if isinstance(node.schema, dict):
                schema = node.schema
            elif hasattr(node.schema, 'pairs'):  # Map object
                # Convert Map to dict
                schema = {}
                for key, val in node.schema.pairs:
                    key_str = key.value if hasattr(key, 'value') else str(key)
                    schema[key_str] = val
            else:
                schema = self.eval_node(node.schema, env, stack_trace)
        
        # Validate data
        try:
            if isinstance(data, String):
                # Validate string against pattern or standard validator
                if isinstance(schema, String):
                    validator_name = schema.value
                    if hasattr(StandardValidators, validator_name.upper()):
                        validator = getattr(StandardValidators, validator_name.upper())
                        if validator.validate(data.value):
                            return String(f"Validation passed for {validator_name}")
                        else:
                            return String(f"Validation failed: {validator.get_error_message()}")
            
            # For complex validation, use schema
            if schema and hasattr(data, '__dict__'):
                manager.validate_schema(vars(data), str(schema) if not isinstance(schema, dict) else "custom")
            
            debug_log("eval_validate_statement", "Validation passed")
            return String("Validation passed")
        
        except ValidationError as e:
            debug_log("eval_validate_statement", f"Validation error: {e}")
            return String(f"Validation failed: {e}")

    def eval_sanitize_statement(self, node, env, stack_trace):
        """Evaluate sanitize statement - sanitize untrusted input."""
        from ..validation_system import Sanitizer, Encoding
        
        # Evaluate data to sanitize
        data = self.eval_node(node.data, env, stack_trace)
        
        # Convert to string
        if hasattr(data, 'value'):
            data_str = str(data.value)
        else:
            data_str = str(data)
        
        # Determine encoding
        encoding = Encoding.HTML  # Default
        if node.encoding:
            enc_val = self.eval_node(node.encoding, env, stack_trace)
            if hasattr(enc_val, 'value'):
                enc_name = enc_val.value.upper()
                try:
                    encoding = Encoding[enc_name]
                except KeyError:
                    encoding = Encoding.HTML
        
        # Sanitize
        try:
            sanitized = Sanitizer.sanitize_string(data_str, encoding)
            debug_log("eval_sanitize_statement", f"Sanitized {len(data_str)} chars with {encoding.value}")
            return String(sanitized)
        except Exception as e:
            debug_log("eval_sanitize_statement", f"Sanitization error: {e}")
            return String(data_str)  # Return original if sanitization fails

    def eval_inject_statement(self, node, env, stack_trace):
        """Evaluate inject statement - full dependency injection with mode-aware resolution."""
        from ..dependency_injection import get_di_registry, ExecutionMode
        from ..object import String as StringObj, Null as NullObj
        
        # Get dependency name
        dep_name = node.dependency.value if hasattr(node.dependency, 'value') else str(node.dependency)
        
        debug_log("eval_inject_statement", f"Resolving dependency: {dep_name}")
        
        # Get DI registry and current module context
        di_registry = get_di_registry()
        
        # Determine module name from environment context
        module_name = env.get("__module__") 
        module_name = module_name.value if module_name and hasattr(module_name, 'value') else "__main__"
        
        # Get or create container for this module
        container = di_registry.get_container(module_name)
        
        # Determine execution mode from environment or default to PRODUCTION
        mode_obj = env.get("__execution_mode__")
        if mode_obj and hasattr(mode_obj, 'value'):
            mode_str = mode_obj.value.upper()
            try:
                execution_mode = ExecutionMode[mode_str]
            except KeyError:
                execution_mode = ExecutionMode.PRODUCTION
        else:
            execution_mode = ExecutionMode.PRODUCTION
        
        # Set container's execution mode
        container.execution_mode = execution_mode
        
        try:
            # Attempt to resolve dependency
            resolved = container.get(dep_name)
            
            if resolved is not None:
                # Successfully resolved - store in environment
                env.set(dep_name, resolved)
                debug_log("eval_inject_statement", f"✓ Injected {dep_name} from container (mode: {execution_mode.name})")
                return StringObj(f"Dependency '{dep_name}' injected ({execution_mode.name} mode)")
            else:
                # Dependency not registered - create null placeholder
                debug_log("eval_inject_statement", f"⚠ Dependency {dep_name} not registered, using null")
                env.set(dep_name, NullObj())
                return StringObj(f"Warning: Dependency '{dep_name}' not registered")
                
        except Exception as e:
            # Error during resolution
            debug_log("eval_inject_statement", f"✗ Error injecting {dep_name}: {e}")
            env.set(dep_name, NullObj())
            return StringObj(f"Error: Could not inject '{dep_name}': {str(e)}")

    def eval_immutable_statement(self, node, env, stack_trace):
        """Evaluate immutable statement - declare variable as immutable."""
        from ..purity_system import get_immutability_manager
        
        manager = get_immutability_manager()
        
        # Get variable name
        var_name = node.target.value if hasattr(node.target, 'value') else str(node.target)
        
        # Evaluate and assign value if provided
        if node.value:
            value = self.eval_node(node.value, env, stack_trace)
            env.set(var_name, value)
            
            # Mark as immutable
            manager.mark_immutable(value)
            debug_log("eval_immutable_statement", f"Created immutable: {var_name}")
            return String(f"Immutable variable '{var_name}' created")
        else:
            # Mark existing variable as immutable
            try:
                value = env.get(var_name)
                manager.mark_immutable(value)
                debug_log("eval_immutable_statement", f"Marked immutable: {var_name}")
                return String(f"Variable '{var_name}' marked as immutable")
            except Exception as e:
                return String(f"Error: Variable '{var_name}' not found")


    # === COMPLEXITY & LARGE PROJECT MANAGEMENT STATEMENT EVALUATORS ===

    def eval_interface_statement(self, node, env, stack_trace):
        """Evaluate interface statement - define a contract/interface."""
        from ..complexity_system import get_complexity_manager
        
        manager = get_complexity_manager()
        
        # Get interface name
        interface_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Create interface from AST node
        from ..complexity_system import Interface
        interface = Interface(
            name=interface_name,
            methods=node.methods if hasattr(node, 'methods') else [],
            properties=node.properties if hasattr(node, 'properties') else {}
        )
        
        # Register interface
        manager.register_interface(interface)
        debug_log("eval_interface_statement", f"Registered interface: {interface_name}")
        
        # Store in environment
        env.set(interface_name, interface)
        return String(f"Interface '{interface_name}' defined")

    def eval_type_alias_statement(self, node, env, stack_trace):
        """Evaluate type alias statement - create type name shortcuts."""
        from ..complexity_system import get_complexity_manager
        
        manager = get_complexity_manager()
        
        # Get type alias name
        alias_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Get base type (just the string name, don't evaluate as expression)
        base_type = node.base_type.value if hasattr(node.base_type, 'value') else str(node.base_type)
        
        # Create type alias
        from ..complexity_system import TypeAlias
        alias = TypeAlias(
            name=alias_name,
            base_type=base_type
        )
        
        # Register type alias
        manager.register_type_alias(alias)
        debug_log("eval_type_alias_statement", f"Registered type alias: {alias_name} -> {base_type}")
        
        # Store in environment
        env.set(alias_name, alias)
        return String(f"Type alias '{alias_name}' defined")

    def eval_module_statement(self, node, env, stack_trace):
        """Evaluate module statement - create namespaced module."""
        from ..complexity_system import Module, ModuleMember, Visibility
        
        # Get module name
        module_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Create module
        module = Module(name=module_name)
        
        # Execute module body in new environment
        module_env = Environment(outer=env)
        
        # Track current module for export statement handling
        module_env.set('__current_module__', module)
        
        if hasattr(node, 'body') and node.body:
            self.eval_node(node.body, module_env, stack_trace)
        
        # Collect module members using AST modifiers when available
        seen = set()
        if hasattr(node, 'body') and getattr(node.body, 'statements', None):
            for stmt in node.body.statements:
                # Determine declared name and modifiers if present
                declared_name = None
                modifiers = getattr(stmt, 'modifiers', []) or []

                # Function / Action declarations
                if type(stmt).__name__ in ('FunctionStatement', 'ActionStatement'):
                    if hasattr(stmt.name, 'value'):
                        declared_name = stmt.name.value
                    else:
                        declared_name = str(stmt.name)
                    member_type = 'function'

                # Let / Const declarations
                elif type(stmt).__name__ in ('LetStatement', 'ConstStatement'):
                    if hasattr(stmt.name, 'value'):
                        declared_name = stmt.name.value
                    else:
                        declared_name = str(stmt.name)
                    member_type = 'variable'

                else:
                    # Not a direct declaration we can extract; skip to env-scan fallback
                    continue

                if declared_name:
                    seen.add(declared_name)
                    try:
                        value = module_env.get(declared_name)
                    except Exception:
                        value = None

                    # Map modifiers to visibility
                    vis = Visibility.PUBLIC
                    lower_mods = [m.lower() for m in modifiers]
                    if 'private' in lower_mods or 'internal' in lower_mods:
                        vis = Visibility.INTERNAL
                    elif 'protected' in lower_mods:
                        vis = Visibility.PROTECTED

                    member = ModuleMember(
                        name=declared_name,
                        member_type=member_type,
                        visibility=vis,
                        value=value
                    )
                    module.add_member(member)

        # Fallback: include any remaining env keys not discovered via AST
        for key in module_env.store:
            if key.startswith('_') or key in seen:
                continue
            try:
                value = module_env.get(key)
            except Exception:
                value = None
            try:
                is_callable = callable(value)
            except Exception:
                is_callable = False
            member_type = 'function' if is_callable else 'variable'
            member = ModuleMember(
                name=key,
                member_type=member_type,
                visibility=Visibility.PUBLIC,
                value=value
            )
            module.add_member(member)
        
        # Note: Module is stored directly in environment; manager integration can be enhanced later
        debug_log("eval_module_statement", f"Created module: {module_name}")
        debug_log("eval_module_statement", f"Module members: {list(module.members.keys())}")
        
        # Store in environment
        env.set(module_name, module)
        return String(f"Module '{module_name}' created")

    def eval_package_statement(self, node, env, stack_trace):
        """Evaluate package statement - create package with hierarchical support."""
        from ..complexity_system import Package
        
        # Get package name (may be dotted like app.api.v1)
        package_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Parse hierarchical package names
        name_parts = package_name.split('.')
        
        # Create the leaf package with body content
        leaf_package = Package(name=name_parts[-1])
        
        # Execute package body in new environment
        package_env = Environment(outer=env)
        
        if hasattr(node, 'body') and node.body:
            self.eval_node(node.body, package_env, stack_trace)
        
        # Collect package members from package environment
        for key in package_env.store:
            if not key.startswith('_'):
                value = package_env.get(key)
                leaf_package.modules[key] = value
        
        debug_log("eval_package_statement", f"Created package: {package_name} with members: {list(leaf_package.modules.keys())}")
        
        # Handle hierarchical package structure
        if len(name_parts) == 1:
            # Simple package (no hierarchy)
            env.set(package_name, leaf_package)
        else:
            # Hierarchical package - ensure all ancestors exist
            # Start from root and work down to leaf
            root_name = name_parts[0]
            root_package = env.get(root_name)
            
            if root_package is None or not hasattr(root_package, 'modules'):
                # Create root package if it doesn't exist
                root_package = Package(name=root_name)
                env.set(root_name, root_package)
                debug_log("eval_package_statement", f"Created root package: {root_name}")
            
            # Navigate/create intermediate packages
            current = root_package
            for i in range(1, len(name_parts)):
                part_name = name_parts[i]
                
                if i == len(name_parts) - 1:
                    # This is the leaf - add it
                    current.modules[part_name] = leaf_package
                    debug_log("eval_package_statement", f"Added {part_name} to {name_parts[i-1]}")
                else:
                    # This is an intermediate package
                    if part_name not in current.modules:
                        # Create intermediate package
                        intermediate = Package(name=part_name)
                        current.modules[part_name] = intermediate
                        debug_log("eval_package_statement", f"Created intermediate package: {part_name}")
                    current = current.modules[part_name]
        
        return String(f"Package '{package_name}' created")

    def eval_using_statement(self, node, env, stack_trace):
        """Evaluate using statement - RAII pattern for resource management."""
        from ..complexity_system import get_complexity_manager
        
        manager = get_complexity_manager()
        
        # Get resource name
        resource_name = node.resource_name.value if hasattr(node.resource_name, 'value') else str(node.resource_name)
        
        # Acquire resource
        resource = self.eval_node(node.resource_expr, env, stack_trace)
        
        # Store resource in environment
        env.set(resource_name, resource)
        
        try:
            debug_log("eval_using_statement", f"Acquired resource: {resource_name}")
            
            # Execute body in using block
            if hasattr(node, 'body') and node.body:
                body_result = self.eval_node(node.body, env, stack_trace)
            
            return body_result if 'body_result' in locals() else NULL
        
        finally:
            # Cleanup resource (RAII)
            if hasattr(resource, 'close'):
                try:
                    resource.close()
                    debug_log("eval_using_statement", f"Closed resource: {resource_name}")
                except Exception as e:
                    debug_log("eval_using_statement", f"Error closing resource: {e}")
            elif hasattr(resource, 'cleanup'):
                try:
                    resource.cleanup()
                    debug_log("eval_using_statement", f"Cleaned up resource: {resource_name}")
                except Exception as e:
                    debug_log("eval_using_statement", f"Error cleaning up resource: {e}")

    # === CONCURRENCY & PERFORMANCE STATEMENT EVALUATORS ===

    def eval_channel_statement(self, node, env, stack_trace):
        """Evaluate channel statement - declare a message passing channel."""
        from ..concurrency_system import get_concurrency_manager
        
        manager = get_concurrency_manager()
        
        # Get channel name
        channel_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Get element type if specified
        element_type = None
        if hasattr(node, 'element_type') and node.element_type:
            element_type = str(node.element_type)
        
        # Get capacity if specified
        capacity = 0
        if hasattr(node, 'capacity') and node.capacity:
            capacity = self.eval_node(node.capacity, env, stack_trace)
            if isinstance(capacity, Integer):
                capacity = capacity.value
            else:
                capacity = 0
        
        # Create channel
        channel = manager.create_channel(channel_name, element_type, capacity)
        debug_log("eval_channel_statement", f"Created channel: {channel_name}")
        
        # Store in environment
        env.set(channel_name, channel)
        return String(f"Channel '{channel_name}' created")

    def eval_send_statement(self, node, env, stack_trace):
        """Evaluate send statement - send value to a channel."""
        
        # Evaluate channel expression
        channel = self.eval_node(node.channel_expr, env, stack_trace)
        
        # Evaluate value to send
        value = self.eval_node(node.value_expr, env, stack_trace)
        
        # Send to channel
        if hasattr(channel, 'send'):
            try:
                channel.send(value, timeout=5.0)
                debug_log("eval_send_statement", f"Sent to channel: {value}")
                return String(f"Value sent to channel")
            except Exception as e:
                return String(f"Error sending to channel: {e}")
        else:
            return String(f"Error: not a valid channel")

    def eval_receive_statement(self, node, env, stack_trace):
        """Evaluate receive statement - receive value from a channel."""
        
        # Evaluate channel expression
        channel = self.eval_node(node.channel_expr, env, stack_trace)
        
        # Receive from channel
        if hasattr(channel, 'receive'):
            try:
                value = channel.receive(timeout=5.0)
                debug_log("eval_receive_statement", f"Received from channel: {value}")
                
                # Bind to target if specified
                if hasattr(node, 'target') and node.target:
                    target_name = node.target.value if hasattr(node.target, 'value') else str(node.target)
                    env.set(target_name, value)
                
                return value if value is not None else NULL
            except Exception as e:
                return String(f"Error receiving from channel: {e}")
        else:
            return String(f"Error: not a valid channel")

    def eval_atomic_statement(self, node, env, stack_trace):
        """Evaluate atomic statement - execute indivisible operation."""
        from ..concurrency_system import get_concurrency_manager
        
        manager = get_concurrency_manager()
        
        # Create/get atomic region
        atomic_id = f"atomic_{id(node)}"
        atomic = manager.create_atomic(atomic_id)
        
        # Execute atomically
        def execute_block():
            if hasattr(node, 'body') and node.body:
                # Atomic block
                return self.eval_node(node.body, env, stack_trace)
            elif hasattr(node, 'expr') and node.expr:
                # Atomic expression
                return self.eval_node(node.expr, env, stack_trace)
            return NULL
        
        result = atomic.execute(execute_block)
        debug_log("eval_atomic_statement", "Atomic operation completed")
        
        return result if result is not NULL else NULL

    # === BLOCKCHAIN STATEMENT EVALUATION ===
    
    def eval_ledger_statement(self, node, env, stack_trace):
        """Evaluate ledger statement - create immutable ledger variable.
        
        ledger balances = {};
        ledger state_root;
        """
        from ..blockchain import Ledger
        from ..blockchain.transaction import get_current_tx, create_tx_context
        
        debug_log("eval_ledger_statement", f"ledger {node.name.value}")
        
        # Ensure TX context exists
        tx = get_current_tx()
        if tx is None:
            tx = create_tx_context(caller="system", gas_limit=1000000)
        
        # Evaluate initial value if provided
        initial_value = NULL
        if node.initial_value:
            initial_value = self.eval_node(node.initial_value, env, stack_trace)
            if is_error(initial_value):
                return initial_value
        
        # Create ledger instance
        ledger_name = node.name.value
        ledger = Ledger(ledger_name)
        
        # Write initial value if provided
        if initial_value != NULL:
            # Convert Zexus object to Python value for storage
            py_value = _zexus_to_python(initial_value)
            ledger.write(ledger_name, py_value, tx.block_hash)
        
        # Store the value directly in environment (ledger is for tracking history)
        env.set(node.name.value, initial_value)
        
        debug_log("eval_ledger_statement", f"Created ledger: {node.name.value}")
        return NULL
    
    def eval_state_statement(self, node, env, stack_trace):
        """Evaluate state statement - create mutable state variable.
        
        state counter = 0;
        state owner = TX.caller;
        """
        debug_log("eval_state_statement", f"state {node.name.value}")
        
        # Evaluate initial value
        value = NULL
        if node.initial_value:
            value = self.eval_node(node.initial_value, env, stack_trace)
            if is_error(value):
                return value
        
        # Store in environment (regular mutable variable)
        env.set(node.name.value, value)
        
        debug_log("eval_state_statement", f"Created state: {node.name.value}")
        return NULL
    
    def eval_require_statement(self, node, env, stack_trace):
        """Evaluate require statement - prerequisites, dependencies, resources.
        
        Basic:
            require(balance >= amount);
            require(TX.caller == owner, "Only owner");
        
        With tolerance:
            require balance >= 0.1 { tolerance_logic() }
        
        File/Module dependencies:
            require \"file.zx\" imported, \"File required\";
            require module \"db\" available, \"Database required\";
        
        Resource requirements:
            require:balance amount >= minimum;
            require:gas available >= needed;
        """
        debug_log("eval_require_statement", "Checking requirement")
        
        # Handle file dependencies
        if node.requirement_type == 'file' and node.file_path:
            return self._eval_require_file(node, env, stack_trace)
        
        # Handle module dependencies
        if node.requirement_type == 'module' and node.module_name:
            return self._eval_require_module(node, env, stack_trace)
        
        # Handle resource requirements
        if node.requirement_type in ['balance', 'gas', 'prereq']:
            return self._eval_require_resource(node, env, stack_trace)
        
        # Standard condition requirement
        if node.condition:
            # Evaluate condition
            condition = self.eval_node(node.condition, env, stack_trace)
            if is_error(condition):
                return condition
            
            # Check if condition is true
            if not is_truthy(condition):
                # Execute tolerance block if provided (for conditional allowances)
                if node.tolerance_block:
                    print(f"⚡ TOLERANCE BLOCK: type={type(node.tolerance_block).__name__}")
                    debug_log("eval_require_statement", "Condition failed - executing tolerance logic")
                    tolerance_result = self.eval_node(node.tolerance_block, env, stack_trace)
                    print(f"⚡ TOLERANCE RESULT: type={type(tolerance_result).__name__}")
                    
                    # Check if tolerance logic allows proceeding
                    if is_error(tolerance_result):
                        return tolerance_result
                    
                    # Unwrap ReturnValue if present
                    from ..object import ReturnValue
                    if isinstance(tolerance_result, ReturnValue):
                        print(f"⚡ UNWRAPPING ReturnValue")
                        tolerance_result = tolerance_result.value
                        print(f"⚡ UNWRAPPED VALUE: {tolerance_result}")
                    
                    # If tolerance block returns true/truthy, allow it
                    if is_truthy(tolerance_result):
                        print(f"⚡ TOLERANCE APPROVED")
                        debug_log("eval_require_statement", "Tolerance logic approved - allowing requirement")
                        return NULL
                    
                    # If tolerance block returns false, requirement still fails
                    print(f"⚡ TOLERANCE REJECTED")
                    debug_log("eval_require_statement", "Tolerance logic rejected - requirement fails")
                    # Fall through to error below
                
                # Evaluate error message
                message = "Requirement not met"
                if node.message:
                    msg_val = self.eval_node(node.message, env, stack_trace)
                    if isinstance(msg_val, String):
                        message = msg_val.value
                    elif not is_error(msg_val):
                        message = str(msg_val.inspect() if hasattr(msg_val, 'inspect') else msg_val)
                
                # Trigger revert
                debug_log("eval_require_statement", f"REVERT: {message}")
                return EvaluationError(f"Requirement failed: {message}", stack_trace=stack_trace)
            
            debug_log("eval_require_statement", "Requirement satisfied")
            return NULL
        
        # No condition or special type
        return EvaluationError("Invalid require statement: missing condition")
    
    def _eval_require_file(self, node, env, stack_trace):
        """Evaluate file dependency requirement."""
        from ..object import Boolean, String
        import os
        
        file_path = node.file_path
        debug_log("_eval_require_file", f"Checking if {file_path} is imported")
        
        # Check if file was imported
        # Look for the file in imported modules
        imported_files = env.get('__imported_files__') if hasattr(env, 'get') else set()
        
        # Also check if file exists
        file_exists = os.path.exists(file_path)
        file_imported = file_path in imported_files if isinstance(imported_files, set) else False
        
        if not file_imported and not file_exists:
            message = f"Required file '{file_path}' not imported"
            if node.message:
                msg_val = self.eval_node(node.message, env, stack_trace)
                if isinstance(msg_val, String):
                    message = msg_val.value
            
            return EvaluationError(f"File dependency: {message}", stack_trace=stack_trace)
        
        debug_log("_eval_require_file", f"File {file_path} available")
        return NULL
    
    def _eval_require_module(self, node, env, stack_trace):
        """Evaluate module dependency requirement."""
        from ..object import Boolean, String
        
        module_name = node.module_name
        debug_log("_eval_require_module", f"Checking if module '{module_name}' is available")
        
        # Check if module is loaded/available
        # Look in environment for the module
        module_available = False
        
        if hasattr(env, 'get'):
            module_obj = env.get(module_name)
            module_available = module_obj is not None
        
        # Also check Python sys.modules for Python modules
        if not module_available:
            import sys
            module_available = module_name in sys.modules
        
        if not module_available:
            message = f"Required module '{module_name}' not available"
            if node.message:
                msg_val = self.eval_node(node.message, env, stack_trace)
                if isinstance(msg_val, String):
                    message = msg_val.value
            
            return EvaluationError(f"Module dependency: {message}", stack_trace=stack_trace)
        
        debug_log("_eval_require_module", f"Module '{module_name}' available")
        return NULL
    
    def _eval_require_resource(self, node, env, stack_trace):
        """Evaluate resource requirement (balance, gas, prerequisites)."""
        from ..object import Boolean, String
        
        req_type = node.requirement_type
        debug_log("_eval_require_resource", f"Checking {req_type} requirement")
        
        # Evaluate condition
        condition = self.eval_node(node.condition, env, stack_trace)
        if is_error(condition):
            return condition
        
        # Check condition
        if not is_truthy(condition):
            # Execute tolerance block if provided
            if node.tolerance_block:
                debug_log("_eval_require_resource", f"{req_type} requirement not met - checking tolerance")
                tolerance_result = self.eval_node(node.tolerance_block, env, stack_trace)
                
                if is_error(tolerance_result):
                    return tolerance_result
                
                if is_truthy(tolerance_result):
                    debug_log("_eval_require_resource", f"Tolerance approved for {req_type}")
                    return NULL
            
            # Requirement not met
            message = f"{req_type.capitalize()} requirement not met"
            if node.message:
                msg_val = self.eval_node(node.message, env, stack_trace)
                if isinstance(msg_val, String):
                    message = msg_val.value
            
            return EvaluationError(f"Resource requirement: {message}", stack_trace=stack_trace)
        
        debug_log("_eval_require_resource", f"{req_type} requirement satisfied")
        return NULL
    
    def eval_revert_statement(self, node, env, stack_trace):
        """Evaluate revert statement - rollback transaction.
        
        revert();
        revert("Unauthorized");
        """
        debug_log("eval_revert_statement", "Reverting transaction")
        
        # Evaluate revert reason if provided
        reason = "Transaction reverted"
        if node.reason:
            reason_val = self.eval_node(node.reason, env, stack_trace)
            if isinstance(reason_val, String):
                reason = reason_val.value
            elif not is_error(reason_val):
                reason = str(reason_val.inspect() if hasattr(reason_val, 'inspect') else reason_val)
        
        debug_log("eval_revert_statement", f"REVERT: {reason}")
        return EvaluationError(f"Transaction reverted: {reason}", stack_trace=stack_trace)
    
    def eval_limit_statement(self, node, env, stack_trace):
        """Evaluate limit statement - set gas limit.
        
        limit(5000);
        """
        from ..blockchain import get_current_tx
        
        debug_log("eval_limit_statement", "Setting gas limit")
        
        # Evaluate gas limit amount
        limit_val = self.eval_node(node.amount, env, stack_trace)
        if is_error(limit_val):
            return limit_val
        
        # Extract numeric value
        if isinstance(limit_val, Integer):
            limit_amount = limit_val.value
        else:
            return EvaluationError(f"Gas limit must be an integer, got {type(limit_val).__name__}")
        
        # Get current transaction context
        tx = get_current_tx()
        if tx:
            tx.gas_limit = limit_amount
            debug_log("eval_limit_statement", f"Set gas limit to {limit_amount}")
        else:
            debug_log("eval_limit_statement", "No active TX context, limit statement ignored")
        
        return NULL
    
    # === BLOCKCHAIN EXPRESSION EVALUATION ===
    
    def eval_tx_expression(self, node, env, stack_trace):
        """Evaluate TX expression - access transaction context.
        
        TX.caller
        TX.timestamp
        TX.gas_remaining
        """
        from ..blockchain import get_current_tx
        
        tx = get_current_tx()
        if not tx:
            debug_log("eval_tx_expression", "No active TX context")
            return NULL
        
        # If no property specified, return the TX object itself
        if not node.property_name:
            # Wrap TX context as Zexus object
            return _python_to_zexus(tx)
        
        # Access specific property
        prop = node.property_name
        if prop == "caller":
            return String(tx.caller)
        elif prop == "timestamp":
            return Integer(tx.timestamp)
        elif prop == "block_hash":
            return String(tx.block_hash)
        elif prop == "gas_limit":
            return Integer(tx.gas_limit)
        elif prop == "gas_used":
            return Integer(tx.gas_used)
        elif prop == "gas_remaining":
            return Integer(tx.gas_remaining)
        else:
            return EvaluationError(f"Unknown TX property: {prop}")
    
    def eval_hash_expression(self, node, env, stack_trace):
        """Evaluate hash expression - cryptographic hashing.
        
        hash(data, "SHA256")
        hash(message, "KECCAK256")
        """
        from ..blockchain.crypto import CryptoPlugin
        
        # Evaluate data to hash
        data_val = self.eval_node(node.data, env, stack_trace)
        if is_error(data_val):
            return data_val
        
        # Evaluate algorithm
        algorithm_val = self.eval_node(node.algorithm, env, stack_trace)
        if is_error(algorithm_val):
            return algorithm_val
        
        # Convert to string values
        if isinstance(data_val, String):
            data = data_val.value
        else:
            data = str(data_val.inspect() if hasattr(data_val, 'inspect') else data_val)
        
        if isinstance(algorithm_val, String):
            algorithm = algorithm_val.value
        else:
            algorithm = str(algorithm_val)
        
        # Perform hashing
        try:
            hash_result = CryptoPlugin.hash_data(data, algorithm)
            return String(hash_result)
        except Exception as e:
            return EvaluationError(f"Hash error: {str(e)}")
    
    def eval_signature_expression(self, node, env, stack_trace):
        """Evaluate signature expression - create digital signature.
        
        signature(data, private_key, "ECDSA")
        """
        from ..blockchain.crypto import CryptoPlugin
        
        # Evaluate arguments
        data_val = self.eval_node(node.data, env, stack_trace)
        if is_error(data_val):
            return data_val
        
        key_val = self.eval_node(node.private_key, env, stack_trace)
        if is_error(key_val):
            return key_val
        
        algorithm_val = self.eval_node(node.algorithm, env, stack_trace) if node.algorithm else String("ECDSA")
        if is_error(algorithm_val):
            return algorithm_val
        
        # Convert to string values
        data = data_val.value if isinstance(data_val, String) else str(data_val)
        private_key = key_val.value if isinstance(key_val, String) else str(key_val)
        algorithm = algorithm_val.value if isinstance(algorithm_val, String) else str(algorithm_val)
        
        # Create signature
        try:
            signature = CryptoPlugin.sign_data(data, private_key, algorithm)
            return String(signature)
        except Exception as e:
            return EvaluationError(f"Signature error: {str(e)}")
    
    def eval_verify_signature_expression(self, node, env, stack_trace):
        """Evaluate verify_sig expression - verify digital signature.
        
        verify_sig(data, signature, public_key, "ECDSA")
        """
        from ..blockchain.crypto import CryptoPlugin
        
        # Evaluate arguments
        data_val = self.eval_node(node.data, env, stack_trace)
        if is_error(data_val):
            return data_val
        
        sig_val = self.eval_node(node.signature, env, stack_trace)
        if is_error(sig_val):
            return sig_val
        
        key_val = self.eval_node(node.public_key, env, stack_trace)
        if is_error(key_val):
            return key_val
        
        algorithm_val = self.eval_node(node.algorithm, env, stack_trace) if node.algorithm else String("ECDSA")
        if is_error(algorithm_val):
            return algorithm_val
        
        # Convert to string values
        data = data_val.value if isinstance(data_val, String) else str(data_val)
        signature = sig_val.value if isinstance(sig_val, String) else str(sig_val)
        public_key = key_val.value if isinstance(key_val, String) else str(key_val)
        algorithm = algorithm_val.value if isinstance(algorithm_val, String) else str(algorithm_val)
        
        # Verify signature
        try:
            is_valid = CryptoPlugin.verify_signature(data, signature, public_key, algorithm)
            return TRUE if is_valid else FALSE
        except Exception as e:
            return EvaluationError(f"Signature verification error: {str(e)}")
    
    def eval_gas_expression(self, node, env, stack_trace):
        """Evaluate gas expression - access gas tracking.
        
        gas.used
        gas.remaining
        gas.limit
        """
        from ..blockchain import get_current_tx
        
        tx = get_current_tx()
        if not tx:
            debug_log("eval_gas_expression", "No active TX context")
            return NULL
        
        # If no property specified, return gas info as object
        if not node.property_name:
            gas_info = {
                "limit": Integer(tx.gas_limit),
                "used": Integer(tx.gas_used),
                "remaining": Integer(tx.gas_remaining)
            }
            return Map(gas_info)
        
        # Access specific property
        prop = node.property_name
        if prop == "limit":
            return Integer(tx.gas_limit)
        elif prop == "used":
            return Integer(tx.gas_used)
        elif prop == "remaining":
            return Integer(tx.gas_remaining)
        else:
            return EvaluationError(f"Unknown gas property: {prop}")

    def eval_protocol_statement(self, node, env, stack_trace):
        """Evaluate PROTOCOL statement - define an interface/trait.
        
        protocol Transferable {
            action transfer(to, amount)
            action balance() -> int
        }
        """
        from ..object import String as StringObj
        
        # Create protocol definition (similar to entity but for interfaces)
        protocol_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Store method signatures
        methods = {}
        for method in node.methods:
            method_name = method.name.value if hasattr(method.name, 'value') else str(method.name)
            methods[method_name] = {
                'params': method.parameters if hasattr(method, 'parameters') else [],
                'return_type': method.return_type if hasattr(method, 'return_type') else None
            }
        
        # Create protocol object
        protocol_def = {
            'type': 'protocol',
            'name': protocol_name,
            'methods': methods
        }
        
        # Store in environment
        env.set(protocol_name, protocol_def)
        
        return StringObj(f"Protocol '{protocol_name}' defined with {len(methods)} methods")
    
    def eval_persistent_statement(self, node, env, stack_trace):
        """Evaluate PERSISTENT statement - declare persistent storage in contracts.
        
        persistent storage balances: map
        persistent storage owner: string = "0x0"
        """
        from ..object import NULL
        
        # Get variable name
        var_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Evaluate initial value if provided
        value = NULL
        if node.initial_value:
            value = self.eval_node(node.initial_value, env, stack_trace)
            if is_error(value):
                return value
        
        # Mark as persistent in environment (special marker)
        env.set(f"__persistent_{var_name}__", True)
        
        # Store the actual value
        env.set(var_name, value)
        
        return NULL

    def eval_this_expression(self, node, env, stack_trace):
        """Evaluate THIS expression - reference to current contract instance, data method instance, or entity instance.
        
        this.balances[account]
        this.owner = TX.caller
        this.width  # in data method
        this.logger.log()  # in entity method
        """
        from ..object import EvaluationError
        
        # Look for current contract instance in environment
        contract_instance = env.get("__contract_instance__")
        
        if contract_instance is not None:
            return contract_instance
        
        # For data methods and entity methods, look for 'this' binding
        instance = env.get("this")
        
        if instance is not None:
            return instance
        
        return EvaluationError("'this' can only be used inside a contract, data method, or entity method")

    def eval_emit_statement(self, node, env, stack_trace):
        """Evaluate EMIT statement - emit an event.
        
        emit Transfer(from, to, amount);
        emit StateChange("balance_updated", new_balance);
        """
        from ..object import NULL, String
        
        # Get event name
        event_name = node.event_name.value if hasattr(node.event_name, 'value') else str(node.event_name)
        
        # Evaluate arguments
        args = []
        for arg in node.arguments:
            val = self.eval_node(arg, env, stack_trace)
            if is_error(val):
                return val
            args.append(val)
        
        # Get or create event log in environment
        event_log = env.get("__event_log__")
        if event_log is None:
            event_log = []
            env.set("__event_log__", event_log)
        
        # Add event to log
        event_data = {
            "event": event_name,
            "args": args
        }
        event_log.append(event_data)
        
        # Print event for debugging (optional)
        args_str = ", ".join(str(arg.inspect() if hasattr(arg, 'inspect') else arg) for arg in args)
        print(f"📢 Event: {event_name}({args_str})")
        
        return NULL

    def eval_modifier_declaration(self, node, env, stack_trace):
        """Evaluate MODIFIER declaration - store modifier for later use.
        
        modifier onlyOwner {
            require(TX.caller == owner, \"Not owner\");
        }
        """
        from ..object import Modifier, NULL
        
        # Get modifier name
        modifier_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Create modifier object
        modifier = Modifier(
            name=modifier_name,
            parameters=node.parameters,
            body=node.body,
            env=env
        )
        
        # Store modifier in environment
        env.set(modifier_name, modifier)
        
        return NULL
    # === CONCURRENCY STATEMENT EVALUATORS ===
    
    def eval_channel_statement(self, node, env, stack_trace):
        """Evaluate channel declaration: channel<T> name [= capacity]
        
        Examples:
            channel<integer> numbers
            channel<string> messages = 10
        """
        from ..concurrency_system import Channel
        
        # Get channel name
        channel_name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        
        # Get element type (optional)
        element_type = node.element_type if hasattr(node, 'element_type') else None
        if element_type and hasattr(element_type, 'value'):
            element_type = element_type.value
        
        # Get capacity (optional, default 0 = unbuffered)
        capacity = 0
        if hasattr(node, 'capacity') and node.capacity:
            cap_val = self.eval_node(node.capacity, env, stack_trace)
            if is_error(cap_val):
                return cap_val
            if isinstance(cap_val, Integer):
                capacity = cap_val.value
        
        # Create channel object
        channel = Channel(
            name=channel_name,
            element_type=element_type,
            capacity=capacity
        )
        
        # Store in environment
        env.set(channel_name, channel)
        
        debug_log("eval_channel_statement", f"Created channel '{channel_name}' (capacity={capacity})")
        return NULL

    def eval_send_statement(self, node, env, stack_trace):
        """Evaluate send statement: send(channel, value)
        
        This is for the statement form, not the builtin function.
        """
        # Evaluate channel
        channel = self.eval_node(node.channel_expr, env, stack_trace)
        if is_error(channel):
            return channel
        
        # Evaluate value
        value = self.eval_node(node.value_expr, env, stack_trace)
        if is_error(value):
            return value
        
        # Send to channel
        if not hasattr(channel, 'send'):
            return EvaluationError(f"send target is not a channel: {type(channel).__name__}")
        
        try:
            channel.send(value, timeout=5.0)
            return NULL
        except Exception as e:
            return EvaluationError(f"send error: {str(e)}")

    def eval_receive_statement(self, node, env, stack_trace):
        """Evaluate receive statement: value = receive(channel)
        
        This is for the statement form, not the builtin function.
        """
        # Evaluate channel
        channel = self.eval_node(node.channel_expr, env, stack_trace)
        if is_error(channel):
            return channel
        
        # Receive from channel
        if not hasattr(channel, 'receive'):
            return EvaluationError(f"receive target is not a channel: {type(channel).__name__}")
        
        try:
            value = channel.receive(timeout=5.0)
            return value if value is not None else NULL
        except Exception as e:
            return EvaluationError(f"receive error: {str(e)}")

    def eval_atomic_statement(self, node, env, stack_trace):
        """Evaluate atomic block: atomic { statements }
        
        Ensures all statements execute atomically (thread-safe).
        Uses a global lock to serialize atomic blocks.
        """
        from threading import Lock
        
        # Global atomic lock (class-level to share across all evaluators)
        if not hasattr(self.__class__, '_atomic_lock'):
            self.__class__._atomic_lock = Lock()
        
        # Execute block under lock
        with self.__class__._atomic_lock:
            result = self.eval_node(node.body, env, stack_trace)
            if is_error(result):
                return result
            return result if not isinstance(result, ReturnValue) else result