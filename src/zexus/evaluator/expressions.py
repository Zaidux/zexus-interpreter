# src/zexus/evaluator/expressions.py
from ..zexus_ast import (
    IntegerLiteral, FloatLiteral, StringLiteral, ListLiteral, MapLiteral, 
    Identifier, PrefixExpression, InfixExpression, IfExpression, 
    Boolean as AST_Boolean, EmbeddedLiteral, ActionLiteral, LambdaExpression
)
from ..object import (
    Integer, Float, String, List, Map,
    EvaluationError, Builtin, DateTime
)
from .utils import is_error, debug_log, NULL, TRUE, FALSE, is_truthy

class ExpressionEvaluatorMixin:
    """Handles evaluation of expressions: Literals, Math, Logic, Identifiers."""
    
    def eval_identifier(self, node, env):
        debug_log("eval_identifier", f"Looking up: {node.value}")
        
        # Special case: 'this' keyword should be treated like ThisExpression
        if node.value == "this":
            # Look for contract instance first
            contract_instance = env.get("__contract_instance__")
            if contract_instance is not None:
                return contract_instance
            
            # Then look for data method instance
            data_instance = env.get("this")
            if data_instance is not None:
                return data_instance
        
        # First, check environment for user-defined variables (including DATA dataclasses)
        val = env.get(node.value)
        if val:
            debug_log("  Found in environment", f"{node.value} = {val}")
            return val
        
        # Check builtins (self.builtins should be defined in FunctionEvaluatorMixin)
        if hasattr(self, 'builtins'):
            builtin = self.builtins.get(node.value)
            if builtin:
                debug_log("  Found builtin", f"{node.value} = {builtin}")
                return builtin
        
        # Special handling for TX - ONLY if not already defined by user
        # This provides blockchain transaction context when TX is not a user dataclass
        if node.value == "TX":
            from ..blockchain.transaction import get_current_tx, create_tx_context
            tx = get_current_tx()
            if tx is None:
                # Auto-create TX context if not exists
                tx = create_tx_context(caller="system", gas_limit=1000000)
            # Wrap TX context as a Zexus Map object for property access
            # Use plain string keys (not String objects) for Map.get() compatibility
            return Map({
                "caller": String(tx.caller),
                "timestamp": Integer(int(tx.timestamp)),
                "block_hash": String(tx.block_hash),
                "gas_used": Integer(tx.gas_used),
                "gas_remaining": Integer(tx.gas_remaining),
                "gas_limit": Integer(tx.gas_limit)
            })
        
        try:
            env_keys = []
            if hasattr(env, 'store'):
                env_keys = list(env.store.keys())
            # Use direct print to ensure visibility during debugging
            import traceback as _tb
            stack_snip = ''.join(_tb.format_stack(limit=5)[-3:])
            # print(f"[DEBUG] Identifier not found: {node.value}; env_keys={env_keys}\nStack snippet:\n{stack_snip}")
        except Exception:
            pass # print(f"[DEBUG] Identifier not found: {node.value}")
        
        # Try to find similar names for helpful suggestion
        suggestion = None
        if hasattr(env, 'store'):
            env_keys = list(env.store.keys())
            
            # Find similar variable names (simple approach)
            def similarity(a, b):
                a, b = a.lower(), b.lower()
                if a == b:
                    return 1.0
                if a in b or b in a:
                    return 0.8
                if len(a) > 2 and len(b) > 2:
                    if a[:3] == b[:3] or a[-3:] == b[-3:]:
                        return 0.6
                return 0.0
            
            similar = [(key, similarity(node.value, key)) for key in env_keys]
            similar = [(k, s) for k, s in similar if s > 0.5]
            similar.sort(key=lambda x: x[1], reverse=True)
            
            if similar:
                suggestion = f"Did you mean '{similar[0][0]}'?"
            elif env_keys:
                suggestion = f"Declare the variable first with 'let' or 'const'. Available: {', '.join(env_keys[:5])}"
            else:
                suggestion = "No variables declared yet. Use 'let variableName = value' to create one."
        
        return EvaluationError(
            f"Identifier '{node.value}' not found",
            suggestion=suggestion
        )
    
    def eval_integer_infix(self, operator, left, right):
        left_val = left.value
        right_val = right.value
        
        if operator == "+": 
            return Integer(left_val + right_val)
        elif operator == "-": 
            return Integer(left_val - right_val)
        elif operator == "*": 
            return Integer(left_val * right_val)
        elif operator == "/":
            if right_val == 0: 
                return EvaluationError(
                    "Division by zero",
                    suggestion="Check your divisor value. Consider adding a condition: if (divisor != 0) { ... }"
                )
            return Integer(left_val // right_val)
        elif operator == "%":
            if right_val == 0: 
                return EvaluationError(
                    "Modulo by zero",
                    suggestion="Check your divisor value. Modulo operation requires a non-zero divisor."
                )
            return Integer(left_val % right_val)
        elif operator == "<": 
            return TRUE if left_val < right_val else FALSE
        elif operator == ">": 
            return TRUE if left_val > right_val else FALSE
        elif operator == "<=": 
            return TRUE if left_val <= right_val else FALSE
        elif operator == ">=": 
            return TRUE if left_val >= right_val else FALSE
        elif operator == "==": 
            return TRUE if left_val == right_val else FALSE
        elif operator == "!=": 
            return TRUE if left_val != right_val else FALSE
        
        return EvaluationError(f"Unknown integer operator: {operator}")
    
    def eval_float_infix(self, operator, left, right):
        left_val = left.value
        right_val = right.value
        
        if operator == "+": 
            return Float(left_val + right_val)
        elif operator == "-": 
            return Float(left_val - right_val)
        elif operator == "*": 
            return Float(left_val * right_val)
        elif operator == "/":
            if right_val == 0: 
                return EvaluationError("Division by zero")
            return Float(left_val / right_val)
        elif operator == "<": 
            return TRUE if left_val < right_val else FALSE
        elif operator == ">": 
            return TRUE if left_val > right_val else FALSE
        elif operator == "<=": 
            return TRUE if left_val <= right_val else FALSE
        elif operator == ">=": 
            return TRUE if left_val >= right_val else FALSE
        elif operator == "==": 
            return TRUE if left_val == right_val else FALSE
        elif operator == "!=": 
            return TRUE if left_val != right_val else FALSE
        
        return EvaluationError(f"Unknown float operator: {operator}")
    
    def eval_string_infix(self, operator, left, right):
        if operator == "+": 
            return String(left.value + right.value)
        elif operator == "==": 
            return TRUE if left.value == right.value else FALSE
        elif operator == "!=": 
            return TRUE if left.value != right.value else FALSE
        elif operator == "*":
            # String repetition: "x" * 3 = "xxx"
            # Only works with String * Integer, not String * String
            return EvaluationError(f"Type mismatch: STRING * STRING (use STRING * INTEGER for repetition)")
        return EvaluationError(f"Unknown string operator: {operator}")
    
    def eval_infix_expression(self, node, env, stack_trace):
        debug_log("eval_infix_expression", f"{node.left} {node.operator} {node.right}")
        
        left = self.eval_node(node.left, env, stack_trace)
        if is_error(left): 
            return left
        
        right = self.eval_node(node.right, env, stack_trace)
        if is_error(right): 
            return right

        # (removed debug instrumentation)
        
        operator = node.operator
        
        # Check for operator overloading in left operand (for dataclasses)
        if isinstance(left, Map) and hasattr(left, 'pairs'):
            operator_key = String(f"__operator_{operator}__")
            if operator_key in left.pairs:
                operator_method = left.pairs[operator_key]
                if isinstance(operator_method, Builtin):
                    # Call the operator method with right operand
                    result = operator_method.fn(right)
                    debug_log("  Operator overload called", f"{operator} on {left}")
                    return result
        
        # Logical Operators (short-circuiting)
        if operator == "&&":
            return TRUE if is_truthy(left) and is_truthy(right) else FALSE
        elif operator == "||":
            return TRUE if is_truthy(left) or is_truthy(right) else FALSE
        
        # Equality operators
        elif operator == "==":
            if hasattr(left, 'value') and hasattr(right, 'value'):
                return TRUE if left.value == right.value else FALSE
            return TRUE if left == right else FALSE
        elif operator == "!=":
            if hasattr(left, 'value') and hasattr(right, 'value'):
                return TRUE if left.value != right.value else FALSE
            return TRUE if left != right else FALSE
        
        # Type-specific dispatch
        if isinstance(left, Integer) and isinstance(right, Integer):
            return self.eval_integer_infix(operator, left, right)
        elif isinstance(left, Float) and isinstance(right, Float):
            return self.eval_float_infix(operator, left, right)
        elif isinstance(left, String) and isinstance(right, String):
            return self.eval_string_infix(operator, left, right)
        
        # String repetition: "x" * 100 or 100 * "x"
        elif operator == "*":
            if isinstance(left, String) and isinstance(right, Integer):
                # "x" * 100
                return String(left.value * right.value)
            elif isinstance(left, Integer) and isinstance(right, String):
                # 100 * "x"
                return String(right.value * left.value)
        
        # Array Concatenation
        elif operator == "+" and isinstance(left, List) and isinstance(right, List):
            # Concatenate two arrays: [1, 2] + [3, 4] = [1, 2, 3, 4]
            new_elements = left.elements[:] + right.elements[:]
            return List(new_elements)
        
        # DateTime arithmetic
        elif isinstance(left, DateTime) and isinstance(right, DateTime):
            # DateTime - DateTime = time difference in seconds (as Float)
            if operator == "-":
                diff = left.timestamp - right.timestamp
                # Return the difference as a Float in seconds
                return Float(diff)
            else:
                return EvaluationError(f"Unsupported operation: DATETIME {operator} DATETIME")
        elif isinstance(left, DateTime) and isinstance(right, (Integer, Float)):
            # DateTime + Number or DateTime - Number (add/subtract seconds)
            if operator == "+":
                new_timestamp = left.timestamp + float(right.value)
                return DateTime(new_timestamp)
            elif operator == "-":
                new_timestamp = left.timestamp - float(right.value)
                return DateTime(new_timestamp)
            else:
                return EvaluationError(f"Unsupported operation: DATETIME {operator} {right.type()}")
        elif isinstance(left, (Integer, Float)) and isinstance(right, DateTime):
            # Number + DateTime (add seconds to datetime)
            if operator == "+":
                new_timestamp = right.timestamp + float(left.value)
                return DateTime(new_timestamp)
            else:
                return EvaluationError(f"Unsupported operation: {left.type()} {operator} DATETIME")
        
        # Mixed String Concatenation
        elif operator == "+":
            if isinstance(left, String):
                right_str = right.inspect() if not isinstance(right, String) else right.value
                return String(left.value + str(right_str))
            elif isinstance(right, String):
                left_str = left.inspect() if not isinstance(left, String) else left.value
                return String(str(left_str) + right.value)
            # Mixed Numeric
            elif isinstance(left, (Integer, Float)) and isinstance(right, (Integer, Float)):
                l_val = float(left.value)
                r_val = float(right.value)
                return Float(l_val + r_val)
        
        # Mixed arithmetic operations (String coerced to number for *, -, /, %)
        elif operator in ("*", "-", "/", "%"):
            # Try to coerce strings to numbers for arithmetic
            l_val = None
            r_val = None
            
            # Get left value
            if isinstance(left, (Integer, Float)):
                l_val = float(left.value)
            elif isinstance(left, String):
                try:
                    l_val = float(left.value)
                except ValueError:
                    pass
            
            # Get right value
            if isinstance(right, (Integer, Float)):
                r_val = float(right.value)
            elif isinstance(right, String):
                try:
                    r_val = float(right.value)
                except ValueError:
                    pass
            
            # Perform operation if both values could be coerced
            if l_val is not None and r_val is not None:
                try:
                    if operator == "*":
                        result = l_val * r_val
                    elif operator == "-":
                        result = l_val - r_val
                    elif operator == "/":
                        if r_val == 0:
                            return EvaluationError("Division by zero")
                        result = l_val / r_val
                    elif operator == "%":
                        if r_val == 0:
                            return EvaluationError("Modulo by zero")
                        result = l_val % r_val
                    
                    # Return Integer if result is whole number, Float otherwise
                    if result == int(result):
                        return Integer(int(result))
                    return Float(result)
                except Exception as e:
                    return EvaluationError(f"Arithmetic error: {str(e)}")
        
        # Comparison with mixed numeric types
        elif operator in ("<", ">", "<=", ">="):
            if isinstance(left, (Integer, Float)) and isinstance(right, (Integer, Float)):
                l_val = float(left.value)
                r_val = float(right.value)
                if operator == "<": return TRUE if l_val < r_val else FALSE
                elif operator == ">": return TRUE if l_val > r_val else FALSE
                elif operator == "<=": return TRUE if l_val <= r_val else FALSE
                elif operator == ">=": return TRUE if l_val >= r_val else FALSE
            
            # Mixed String/Number comparison (Coerce to float)
            elif (isinstance(left, (Integer, Float)) and isinstance(right, String)) or \
                 (isinstance(left, String) and isinstance(right, (Integer, Float))):
                try:
                    l_val = float(left.value)
                    r_val = float(right.value)
                    if operator == "<": return TRUE if l_val < r_val else FALSE
                    elif operator == ">": return TRUE if l_val > r_val else FALSE
                    elif operator == "<=": return TRUE if l_val <= r_val else FALSE
                    elif operator == ">=": return TRUE if l_val >= r_val else FALSE
                except ValueError:
                    # If conversion fails, return FALSE (NaN comparison behavior)
                    return FALSE

        return EvaluationError(f"Type mismatch: {left.type()} {operator} {right.type()}")
    
    def eval_prefix_expression(self, node, env, stack_trace):
        debug_log("eval_prefix_expression", f"{node.operator} {node.right}")
        
        right = self.eval_node(node.right, env, stack_trace)
        if is_error(right): 
            return right
        
        operator = node.operator
        
        if operator == "!":
            # !true = false, !false = true, !null = true, !anything_else = false
            if right == TRUE:
                return FALSE
            elif right == FALSE or right == NULL:
                return TRUE
            else:
                return FALSE
        elif operator == "-":
            if isinstance(right, Integer):
                return Integer(-right.value)
            elif isinstance(right, Float):
                return Float(-right.value)
            return EvaluationError(f"Unknown operator: -{right.type()}")
        
        return EvaluationError(f"Unknown operator: {operator}{right.type()}")
    
    def eval_if_expression(self, node, env, stack_trace):
        debug_log("eval_if_expression", "Evaluating condition")
        
        condition = self.eval_node(node.condition, env, stack_trace)
        if is_error(condition): 
            return condition
        
        if is_truthy(condition):
            debug_log("  Condition true, evaluating consequence")
            return self.eval_node(node.consequence, env, stack_trace)
        elif node.alternative:
            debug_log("  Condition false, evaluating alternative")
            return self.eval_node(node.alternative, env, stack_trace)
        
        debug_log("  Condition false, no alternative")
        return NULL
    
    def eval_expressions(self, exps, env):
        results = []
        for e in exps:
            val = self.eval_node(e, env)
            if is_error(val): 
                return val
            results.append(val)
        return results

    def eval_ternary_expression(self, node, env, stack_trace):
        """Evaluate ternary expression: condition ? true_value : false_value"""
        from .utils import is_truthy
        
        condition = self.eval_node(node.condition, env, stack_trace)
        if is_error(condition):
            return condition
        
        if is_truthy(condition):
            return self.eval_node(node.true_value, env, stack_trace)
        else:
            return self.eval_node(node.false_value, env, stack_trace)

    def eval_nullish_expression(self, node, env, stack_trace):
        """Evaluate nullish coalescing: value ?? default
        Returns default if value is null/undefined, otherwise returns value"""
        left = self.eval_node(node.left, env, stack_trace)
        
        # If left is an error, return the error
        if is_error(left):
            return left
        
        # Check if left is null or undefined (NULL)
        if left is NULL or left is None or (hasattr(left, 'type') and left.type() == 'NULL'):
            return self.eval_node(node.right, env, stack_trace)
        
        return left

    def eval_await_expression(self, node, env, stack_trace):
        """Evaluate await expression: await <expression>
        
        Await can handle:
        1. Promise objects - waits for resolution
        2. Coroutine objects - resumes until complete
        3. Async action calls - wraps in Promise
        4. Regular values - returns immediately
        """
        from ..object import Promise, Coroutine, EvaluationError
        
        # Evaluate the expression to await
        awaitable = self.eval_node(node.expression, env, stack_trace)
        
        # Check for errors
        if is_error(awaitable):
            return awaitable
        
        # Handle different awaitable types
        if hasattr(awaitable, 'type'):
            obj_type = awaitable.type()
            
            # Await a Promise
            if obj_type == "PROMISE":
                # Since promises execute immediately in executor, they should be resolved
                if awaitable.is_resolved():
                    try:
                        result = awaitable.get_value()
                        return result if result is not None else NULL
                    except Exception as e:
                        # Propagate error with stack trace context
                        error_msg = f"Promise rejected: {e}"
                        if hasattr(awaitable, 'stack_trace') and awaitable.stack_trace:
                            error_msg += f"\n  Promise created at: {awaitable.stack_trace}"
                        return EvaluationError(error_msg)
                else:
                    # Promise is still pending - this shouldn't happen with current implementation
                    # but we can spin-wait briefly
                    import time
                    max_wait = 1.0  # 1 second timeout
                    waited = 0.0
                    while not awaitable.is_resolved() and waited < max_wait:
                        time.sleep(0.001)  # 1ms
                        waited += 0.001
                    
                    if awaitable.is_resolved():
                        try:
                            result = awaitable.get_value()
                            return result if result is not None else NULL
                        except Exception as e:
                            return EvaluationError(f"Promise rejected: {e}")
                    else:
                        return EvaluationError("Await timeout: promise did not resolve")
            
            # Await a Coroutine
            elif obj_type == "COROUTINE":
                # Resume coroutine until complete
                while not awaitable.is_complete:
                    is_done, value = awaitable.resume()
                    if is_done:
                        # Check if there was an error
                        if awaitable.error:
                            return EvaluationError(f"Coroutine error: {awaitable.error}")
                        return value if value is not None else NULL
                    
                    # If coroutine yielded a value, it might be another awaitable
                    if hasattr(value, 'type') and value.type() == "PROMISE":
                        # Wait for the promise
                        if value.is_resolved():
                            try:
                                resume_value = value.get_value()
                                # Send the value back to the coroutine
                                is_done, result = awaitable.resume(resume_value)
                                if is_done:
                                    return result if result is not None else NULL
                            except Exception as e:
                                return EvaluationError(f"Promise error in coroutine: {e}")
                
                return awaitable.result if awaitable.result is not None else NULL
            
            # Regular value - return immediately
            else:
                return awaitable
        
        # No type method - return as-is
        return awaitable

    def eval_file_import_expression(self, node, env, stack_trace):
        """Evaluate file import expression: let code << "filename.ext"
        
        Reads the file contents and returns as a String object.
        Supports any file extension - returns raw file content.
        """
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
        
        # 4. Check if file exists
        if not os.path.exists(filepath):
            return new_error(f"Cannot import file '{filepath}': File not found", stack_trace)
        
        # 5. Read file contents
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return String(content)
        except UnicodeDecodeError:
            # Try binary mode for non-text files
            try:
                with open(filepath, 'rb') as f:
                    content = f.read()
                # Return as string representation of bytes
                return String(str(content))
            except Exception as e:
                return new_error(f"Error reading file '{filepath}': {e}", stack_trace)
        except Exception as e:
            return new_error(f"Error importing file '{filepath}': {e}", stack_trace)
    def eval_match_expression(self, node, env, stack_trace):
        """Evaluate match expression with pattern matching
        
        match value {
            Point(x, y) => x + y,
            User(name, _) => name,
            42 => "the answer",
            _ => "default"
        }
        """
        from .. import zexus_ast
        from ..object import Map, String, Integer, Boolean as BooleanObj, Environment
        
        debug_log("eval_match_expression", "Evaluating match expression")
        
        # Evaluate the value to match against
        match_value = self.eval_node(node.value, env, stack_trace)
        if is_error(match_value):
            return match_value
        
        debug_log("  Match value", str(match_value))
        
        # Try each case in order
        for case in node.cases:
            pattern = case.pattern
            result_expr = case.result
            
            # Try to match the pattern
            match_result = self._match_pattern(pattern, match_value, env)
            
            if match_result is not None:
                # Pattern matched! Create new environment with bindings
                new_env = Environment(outer=env)
                
                # Add all bindings to the new environment
                for var_name, var_value in match_result.items():
                    new_env.set(var_name, var_value)
                
                # Evaluate and return the result expression
                result = self.eval_node(result_expr, new_env, stack_trace)
                debug_log("  ✅ Pattern matched", f"Result: {result}")
                return result
        
        # No pattern matched
        return EvaluationError("Match expression: no pattern matched")
    
    def _match_pattern(self, pattern, value, env):
        """Try to match a pattern against a value
        
        Returns:
            dict: Bindings if matched (variable name -> value)
            None: If pattern doesn't match
        """
        from .. import zexus_ast
        from ..object import Map, String, Integer, Float, Boolean as BooleanObj
        
        # Wildcard pattern: always matches, no bindings
        if isinstance(pattern, zexus_ast.WildcardPattern):
            debug_log("  🎯 Wildcard pattern matched", "_")
            return {}
        
        # Variable pattern: always matches, bind variable
        if isinstance(pattern, zexus_ast.VariablePattern):
            debug_log("  🎯 Variable pattern matched", f"{pattern.name} = {value}")
            return {pattern.name: value}
        
        # Literal pattern: check equality
        if isinstance(pattern, zexus_ast.LiteralPattern):
            pattern_value = self.eval_node(pattern.value, env, [])
            
            # Compare values
            matches = False
            if isinstance(value, Integer) and isinstance(pattern_value, Integer):
                matches = value.value == pattern_value.value
            elif isinstance(value, Float) and isinstance(pattern_value, Float):
                matches = value.value == pattern_value.value
            elif isinstance(value, String) and isinstance(pattern_value, String):
                matches = value.value == pattern_value.value
            elif isinstance(value, BooleanObj) and isinstance(pattern_value, BooleanObj):
                matches = value.value == pattern_value.value
            
            if matches:
                debug_log("  🎯 Literal pattern matched", str(pattern_value))
                return {}
            else:
                debug_log("  ❌ Literal pattern didn't match", f"{pattern_value} != {value}")
                return None
        
        # Constructor pattern: match dataclass instances
        if isinstance(pattern, zexus_ast.ConstructorPattern):
            # Check if value is a Map (dataclass instance)
            if not isinstance(value, Map):
                debug_log("  ❌ Constructor pattern: value is not a dataclass", type(value).__name__)
                return None
            
            # Check if value has __type__ field matching constructor name
            type_key = String("__type__")
            if type_key not in value.pairs:
                debug_log("  ❌ Constructor pattern: no __type__ field", "")
                return None
            
            type_value = value.pairs[type_key]
            if not isinstance(type_value, String):
                debug_log("  ❌ Constructor pattern: __type__ is not a string", type(type_value).__name__)
                return None
            
            # Extract actual type name (handle specialized generics like "Point<number>")
            actual_type = type_value.value
            if '<' in actual_type:
                # Strip generic parameters for matching
                actual_type = actual_type.split('<')[0]
            
            if actual_type != pattern.constructor_name:
                debug_log("  ❌ Constructor pattern: type mismatch", f"{actual_type} != {pattern.constructor_name}")
                return None
            
            debug_log("  ✅ Constructor type matched", pattern.constructor_name)
            
            # Extract field values and match against bindings
            bindings = {}
            
            # Get all non-internal, non-method fields from the dataclass
            # Maintain original field order from the dataclass definition
            fields = []
            field_dict = {}
            
            for key, val in value.pairs.items():
                if isinstance(key, String):
                    field_name = key.value
                    # Skip internal fields (__type__, __immutable__, etc.)
                    if field_name.startswith("__"):
                        continue
                    # Skip auto-generated methods (toString, toJSON, clone, equals, hash, verify, fromJSON)
                    if field_name in {"toString", "toJSON", "clone", "equals", "hash", "verify", "fromJSON"}:
                        continue
                    field_dict[field_name] = val
            
            # Try to get field order from __field_order__ metadata if available
            # Otherwise, use the order they appear in the Map (which should be insertion order in Python 3.7+)
            field_order_key = String("__field_order__")
            if field_order_key in value.pairs:
                # Use explicit field order if available
                field_order = value.pairs[field_order_key]
                if isinstance(field_order, List):
                    for field_name_obj in field_order.elements:
                        if isinstance(field_name_obj, String):
                            field_name = field_name_obj.value
                            if field_name in field_dict:
                                fields.append((field_name, field_dict[field_name]))
            else:
                # Use insertion order (dict maintains order in Python 3.7+)
                fields = [(k, v) for k, v in field_dict.items()]
            
            # Match each binding pattern against corresponding field value
            if len(pattern.bindings) != len(fields):
                debug_log("  ❌ Constructor pattern: binding count mismatch", f"{len(pattern.bindings)} != {len(fields)}")
                return None
            
            for i, (field_name, field_value) in enumerate(fields):
                binding_pattern = pattern.bindings[i]
                
                # Recursively match the binding pattern
                binding_result = self._match_pattern(binding_pattern, field_value, env)
                
                if binding_result is None:
                    debug_log("  ❌ Constructor pattern: binding didn't match", f"field {field_name}")
                    return None
                
                # Merge bindings
                bindings.update(binding_result)
            
            debug_log("  🎯 Constructor pattern fully matched", f"{pattern.constructor_name} with {len(bindings)} bindings")
            return bindings
        
        # Unknown pattern type
        debug_log("  ❌ Unknown pattern type", type(pattern).__name__)
        return None
    def eval_async_expression(self, node, env, stack_trace):
        """Evaluate async expression: async <expression>
        
        Executes the expression in a background thread.
        Example: async producer()
        """
        import threading
        import sys
        
        # For call expressions, we need to defer evaluation to the thread
        # Otherwise evaluating here will execute the action in the main thread
        if type(node.expression).__name__ == 'CallExpression':
            def run_in_thread():
                try:
                    result = self.eval_node(node.expression, env, stack_trace)
                    
                    # If it's a Coroutine (from async action), execute it
                    if hasattr(result, '__class__') and result.__class__.__name__ == 'Coroutine':
                        try:
                            # Prime the generator
                            next(result.generator)
                            # Execute until completion
                            while True:
                                next(result.generator)
                        except StopIteration:
                            pass  # Coroutine completed
                    
                except StopIteration:
                    pass  # Normal coroutine completion
                except Exception as e:
                    import sys
                    print(f"[ASYNC ERROR] {type(e).__name__}: {str(e)}", file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
            
            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()
            return NULL
        
        # For other expressions, evaluate first then check if it's a Coroutine
        result = self.eval_node(node.expression, env, stack_trace)
        
        if is_error(result):
            return result
        
        # print(f"[ASYNC EXPR] Expression evaluated to: {type(result).__name__}", file=sys.stderr)
        
        # If it's a Coroutine (from calling an async action), execute it in a thread
        if hasattr(result, '__class__') and result.__class__.__name__ == 'Coroutine':
            def run_coroutine():
                try:
                    # Prime the generator
                    next(result.generator)
                    # Execute until completion
                    while True:
                        next(result.generator)
                except StopIteration:
                    pass  # Coroutine completed normally
                except Exception as e:
                    import sys
                    print(f"[ASYNC ERROR] {type(e).__name__}: {str(e)}", file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)
            
            thread = threading.Thread(target=run_coroutine, daemon=True)
            thread.start()
            return NULL
        
        # For any other result (including NULL from regular actions),
        # we can't execute it asynchronously since it already executed.
        # Just return NULL to indicate "async operation initiated"
        # print(f"[ASYNC EXPR] Result is not a coroutine, returning NULL", file=sys.stderr)
        return NULL
