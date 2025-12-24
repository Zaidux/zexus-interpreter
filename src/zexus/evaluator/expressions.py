# src/zexus/evaluator/expressions.py
from .. import zexus_ast
from ..zexus_ast import (
    IntegerLiteral, FloatLiteral, StringLiteral, ListLiteral, MapLiteral, 
    Identifier, PrefixExpression, InfixExpression, IfExpression, 
    Boolean as AST_Boolean, EmbeddedLiteral, ActionLiteral, LambdaExpression
)
from ..object import (
    Integer, Float, String, List, Map, Boolean as BooleanObj,
    Null, Action, LambdaFunction, EmbeddedCode, EvaluationError, Builtin
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
            print(f"[DEBUG] Identifier not found: {node.value}; env_keys={env_keys}\nStack snippet:\n{stack_snip}")
        except Exception:
            print(f"[DEBUG] Identifier not found: {node.value}")
        return EvaluationError(f"Identifier '{node.value}' not found")
    
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
                return EvaluationError("Division by zero")
            return Integer(left_val // right_val)
        elif operator == "%":
            if right_val == 0: 
                return EvaluationError("Modulo by zero")
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
        
        # Array Concatenation
        elif operator == "+" and isinstance(left, List) and isinstance(right, List):
            # Concatenate two arrays: [1, 2] + [3, 4] = [1, 2, 3, 4]
            new_elements = left.elements[:] + right.elements[:]
            return List(new_elements)
        
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
                # Propagate stack trace context from promise
                if hasattr(awaitable, 'stack_trace') and awaitable.stack_trace:
                    # Merge promise's stack trace with current context
                    stack_trace = stack_trace + [f"  at await <promise>"]
                
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
