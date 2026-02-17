# src/zexus/evaluator/expressions.py
import os

from ..zexus_ast import (
    IntegerLiteral, FloatLiteral, StringLiteral, ListLiteral, MapLiteral,
    Identifier, PrefixExpression, InfixExpression, IfExpression,
    Boolean as AST_Boolean, EmbeddedLiteral, ActionLiteral, LambdaExpression,
    PropertyAccessExpression,
)
from ..object import (
    Integer, Float, String, List, Map,
    EvaluationError, Builtin, DateTime
)
from ..config import config as zexus_config
from .utils import is_error, debug_log, NULL, TRUE, FALSE, is_truthy, _python_to_zexus

class ExpressionEvaluatorMixin:
    """Handles evaluation of expressions: Literals, Math, Logic, Identifiers."""
    
    def eval_property_access_expression(self, node, env, stack_trace=None):
        obj = self.eval_node(node.object, env, stack_trace)
        if is_error(obj): return obj

        from ..object import String, Map, List, EvaluationError

        if node.computed:
            # Index notation: obj[expr]
            idx = self.eval_node(node.property, env, stack_trace)
            if is_error(idx): return idx
            
            if isinstance(obj, List):
                 return obj.get(idx)
            elif isinstance(obj, Map):
                 key = idx
                 if isinstance(key, str): key = String(key)
                 return obj.get(key) or NULL
            elif isinstance(obj, String):
                 return obj.get(idx)
            else:
                 # Fallback for Python objects
                 try:
                     raw_idx = idx.value if hasattr(idx, 'value') else idx
                     return obj[raw_idx]
                 except (IndexError, KeyError, TypeError):
                     return NULL
        else:
            # Dot notation: obj.prop
            if not hasattr(node.property, 'value'):
                 return EvaluationError(f"Invalid property identifier: {node.property}")
            
            prop_name = str(node.property.value)
            
            # Check security restrictions (redact, read-only, etc.)
            try:
                from ..security import get_security_context
                ctx = get_security_context()
                target = f"{getattr(node.object, 'value', str(node.object))}.{prop_name}"
                restriction = ctx.get_restriction(target)
                if restriction:
                    rule = restriction.get('restriction')
                    if rule == 'redact':
                        return String('***REDACTED***')
            except Exception:
                pass
            
            if isinstance(obj, Map):
                 return obj.get(String(prop_name)) or NULL
            elif isinstance(obj, dict):
                 return obj.get(prop_name, NULL)
            else:
                 # Try getattr (for data classes, etc.)
                 try:
                     res = getattr(obj, prop_name, NULL)
                     return res
                 except Exception:
                     return NULL

    def eval_identifier(self, node, env):
        name = node.value
        if zexus_config.fast_debug_enabled:
            debug_log("eval_identifier", f"Looking up: {name}")
        
        # Special case: 'this' keyword should be treated like ThisExpression
        if name == "this":
            # Look for contract instance first
            contract_instance = env.get("__contract_instance__")
            if contract_instance is not None:
                return contract_instance
            
            # Then look for data method instance
            data_instance = env.get("this")
            if data_instance is not None:
                return data_instance
        
        # First, check environment for user-defined variables (including DATA dataclasses)
        val = env.get(name)
        if val:
            if zexus_config.fast_debug_enabled:
                 debug_log("  Found in environment", f"{name} = {val}")
            return val
        
        # Check builtins (self.builtins should be defined in FunctionEvaluatorMixin)
        if hasattr(self, 'builtins'):
            builtin = self.builtins.get(name)
            if builtin:
                if zexus_config.fast_debug_enabled:
                    debug_log("  Found builtin", f"{name} = {builtin}")
                return builtin
        
        # Special handling for TX - ONLY if not already defined by user
        # This provides blockchain transaction context when TX is not a user dataclass
        if name == "TX":
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
        
        # SECURITY FIX #6: Integer overflow protection
        # Python integers have arbitrary precision, but we enforce safe ranges
        # to prevent resource exhaustion and match real-world integer behavior
        MAX_SAFE_BIT_LENGTH = 4096  # Allow values up to ~10^1233 before flagging overflow

        def check_overflow(result, operation):
            """Check if integer operation resulted in overflow"""
            if isinstance(result, int):
                bit_length = abs(result).bit_length()
                if bit_length > MAX_SAFE_BIT_LENGTH:
                    # Use a quick log10 approximation without importing decimal-heavy helpers
                    approx_digits = int(bit_length * 0.30103) + 1 if bit_length else 1
                    return EvaluationError(
                        f"Integer overflow in {operation}",
                        suggestion=(
                            f"Result requires {bit_length} bits (~{approx_digits} digits), exceeding the safe limit of "
                            f"{MAX_SAFE_BIT_LENGTH} bits. Break the calculation into smaller parts or enable big-integer support."
                        )
                    )
            return Integer(result)
        
        if operator == "+":
            result = left_val + right_val
            return check_overflow(result, "addition")
        elif operator == "-":
            result = left_val - right_val
            return check_overflow(result, "subtraction")
        elif operator == "*":
            result = left_val * right_val
            return check_overflow(result, "multiplication")
        elif operator == "/":
            if right_val == 0: 
                return EvaluationError(
                    "Division by zero",
                    suggestion="Check your divisor value. Consider adding a condition: if (divisor != 0) { ... }"
                )
            result = left_val // right_val
            return check_overflow(result, "division")
        elif operator == "%":
            if right_val == 0: 
                return EvaluationError(
                    "Modulo by zero",
                    suggestion="Check your divisor value. Modulo operation requires a non-zero divisor."
                )
            return Integer(left_val % right_val)
        elif operator == "**":
            if right_val < 0:
                # Negative exponent returns float
                return Float(left_val ** right_val)
            result = left_val ** right_val
            return check_overflow(result, "exponentiation")
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
        elif operator == "**":
            try:
                return Float(left_val ** right_val)
            except (OverflowError, ValueError) as e:
                return EvaluationError(f"Exponentiation error: {e}")
        
        return EvaluationError(f"Unknown float operator: {operator}")
    
    def eval_string_infix(self, operator, left, right):
        if operator == "+":
            # SECURITY ENFORCEMENT: Check sanitization before concatenation
            from ..security_enforcement import check_string_concatenation
            check_string_concatenation(left, right)
            
            # Propagate sanitization status to result
            result = String(left.value + right.value)
            
            # Result is trusted only if both inputs are trusted
            result.is_trusted = left.is_trusted and right.is_trusted
            
            # Propagate sanitization context intelligently:
            # - If both have same sanitization, inherit it
            # - If one is trusted (literal) and other is sanitized, inherit the sanitization
            # - Otherwise, no sanitization (both must be sanitized or one trusted + one sanitized)
            if left.sanitized_for == right.sanitized_for and left.sanitized_for is not None:
                result.sanitized_for = left.sanitized_for
            elif left.is_trusted and right.sanitized_for is not None:
                result.sanitized_for = right.sanitized_for
            elif right.is_trusted and left.sanitized_for is not None:
                result.sanitized_for = left.sanitized_for
            
            return result
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
        
        # SECURITY FIX #8: Type Safety - No Implicit Coercion
        # Addition operator: String+String OR Number+Number only
        elif operator == "+":
            # String concatenation requires both operands to be strings
            if isinstance(left, String) or isinstance(right, String):
                # If either is a string, both must be strings
                if not (isinstance(left, String) and isinstance(right, String)):
                    left_type = "STRING" if isinstance(left, String) else type(left).__name__.upper()
                    right_type = "STRING" if isinstance(right, String) else type(right).__name__.upper()
                    
                    return EvaluationError(
                        f"Type mismatch: cannot add {left_type} and {right_type}\n"
                        f"Use explicit conversion: string(value) to convert to string before concatenation\n"
                        f"Example: string({left.value if hasattr(left, 'value') else left}) + string({right.value if hasattr(right, 'value') else right})"
                    )
            
            # Numeric addition: Integer + Integer OR Float + Float OR Integer + Float
            elif isinstance(left, (Integer, Float)) and isinstance(right, (Integer, Float)):
                # Mixed Integer/Float operations return Float
                if isinstance(left, Float) or isinstance(right, Float):
                    result = float(left.value) + float(right.value)
                    return Float(result)
                # Both integers handled by eval_integer_infix
                else:
                    return EvaluationError("Internal error: Integer + Integer should be handled by eval_integer_infix")
            
            # Invalid type combination
            else:
                left_type = type(left).__name__.replace("Obj", "").upper()
                right_type = type(right).__name__.replace("Obj", "").upper()
                return EvaluationError(
                    f"Type error: cannot add {left_type} and {right_type}\n"
                    f"Addition requires matching types: STRING + STRING or NUMBER + NUMBER"
                )
        
        # SECURITY FIX #8: Strict Type Checking for Arithmetic
        # All arithmetic operations require numeric types (Integer or Float)
        elif operator in ("*", "-", "/", "%", "**"):
            # Get type names for error messages
            left_type = type(left).__name__.replace("Obj", "").upper()
            right_type = type(right).__name__.replace("Obj", "").upper()
            
            # Only allow arithmetic between numbers (Integer or Float)
            if not isinstance(left, (Integer, Float)):
                return EvaluationError(
                    f"Type error: {operator} requires numeric operands, got {left_type}\n"
                    f"Use explicit conversion: int(value) or float(value)"
                )
            
            if not isinstance(right, (Integer, Float)):
                return EvaluationError(
                    f"Type error: {operator} requires numeric operands, got {right_type}\n"
                    f"Use explicit conversion: int(value) or float(value)"
                )
            
            # Both are numbers - perform operation
            # Mixed Integer/Float operations return Float
            if isinstance(left, Float) or isinstance(right, Float):
                l_val = float(left.value)
                r_val = float(right.value)
                
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
                    elif operator == "**":
                        result = l_val ** r_val
                    
                    # Return Integer if result is whole number, Float otherwise
                    if result == int(result) and operator not in ("/", "**"):  # Division/power always returns float
                        return Integer(int(result))
                    return Float(result)
                except Exception as e:
                    return EvaluationError(f"Arithmetic error: {str(e)}")
            else:
                # Both integers - integer arithmetic (already handled by eval_integer_infix)
                return EvaluationError(f"Internal error: Integer {operator} Integer should be handled by eval_integer_infix")
        
        # Comparison with mixed numeric types (Integer/Float comparison allowed)
        elif operator in ("<", ">", "<=", ">="):
            # Safe null handling: Any comparison with NULL is False (except != handled above)
            if left == NULL or right == NULL:
                return FALSE

            if isinstance(left, (Integer, Float)) and isinstance(right, (Integer, Float)):
                l_val = float(left.value)
                r_val = float(right.value)
                if operator == "<": return TRUE if l_val < r_val else FALSE
                elif operator == ">": return TRUE if l_val > r_val else FALSE
                elif operator == "<=": return TRUE if l_val <= r_val else FALSE
                elif operator == ">=": return TRUE if l_val >= r_val else FALSE
            
            # SECURITY FIX #8: No implicit coercion for comparisons
            else:
                left_type = type(left).__name__.replace("Obj", "").upper()
                right_type = type(right).__name__.replace("Obj", "").upper()
                return EvaluationError(
                    f"Type error: cannot compare {left_type} {operator} {right_type}\n"
                    f"Use explicit conversion if needed: int(value) or float(value)"
                )

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

    def _current_directory_from_env(self, env):
        if not env or not hasattr(env, 'get'):
            return None
        try:
            file_obj = env.get("__file__")
        except Exception:
            file_obj = None
        if not file_obj:
            return None
        path = file_obj.value if hasattr(file_obj, 'value') else str(file_obj)
        if not path:
            return None
        return os.path.dirname(path)

    def _evaluate_expression_to_string(self, expr, env, stack_trace, literal_identifiers=False):
        if literal_identifiers and isinstance(expr, Identifier):
            return expr.value, None
        if isinstance(expr, StringLiteral):
            return expr.value, None
        if isinstance(expr, IntegerLiteral):
            return str(expr.value), None
        if isinstance(expr, FloatLiteral):
            return str(expr.value), None
        if isinstance(expr, AST_Boolean):
            return "true" if expr.value else "false", None

        value = self.eval_node(expr, env, stack_trace)
        if is_error(value):
            return None, value

        if hasattr(value, 'value'):
            return str(value.value), None
        return str(value), None

    def _flatten_property_chain(self, expr, env, stack_trace):
        segments = []
        current = expr
        while isinstance(current, PropertyAccessExpression):
            literal_mode = not getattr(current, 'computed', False)
            segment, error = self._evaluate_expression_to_string(
                current.property,
                env,
                stack_trace,
                literal_identifiers=literal_mode,
            )
            if error:
                return None, error
            segments.insert(0, segment)
            current = current.object

        segment, error = self._evaluate_expression_to_string(
            current,
            env,
            stack_trace,
            literal_identifiers=True,
        )
        if error:
            return None, error
        segments.insert(0, segment)
        return segments, None

    def eval_find_expression(self, node, env, stack_trace):
        debug_log("eval_find_expression", "find keyword")

        pattern, error = self._evaluate_expression_to_string(
            node.target,
            env,
            stack_trace,
            literal_identifiers=True,
        )
        if error:
            return error

        if pattern is None or not str(pattern).strip():
            return EvaluationError("find requires a target path")

        pattern = str(pattern).strip()

        scope = None
        if getattr(node, 'scope', None) is not None:
            scope, error = self._evaluate_expression_to_string(
                node.scope,
                env,
                stack_trace,
                literal_identifiers=True,
            )
            if error:
                return error
            scope = str(scope).strip() if scope else None

        current_dir = self._current_directory_from_env(env)

        from .. import module_manager

        matches = module_manager.find_files(
            pattern,
            current_dir=current_dir,
            scope=scope,
        )

        if not matches:
            suggestion = "Verify the file exists or adjust the scope provided to find."
            return EvaluationError(
                f"find could not locate '{pattern}'",
                suggestion=suggestion,
            )

        if len(matches) > 1:
            preview = ", ".join(matches[:3])
            suggestion = "Provide a more specific path or scope to disambiguate the match."
            return EvaluationError(
                f"find found multiple matches for '{pattern}': {preview}",
                suggestion=suggestion,
            )

        return String(matches[0], is_trusted=True)

    def eval_load_expression(self, node, env, stack_trace):
        debug_log("eval_load_expression", "load keyword")

        from ..runtime import get_load_manager

        provider_hint = getattr(node, 'provider_hint', None)
        provider = provider_hint.lower() if isinstance(provider_hint, str) else None

        segments = None
        if isinstance(node.target, PropertyAccessExpression):
            segments, error = self._flatten_property_chain(node.target, env, stack_trace)
            if error:
                return error

        manager = get_load_manager()

        key = None
        if segments:
            candidate = (segments[0] or "").lower()
            if provider is None and manager.is_provider_registered(candidate):
                provider = candidate
                remainder = [seg for seg in segments[1:] if seg is not None]
                key = ".".join(remainder)
            else:
                key = ".".join(seg for seg in segments if seg is not None)

        if key is None:
            key, error = self._evaluate_expression_to_string(
                node.target,
                env,
                stack_trace,
                literal_identifiers=True,
            )
            if error:
                return error

        if isinstance(key, str):
            key = key.strip()

        if provider:
            provider = provider.strip().lower()

        if provider and not key:
            return EvaluationError(
                "load requires a key when a provider is specified",
                suggestion="Append the key after the provider, for example load env.API_KEY.",
            )

        source = None
        if getattr(node, 'source', None) is not None:
            source, error = self._evaluate_expression_to_string(
                node.source,
                env,
                stack_trace,
            )
            if error:
                return error
            if isinstance(source, str):
                source = source.strip()

        current_dir = self._current_directory_from_env(env)

        if provider and not manager.is_provider_registered(provider):
            return EvaluationError(
                f"Unknown load provider '{provider}'",
                suggestion="Register a provider before using it or choose a supported provider.",
            )

        try:
            value = manager.load(
                key,
                provider=provider,
                source=source,
                current_dir=current_dir,
            )
        except FileNotFoundError as exc:
            suggestion = "Check the referenced file path or provide an absolute path."
            return EvaluationError(str(exc), suggestion=suggestion)
        except KeyError:
            missing = key or "<empty>"
            message = f"load could not resolve '{missing}'"
            if provider:
                message += f" via provider '{provider}'"
            if source:
                message += f" from '{source}'"
            suggestion = "Ensure the value exists or configure a fallback source."
            return EvaluationError(message, suggestion=suggestion)
        except Exception as exc:
            return EvaluationError(f"load failed: {exc}")

        return _python_to_zexus(value, mark_untrusted=True)

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
        
        Schedules the expression on the shared Zexus event loop.
        For call expressions, evaluation is deferred entirely to the loop.
        For coroutine results, driving is done inside a loop task.
        """
        from ..event_loop import spawn as _spawn
        import asyncio
        import sys

        # For call expressions, we need to defer evaluation to the event loop
        if type(node.expression).__name__ == 'CallExpression':
            async def _run_call():
                try:
                    result = self.eval_node(node.expression, env, stack_trace)

                    # If it's a Coroutine (from async action), execute it
                    if hasattr(result, '__class__') and result.__class__.__name__ == 'Coroutine':
                        try:
                            next(result.generator)
                            while True:
                                next(result.generator)
                        except StopIteration:
                            pass
                except StopIteration:
                    pass
                except Exception as e:
                    print(f"[ASYNC ERROR] {type(e).__name__}: {str(e)}", file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)

            _spawn(_run_call())
            return NULL

        # For other expressions, evaluate first then check if it's a Coroutine
        previous_allow = getattr(self, "_allow_coroutine_result", False)
        self._allow_coroutine_result = True
        try:
            result = self.eval_node(node.expression, env, stack_trace)
        finally:
            self._allow_coroutine_result = previous_allow

        if is_error(result):
            return result

        # If it's a Coroutine, drive it on the shared event loop
        if hasattr(result, '__class__') and result.__class__.__name__ == 'Coroutine':
            async def _drive_coroutine():
                try:
                    next(result.generator)
                    while True:
                        next(result.generator)
                except StopIteration:
                    pass
                except Exception as e:
                    print(f"[ASYNC ERROR] {type(e).__name__}: {str(e)}", file=sys.stderr, flush=True)
                    import traceback
                    traceback.print_exc(file=sys.stderr)

            _spawn(_drive_coroutine())
            return NULL

        return NULL
