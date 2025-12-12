# src/zexus/evaluator/expressions.py
from .. import zexus_ast
from ..zexus_ast import (
    IntegerLiteral, FloatLiteral, StringLiteral, ListLiteral, MapLiteral, 
    Identifier, PrefixExpression, InfixExpression, IfExpression, 
    Boolean as AST_Boolean, EmbeddedLiteral, ActionLiteral, LambdaExpression
)
from ..object import (
    Integer, Float, String, List, Map, Boolean as BooleanObj,
    Null, Action, LambdaFunction, EmbeddedCode, EvaluationError
)
from .utils import is_error, debug_log, NULL, TRUE, FALSE, is_truthy

class ExpressionEvaluatorMixin:
    """Handles evaluation of expressions: Literals, Math, Logic, Identifiers."""
    
    def eval_identifier(self, node, env):
        debug_log("eval_identifier", f"Looking up: {node.value}")
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
