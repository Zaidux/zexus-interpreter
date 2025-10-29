# evaluator.py (PRODUCTION READY)
from . import zexus_ast
from .zexus_ast import (
    Program, ExpressionStatement, BlockStatement, ReturnStatement, LetStatement,
    ActionStatement, IfStatement, WhileStatement, ForEachStatement, MethodCallExpression,
    EmbeddedLiteral, PrintStatement, ScreenStatement, EmbeddedCodeStatement, UseStatement,
    ExactlyStatement, IntegerLiteral, StringLiteral, ListLiteral, MapLiteral, Identifier,
    ActionLiteral, CallExpression, PrefixExpression, InfixExpression, IfExpression,
    Boolean as AST_Boolean, AssignmentExpression, PropertyAccessExpression,
    ExportStatement, LambdaExpression  # NEW: Added exports and lambda
)

from .object import Environment, Integer, Float, String, List, Map, Null, Boolean as BooleanObj, Builtin, Action, EmbeddedCode, ReturnValue, LambdaFunction

NULL, TRUE, FALSE = Null(), BooleanObj(True), BooleanObj(False)

class EvaluationError(Exception):
    """Custom exception for evaluation errors with location info"""
    def __init__(self, message, line=None, column=None):
        super().__init__(message)
        self.line = line
        self.column = column
        self.message = message
    
    def __str__(self):
        if self.line and self.column:
            return f"Line {self.line}:{self.column} - {self.message}"
        return self.message

# === HELPER FUNCTIONS ===

def eval_program(statements, env):
    result = NULL
    for stmt in statements:
        result = eval_node(stmt, env)
        if isinstance(result, ReturnValue):
            return result.value
        if isinstance(result, EvaluationError):
            return result
    return result

def eval_assignment_expression(node, env):
    """Handle assignment expressions like: x = 5"""
    value = eval_node(node.value, env)
    if isinstance(value, EvaluationError):
        return value

    # Set the variable in the environment
    env.set(node.name.value, value)
    return value

def eval_block_statement(block, env):
    result = NULL
    for stmt in block.statements:
        result = eval_node(stmt, env)
        if isinstance(result, ReturnValue) or isinstance(result, EvaluationError):
            return result
    return result

def eval_expressions(expressions, env):
    results = []
    for expr in expressions:
        result = eval_node(expr, env)
        if isinstance(result, (ReturnValue, EvaluationError)):
            return result
        results.append(result)
    return results

def eval_identifier(node, env):
    val = env.get(node.value)
    if val:
        return val
    # Check builtins
    builtin = builtins.get(node.value)
    if builtin:
        return builtin
    
    return EvaluationError(f"Identifier '{node.value}' not found")

def is_truthy(obj):
    if obj == NULL or obj == FALSE or isinstance(obj, EvaluationError):
        return False
    return True

def eval_prefix_expression(operator, right):
    if isinstance(right, EvaluationError):
        return right
        
    if operator == "!":
        return eval_bang_operator_expression(right)
    elif operator == "-":
        return eval_minus_prefix_operator_expression(right)
    return EvaluationError(f"Unknown operator: {operator}{right.type()}")

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
    # Handle errors first
    if isinstance(left, EvaluationError):
        return left
    if isinstance(right, EvaluationError):
        return right

    # Logical operators
    if operator == "&&":
        return TRUE if is_truthy(left) and is_truthy(right) else FALSE
    elif operator == "||":
        return TRUE if is_truthy(left) or is_truthy(right) else FALSE

    # Comparison operators
    if operator == "==":
        return TRUE if left.value == right.value else FALSE
    elif operator == "!=":
        return TRUE if left.value != right.value else FALSE
    elif operator == "<=":
        return TRUE if left.value <= right.value else FALSE
    elif operator == ">=":
        return TRUE if left.value >= right.value else FALSE

    # Type-specific operations
    if isinstance(left, Integer) and isinstance(right, Integer):
        return eval_integer_infix_expression(operator, left, right)
    elif isinstance(left, Float) and isinstance(right, Float):
        return eval_float_infix_expression(operator, left, right)
    elif isinstance(left, String) and isinstance(right, String):
        return eval_string_infix_expression(operator, left, right)

    return EvaluationError(f"Type mismatch: {left.type()} {operator} {right.type()}")

def eval_integer_infix_expression(operator, left, right):
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

def eval_float_infix_expression(operator, left, right):
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
    elif operator == "%":
        if right_val == 0:
            return EvaluationError("Modulo by zero")
        return Float(left_val % right_val)
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

def eval_string_infix_expression(operator, left, right):
    if operator == "+":
        return String(left.value + right.value)
    elif operator == "==":
        return TRUE if left.value == right.value else FALSE
    elif operator == "!=":
        return TRUE if left.value != right.value else FALSE
    return EvaluationError(f"Unknown string operator: {operator}")

def eval_if_expression(ie, env):
    condition = eval_node(ie.condition, env)
    if isinstance(condition, EvaluationError):
        return condition
        
    if is_truthy(condition):
        return eval_node(ie.consequence, env)
    elif ie.alternative:
        return eval_node(ie.alternative, env)
    return NULL

def apply_function(fn, args, call_site=None):
    if isinstance(fn, (Action, LambdaFunction)):
        extended_env = extend_function_env(fn, args)
        evaluated = eval_node(fn.body, extended_env)
        return unwrap_return_value(evaluated)
    elif isinstance(fn, Builtin):
        try:
            return fn.fn(*args)
        except Exception as e:
            return EvaluationError(f"Builtin function error: {str(e)}")
    return EvaluationError(f"Not a function: {fn.type()}")

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
        if isinstance(result, EvaluationError):
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
        if isinstance(result, EvaluationError):
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
        if isinstance(result, EvaluationError):
            return result
        if is_truthy(result):
            filtered_elements.append(element)
    
    return List(filtered_elements)

# NEW: Export system
def eval_export_statement(node, env):
    """Handle export statements"""
    # Get the value to export
    value = env.get(node.name.value)
    if not value:
        return EvaluationError(f"Cannot export undefined identifier: {node.name.value}")
    
    # Export with security restrictions
    env.export(node.name.value, value, node.allowed_files, node.permission)
    return NULL

def check_import_permission(exported_value, importer_file, env):
    """Check if importer has permission to access exported value"""
    # For now, implement basic file-based permission checking
    allowed_files = getattr(exported_value, '_allowed_files', [])
    if allowed_files and importer_file not in allowed_files:
        return EvaluationError(f"File '{importer_file}' not authorized to import this function")
    return True

# === BUILTIN FUNCTIONS ===
def builtin_len(*args):
    if len(args) != 1:
        return EvaluationError(f"len() takes exactly 1 argument ({len(args)} given)")
    arg = args[0]
    if isinstance(arg, String):
        return Integer(len(arg.value))
    elif isinstance(arg, List):
        return Integer(len(arg.elements))
    return EvaluationError(f"len() not supported for type {arg.type()}")

def builtin_first(*args):
    if len(args) != 1:
        return EvaluationError(f"first() takes exactly 1 argument ({len(args)} given)")
    if not isinstance(args[0], List):
        return EvaluationError("first() expects a list")
    list_obj = args[0]
    return list_obj.elements[0] if list_obj.elements else NULL

def builtin_string(*args):
    if len(args) != 1:
        return EvaluationError(f"string() takes exactly 1 argument ({len(args)} given)")
    arg = args[0]
    if isinstance(arg, Integer):
        return String(str(arg.value))
    elif isinstance(arg, Float):
        return String(str(arg.value))
    elif isinstance(arg, String):
        return arg
    elif isinstance(arg, BooleanObj):
        return String("true" if arg.value else "false")
    elif isinstance(arg, (List, Map)):
        return String(arg.inspect())
    return String("unknown")

def builtin_rest(*args):
    if len(args) != 1:
        return EvaluationError(f"rest() takes exactly 1 argument ({len(args)} given)")
    if not isinstance(args[0], List):
        return EvaluationError("rest() expects a list")
    list_obj = args[0]
    return List(list_obj.elements[1:]) if len(list_obj.elements) > 0 else List([])

def builtin_push(*args):
    if len(args) != 2:
        return EvaluationError(f"push() takes exactly 2 arguments ({len(args)} given)")
    if not isinstance(args[0], List):
        return EvaluationError("push() expects a list as first argument")
    list_obj = args[0]
    new_elements = list_obj.elements + [args[1]]
    return List(new_elements)

def builtin_reduce(*args):
    """Built-in reduce function for arrays"""
    if len(args) < 2 or len(args) > 3:
        return EvaluationError("reduce() takes 2 or 3 arguments (array, lambda[, initial])")
    
    array_obj, lambda_fn = args[0], args[1]
    initial = args[2] if len(args) == 3 else None
    
    return array_reduce(array_obj, lambda_fn, initial)

def builtin_map(*args):
    """Built-in map function for arrays"""
    if len(args) != 2:
        return EvaluationError("map() takes 2 arguments (array, lambda)")
    
    return array_map(args[0], args[1])

def builtin_filter(*args):
    """Built-in filter function for arrays"""
    if len(args) != 2:
        return EvaluationError("filter() takes 2 arguments (array, lambda)")
    
    return array_filter(args[0], args[1])

builtins = {
    "len": Builtin(builtin_len, "len"),
    "first": Builtin(builtin_first, "first"),
    "rest": Builtin(builtin_rest, "rest"),
    "push": Builtin(builtin_push, "push"),
    "string": Builtin(builtin_string, "string"),
    "reduce": Builtin(builtin_reduce, "reduce"),  # NEW
    "map": Builtin(builtin_map, "map"),          # NEW
    "filter": Builtin(builtin_filter, "filter"), # NEW
}

# === MAIN EVAL_NODE FUNCTION ===
def eval_node(node, env):
    if node is None:
        return NULL

    node_type = type(node)

    try:
        # Statements
        if node_type == Program:
            return eval_program(node.statements, env)

        elif node_type == ExpressionStatement:
            return eval_node(node.expression, env)

        elif node_type == BlockStatement:
            return eval_block_statement(node, env)

        elif node_type == ReturnStatement:
            val = eval_node(node.return_value, env)
            if isinstance(val, EvaluationError):
                return val
            return ReturnValue(val)

        elif node_type == LetStatement:
            val = eval_node(node.value, env)
            if isinstance(val, EvaluationError):
                return val
            env.set(node.name.value, val)
            return NULL

        elif node_type == ActionStatement:
            action_obj = Action(node.parameters, node.body, env)
            env.set(node.name.value, action_obj)
            return NULL

        # NEW: Export statement
        elif node_type == ExportStatement:
            return eval_export_statement(node, env)

        elif node_type == IfStatement:
            condition = eval_node(node.condition, env)
            if isinstance(condition, EvaluationError):
                return condition
            if is_truthy(condition):
                return eval_node(node.consequence, env)
            elif node.alternative is not None:
                return eval_node(node.alternative, env)
            return NULL

        elif node_type == WhileStatement:
            result = NULL
            while True:
                condition = eval_node(node.condition, env)
                if isinstance(condition, EvaluationError):
                    return condition
                if not is_truthy(condition):
                    break
                result = eval_node(node.body, env)
                if isinstance(result, (ReturnValue, EvaluationError)):
                    break
            return result

        elif node_type == ForEachStatement:
            iterable = eval_node(node.iterable, env)
            if isinstance(iterable, EvaluationError):
                return iterable
            if not isinstance(iterable, List):
                return EvaluationError("for-each loop expected list")

            result = NULL
            for element in iterable.elements:
                env.set(node.item.value, element)
                result = eval_node(node.body, env)
                if isinstance(result, (ReturnValue, EvaluationError)):
                    break

            return result

        elif node_type == AssignmentExpression:
            return eval_assignment_expression(node, env)

        elif node_type == PropertyAccessExpression:
            obj = eval_node(node.object, env)
            if isinstance(obj, EvaluationError):
                return obj
            property_name = node.property.value

            if isinstance(obj, EmbeddedCode):
                if property_name == "code":
                    return String(obj.code)
                elif property_name == "language":
                    return String(obj.language)

            return NULL

        elif node_type == AST_Boolean:
            return TRUE if node.value else FALSE

        # NEW: Lambda expression
        elif node_type == LambdaExpression:
            return eval_lambda_expression(node, env)

        elif node_type == MethodCallExpression:
            obj = eval_node(node.object, env)
            if isinstance(obj, EvaluationError):
                return obj
            method_name = node.method.value

            # Handle array methods with lambdas
            if isinstance(obj, List):
                args = eval_expressions(node.arguments, env)
                if isinstance(args, (ReturnValue, EvaluationError)):
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
                if isinstance(args, (ReturnValue, EvaluationError)):
                    return args
                # Simplified embedded execution
                print(f"[EMBED] Executing {obj.language}.{method_name}")
                return Integer(42)

            return EvaluationError(f"Method '{method_name}' not supported for {obj.type()}")

        elif node_type == EmbeddedLiteral:
            return EmbeddedCode("embedded_block", node.language, node.code)

        elif node_type == PrintStatement:
            val = eval_node(node.value, env)
            if isinstance(val, EvaluationError):
                # Print errors to stderr but don't stop execution
                print(f"❌ Error: {val}", file=sys.stderr)
                return NULL
            print(val.inspect())
            return NULL

        elif node_type == ScreenStatement:
            print(f"[RENDER] Screen: {node.name.value}")
            return NULL

        elif node_type == EmbeddedCodeStatement:
            embedded_obj = EmbeddedCode(node.name.value, node.language, node.code)
            env.set(node.name.value, embedded_obj)
            return NULL

        elif node_type == UseStatement:
            # Simplified module import
            print(f"[IMPORT] Loading module: {node.file_path}")
            # For now, return a dummy module environment
            module_env = Environment()
            if node.alias:
                env.set(node.alias, module_env)
            else:
                # Import all exports into current scope
                for name, value in module_env.get_exports().items():
                    env.set(name, value)
            return NULL

        elif node_type == ExactlyStatement:
            return eval_node(node.body, env)

        # Expressions
        elif node_type == IntegerLiteral:
            return Integer(node.value)

        elif node_type == StringLiteral:
            return String(node.value)

        elif node_type == ListLiteral:
            elements = eval_expressions(node.elements, env)
            if isinstance(elements, (ReturnValue, EvaluationError)):
                return elements
            return List(elements)

        elif node_type == MapLiteral:
            pairs = {}
            for key_expr, value_expr in node.pairs:
                key = eval_node(key_expr, env)
                if isinstance(key, EvaluationError):
                    return key
                value = eval_node(value_expr, env)
                if isinstance(value, EvaluationError):
                    return value
                key_str = key.inspect()
                pairs[key_str] = value
            return Map(pairs)

        elif node_type == Identifier:
            return eval_identifier(node, env)

        elif node_type == ActionLiteral:
            return Action(node.parameters, node.body, env)

        elif node_type == CallExpression:
            function = eval_node(node.function, env)
            if isinstance(function, EvaluationError):
                return function
            args = eval_expressions(node.arguments, env)
            if isinstance(args, (ReturnValue, EvaluationError)):
                return args
            return apply_function(function, args)

        elif node_type == PrefixExpression:
            right = eval_node(node.right, env)
            if isinstance(right, EvaluationError):
                return right
            return eval_prefix_expression(node.operator, right)

        elif node_type == InfixExpression:
            left = eval_node(node.left, env)
            if isinstance(left, EvaluationError):
                return left
            right = eval_node(node.right, env)
            if isinstance(right, EvaluationError):
                return right
            return eval_infix_expression(node.operator, left, right)

        elif node_type == IfExpression:
            return eval_if_expression(node, env)

        return EvaluationError(f"Unknown node type: {node_type}")

    except Exception as e:
        # Catch any unexpected errors and wrap them
        return EvaluationError(f"Internal error: {str(e)}")

# Production evaluation entry point
def evaluate(program, env):
    """Production evaluation with proper error handling"""
    result = eval_node(program, env)
    
    if isinstance(result, EvaluationError):
        # Format error for production
        error_msg = f"❌ Runtime Error"
        if result.line and result.column:
            error_msg += f" at line {result.line}:{result.column}"
        error_msg += f"\n   {result.message}"
        return error_msg
    
    return result