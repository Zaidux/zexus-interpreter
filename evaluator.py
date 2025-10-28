# evaluator.py (COMPLETE FIXED VERSION)
import zexus_ast
from zexus_ast import (
    Program, ExpressionStatement, BlockStatement, ReturnStatement, LetStatement,
    ActionStatement, IfStatement, WhileStatement, ForEachStatement, MethodCallExpression,
    EmbeddedLiteral, PrintStatement, ScreenStatement, EmbeddedCodeStatement, UseStatement,
    ExactlyStatement, IntegerLiteral, StringLiteral, ListLiteral, MapLiteral, Identifier,
    ActionLiteral, CallExpression, PrefixExpression, InfixExpression, IfExpression,
    Boolean as AST_Boolean, AssignmentExpression, PropertyAccessExpression
)

from object import Environment, Integer, Float, String, List, Map, Null, Boolean as BooleanObj, Builtin, Action, EmbeddedCode, ReturnValue

NULL, TRUE, FALSE = Null(), BooleanObj(True), BooleanObj(False)

# === CONSTANT CACHE FOR PERFORMANCE ===
constant_cache = {}

def get_cached_integer(value):
    """Cache integer objects for performance"""
    if value not in constant_cache:
        constant_cache[value] = Integer(value)
    return constant_cache[value]

def get_cached_boolean(value):
    """Cache boolean objects for performance"""
    key = f"bool_{value}"
    if key not in constant_cache:
        constant_cache[key] = TRUE if value else FALSE
    return constant_cache[key]

# === HELPER FUNCTIONS ===

def eval_program(statements, env):
    result = NULL
    for stmt in statements:
        result = eval_node(stmt, env)
        if isinstance(result, ReturnValue):
            return result.value
    return result

def eval_assignment_expression(node, env):
    """Handle assignment expressions like: x = 5"""
    value = eval_node(node.value, env)
    if value is None:
        return NULL

    # Set the variable in the environment
    env.set(node.name.value, value)
    return value

def eval_block_statement(block, env):
    result = NULL
    for stmt in block.statements:
        result = eval_node(stmt, env)
        if isinstance(result, ReturnValue):
            return result
    return result

def eval_expressions(expressions, env):
    results = []
    for expr in expressions:
        result = eval_node(expr, env)
        if isinstance(result, ReturnValue):
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
    print(f"Identifier not found: {node.value}")
    return NULL

def is_truthy(obj):
    if obj == NULL or obj == FALSE:
        return False
    return True

def eval_prefix_expression(operator, right):
    if operator == "!":
        return eval_bang_operator_expression(right)
    elif operator == "-":
        return eval_minus_prefix_operator_expression(right)
    return NULL

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
    return NULL

def eval_infix_expression(operator, left, right):
    print(f"DEBUG: eval_infix_expression: {left.inspect()} {operator} {right.inspect()}")
    
    # ✅ ADD logical operators first
    if operator == "&&":
        result = TRUE if is_truthy(left) and is_truthy(right) else FALSE
        print(f"DEBUG: logical AND result: {result.inspect()}")
        return result
    elif operator == "||":
        result = TRUE if is_truthy(left) or is_truthy(right) else FALSE
        print(f"DEBUG: logical OR result: {result.inspect()}")
        return result

    # Handle comparison operators for all types
    if operator == "==":
        result = TRUE if left.value == right.value else FALSE
        print(f"DEBUG: equality result: {result.inspect()}")
        return result
    elif operator == "!=":
        result = TRUE if left.value != right.value else FALSE
        print(f"DEBUG: inequality result: {result.inspect()}")
        return result
    elif operator == "<=":
        result = TRUE if left.value <= right.value else FALSE
        print(f"DEBUG: less-equal result: {result.inspect()}")
        return result
    elif operator == ">=":
        result = TRUE if left.value >= right.value else FALSE
        print(f"DEBUG: greater-equal result: {result.inspect()}")
        return result

    # Type-specific operations
    if isinstance(left, Integer) and isinstance(right, Integer):
        result = eval_integer_infix_expression(operator, left, right)
        print(f"DEBUG: integer operation result: {result.inspect()}")
        return result
    elif isinstance(left, Float) and isinstance(right, Float):
        result = eval_float_infix_expression(operator, left, right)
        print(f"DEBUG: float operation result: {result.inspect()}")
        return result
    elif isinstance(left, String) and isinstance(right, String):
        result = eval_string_infix_expression(operator, left, right)
        print(f"DEBUG: string operation result: {result.inspect()}")
        return result
    
    print(f"DEBUG: unsupported operation: {left.type()} {operator} {right.type()}")
    return NULL

def eval_integer_infix_expression(operator, left, right):
    left_val = left.value
    right_val = right.value

    print(f"DEBUG: integer operation: {left_val} {operator} {right_val}")

    if operator == "+":
        return Integer(left_val + right_val)
    elif operator == "-":
        return Integer(left_val - right_val)
    elif operator == "*":
        return Integer(left_val * right_val)
    elif operator == "/":
        return Integer(left_val // right_val)
    elif operator == "%":  # ✅ ADD modulo operator
        if right_val == 0:
            print("Error: Modulo by zero")
            return NULL
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
    return NULL

def eval_float_infix_expression(operator, left, right):
    left_val = left.value
    right_val = right.value

    print(f"DEBUG: float operation: {left_val} {operator} {right_val}")

    if operator == "+":
        return Float(left_val + right_val)
    elif operator == "-":
        return Float(left_val - right_val)
    elif operator == "*":
        return Float(left_val * right_val)
    elif operator == "/":
        return Float(left_val / right_val)
    elif operator == "%":  # ✅ ADD modulo operator for floats
        if right_val == 0:
            print("Error: Modulo by zero")
            return NULL
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
    return NULL

def eval_string_infix_expression(operator, left, right):
    print(f"DEBUG: string operation: '{left.value}' {operator} '{right.value}'")
    
    if operator == "+":
        return String(left.value + right.value)
    elif operator == "==":
        return TRUE if left.value == right.value else FALSE
    elif operator == "!=":
        return TRUE if left.value != right.value else FALSE
    return NULL

def eval_if_expression(ie, env):
    condition = eval_node(ie.condition, env)
    if is_truthy(condition):
        return eval_node(ie.consequence, env)
    elif ie.alternative:
        return eval_node(ie.alternative, env)
    return NULL

def apply_function(fn, args):
    if isinstance(fn, Action):
        extended_env = extend_function_env(fn, args)
        evaluated = eval_node(fn.body, extended_env)
        return unwrap_return_value(evaluated)
    elif isinstance(fn, Builtin):
        return fn.fn(*args)
    print(f"Not a function: {fn.type()}")
    return NULL

def extend_function_env(fn, args):
    env = Environment(outer=fn.env)
    for param, arg in zip(fn.parameters, args):
        env.set(param.value, arg)
    return env

def unwrap_return_value(obj):
    if isinstance(obj, ReturnValue):
        return obj.value
    return obj

def execute_embedded_function(embedded_obj, method, args):
    # Simplified embedded function execution
    print(f"[EMBED] Executing {embedded_obj.language}.{method} with args {[arg.inspect() for arg in args]}")
    # For now, return a dummy value
    return Integer(42)

# === BUILTIN FUNCTIONS ===
def builtin_len(*args):
    if len(args) != 1:
        return NULL
    arg = args[0]
    if isinstance(arg, String):
        return Integer(len(arg.value))
    elif isinstance(arg, List):
        return Integer(len(arg.elements))
    return NULL

def builtin_first(*args):
    if len(args) != 1 or not isinstance(args[0], List):
        return NULL
    list_obj = args[0]
    if len(list_obj.elements) > 0:
        return list_obj.elements[0]
    return NULL

def builtin_string(*args):
    if len(args) != 1:
        return NULL
    arg = args[0]
    if isinstance(arg, Integer):
        return String(str(arg.value))
    elif isinstance(arg, Float):
        return String(str(arg.value))
    elif isinstance(arg, String):
        return arg
    elif isinstance(arg, BooleanObj):
        return String("true" if arg.value else "false")
    elif isinstance(arg, Map):
        return String(arg.inspect())
    elif isinstance(arg, List):
        return String(arg.inspect())
    return String("unknown")

def builtin_rest(*args):
    if len(args) != 1 or not isinstance(args[0], List):
        return NULL
    list_obj = args[0]
    if len(list_obj.elements) > 0:
        new_elements = list_obj.elements[1:]
        return List(new_elements)
    return NULL

def builtin_push(*args):
    if len(args) != 2 or not isinstance(args[0], List):
        return NULL
    list_obj = args[0]
    new_element = args[1]
    new_elements = list_obj.elements + [new_element]
    return List(new_elements)

def builtin_map_get(*args):
    if len(args) != 2:
        return NULL
    map_obj, key = args[0], args[1]
    if not isinstance(map_obj, Map):
        return NULL
    key_str = key.inspect()
    return map_obj.pairs.get(key_str, NULL)

def builtin_map_set(*args):
    if len(args) != 3:
        return NULL
    map_obj, key, value = args[0], args[1], args[2]
    if not isinstance(map_obj, Map):
        return NULL
    key_str = key.inspect()
    map_obj.pairs[key_str] = value
    return map_obj

def builtin_map_keys(*args):
    if len(args) != 1:
        return NULL
    map_obj = args[0]
    if not isinstance(map_obj, Map):
        return NULL
    keys = [String(key) for key in map_obj.pairs.keys()]
    return List(keys)

builtins = {
    "len": Builtin(builtin_len),
    "first": Builtin(builtin_first),
    "rest": Builtin(builtin_rest),
    "push": Builtin(builtin_push),
    "string": Builtin(builtin_string),
    "map_get": Builtin(builtin_map_get),
    "map_set": Builtin(builtin_map_set),
    "map_keys": Builtin(builtin_map_keys),
}

# === MAIN EVAL_NODE FUNCTION ===
def eval_node(node, env):
    if node is None:
        return NULL

    node_type = type(node)

    # Statements
    if node_type == Program:
        return eval_program(node.statements, env)

    elif node_type == ExpressionStatement:
        return eval_node(node.expression, env)

    elif node_type == BlockStatement:
        return eval_block_statement(node, env)

    elif node_type == ReturnStatement:
        val = eval_node(node.return_value, env)
        return ReturnValue(val)

    elif node_type == LetStatement:
        val = eval_node(node.value, env)
        env.set(node.name.value, val)
        return NULL

    elif node_type == ActionStatement:
        action_obj = Action(node.parameters, node.body, env)
        env.set(node.name.value, action_obj)
        return NULL

    elif node_type == IfStatement:
        condition = eval_node(node.condition, env)
        if is_truthy(condition):
            return eval_node(node.consequence, env)
        elif node.alternative is not None:
            return eval_node(node.alternative, env)
        return NULL

    elif node_type == WhileStatement:
        result = NULL
        while True:
            condition = eval_node(node.condition, env)
            if not is_truthy(condition):
                break
            result = eval_node(node.body, env)
        return result

    elif node_type == ForEachStatement:
        # Evaluate the iterable
        iterable = eval_node(node.iterable, env)
        if not isinstance(iterable, List):
            print(f"Error: for-each loop expected list, got {iterable.type()}")
            return NULL

        result = NULL
        # For each element in the iterable
        for element in iterable.elements:
            # Set the loop variable
            env.set(node.item.value, element)
            # Execute the loop body
            result = eval_node(node.body, env)

        return result

    # ✅ FIXED: Only one AssignmentExpression case with proper indentation
    elif node_type == AssignmentExpression:
        return eval_assignment_expression(node, env)

    elif node_type == PropertyAccessExpression:
        obj = eval_node(node.object, env)
        property_name = node.property.value

        # Handle embedded code properties
        if isinstance(obj, EmbeddedCode):
            if property_name == "code":
                return String(obj.code)
            elif property_name == "language":
                return String(obj.language)

        # For now, return NULL for other property access
        return NULL

    elif node_type == AST_Boolean:
        return get_cached_boolean(node.value)

    elif node_type == MethodCallExpression:
        obj = eval_node(node.object, env)
        method_name = node.method.value

        # Handle embedded code method calls
        if isinstance(obj, EmbeddedCode):
            args = eval_expressions(node.arguments, env)
            return execute_embedded_function(obj, method_name, args)

        # Handle regular method calls
        print(f"Method call on {obj.type()}.{method_name} not implemented yet")
        return NULL

    elif node_type == EmbeddedLiteral:
        embedded_obj = EmbeddedCode("embedded_block", node.language, node.code)
        return embedded_obj

    elif node_type == PrintStatement:
        val = eval_node(node.value, env)
        if val:
            print(val.inspect())
        return NULL

    elif node_type == ScreenStatement:
        print(f"[RENDER] Simulating screen '{node.name.value}'...")
        return NULL

    elif node_type == EmbeddedCodeStatement:
        embedded_obj = EmbeddedCode(node.name.value, node.language, node.code)
        env.set(node.name.value, embedded_obj)
        print(f"[EMBED] Stored {node.language} code as '{node.name.value}'")
        return NULL

    elif node_type == UseStatement:
        embedded_obj = env.get(node.embedded_ref.value)
        if not embedded_obj or not isinstance(embedded_obj, EmbeddedCode):
            print(f"[ERROR] Embedded code '{node.embedded_ref.value}' not found")
            return NULL
        args = eval_expressions(node.arguments, env)
        result = execute_embedded_function(embedded_obj, node.method, args)
        return result

    elif node_type == ExactlyStatement:
        print(f"[EXACTLY] Executing exact block '{node.name.value}'")
        return eval_node(node.body, env)

    # Expressions
    elif node_type == IntegerLiteral:
        return get_cached_integer(node.value)

    elif node_type == StringLiteral:
        return String(node.value)

    elif node_type == ListLiteral:
        elements = eval_expressions(node.elements, env)
        return List(elements)

    elif node_type == MapLiteral:
        pairs = {}
        for key_expr, value_expr in node.pairs:
            key = eval_node(key_expr, env)
            value = eval_node(value_expr, env)
            key_str = key.inspect()
            pairs[key_str] = value
        return Map(pairs)

    elif node_type == Identifier:
        return eval_identifier(node, env)

    elif node_type == ActionLiteral:
        params = node.parameters
        body = node.body
        return Action(params, body, env)

    elif node_type == CallExpression:
        function = eval_node(node.function, env)
        args = eval_expressions(node.arguments, env)
        return apply_function(function, args)

    elif node_type == PrefixExpression:
        right = eval_node(node.right, env)
        return eval_prefix_expression(node.operator, right)

    elif node_type == InfixExpression:
        left = eval_node(node.left, env)
        right = eval_node(node.right, env)
        return eval_infix_expression(node.operator, left, right)

    elif node_type == IfExpression:
        return eval_if_expression(node, env)

    print(f"Unknown node type: {node_type}")
    return NULL