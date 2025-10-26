# evaluator.py (COMPLETE FIXED VERSION WITH IF/WHILE SUPPORT)
from zexus_ast import *
from object import *

NULL, TRUE, FALSE = Null(), Boolean(True), Boolean(False)

# --- Updated Built-in Functions Section ---
def builtin_len(*args):
    if len(args) != 1:
        return NULL
    arg = args[0]
    if isinstance(arg, String):
        return Integer(len(arg.value))
    elif isinstance(arg, List):
        return Integer(len(arg.elements))
    return NULL


def builtin_first(*args):  # NEW
    if len(args) != 1 or not isinstance(args[0], List):
        return NULL
    list_obj = args[0]
    if len(list_obj.elements) > 0:
        return list_obj.elements[0]
    return NULL


def builtin_string(*args):  # NEW
    if len(args) != 1:
        return NULL
    arg = args[0]
    if isinstance(arg, Integer):
        return String(str(arg.value))
    elif isinstance(arg, Float):
        return String(str(arg.value))
    elif isinstance(arg, String):
        return arg
    elif isinstance(arg, Boolean):
        return String("true" if arg.value else "false")  # ✅ FIXED
    elif isinstance(arg, Map):
        return String(arg.inspect())
    elif isinstance(arg, List):
        return String(arg.inspect())
    return String("unknown")


def builtin_rest(*args):  # NEW
    if len(args) != 1 or not isinstance(args[0], List):
        return NULL
    list_obj = args[0]
    if len(list_obj.elements) > 0:
        new_elements = list_obj.elements[1:]
        return List(new_elements)
    return NULL


def builtin_push(*args):  # NEW
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
    "last": Builtin(builtin_first),  # Assuming builtin_last will be similar
    "rest": Builtin(builtin_rest),
    "push": Builtin(builtin_push),
    "string": Builtin(builtin_string),
    "map_get": Builtin(builtin_map_get),
    "map_set": Builtin(builtin_map_set),
    "map_keys": Builtin(builtin_map_keys),
}
# --- End of Built-ins Section ---


def eval_node(node, env):
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
   
    # Add to evaluator.py in eval_node function
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

    # evaluator.py - ADD this case to your eval_node function
    elif node_type == ExactlyStatement:
        print(f"[EXACTLY] Executing exact block '{node.name.value}'")
        return eval_node(node.body, env)

    # Expressions
    elif node_type == IntegerLiteral:
        return Integer(node.value)

    elif node_type == StringLiteral:
        return String(node.value)

    elif node_type == Boolean:
        return TRUE if node.value else FALSE

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

    return NULL