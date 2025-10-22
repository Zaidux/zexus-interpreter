# evaluator.py (COMPLETE FIXED VERSION WITH IF/WHILE SUPPORT)
from zexus_ast import *
from object import *

NULL, TRUE, FALSE = Null(), Boolean(True), Boolean(False)

# --- Updated Built-in Functions Section ---
def builtin_len(*args):
    if len(args) != 1: return NULL
    arg = args[0]
    if isinstance(arg, String): return Integer(len(arg.value))
    elif isinstance(arg, List): return Integer(len(arg.elements))
    return NULL

def builtin_first(*args): # NEW
    if len(args) != 1 or not isinstance(args[0], List): return NULL
    list_obj = args[0]
    if len(list_obj.elements) > 0:
        return list_obj.elements[0]
    return NULL

# In evaluator.py - update builtin_string function
def builtin_string(*args): # NEW
    if len(args) != 1: return NULL
    arg = args[0]
    if isinstance(arg, Integer): return String(str(arg.value))
    elif isinstance(arg, Float): return String(str(arg.value))
    elif isinstance(arg, String): return arg
    elif isinstance(arg, Boolean): return String("true" if arg.value else "false")  # ✅ FIXED
    elif isinstance(arg, Map): return String(arg.inspect())
    elif isinstance(arg, List): return String(arg.inspect())
    return String("unknown")

def builtin_rest(*args): # NEW
    if len(args) != 1 or not isinstance(args[0], List): return NULL
    list_obj = args[0]
    if len(list_obj.elements) > 0:
        new_elements = list_obj.elements[1:]
        return List(new_elements)
    return NULL

def builtin_push(*args): # NEW
    if len(args) != 2 or not isinstance(args[0], List): return NULL
    list_obj = args[0]
    new_element = args[1]
    new_elements = list_obj.elements + [new_element]
    return List(new_elements)

def builtin_string(*args): # NEW
    if len(args) != 1: return NULL
    arg = args[0]
    if isinstance(arg, Integer): return String(str(arg.value))
    elif isinstance(arg, Float): return String(str(arg.value))  # ADD THIS
    elif isinstance(arg, String): return arg
    elif isinstance(arg, Boolean): return String("true" if arg.value else "false")
    elif isinstance(arg, Map): return String(arg.inspect())  # ADD THIS FOR MAPS
    elif isinstance(arg, List): return String(arg.inspect())
    return String("unknown")

def builtin_map_get(*args):
    if len(args) != 2: return NULL
    map_obj, key = args[0], args[1]
    if not isinstance(map_obj, Map): return NULL
    key_str = key.inspect()
    return map_obj.pairs.get(key_str, NULL)

def builtin_map_set(*args):
    if len(args) != 3: return NULL
    map_obj, key, value = args[0], args[1], args[2]
    if not isinstance(map_obj, Map): return NULL
    key_str = key.inspect()
    map_obj.pairs[key_str] = value
    return map_obj

def builtin_map_keys(*args):
    if len(args) != 1: return NULL
    map_obj = args[0]
    if not isinstance(map_obj, Map): return NULL
    # Convert keys back to String objects
    keys = [String(key) for key in map_obj.pairs.keys()]
    return List(keys)

builtins = {
    "len": Builtin(builtin_len),
    "first": Builtin(builtin_first),
    "last": Builtin(builtin_last),
    "rest": Builtin(builtin_rest),
    "push": Builtin(builtin_push),
    "string": Builtin(builtin_string),
    "map_get": Builtin(builtin_map_get),      # ADD THESE
    "map_set": Builtin(builtin_map_set),      # ADD THESE
    "map_keys": Builtin(builtin_map_keys),    # ADD THESE
}
# --- End of Built-ins Section ---

def eval_node(node, env):
    """
    Evaluates an AST node in the given environment.
    """
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
        # Unwrap if it's an error propagating
        # if isinstance(val, Error): return val
        return ReturnValue(val)

    elif node_type == LetStatement:
        # print(f"DEBUG: LetStatement - storing '{node.name.value}'")
        val = eval_node(node.value, env)
        # print(f"DEBUG: Storing '{node.name.value}' = {val}")
        # if isinstance(val, Error): return val
        env.set(node.name.value, val)
        return NULL

    # ✅ FIX: Add ActionStatement handling
    elif node_type == ActionStatement:
        # print(f"DEBUG: Processing ActionStatement - storing '{node.name.value}'")
        action_obj = Action(node.parameters, node.body, env)
        # print(f"DEBUG: Storing function '{node.name.value}' = {action_obj}")
        env.set(node.name.value, action_obj)
        return NULL

    # ✅ NEW: IfStatement handling
    elif node_type == IfStatement:
        condition = eval_node(node.condition, env)
        if is_truthy(condition):
            return eval_node(node.consequence, env)
        elif node.alternative is not None:
            return eval_node(node.alternative, env)
        return NULL

    # ✅ NEW: WhileStatement handling
    elif node_type == WhileStatement:
        result = NULL
        while True:
            condition = eval_node(node.condition, env)
            if not is_truthy(condition):
                break
            result = eval_node(node.body, env)
            # Handle break/continue in the future
        return result

    elif node_type == PrintStatement:
        val = eval_node(node.value, env)
        # if isinstance(val, Error): return val
        if val: print(val.inspect())
        return NULL

    elif node_type == ScreenStatement: # NEW
        # For now, we simulate rendering by printing a description.
        # In v1.5, this would call the real rendering engine.
        print(f"[RENDER] Simulating screen '{node.name.value}'...")
        # We could optionally evaluate the body to find UI components
        # eval_node(node.body, env)
        return NULL
    elif node_type == EmbeddedCodeStatement:
    # Store the embedded code in the environment
    embedded_obj = EmbeddedCode(node.name.value, node.language, node.code)
    env.set(node.name.value, embedded_obj)
    print(f"[EMBED] Stored {node.language} code as '{node.name.value}'")
    return NULL

    elif node_type == UseStatement:
    # Look up the embedded code block
    embedded_obj = env.get(node.embedded_ref.value)
    if not embedded_obj or not isinstance(embedded_obj, EmbeddedCode):
        print(f"[ERROR] Embedded code '{node.embedded_ref.value}' not found")
        return NULL
        
    # Execute the embedded function
    args =         eval_expressions(node.arguments, env)
    result = execute_embedded_function(embedded_obj, node.method, args)
    return result

    # Expressions
    elif node_type == IntegerLiteral:
        return Integer(node.value)

    elif node_type == StringLiteral:
        return String(node.value)

    elif node_type == Boolean:
        return TRUE if node.value else FALSE

    elif node_type == ListLiteral:
        elements = eval_expressions(node.elements, env)
        # if len(elements) == 1 and isinstance(elements[0], Error): return elements[0]
        return List(elements)

    elif node_type == MapLiteral:
        pairs = {}
        for key_expr, value_expr in node.pairs:
            key = eval_node(key_expr, env)
            value = eval_node(value_expr, env)
        # Use string representation as hashable key for now
            key_str = key.inspect()
            pairs[key_str] = value
        return Map(pairs)

    elif node_type == Identifier: # UPDATED: Now checks for built-ins
        return eval_identifier(node, env)

    elif node_type == ActionLiteral: # Action (function) definition
        # print(f"DEBUG: Processing ActionLiteral - params: {node.parameters}")
        params = node.parameters
        body = node.body
        action_obj = Action(params, body, env)
        # print(f"DEBUG: Created Action object: {action_obj}")
        return action_obj

    elif node_type == CallExpression: # UPDATED: Renamed to apply_function
        function = eval_node(node.function, env)
        # if isinstance(function, Error): return function
        args = eval_expressions(node.arguments, env)
        # if len(args) == 1 and isinstance(args[0], Error): return args[0]
        return apply_function(function, args)

    elif node_type == PrefixExpression:
        right = eval_node(node.right, env)
        # if isinstance(right, Error): return right
        return eval_prefix_expression(node.operator, right)

    elif node_type == InfixExpression:
        left = eval_node(node.left, env)
        # if isinstance(left, Error): return left
        right = eval_node(node.right, env)
        # if isinstance(right, Error): return right
        return eval_infix_expression(node.operator, left, right)

    elif node_type == IfExpression:
        return eval_if_expression(node, env)

    # In a full language, you'd return an error for unknown node types.
    return NULL

def eval_program(stmts, env):
    """
    Evaluates a program (list of statements).
    """
    result = NULL
    for statement in stmts:
        result = eval_node(statement, env)
        # Handle early returns or errors
        if isinstance(result, ReturnValue):
            return result.value
        # if isinstance(result, Error): return result
    return result

def eval_block_statement(block, env):
    """
    Evaluates a block of statements.
    """
    result = NULL
    for statement in block.statements:
        result = eval_node(statement, env)
        # Propagate return values or errors
        if result and (isinstance(result, ReturnValue) or isinstance(result, Error)):
            return result
    return result

def apply_function(fn, args): # UPDATED: Handles both Actions and Builtins
    # print(f"DEBUG: Applying function {fn}, args: {args}")
    # print(f"DEBUG: Function type: {type(fn).__name__}")
    
    """
    Applies an Action (user-defined function) or a Builtin function.
    """
    if isinstance(fn, Action):
        # print(f"DEBUG: Action params: {fn.parameters}, body: {fn.body}")
        extended_env = extend_action_env(fn, args)
        evaluated = eval_node(fn.body, extended_env)
        # print(f"DEBUG: Action evaluated to: {evaluated}")

        # Unwrap ReturnValue if present
        if isinstance(evaluated, ReturnValue):
            return evaluated.value
        return evaluated
    elif isinstance(fn, Builtin): # NEW: Handle built-in functions
        return fn.fn(*args)

    # In a full language, return an error object.
    # return Error("not an action or builtin: {}".format(fn.type()))
    return NULL # Or an error: not a function

def extend_action_env(fn, args): # For Action (user-defined function) calls
    """
    Creates a new environment for an action call, extending the action's environment
    and binding arguments to parameters.
    """
    # print(f"DEBUG: Extending environment for {fn.parameters} with args {args}")

    env = Environment(outer=fn.env)
    for i, param in enumerate(fn.parameters):
        # In a full language, check if len(args) matches len(fn.parameters)
        # and handle argument errors.
        if i < len(args):
            # print(f"DEBUG: Setting {param.value} = {args[i]}")
            env.set(param.value, args[i])
        # else: handle missing arguments
    return env

def execute_embedded_function(embedded_obj, method, args):
    """
    Executes a function from embedded code.
    For now, we'll simulate this. In a real implementation, you'd use:
    - exec() for Python
    - eval() for JavaScript  
    - etc.
    """
    print(f"[EMBED] Executing {embedded_obj.language}.{method} with args {args}")
    
    # Simulate execution for different languages
    if embedded_obj.language == "javascript":
        if method == "add":
            if len(args) == 2 and args[0].type() == "INTEGER" and args[1].type() == "INTEGER":
                return Integer(args[0].value + args[1].value)
        elif method == "multiply":
            if len(args) == 2 and args[0].type() == "INTEGER" and args[1].type() == "INTEGER":
                return Integer(args[0].value * args[1].value)
                
    elif embedded_obj.language == "python":
        if method == "process_data":
            if len(args) == 1 and args[0].type() == "LIST":
                # Simulate processing: double each element
                new_elements = [Integer(el.value * 2) for el in args[0].elements]
                return List(new_elements)
                
    # Default: return first argument (for simulation)
    if args:
        return args[0]
    return NULL

def eval_identifier(node, env): # UPDATED: Now checks for built-ins
    """
    Evaluates an identifier by looking it up in the environment or built-ins.
    """
    # print(f"DEBUG: Looking up identifier '{node.value}' in environment")

    val = env.get(node.value)
    if val:
        # print(f"DEBUG: Found '{node.value}' = {val}")
        return val

    builtin = builtins.get(node.value)
    if builtin:
        # print(f"DEBUG: Found builtin '{node.value}'")
        return builtin

    # print(f"DEBUG: Identifier '{node.value}' NOT FOUND - returning NULL")
    # In a full language, return an error for undefined identifier.
    # return Error("identifier not found: " + node.value)
    return NULL

def eval_expressions(exps, env):
    """
    Evaluates a list of expressions.
    """
    result = []
    for e in exps:
        evaluated = eval_node(e, env)
        # if isinstance(evaluated, Error): return [evaluated] # Propagate error
        result.append(evaluated)
    return result

def eval_prefix_expression(operator, right):
    """
    Evaluates a prefix expression (e.g., !true, -5).
    """
    if operator == "!": return eval_bang_operator_expression(right)
    if operator == "-": return eval_minus_prefix_operator(right)

    # In a full language, return an error for unknown operator.
    # return Error("unknown operator: {}{}".format(operator, right.type()))
    return NULL

def eval_infix_expression(operator, left, right):
    """
    Evaluates an infix expression (e.g., 5 + 5, true == false).
    """
    # UPDATE this condition to handle both integers and floats
    if (left.type() == "INTEGER" or left.type() == "FLOAT") and \
       (right.type() == "INTEGER" or right.type() == "FLOAT"):
        return eval_integer_infix_expression(operator, left, right)
    elif left.type() == "STRING" and right.type() == "STRING":
        if operator == "+": return String(left.value + right.value)
    elif left.type() == "STRING" and right.type() == "INTEGER":
        if operator == "+": return String(left.value + str(right.value))
    elif left.type() == "STRING" and right.type() == "FLOAT":
        if operator == "+": return String(left.value + str(right.value))
    elif left.type() == "INTEGER" and right.type() == "STRING":
        if operator == "+": return String(str(left.value) + right.value)
    elif left.type() == "FLOAT" and right.type() == "STRING":
        if operator == "+": return String(str(left.value) + right.value)
    elif operator == "==": return native_bool_to_boolean_object(left.value == right.value)
    elif operator == "!=": return native_bool_to_boolean_object(left.value != right.value)
    # In a full language, return an error for type mismatch or unknown operator.
    # return Error("type mismatch: {} {} {}".format(left.type(), operator, right.type()))
    return NULL

def eval_if_expression(if_node, env):
    """
    Evaluates an if-else expression.
    """
    condition = eval_node(if_node.condition, env)
    # if isinstance(condition, Error): return condition

    if is_truthy(condition):
        return eval_node(if_node.consequence, env)
    elif if_node.alternative is not None:
        return eval_node(if_node.alternative, env)
    else:
        return NULL

def is_truthy(obj):
    """
    Determines if an object is truthy.
    """
    return obj not in [NULL, FALSE]

def eval_bang_operator_expression(right):
    """
    Evaluates the '!' (bang) operator.
    """
    if right == TRUE: return FALSE
    if right == FALSE: return TRUE
    if right == NULL: return TRUE # Monkey/Maki language treats NULL as falsy, so !NULL is true
    return FALSE # For any other object, treat as truthy, so !obj is false

def eval_minus_prefix_operator(right):
    """
    Evaluates the '-' (minus) prefix operator.
    """
    # UPDATE to handle both integers and floats
    if right.type() == "INTEGER":
        return Integer(-right.value)
    elif right.type() == "FLOAT":
        return Float(-right.value)
    else:
        # In a full language, return an error object.
        # return Error("unknown operator: -{}".format(right.type()))
        return NULL

def eval_integer_infix_expression(operator, left, right):
    """
    Evaluates infix expressions for integers AND floats.
    """
    # Handle mixed int/float operations by converting to float
    if left.type() == "INTEGER":
        left_val = float(left.value)
    else:
        left_val = left.value

    if right.type() == "INTEGER":
        right_val = float(right.value)
    else:
        right_val = right.value

    # Perform the operation
    if operator == "+":
        result = left_val + right_val
    elif operator == "-":
        result = left_val - right_val
    elif operator == "*":
        result = left_val * right_val
    elif operator == "/":
        result = left_val / right_val
    elif operator == "<":
        return native_bool_to_boolean_object(left_val < right_val)
    elif operator == ">":
        return native_bool_to_boolean_object(left_val > right_val)
    elif operator == "==":
        return native_bool_to_boolean_object(left_val == right_val)
    elif operator == "!=":
        return native_bool_to_boolean_object(left_val != right_val)
    else:
        return NULL

    # Return appropriate type (int if whole number, float otherwise)
    if result.is_integer():
        return Integer(int(result))
    else:
        return Float(result)

def native_bool_to_boolean_object(value):
    """
    Converts a Python boolean to a Boolean object.
    """
    return TRUE if value else FALSE
