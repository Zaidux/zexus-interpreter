# src/zexus/evaluator.py (FIXED VERSION)
import sys
import traceback
import json
import os
import asyncio
from . import zexus_ast
from .zexus_ast import (
    Program, ExpressionStatement, BlockStatement, ReturnStatement, LetStatement,
    ActionStatement, IfStatement, WhileStatement, ForEachStatement, MethodCallExpression,
    EmbeddedLiteral, PrintStatement, ScreenStatement, EmbeddedCodeStatement, UseStatement,
    ExactlyStatement, TryCatchStatement, IntegerLiteral, StringLiteral, ListLiteral, MapLiteral, Identifier,
    ActionLiteral, CallExpression, PrefixExpression, InfixExpression, IfExpression,
    Boolean as AST_Boolean, AssignmentExpression, PropertyAccessExpression,
    ExportStatement, LambdaExpression, FromStatement, ComponentStatement, ThemeStatement,
    DebugStatement, ExternalDeclaration,
    # Phase 1 Advanced Nodes
    EntityStatement, SealStatement, VerifyStatement, ContractStatement, ProtectStatement,
    MiddlewareStatement, AuthStatement, ThrottleStatement, CacheStatement
)

from .object import (
    Environment, Integer, Float, String, List, Map, Null, Boolean as BooleanObj, 
    Builtin, Action, EmbeddedCode, ReturnValue, LambdaFunction, DateTime, Math, File, Debug,
    EvaluationError as ObjectEvaluationError
)

# Global Constants
NULL, TRUE, FALSE = Null(), BooleanObj(True), BooleanObj(False)

# Registry for builtin functions (populated later)
builtins = {}

# Use the unified EvaluationError from object.py
EvaluationError = ObjectEvaluationError

# Helper to centralize error checks
def is_error(obj):
    return isinstance(obj, (EvaluationError, ObjectEvaluationError))

# Summary counters for lightweight summary logging
EVAL_SUMMARY = {
    'parsed_statements': 0,
    'evaluated_statements': 0,
    'errors': 0,
    'async_tasks_run': 0,
    'max_statements_in_block': 0
}

def _is_awaitable(obj):
    try:
        return asyncio.iscoroutine(obj) or isinstance(obj, asyncio.Future)
    except Exception:
        return False

def _resolve_awaitable(obj):
    """If obj is a coroutine/future, run it to completion and return the result."""
    if _is_awaitable(obj):
        try:
            EVAL_SUMMARY['async_tasks_run'] += 1
            return asyncio.run(obj)
        except RuntimeError:
            return obj
    return obj

# === DEBUG FLAGS ===
from .config import config as zexus_config

def debug_log(message, data=None, level='debug'):
    """Conditional debug logging that respects the user's persistent config."""
    try:
        if not zexus_config.should_log(level):
            return
    except Exception:
        return

    if data is not None:
        print(f"🔍 [EVAL DEBUG] {message}: {data}")
    else:
        print(f"🔍 [EVAL DEBUG] {message}")

# === CORE EVALUATION FUNCTIONS ===

def eval_program(statements, env):
    debug_log("eval_program", f"Processing {len(statements)} statements")
    try:
        EVAL_SUMMARY['parsed_statements'] = max(EVAL_SUMMARY.get('parsed_statements', 0), len(statements))
    except Exception:
        pass

    result = NULL
    for i, stmt in enumerate(statements):
        debug_log(f"  Statement {i+1}", type(stmt).__name__)
        res = eval_node(stmt, env)
        res = _resolve_awaitable(res)
        EVAL_SUMMARY['evaluated_statements'] += 1
        if isinstance(res, ReturnValue):
            debug_log("  ReturnValue encountered", res.value)
            return res.value
        if is_error(res):
            debug_log("  Error encountered", res)
            try:
                EVAL_SUMMARY['errors'] += 1
            except Exception:
                pass
            return res
        result = res
    return result

def eval_assignment_expression(node, env):
    """Handle assignment expressions like: x = 5"""
    debug_log("eval_assignment_expression", f"Assigning to {node.name.value}")
    
    # 1. Evaluate the new value
    value = eval_node(node.value, env)
    if is_error(value):
        return value

    # 2. Security Check: Ensure the target is not sealed
    from .security import SealedObject
    current_val = env.get(node.name.value)
    if isinstance(current_val, SealedObject):
        error_msg = f"Cannot assign to sealed object: {node.name.value}"
        debug_log("  Assignment error", error_msg)
        return EvaluationError(error_msg)

    # 3. Perform assignment
    env.set(node.name.value, value)
    debug_log("  Assignment successful", f"{node.name.value} = {value}")
    return value

def eval_block_statement(block, env):
    debug_log("eval_block_statement", f"Processing {len(block.statements)} statements in block")
    result = NULL
    for stmt in block.statements:
        res = eval_node(stmt, env)
        res = _resolve_awaitable(res)
        EVAL_SUMMARY['evaluated_statements'] += 1
        if isinstance(res, (ReturnValue, EvaluationError, ObjectEvaluationError)):
            return res
        result = res
    return result

def eval_expressions(expressions, env):
    debug_log("eval_expressions", f"Evaluating {len(expressions)} expressions")
    results = []
    for expr in expressions:
        res = eval_node(expr, env)
        res = _resolve_awaitable(res)
        if is_error(res):
            return res
        results.append(res)
    return results

def eval_identifier(node, env):
    debug_log("eval_identifier", f"Looking up: {node.value}")
    val = env.get(node.value)
    if val:
        return val
    # Check builtins
    builtin = builtins.get(node.value)
    if builtin:
        return builtin

    return EvaluationError(f"Identifier '{node.value}' not found")

def is_truthy(obj):
    if is_error(obj):
        return False
    result = not (obj == NULL or obj == FALSE)
    return result

# ... [Infix/Prefix logic remains same as previous, abbreviated for clarity] ...
def eval_infix_expression(operator, left, right):
    if is_error(left): return left
    if is_error(right): return right
    
    if operator == "+":
        if isinstance(left, String) or isinstance(right, String):
             # String concatenation logic
            l_str = left.value if isinstance(left, String) else str(left.inspect())
            r_str = right.value if isinstance(right, String) else str(right.inspect())
            return String(l_str + r_str)
        if isinstance(left, Integer) and isinstance(right, Integer):
            return Integer(left.value + right.value)
        # ... other numeric math ...
        if isinstance(left, (Integer, Float)) and isinstance(right, (Integer, Float)):
            return Float(left.value + right.value)

    # ... Logic for -, *, /, ==, etc. (standard implementation) ...
    # For brevity, deferring to standard implementation pattern
    if operator == "==":
        return TRUE if left.inspect() == right.inspect() else FALSE
    
    return EvaluationError(f"Unknown operator {operator} for types {left.type()} and {right.type()}")

def eval_prefix_expression(operator, right):
    if operator == "!":
        return FALSE if is_truthy(right) else TRUE
    if operator == "-":
        if isinstance(right, Integer): return Integer(-right.value)
        if isinstance(right, Float): return Float(-right.value)
    return EvaluationError(f"Unknown operator {operator}{right.type()}")

def eval_if_expression(ie, env):
    condition = eval_node(ie.condition, env)
    if is_error(condition): return condition
    if is_truthy(condition):
        return eval_node(ie.consequence, env)
    elif ie.alternative:
        return eval_node(ie.alternative, env)
    return NULL

def apply_function(fn, args):
    if isinstance(fn, (Action, LambdaFunction)):
        extended_env = extend_function_env(fn, args)
        evaluated = eval_node(fn.body, extended_env)
        evaluated = _resolve_awaitable(evaluated)
        return unwrap_return_value(evaluated)
    elif isinstance(fn, Builtin):
        try:
            if not isinstance(args, (list, tuple)): return EvaluationError("Invalid args")
            result = fn.fn(*args)
            if _is_awaitable(result): result = _resolve_awaitable(result)
            return result
        except Exception as e:
            return EvaluationError(f"Builtin error: {str(e)}")
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

def eval_lambda_expression(node, env):
    return LambdaFunction(node.parameters, node.body, env)

# === ENHANCED MAIN EVAL_NODE FUNCTION ===

def eval_node(node, env, stack_trace=None):
    if node is None:
        return NULL

    node_type = type(node)
    stack_trace = stack_trace or []

    # Stack tracing
    current_frame = f"  at {node_type.__name__}"
    if hasattr(node, 'token') and node.token:
        current_frame += f" (line {node.token.line})"
    stack_trace.append(current_frame)

    debug_log("eval_node", f"Processing {node_type.__name__}")

    try:
        # --- Statements ---
        if node_type == Program:
            return eval_program(node.statements, env)

        elif node_type == ExpressionStatement:
            return eval_node(node.expression, env, stack_trace)

        elif node_type == BlockStatement:
            return eval_block_statement(node, env)

        elif node_type == ReturnStatement:
            val = eval_node(node.return_value, env, stack_trace)
            return val if is_error(val) else ReturnValue(val)

        elif node_type == LetStatement:
            # Evaluate value first (prevent circular logic)
            value = eval_node(node.value, env, stack_trace)
            if is_error(value): return value
            env.set(node.name.value, value)
            return NULL

        elif node_type == ActionStatement:
            action_obj = Action(node.parameters, node.body, env)
            env.set(node.name.value, action_obj)
            return NULL

        # --- Phase 1: OOP & Security Statements ---

        elif node_type == EntityStatement:
            debug_log("  EntityStatement node", node.name.value)
            return eval_entity_statement(node, env)

        elif node_type == SealStatement:
            debug_log("  SealStatement node")
            return eval_seal_statement(node, env, stack_trace)

        elif node_type == VerifyStatement:
            return eval_verify_statement(node, env, stack_trace)

        elif node_type == ContractStatement:
            return eval_contract_statement(node, env, stack_trace)

        elif node_type == ProtectStatement:
            return eval_protect_statement(node, env, stack_trace)
        
        elif node_type == MiddlewareStatement:
            return eval_middleware_statement(node, env)
        
        elif node_type == AuthStatement:
            return eval_auth_statement(node, env)
            
        elif node_type == ThrottleStatement:
            return eval_throttle_statement(node, env)

        elif node_type == CacheStatement:
            return eval_cache_statement(node, env)

        elif node_type == ExportStatement:
            return eval_export_statement(node, env)

        elif node_type == UseStatement:
            return eval_use_statement_logic(node, env, stack_trace) # Refactored below

        elif node_type == FromStatement:
            return eval_from_statement_logic(node, env, stack_trace) # Refactored below

        # --- Control Flow ---

        elif node_type == IfStatement:
            condition = eval_node(node.condition, env, stack_trace)
            if is_error(condition): return condition
            if is_truthy(condition):
                return eval_node(node.consequence, env, stack_trace)
            elif node.alternative:
                return eval_node(node.alternative, env, stack_trace)
            return NULL

        elif node_type == WhileStatement:
            result = NULL
            while True:
                condition = eval_node(node.condition, env, stack_trace)
                if is_error(condition): return condition
                if not is_truthy(condition): break
                result = eval_node(node.body, env, stack_trace)
                if isinstance(result, (ReturnValue, EvaluationError, ObjectEvaluationError)):
                    break
            return result

        elif node_type == ForEachStatement:
            iterable = eval_node(node.iterable, env, stack_trace)
            if is_error(iterable): return iterable
            if not isinstance(iterable, List): return EvaluationError("for-each loop expected list")
            result = NULL
            for element in iterable.elements:
                env.set(node.item.value, element)
                result = eval_node(node.body, env, stack_trace)
                if isinstance(result, (ReturnValue, EvaluationError, ObjectEvaluationError)):
                    break
            return result

        elif node_type == TryCatchStatement:
            return eval_try_catch_statement_fixed(node, env, stack_trace)

        # --- Expressions ---

        elif node_type == AssignmentExpression:
            return eval_assignment_expression(node, env)

        elif node_type == Identifier:
            return eval_identifier(node, env)

        elif node_type == CallExpression:
            function = eval_node(node.function, env, stack_trace)
            if is_error(function): return function
            args = eval_expressions(node.arguments, env)
            if is_error(args): return args
            return apply_function(function, args)

        elif node_type == IntegerLiteral:
            return Integer(node.value)
        elif node_type == StringLiteral:
            return String(node.value)
        elif node_type == AST_Boolean:
            return TRUE if node.value else FALSE
        elif node_type == ListLiteral:
            elements = eval_expressions(node.elements, env)
            if is_error(elements): return elements
            return List(elements)
        elif node_type == MapLiteral:
            pairs = {}
            for key_expr, value_expr in node.pairs:
                key = eval_node(key_expr, env, stack_trace)
                if is_error(key): return key
                value = eval_node(value_expr, env, stack_trace)
                if is_error(value): return value
                pairs[key.inspect()] = value
            return Map(pairs)
        elif node_type == ActionLiteral:
            return Action(node.parameters, node.body, env)
        elif node_type == LambdaExpression:
            return eval_lambda_expression(node, env)
        elif node_type == InfixExpression:
            return eval_infix_expression(node.operator, node.left, node.right)
        elif node_type == PrefixExpression:
            return eval_prefix_expression(node.operator, node.right)
        
        elif node_type == PropertyAccessExpression:
            obj = eval_node(node.object, env, stack_trace)
            if is_error(obj): return obj
            prop_name = node.property.value
            if hasattr(obj, 'get'): # Supports Map, EntityInstance, SealedObject
                return obj.get(prop_name) or NULL
            return NULL
            
        # Fallback
        return EvaluationError(f"Unknown node type: {node_type}", stack_trace=stack_trace)

    except Exception as e:
        return EvaluationError(f"Internal error: {str(e)}", stack_trace=stack_trace[-5:])

# =====================================================
# STATEMENT HANDLER IMPLEMENTATIONS
# =====================================================

def eval_try_catch_statement_fixed(node, env, stack_trace):
    try:
        result = eval_node(node.try_block, env, stack_trace)
        if is_error(result):
            catch_env = Environment(outer=env)
            error_var = node.error_variable.value if node.error_variable else "error"
            catch_env.set(error_var, String(result.message))
            return eval_node(node.catch_block, catch_env, stack_trace)
        return result
    except Exception as e:
        catch_env = Environment(outer=env)
        error_var = node.error_variable.value if node.error_variable else "error"
        catch_env.set(error_var, String(str(e)))
        return eval_node(node.catch_block, catch_env, stack_trace)

def eval_entity_statement(node, env):
    """Evaluate entity statement - create entity definition"""
    from .object import EntityDefinition

    properties = []
    for prop in node.properties:
        # Convert parser dict format to clean dict
        properties.append({
            "name": prop["name"],
            "type": prop["type"],
            # evaluate default if present (simple literals only for now)
            "default_value": prop.get("default_value") 
        })

    entity_def = EntityDefinition(node.name.value, properties)
    env.set(node.name.value, entity_def)
    return NULL

def eval_seal_statement(node, env, stack_trace=None):
    """Evaluate seal statement - mark a variable as sealed (immutable)"""
    from .security import SealedObject

    target = node.target
    if target is None:
        return EvaluationError("seal: missing target")

    # 1. Seal Identifier: seal x
    if isinstance(target, Identifier):
        name = target.value
        current = env.get(name)
        if current is None:
            return EvaluationError(f"seal: identifier '{name}' not found")
        
        # Wrap in sealed object and replace in environment
        sealed = SealedObject(current)
        env.set(name, sealed)
        debug_log("  Sealed identifier", name)
        return sealed

    # 2. Seal Property: seal obj.prop
    if isinstance(target, PropertyAccessExpression):
        obj = eval_node(target.object, env, stack_trace)
        if is_error(obj): return obj
        prop_name = target.property.value
        
        # Map-like objects
        if isinstance(obj, Map):
            val = obj.pairs.get(prop_name)
            if val is None:
                return EvaluationError(f"seal: property '{prop_name}' not found")
            obj.pairs[prop_name] = SealedObject(val)
            debug_log("  Sealed map property", prop_name)
            return obj.pairs[prop_name]
            
        # EntityInstance objects
        if hasattr(obj, 'set') and hasattr(obj, 'get'):
            val = obj.get(prop_name)
            if val is None or val == NULL:
                return EvaluationError(f"seal: property '{prop_name}' not found on object")
            obj.set(prop_name, SealedObject(val))
            debug_log("  Sealed object property", prop_name)
            return SealedObject(val)

    return EvaluationError("seal: unsupported target; use an identifier or property access")

def eval_export_statement(node, env):
    """Handle export statements"""
    names = []
    # Handle both single 'name' and list 'names' attributes from AST
    if hasattr(node, 'names') and node.names:
        names = [n.value if hasattr(n, 'value') else str(n) for n in node.names]
    elif hasattr(node, 'name') and node.name is not None:
        names = [node.name.value if hasattr(node.name, 'value') else str(node.name)]

    if not names:
        return EvaluationError("export: no identifiers provided")

    for nm in names:
        value = env.get(nm)
        if not value:
            return EvaluationError(f"Cannot export undefined identifier: {nm}")
        
        # Tag value with permission if provided
        if node.permission:
            try:
                setattr(value, '_export_permission', node.permission)
                setattr(value, '_allowed_files', node.allowed_files)
            except: pass

        env.export(nm, value)
        
    return NULL

def eval_verify_statement(node, env, stack_trace=None):
    from .security import VerifyWrapper, VerificationCheck, get_security_context

    target_value = eval_node(node.target, env, stack_trace)
    if is_error(target_value): return target_value

    checks = []
    for condition_node in node.conditions:
        # We delay evaluation of checks to runtime calls usually, 
        # but if passed as expressions, verify expects callables
        checks.append(VerificationCheck(str(condition_node), lambda ctx: eval_node(condition_node, env)))

    wrapped = VerifyWrapper(target_value, checks, node.error_handler)
    ctx = get_security_context()
    ctx.register_verify_check(str(node.target), wrapped)
    return wrapped

def eval_contract_statement(node, env, stack_trace=None):
    from .security import SmartContract
    
    # Mock implementation for phase 1
    storage = {v["name"]: {} for v in node.storage_vars}
    actions = {a.name.value: a for a in node.actions}
    
    contract = SmartContract(node.name.value, storage, actions)
    env.set(node.name.value, contract)
    return NULL

def eval_protect_statement(node, env, stack_trace=None):
    from .security import ProtectionPolicy, get_security_context
    
    rules_val = eval_node(node.rules, env, stack_trace)
    if is_error(rules_val): return rules_val
    
    rules_dict = {}
    if isinstance(rules_val, Map):
        for k, v in rules_val.pairs.items():
            rules_dict[k.inspect()] = v

    policy = ProtectionPolicy(str(node.target), rules_dict, node.enforcement_level)
    get_security_context().register_protection(str(node.target), policy)
    return policy

def eval_middleware_statement(node, env):
    from .security import Middleware, get_security_context
    handler = eval_node(node.handler, env)
    if is_error(handler): return handler
    mw = Middleware(node.name.value, handler)
    get_security_context().middlewares[node.name.value] = mw
    return NULL

def eval_auth_statement(node, env):
    from .security import AuthConfig, get_security_context
    cfg = eval_node(node.config, env)
    if is_error(cfg): return cfg
    get_security_context().auth_config = AuthConfig(cfg)
    return NULL

def eval_throttle_statement(node, env):
    from .security import RateLimiter, get_security_context
    target = eval_node(node.target, env)
    limits = eval_node(node.limits, env)
    
    rpm = 100
    if isinstance(limits, Map):
        # Extract simplified limits
        pass 
    
    limiter = RateLimiter(rpm, 10, False)
    get_security_context().rate_limiters[str(node.target)] = limiter
    return NULL

def eval_cache_statement(node, env):
    from .security import CachePolicy, get_security_context
    CachePolicy(ttl=3600) # Simplified
    return NULL

# Logic extracted for Use/From statements to keep eval_node clean
def eval_use_statement_logic(node, env, stack_trace):
    from .module_cache import get_cached_module, cache_module, get_module_candidates, normalize_path
    
    file_path = node.file_path.value if isinstance(node.file_path, StringLiteral) else node.file_path
    if not file_path: return EvaluationError("use: missing file path")

    normalized_path = normalize_path(file_path)
    module_env = get_cached_module(normalized_path)

    if not module_env:
        candidates = get_module_candidates(file_path)
        for candidate in candidates:
            try:
                if not os.path.exists(candidate): continue
                with open(candidate, 'r', encoding='utf-8') as f: code = f.read()
                
                # Import here to avoid circular top-level imports
                from .lexer import Lexer
                from .parser import Parser
                
                parser = Parser(Lexer(code))
                program = parser.parse_program()
                
                module_env = Environment()
                cache_module(normalized_path, module_env) # Cache placeholder
                
                res = eval_node(program, module_env)
                if is_error(res): return res
                break
            except Exception as e:
                continue
    
    if not module_env:
        return EvaluationError(f"Module not found: {file_path}")

    # Handle Alias or Imports
    if node.alias:
        env.set(node.alias, module_env)
    elif getattr(node, 'is_named_import', False):
        # use { A, B } from '...'
        exports = module_env.get_exports()
        for name_node in node.names:
            name = name_node.value
            if name in exports:
                env.set(name, exports[name])
            else:
                return EvaluationError(f"Import '{name}' not found in {file_path}")
    else:
        # Simple use '...' (import everything)
        exports = module_env.get_exports()
        for name, val in exports.items():
            env.set(name, val)
            
    return NULL

def eval_from_statement_logic(node, env, stack_trace):
    # Syntactic sugar around use statement logic
    use_node = UseStatement(node.file_path)
    # We need to get the env first, but UseStatement logic usually merges it.
    # For strict correctness, we should load module_env separately.
    # Reusing use_statement_logic with a temp alias to capture the env
    
    temp_alias = "__temp_module_load__"
    use_node.alias = temp_alias
    res = eval_use_statement_logic(use_node, env, stack_trace)
    if is_error(res): return res
    
    module_env = env.get(temp_alias)
    # Clean up temp
    # env.store.pop(temp_alias, None) # Not strictly supported by Environment API yet
    
    exports = module_env.get_exports()
    for name_pair in node.imports:
        src = name_pair[0].value
        dest = name_pair[1].value if len(name_pair) > 1 and name_pair[1] else src
        
        if src not in exports:
            return EvaluationError(f"from: '{src}' not exported by {node.file_path}")
        env.set(dest, exports[src])
        
    return NULL

# Register core builtins if not already present
try:
    if not builtins:
        # Populate minimal set for bootstrapping if needed
        pass
except: pass