# src/zexus/evaluator/statements.py
import os
import sys

from .. import zexus_ast
from ..zexus_ast import (
    Program, ExpressionStatement, BlockStatement, ReturnStatement, LetStatement, ConstStatement,
    ActionStatement, IfStatement, WhileStatement, ForEachStatement,
    TryCatchStatement, UseStatement, FromStatement, ExportStatement,
    ContractStatement, EntityStatement, VerifyStatement, ProtectStatement,
    SealStatement, MiddlewareStatement, AuthStatement, ThrottleStatement, CacheStatement,
    ComponentStatement, ThemeStatement, DebugStatement, ExternalDeclaration, AssignmentExpression,
    PrintStatement, ScreenStatement, EmbeddedCodeStatement, ExactlyStatement,
    Identifier, PropertyAccessExpression
)
from ..object import (
    Environment, Integer, String, Boolean as BooleanObj, ReturnValue,
    Action, List, Map, EvaluationError, EntityDefinition, EmbeddedCode, Builtin
)
from ..security import (
    SealedObject, SmartContract, VerifyWrapper, VerificationCheck, get_security_context,
    ProtectionPolicy, Middleware, AuthConfig, RateLimiter, CachePolicy
)
from .utils import is_error, debug_log, EVAL_SUMMARY, NULL, TRUE, FALSE, _resolve_awaitable, _zexus_to_python, _python_to_zexus, is_truthy

class StatementEvaluatorMixin:
    """Handles evaluation of statements, flow control, module loading, and security features."""
    
    def eval_program(self, statements, env):
        debug_log("eval_program", f"Processing {len(statements)} statements")
        
        try:
            EVAL_SUMMARY['parsed_statements'] = max(EVAL_SUMMARY.get('parsed_statements', 0), len(statements))
        except Exception: 
            pass
        
        result = NULL
        for i, stmt in enumerate(statements):
            debug_log(f"  Statement {i+1}", type(stmt).__name__)
            res = self.eval_node(stmt, env)
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
        
        debug_log("eval_program completed", result)
        return result
    
    def eval_block_statement(self, block, env):
        debug_log("eval_block_statement", f"len={len(block.statements)}")
        
        try:
            EVAL_SUMMARY['max_statements_in_block'] = max(EVAL_SUMMARY.get('max_statements_in_block', 0), len(block.statements))
        except Exception:
            pass
        
        result = NULL
        for stmt in block.statements:
            res = self.eval_node(stmt, env)
            res = _resolve_awaitable(res)
            EVAL_SUMMARY['evaluated_statements'] += 1
            
            if isinstance(res, (ReturnValue, EvaluationError)):
                debug_log("  Block interrupted", res)
                if is_error(res):
                    try:
                        EVAL_SUMMARY['errors'] += 1
                    except Exception:
                        pass
                return res
            result = res
        
        debug_log("  Block completed", result)
        return result
    
    def eval_expression_statement(self, node, env, stack_trace):
        return self.eval_node(node.expression, env, stack_trace)
    
    # === VARIABLE & CONTROL FLOW ===
    
    def eval_let_statement(self, node, env, stack_trace):
        debug_log("eval_let_statement", f"let {node.name.value}")
        
        # FIXED: Evaluate value FIRST to prevent recursion issues
        value = self.eval_node(node.value, env, stack_trace)
        if is_error(value): 
            return value
        
        env.set(node.name.value, value)
        return NULL
    
    def eval_const_statement(self, node, env, stack_trace):
        debug_log("eval_const_statement", f"const {node.name.value}")
        
        # Evaluate value FIRST
        value = self.eval_node(node.value, env, stack_trace)
        if is_error(value): 
            return value
        
        # Set as const in environment
        env.set_const(node.name.value, value)
        return NULL
    
    def eval_return_statement(self, node, env, stack_trace):
        val = self.eval_node(node.return_value, env, stack_trace)
        if is_error(val): 
            return val
        return ReturnValue(val)
    
    def eval_assignment_expression(self, node, env, stack_trace):
        debug_log("eval_assignment_expression", f"Assigning to {node.name.value}")
        
        target_obj = env.get(node.name.value)
        if isinstance(target_obj, SealedObject):
            return EvaluationError(f"Cannot assign to sealed object: {node.name.value}")
        
        value = self.eval_node(node.value, env, stack_trace)
        if is_error(value): 
            return value
        
        env.set(node.name.value, value)
        return value
    
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
            if isinstance(result, (ReturnValue, EvaluationError)):
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
            if isinstance(result, (ReturnValue, EvaluationError)):
                return result
        
        return result
    
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
        
        # 1. Determine File Path
        file_path_attr = getattr(node, 'file_path', None) or getattr(node, 'embedded_ref', None)
        file_path = file_path_attr.value if hasattr(file_path_attr, 'value') else file_path_attr
        if not file_path: 
            return EvaluationError("use: missing file path")
        
        debug_log("  UseStatement loading", file_path)
        normalized_path = normalize_path(file_path)
        
        # 2. Check Cache
        module_env = get_cached_module(normalized_path)
        
        # 3. Load if not cached
        if not module_env:
            candidates = get_module_candidates(file_path)
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
        alias = getattr(node, 'alias', None)
        if alias:
            env.set(alias, module_env)
        else:
            # Import all exports into current scope
            try:
                exports = module_env.get_exports()
                importer_file = env.get("__file__").value if env.get("__file__") else None
                
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
            candidates = get_module_candidates(file_path)
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
        importer_file = env.get("__file__").value if env.get("__file__") else None
        
        for name_pair in node.imports:
            # name_pair is [source_name, dest_name] (dest_name optional)
            src = name_pair[0].value if hasattr(name_pair[0], 'value') else str(name_pair[0])
            dest = name_pair[1].value if len(name_pair) > 1 and name_pair[1] else src
            
            # Retrieve from module exports
            exports = module_env.get_exports() if hasattr(module_env, 'get_exports') else {}
            val = exports.get(src)
            
            if val is None:
                # Fallback: check if it's in the environment directly
                val = module_env.get(src)
            
            if val is None:
                return EvaluationError(f"From import: '{src}' not found in {file_path}")
            
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
        
        for nm in names:
            val = env.get(nm)
            if not val: 
                return EvaluationError(f"Cannot export undefined: {nm}")
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
            
            prop_key = target_node.property.value  # Assuming Identifier
            
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
        
        # Create audit log entry as a Map object
        audit_log_pairs = {
            "data_name": String(data_name),
            "action": String(action_type),
            "timestamp": String(timestamp),
            "data_type": String(type(audited_data).__name__),
        }
        
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
        
        # Return restriction entry
        return Map({
            "target": String(f"{obj_name}.{field_name}"),
            "field": String(field_name),
            "restriction": String(restriction),
            "status": String("applied"),
            "timestamp": String(datetime.now(timezone.utc).isoformat())
        })
    
    def eval_sandbox_statement(self, node, env, stack_trace):
        """Evaluate sandbox statement for isolated execution environments.
        
        Syntax: sandbox { code }
        
        Creates a new isolated environment and executes code within it.
        """
        # Create isolated environment (child of current)
        sandbox_env = Environment(parent=env)
        
        # Execute body in sandbox
        if node.body is None:
            return NULL
        
        result = self.eval_node(node.body, sandbox_env, stack_trace)
        
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
        
        # Create trail configuration entry
        trail_config = {
            "type": String(trail_type),
            "filter": String(filter_key) if filter_key else String("*"),
            "enabled": String("true"),
            "timestamp": String(datetime.now(timezone.utc).isoformat())
        }
        
        return Map(trail_config)
    
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
        env.set(node.name.value, contract)
        return NULL
    
    def eval_entity_statement(self, node, env, stack_trace):
        props = {}
        for prop in node.properties:
            p_name = prop.name.value
            p_type = prop.type.value
            def_val = NULL
            
            if getattr(prop, 'default_value', None):
                def_val = self.eval_node(prop.default_value, env, stack_trace)
                if is_error(def_val): 
                    return def_val
            
            props[p_name] = {"type": p_type, "default_value": def_val}
        
        entity = EntityDefinition(node.name.value, props)
        env.set(node.name.value, entity)
        return NULL
    
    def eval_verify_statement(self, node, env, stack_trace):
        target = self.eval_node(node.target, env, stack_trace)
        if is_error(target): 
            return target
        
        checks = []
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
    
    def eval_protect_statement(self, node, env, stack_trace):
        target = self.eval_node(node.target, env, stack_trace)
        if is_error(target): 
            return target
        
        rules_val = self.eval_node(node.rules, env, stack_trace)
        if is_error(rules_val): 
            return rules_val
        
        rules_dict = {}
        if isinstance(rules_val, Map):
            for k, v in rules_val.pairs.items():
                rules_dict[k.value if isinstance(k, String) else str(k)] = v
        
        policy = ProtectionPolicy(str(node.target), rules_dict, node.enforcement_level)
        get_security_context().register_protection(str(node.target), policy)
        return policy
    
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
        target = self.eval_node(node.target, env)
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
                    per_user = True if (isinstance(v, BooleanObj) and v.value) else False
        
        limiter = RateLimiter(rpm, burst, per_user)
        ctx = get_security_context()
        if not hasattr(ctx, 'rate_limiters'): 
            ctx.rate_limiters = {}
        ctx.rate_limiters[str(node.target)] = limiter
        return NULL
    
    def eval_cache_statement(self, node, env, stack_trace):
        target = self.eval_node(node.target, env)
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
        val = self.eval_node(node.value, env, stack_trace)
        if is_error(val):
            print(f"❌ Error: {val}", file=sys.stderr)
            return NULL
        
        print(val.inspect())
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
        val = self.eval_node(node.value, env, stack_trace)
        if is_error(val): 
            return val
        
        from ..object import Debug
        Debug.log(String(str(val)))
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
        env.set(node.name.value, action)
        return NULL
