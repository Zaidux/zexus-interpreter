# src/zexus/evaluator/core.py
import traceback
import asyncio
from .. import zexus_ast
from ..object import Environment, EvaluationError, Null, Boolean as BooleanObj, Map, EmbeddedCode, List, Action, LambdaFunction, String
from .utils import is_error, debug_log, EVAL_SUMMARY, NULL
from .expressions import ExpressionEvaluatorMixin
from .statements import StatementEvaluatorMixin
from .functions import FunctionEvaluatorMixin
from .integration import EvaluationContext, get_integration

# Import VM and bytecode compiler
try:
    from ..vm.vm import VM
    from .bytecode_compiler import EvaluatorBytecodeCompiler, should_use_vm_for_node
    VM_AVAILABLE = True
except ImportError as e:
    VM_AVAILABLE = False
    VM = None
    EvaluatorBytecodeCompiler = None
    should_use_vm_for_node = lambda node: False
    print(f"⚠️  VM not available in evaluator: {e}")

class Evaluator(ExpressionEvaluatorMixin, StatementEvaluatorMixin, FunctionEvaluatorMixin):
    def __init__(self, trusted: bool = False, use_vm: bool = True):
        # Initialize mixins (FunctionEvaluatorMixin sets up builtins)
        FunctionEvaluatorMixin.__init__(self)
        
        # Initialize 10-phase integration
        self.integration_context = EvaluationContext("evaluator")
        
        # Setup security context
        if trusted:
            self.integration_context.setup_for_trusted_code()
        else:
            self.integration_context.setup_for_untrusted_code()
        
        # VM integration
        self.use_vm = use_vm and VM_AVAILABLE
        self.vm_instance = None
        self.bytecode_compiler = None
        
        # VM execution statistics
        self.vm_stats = {
            'bytecode_compiles': 0,
            'vm_executions': 0,
            'vm_fallbacks': 0,
            'direct_evals': 0
        }
        
        # CONTINUE keyword support - error recovery mode
        self.continue_on_error = False
        self.error_log = []  # Store errors when continue_on_error is enabled
        
        if self.use_vm and VM_AVAILABLE:
            self._initialize_vm()
    
    def eval_node(self, node, env, stack_trace=None):
        if node is None: 
            debug_log("eval_node", "Node is None, returning NULL")
            return NULL
        
        stack_trace = stack_trace or []
        node_type = type(node)
        
        # Add to stack trace for better error reporting
        current_frame = f"  at {node_type.__name__}"
        if hasattr(node, 'token') and node.token:
            current_frame += f" (line {node.token.line})"
        stack_trace.append(current_frame)
        
        debug_log("eval_node", f"Processing {node_type.__name__}")
        
        try:
            # === STATEMENTS ===
            if isinstance(node, zexus_ast.Program):
                debug_log("  Program node", f"{len(node.statements)} statements")
                return self.ceval_program(node.statements, env)
            
            elif isinstance(node, zexus_ast.ExpressionStatement):
                debug_log("  ExpressionStatement node")
                return self.eval_node(node.expression, env, stack_trace)
            
            elif isinstance(node, zexus_ast.BlockStatement):
                debug_log("  BlockStatement node", f"{len(node.statements)} statements")
                return self.eval_block_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ReturnStatement):
                debug_log("  ReturnStatement node")
                return self.eval_return_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ContinueStatement):
                debug_log("  ContinueStatement node")
                return self.eval_continue_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.BreakStatement):
                debug_log("  BreakStatement node")
                return self.eval_break_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.LetStatement):
                return self.eval_let_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ConstStatement):
                return self.eval_const_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.DataStatement):
                return self.eval_data_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.AssignmentExpression):
                debug_log("  AssignmentExpression node")
                return self.eval_assignment_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.TryCatchStatement):
                return self.eval_try_catch_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ThrowStatement):
                debug_log("  ThrowStatement node")
                return self.eval_throw_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.IfStatement):
                debug_log("  IfStatement node")
                return self.eval_if_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.WhileStatement):
                debug_log("  WhileStatement node")
                return self.eval_while_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ForEachStatement):
                debug_log("  ForEachStatement node", f"for each {node.item.value}")
                return self.eval_foreach_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.WatchStatement):
                debug_log("  WatchStatement node")
                return self.eval_watch_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.SealStatement):
                return self.eval_seal_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.RestrictStatement):
                return self.eval_restrict_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.SandboxStatement):
                return self.eval_sandbox_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.TrailStatement):
                return self.eval_trail_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.TxStatement):
                return self.eval_tx_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.EntityStatement):
                return self.eval_entity_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ContractStatement):
                return self.eval_contract_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ExportStatement):
                return self.eval_export_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.UseStatement):
                debug_log("  UseStatement node", node.file_path)
                return self.eval_use_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.FromStatement):
                debug_log("  FromStatement node", node.file_path)
                return self.eval_from_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.VerifyStatement):
                return self.eval_verify_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ProtectStatement):
                return self.eval_protect_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.MiddlewareStatement):
                return self.eval_middleware_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.AuthStatement):
                return self.eval_auth_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ThrottleStatement):
                return self.eval_throttle_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.CacheStatement):
                return self.eval_cache_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.PrintStatement):
                debug_log("  PrintStatement node")
                return self.eval_print_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ScreenStatement):
                debug_log("  ScreenStatement node", node.name.value)
                return self.eval_screen_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.EmbeddedCodeStatement):
                debug_log("  EmbeddedCodeStatement node", node.name.value)
                return self.eval_embedded_code_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ComponentStatement):
                debug_log("  ComponentStatement node", node.name.value)
                return self.eval_component_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ThemeStatement):
                debug_log("  ThemeStatement node", node.name.value)
                return self.eval_theme_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.DebugStatement):
                debug_log("  DebugStatement node")
                return self.eval_debug_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ExternalDeclaration):
                debug_log("  ExternalDeclaration node", node.name.value)
                return self.eval_external_declaration(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ExactlyStatement):
                debug_log("  ExactlyStatement node")
                return self.eval_exactly_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ActionStatement):
                debug_log("  ActionStatement node", f"action {node.name.value}")
                return self.eval_action_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.FunctionStatement):
                debug_log("  FunctionStatement node", f"function {node.name.value}")
                return self.eval_function_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.NativeStatement):
                debug_log("  NativeStatement node", f"native {node.function_name}")
                return self.eval_native_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.GCStatement):
                debug_log("  GCStatement node", f"gc {node.action}")
                return self.eval_gc_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.InlineStatement):
                debug_log("  InlineStatement node", f"inline {node.function_name}")
                return self.eval_inline_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.BufferStatement):
                debug_log("  BufferStatement node", f"buffer {node.buffer_name}")
                return self.eval_buffer_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.SIMDStatement):
                debug_log("  SIMDStatement node")
                return self.eval_simd_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.DeferStatement):
                debug_log("  DeferStatement node")
                return self.eval_defer_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.PatternStatement):
                debug_log("  PatternStatement node")
                return self.eval_pattern_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.EnumStatement):
                debug_log("  EnumStatement node", f"enum {node.name}")
                return self.eval_enum_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.StreamStatement):
                debug_log("  StreamStatement node", f"stream {node.stream_name}")
                return self.eval_stream_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.WatchStatement):
                debug_log("  WatchStatement node")
                return self.eval_watch_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.LogStatement):
                debug_log("  LogStatement node")
                return self.eval_log_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ImportLogStatement):
                debug_log("  ImportLogStatement node", "log <<")
                return self.eval_import_log_statement(node, env, stack_trace)
            
            # === NEW SECURITY STATEMENTS ===
            elif isinstance(node, zexus_ast.CapabilityStatement):
                debug_log("  CapabilityStatement node", f"capability {node.name}")
                return self.eval_capability_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.GrantStatement):
                debug_log("  GrantStatement node", f"grant {node.entity_name}")
                return self.eval_grant_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.RevokeStatement):
                debug_log("  RevokeStatement node", f"revoke {node.entity_name}")
                return self.eval_revoke_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ValidateStatement):
                debug_log("  ValidateStatement node")
                return self.eval_validate_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.SanitizeStatement):
                debug_log("  SanitizeStatement node")
                return self.eval_sanitize_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.InjectStatement):
                debug_log("  InjectStatement node")
                return self.eval_inject_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ImmutableStatement):
                debug_log("  ImmutableStatement node", f"immutable {node.target}")
                return self.eval_immutable_statement(node, env, stack_trace)
            
            # === COMPLEXITY & LARGE PROJECT MANAGEMENT STATEMENTS ===
            elif isinstance(node, zexus_ast.InterfaceStatement):
                debug_log("  InterfaceStatement node", f"interface {node.name.value}")
                return self.eval_interface_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.TypeAliasStatement):
                debug_log("  TypeAliasStatement node", f"type_alias {node.name.value}")
                return self.eval_type_alias_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ModuleStatement):
                debug_log("  ModuleStatement node", f"module {node.name.value}")
                return self.eval_module_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.PackageStatement):
                debug_log("  PackageStatement node", f"package {node.name.value}")
                return self.eval_package_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.UsingStatement):
                debug_log("  UsingStatement node", f"using {node.resource_name.value}")
                return self.eval_using_statement(node, env, stack_trace)
            
            # === EXPRESSIONS ===
            # === CONCURRENCY & PERFORMANCE STATEMENTS ===
            elif isinstance(node, zexus_ast.ChannelStatement):
                debug_log("  ChannelStatement node", f"channel {node.name.value}")
                return self.eval_channel_statement(node, env, stack_trace)

            elif isinstance(node, zexus_ast.SendStatement):
                debug_log("  SendStatement node", "send to channel")
                return self.eval_send_statement(node, env, stack_trace)

            elif isinstance(node, zexus_ast.ReceiveStatement):
                debug_log("  ReceiveStatement node", "receive from channel")
                return self.eval_receive_statement(node, env, stack_trace)

            elif isinstance(node, zexus_ast.AtomicStatement):
                debug_log("  AtomicStatement node", "atomic operation")
                return self.eval_atomic_statement(node, env, stack_trace)

            # === BLOCKCHAIN STATEMENTS ===
            elif isinstance(node, zexus_ast.LedgerStatement):
                debug_log("  LedgerStatement node", f"ledger {node.name.value}")
                return self.eval_ledger_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.StateStatement):
                debug_log("  StateStatement node", f"state {node.name.value}")
                return self.eval_state_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.RequireStatement):
                debug_log("  RequireStatement node", "require condition")
                return self.eval_require_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.RevertStatement):
                debug_log("  RevertStatement node", "revert transaction")
                return self.eval_revert_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.LimitStatement):
                debug_log("  LimitStatement node", "set gas limit")
                return self.eval_limit_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ProtocolStatement):
                debug_log("  ProtocolStatement node", f"protocol {node.name}")
                return self.eval_protocol_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.PersistentStatement):
                debug_log("  PersistentStatement node", f"persistent {node.name}")
                return self.eval_persistent_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.EmitStatement):
                debug_log("  EmitStatement node", f"emit {node.event_name}")
                return self.eval_emit_statement(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ModifierDeclaration):
                debug_log("  ModifierDeclaration node", f"modifier {node.name}")
                return self.eval_modifier_declaration(node, env, stack_trace)

            # === EXPRESSIONS ===
            elif isinstance(node, zexus_ast.Identifier):
                debug_log("  Identifier node", node.value)
                return self.eval_identifier(node, env)
            
            elif isinstance(node, zexus_ast.IntegerLiteral):
                debug_log("  IntegerLiteral node", node.value)
                from ..object import Integer
                return Integer(node.value)
            
            elif node_type == zexus_ast.FloatLiteral or isinstance(node, zexus_ast.FloatLiteral):
                debug_log("  FloatLiteral node", getattr(node, 'value', 'unknown'))
                from ..object import Float
                try:
                    val = getattr(node, 'value', None)
                    return Float(val)
                except Exception:
                    return EvaluationError(f"Invalid float literal: {getattr(node, 'value', None)}")
            
            elif isinstance(node, zexus_ast.StringLiteral):
                debug_log("  StringLiteral node", node.value)
                from ..object import String
                # Process escape sequences in the string
                value = node.value
                value = value.replace('\\n', '\n')
                value = value.replace('\\t', '\t')
                value = value.replace('\\r', '\r')
                value = value.replace('\\\\', '\\')
                value = value.replace('\\"', '"')
                value = value.replace("\\'", "'")
                return String(value)
            
            elif isinstance(node, zexus_ast.Boolean):
                debug_log("  Boolean node", f"value: {node.value}")
                from ..object import Boolean
                return Boolean(node.value)
            
            elif isinstance(node, zexus_ast.NullLiteral):
                debug_log("  NullLiteral node")
                return NULL
            
            elif isinstance(node, zexus_ast.ThisExpression):
                debug_log("  ThisExpression node")
                return self.eval_this_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.InfixExpression):
                debug_log("  InfixExpression node", f"{node.left} {node.operator} {node.right}")
                return self.eval_infix_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.PrefixExpression):
                debug_log("  PrefixExpression node", f"{node.operator} {node.right}")
                return self.eval_prefix_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.IfExpression):
                debug_log("  IfExpression node")
                return self.eval_if_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.TernaryExpression):
                debug_log("  TernaryExpression node")
                return self.eval_ternary_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.NullishExpression):
                debug_log("  NullishExpression node")
                return self.eval_nullish_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.AwaitExpression):
                debug_log("  AwaitExpression node")
                return self.eval_await_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.FileImportExpression):
                debug_log("  FileImportExpression node", "<< file import")
                return self.eval_file_import_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.MethodCallExpression):
                debug_log("  MethodCallExpression node", f"{node.object}.{node.method}")
                return self.eval_method_call_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.CallExpression):
                debug_log("🚀 CallExpression node", f"Calling {node.function}")
                return self.eval_call_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.AsyncExpression):
                debug_log("⚡ AsyncExpression node", f"Async execution of {node.expression}")
                return self.eval_async_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.MatchExpression):
                debug_log("🎯 MatchExpression node", "Pattern matching")
                return self.eval_match_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.ListLiteral):
                debug_log("  ListLiteral node", f"{len(node.elements)} elements")
                elems = self.eval_expressions(node.elements, env)
                if is_error(elems):
                    return elems
                return List(elems)
            
            elif isinstance(node, zexus_ast.MapLiteral):
                debug_log("  MapLiteral node", f"{len(node.pairs)} pairs")
                pairs = {}
                for k, v in node.pairs:
                    # If the key is a bare identifier (e.g. io.read) treat it as a string key
                    if isinstance(k, zexus_ast.Identifier):
                        key_str = k.value
                    else:
                        key = self.eval_node(k, env, stack_trace)
                        if is_error(key):
                            return key
                        key_str = key.inspect()

                    val = self.eval_node(v, env, stack_trace)
                    if is_error(val):
                        return val
                    pairs[key_str] = val
                return Map(pairs)
            
            elif isinstance(node, zexus_ast.ActionLiteral):
                debug_log("  ActionLiteral node")
                return Action(node.parameters, node.body, env)
            
            elif isinstance(node, zexus_ast.LambdaExpression):
                debug_log("  LambdaExpression node")
                return LambdaFunction(node.parameters, node.body, env)
            
            elif isinstance(node, zexus_ast.EmbeddedLiteral):
                debug_log("  EmbeddedLiteral node")
                return EmbeddedCode("embedded_block", node.language, node.code)
            
            elif isinstance(node, zexus_ast.PropertyAccessExpression):
                debug_log("  PropertyAccessExpression node", f"{node.object}.{node.property}")
                obj = self.eval_node(node.object, env, stack_trace)
                if is_error(obj): 
                    return obj
                
                # Safely extract property name - property can be Identifier, IntegerLiteral, or other expression
                if hasattr(node.property, 'value'):
                    property_name = node.property.value
                elif isinstance(node.property, zexus_ast.PropertyAccessExpression):
                    # Nested property access - evaluate it
                    prop_result = self.eval_node(node.property, env, stack_trace)
                    if is_error(prop_result):
                        return prop_result
                    property_name = prop_result.inspect() if hasattr(prop_result, 'inspect') else str(prop_result)
                else:
                    # Evaluate the property expression to get the key
                    prop_result = self.eval_node(node.property, env, stack_trace)
                    if is_error(prop_result):
                        return prop_result
                    property_name = prop_result.value if hasattr(prop_result, 'value') else str(prop_result)
                
                if isinstance(obj, EmbeddedCode):
                    if property_name == "code":
                        return String(obj.code)
                    elif property_name == "language":
                        return String(obj.language)
                
                # Enforcement: consult security restrictions before returning
                try:
                    from ..security import get_security_context
                    ctx = get_security_context()
                    target = f"{getattr(node.object, 'value', str(node.object))}.{property_name}"
                    restriction = ctx.get_restriction(target)
                except Exception:
                    restriction = None

                # Handle Builtin objects (for static methods like TypeName.default())
                from ..object import Builtin
                if isinstance(obj, Builtin):
                    if hasattr(obj, 'static_methods') and property_name in obj.static_methods:
                        return obj.static_methods[property_name]
                    return NULL

                # Handle Module objects
                from ..complexity_system import Module
                if isinstance(obj, Module):
                    val = obj.get(property_name)
                    if val is None:
                        return NULL
                    if restriction:
                        rule = restriction.get('restriction')
                        if rule == 'redact':
                            from ..object import String
                            return String('***REDACTED***')
                        if rule == 'admin-only':
                            is_admin = bool(env.get('__is_admin__')) if env and hasattr(env, 'get') else False
                            if not is_admin:
                                return EvaluationError('Access denied: admin required')
                    return val

                # Handle Map objects
                if isinstance(obj, Map):
                    from ..object import String
                    
                    # Try string key first
                    val = obj.pairs.get(property_name, NULL)
                    if val == NULL:
                        # Try with String object key (for dataclasses and other String-keyed maps)
                        str_key = String(property_name)
                        val = obj.pairs.get(str_key, NULL)
                    
                    # Check if this is a computed property
                    computed_props = obj.pairs.get(String("__computed__"))
                    if computed_props and isinstance(computed_props, dict) and property_name in computed_props:
                        # Evaluate computed property
                        computed_expr = computed_props[property_name]
                        
                        # Create environment with all field values in scope
                        from ..environment import Environment
                        compute_env = Environment(outer=env)
                        compute_env.set('this', obj)
                        
                        # Add all regular fields to environment
                        for key, value in obj.pairs.items():
                            if isinstance(key, String) and not key.value.startswith('__'):
                                compute_env.set(key.value, value)
                        
                        # Evaluate the computed expression
                        result = self.eval_node(computed_expr, compute_env, stack_trace)
                        return result if not is_error(result) else NULL
                    
                    # apply restriction if present
                    if restriction:
                        rule = restriction.get('restriction')
                        if rule == 'redact':
                            from ..object import String
                            return String('***REDACTED***')
                        if rule == 'admin-only':
                            # check environment flag for admin
                            is_admin = bool(env.get('__is_admin__')) if env and hasattr(env, 'get') else False
                            if not is_admin:
                                return EvaluationError('Access denied: admin required')
                    return val

                if hasattr(obj, 'get') and callable(obj.get):
                    val = obj.get(property_name)
                    if restriction:
                        rule = restriction.get('restriction')
                        if rule == 'redact':
                            from ..object import String
                            return String('***REDACTED***')
                        if rule == 'admin-only':
                            is_admin = bool(env.get('__is_admin__')) if env and hasattr(env, 'get') else False
                            if not is_admin:
                                return EvaluationError('Access denied: admin required')
                    return val

                return NULL
            
            # === BLOCKCHAIN EXPRESSIONS ===
            elif isinstance(node, zexus_ast.TXExpression):
                debug_log("  TXExpression node", f"tx.{node.property_name}")
                return self.eval_tx_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.HashExpression):
                debug_log("  HashExpression node", "hash()")
                return self.eval_hash_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.SignatureExpression):
                debug_log("  SignatureExpression node", "signature()")
                return self.eval_signature_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.VerifySignatureExpression):
                debug_log("  VerifySignatureExpression node", "verify_sig()")
                return self.eval_verify_signature_expression(node, env, stack_trace)
            
            elif isinstance(node, zexus_ast.GasExpression):
                debug_log("  GasExpression node", f"gas.{node.property_name}")
                return self.eval_gas_expression(node, env, stack_trace)
            
            # Fallback
            debug_log("  Unknown node type", node_type)
            return EvaluationError(f"Unknown node type: {node_type}", stack_trace=stack_trace)
        
        except Exception as e:
            # Enhanced error with stack trace
            error_msg = f"Internal error: {str(e)}"
            debug_log("  Exception in eval_node", error_msg)
            traceback.print_exc()
            return EvaluationError(error_msg, stack_trace=stack_trace[-5:])  # Last 5 frames

# Additional VM-related methods
    def _initialize_vm(self):
        """Initialize VM components for bytecode execution"""
        try:
            # Create bytecode compiler with caching enabled
            self.bytecode_compiler = EvaluatorBytecodeCompiler(
                use_cache=True,
                cache_size=1000
            )
            # Create VM instance with JIT enabled
            self.vm_instance = VM(use_jit=True, jit_threshold=100)
            debug_log("Evaluator", "VM integration initialized successfully (cache + JIT)")
        except Exception as e:
            debug_log("Evaluator", f"Failed to initialize VM: {e}")
            self.use_vm = False
    
    def _should_use_vm(self, node) -> bool:
        """
        Determine if this node should be executed via VM.
        Uses heuristics to decide when VM execution is beneficial.
        """
        if not self.use_vm or not VM_AVAILABLE:
            return False
        
        # Check if the compiler can handle this node
        if self.bytecode_compiler and not self.bytecode_compiler.can_compile(node):
            return False
        
        # Use external heuristics
        return should_use_vm_for_node(node)
    
    def _execute_via_vm(self, node, env, debug_mode=False):
        """
        Compile node to bytecode and execute via VM.
        Falls back to direct evaluation on error.
        """
        try:
            debug_log("VM Execution", f"Compiling {type(node).__name__} to bytecode")
            
            # Compile to bytecode
            bytecode = self.bytecode_compiler.compile(node, optimize=True)
            
            if bytecode is None or self.bytecode_compiler.errors:
                debug_log("VM Execution", f"Compilation failed: {self.bytecode_compiler.errors}")
                self.vm_stats['vm_fallbacks'] += 1
                return None  # Signal fallback
            
            self.vm_stats['bytecode_compiles'] += 1
            
            # Convert environment to dict for VM
            vm_env = self._env_to_dict(env)
            
            # Add builtins to VM environment
            vm_builtins = {}
            if hasattr(self, 'builtins') and self.builtins:
                vm_builtins = {k: v for k, v in self.builtins.items()}
            
            # Use shared VM instance (has JIT, optimizer, etc.)
            if not self.vm_instance:
                self.vm_instance = VM(use_jit=True, jit_threshold=100)
            
            self.vm_instance.builtins = vm_builtins
            self.vm_instance.env = vm_env
            result = self.vm_instance.execute(bytecode, debug=debug_mode)
            
            self.vm_stats['vm_executions'] += 1
            
            # Update environment with VM changes
            self._update_env_from_dict(env, self.vm_instance.env)
            
            # Convert VM result back to evaluator objects
            return self._vm_result_to_evaluator(result)
            
        except Exception as e:
            debug_log("VM Execution", f"VM execution error: {e}")
            self.vm_stats['vm_fallbacks'] += 1
            return None  # Signal fallback
    
    def _env_to_dict(self, env):
        """Convert Environment object to dict for VM"""
        result = {}
        try:
            from ..object import String, Integer, Float, Boolean, List, Map, NULL
            
            def to_python(obj):
                """Convert evaluator object to Python primitive"""
                if obj is NULL or obj is None:
                    return None
                elif isinstance(obj, (String, Integer, Float, Boolean)):
                    return obj.value
                elif isinstance(obj, List):
                    return [to_python(e) for e in obj.elements]
                elif isinstance(obj, Map):
                    return {k: to_python(v) for k, v in obj.pairs.items()}
                else:
                    # Return as-is for complex objects
                    return obj
            
            # Get all variables from environment
            if hasattr(env, 'store') and isinstance(env.store, dict):
                for k, v in env.store.items():
                    result[k] = to_python(v)
            
            # Get outer environment if available
            if hasattr(env, 'outer') and env.outer:
                outer_dict = self._env_to_dict(env.outer)
                # Don't overwrite inner scope
                for k, v in outer_dict.items():
                    if k not in result:
                        result[k] = v
        except Exception as e:
            debug_log("_env_to_dict", f"Error: {e}")
        
        return result
    
    def _update_env_from_dict(self, env, vm_env: dict):
        """Update Environment object from VM's dict"""
        try:
            for key, value in vm_env.items():
                # Convert VM result back to evaluator object
                evaluator_value = self._vm_result_to_evaluator(value)
                # Update environment
                if hasattr(env, 'set'):
                    env.set(key, evaluator_value)
                elif hasattr(env, 'store'):
                    env.store[key] = evaluator_value
        except Exception as e:
            debug_log("_update_env_from_dict", f"Error: {e}")
    
    def _vm_result_to_evaluator(self, result):
        """Convert VM result to evaluator object"""
        from ..object import String, Integer, Float, Boolean, List, Map, NULL
        
        if result is None:
            return NULL
        elif isinstance(result, bool):
            return Boolean(result)
        elif isinstance(result, int):
            return Integer(result)
        elif isinstance(result, float):
            return Float(result)
        elif isinstance(result, str):
            return String(result)
        elif isinstance(result, list):
            return List([self._vm_result_to_evaluator(e) for e in result])
        elif isinstance(result, dict):
            converted = {k: self._vm_result_to_evaluator(v) for k, v in result.items()}
            return Map(converted)
        else:
            # Return as-is for complex objects
            return result
    
    def eval_with_vm_support(self, node, env, stack_trace=None, debug_mode=False):
        """
        Evaluate node with optional VM execution.
        Tries VM first if beneficial, falls back to direct evaluation.
        """
        # Check if we should use VM
        if self._should_use_vm(node):
            result = self._execute_via_vm(node, env, debug_mode)
            if result is not None:
                return result
            # Fall through to direct evaluation
        
        # Direct evaluation
        self.vm_stats['direct_evals'] += 1
        return self.eval_node(node, env, stack_trace)
    
    def get_vm_stats(self) -> dict:
        """Return VM execution statistics"""
        return self.vm_stats.copy()


    def get_full_vm_statistics(self):
        """Get comprehensive statistics from all VM components"""
        stats = {
            'evaluator': self.vm_stats.copy(),
            'cache': None,
            'jit': None,
            'optimizer': None
        }
        
        # Cache statistics
        if self.bytecode_compiler and self.bytecode_compiler.cache:
            stats['cache'] = self.bytecode_compiler.get_cache_stats()
        
        # JIT statistics
        if self.vm_instance:
            stats['jit'] = self.vm_instance.get_jit_stats()
        
        return stats

# Global Entry Point
def evaluate(program, env, debug_mode=False, use_vm=True):
    if debug_mode: 
        env.enable_debug()
    
    # Instantiate the Modular Evaluator with VM support
    evaluator = Evaluator(use_vm=use_vm)

    # Merge any module-level builtin injections (tests may add async helpers, etc.)
    try:
        from . import builtins as injected_builtins
        if isinstance(injected_builtins, dict):
            evaluator.builtins.update(injected_builtins)
    except Exception:
        pass
    
    # Try VM-accelerated execution for the whole program if beneficial
    if use_vm and VM_AVAILABLE:
        result = evaluator.eval_with_vm_support(program, env, debug_mode=debug_mode)
    else:
        result = evaluator.eval_node(program, env)
    
    if debug_mode: 
        env.disable_debug()
    
    return result
