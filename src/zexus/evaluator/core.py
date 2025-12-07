# src/zexus/evaluator/core.py
import traceback
from .. import zexus_ast
from ..object import Environment, EvaluationError, Null, Boolean as BooleanObj
from .utils import is_error, debug_log, EVAL_SUMMARY, NULL
from .expressions import ExpressionEvaluatorMixin
from .statements import StatementEvaluatorMixin
from .functions import FunctionEvaluatorMixin

class Evaluator(ExpressionEvaluatorMixin, StatementEvaluatorMixin, FunctionEvaluatorMixin):
    def __init__(self):
        # Initialize mixins (FunctionEvaluatorMixin sets up builtins)
        FunctionEvaluatorMixin.__init__(self)
    
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
            if node_type == zexus_ast.Program:
                debug_log("  Program node", f"{len(node.statements)} statements")
                return self.eval_program(node.statements, env)
            
            elif node_type == zexus_ast.ExpressionStatement:
                debug_log("  ExpressionStatement node")
                return self.eval_node(node.expression, env, stack_trace)
            
            elif node_type == zexus_ast.BlockStatement:
                debug_log("  BlockStatement node", f"{len(node.statements)} statements")
                return self.eval_block_statement(node, env)
            
            elif node_type == zexus_ast.ReturnStatement:
                debug_log("  ReturnStatement node")
                return self.eval_return_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.LetStatement:
                return self.eval_let_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ConstStatement:
                return self.eval_const_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.AssignmentExpression:
                debug_log("  AssignmentExpression node")
                return self.eval_assignment_expression(node, env, stack_trace)
            
            elif node_type == zexus_ast.TryCatchStatement:
                return self.eval_try_catch_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.IfStatement:
                debug_log("  IfStatement node")
                return self.eval_if_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.WhileStatement:
                debug_log("  WhileStatement node")
                return self.eval_while_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ForEachStatement:
                debug_log("  ForEachStatement node", f"for each {node.item.value}")
                return self.eval_foreach_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.SealStatement:
                return self.eval_seal_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.RestrictStatement:
                return self.eval_restrict_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.SandboxStatement:
                return self.eval_sandbox_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.TrailStatement:
                return self.eval_trail_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.EntityStatement:
                return self.eval_entity_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ContractStatement:
                return self.eval_contract_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ExportStatement:
                return self.eval_export_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.UseStatement:
                debug_log("  UseStatement node", node.file_path)
                return self.eval_use_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.FromStatement:
                debug_log("  FromStatement node", node.file_path)
                return self.eval_from_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.VerifyStatement:
                return self.eval_verify_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ProtectStatement:
                return self.eval_protect_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.MiddlewareStatement:
                return self.eval_middleware_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.AuthStatement:
                return self.eval_auth_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ThrottleStatement:
                return self.eval_throttle_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.CacheStatement:
                return self.eval_cache_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.PrintStatement:
                debug_log("  PrintStatement node")
                return self.eval_print_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ScreenStatement:
                debug_log("  ScreenStatement node", node.name.value)
                return self.eval_screen_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.EmbeddedCodeStatement:
                debug_log("  EmbeddedCodeStatement node", node.name.value)
                return self.eval_embedded_code_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ComponentStatement:
                debug_log("  ComponentStatement node", node.name.value)
                return self.eval_component_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ThemeStatement:
                debug_log("  ThemeStatement node", node.name.value)
                return self.eval_theme_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.DebugStatement:
                debug_log("  DebugStatement node")
                return self.eval_debug_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ExternalDeclaration:
                debug_log("  ExternalDeclaration node", node.name.value)
                return self.eval_external_declaration(node, env, stack_trace)
            
            elif node_type == zexus_ast.ExactlyStatement:
                debug_log("  ExactlyStatement node")
                return self.eval_exactly_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.ActionStatement:
                debug_log("  ActionStatement node", f"action {node.name.value}")
                return self.eval_action_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.NativeStatement:
                debug_log("  NativeStatement node", f"native {node.function_name}")
                return self.eval_native_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.GCStatement:
                debug_log("  GCStatement node", f"gc {node.action}")
                return self.eval_gc_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.InlineStatement:
                debug_log("  InlineStatement node", f"inline {node.function_name}")
                return self.eval_inline_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.BufferStatement:
                debug_log("  BufferStatement node", f"buffer {node.buffer_name}")
                return self.eval_buffer_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.SIMDStatement:
                debug_log("  SIMDStatement node")
                return self.eval_simd_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.DeferStatement:
                debug_log("  DeferStatement node")
                return self.eval_defer_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.PatternStatement:
                debug_log("  PatternStatement node")
                return self.eval_pattern_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.EnumStatement:
                debug_log("  EnumStatement node", f"enum {node.name}")
                return self.eval_enum_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.StreamStatement:
                debug_log("  StreamStatement node", f"stream {node.stream_name}")
                return self.eval_stream_statement(node, env, stack_trace)
            
            elif node_type == zexus_ast.WatchStatement:
                debug_log("  WatchStatement node")
                return self.eval_watch_statement(node, env, stack_trace)
            
            # === EXPRESSIONS ===
            elif node_type == zexus_ast.Identifier:
                debug_log("  Identifier node", node.value)
                return self.eval_identifier(node, env)
            
            elif node_type == zexus_ast.IntegerLiteral:
                debug_log("  IntegerLiteral node", node.value)
                from ..object import Integer
                return Integer(node.value)
            
            elif node_type == zexus_ast.FloatLiteral or node_type.__name__ == 'FloatLiteral':
                debug_log("  FloatLiteral node", getattr(node, 'value', 'unknown'))
                from ..object import Float
                try:
                    val = getattr(node, 'value', None)
                    return Float(val)
                except Exception:
                    return EvaluationError(f"Invalid float literal: {getattr(node, 'value', None)}")
            
            elif node_type == zexus_ast.StringLiteral:
                debug_log("  StringLiteral node", node.value)
                from ..object import String
                return String(node.value)
            
            elif node_type == zexus_ast.Boolean:
                debug_log("  Boolean node", f"value: {node.value}")
                from ..object import Boolean
                return Boolean(node.value)
            
            elif node_type == zexus_ast.InfixExpression:
                debug_log("  InfixExpression node", f"{node.left} {node.operator} {node.right}")
                return self.eval_infix_expression(node, env, stack_trace)
            
            elif node_type == zexus_ast.PrefixExpression:
                debug_log("  PrefixExpression node", f"{node.operator} {node.right}")
                return self.eval_prefix_expression(node, env, stack_trace)
            
            elif node_type == zexus_ast.IfExpression:
                debug_log("  IfExpression node")
                return self.eval_if_expression(node, env, stack_trace)
            
            elif node_type == zexus_ast.CallExpression:
                debug_log("🚀 CallExpression node", f"Calling {node.function}")
                return self.eval_call_expression(node, env, stack_trace)
            
            elif node_type == zexus_ast.MethodCallExpression:
                debug_log("  MethodCallExpression node", f"{node.object}.{node.method}")
                return self.eval_method_call_expression(node, env, stack_trace)
            
            elif node_type == zexus_ast.ListLiteral:
                debug_log("  ListLiteral node", f"{len(node.elements)} elements")
                from ..object import List
                elems = self.eval_expressions(node.elements, env)
                if is_error(elems): 
                    return elems
                return List(elems)
            
            elif node_type == zexus_ast.MapLiteral:
                debug_log("  MapLiteral node", f"{len(node.pairs)} pairs")
                from ..object import Map
                pairs = {}
                for k, v in node.pairs:
                    key = self.eval_node(k, env, stack_trace)
                    if is_error(key): 
                        return key
                    val = self.eval_node(v, env, stack_trace)
                    if is_error(val): 
                        return val
                    key_str = key.inspect()
                    pairs[key_str] = val
                return Map(pairs)
            
            elif node_type == zexus_ast.ActionLiteral:
                debug_log("  ActionLiteral node")
                from ..object import Action
                return Action(node.parameters, node.body, env)
            
            elif node_type == zexus_ast.LambdaExpression:
                debug_log("  LambdaExpression node")
                from ..object import LambdaFunction
                return LambdaFunction(node.parameters, node.body, env)
            
            elif node_type == zexus_ast.EmbeddedLiteral:
                debug_log("  EmbeddedLiteral node")
                from ..object import EmbeddedCode
                return EmbeddedCode("embedded_block", node.language, node.code)
            
            elif node_type == zexus_ast.PropertyAccessExpression:
                debug_log("  PropertyAccessExpression node", f"{node.object}.{node.property}")
                obj = self.eval_node(node.object, env, stack_trace)
                if is_error(obj): 
                    return obj
                
                property_name = node.property.value
                
                if isinstance(obj, EmbeddedCode):
                    if property_name == "code":
                        from ..object import String
                        return String(obj.code)
                    elif property_name == "language":
                        from ..object import String
                        return String(obj.language)
                
                # Enforcement: consult security restrictions before returning
                try:
                    from ..security import get_security_context
                    ctx = get_security_context()
                    target = f"{getattr(node.object, 'value', str(node.object))}.{property_name}"
                    restriction = ctx.get_restriction(target)
                except Exception:
                    restriction = None

                # Handle Map objects
                if isinstance(obj, Map):
                    val = obj.pairs.get(property_name, NULL)
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
                                from ..object import EvaluationError
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
                                from ..object import EvaluationError
                                return EvaluationError('Access denied: admin required')
                    return val

                return NULL
            
            # Fallback
            debug_log("  Unknown node type", node_type)
            return EvaluationError(f"Unknown node type: {node_type}", stack_trace=stack_trace)
        
        except Exception as e:
            # Enhanced error with stack trace
            error_msg = f"Internal error: {str(e)}"
            debug_log("  Exception in eval_node", error_msg)
            traceback.print_exc()
            return EvaluationError(error_msg, stack_trace=stack_trace[-5:])  # Last 5 frames

# Global Entry Point
def evaluate(program, env, debug_mode=False):
    if debug_mode: 
        env.enable_debug()
    
    # Instantiate the Modular Evaluator
    evaluator = Evaluator()
    result = evaluator.eval_node(program, env)
    
    if debug_mode: 
        env.disable_debug()
    
    return result
