# src/zexus/evaluator/core.py
import traceback
import asyncio
import os
import re
import sys
from .. import zexus_ast
from ..object import Environment, EvaluationError, Null, Boolean as BooleanObj, Map, EmbeddedCode, List, Action, LambdaFunction, String, ReturnValue, Builtin, Integer, Float
from .utils import is_error, debug_log, EVAL_SUMMARY, NULL
from ..config import config as zexus_config
from .expressions import ExpressionEvaluatorMixin
from .statements import StatementEvaluatorMixin
from .functions import FunctionEvaluatorMixin
from .integration import EvaluationContext, get_integration
from .resource_limiter import ResourceLimiter, ResourceError, TimeoutError

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
    def __init__(self, trusted: bool = False, use_vm: bool = True, resource_limiter=None):
        # Initialize mixins (FunctionEvaluatorMixin sets up builtins)
        FunctionEvaluatorMixin.__init__(self)

        self._ensure_recursion_headroom()
        
        # Initialize 10-phase integration
        self.integration_context = EvaluationContext("evaluator")
        
        # Setup security context
        if trusted:
            self.integration_context.setup_for_trusted_code()
        else:
            self.integration_context.setup_for_untrusted_code()
        
        # Resource limiting (Security Fix #7)
        self.resource_limiter = resource_limiter or ResourceLimiter()
        
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
        
        # Initialize unified executor (automatic VM integration)
        self.unified_executor = None
        try:
            from .unified_execution import create_unified_executor
            self.unified_executor = create_unified_executor(self)
        except ImportError:
            debug_log("Evaluator", "Unified executor not available")
        
        # Dispatch table for hot node types (avoids repeated isinstance chains)
        self._node_handlers = {}
        self._initialize_dispatch_table()

        if self.use_vm and VM_AVAILABLE:
            self._initialize_vm()

    # ------------------------------------------------------------------
    # Backward compatible entrypoints
    # ------------------------------------------------------------------

    def eval(self, program, env, *, debug_mode: bool = False, use_vm: bool | None = None):
        """Evaluate an AST node, mirroring the legacy Evaluator API."""
        previous_use_vm = self.use_vm
        try:
            if use_vm is not None:
                self.use_vm = bool(use_vm) and VM_AVAILABLE

            if debug_mode and hasattr(env, "enable_debug"):
                env.enable_debug()

            return self.eval_with_vm_support(program, env, debug_mode=debug_mode)
        finally:
            if debug_mode and hasattr(env, "disable_debug"):
                env.disable_debug()
            self.use_vm = previous_use_vm

    def _ensure_recursion_headroom(self, minimum: int = 5000):
        """(M8) Keep recursion usable, but avoid an unbounded recursion-limit jump.

        The interpreter's evaluation strategy uses Python recursion for AST
        traversal and function calls. With the default Python recursion limit,
        relatively small program recursion (e.g., factorial(100)) can trigger
        `RecursionError`.

        Default behavior:
        - Raise the recursion limit to a conservative target (3000) if the
          current limit is lower.
        - Never raise above `minimum`.

        Opt-in override:
        - Set `ZEXUS_PYTHON_RECURSION_LIMIT=<int>` to request a specific limit.

        Opt-out:
        - Set `ZEXUS_DISABLE_RECURSION_LIMIT_RAISE=1` to disable any adjustment.
        """
        if os.getenv("ZEXUS_DISABLE_RECURSION_LIMIT_RAISE") == "1":
            return

        requested = os.getenv("ZEXUS_PYTHON_RECURSION_LIMIT")
        default_target = 3000

        try:
            current = sys.getrecursionlimit()
            requested_limit = int(requested) if requested else default_target
            if requested_limit <= 0:
                return

            target = min(max(current, requested_limit), minimum)
            if target > current:
                sys.setrecursionlimit(target)
        except Exception:
            return

    def _initialize_dispatch_table(self):
        """Precompute handlers for AST node types to eliminate isinstance overhead.

        (M4) This table is the single dispatch source of truth.
        """

        def _camel_to_snake(name: str) -> str:
            # Handles acronyms reasonably (e.g., TXExpression -> tx_expression)
            s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
            return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

        def _ast(name: str):
            return getattr(zexus_ast, name, None)

        self._node_handlers = {}

        # Core hot-path nodes with dedicated handler methods
        _core = {
            "Program": self._handle_program_node,
            "ExpressionStatement": self._handle_expression_statement,
            "BlockStatement": self._handle_block_statement,
            "ReturnStatement": self._handle_return_statement,
            "LetStatement": self._handle_let_statement,
            "ConstStatement": self._handle_const_statement,
            "AssignmentExpression": self._handle_assignment_expression,
            "IfStatement": self._handle_if_statement,
            "WhileStatement": self._handle_while_statement,
            "ActionStatement": self._handle_action_statement,
            "FunctionStatement": self._handle_function_statement,
            "Identifier": self._handle_identifier,
            "IntegerLiteral": self._handle_integer_literal,
            "Boolean": self._handle_boolean_literal,
            "NullLiteral": self._handle_null_literal,
            "ThisExpression": self._handle_this_expression,
            "InfixExpression": self._handle_infix_expression,
            "PrefixExpression": self._handle_prefix_expression,
            "CallExpression": self._handle_call_expression,
            "PropertyAccessExpression": self._handle_property_access_expression,
            "MethodCallExpression": self._handle_method_call_expression,
            "ListLiteral": self._handle_list_literal,
            "StringLiteral": self._handle_string_literal,
            "StringInterpolationExpression": self._handle_string_interpolation,
            "FloatLiteral": self._handle_float_literal,
            "MapLiteral": self._handle_map_literal,
            "LambdaExpression": self._handle_lambda_expression,
            "ActionLiteral": self._handle_action_literal,
            "ForEachStatement": self._handle_foreach_statement,
            "PrintStatement": self._handle_print_statement,
            "DataStatement": self._handle_data_statement,
            "TryCatchStatement": self._handle_try_catch_statement,
            "ThrowStatement": self._handle_throw_statement,
            "ContractStatement": self._handle_contract_statement,
            "ExportStatement": self._handle_export_statement,
            "UseStatement": self._handle_use_statement,
            "FromStatement": self._handle_from_statement,
            "IfExpression": self._handle_if_expression,
            "TernaryExpression": self._handle_ternary_expression,
            "ContinueStatement": self._handle_continue_statement,
            "BreakStatement": self._handle_break_statement,
        }
        for ast_name, handler in _core.items():
            ast_type = _ast(ast_name)
            if ast_type is not None:
                self._node_handlers[ast_type] = handler

        # Auto-register remaining node classes that follow the eval_<snake_case> convention.
        base_node = getattr(zexus_ast, "Node", None)
        for ast_name, ast_type in vars(zexus_ast).items():
            if not isinstance(ast_type, type):
                continue
            if ast_type in self._node_handlers:
                continue
            if base_node is not None and ast_type is base_node:
                continue

            try:
                if base_node is not None and not issubclass(ast_type, base_node):
                    continue
            except Exception:
                continue

            method_name = f"eval_{_camel_to_snake(ast_name)}"
            method = getattr(self, method_name, None)
            if callable(method):
                self._node_handlers[ast_type] = lambda node, env, st, m=method: m(node, env, st)

    def _handle_program_node(self, node, env, stack_trace):
        debug_log("  Program node", f"{len(node.statements)} statements")
        return self.ceval_program(node.statements, env)

    def _handle_expression_statement(self, node, env, stack_trace):
        debug_log("  ExpressionStatement node")
        return self.eval_node(node.expression, env, stack_trace)

    def _handle_block_statement(self, node, env, stack_trace):
        debug_log("  BlockStatement node", f"{len(node.statements)} statements")
        return self.eval_block_statement(node, env, stack_trace)

    def _handle_return_statement(self, node, env, stack_trace):
        debug_log("  ReturnStatement node")
        return self.eval_return_statement(node, env, stack_trace)

    def _handle_let_statement(self, node, env, stack_trace):
        return self.eval_let_statement(node, env, stack_trace)

    def _handle_const_statement(self, node, env, stack_trace):
        return self.eval_const_statement(node, env, stack_trace)

    def _handle_assignment_expression(self, node, env, stack_trace):
        debug_log("  AssignmentExpression node")
        return self.eval_assignment_expression(node, env, stack_trace)

    def _handle_if_statement(self, node, env, stack_trace):
        debug_log("  IfStatement node")
        return self.eval_if_statement(node, env, stack_trace)

    def _handle_while_statement(self, node, env, stack_trace):
        debug_log("  WhileStatement node")
        return self.eval_while_statement(node, env, stack_trace)

    def _handle_action_statement(self, node, env, stack_trace):
        debug_log("  ActionStatement node", f"action {node.name.value}")
        return self.eval_action_statement(node, env, stack_trace)

    def _handle_function_statement(self, node, env, stack_trace):
        debug_log("  FunctionStatement node", f"function {node.name.value}")
        debug_log("  FunctionStatement evaluate", f"{node.name.value} modifiers={getattr(node, 'modifiers', [])}")
        return self.eval_function_statement(node, env, stack_trace)

    def _handle_identifier(self, node, env, stack_trace):
        debug_log("  Identifier node", node.value)
        return self.eval_identifier(node, env)

    def _handle_integer_literal(self, node, env, stack_trace):
        return Integer(node.value)

    def _handle_boolean_literal(self, node, env, stack_trace):
        return BooleanObj(node.value)

    def _handle_null_literal(self, node, env, stack_trace):
        debug_log("  NullLiteral node")
        return NULL

    def _handle_this_expression(self, node, env, stack_trace):
        debug_log("  ThisExpression node")
        return self.eval_this_expression(node, env, stack_trace)

    def _handle_infix_expression(self, node, env, stack_trace):
        debug_log("  InfixExpression node", f"{node.left} {node.operator} {node.right}")
        return self.eval_infix_expression(node, env, stack_trace)

    def _handle_prefix_expression(self, node, env, stack_trace):
        debug_log("  PrefixExpression node", f"{node.operator} {node.right}")
        return self.eval_prefix_expression(node, env, stack_trace)

    def _handle_call_expression(self, node, env, stack_trace):
        debug_log("🚀 CallExpression node", f"Calling {node.function}")
        return self.eval_call_expression(node, env, stack_trace)

    def _handle_property_access_expression(self, node, env, stack_trace):
        debug_log("  PropertyAccessExpression node")
        return self.eval_property_access_expression(node, env, stack_trace)

    def _handle_method_call_expression(self, node, env, stack_trace):
        debug_log("  MethodCallExpression node", f"{node.object}.{node.method}")
        return self.eval_method_call_expression(node, env, stack_trace)

    def _handle_list_literal(self, node, env, stack_trace):
        elems = self.eval_expressions(node.elements, env)
        if is_error(elems):
            return elems
        return List(elems)

    # --- VM statement wrappers (moved out of eval_node for performance) ---

    @staticmethod
    def _wrap_vm_statement(statement, expected_type):
        if isinstance(statement, expected_type):
            return statement
        if isinstance(statement, dict):
            try:
                if expected_type is zexus_ast.GCStatement:
                    return zexus_ast.GCStatement(statement.get("action"))
                if expected_type is zexus_ast.InlineStatement:
                    return zexus_ast.InlineStatement(statement.get("function_name") or statement.get("name"))
                if expected_type is zexus_ast.DeferStatement:
                    return zexus_ast.DeferStatement(statement.get("code"))
                if expected_type is zexus_ast.NativeStatement:
                    return zexus_ast.NativeStatement(
                        statement.get("library_name") or statement.get("library"),
                        statement.get("function_name") or statement.get("function"),
                        args=statement.get("args"),
                        alias=statement.get("alias")
                    )
                if expected_type is zexus_ast.BufferStatement:
                    return zexus_ast.BufferStatement(
                        statement.get("buffer_name") or statement.get("name"),
                        statement.get("operation"),
                        statement.get("arguments"),
                    )
                if expected_type is zexus_ast.SIMDStatement:
                    return zexus_ast.SIMDStatement(
                        statement.get("operation"),
                        operands=statement.get("operands")
                    )
                if expected_type is zexus_ast.PatternStatement:
                    return zexus_ast.PatternStatement(
                        statement.get("expression"),
                        statement.get("cases") or []
                    )
            except Exception as exc:
                return EvaluationError(str(exc))
        return EvaluationError(f"Invalid payload for {expected_type.__name__}")

    # --- Extended dispatch handlers (avoid isinstance chain) ---

    def _handle_string_literal(self, node, env, stack_trace):
        value = node.value
        value = value.replace('\\n', '\n')
        value = value.replace('\\t', '\t')
        value = value.replace('\\r', '\r')
        value = value.replace('\\\\', '\\')
        value = value.replace('\\"', '"')
        value = value.replace("\\'", "'")
        return String(value, is_trusted=True)

    def _handle_string_interpolation(self, node, env, stack_trace):
        """Evaluate string interpolation: "hello ${name}" """
        result_parts = []
        for part_type, part_value in node.parts:
            if part_type == "str":
                result_parts.append(part_value)
            elif part_type == "expr":
                val = self.eval_node(part_value, env, stack_trace)
                if is_error(val):
                    return val
                if hasattr(val, 'value'):
                    result_parts.append(str(val.value))
                elif val is None:
                    result_parts.append("null")
                else:
                    result_parts.append(str(val))
        return String(''.join(result_parts))

    def _handle_float_literal(self, node, env, stack_trace):
        try:
            return Float(node.value)
        except Exception:
            return EvaluationError(f"Invalid float literal: {getattr(node, 'value', None)}")

    def _handle_map_literal(self, node, env, stack_trace):
        pairs = {}
        for k, v in node.pairs:
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

    def _handle_lambda_expression(self, node, env, stack_trace):
        return LambdaFunction(node.parameters, node.body, env)

    def _handle_action_literal(self, node, env, stack_trace):
        return Action(node.parameters, node.body, env)

    def _handle_foreach_statement(self, node, env, stack_trace):
        return self.eval_foreach_statement(node, env, stack_trace)

    def _handle_print_statement(self, node, env, stack_trace):
        return self.eval_print_statement(node, env, stack_trace)

    def _handle_data_statement(self, node, env, stack_trace):
        return self.eval_data_statement(node, env, stack_trace)

    def _handle_try_catch_statement(self, node, env, stack_trace):
        return self.eval_try_catch_statement(node, env, stack_trace)

    def _handle_throw_statement(self, node, env, stack_trace):
        return self.eval_throw_statement(node, env, stack_trace)

    def _handle_contract_statement(self, node, env, stack_trace):
        return self.eval_contract_statement(node, env, stack_trace)

    def _handle_export_statement(self, node, env, stack_trace):
        return self.eval_export_statement(node, env, stack_trace)

    def _handle_use_statement(self, node, env, stack_trace):
        return self.eval_use_statement(node, env, stack_trace)

    def _handle_from_statement(self, node, env, stack_trace):
        return self.eval_from_statement(node, env, stack_trace)

    def _handle_if_expression(self, node, env, stack_trace):
        return self.eval_if_expression(node, env, stack_trace)

    def _handle_ternary_expression(self, node, env, stack_trace):
        return self.eval_ternary_expression(node, env, stack_trace)

    def _handle_continue_statement(self, node, env, stack_trace):
        return self.eval_continue_statement(node, env, stack_trace)

    def _handle_break_statement(self, node, env, stack_trace):
        return self.eval_break_statement(node, env, stack_trace)

    def eval_node(self, node, env, stack_trace=None):
        if node is None: 
            return NULL

        # DAP debug hook — check breakpoints / stepping before dispatch
        _dbg = getattr(self, "_debug_engine", None)
        if _dbg is not None and hasattr(node, "__class__"):
            _line = getattr(node, "line", None) or getattr(node, "token_line", 0)
            if _line:
                _file = getattr(self, "_debug_file", "<unknown>")
                _dbg.check(_file, _line, env)

        node_type = type(node)

        handler = self._node_handlers.get(node_type)
        if handler is None:
            return EvaluationError(f"Unsupported AST node type: {node_type.__name__}")

        try:
            return handler(node, env, stack_trace)
        except RecursionError:
            return EvaluationError(
                "Maximum recursion depth exceeded while evaluating AST (Python recursion limit). "
                "Consider simplifying deeply nested expressions or enabling the VM.",
                stack_trace=stack_trace,
            )
        except Exception as e:
            error_msg = f"Internal error: {str(e)}"
            debug_log("  Exception in eval_node", error_msg)
            traceback.print_exc()
            trimmed_trace = stack_trace[-5:] if isinstance(stack_trace, list) else stack_trace
            return EvaluationError(error_msg, stack_trace=trimmed_trace)

# Additional VM-related methods
    def _initialize_vm(self):
        """Initialize VM components for bytecode execution"""
        try:
            # Create bytecode compiler with caching enabled
            self.bytecode_compiler = EvaluatorBytecodeCompiler(
                use_cache=True,
                cache_size=1000
            )
            # Create VM instance with JIT and gas metering enabled
            self.vm_instance = VM(
                use_jit=True, 
                jit_threshold=100,
                enable_gas_metering=True,
                gas_limit=1_000_000  # Default 1M gas limit
            )
            debug_log("Evaluator", "VM integration initialized successfully (cache + JIT + gas metering)")
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
    
    def _execute_via_vm(self, node, env, debug_mode=False, file_path=None):
        """
        Compile node to bytecode and execute via VM.
        Falls back to direct evaluation on error.
        Stores compiled bytecode in file cache if file_path is provided.
        """
        try:
            debug_log("VM Execution", f"Compiling {type(node).__name__} to bytecode")
            
            # Compile to bytecode
            bytecode = self.bytecode_compiler.compile(node, optimize=True)
            
            if bytecode is None or self.bytecode_compiler.errors:
                debug_log("VM Execution", f"Compilation failed: {self.bytecode_compiler.errors}")
                if os.environ.get("ZEXUS_VM_FALLBACK_DEBUG"):
                    print(
                        "[VM FALLBACK] compile_failed "
                        f"node={type(node).__name__} file={file_path} "
                        f"errors={self.bytecode_compiler.errors}"
                    )
                self.vm_stats['vm_fallbacks'] += 1
                return None  # Signal fallback
            
            self.vm_stats['bytecode_compiles'] += 1
            
            # Store in file cache for faster repeat runs
            if file_path and self.bytecode_compiler.cache:
                self.bytecode_compiler.cache.put_by_file(file_path, [bytecode])
            
            # Convert environment to dict for VM
            vm_env = self._env_to_dict(env)
            
            # Add builtins to VM environment
            vm_builtins = {}
            if hasattr(self, 'builtins') and self.builtins:
                vm_builtins = {k: v for k, v in self.builtins.items()}

            vm_builtins.update(self._create_vm_keyword_builtins(env))
            
            # Use shared VM instance (has JIT, optimizer, etc.)
            if not self.vm_instance:
                self.vm_instance = VM(use_jit=True, jit_threshold=100)
            
            self.vm_instance.builtins = vm_builtins
            self.vm_instance.env = vm_env
            result = self.vm_instance.execute(bytecode, debug=debug_mode)
            profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
            verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
            if (
                profile_flag and profile_flag.lower() not in ("0", "false", "off")
                and verbose_flag and verbose_flag.lower() not in ("0", "false", "off")
            ):
                profile = getattr(self.vm_instance, "_last_opcode_profile", None)
                size = len(profile) if profile else 0
                print(f"[VM DEBUG] evaluator opcode profile size={size}")
            
            self.vm_stats['vm_executions'] += 1
            
            # Update environment with VM changes
            self._update_env_from_dict(env, self.vm_instance.env)
            
            # Convert VM result back to evaluator objects
            return self._vm_result_to_evaluator(result)
            
        except Exception as e:
            debug_log("VM Execution", f"VM execution error: {e}")
            if os.environ.get("ZEXUS_VM_FALLBACK_DEBUG"):
                print(
                    "[VM FALLBACK] execute_failed "
                    f"node={type(node).__name__} file={file_path} error={e}"
                )
            self.vm_stats['vm_fallbacks'] += 1
            return None  # Signal fallback

    def _execute_bytecode_sequence(self, bytecodes, env, debug_mode=False):
        """Execute a sequence of bytecode objects in the VM for cached file runs."""
        try:
            if not bytecodes:
                return None

            # Convert environment to dict for VM
            vm_env = self._env_to_dict(env)

            # Add builtins to VM environment
            vm_builtins = {}
            if hasattr(self, 'builtins') and self.builtins:
                vm_builtins = {k: v for k, v in self.builtins.items()}
            vm_builtins.update(self._create_vm_keyword_builtins(env))

            if not self.vm_instance:
                self.vm_instance = VM(use_jit=True, jit_threshold=100)

            self.vm_instance.builtins = vm_builtins
            self.vm_instance.env = vm_env

            result = None
            for bc in bytecodes:
                if bc is None:
                    continue
                result = self.vm_instance.execute(bc, debug=debug_mode)

            self.vm_stats['vm_executions'] += 1
            self._update_env_from_dict(env, self.vm_instance.env)
            return self._vm_result_to_evaluator(result)
        except Exception as e:
            debug_log("VM Execution", f"VM cached execution error: {e}")
            if os.environ.get("ZEXUS_VM_FALLBACK_DEBUG"):
                print(f"[VM FALLBACK] cached_execute_failed error={e}")
            self.vm_stats['vm_fallbacks'] += 1
            return None
    
    def _create_vm_keyword_builtins(self, env):
        """Expose keyword helpers to the VM so bytecode can reuse evaluator logic"""
        def _vm_find(node_ref):
            return self.eval_find_expression(node_ref, env, stack_trace=[])

        def _vm_load(node_ref):
            return self.eval_load_expression(node_ref, env, stack_trace=[])

        def _vm_use_module(spec):
            if spec is None:
                return NULL
            try:
                from .. import zexus_ast

                file_path = spec.get("file", "") if isinstance(spec, dict) else ""
                alias_raw = spec.get("alias") if isinstance(spec, dict) else None
                names_raw = spec.get("names") if isinstance(spec, dict) else []
                is_named = bool(spec.get("is_named")) if isinstance(spec, dict) else False

                alias_node = zexus_ast.Identifier(alias_raw) if alias_raw else None
                name_nodes = []
                if isinstance(names_raw, list):
                    for name in names_raw:
                        if name:
                            name_nodes.append(zexus_ast.Identifier(name))

                use_node = zexus_ast.UseStatement(
                    file_path,
                    alias=alias_node,
                    names=name_nodes,
                    is_named_import=is_named
                )
                return self.eval_use_statement(use_node, env, stack_trace=[])
            except Exception as exc:
                return EvaluationError(str(exc))

        def _vm_from_module(spec):
            if spec is None:
                return NULL
            try:
                from .. import zexus_ast

                file_path = spec.get("file", "") if isinstance(spec, dict) else ""
                raw_imports = spec.get("imports") if isinstance(spec, dict) else []

                imports = []
                if isinstance(raw_imports, list):
                    for entry in raw_imports:
                        if isinstance(entry, dict):
                            name = entry.get("name")
                            alias = entry.get("alias")
                        elif isinstance(entry, (list, tuple)):
                            name = entry[0] if len(entry) > 0 else None
                            alias = entry[1] if len(entry) > 1 else None
                        else:
                            name = entry
                            alias = None

                        name_node = zexus_ast.Identifier(name) if name else None
                        alias_node = zexus_ast.Identifier(alias) if alias else None
                        imports.append((name_node, alias_node))

                from_node = zexus_ast.FromStatement(file_path, imports=imports)
                return self.eval_from_statement(from_node, env, stack_trace=[])
            except Exception as exc:
                return EvaluationError(str(exc))

        def _wrap_statement(statement, expected_type):
            if isinstance(statement, expected_type):
                return statement
            if isinstance(statement, dict):
                try:
                    if expected_type is zexus_ast.GCStatement:
                        return zexus_ast.GCStatement(statement.get("action"))
                    if expected_type is zexus_ast.InlineStatement:
                        return zexus_ast.InlineStatement(statement.get("function_name") or statement.get("name"))
                    if expected_type is zexus_ast.DeferStatement:
                        return zexus_ast.DeferStatement(statement.get("code"))
                    if expected_type is zexus_ast.NativeStatement:
                        return zexus_ast.NativeStatement(
                            statement.get("library_name") or statement.get("library"),
                            statement.get("function_name") or statement.get("function"),
                            args=statement.get("args"),
                            alias=statement.get("alias")
                        )
                    if expected_type is zexus_ast.BufferStatement:
                        return zexus_ast.BufferStatement(
                            statement.get("buffer_name") or statement.get("name"),
                            statement.get("operation"),
                            statement.get("arguments"),
                        )
                    if expected_type is zexus_ast.SIMDStatement:
                        return zexus_ast.SIMDStatement(
                            statement.get("operation"),
                            operands=statement.get("operands")
                        )
                    if expected_type is zexus_ast.PatternStatement:
                        return zexus_ast.PatternStatement(
                            statement.get("expression"),
                            statement.get("cases") or []
                        )
                except Exception as exc:
                    return EvaluationError(str(exc))
            return EvaluationError(f"Invalid payload for {expected_type.__name__}")

        def _vm_native_statement(node, _env=env):
            stmt = _wrap_statement(node, zexus_ast.NativeStatement)
            if isinstance(stmt, EvaluationError):
                return stmt
            return self.eval_native_statement(stmt, _env, stack_trace=[])

        def _vm_gc_statement(node, _env=env):
            stmt = _wrap_statement(node, zexus_ast.GCStatement)
            if isinstance(stmt, EvaluationError):
                return stmt
            return self.eval_gc_statement(stmt, _env, stack_trace=[])

        def _vm_inline_statement(node, _env=env):
            stmt = _wrap_statement(node, zexus_ast.InlineStatement)
            if isinstance(stmt, EvaluationError):
                return stmt
            return self.eval_inline_statement(stmt, _env, stack_trace=[])

        def _vm_buffer_statement(node, _env=env):
            stmt = _wrap_statement(node, zexus_ast.BufferStatement)
            if isinstance(stmt, EvaluationError):
                return stmt
            return self.eval_buffer_statement(stmt, _env, stack_trace=[])

        def _vm_simd_statement(node, _env=env):
            stmt = _wrap_statement(node, zexus_ast.SIMDStatement)
            if isinstance(stmt, EvaluationError):
                return stmt
            return self.eval_simd_statement(stmt, _env, stack_trace=[])

        def _vm_defer_statement(node, _env=env):
            stmt = _wrap_statement(node, zexus_ast.DeferStatement)
            if isinstance(stmt, EvaluationError):
                return stmt
            return self.eval_defer_statement(stmt, _env, stack_trace=[])

        def _vm_pattern_statement(node, _env=env):
            stmt = _wrap_statement(node, zexus_ast.PatternStatement)
            if isinstance(stmt, EvaluationError):
                return stmt
            return self.eval_pattern_statement(stmt, _env, stack_trace=[])

        return {
            "__keyword_find__": Builtin(_vm_find, "__keyword_find__"),
            "__keyword_load__": Builtin(_vm_load, "__keyword_load__"),
            "__vm_use_module__": Builtin(_vm_use_module, "__vm_use_module__"),
            "__vm_from_module__": Builtin(_vm_from_module, "__vm_from_module__"),
            "__vm_native_statement__": Builtin(_vm_native_statement, "__vm_native_statement__"),
            "__vm_gc_statement__": Builtin(_vm_gc_statement, "__vm_gc_statement__"),
            "__vm_inline_statement__": Builtin(_vm_inline_statement, "__vm_inline_statement__"),
            "__vm_buffer_statement__": Builtin(_vm_buffer_statement, "__vm_buffer_statement__"),
            "__vm_simd_statement__": Builtin(_vm_simd_statement, "__vm_simd_statement__"),
            "__vm_defer_statement__": Builtin(_vm_defer_statement, "__vm_defer_statement__"),
            "__vm_pattern_statement__": Builtin(_vm_pattern_statement, "__vm_pattern_statement__"),
        }

    def _env_to_dict(self, env, *, _depth: int = 0, _max_depth: int = 64, _seen=None):
        """Convert Environment object to dict for VM.

        LI7: Guard against cycles and unbounded `outer` chains.
        """
        result = {}
        try:
            from ..object import String, Integer, Float, Boolean, List, Map, NULL

            if env is None:
                return result

            if _seen is None:
                _seen = set()
            env_id = id(env)
            if env_id in _seen:
                return result
            _seen.add(env_id)

            if _depth >= _max_depth:
                return result
            
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
                outer_dict = self._env_to_dict(
                    env.outer,
                    _depth=_depth + 1,
                    _max_depth=_max_depth,
                    _seen=_seen,
                )
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
        Uses file-based cache for repeat runs, then tries VM for beneficial nodes.
        """
        # Try file-based cache first for whole-file acceleration
        file_path = None
        try:
            file_obj = env.get("__file__")
            if file_obj and hasattr(file_obj, 'value'):
                file_path = file_obj.value
        except Exception:
            pass
        
        # For program nodes with file context, try cached execution
        if (
            self.use_vm
            and VM_AVAILABLE
            and file_path
            and self.bytecode_compiler
            and type(node).__name__ == 'Program'
            and self.bytecode_compiler.cache
        ):
            cached_bytecodes = None
            if self.bytecode_compiler.cache.is_file_cached(file_path):
                cached_bytecodes = self.bytecode_compiler.cache.get_by_file(file_path)
                if cached_bytecodes:
                    debug_log("VM Execution", f"Using cached bytecode for {file_path}")
                    result = self._execute_bytecode_sequence(cached_bytecodes, env, debug_mode)
                    if result is not None:
                        return result

            # Cache miss or invalid - compile full file and store
            compiled = self.bytecode_compiler.compile_file(file_path, node, optimize=True)
            if compiled:
                debug_log("VM Execution", f"Cached compilation for {file_path} ({len(compiled)} bytecodes)")
                result = self._execute_bytecode_sequence(compiled, env, debug_mode)
                if result is not None:
                    return result
        
        # Check if we should use VM for this node
        if self._should_use_vm(node):
            result = self._execute_via_vm(node, env, debug_mode, file_path)
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
            'optimizer': None,
            'fast_loop': None
        }
        
        # Cache statistics
        if self.bytecode_compiler and self.bytecode_compiler.cache:
            stats['cache'] = self.bytecode_compiler.get_cache_stats()
        
        # JIT statistics
        if self.vm_instance:
            stats['jit'] = self.vm_instance.get_jit_stats()
            stats['fast_loop'] = getattr(self.vm_instance, "_fast_loop_stats", None)
        
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
            from ..object import Builtin

            for name, value in injected_builtins.items():
                if isinstance(value, Builtin):
                    evaluator.builtins[name] = value
                elif callable(value):
                    evaluator.builtins[name] = Builtin(value, name)
                else:
                    evaluator.builtins[name] = value
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
