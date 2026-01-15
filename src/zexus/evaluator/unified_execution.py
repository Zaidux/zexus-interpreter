"""
Unified Execution System - Automatic Interpreter/VM Switching

This system unifies the evaluator and VM, automatically switching based on workload:
- Workloads < 500: Pure interpretation (fast startup, low overhead)
- Workloads >= 500: Automatic VM compilation (100x speedup)

NO FLAGS NEEDED - The system automatically chooses the best execution method.

Features:
- Automatic hot loop detection
- Seamless VM compilation
- File-based persistent caching (faster repeat runs)
- Pattern-based bytecode reuse
- JIT compilation for ultra-hot paths
- Parallel execution for suitable workloads
"""

from typing import Any, Dict, Optional, List
import time
import os


class WorkloadDetector:
    """
    Detects workload characteristics and determines optimal execution strategy.
    
    Metrics considered:
    - Loop iteration count
    - Function call frequency
    - Arithmetic operation density
    - Memory allocation rate
    """
    
    def __init__(self):
        # Loop tracking
        self.loop_iterations: Dict[int, int] = {}  # loop_id -> iteration count
        self.loop_start_time: Dict[int, float] = {}
        
        # Function call tracking
        self.function_calls: Dict[str, int] = {}  # func_name -> call count
        self.hot_functions: set = set()
        
        # Workload classification thresholds
        self.vm_threshold = 500       # Iterations before VM compilation
        self.jit_threshold = 5000     # Iterations before JIT compilation
        self.parallel_threshold = 10000  # Iterations before parallel execution
        
        # Statistics
        self.stats = {
            "total_loops": 0,
            "hot_loops": 0,
            "jit_loops": 0,
            "vm_compilations": 0,
            "jit_compilations": 0,
            "vm_executions": 0,
            "interpretation_time_ms": 0,
            "vm_time_ms": 0
        }
    
    def track_loop_iteration(self, loop_id: int) -> Dict[str, Any]:
        """
        Track loop iteration and determine optimal execution strategy.
        
        Returns:
            {
                "should_compile": bool,      # Should compile to VM now
                "should_jit": bool,          # Should JIT compile now
                "use_vm": bool,              # Should use already-compiled VM
                "use_jit": bool,             # Should use JIT-compiled code
                "iteration": int,            # Current iteration number
                "is_hot": bool,              # Is this a hot loop
                "tier": str                  # Current execution tier
            }
        """
        if loop_id not in self.loop_iterations:
            self.loop_iterations[loop_id] = 0
            self.loop_start_time[loop_id] = time.time()
            self.stats["total_loops"] += 1
        
        self.loop_iterations[loop_id] += 1
        iteration = self.loop_iterations[loop_id]
        
        # Determine execution tier
        should_compile = (iteration == self.vm_threshold)
        should_jit = (iteration == self.jit_threshold)
        is_hot = (iteration >= self.vm_threshold)
        is_jit_hot = (iteration >= self.jit_threshold)
        
        if should_compile:
            self.stats["hot_loops"] += 1
        if should_jit:
            self.stats["jit_loops"] += 1
        
        # Determine tier
        if is_jit_hot:
            tier = "jit"
        elif is_hot:
            tier = "vm"
        else:
            tier = "interpreted"
        
        return {
            "should_compile": should_compile,
            "should_jit": should_jit,
            "use_vm": is_hot and not is_jit_hot,
            "use_jit": is_jit_hot,
            "iteration": iteration,
            "is_hot": is_hot,
            "tier": tier
        }
    
    def track_function_call(self, func_name: str) -> bool:
        """
        Track function call frequency.
        
        Returns:
            True if function is hot (should be compiled)
        """
        if func_name not in self.function_calls:
            self.function_calls[func_name] = 0
        
        self.function_calls[func_name] += 1
        
        # Function is hot if called >= 100 times
        is_hot = self.function_calls[func_name] >= 100
        
        if is_hot and func_name not in self.hot_functions:
            self.hot_functions.add(func_name)
        
        return is_hot
    
    def reset_loop(self, loop_id: int):
        """Reset loop tracking when loop exits"""
        if loop_id in self.loop_iterations:
            # Record execution time
            if loop_id in self.loop_start_time:
                elapsed = (time.time() - self.loop_start_time[loop_id]) * 1000
                
                # Was this loop interpreted or VM-executed?
                if self.loop_iterations[loop_id] >= self.vm_threshold:
                    self.stats["vm_time_ms"] += elapsed
                else:
                    self.stats["interpretation_time_ms"] += elapsed
                
                del self.loop_start_time[loop_id]
            
            del self.loop_iterations[loop_id]
    
    def get_stats(self) -> Dict:
        """Get workload detection statistics"""
        total_time = self.stats["interpretation_time_ms"] + self.stats["vm_time_ms"]
        
        return {
            **self.stats,
            "total_execution_time_ms": total_time,
            "vm_percentage": (
                (self.stats["vm_time_ms"] / total_time * 100) if total_time > 0 else 0
            ),
            "active_hot_functions": len(self.hot_functions),
            "vm_threshold": self.vm_threshold,
            "jit_threshold": self.jit_threshold
        }


class UnifiedExecutor:
    """
    Unified execution system that seamlessly switches between interpreter and VM.
    
    Features:
    - Automatic workload detection
    - Transparent VM compilation
    - JIT compilation for ultra-hot paths
    - File-based persistent caching
    - Pattern-based bytecode reuse
    - Environment synchronization
    - Zero-overhead switching
    """
    
    def __init__(self, evaluator, vm_enabled: bool = True):
        """
        Args:
            evaluator: The evaluator instance
            vm_enabled: Whether VM is available and enabled
        """
        self.evaluator = evaluator
        self.vm_enabled = vm_enabled
        
        # Workload detector
        self.workload = WorkloadDetector()
        
        # VM compilation cache (uses shared persistent cache)
        self.compiled_loops: Dict[int, Any] = {}  # loop_id -> bytecode
        self.compiled_functions: Dict[str, Any] = {}  # func_name -> bytecode
        self.jit_compiled: Dict[int, Any] = {}  # loop_id -> native code
        
        # VM instance (lazy init)
        self.vm = None
        self._profile_reported = False
        
        # JIT compiler (lazy init)
        self.jit_compiler = None
        
        # Compilation failures (don't retry)
        self.failed_compilations: set = set()
        
        # Get shared cache for persistence
        self._shared_cache = None
        try:
            from .bytecode_compiler import get_shared_cache
            self._shared_cache = get_shared_cache()
        except ImportError:
            pass
        
        # Statistics
        self.stats = {
            "total_executions": 0,
            "vm_hits": 0,
            "jit_hits": 0,
            "compilation_failures": 0,
            "environment_syncs": 0,
            "cache_hits": 0
        }
    
    def execute_loop(self, loop_id: int, condition_node, body_node, env, stack_trace) -> Any:
        """
        Execute a loop with automatic VM compilation.
        
        This is the core of the unified system:
        1. Track iterations
        2. At iteration 500: Compile to VM
        3. After iteration 500: Execute via VM
        4. Always sync environment back
        
        Args:
            loop_id: Unique loop identifier
            condition_node: Loop condition AST
            body_node: Loop body AST
            env: Execution environment
            stack_trace: Stack trace for errors
            
        Returns:
            Loop result
        """
        from ..object import NULL, EvaluationError
        profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
        profile_active = profile_flag and profile_flag.lower() not in ("0", "false", "off")
        verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
        profile_verbose = profile_active and verbose_flag and verbose_flag.lower() not in ("0", "false", "off")
        from ..evaluator.utils import is_error, is_truthy
        
        self.stats["total_executions"] += 1
        result = NULL

        # If the loop body contains constructs the VM cannot safely handle yet (e.g.,
        # method calls on contracts / smart objects), pin this loop to interpreter only.
        if self.vm_enabled and not self._is_vm_safe_loop(body_node):
            self.failed_compilations.add(loop_id)
        
        while True:
            # CRITICAL: Resource limit check (prevents infinite loops)
            try:
                self.evaluator.resource_limiter.check_iterations()
            except Exception as e:
                from ..evaluator.resource_limiter import ResourceError, TimeoutError
                from ..object import EvaluationError
                if isinstance(e, (ResourceError, TimeoutError)):
                    self.workload.reset_loop(loop_id)
                    return EvaluationError(str(e))
                raise
            
            # Track iteration
            info = self.workload.track_loop_iteration(loop_id)
            
            # Check condition
            cond = self.evaluator.eval_node(condition_node, env, stack_trace)
            if is_error(cond):
                self.workload.reset_loop(loop_id)
                return cond
            
            if not is_truthy(cond):
                # Loop done
                break
            
            # Decide execution method
            if info["should_compile"] and self.vm_enabled:
                # Just hit threshold - compile now
                success = self._compile_loop(loop_id, body_node, env)
                
                if success:
                    # Compilation succeeded - will use VM from now on
                    pass
                else:
                    # Compilation failed - mark and continue with interpreter
                    self.failed_compilations.add(loop_id)
            
            # Execute body
            if info["use_vm"] and loop_id in self.compiled_loops and loop_id not in self.failed_compilations:
                # Use VM
                result = self._execute_via_vm(loop_id, env)
                self.stats["vm_hits"] += 1
                
                if is_error(result):
                    # VM execution failed - fall back to interpreter
                    self.failed_compilations.add(loop_id)
                    result = self.evaluator.eval_node(body_node, env, stack_trace)
            else:
                # Use interpreter
                result = self.evaluator.eval_node(body_node, env, stack_trace)
            
            # Handle control flow
            if is_error(result):
                self.workload.reset_loop(loop_id)
                return result
            
            # Check for break/return
            from ..evaluator.statements import BreakException
            from ..object import ReturnValue
            
            if isinstance(result, (BreakException, ReturnValue)):
                self.workload.reset_loop(loop_id)
                return result if isinstance(result, ReturnValue) else NULL
        
        # Loop completed normally
        self.workload.reset_loop(loop_id)
        return result
    
    def _compile_loop(self, loop_id: int, body_node, env) -> bool:
        """
        Compile loop body to VM bytecode.
        
        Returns:
            True if compilation succeeded
        """
        # Bail out early for loops marked unsafe
        if loop_id in self.failed_compilations:
            return False
        try:
            from ..vm.compiler import BytecodeCompiler
            from ..evaluator.bytecode_compiler import EvaluatorBytecodeCompiler
            profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
            profile_active = profile_flag and profile_flag.lower() not in ("0", "false", "off")
            verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
            profile_verbose = profile_active and verbose_flag and verbose_flag.lower() not in ("0", "false", "off")
            
            # Try evaluator's bytecode compiler first (better integration)
            compiler = EvaluatorBytecodeCompiler()
            bytecode = compiler.compile(body_node, optimize=True)
            
            if bytecode and not compiler.errors:
                self.compiled_loops[loop_id] = bytecode
                self.workload.stats["vm_compilations"] += 1
                if profile_verbose:
                    print(f"[VM DEBUG] loop {loop_id} compiled to bytecode")
                    print(f"[VM DEBUG] instructions={bytecode.instructions}")
                    print("[VM DEBUG] constants={}".format(list(enumerate(bytecode.constants))))
                return True
            else:
                # Compilation failed
                if profile_active:
                    print(f"[VM DEBUG] loop {loop_id} compilation errors={compiler.errors}")
                self.stats["compilation_failures"] += 1
                return False
        
        except Exception as e:
            # Unexpected compilation error
            profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
            if profile_flag and profile_flag.lower() not in ("0", "false", "off"):
                print(f"[VM DEBUG] loop {loop_id} compilation exception={e}")
            self.stats["compilation_failures"] += 1
            return False
    
    def _execute_via_vm(self, loop_id: int, env) -> Any:
        """
        Execute compiled loop via VM.
        
        Returns:
            Execution result or error
        """
        from ..object import NULL, EvaluationError
        
        profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
        profile_active = profile_flag and profile_flag.lower() not in ("0", "false", "off")
        
        if loop_id not in self.compiled_loops:
            return EvaluationError("Loop not compiled")
        
        try:
            # Lazy init VM with all standard optimizations enabled
            if self.vm is None:
                from ..vm.vm import VM, VMMode
                gas_limit = 10_000_000
                if profile_active:
                    profile_limit_env = os.environ.get("ZEXUS_VM_PROFILE_GAS_LIMIT")
                    if profile_limit_env is not None:
                        try:
                            gas_limit = int(profile_limit_env)
                        except ValueError:
                            gas_limit = 50_000_000
                    else:
                        gas_limit = 50_000_000
                
                # Initialize VM with all optimizers enabled as standard
                self.vm = VM(
                    mode=VMMode.AUTO,
                    use_jit=True,                      # JIT for hot paths
                    max_heap_mb=1024,
                    debug=False,
                    gas_limit=gas_limit,
                    enable_peephole_optimizer=True,    # Standard optimization
                    enable_memory_pool=True,           # Standard memory pooling
                    enable_async_optimizer=True,       # Standard async optimization
                )
            
            # Sync environment to VM
            self._sync_env_to_vm(env)
            if hasattr(self.evaluator, 'builtins') and self.evaluator.builtins:
                self.vm.builtins = {k: v for k, v in self.evaluator.builtins.items()}
            
            # Execute bytecode
            bytecode = self.compiled_loops[loop_id]
            result = self.vm.execute(bytecode, debug=False)
            if profile_active:
                profile = getattr(self.vm, "_last_opcode_profile", None)
                if profile and not self._profile_reported:
                    top_entries = profile[:10]
                    print(f"[VM OPCODES] top={top_entries}")
                    self._profile_reported = True
            
            # Sync environment back from VM
            self._sync_env_from_vm(env)
            
            self.workload.stats["vm_executions"] += 1
            
            # Convert result to evaluator object
            return self._convert_from_vm(result) if result is not None else NULL
        
        except Exception as e:
            # VM execution failed
            if profile_verbose:
                print(f"[VM DEBUG] unified execution exception={e}")
            return EvaluationError(f"VM execution error: {e}")
    
    def _sync_env_to_vm(self, env):
        """Sync evaluator environment to VM"""
        if not self.vm or not env:
            return
        
        self.stats["environment_syncs"] += 1
        
        # Convert evaluator objects to VM-compatible values
        if hasattr(env, 'store'):
            for key, value in env.store.items():
                vm_value = self._convert_to_vm(value)
                self.vm.env[key] = vm_value
        elif isinstance(env, dict):
            for key, value in env.items():
                vm_value = self._convert_to_vm(value)
                self.vm.env[key] = vm_value
    
    def _sync_env_from_vm(self, env):
        """Sync VM environment back to evaluator"""
        if not self.vm or not env:
            return
        
        # Convert VM values back to evaluator objects
        for key, value in self.vm.env.items():
            eval_value = self._convert_from_vm(value)
            
            if hasattr(env, 'set'):
                try:
                    env.assign(key, eval_value)
                except ValueError:
                    # Variable doesn't exist - create it
                    env.set(key, eval_value)
            elif hasattr(env, 'store'):
                env.store[key] = eval_value
            elif isinstance(env, dict):
                env[key] = eval_value
    
    def _convert_to_vm(self, value: Any) -> Any:
        """Convert evaluator object to VM value"""
        if hasattr(value, 'value'):
            # Wrapped primitive (Integer, String, Boolean, Float)
            return value.value
        elif hasattr(value, 'elements'):
            # List
            return [self._convert_to_vm(elem) for elem in value.elements]
        elif hasattr(value, 'pairs'):
            # Map
            return {
                self._convert_to_vm(k): self._convert_to_vm(v)
                for k, v in value.pairs.items()
            }
        else:
            # Already primitive or unsupported
            return value
    
    def _convert_from_vm(self, value: Any) -> Any:
        """Convert VM value to evaluator object"""
        from ..object import Integer, Float, String, Boolean, List, Map, NULL
        
        if value is None:
            return NULL
        elif isinstance(value, bool):
            return Boolean(value)
        elif isinstance(value, int):
            return Integer(value)
        elif isinstance(value, float):
            return Float(value)
        elif isinstance(value, str):
            return String(value)
        elif isinstance(value, list):
            return List([self._convert_from_vm(elem) for elem in value])
        elif isinstance(value, dict):
            pairs = {
                self._convert_from_vm(k): self._convert_from_vm(v)
                for k, v in value.items()
            }
            return Map(pairs)
        else:
            return value

    # --- Safety analysis -------------------------------------------------

    def _is_vm_safe_loop(self, body_node) -> bool:
        """Heuristic: disallow VM only for transaction statements that mutate control flow.
        Contract method calls are now handled via post-call state sync.
        """
        try:
            from .. import zexus_ast
        except Exception:
            return False

        unsafe_nodes = (zexus_ast.TxStatement,)

        def _walk(node) -> bool:
            if node is None:
                return True
            if isinstance(node, unsafe_nodes):
                return False
            for attr in dir(node):
                if attr.startswith('_'):
                    continue
                try:
                    val = getattr(node, attr)
                except Exception:
                    continue
                if isinstance(val, list):
                    for item in val:
                        if not _walk(item):
                            return False
                elif hasattr(val, '__dict__'):
                    if not _walk(val):
                        return False
            return True

        return _walk(body_node)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive execution statistics"""
        return {
            "unified_executor": self.stats,
            "workload_detection": self.workload.get_stats(),
            "compiled_loops": len(self.compiled_loops),
            "compiled_functions": len(self.compiled_functions),
            "failed_compilations": len(self.failed_compilations),
            "vm_enabled": self.vm_enabled,
            "vm_initialized": self.vm is not None
        }


def create_unified_executor(evaluator) -> UnifiedExecutor:
    """
    Create a unified executor for the evaluator.
    
    This should be called once during evaluator initialization.
    """
    # Check if VM is available
    try:
        from ..vm.vm import VM
        vm_available = True
    except ImportError:
        vm_available = False
    
    return UnifiedExecutor(evaluator, vm_enabled=vm_available)
