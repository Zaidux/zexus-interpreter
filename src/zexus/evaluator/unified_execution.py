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
        
        # Workload classification thresholds (tuned for faster VM promotion)
        self.vm_threshold = 120       # Iterations before VM compilation
        self.jit_threshold = 1200     # Iterations before JIT compilation
        self.parallel_threshold = 6000  # Iterations before parallel execution
        
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
        self._force_vm_loops: set[int] = set()  # loop_ids promoted immediately
        self._loop_sync_keys: Dict[int, set[str]] = {}
        self.force_all_vm_loops = False
        self._compile_errors: Dict[int, Any] = {}
        self._compile_info: Dict[int, Any] = {}
        self._logged_vm_loops: set[int] = set()
        self._logged_vm_loop_env: set[int] = set()
        
        # VM instance (lazy init)
        self.vm = None
        self._profile_reported = False

        # VM configuration overrides (applied on next VM init)
        self.vm_config: Dict[str, Any] = {
            "mode": "auto",
            "use_jit": True,
            "max_heap_mb": 1024,
            "debug": False,
            "use_memory_manager": False,
            "gc_threshold": 1000,
            "enable_gas_metering": True,
            "enable_peephole_optimizer": True,
            "enable_bytecode_optimizer": False,
            "optimizer_level": 2,
            "enable_memory_pool": True,
            "enable_async_optimizer": True,
            "enable_profiling": False,
            "profiling_level": "DETAILED",
            "profiling_sample_rate": 1.0,
            "profiling_max_samples": 2048,
            "profiling_track_overhead": False,
            "enable_ssa": False,
            "enable_register_allocation": False,
            "enable_bytecode_converter": False,
            "converter_aggressive": False,
            "vm_full_loop": True,
            "vm_sync_all": False,
            "vm_allow_unsafe_loops": False,
            "vm_dump_bytecode": False,
            "vm_single_shot": False,
            "vm_single_shot_min_instructions": 24,
            "fast_single_shot": False,
            "single_shot_max_instructions": 64,
            "vm_action_cache": False,
            "vm_action_sync_all": False,
        }
        
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

    def configure_vm(self, config: Dict[str, Any]) -> None:
        """Apply VM configuration overrides for future VM initialization."""
        if not isinstance(config, dict):
            return
        for key, value in config.items():
            self.vm_config[key] = value
        # Reset VM so config changes take effect
        self.vm = None

    def reset_vm(self) -> None:
        """Drop the cached VM instance to force reinitialization."""
        self.vm = None

    def set_force_vm_loops(self, flag: bool) -> None:
        """Force all loops to attempt VM compilation/execution (debug/profiling)."""
        self.force_all_vm_loops = bool(flag)

    def ensure_vm(self, profile_active: bool = False) -> None:
        """Ensure a VM instance is initialized with current configuration."""
        if self.vm is not None:
            return
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

        mode_value = self.vm_config.get("mode", "auto")
        if isinstance(mode_value, str):
            mode = getattr(VMMode, mode_value.upper(), VMMode.AUTO)
        else:
            mode = mode_value
        self.vm = VM(
            mode=mode,
            use_jit=bool(self.vm_config.get("use_jit", True)),
            max_heap_mb=int(self.vm_config.get("max_heap_mb", 1024)),
            debug=bool(self.vm_config.get("debug", False)),
            use_memory_manager=bool(self.vm_config.get("use_memory_manager", False)),
            gc_threshold=int(self.vm_config.get("gc_threshold", 1000)),
            gas_limit=gas_limit,
            enable_gas_metering=bool(self.vm_config.get("enable_gas_metering", True)),
            enable_peephole_optimizer=bool(self.vm_config.get("enable_peephole_optimizer", True)),
            enable_bytecode_optimizer=bool(self.vm_config.get("enable_bytecode_optimizer", False)),
            optimizer_level=int(self.vm_config.get("optimizer_level", 2)),
            enable_memory_pool=bool(self.vm_config.get("enable_memory_pool", True)),
            enable_async_optimizer=bool(self.vm_config.get("enable_async_optimizer", True)),
            enable_profiling=bool(self.vm_config.get("enable_profiling", False)),
            profiling_level=str(self.vm_config.get("profiling_level", "DETAILED")),
            profiling_sample_rate=float(self.vm_config.get("profiling_sample_rate", 1.0)),
            profiling_max_samples=int(self.vm_config.get("profiling_max_samples", 2048)),
            profiling_track_overhead=bool(self.vm_config.get("profiling_track_overhead", False)),
            enable_ssa=bool(self.vm_config.get("enable_ssa", False)),
            enable_register_allocation=bool(self.vm_config.get("enable_register_allocation", False)),
            enable_bytecode_converter=bool(self.vm_config.get("enable_bytecode_converter", False)),
            converter_aggressive=bool(self.vm_config.get("converter_aggressive", False)),
            fast_single_shot=bool(self.vm_config.get("fast_single_shot", False)),
            single_shot_max_instructions=int(self.vm_config.get("single_shot_max_instructions", 64)),
        )
    
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
        from ..evaluator.utils import is_error, is_truthy
        
        self.stats["total_executions"] += 1
        result = NULL

        # If the loop body contains constructs the VM cannot safely handle yet
        # (e.g., transaction blocks or break/continue), pin this loop to interpreter only.
        allow_unsafe = bool(self.vm_config.get("vm_allow_unsafe_loops", False))
        if self.vm_enabled and not self.force_all_vm_loops and not allow_unsafe and not self._is_vm_safe_loop(body_node):
            self.failed_compilations.add(loop_id)
            self._compile_errors[loop_id] = "unsafe loop (control-flow or tx statement)"

        # Fast-path: promote obviously hot loops before iteration threshold
        force_vm_loop = loop_id in self._force_vm_loops or self.force_all_vm_loops
        if (
            self.vm_enabled
            and loop_id not in self.failed_compilations
            and loop_id not in self.compiled_loops
            and not force_vm_loop
        ):
            try:
                from .bytecode_compiler import should_use_vm_for_node
                from .. import zexus_ast
                loop_node = zexus_ast.WhileStatement(condition_node, body_node)

                if should_use_vm_for_node(body_node):
                    full_loop_mode = bool(self.vm_config.get("vm_full_loop", True))
                    promoted = self._compile_loop(loop_id, loop_node, env, full_loop=full_loop_mode)
                    if promoted:
                        self._force_vm_loops.add(loop_id)
                        force_vm_loop = True
            except ImportError:
                force_vm_loop = loop_id in self._force_vm_loops
        
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
            force_vm_loop = loop_id in self._force_vm_loops or force_vm_loop or self.force_all_vm_loops
            single_shot_active = False

            # Promote to parallel VM mode for very hot loops
            if (
                self.vm_enabled
                and info["iteration"] == self.workload.parallel_threshold
            ):
                self.configure_vm({"mode": "parallel"})
            
            # Check condition
            cond = self.evaluator.eval_node(condition_node, env, stack_trace)
            if is_error(cond):
                self.workload.reset_loop(loop_id)
                return cond
            
            if not is_truthy(cond):
                # Loop done
                break
            
            # Decide execution method
            if (
                info["should_compile"]
                and self.vm_enabled
                and loop_id not in self.compiled_loops
                and loop_id not in self.failed_compilations
            ):
                # Just hit threshold - compile now
                try:
                    from .. import zexus_ast
                    loop_node = zexus_ast.WhileStatement(condition_node, body_node)
                except Exception:
                    loop_node = body_node
                full_loop_mode = bool(self.vm_config.get("vm_full_loop", True))
                success = self._compile_loop(loop_id, loop_node, env, full_loop=full_loop_mode)
                
                if success:
                    # Compilation succeeded - will use VM from now on
                    pass
                else:
                    # Compilation failed - mark and continue with interpreter
                    self.failed_compilations.add(loop_id)
                    if loop_id not in self._compile_errors:
                        self._compile_errors[loop_id] = "compile returned false"

            # Single-shot VM: allow VM execution for one-iteration loops
            if (
                not info["use_vm"]
                and self.vm_enabled
                and bool(self.vm_config.get("vm_single_shot", False))
                and info["iteration"] == 1
                and loop_id not in self.compiled_loops
                and loop_id not in self.failed_compilations
            ):
                try:
                    from .bytecode_compiler import should_use_vm_for_node
                    if should_use_vm_for_node(body_node):
                        success = self._compile_loop(loop_id, body_node, env, full_loop=False)
                        if success:
                            instr_count = self._compile_info.get(loop_id, {}).get("instructions", 0)
                            min_instr = int(self.vm_config.get("vm_single_shot_min_instructions", 24))
                            if instr_count >= min_instr:
                                single_shot_active = True
                            else:
                                # Too small to justify VM overhead
                                self.compiled_loops.pop(loop_id, None)
                except Exception:
                    pass
            
            # Execute body
            use_vm_now = (
                loop_id in self.compiled_loops
                and loop_id not in self.failed_compilations
                and (force_vm_loop or info["use_vm"] or single_shot_active)
            )

            if use_vm_now:
                sync_keys = None
                if not bool(self.vm_config.get("vm_sync_all", False)):
                    sync_keys = self._loop_sync_keys.get(loop_id)
                    if sync_keys is None:
                        sync_keys = self._collect_assigned_names(body_node)
                        self._loop_sync_keys[loop_id] = sync_keys
                verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
                if verbose_flag and verbose_flag.lower() not in ("0", "false", "off"):
                    if loop_id not in self._logged_vm_loops:
                        info = self._compile_info.get(loop_id, {})
                        instr_count = info.get("instructions", "?")
                        print(f"[VM TRACE] executing loop_id={loop_id} instr={instr_count}")
                        self._logged_vm_loops.add(loop_id)
                entry = self.compiled_loops[loop_id]
                is_full_loop = isinstance(entry, dict) and entry.get("full_loop") is True
                # Use VM
                result = self._execute_via_vm(loop_id, env, sync_keys=sync_keys)
                self.stats["vm_hits"] += 1
                if is_error(result):
                    # VM execution failed - fall back to interpreter
                    self.failed_compilations.add(loop_id)
                    self._compile_errors[loop_id] = str(result)
                    result = self.evaluator.eval_node(body_node, env, stack_trace)
                elif is_full_loop:
                    # Full loop executed in VM, return immediately
                    self.workload.reset_loop(loop_id)
                    return result
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
        return result
    
    def _compile_loop(self, loop_id: int, node_to_compile, env, full_loop: bool = False) -> bool:
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
            compiler = EvaluatorBytecodeCompiler(use_cache=True)
            bytecode = compiler.compile(node_to_compile, optimize=True)

            if bytecode and not compiler.errors:
                try:
                    instruction_count = len(bytecode.instructions) if hasattr(bytecode, "instructions") else 0
                    constant_count = len(bytecode.constants) if hasattr(bytecode, "constants") else 0
                    ops_preview = []
                    if hasattr(bytecode, "instructions"):
                        for instr in bytecode.instructions[:40]:
                            if instr is None:
                                continue
                            op = instr[0] if isinstance(instr, tuple) and len(instr) >= 1 else instr
                            op_name = op.name if hasattr(op, "name") else str(op)
                            ops_preview.append(op_name)
                except Exception:
                    instruction_count = 0
                    constant_count = 0
                    ops_preview = []
                self._compile_info[loop_id] = {
                    "instructions": instruction_count,
                    "constants": constant_count,
                    "full_loop": bool(full_loop),
                    "ops_preview": ops_preview
                }
                if bool(self.vm_config.get("vm_dump_bytecode", False)):
                    try:
                        dump_lines = []
                        dump_lines.append(f"Loop {loop_id} | full_loop={bool(full_loop)}")
                        dump_lines.append(f"Instructions: {instruction_count} | Constants: {constant_count}")
                        dump_lines.append("\nConstants:")
                        for i, const in enumerate(getattr(bytecode, "constants", [])):
                            dump_lines.append(f"  {i:04d}: {const!r}")
                        dump_lines.append("\nInstructions:")
                        for idx, instr in enumerate(getattr(bytecode, "instructions", [])):
                            if instr is None:
                                continue
                            if isinstance(instr, tuple) and len(instr) >= 2:
                                op = instr[0]
                                operand = instr[1]
                                op_name = op.name if hasattr(op, "name") else str(op)
                                dump_lines.append(f"  {idx:04d}: {op_name} {operand}")
                            else:
                                dump_lines.append(f"  {idx:04d}: {instr}")
                        dump_path = f"/tmp/zexus_vm_dump_{loop_id}.txt"
                        with open(dump_path, "w", encoding="utf-8") as handle:
                            handle.write("\n".join(dump_lines))
                    except Exception:
                        pass
                if full_loop:
                    self.compiled_loops[loop_id] = {
                        "bytecode": bytecode,
                        "full_loop": True
                    }
                else:
                    self.compiled_loops[loop_id] = bytecode
                self.workload.stats["vm_compilations"] += 1
                if profile_verbose:
                    print(f"[VM DEBUG] loop {loop_id} compiled to bytecode")
                    print(f"[VM DEBUG] instructions={bytecode.instructions}")
                    print("[VM DEBUG] constants={}".format(list(enumerate(bytecode.constants))))
                return True
            else:
                if profile_active:
                    print(f"[VM DEBUG] loop {loop_id} compilation errors={compiler.errors}")
                if compiler.errors:
                    self._compile_errors[loop_id] = [str(err) for err in compiler.errors]
                else:
                    self._compile_errors[loop_id] = "compile returned no bytecode"
                self.stats["compilation_failures"] += 1
                return False

        except Exception as e:
            profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
            if profile_flag and profile_flag.lower() not in ("0", "false", "off"):
                print(f"[VM DEBUG] loop {loop_id} compilation exception={e}")
            self._compile_errors[loop_id] = str(e)
            self.stats["compilation_failures"] += 1
            return False
    
    def _execute_via_vm(self, loop_id: int, env, sync_keys: Optional[set[str]] = None) -> Any:
        """
        Execute compiled loop via VM.
        
        Returns:
            Execution result or error
        """
        from ..object import NULL, EvaluationError
        
        profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
        profile_active = profile_flag and profile_flag.lower() not in ("0", "false", "off")
        verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
        profile_verbose = profile_active and verbose_flag and verbose_flag.lower() not in ("0", "false", "off")
        
        if loop_id not in self.compiled_loops:
            return EvaluationError("Loop not compiled")
        
        try:
            # Lazy init VM with all standard optimizations enabled
            self.ensure_vm(profile_active=profile_active)

            # Sync environment to VM
            self._sync_env_to_vm(env)
            if hasattr(self.evaluator, 'builtins') and self.evaluator.builtins:
                self.vm.builtins = {k: v for k, v in self.evaluator.builtins.items()}

            verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
            if verbose_flag and verbose_flag.lower() not in ("0", "false", "off"):
                if loop_id not in self._logged_vm_loop_env:
                    i_val = self.vm.env.get("i")
                    count_val = self.vm.env.get("PERF_TRANSACTION_COUNT")
                    use_bulk_all = self.vm.env.get("use_bulk_all")
                    use_fast_blocks = self.vm.env.get("use_fast_blocks")
                    blockchain_val = self.vm.env.get("blockchain")
                    if i_val is not None or count_val is not None or use_bulk_all is not None:
                        chain_type = type(blockchain_val).__name__ if blockchain_val is not None else None
                        print(
                            f"[VM TRACE] loop_id={loop_id} i={i_val} PERF_TRANSACTION_COUNT={count_val} "
                            f"use_bulk_all={use_bulk_all} use_fast_blocks={use_fast_blocks} blockchain={chain_type}"
                        )
                    self._logged_vm_loop_env.add(loop_id)

            # Execute bytecode
            entry = self.compiled_loops[loop_id]
            if isinstance(entry, dict):
                bytecode = entry.get("bytecode")
            else:
                bytecode = entry
            result = self.vm.execute(bytecode, debug=False)
            if profile_active:
                profile = getattr(self.vm, "_last_opcode_profile", None)
                if profile and not self._profile_reported:
                    top_entries = profile[:10]
                    print(f"[VM OPCODES] top={top_entries}")
                    self._profile_reported = True
            
            # Sync environment back from VM
            self._sync_env_from_vm(env, sync_keys=sync_keys)
            
            self.workload.stats["vm_executions"] += 1
            
            # Convert result to evaluator object
            return self._convert_from_vm(result) if result is not None else NULL
        
        except Exception as e:
            if profile_verbose:
                print(f"[VM DEBUG] unified execution exception={e}")
            return EvaluationError(f"VM execution error: {e}")

    def execute_action_bytecode(self, bytecode, env, sync_keys: Optional[set[str]] = None) -> Any:
        """Execute cached action bytecode via VM with env sync."""
        from ..object import NULL, EvaluationError

        profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
        profile_active = profile_flag and profile_flag.lower() not in ("0", "false", "off")
        verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
        profile_verbose = profile_active and verbose_flag and verbose_flag.lower() not in ("0", "false", "off")

        try:
            self.ensure_vm(profile_active=profile_active)
            self._sync_env_to_vm(env)
            if hasattr(self.evaluator, 'builtins') and self.evaluator.builtins:
                self.vm.builtins = {k: v for k, v in self.evaluator.builtins.items()}

            result = self.vm.execute(bytecode, debug=False)

            if sync_keys is None and not bool(self.vm_config.get("vm_action_sync_all", False)):
                sync_keys = None
            self._sync_env_from_vm(env, sync_keys=sync_keys)
            self.workload.stats["vm_executions"] += 1
            self.stats["vm_hits"] += 1
            return self._convert_from_vm(result) if result is not None else NULL
        except Exception as e:
            if profile_verbose:
                print(f"[VM DEBUG] action execution exception={e}")
            return EvaluationError(f"VM execution error: {e}")
    
    def _sync_env_to_vm(self, env):
        """Sync evaluator environment to VM"""
        if not self.vm or not env:
            return
        
        self.stats["environment_syncs"] += 1
        
        # Convert evaluator objects to VM-compatible values
        if hasattr(env, 'store'):
            scopes = []
            scope = env
            while scope is not None and hasattr(scope, 'store'):
                scopes.append(scope)
                scope = getattr(scope, 'outer', None)
            # Apply outer scopes first, then inner scopes override
            for scope in reversed(scopes):
                for key, value in scope.store.items():
                    vm_value = self._convert_to_vm(value)
                    self.vm.env[key] = vm_value
        elif isinstance(env, dict):
            for key, value in env.items():
                vm_value = self._convert_to_vm(value)
                self.vm.env[key] = vm_value
    
    def _sync_env_from_vm(self, env, sync_keys: Optional[set[str]] = None):
        """Sync VM environment back to evaluator"""
        if not self.vm or not env:
            return
        
        if sync_keys is not None:
            if len(sync_keys) == 0:
                return
            # Only sync tracked keys to reduce overhead
            for key in sync_keys:
                if key not in self.vm.env:
                    continue
                value = self.vm.env.get(key)
                eval_value = self._convert_from_vm(value)
                if hasattr(env, 'set'):
                    try:
                        env.assign(key, eval_value)
                    except ValueError:
                        env.set(key, eval_value)
                elif hasattr(env, 'store'):
                    env.store[key] = eval_value
                elif isinstance(env, dict):
                    env[key] = eval_value
            return

        # Convert VM values back to evaluator objects (full sync)
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
        from ..object import List as ZList, Map as ZMap
        if hasattr(value, "call_method") or hasattr(value, "get_attr"):
            return value
        if isinstance(value, (ZList, ZMap)):
            return value
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
        if isinstance(value, (List, Map)):
            return value
        
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

    def _collect_assigned_names(self, node) -> set[str]:
        """Collect variable names assigned within a node (for selective VM sync)."""
        try:
            from .. import zexus_ast
        except Exception:
            return set()

        assigned: set[str] = set()

        def _walk(n) -> None:
            if n is None:
                return
            if isinstance(n, zexus_ast.LetStatement) and hasattr(n, "name"):
                name = getattr(n.name, "value", None) or str(n.name)
                assigned.add(name)
            elif isinstance(n, zexus_ast.ConstStatement) and hasattr(n, "name"):
                name = getattr(n.name, "value", None) or str(n.name)
                assigned.add(name)
            elif isinstance(n, zexus_ast.MethodCallExpression):
                obj = getattr(n, "object", None)
                if isinstance(obj, zexus_ast.Identifier):
                    assigned.add(obj.value)
            elif isinstance(n, zexus_ast.CallExpression):
                func = getattr(n, "function", None)
                if isinstance(func, zexus_ast.PropertyAccessExpression):
                    obj = getattr(func, "object", None)
                    if isinstance(obj, zexus_ast.Identifier):
                        assigned.add(obj.value)
            elif isinstance(n, zexus_ast.AssignmentExpression):
                if isinstance(n.name, zexus_ast.Identifier):
                    assigned.add(n.name.value)
                elif isinstance(n.name, zexus_ast.PropertyAccessExpression):
                    obj = n.name.object
                    if isinstance(obj, zexus_ast.Identifier):
                        assigned.add(obj.value)
                elif isinstance(n.name, zexus_ast.IndexExpression):
                    left = n.name.left
                    if isinstance(left, zexus_ast.Identifier):
                        assigned.add(left.value)

            for attr in dir(n):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(n, attr)
                except Exception:
                    continue
                if isinstance(val, list):
                    for item in val:
                        if hasattr(item, "__dict__"):
                            _walk(item)
                elif hasattr(val, "__dict__"):
                    _walk(val)

        _walk(node)
        return assigned

    # --- Safety analysis -------------------------------------------------

    def _is_vm_safe_loop(self, body_node) -> bool:
        """Heuristic: disallow VM only for transaction statements that mutate control flow.
        Contract method calls are now handled via post-call state sync.
        """
        try:
            from .. import zexus_ast
        except Exception:
            return False

        unsafe_nodes = (zexus_ast.TxStatement, zexus_ast.BreakStatement, zexus_ast.ContinueStatement)

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
            "compile_errors": {str(k): v for k, v in self._compile_errors.items()},
            "compile_info": {str(k): v for k, v in self._compile_info.items()},
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
