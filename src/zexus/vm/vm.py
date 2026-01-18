"""
Integrated Extended VM for Zexus.

Capabilities:
 - Architecture: Stack, Register, and Parallel execution modes.
 - Compilation: Tiered compilation with JIT (Hot path detection).
        self._ensure_recursion_headroom()
 - Memory: Managed memory with Garbage Collection.
 - Formats: High-level ops list and Low-level Bytecode.
 - Features: Async primitives (SPAWN/AWAIT), Event System, Module Imports.
 - Blockchain: Ziver-Chain specific opcodes (Merkle, Hash, State, Gas).
"""

import os
import sys
import time
import asyncio
import threading
import importlib
import hashlib
import types
from typing import List, Any, Dict, Tuple, Optional, Union, Callable
from enum import Enum

from ..object import (
    Integer as ZInteger,
    Float as ZFloat,
    Boolean as ZBoolean,
    String as ZString,
    List as ZList,
    Map as ZMap,
    Null as ZNull,
    EvaluationError as ZEvaluationError,
)

# ==================== Backend / Optional Imports ====================

# JIT Compiler
try:
    from .jit import JITCompiler, ExecutionTier
    _JIT_AVAILABLE = True
except ImportError:
    _JIT_AVAILABLE = False
    JITCompiler = None
    ExecutionTier = Enum('ExecutionTier', ['INTERPRETED', 'BYTECODE', 'JIT_NATIVE'])

# Memory Manager
try:
    from .memory_manager import create_memory_manager
    _MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    _MEMORY_MANAGER_AVAILABLE = False

# Register VM (Phase 5)
try:
    from .register_vm import RegisterVM
    _REGISTER_VM_AVAILABLE = True
except ImportError:
    _REGISTER_VM_AVAILABLE = False

# Parallel VM (Phase 6)
try:
    from .parallel_vm import ParallelVM, ExecutionMode
    _PARALLEL_VM_AVAILABLE = True
except ImportError:
    _PARALLEL_VM_AVAILABLE = False

# Profiler (Phase 8)
try:
    from .profiler import InstructionProfiler, ProfilingLevel
    _PROFILER_AVAILABLE = True
except ImportError:
    _PROFILER_AVAILABLE = False
    InstructionProfiler = None
    ProfilingLevel = None

# Memory Pool (Phase 8)
try:
    from .memory_pool import IntegerPool, StringPool, ListPool
    _MEMORY_POOL_AVAILABLE = True
except ImportError:
    _MEMORY_POOL_AVAILABLE = False
    IntegerPool = None
    StringPool = None
    ListPool = None

# Peephole Optimizer (Phase 8)
try:
    from .peephole_optimizer import PeepholeOptimizer, OptimizationLevel
    _PEEPHOLE_OPTIMIZER_AVAILABLE = True
except ImportError:
    _PEEPHOLE_OPTIMIZER_AVAILABLE = False
    PeepholeOptimizer = None
    OptimizationLevel = None

# Bytecode Optimizer (Phase 8)
try:
    from .optimizer import BytecodeOptimizer
    _BYTECODE_OPTIMIZER_AVAILABLE = True
except ImportError:
    _BYTECODE_OPTIMIZER_AVAILABLE = False
    BytecodeOptimizer = None

# Cython fast-path (optional)
try:
    from . import fastops as _fastops
    _FASTOPS_AVAILABLE = True
except Exception:
    _FASTOPS_AVAILABLE = False
    _fastops = None

# Async Optimizer (Phase 8)
try:
    from .async_optimizer import AsyncOptimizer, AsyncOptimizationLevel
    _ASYNC_OPTIMIZER_AVAILABLE = True
except ImportError:
    _ASYNC_OPTIMIZER_AVAILABLE = False
    AsyncOptimizer = None
    AsyncOptimizationLevel = None

# SSA Converter & Register Allocator (Phase 8.5)
try:
    from .ssa_converter import SSAConverter, SSAProgram, destruct_ssa
    from .register_allocator import RegisterAllocator, compute_live_ranges, AllocationResult
    _SSA_AVAILABLE = True
except ImportError:
    _SSA_AVAILABLE = False
    SSAConverter = None
    RegisterAllocator = None

# Bytecode Converter (Stack -> Register)
try:
    from .bytecode_converter import BytecodeConverter
    _BYTECODE_CONVERTER_AVAILABLE = True
except ImportError:
    _BYTECODE_CONVERTER_AVAILABLE = False
    BytecodeConverter = None

# Renderer Backend
try:
    from ..renderer import backend as _BACKEND
    _BACKEND_AVAILABLE = True
except Exception:
    _BACKEND_AVAILABLE = False
    _BACKEND = None

# Gas Metering
try:
    from .gas_metering import GasMetering, OutOfGasError, OperationLimitExceededError
    _GAS_METERING_AVAILABLE = True
except ImportError:
    _GAS_METERING_AVAILABLE = False
    GasMetering = None
    OutOfGasError = None
    OperationLimitExceededError = None


# ==================== Core Definitions ====================

class VMMode(Enum):
    """Execution modes for the VM"""
    STACK = "stack"          # Stack-based execution (standard)
    REGISTER = "register"    # Register-based execution (optimized)
    PARALLEL = "parallel"    # Parallel execution (multi-core)
    AUTO = "auto"            # Automatically choose best mode

class Cell:
    """Mutable cell used for proper closure capture semantics"""
    def __init__(self, value):
        self.value = value
    
    def __repr__(self):
        return f"<Cell {self.value!r}>"


def _to_string_value(arg):
    if isinstance(arg, ZString):
        return arg.value
    if isinstance(arg, ZInteger):
        return str(arg.value)
    if isinstance(arg, ZFloat):
        return str(arg.value)
    if isinstance(arg, ZBoolean):
        return "true" if getattr(arg, "value", False) else "false"
    if isinstance(arg, ZList):
        try:
            return arg.inspect()
        except Exception:
            return str(arg)
    if isinstance(arg, ZMap):
        try:
            return arg.inspect()
        except Exception:
            return str(arg)
    if arg is None or isinstance(arg, ZNull):
        return "null"
    if isinstance(arg, bool):
        return "true" if arg else "false"
    if isinstance(arg, (int, float)):
        return str(arg)
    if hasattr(arg, "inspect") and callable(getattr(arg, "inspect")):
        try:
            return arg.inspect()
        except Exception:
            return str(arg)
    return str(arg)


def _fallback_string(args):
    if len(args) != 1:
        return ZEvaluationError("string() takes exactly 1 argument")
    return ZString(_to_string_value(args[0]))


def _fallback_int(args):
    if len(args) != 1:
        return ZEvaluationError("int() takes exactly 1 argument")
    value = args[0]
    if isinstance(value, ZInteger):
        return value
    if isinstance(value, ZFloat):
        return ZInteger(int(value.value))
    if isinstance(value, ZString):
        try:
            return ZInteger(int(value.value))
        except ValueError:
            return ZEvaluationError(f"Cannot convert '{value.value}' to integer")
    if isinstance(value, (int, float)):
        return ZInteger(int(value))
    return ZEvaluationError(f"int() not supported for type {type(value).__name__}")


def _fallback_float(args):
    if len(args) != 1:
        return ZEvaluationError("float() takes exactly 1 argument")
    value = args[0]
    if isinstance(value, ZFloat):
        return value
    if isinstance(value, ZInteger):
        return ZFloat(float(value.value))
    if isinstance(value, ZString):
        try:
            return ZFloat(float(value.value))
        except ValueError:
            return ZEvaluationError(f"Cannot convert '{value.value}' to float")
    if isinstance(value, (int, float)):
        return ZFloat(float(value))
    return ZEvaluationError(f"float() not supported for type {type(value).__name__}")


def _fallback_len(args):
    if len(args) != 1:
        return ZEvaluationError("len() takes exactly 1 argument")
    value = args[0]
    if isinstance(value, ZString):
        return ZInteger(len(value.value))
    if isinstance(value, ZList):
        return ZInteger(len(getattr(value, "elements", [])))
    if isinstance(value, ZMap):
        return ZInteger(len(getattr(value, "pairs", {})))
    if isinstance(value, str):
        return ZInteger(len(value))
    if isinstance(value, list):
        return ZInteger(len(value))
    if isinstance(value, dict):
        return ZInteger(len(value))
    return ZEvaluationError(f"len() not supported for type {type(value).__name__}")


def _fallback_type(args):
    if len(args) != 1:
        return ZEvaluationError("type() takes exactly 1 argument")
    value = args[0]
    if isinstance(value, ZInteger):
        return ZString("Integer")
    if isinstance(value, ZFloat):
        return ZString("Float")
    if isinstance(value, ZString):
        return ZString("String")
    if isinstance(value, ZBoolean):
        return ZString("Boolean")
    if isinstance(value, ZList):
        return ZString("List")
    if isinstance(value, ZMap):
        return ZString("Map")
    if value is None or isinstance(value, ZNull):
        return ZString("Null")
    return ZString(type(value).__name__)


_FALLBACK_BUILTINS = {
    "string": _fallback_string,
    "int": _fallback_int,
    "float": _fallback_float,
    "len": _fallback_len,
    "type": _fallback_type,
}


class VM:
    """
    Main Virtual Machine integrating advanced architecture with rich feature set.
    """
    
    def __init__(
        self,
        builtins: Dict[str, Any] = None,
        env: Dict[str, Any] = None,
        parent_env: Dict[str, Any] = None,
        use_jit: bool = True,
        jit_threshold: int = 100,
        use_memory_manager: bool = False,
        max_heap_mb: int = 100,
        gc_threshold: int = 1000,
        mode: VMMode = VMMode.AUTO,
        worker_count: int = None,
        chunk_size: int = 50,
        num_registers: int = 16,
        hybrid_mode: bool = True,
        debug: bool = False,
        enable_profiling: bool = False,
        profiling_level: str = "DETAILED",
        profiling_sample_rate: float = 1.0,
        profiling_max_samples: int = 2048,
        profiling_track_overhead: bool = False,
        enable_memory_pool: bool = True,
        pool_max_size: int = 1000,
        enable_peephole_optimizer: bool = True,
        enable_bytecode_optimizer: bool = False,
        optimizer_level: int = 2,
        optimization_level: str = "MODERATE",
        enable_async_optimizer: bool = True,
        async_optimization_level: str = "MODERATE",
        enable_ssa: bool = False,
        enable_register_allocation: bool = False,
        enable_bytecode_converter: bool = False,
        converter_aggressive: bool = False,
        fast_single_shot: bool = False,
        single_shot_max_instructions: int = 64,
        num_allocator_registers: int = 16,
        enable_gas_metering: bool = True,
        gas_limit: int = None,
        enable_timeout: bool = True
    ):
        """
        Initialize the enhanced VM.
        """
        # --- Environment Setup ---
        self.builtins = builtins or {}
        self.env = env or {}
        self._parent_env = parent_env
        self.debug = debug
        
        # --- Gas Metering (Security) ---
        self.enable_gas_metering = enable_gas_metering and _GAS_METERING_AVAILABLE
        self.gas_metering = None
        if self.enable_gas_metering:
            self.gas_metering = GasMetering(gas_limit=gas_limit, enable_timeout=enable_timeout)
            if debug:
                print(f"[VM] Gas metering enabled: limit={gas_limit or 'unlimited'}")
            profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
            if profile_flag and profile_flag.lower() not in ("0", "false", "off"):
                profile_max_ops = os.environ.get("ZEXUS_VM_PROFILE_OPS_MAX", "50000000")
                try:
                    parsed_max = int(profile_max_ops)
                except ValueError:
                    parsed_max = 50_000_000
                self.gas_metering.max_operations = max(parsed_max, self.gas_metering.max_operations)
                profile_gas_limit = os.environ.get("ZEXUS_VM_PROFILE_GAS_LIMIT")
                if profile_gas_limit is not None:
                    try:
                        parsed_limit = int(profile_gas_limit)
                        self.gas_metering.set_limit(parsed_limit)
                    except ValueError:
                        pass
        
        # --- State Tracking ---
        self._events: Dict[str, List[Any]] = {}  # Event registry
        self._tasks: Dict[str, asyncio.Task] = {} # Async tasks
        self._task_counter = 0
        self._closure_cells: Dict[str, Cell] = {} # Closure storage
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._mode_usage = {m.value: 0 for m in VMMode}
        self._last_opcode_profile = None
        self._call_method_trace_count = 0
        self._action_evaluator = None
        
        # --- JIT Compilation (Phase 2) ---
        self.use_jit = use_jit and _JIT_AVAILABLE
        self._jit_lock = None  # Thread lock for JIT compilation
        if self.use_jit:
            import threading
            self._jit_lock = threading.Lock()
            self.jit_compiler = JITCompiler(
                hot_threshold=jit_threshold,
                optimization_level=1,
                debug=debug
            )
            self._jit_execution_stats: Dict[str, List[float]] = {}
            self._execution_times: Dict[str, float] = {}
        else:
            self.jit_compiler = None
        
        # --- Memory Management (Phase 7) ---
        self.use_memory_manager = use_memory_manager and _MEMORY_MANAGER_AVAILABLE
        self.memory_manager = None
        self._managed_objects: Dict[str, int] = {}
        self._memory_lock = None  # Thread lock for memory operations
        if self.use_memory_manager:
            import threading
            self._memory_lock = threading.Lock()
            self.memory_manager = create_memory_manager(
                max_heap_mb=max_heap_mb,
                gc_threshold=gc_threshold
            )

        # --- Profiler (Phase 8) ---
        self.enable_profiling = enable_profiling and _PROFILER_AVAILABLE
        self.profiler = None
        if self.enable_profiling:
            try:
                level = getattr(ProfilingLevel, profiling_level, ProfilingLevel.DETAILED)
                self.profiler = InstructionProfiler(
                    level=level,
                    sample_rate=profiling_sample_rate,
                    max_samples=profiling_max_samples,
                    track_overhead=profiling_track_overhead
                )
                if debug:
                    print(f"[VM] Profiler enabled: {profiling_level}")
            except Exception as e:
                if debug:
                    print(f"[VM] Failed to enable profiler: {e}")
                self.enable_profiling = False

        # --- Memory Pool (Phase 8) ---
        self.enable_memory_pool = enable_memory_pool and _MEMORY_POOL_AVAILABLE
        self.integer_pool = None
        self.string_pool = None
        self.list_pool = None
        if self.enable_memory_pool:
            try:
                self.integer_pool = IntegerPool(max_size=pool_max_size)
                self.string_pool = StringPool(max_size=pool_max_size)
                self.list_pool = ListPool(max_pool_size=pool_max_size)
                if debug:
                    print(f"[VM] Memory pools enabled: max_size={pool_max_size}")
            except Exception as e:
                if debug:
                    print(f"[VM] Failed to enable memory pools: {e}")
                self.enable_memory_pool = False

        # --- Peephole Optimizer (Phase 8) ---
        self.enable_peephole_optimizer = enable_peephole_optimizer and _PEEPHOLE_OPTIMIZER_AVAILABLE
        self.peephole_optimizer = None
        if self.enable_peephole_optimizer:
            try:
                level = getattr(OptimizationLevel, optimization_level, OptimizationLevel.MODERATE)
                self.peephole_optimizer = PeepholeOptimizer(level=level)
                if debug:
                    print(f"[VM] Peephole optimizer enabled: {optimization_level}")
            except Exception as e:
                if debug:
                    print(f"[VM] Failed to enable peephole optimizer: {e}")
                self.enable_peephole_optimizer = False

        # --- Bytecode Optimizer (Phase 8) ---
        self.enable_bytecode_optimizer = enable_bytecode_optimizer and _BYTECODE_OPTIMIZER_AVAILABLE
        self.bytecode_optimizer = None
        if self.enable_bytecode_optimizer:
            try:
                level = max(0, min(3, int(optimizer_level)))
                self.bytecode_optimizer = BytecodeOptimizer(level=level, max_passes=5, debug=debug)
                if debug:
                    print(f"[VM] Bytecode optimizer enabled: level={level}")
            except Exception as e:
                if debug:
                    print(f"[VM] Failed to enable bytecode optimizer: {e}")
                self.enable_bytecode_optimizer = False

        # --- Async Optimizer (Phase 8) ---
        self.enable_async_optimizer = enable_async_optimizer and _ASYNC_OPTIMIZER_AVAILABLE
        self.async_optimizer = None
        if self.enable_async_optimizer:
            try:
                level = getattr(AsyncOptimizationLevel, async_optimization_level, AsyncOptimizationLevel.MODERATE)
                self.async_optimizer = AsyncOptimizer(level=level, pool_size=pool_max_size)
                if debug:
                    print(f"[VM] Async optimizer enabled: {async_optimization_level}")
            except Exception as e:
                if debug:
                    print(f"[VM] Failed to enable async optimizer: {e}")
                self.enable_async_optimizer = False

        # --- SSA Converter & Register Allocator (Phase 8.5) ---
        self.enable_ssa = enable_ssa and _SSA_AVAILABLE
        self.enable_register_allocation = enable_register_allocation and _SSA_AVAILABLE
        self.ssa_converter = None
        self.register_allocator = None
        self.enable_bytecode_converter = enable_bytecode_converter and _BYTECODE_CONVERTER_AVAILABLE
        self.bytecode_converter = None

        # --- Fast single-shot execution ---
        self.fast_single_shot = bool(fast_single_shot)
        try:
            self.single_shot_max_instructions = int(single_shot_max_instructions)
        except Exception:
            self.single_shot_max_instructions = 64
        
        if self.enable_ssa:
            try:
                self.ssa_converter = SSAConverter(optimize=True)
                if debug:
                    print("[VM] SSA converter enabled")
            except Exception as e:
                if debug:
                    print(f"[VM] Failed to enable SSA converter: {e}")
                self.enable_ssa = False
        
        if self.enable_register_allocation:
            try:
                self.register_allocator = RegisterAllocator(
                    num_registers=num_allocator_registers,
                    num_temp_registers=8
                )
                self._last_register_allocation = None
                if debug:
                    print(f"[VM] Register allocator enabled: {num_allocator_registers} registers")
            except Exception as e:
                if debug:
                    print(f"[VM] Failed to enable register allocator: {e}")
                self.enable_register_allocation = False
                self._last_register_allocation = None

        if self.enable_bytecode_converter:
            try:
                self.bytecode_converter = BytecodeConverter(
                    num_registers=num_registers,
                    aggressive=converter_aggressive,
                    debug=debug
                )
                if debug:
                    print(f"[VM] Bytecode converter enabled: aggressive={converter_aggressive}")
            except Exception as e:
                if debug:
                    print(f"[VM] Failed to enable bytecode converter: {e}")
                self.enable_bytecode_converter = False

        # --- Execution Mode Configuration ---
        self.mode = mode
        self.worker_count = worker_count
        self.chunk_size = chunk_size
        self.num_registers = num_registers
        self.hybrid_mode = hybrid_mode
        
        # Initialize specialized VMs
        self._register_vm = None
        self._parallel_vm = None
        
        if _REGISTER_VM_AVAILABLE and (mode == VMMode.REGISTER or mode == VMMode.AUTO):
            self._register_vm = RegisterVM(
                num_registers=num_registers,
                hybrid_mode=hybrid_mode
            )
        
        if _PARALLEL_VM_AVAILABLE and (mode == VMMode.PARALLEL or mode == VMMode.AUTO):
            self._parallel_vm = ParallelVM(
                worker_count=worker_count or self._get_cpu_count(),
                chunk_size=chunk_size
            )

        if debug:
            print(f"[VM] Initialized | Mode: {mode.value} | JIT: {self.use_jit} | MemMgr: {self.use_memory_manager}")

    # ==================== VM <-> Evaluator Conversions ====================

    @staticmethod
    def _wrap_for_builtin(value: Any) -> Any:
        if isinstance(value, (ZInteger, ZFloat, ZString, ZBoolean, ZList, ZMap)) or value is None:
            return value
        if isinstance(value, bool):
            return ZBoolean(value)
        if isinstance(value, int):
            return ZInteger(value)
        if isinstance(value, float):
            return ZFloat(value)
        if isinstance(value, str):
            return ZString(value)
        if isinstance(value, list):
            return ZList([VM._wrap_for_builtin(elem) for elem in value])
        if isinstance(value, dict):
            pairs = {}
            for key, val in value.items():
                wrapped_key = VM._wrap_for_builtin(key)
                wrapped_val = VM._wrap_for_builtin(val)
                pairs[wrapped_key] = wrapped_val
            return ZMap(pairs)
        return value

    @staticmethod
    def _unwrap_after_builtin(value: Any) -> Any:
        if isinstance(value, (ZInteger, ZFloat, ZBoolean, ZString)):
            return value.value
        if isinstance(value, ZNull):
            return None
        if isinstance(value, ZList):
            return [VM._unwrap_after_builtin(elem) for elem in value.elements]
        if isinstance(value, ZMap):
            return {
                VM._unwrap_after_builtin(k): VM._unwrap_after_builtin(v)
                for k, v in value.pairs.items()
            }
        # Fallback for simple wrapper objects that expose a value attr
        if hasattr(value, "value") and not callable(getattr(value, "value")):
            return value.value
        return value

    def _get_cpu_count(self) -> int:
        import os
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1

    # ==================== Public Execution API ====================

    def _run_coroutine_sync(self, coro):
        """Run a coroutine from sync code, even if an event loop is already running."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, Exception] = {}

        def _runner():
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                result_holder["result"] = loop.run_until_complete(coro)
            except Exception as exc:
                error_holder["error"] = exc
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("result")

    def execute(self, code: Union[List[Tuple], Any], debug: bool = False) -> Any:
        """
        Execute code (High-level ops or Bytecode) using optimal execution mode.
        Blocks until completion (wraps async execution).
        """
        start_time = time.perf_counter()
        self._execution_count += 1
        
        # Handle High-Level Ops (List format)
        if isinstance(code, list) and not hasattr(code, "instructions"):
            if debug or self.debug:
                print("[VM] Executing High-Level Ops")
            try:
                # Run purely async internally, execute blocks
                return self._run_coroutine_sync(self._run_high_level_ops(code, debug or self.debug))
            except Exception as e:
                if debug or self.debug: print(f"[VM HL Error] {e}")
                raise e

        # Handle Low-Level Bytecode (Bytecode Object)
        try:
            execution_mode = self._select_execution_mode(code)
            self._mode_usage[execution_mode.value] += 1
            
            if debug or self.debug:
                print(f"[VM] Executing Bytecode | Mode: {execution_mode.value}")
            
            # 1. Register Mode (Optimized)
            if execution_mode == VMMode.REGISTER and self._register_vm:
                result = self._execute_register(code, debug)
            
            # 2. Parallel Mode (Multi-core)
            elif execution_mode == VMMode.PARALLEL and self._parallel_vm:
                result = self._execute_parallel(code, debug)
            
            # 3. Fast synchronous path for performance mode (no async overhead)
            elif getattr(self, '_perf_fast_dispatch', False):
                result = self._run_stack_bytecode_sync(code, debug)
            
            # 4. Stack Mode (Standard/Fallback + Async Support)
            else:
                result = self._run_coroutine_sync(self._execute_stack(code, debug))
            
            # JIT Tracking
            if self.use_jit and hasattr(code, 'instructions'):
                execution_time = time.perf_counter() - start_time
                self._track_execution_for_jit(code, execution_time, execution_mode)
            
            return result
            
        finally:
            self._total_execution_time += (time.perf_counter() - start_time)

    def _select_execution_mode(self, code) -> VMMode:
        if self.mode != VMMode.AUTO:
            return self.mode
        
        if self.use_jit:
            return VMMode.STACK

        if hasattr(code, 'instructions'):
            instructions = code.instructions
            if self._parallel_vm and self._is_parallelizable(instructions):
                return VMMode.PARALLEL
            if self._register_vm and self._is_register_friendly(instructions):
                return VMMode.REGISTER
        
        return VMMode.STACK

    # ==================== Specialized Execution Methods ====================

    async def _execute_stack(self, code, debug: bool = False):
        """Async wrapper for the core stack VM"""
        if hasattr(code, "instructions"):
            return await self._run_stack_bytecode(code, debug)
        return None

    def _execute_register(self, bytecode, debug: bool = False):
        """Execute using register-based VM"""
        try:
            if self.enable_bytecode_converter and self.bytecode_converter and hasattr(bytecode, "instructions"):
                try:
                    if not bytecode.metadata.get("converted_to_register"):
                        bytecode = self.bytecode_converter.convert(bytecode)
                except Exception:
                    pass
            if self.enable_register_allocation and self.register_allocator and hasattr(bytecode, "instructions"):
                try:
                    self._last_register_allocation = self.allocate_registers(bytecode.instructions)
                except Exception:
                    self._last_register_allocation = None
            # Ensure register VM has current environment and builtins
            self._register_vm.env = self.env.copy()
            self._register_vm.builtins = self.builtins.copy()
            if hasattr(self._register_vm, '_parent_env'):
                self._register_vm._parent_env = self._parent_env
            
            result = self._register_vm.execute(bytecode)
            
            # Sync back environment changes
            self.env.update(self._register_vm.env)
            
            return result
        except Exception as e:
            if debug: print(f"[VM Register] Failed: {e}, falling back to stack")
            return self._run_coroutine_sync(self._run_stack_bytecode(bytecode, debug))

    def _execute_parallel(self, bytecode, debug: bool = False):
        """Execute using parallel VM"""
        try:
            optimized_bytecode = self._optimize_bytecode_for_parallel(bytecode)
            return self._parallel_vm.execute(
                optimized_bytecode,
                initial_state={
                    "env": self.env.copy(),
                    "builtins": self.builtins.copy(),
                    "parent_env": self._parent_env,
                },
            )
        except Exception as e:
            if debug: print(f"[VM Parallel] Failed: {e}, falling back to stack")
            return self._run_coroutine_sync(self._run_stack_bytecode(bytecode, debug))

    def _optimize_bytecode_for_parallel(self, bytecode):
        """Apply peephole/SSA optimizations and map opcodes for parallel execution."""
        from .bytecode import Bytecode, Opcode

        consts = list(getattr(bytecode, "constants", []))
        instrs = list(getattr(bytecode, "instructions", []))

        if self.enable_bytecode_optimizer and self.bytecode_optimizer:
            try:
                normalized_for_opt: List[Tuple[Any, Any]] = []
                for instr in instrs:
                    if instr is None:
                        continue
                    if isinstance(instr, tuple) and len(instr) >= 2:
                        op = instr[0]
                        operand = instr[1] if len(instr) == 2 else tuple(instr[1:])
                        op_name = op.name if hasattr(op, "name") else op
                        normalized_for_opt.append((str(op_name), operand))
                instrs = self.bytecode_optimizer.optimize(normalized_for_opt, consts)
            except Exception:
                pass

        if self.enable_peephole_optimizer and self.peephole_optimizer:
            try:
                instrs, consts = self.peephole_optimizer.optimize_bytecode(instrs, consts)
            except Exception:
                pass

        normalized: List[Tuple[Any, Any]] = []
        for instr in instrs:
            if instr is None:
                continue
            if isinstance(instr, tuple) and len(instr) >= 2:
                op = instr[0]
                operand = instr[1] if len(instr) == 2 else tuple(instr[1:])
                op_name = op.name if hasattr(op, "name") else op
                normalized.append((op_name, operand))

        instrs = normalized

        if self.enable_ssa and self.ssa_converter:
            try:
                ssa_program = self.ssa_converter.convert_to_ssa(instrs)
                ssa_instrs = destruct_ssa(ssa_program)
                instrs, consts = self._normalize_ssa_instructions(ssa_instrs, consts)
            except Exception:
                pass

        mapped: List[Tuple[Any, Any]] = []
        for op, operand in instrs:
            if isinstance(op, str) and op in Opcode.__members__:
                mapped.append((Opcode[op], operand))
            else:
                mapped.append((op, operand))

        return Bytecode(instructions=mapped, constants=consts)

    # ==================== JIT & Optimization Heuristics ====================

    def _is_parallelizable(self, instructions) -> bool:
        if len(instructions) < 100: return False
        independent_ops = sum(1 for op, _ in instructions if op in ['LOAD_CONST', 'ADD', 'SUB', 'MUL', 'HASH_BLOCK'])
        return independent_ops / len(instructions) > 0.3

    def _is_register_friendly(self, instructions) -> bool:
        arith_ops = sum(1 for op, _ in instructions if op in ['ADD', 'SUB', 'MUL', 'DIV', 'EQ', 'LT'])
        return arith_ops / max(len(instructions), 1) > 0.4

    def _track_execution_for_jit(self, bytecode, execution_time: float, execution_mode: VMMode):
        if not self.use_jit or not self.jit_compiler: return
        
        with self._jit_lock:
            hot_path_info = self.jit_compiler.track_execution(bytecode, execution_time)
            bytecode_hash = getattr(hot_path_info, 'bytecode_hash', None) or self.jit_compiler._hash_bytecode(bytecode)
            
            if bytecode_hash not in self._jit_execution_stats:
                self._jit_execution_stats[bytecode_hash] = []
            self._jit_execution_stats[bytecode_hash].append(execution_time)
            
            # Check if should compile (outside lock to avoid holding during compilation)
            should_compile = self.jit_compiler.should_compile(bytecode_hash)
        
        # Compile outside the lock to prevent blocking other executions
        if should_compile:
            if self.debug: print(f"[VM JIT] Compiling hot path: {bytecode_hash[:8]}")
            with self._jit_lock:
                # Double-check it hasn't been compiled by another thread
                if self.jit_compiler.should_compile(bytecode_hash):
                    self.jit_compiler.compile_hot_path(bytecode)

    def _normalize_ssa_instructions(self, instructions: List[Tuple], consts: List[Any]) -> Tuple[List[Tuple], List[Any]]:
        """Normalize SSA-destructed instructions to (opcode, operand) format."""
        def _const_index(value: Any) -> int:
            for i, const in enumerate(consts):
                if const == value and type(const) == type(value):
                    return i
            consts.append(value)
            return len(consts) - 1

        normalized: List[Tuple] = []
        for instr in instructions:
            if instr is None:
                continue
            if not isinstance(instr, tuple):
                continue
            op = instr[0]
            if len(instr) == 2:
                normalized.append((op, instr[1]))
                continue
            if op == "MOVE" and len(instr) >= 3:
                src = instr[1]
                dest = instr[2]
                src_idx = _const_index(src)
                dest_idx = _const_index(dest)
                normalized.append(("LOAD_NAME", src_idx))
                normalized.append(("STORE_NAME", dest_idx))
                continue

            operand = tuple(instr[1:])
            normalized.append((op, operand))

        return normalized, consts

    def get_jit_stats(self) -> Dict[str, Any]:
        if self.use_jit and self.jit_compiler:
            stats = self.jit_compiler.get_stats()
            stats['vm_hot_paths_tracked'] = len(self._jit_execution_stats)
            stats['jit_enabled'] = True
            return stats
        return {'jit_enabled': False}

    def _ensure_recursion_headroom(self, minimum: int = 5000):
        try:
            current = sys.getrecursionlimit()
            if current < minimum:
                sys.setrecursionlimit(minimum)
        except Exception:
            pass

    def clear_jit_cache(self):
        if self.use_jit and self.jit_compiler:
            with self._jit_lock:
                self.jit_compiler.clear_cache()
                self._jit_execution_stats.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive VM statistics"""
        stats = {
            'execution_count': self._execution_count,
            'total_execution_time': self._total_execution_time,
            'mode_usage': self._mode_usage.copy(),
            'jit_enabled': self.use_jit,
            'memory_manager_enabled': self.use_memory_manager
        }
        
        if self.use_jit:
            stats['jit_stats'] = self.get_jit_stats()
        
        if self.use_memory_manager:
            stats['memory_stats'] = self.get_memory_stats()
        
        return stats

    # ==================== Memory Management API ====================

    def get_memory_stats(self) -> Dict[str, Any]:
        if self.use_memory_manager and self.memory_manager:
            with self._memory_lock:
                stats = self.memory_manager.get_stats()
                stats['managed_objects_count'] = len(self._managed_objects)
                stats['memory_manager_enabled'] = True
            return stats
        return {'memory_manager_enabled': False}
    
    def get_memory_report(self) -> str:
        """Get detailed memory report"""
        if self.use_memory_manager and self.memory_manager:
            stats = self.get_memory_stats()
            report = f"Memory Manager Report:\n"
            report += f"  Managed Objects: {stats.get('managed_objects_count', 0)}\n"
            report += f"  Total Allocations: {stats.get('total_allocations', 0)}\n"
            report += f"  Active Objects: {stats.get('active_objects', 0)}\n"
            return report
        return "Memory manager disabled"

    def collect_garbage(self, force: bool = False) -> Dict[str, Any]:
        if self.use_memory_manager and self.memory_manager:
            collected, gc_time = self.memory_manager.collect_garbage(force=force)
            # Cleanup local references to collected objects
            collected_ids = getattr(self.memory_manager, '_last_collected_ids', set())
            for name, obj_id in list(self._managed_objects.items()):
                if obj_id in collected_ids:
                    del self._managed_objects[name]
            return {'collected': collected, 'gc_time': gc_time}
        
        # Fallback: Manual environment cleanup for non-managed memory
        # Clear variables that are no longer referenced
        if force:
            initial_count = len(self.env)
            # Keep only builtins and parent env references
            keys_to_remove = []
            for key in list(self.env.keys()):
                # Don't remove special keys or builtins
                if not key.startswith('_') and key not in self.builtins:
                    keys_to_remove.append(key)
            
            # Remove temporary variables
            for key in keys_to_remove:
                del self.env[key]
            
            cleared = initial_count - len(self.env)
            return {'collected': cleared, 'message': 'Environment variables cleared'}
        
        return {'collected': 0, 'message': 'Memory manager disabled or not forced'}


    def _allocate_managed(self, value: Any, name: str = None, root: bool = False) -> int:
        if not self.use_memory_manager or not self.memory_manager: return -1
        try:
            with self._memory_lock:
                if name and name in self._managed_objects:
                    self.memory_manager.deallocate(self._managed_objects[name])
                obj_id = self.memory_manager.allocate(value, root=root)
                if name: self._managed_objects[name] = obj_id
                return obj_id
        except Exception:
            return -1

    def _get_managed(self, name: str) -> Any:
        if not self.use_memory_manager or not self.memory_manager: return None
        with self._memory_lock:
            obj_id = self._managed_objects.get(name)
            if obj_id is not None:
                return self.memory_manager.get(obj_id)
            return None

    # ==================== Core Execution: High-Level Ops ====================

    async def _run_high_level_ops(self, ops: List[Tuple], debug: bool = False):
        last = None
        for i, op in enumerate(ops):
            if not isinstance(op, (list, tuple)) or len(op) == 0: continue
            code = op[0]
            if debug: print(f"[VM HL] op#{i}: {op}")
            try:
                if code == "DEFINE_SCREEN":
                    _, name, props = op
                    if _BACKEND_AVAILABLE: _BACKEND.define_screen(name, props)
                    else: self.env.setdefault("screens", {})[name] = props
                    last = None
                elif code == "DEFINE_COMPONENT":
                    _, name, props = op
                    if _BACKEND_AVAILABLE: _BACKEND.define_component(name, props)
                    else: self.env.setdefault("components", {})[name] = props
                    last = None
                elif code == "DEFINE_THEME":
                    _, name, props = op
                    self.env.setdefault("themes", {})[name] = props
                elif code == "CALL_BUILTIN":
                    _, name, arg_ops = op
                    args = [self._eval_hl_op(a) for a in arg_ops]
                    last = await self._call_builtin_async(name, args)
                elif code == "LET":
                    _, name, val_op = op
                    val = self._eval_hl_op(val_op)
                    # If val is a coroutine, await it
                    if asyncio.iscoroutine(val) or isinstance(val, asyncio.Future):
                        val = await val
                    self.env[name] = val
                    last = None
                elif code == "EXPR":
                    _, expr_op = op
                    last = self._eval_hl_op(expr_op)
                    # If last is a coroutine, await it
                    if asyncio.iscoroutine(last) or isinstance(last, asyncio.Future):
                        last = await last
                elif code == "REGISTER_EVENT":
                    _, name, props = op
                    self._events.setdefault(name, [])
                elif code == "EMIT_EVENT":
                    _, name, payload_op = op
                    payload = self._eval_hl_op(payload_op)
                    handlers = self._events.get(name, [])
                    for h in handlers:
                        await self._call_builtin_async(h, [payload])
                elif code == "IMPORT":
                    _, module_path, alias = op
                    try:
                        mod = importlib.import_module(module_path)
                        self.env[alias or module_path] = mod
                    except Exception:
                        self.env[alias or module_path] = None
                elif code == "DEFINE_ENUM":
                    _, name, members = op
                    enum_registry = self.env.setdefault("enums", {})
                    enum_registry[name] = members
                    self.env[name] = members
                elif code == "DEFINE_PROTOCOL":
                    _, name, spec = op
                    self.env.setdefault("protocols", {})[name] = spec
                elif code == "AWAIT":
                    _, inner_op = op
                    evaluated = self._eval_hl_op(inner_op)
                    last = await evaluated if (asyncio.iscoroutine(evaluated) or isinstance(evaluated, asyncio.Future)) else evaluated
                else:
                    last = None
            except Exception as e:
                last = e
        return last

    def _eval_hl_op(self, op):
        if not isinstance(op, tuple): return op
        tag = op[0]
        if tag == "LITERAL": return op[1]
        if tag == "IDENT":
            name = op[1]
            if name in self.env: return self.env[name]
            if name in self.builtins: return self.builtins[name]
            return None
        if tag == "CALL_BUILTIN":
            name = op[1]; args = [self._eval_hl_op(a) for a in op[2]]
            # Return a coroutine instead of calling asyncio.run() - let caller handle await
            target = self.builtins.get(name) or self.env.get(name)
            if asyncio.iscoroutinefunction(target):
                return target(*args)
            elif callable(target):
                result = target(*args)
                if asyncio.iscoroutine(result):
                    return result
                return result
            return None
        if tag == "MAP": return {k: self._eval_hl_op(v) for k, v in op[1].items()}
        if tag == "LIST": return [self._eval_hl_op(e) for e in op[1]]
        return None

    # ==================== Fast Synchronous Dispatch (Performance Mode) ====================

    def _run_stack_bytecode_sync(self, bytecode, debug=False):
        """Synchronous fast-path execution without async overhead or gas metering."""
        consts = list(getattr(bytecode, "constants", []))
        instrs = list(getattr(bytecode, "instructions", []))
        
        # Normalize opcodes
        normalized: List[Tuple[str, Any]] = []
        for instr in instrs:
            if instr is None:
                continue
            if isinstance(instr, tuple) and len(instr) >= 2:
                op = instr[0]
                operand = instr[1] if len(instr) == 2 else tuple(instr[1:])
                op_name = op.name if hasattr(op, "name") else op
                normalized.append((op_name, operand))
        instrs = normalized

        # Cython fast-path if available
        if _FASTOPS_AVAILABLE:
            try:
                return _fastops.execute(instrs, consts, self.env, self.builtins, self._closure_cells)
            except NotImplementedError:
                pass
            except Exception:
                pass
        
        # Fast stack implementation
        stack: List[Any] = []
        stack_append = stack.append
        stack_pop = stack.pop
        
        ip = 0
        instr_count = len(instrs)
        env = self.env
        builtins = self.builtins
        
        def const(idx):
            return consts[idx] if isinstance(idx, int) and 0 <= idx < len(consts) else idx
        
        def resolve(name):
            if name in env:
                val = env[name]
                return val.value if isinstance(val, Cell) else val
            if name in self._closure_cells:
                return self._closure_cells[name].value
            return None
        
        def store(name, value):
            if name in env and isinstance(env[name], Cell):
                env[name].value = value
            else:
                env[name] = value
        
        while ip < instr_count:
            op_name, operand = instrs[ip]
            ip += 1
            
            # Hot path: arithmetic and stack ops (inlined)
            if op_name == "LOAD_CONST":
                stack_append(const(operand))
            elif op_name == "LOAD_NAME":
                stack_append(resolve(const(operand)))
            elif op_name == "STORE_NAME":
                store(const(operand), stack_pop() if stack else None)
            elif op_name == "POP":
                if stack: stack_pop()
            elif op_name == "DUP":
                if stack: stack_append(stack[-1])
            elif op_name == "ADD":
                b = stack_pop() if stack else 0
                a = stack_pop() if stack else 0
                if hasattr(a, 'value'): a = a.value
                if hasattr(b, 'value'): b = b.value
                stack_append(a + b)
            elif op_name == "SUB":
                b = stack_pop() if stack else 0
                a = stack_pop() if stack else 0
                if hasattr(a, 'value'): a = a.value
                if hasattr(b, 'value'): b = b.value
                stack_append(a - b)
            elif op_name == "MUL":
                b = stack_pop() if stack else 0
                a = stack_pop() if stack else 0
                if hasattr(a, 'value'): a = a.value
                if hasattr(b, 'value'): b = b.value
                stack_append(a * b)
            elif op_name == "DIV":
                b = stack_pop() if stack else 1
                a = stack_pop() if stack else 0
                if hasattr(a, 'value'): a = a.value
                if hasattr(b, 'value'): b = b.value
                stack_append(a / b if b != 0 else 0)
            elif op_name == "MOD":
                b = stack_pop() if stack else 1
                a = stack_pop() if stack else 0
                stack_append(a % b if b != 0 else 0)
            elif op_name == "EQ":
                b = stack_pop() if stack else None
                a = stack_pop() if stack else None
                stack_append(a == b)
            elif op_name == "NEQ":
                b = stack_pop() if stack else None
                a = stack_pop() if stack else None
                stack_append(a != b)
            elif op_name == "LT":
                b = stack_pop() if stack else 0
                a = stack_pop() if stack else 0
                stack_append(a < b)
            elif op_name == "GT":
                b = stack_pop() if stack else 0
                a = stack_pop() if stack else 0
                stack_append(a > b)
            elif op_name == "LTE":
                b = stack_pop() if stack else 0
                a = stack_pop() if stack else 0
                stack_append(a <= b)
            elif op_name == "GTE":
                b = stack_pop() if stack else 0
                a = stack_pop() if stack else 0
                stack_append(a >= b)
            elif op_name == "NOT":
                a = stack_pop() if stack else False
                stack_append(not a)
            elif op_name == "NEG":
                a = stack_pop() if stack else 0
                stack_append(-a)
            elif op_name == "JUMP":
                ip = operand
            elif op_name == "JUMP_IF_FALSE":
                cond = stack_pop() if stack else None
                if not cond:
                    ip = operand
            elif op_name == "RETURN":
                return stack_pop() if stack else None
            elif op_name == "BUILD_LIST":
                count = operand if operand is not None else 0
                elements = [stack_pop() for _ in range(count)][::-1]
                stack_append(elements)
            elif op_name == "BUILD_MAP":
                count = operand if operand is not None else 0
                result = {}
                for _ in range(count):
                    val = stack_pop()
                    key = stack_pop()
                    result[key] = val
                stack_append(result)
            elif op_name == "INDEX":
                idx = stack_pop()
                obj = stack_pop()
                try:
                    if isinstance(obj, ZList):
                        stack_append(obj.get(idx))
                    elif isinstance(obj, ZMap):
                        stack_append(obj.get(idx))
                    else:
                        stack_append(obj[idx] if obj is not None else None)
                except (IndexError, KeyError, TypeError):
                    stack_append(None)
            elif op_name == "GET_LENGTH":
                obj = stack_pop()
                try:
                    if obj is None:
                        stack_append(0)
                    elif isinstance(obj, ZList):
                        stack_append(len(obj.elements))
                    elif isinstance(obj, ZMap):
                        stack_append(len(obj.pairs))
                    elif hasattr(obj, '__len__'):
                        stack_append(len(obj))
                    else:
                        stack_append(0)
                except Exception:
                    stack_append(0)
            elif op_name == "CALL_NAME":
                name_idx, arg_count = operand
                func_name = const(name_idx)
                args = [stack_pop() for _ in range(arg_count)][::-1] if arg_count else []
                fn = resolve(func_name) or builtins.get(func_name)
                if fn is None:
                    res = self._call_fallback_builtin(func_name, args)
                else:
                    res = self._invoke_callable_sync(fn, args)
                stack_append(res)
            elif op_name == "CALL_TOP":
                arg_count = operand or 0
                args = [stack_pop() for _ in range(arg_count)][::-1] if arg_count else []
                fn_obj = stack_pop() if stack else None
                res = self._invoke_callable_sync(fn_obj, args)
                stack_append(res)
            elif op_name == "CALL_METHOD":
                if not operand:
                    stack_append(None)
                    continue
                method_idx, arg_count = operand
                args = [stack_pop() for _ in range(arg_count)][::-1] if arg_count else []
                target = stack_pop() if stack else None
                method_name = const(method_idx)
                if target is None:
                    stack_append(None)
                    continue
                result = None
                try:
                    if method_name == "set":
                        if isinstance(target, ZMap) and len(args) >= 2:
                            key = args[0]
                            if isinstance(key, str):
                                key = ZString(key)
                            result = target.set(key, args[1])
                        elif isinstance(target, ZList) and len(args) >= 2:
                            result = target.set(args[0], args[1])
                        elif isinstance(target, (dict, list)) and len(args) >= 2:
                            target[args[0]] = args[1]
                            result = args[1]
                    elif hasattr(target, "call_method"):
                        wrapped_args = [self._wrap_for_builtin(arg) for arg in args]
                        result = target.call_method(method_name, wrapped_args)
                    else:
                        attr = getattr(target, method_name, None)
                        if callable(attr):
                            result = attr(*args)
                        elif isinstance(target, dict) and method_name in target:
                            candidate = target[method_name]
                            result = candidate(*args) if callable(candidate) else candidate
                        else:
                            result = attr
                except Exception:
                    result = None
                stack_append(self._unwrap_after_builtin(result))
            elif op_name == "PRINT":
                val = stack_pop() if stack else None
                print(val)
            elif op_name == "GET_ATTR":
                attr = stack_pop() if stack else None
                obj = stack_pop() if stack else None
                if obj is None:
                    stack_append(None)
                else:
                    attr_name = attr.value if hasattr(attr, 'value') else attr
                    try:
                        if isinstance(obj, ZMap):
                            key = attr_name
                            if isinstance(key, str):
                                key = ZString(key)
                            stack_append(obj.get(key))
                        elif isinstance(obj, dict):
                            stack_append(obj.get(attr_name))
                        else:
                            stack_append(getattr(obj, attr_name, None))
                    except Exception:
                        stack_append(None)
            else:
                # Fallback to async path for unsupported ops
                return self._run_coroutine_sync(self._run_stack_bytecode(bytecode, debug))
        
        return stack_pop() if stack else None

    def _invoke_callable_sync(self, fn, args):
        """Synchronous callable invocation for fast dispatch."""
        if fn is None:
            return None
        if callable(fn) and not asyncio.iscoroutinefunction(fn):
            try:
                return fn(*args)
            except Exception:
                return None
        if isinstance(fn, dict):
            # Function descriptor - execute bytecode
            bytecode = fn.get("bytecode")
            if bytecode:
                params = fn.get("parameters", [])
                child_vm = VM(
                    builtins=self.builtins.copy(),
                    env={},
                    parent_env=self,
                    use_jit=False,
                    enable_gas_metering=False,
                    enable_peephole_optimizer=False,
                    enable_bytecode_optimizer=False,
                    enable_profiling=False,
                    enable_memory_pool=False,
                )
                child_vm._perf_fast_dispatch = True
                for i, p in enumerate(params):
                    pname = p.get("name") if isinstance(p, dict) else str(p)
                    child_vm.env[pname] = args[i] if i < len(args) else None
                return child_vm._run_stack_bytecode_sync(bytecode, debug=False)
        # Fallback for async callables
        if asyncio.iscoroutinefunction(fn):
            return self._run_coroutine_sync(fn(*args))
        return None

    # ==================== Core Execution: Stack Bytecode ====================

    async def _run_stack_bytecode(self, bytecode, debug=False):
        # 0. Optional bytecode optimizations (peephole, SSA)
        consts = list(getattr(bytecode, "constants", []))
        instrs = list(getattr(bytecode, "instructions", []))

        fast_single_shot = (
            self.fast_single_shot
            and isinstance(self.single_shot_max_instructions, int)
            and len(instrs) <= self.single_shot_max_instructions
        )

        if not fast_single_shot and self.enable_bytecode_optimizer and self.bytecode_optimizer:
            try:
                normalized_for_opt: List[Tuple[Any, Any]] = []
                for instr in instrs:
                    if instr is None:
                        continue
                    if isinstance(instr, tuple) and len(instr) >= 2:
                        op = instr[0]
                        operand = instr[1] if len(instr) == 2 else tuple(instr[1:])
                        op_name = op.name if hasattr(op, "name") else op
                        normalized_for_opt.append((str(op_name), operand))
                instrs = self.bytecode_optimizer.optimize(normalized_for_opt, consts)
            except Exception:
                pass

        # Peephole optimization with constant pool awareness
        if not fast_single_shot and self.enable_peephole_optimizer and self.peephole_optimizer:
            try:
                instrs, consts = self.peephole_optimizer.optimize_bytecode(instrs, consts)
            except Exception:
                pass

        # Normalize opcodes to names for SSA pipeline and stack dispatch
        normalized_instrs: List[Tuple[Any, Any]] = []
        for instr in instrs:
            if instr is None:
                continue
            if isinstance(instr, tuple) and len(instr) >= 2:
                op = instr[0]
                operand = instr[1] if len(instr) == 2 else tuple(instr[1:])
                op_name = op.name if hasattr(op, "name") else op
                normalized_instrs.append((op_name, operand))

        instrs = normalized_instrs

        if not fast_single_shot and self.enable_ssa and self.ssa_converter:
            try:
                ssa_program = self.ssa_converter.convert_to_ssa(instrs)
                ssa_instrs = destruct_ssa(ssa_program)
                instrs, consts = self._normalize_ssa_instructions(ssa_instrs, consts)
            except Exception:
                pass

        # 1. JIT Check (with thread safety)
        if self.use_jit and self.jit_compiler:
            jit_function = None
            if self.enable_gas_metering and self.gas_metering:
                # JIT currently bypasses gas accounting; fall back to interpreter
                jit_function = None
            else:
                with self._jit_lock:
                    bytecode_hash = self.jit_compiler._hash_bytecode(bytecode)
                    jit_function = self.jit_compiler.compilation_cache.get(bytecode_hash)
            
            if jit_function:
                try:
                    start_t = time.perf_counter()
                    stack = []
                    result = jit_function(self, stack, self.env)
                    with self._jit_lock:
                        self.jit_compiler.stats.cache_hits += 1
                        self.jit_compiler.record_execution_time(bytecode_hash, time.perf_counter() - start_t, ExecutionTier.JIT_NATIVE)
                    if debug: print(f"[VM JIT] Executed cached function")
                    return result
                except Exception as e:
                    if debug: print(f"[VM JIT] Failed: {e}, falling back")

        # 2. Bytecode Execution Setup
        ip = 0
        running = True
        return_value = None
        profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
        profile_ops = profile_flag is not None and profile_flag.lower() not in ("0", "false", "off")
        profile_verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
        profile_verbose = profile_verbose_flag and profile_verbose_flag.lower() not in ("0", "false", "off")
        opcode_counts: Optional[Dict[str, int]] = {} if profile_ops else None
        if profile_ops and profile_verbose:
            print(f"[VM DEBUG] opcode profiling enabled; instrs={len(instrs)}")

        class _EvalStack:
            __slots__ = ("data", "sp")

            def __init__(self, capacity: int):
                base = max(32, capacity)
                self.data = [None] * base
                self.sp = 0

            def _ensure_capacity(self):
                if self.sp >= len(self.data):
                    self.data.extend([None] * len(self.data))

            def append(self, value: Any):
                self._ensure_capacity()
                self.data[self.sp] = value
                self.sp += 1

            def pop(self):
                if self.sp == 0:
                    raise IndexError("pop from empty stack")
                self.sp -= 1
                value = self.data[self.sp]
                self.data[self.sp] = None
                return value

            def peek(self, default: Any = None):
                if self.sp == 0:
                    return default
                return self.data[self.sp - 1]

            def __len__(self) -> int:
                return self.sp

            def __bool__(self) -> bool:
                return self.sp > 0

            def __getitem__(self, index: int):
                if index < 0:
                    index = self.sp + index
                if index < 0 or index >= self.sp:
                    raise IndexError("stack index out of range")
                return self.data[index]

            def snapshot(self) -> List[Any]:
                return self.data[:self.sp]

        stack = _EvalStack(len(instrs) * 2 if instrs else 32)

        def const(idx):
            if isinstance(idx, int):
                return consts[idx] if 0 <= idx < len(consts) else None
            return idx

        # Lexical Resolution Helper (Closures/Cells)
        def _resolve(name):
            # 1. Local
            if name in self.env:
                val = self.env[name]
                return val.value if isinstance(val, Cell) else val
            # 2. Closure Cells (attached to VM)
            if name in self._closure_cells:
                return self._closure_cells[name].value
            # 3. Parent Chain
            p = self._parent_env
            while p is not None:
                if isinstance(p, VM):
                    if name in p.env:
                        val = p.env[name]
                        return val.value if isinstance(val, Cell) else val
                    if name in p._closure_cells:
                        return p._closure_cells[name].value
                    p = p._parent_env
                else:
                    if name in p: return p[name]
                    p = None
            return None

        def _store(name, value):
            # Update existing Cell in local env
            if name in self.env and isinstance(self.env[name], Cell):
                self.env[name].value = value; return
            # Update local non-cell
            if name in self.env:
                self.env[name] = value; return
            # Update Closure Cell
            if name in self._closure_cells:
                self._closure_cells[name].value = value; return
            # Update Parent Chain
            p = self._parent_env
            while p is not None:
                if isinstance(p, VM):
                    if name in p._closure_cells:
                        p._closure_cells[name].value = value; return
                    if name in p.env:
                        p.env[name] = value; return
                    p = p._parent_env
                else:
                    if name in p:
                        p[name] = value; return
                    p = None
            # Default: Create local
            self.env[name] = value

        def _unwrap(value):
            if isinstance(value, ZNull):
                return None
            return value.value if hasattr(value, 'value') else value

        def _binary_op(func):
            def wrapper(_):
                b = _unwrap(stack.pop() if stack else 0)
                a = _unwrap(stack.pop() if stack else 0)
                if isinstance(a, ZEvaluationError):
                    stack.append(a)
                    return
                if isinstance(b, ZEvaluationError):
                    stack.append(b)
                    return
                try:
                    stack.append(func(a, b))
                except Exception as exc:
                    if os.environ.get("ZEXUS_VM_PROFILE_OPS"):
                        print(f"[VM DEBUG] binary op error func={func} a={a!r} b={b!r} exc={exc}")
                    raise
            return wrapper

        def _binary_bool_op(func):
            def wrapper(_):
                b = _unwrap(stack.pop() if stack else None)
                a = _unwrap(stack.pop() if stack else None)
                if isinstance(a, ZEvaluationError):
                    stack.append(a)
                    return
                if isinstance(b, ZEvaluationError):
                    stack.append(b)
                    return
                stack.append(func(a, b))
            return wrapper

        async def _op_call_name(operand):
            if not operand:
                stack.append(None)
                return
            name_idx, arg_count = operand
            func_name = const(name_idx)
            args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
            fn = _resolve(func_name) or self.builtins.get(func_name)
            if fn is None:
                fallback_res = self._call_fallback_builtin(func_name, args)
                stack.append(fallback_res)
                return
            res = await self._invoke_callable_or_funcdesc(fn, args)
            stack.append(res)

        async def _op_call_top(arg_count):
            count = arg_count or 0
            args = [stack.pop() for _ in range(count)][::-1] if count else []
            fn_obj = stack.pop() if stack else None
            res = await self._invoke_callable_or_funcdesc(fn_obj, args)
            stack.append(res)

        async def _op_call_method(operand):
            if not operand:
                stack.append(None)
                return

            method_idx, arg_count = operand
            args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
            target = stack.pop() if stack else None
            method_name = const(method_idx)

            if target is None:
                stack.append(None)
                return

            result = None
            verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
            verbose_active = verbose_flag and verbose_flag.lower() not in ("0", "false", "off")
            if verbose_active and self._call_method_trace_count < 25:
                target_type = type(target).__name__
                preview = []
                for item in args[:3]:
                    try:
                        preview.append(repr(item))
                    except Exception:
                        preview.append(f"<{type(item).__name__}>")
                print(f"[VM TRACE] CALL_METHOD {method_name} target={target_type} args={len(args)} preview={preview}")
                self._call_method_trace_count += 1
            try:
                if method_name == "set":
                    if isinstance(target, ZMap):
                        if len(args) >= 2:
                            key = args[0]
                            if isinstance(key, str):
                                key = ZString(key)
                            result = target.set(key, args[1])
                        else:
                            result = None
                    elif isinstance(target, ZList):
                        if len(args) >= 2:
                            result = target.set(args[0], args[1])
                        else:
                            result = None
                    elif isinstance(target, (dict, list)):
                        if len(args) >= 2:
                            target[args[0]] = args[1]
                            result = args[1]
                        else:
                            result = None
                elif hasattr(target, "call_method"):
                    wrapped_args = [self._wrap_for_builtin(arg) for arg in args]
                    result = target.call_method(method_name, wrapped_args)
                else:
                    attr = getattr(target, method_name, None)
                    if callable(attr):
                        result = attr(*args)
                    elif isinstance(target, dict) and method_name in target:
                        candidate = target[method_name]
                        result = candidate(*args) if callable(candidate) else candidate
                    else:
                        result = attr
                if verbose_active and self._call_method_trace_count <= 25:
                    print(f"[VM TRACE] CALL_METHOD {method_name} result={result}")
            except Exception as exc:
                if debug:
                    print(f"[VM] CALL_METHOD failed for {method_name}: {exc}")
                raise

            stack.append(self._unwrap_after_builtin(result))

        def _op_load_const(idx):
            value = const(idx)
            if self.integer_pool and isinstance(value, int):
                value = self.integer_pool.get(value)
            elif self.string_pool and isinstance(value, str):
                value = self.string_pool.get(value)
            stack.append(value)

        def _op_load_name(idx):
            name = const(idx)
            stack.append(_resolve(name))

        def _op_store_name(idx):
            name = const(idx)
            val = stack.pop() if stack else None
            _store(name, val)
            if self.use_memory_manager and val is not None:
                self._allocate_managed(val, name=name, root=True)

        def _op_pop(_):
            if stack:
                stack.pop()

        def _op_dup(_):
            if stack:
                stack.append(stack[-1])

        def _op_neg(_):
            a = _unwrap(stack.pop() if stack else 0)
            stack.append(-a)

        def _op_not(_):
            a = stack.pop() if stack else False
            stack.append(not a)

        def _op_jump(target):
            nonlocal ip
            ip = target

        def _op_jump_if_false(target):
            nonlocal ip
            cond = stack.pop() if stack else None
            cond_val = _unwrap(cond)
            if not cond_val:
                ip = target

        def _op_return(_):
            nonlocal running, return_value
            return_value = stack.pop() if stack else None
            running = False

        def _op_build_list(count):
            total = count if count is not None else 0
            if self.list_pool:
                lst = self.allocate_list(total)
                if total > 0:
                    for i in range(total - 1, -1, -1):
                        lst[i] = stack.pop()
                stack.append(lst)
            else:
                elements = [stack.pop() for _ in range(total)][::-1]
                stack.append(elements)

        def _op_build_map(count):
            total = count if count is not None else 0
            result = {}
            for _ in range(total):
                val = stack.pop(); key = stack.pop()
                result[key] = val
            stack.append(result)

        def _op_index(_):
            idx = stack.pop(); obj = stack.pop()
            try:
                if isinstance(obj, ZList):
                    stack.append(obj.get(idx))
                elif isinstance(obj, ZMap):
                    key = idx
                    if isinstance(key, str):
                        key = ZString(key)
                    stack.append(obj.get(key))
                elif isinstance(obj, ZString):
                    stack.append(obj[idx])
                else:
                    raw_idx = idx.value if hasattr(idx, "value") else idx
                    stack.append(obj[raw_idx] if obj is not None else None)
            except (IndexError, KeyError, TypeError):
                stack.append(None)

        def _op_get_attr(_):
            attr = stack.pop() if stack else None
            obj = stack.pop() if stack else None
            if obj is None:
                stack.append(None)
                return
            attr_name = _unwrap(attr)
            try:
                if isinstance(obj, ZMap):
                    key = attr_name
                    if isinstance(key, str):
                        key = ZString(key)
                    stack.append(obj.get(key))
                elif isinstance(obj, dict):
                    stack.append(obj.get(attr_name))
                else:
                    stack.append(getattr(obj, attr_name, None))
            except Exception:
                stack.append(None)

        def _op_get_length(_):
            obj = stack.pop()
            try:
                if obj is None:
                    stack.append(0)
                elif isinstance(obj, ZList):
                    stack.append(len(obj.elements))
                elif isinstance(obj, ZMap):
                    stack.append(len(obj.pairs))
                elif isinstance(obj, ZString):
                    stack.append(len(obj.value))
                elif hasattr(obj, '__len__'):
                    stack.append(len(obj))
                else:
                    stack.append(0)
            except (TypeError, AttributeError):
                stack.append(0)

        def _op_read(_):
            path = stack.pop() if stack else None
            try:
                import os
                if path and os.path.exists(path):
                    with open(path, 'r') as f:
                        stack.append(f.read())
                else:
                    stack.append(None)
            except:
                stack.append(None)

        def _op_import(operand):
            # Placeholder
            pass

        def _op_store_func(operand):
            name_idx, func_idx = operand
            name = const(name_idx)
            func = const(func_idx)
            _store(name, func)

        dispatch_table: Dict[str, Callable[[Any], Any]] = {
            "LOAD_CONST": _op_load_const,
            "LOAD_NAME": _op_load_name,
            "STORE_NAME": _op_store_name,
            "POP": _op_pop,
            "DUP": _op_dup,
            "CALL_NAME": _op_call_name,
            "CALL_TOP": _op_call_top,
            "CALL_METHOD": _op_call_method,
            "ADD": _binary_op(lambda a, b: a + b),
            "SUB": _binary_op(lambda a, b: a - b),
            "MUL": _binary_op(lambda a, b: a * b),
            "DIV": _binary_op(lambda a, b: a / b if b != 0 else 0),
            "MOD": _binary_op(lambda a, b: a % b if b != 0 else 0),
            "POW": _binary_op(lambda a, b: a ** b),
            "NEG": _op_neg,
            "EQ": _binary_bool_op(lambda a, b: a == b),
            "NEQ": _binary_bool_op(lambda a, b: a != b),
            "LT": _binary_op(lambda a, b: a < b),
            "GT": _binary_op(lambda a, b: a > b),
            "LTE": _binary_op(lambda a, b: a <= b),
            "GTE": _binary_op(lambda a, b: a >= b),
            "NOT": _op_not,
            "JUMP": _op_jump,
            "JUMP_IF_FALSE": _op_jump_if_false,
            "RETURN": _op_return,
            "BUILD_LIST": _op_build_list,
            "BUILD_MAP": _op_build_map,
            "INDEX": _op_index,
            "GET_ATTR": _op_get_attr,
            "GET_LENGTH": _op_get_length,
            "READ": _op_read,
            "IMPORT": _op_import,
            "STORE_FUNC": _op_store_func,
        }
        async_dispatch_ops = {"CALL_NAME", "CALL_TOP", "CALL_METHOD"}

        # 3. Execution Loop
        prev_ip = None
        while running and ip < len(instrs):
            try:
                current_ip = ip
                op, operand = instrs[current_ip]
                
                # Handle Opcode enum: convert to name for comparison
                if hasattr(op, 'name'):  # Opcode IntEnum
                    op_name = op.name
                else:  # Already a string
                    op_name = op
    
                if profile_ops:
                    opcode_counts[op_name] = opcode_counts.get(op_name, 0) + 1
                
                if debug: print(f"[VM SL] ip={ip} op={op} operand={operand} stack={stack.snapshot()}")
                
                # Profile instruction (if enabled) - start timing
                instr_start_time = None
                if self.enable_profiling and self.profiler and self.profiler.enabled:
                    if self.profiler.level in (ProfilingLevel.DETAILED, ProfilingLevel.FULL):
                        instr_start_time = time.perf_counter()
                    # Record instruction (count only for BASIC level)
                    self.profiler.record_instruction(current_ip, op_name, operand, prev_ip, len(stack))
                
                prev_ip = current_ip
                ip += 1
    
                # === GAS METERING ===
                if self.enable_gas_metering and self.gas_metering:
                    # Determine gas cost parameters
                    gas_kwargs = {}
                    if op_name in ("BUILD_LIST", "BUILD_MAP"):
                        gas_kwargs['count'] = operand if operand is not None else 0
                    elif op_name == "MERKLE_ROOT":
                        gas_kwargs['leaf_count'] = operand if operand is not None else 0
                    elif op_name in ("CALL_NAME", "CALL_TOP", "CALL_METHOD", "CALL_BUILTIN"):
                        if op_name == "CALL_NAME":
                            gas_kwargs['arg_count'] = operand[1] if isinstance(operand, tuple) else 0
                        elif op_name == "CALL_TOP":
                            gas_kwargs['arg_count'] = operand if operand is not None else 0
                        else:
                            gas_kwargs['arg_count'] = operand[1] if isinstance(operand, tuple) else 0
                    
                    # Consume gas for operation
                    if not self.gas_metering.consume(op_name, **gas_kwargs):
                        # Out of gas!
                        if self.gas_metering.operation_count > self.gas_metering.max_operations:
                            raise OperationLimitExceededError(
                                self.gas_metering.operation_count,
                                self.gas_metering.max_operations
                            )
                        else:
                            raise OutOfGasError(
                                self.gas_metering.gas_used,
                                self.gas_metering.gas_limit,
                                op_name
                            )
    
                handler = dispatch_table.get(op_name)
                if handler is not None:
                    if op_name in async_dispatch_ops:
                        await handler(operand)
                    else:
                        handler(operand)
                    if not running:
                        break
                    continue
    
                # --- Basic Stack Ops ---
                if op_name == "LOAD_CONST":
                    stack.append(const(operand))
                elif op_name == "LOAD_NAME":
                    name = const(operand)
                    stack.append(_resolve(name))
                elif op_name == "STORE_NAME":
                    name = const(operand)
                    val = stack.pop() if stack else None
                    _store(name, val)
                    if self.use_memory_manager and val is not None:
                        self._allocate_managed(val, name=name)
                elif op_name == "POP":
                    if stack: stack.pop()
                elif op_name == "DUP":
                    if stack: stack.append(stack[-1])
                elif op_name == "PRINT":
                    val = stack.pop() if stack else None
                    print(val)
                
                # --- Function/Closure Ops ---
                elif op_name == "STORE_FUNC":
                    name_idx, func_idx = operand
                    name = const(name_idx)
                    func_desc = const(func_idx)
                    # Create func descriptor, capturing current VM as parent
                    func_desc_copy = dict(func_desc) if isinstance(func_desc, dict) else {"bytecode": func_desc}
                    closure_snapshot = {}
                    # Snapshot current environment (excluding internal keys)
                    for key, value in self.env.items():
                        if isinstance(key, str) and key.startswith("_"):
                            continue
                        closure_snapshot[key] = value
                    # Include existing closure cells if present
                    for key, cell in self._closure_cells.items():
                        if key not in closure_snapshot:
                            closure_snapshot[key] = cell.value
                    if closure_snapshot:
                        func_desc_copy["closure_snapshot"] = closure_snapshot
                    func_desc_copy["parent_vm"] = self
                    self.env[name] = func_desc_copy
                
                elif op_name == "CALL_NAME":
                    name_idx, arg_count = operand
                    func_name = const(name_idx)
                    args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
                    fn = _resolve(func_name) or self.builtins.get(func_name)
                    if fn is None:
                        res = self._call_fallback_builtin(func_name, args)
                    else:
                        res = await self._invoke_callable_or_funcdesc(fn, args)
                    stack.append(res)
                
                elif op_name == "CALL_TOP":
                    arg_count = operand
                    args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
                    fn_obj = stack.pop() if stack else None
                    res = await self._invoke_callable_or_funcdesc(fn_obj, args)
                    stack.append(res)
    
                # --- Arithmetic & Logic ---
                elif op_name == "ADD":
                    b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                    # Auto-unwrap evaluator objects
                    if hasattr(a, 'value'): a = a.value
                    if hasattr(b, 'value'): b = b.value
                    stack.append(a + b)
                elif op_name == "SUB":
                    b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                    if hasattr(a, 'value'): a = a.value
                    if hasattr(b, 'value'): b = b.value
                    stack.append(a - b)
                elif op_name == "MUL":
                    b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                    if hasattr(a, 'value'): a = a.value
                    if hasattr(b, 'value'): b = b.value
                    stack.append(a * b)
                elif op_name == "DIV":
                    b = stack.pop() if stack else 1; a = stack.pop() if stack else 0
                    if hasattr(a, 'value'): a = a.value
                    if hasattr(b, 'value'): b = b.value
                    stack.append(a / b if b != 0 else 0)
                elif op_name == "MOD":
                    b = stack.pop() if stack else 1; a = stack.pop() if stack else 0
                    stack.append(a % b if b != 0 else 0)
                elif op_name == "POW":
                    b = stack.pop() if stack else 1; a = stack.pop() if stack else 0
                    stack.append(a ** b)
                elif op_name == "NEG":
                    a = stack.pop() if stack else 0
                    stack.append(-a)
                elif op_name == "EQ":
                    b = stack.pop() if stack else None; a = stack.pop() if stack else None
                    stack.append(a == b)
                elif op_name == "NEQ":
                    b = stack.pop() if stack else None; a = stack.pop() if stack else None
                    stack.append(a != b)
                elif op_name == "LT":
                    b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                    stack.append(a < b)
                elif op_name == "GT":
                    b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                    stack.append(a > b)
                elif op_name == "LTE":
                    b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                    stack.append(a <= b)
                elif op_name == "GTE":
                    b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                    stack.append(a >= b)
                elif op_name == "NOT":
                    a = stack.pop() if stack else False
                    stack.append(not a)
    
                # --- Control Flow ---
                elif op_name == "JUMP":
                    ip = operand
                elif op_name == "JUMP_IF_FALSE":
                    cond = stack.pop() if stack else None
                    if not cond: ip = operand
                elif op_name == "RETURN":
                    return stack.pop() if stack else None
    
                # --- Collections ---
                elif op_name == "BUILD_LIST":
                    count = operand if operand is not None else 0
                    elements = [stack.pop() for _ in range(count)][::-1]
                    stack.append(elements)
                elif op_name == "BUILD_MAP":
                    count = operand if operand is not None else 0
                    result = {}
                    for _ in range(count):
                        val = stack.pop(); key = stack.pop()
                        result[key] = val
                    stack.append(result)
                elif op_name == "INDEX":
                    idx = stack.pop(); obj = stack.pop()
                    try:
                        if isinstance(obj, ZList):
                            stack.append(obj.get(idx))
                        elif isinstance(obj, ZMap):
                            stack.append(obj.get(idx))
                        elif isinstance(obj, ZString):
                            stack.append(obj[idx])
                        else:
                            stack.append(obj[idx] if obj is not None else None)
                    except (IndexError, KeyError, TypeError):
                        stack.append(None)
                elif op_name == "GET_LENGTH":
                    obj = stack.pop()
                    try:
                        if obj is None:
                            stack.append(0)
                        elif isinstance(obj, ZList):
                            stack.append(len(obj.elements))
                        elif isinstance(obj, ZMap):
                            stack.append(len(obj.pairs))
                        elif isinstance(obj, ZString):
                            stack.append(len(obj.value))
                        elif hasattr(obj, '__len__'):
                            stack.append(len(obj))
                        else:
                            stack.append(0)
                    except (TypeError, AttributeError):
                        stack.append(0)
    
                # --- Async & Events ---
                elif op_name == "SPAWN":
                    # operand: tuple ("CALL", func_name, arg_count) OR index
                    task_handle = None
                    if isinstance(operand, tuple) and operand[0] == "CALL":
                        fn_name = operand[1]; arg_count = operand[2]
                        args = [stack.pop() for _ in range(arg_count)][::-1]
                        fn = self.builtins.get(fn_name) or self.env.get(fn_name)
                        coro = self._to_coro(fn, args)
                        
                        # Use async optimizer if available
                        if self.async_optimizer:
                            coro = self.async_optimizer.spawn(coro)
                            task = asyncio.create_task(coro)
                        else:
                            task = asyncio.create_task(coro)
                        
                        self._task_counter += 1
                        tid = f"task_{self._task_counter}"
                        self._tasks[tid] = task
                        task_handle = tid
                    stack.append(task_handle)
    
                elif op_name == "AWAIT":
                    # Keep popping until we find a task to await
                    result_found = False
                    temp_stack = self.allocate_list(0)
                    
                    while stack and not result_found:
                        top = stack.pop()
                        
                        if isinstance(top, str) and top in self._tasks:
                            # Use async optimizer if available
                            if self.async_optimizer:
                                res = await self.async_optimizer.await_optimized(self._tasks[top])
                            else:
                                res = await self._tasks[top]
                            # Push back any non-task values we skipped
                            for val in reversed(temp_stack):
                                stack.append(val)
                            stack.append(res)
                            result_found = True
                        elif asyncio.iscoroutine(top) or isinstance(top, asyncio.Future):
                            # Use async optimizer if available
                            if self.async_optimizer:
                                res = await self.async_optimizer.await_optimized(top)
                            else:
                                res = await top
                            # Push back any non-task values we skipped
                            for val in reversed(temp_stack):
                                stack.append(val)
                            stack.append(res)
                            result_found = True
                        else:
                            # Not a task, save it and keep looking
                            temp_stack.append(top)
                    
                    # If no task was found, put everything back
                    if not result_found:
                        for val in reversed(temp_stack):
                            stack.append(val)
    
                    if temp_stack:
                        temp_stack.clear()
                    self.release_list(temp_stack)
    
                elif op_name == "REGISTER_EVENT":
                    event_parts = operand if isinstance(operand, (list, tuple)) else (operand,)
                    event_name = const(event_parts[0]) if event_parts else None
                    handler = const(event_parts[1]) if len(event_parts) > 1 else None
                    if event_name is not None:
                        handlers = self._events.setdefault(event_name, [])
                        if handler and handler not in handlers:
                            handlers.append(handler)
    
                elif op_name == "EMIT_EVENT":
                    event_ref = const(operand[0]) if operand else None
                    payload_ref = None
                    event_name = event_ref
                    if isinstance(event_ref, (list, tuple)) and event_ref:
                        event_name = event_ref[0]
                        if len(event_ref) > 1:
                            payload_ref = event_ref[1]
    
                    payload = None
                    if stack:
                        payload = stack.pop()
                    elif payload_ref is not None:
                        payload = const(payload_ref)
                    elif isinstance(operand, (list, tuple)) and len(operand) > 1:
                        payload = const(operand[1])
    
                    payload = _unwrap(payload)
    
                    handlers = self._events.get(event_name, [])
                    for h in handlers:
                        fn = self.builtins.get(h) or self.env.get(h)
                        if fn is None:
                            continue
                        await self._call_builtin_async_obj(fn, [payload], wrap_args=False)
    
                elif op_name == "IMPORT":
                    mod_name = const(operand[0])
                    alias = const(operand[1]) if isinstance(operand, (list,tuple)) and len(operand) > 1 else None
                    try:
                        mod = importlib.import_module(mod_name)
                        self.env[alias or mod_name] = mod
                    except Exception:
                        self.env[alias or mod_name] = None
    
                elif op_name == "DEFINE_ENUM":
                    enum_name = _unwrap(const(operand[0]))
                    enum_map = const(operand[1])
                    self.env.setdefault("enums", {})[enum_name] = enum_map
                    self.env[enum_name] = enum_map
    
                elif op_name == "ASSERT_PROTOCOL":
                    obj_name = const(operand[0])
                    spec = const(operand[1])
                    obj = self.env.get(obj_name)
                    ok = True
                    missing = []
                    for m in spec.get("methods", []):
                        if not hasattr(obj, m):
                            ok = False; missing.append(m)
                    stack.append((ok, missing))
    
                # --- Blockchain Specific Opcodes ---
                
                elif op_name == "HASH_BLOCK":
                    block_data = stack.pop() if stack else ""
                    if isinstance(block_data, dict):
                        import json; block_data = json.dumps(block_data, sort_keys=True)
                    if not isinstance(block_data, (bytes, str)): block_data = str(block_data)
                    if isinstance(block_data, str): block_data = block_data.encode('utf-8')
                    stack.append(hashlib.sha256(block_data).hexdigest())
    
                elif op_name == "VERIFY_SIGNATURE":
                    if len(stack) >= 3:
                        pk = stack.pop(); msg = stack.pop(); sig = stack.pop()
                        verify_fn = self.builtins.get("verify_sig") or self.env.get("verify_sig")
                        if verify_fn:
                            res = await self._invoke_callable_or_funcdesc(verify_fn, [sig, msg, pk])
                            stack.append(res)
                        else:
                            # Fallback for testing
                            expected = hashlib.sha256(str(msg).encode()).hexdigest()
                            stack.append(sig == expected)
                    else:
                        stack.append(False)
    
                elif op_name == "MERKLE_ROOT":
                    leaf_count = operand if operand is not None else 0
                    if leaf_count <= 0:
                        stack.append("")
                    else:
                        leaves = [stack.pop() for _ in range(leaf_count)][::-1] if len(stack) >= leaf_count else []
                        hashes = []
                        for leaf in leaves:
                            if isinstance(leaf, dict):
                                import json; leaf = json.dumps(leaf, sort_keys=True)
                            if not isinstance(leaf, (str, bytes)): leaf = str(leaf)
                            if isinstance(leaf, str): leaf = leaf.encode('utf-8')
                            hashes.append(hashlib.sha256(leaf).hexdigest())
                        
                        while len(hashes) > 1:
                            if len(hashes) % 2 != 0: hashes.append(hashes[-1])
                            new_hashes = []
                            for i in range(0, len(hashes), 2):
                                combined = (hashes[i] + hashes[i+1]).encode('utf-8')
                                new_hashes.append(hashlib.sha256(combined).hexdigest())
                            hashes = new_hashes
                        stack.append(hashes[0] if hashes else "")
    
                elif op_name == "STATE_READ":
                    if operand is None:
                        key = _unwrap(stack.pop() if stack else None)
                    else:
                        key = const(operand)
                    stack.append(self.env.setdefault("_blockchain_state", {}).get(key))
    
                elif op_name == "STATE_WRITE":
                    val = _unwrap(stack.pop() if stack else None)
                    if operand is None:
                        key = _unwrap(stack.pop() if stack else None)
                    else:
                        key = const(operand)
                    if self.env.get("_in_transaction", False):
                        self.env.setdefault("_tx_pending_state", {})[key] = val
                    else:
                        self.env.setdefault("_blockchain_state", {})[key] = val
    
                elif op_name == "TX_BEGIN":
                    self.env["_in_transaction"] = True
                    self.env["_tx_pending_state"] = {}
                    self.env["_tx_snapshot"] = dict(self.env.get("_blockchain_state", {}))
                    if self.use_memory_manager: self.env["_tx_memory_snapshot"] = dict(self._managed_objects)
    
                elif op_name == "TX_COMMIT":
                    if self.env.get("_in_transaction", False):
                        self.env.setdefault("_blockchain_state", {}).update(self.env.get("_tx_pending_state", {}))
                        self.env["_in_transaction"] = False
                        self.env["_tx_pending_state"] = {}
                        if "_tx_memory_snapshot" in self.env: del self.env["_tx_memory_snapshot"]
    
                elif op_name == "TX_REVERT":
                    if self.env.get("_in_transaction", False):
                        self.env["_blockchain_state"] = dict(self.env.get("_tx_snapshot", {}))
                        self.env["_in_transaction"] = False
                        self.env["_tx_pending_state"] = {}
                        if self.use_memory_manager and "_tx_memory_snapshot" in self.env:
                            self._managed_objects = dict(self.env["_tx_memory_snapshot"])
    
                elif op_name == "ENABLE_ERROR_MODE":
                    self.env["_continue_on_error"] = True
                    if self.debug: print("[VM] Error Recovery Mode ENABLED")
    
                elif op_name == "GAS_CHARGE":
                    amount = operand if operand is not None else 0
                    current = self.env.get("_gas_remaining", float('inf'))
                    if current != float('inf'):
                        new_gas = current - amount
                        if new_gas < 0:
                            # Revert if in TX
                            if self.env.get("_in_transaction", False):
                                self.env["_blockchain_state"] = dict(self.env.get("_tx_snapshot", {}))
                                self.env["_in_transaction"] = False
                            stack.append({"error": "OutOfGas", "required": amount, "remaining": current})
                            return stack[-1]
                        self.env["_gas_remaining"] = new_gas
    
                elif op_name == "REQUIRE":
                    message = stack.pop() if stack else "Requirement failed"
                    if hasattr(message, 'value'): message = message.value
                    condition = stack.pop() if stack else False
                    cond_val = condition.value if hasattr(condition, 'value') else condition
                    
                    if not cond_val:
                        if self.env.get("_in_transaction", False):
                            self.env["_blockchain_state"] = dict(self.env.get("_tx_snapshot", {}))
                            self.env["_in_transaction"] = False
                            self.env["_tx_pending_state"] = {}
                        raise ZEvaluationError(f"Requirement failed: {message}")
    
                elif op_name == "DEFINE_CONTRACT":
                    member_count = operand
                    members = {}
                    for _ in range(member_count):
                        key_obj = stack.pop() if stack else None
                        val_obj = stack.pop() if stack else None
                        key_str = key_obj.value if hasattr(key_obj, 'value') else str(key_obj)
                        members[key_str] = val_obj
                    
                    name_obj = stack.pop() if stack else None
                    stack.append(ZMap(members))
    
                elif op_name == "DEFINE_ENTITY":
                    member_count = operand
                    members = {}
                    for _ in range(member_count):
                        key_obj = stack.pop() if stack else None
                        val_obj = stack.pop() if stack else None
                        key_str = key_obj.value if hasattr(key_obj, 'value') else str(key_obj)
                        members[key_str] = val_obj
                    
                    name_obj = stack.pop() if stack else None
                    # Create Entity (using Map for now, can be specialized Entity class later)
                    members['_type'] = 'entity'
                    members['_name'] = name_obj.value if hasattr(name_obj, 'value') else str(name_obj)
                    stack.append(ZMap(members))
    
                elif op_name == "DEFINE_CAPABILITY":
                    name = stack.pop() if stack else None
                    definition = stack.pop() if stack else {}
                    if hasattr(name, 'value'): name = name.value
                    self.env.setdefault("_capabilities", {})[name] = definition
    
                elif op_name == "GRANT_CAPABILITY":
                    count = operand
                    caps = [stack.pop() for _ in range(count)][::-1]
                    entity_name = stack.pop() if stack else None
                    if hasattr(entity_name, 'value'): entity_name = entity_name.value
                    
                    grants = self.env.setdefault("_grants", {})
                    entity_grants = grants.setdefault(entity_name, set())
                    
                    for cap in caps:
                        c_val = cap.value if hasattr(cap, 'value') else str(cap)
                        entity_grants.add(c_val)
    
                elif op_name == "REVOKE_CAPABILITY":
                    count = operand
                    caps = [stack.pop() for _ in range(count)][::-1]
                    entity_name = stack.pop() if stack else None
                    if hasattr(entity_name, 'value'): entity_name = entity_name.value
                    
                    if "_grants" in self.env and entity_name in self.env["_grants"]:
                        entity_grants = self.env["_grants"][entity_name]
                        for cap in caps:
                            c_val = cap.value if hasattr(cap, 'value') else str(cap)
                            if c_val in entity_grants:
                                entity_grants.remove(c_val)
    
                elif op_name == "AUDIT_LOG":
                    ts = stack.pop()
                    action = stack.pop()
                    data = stack.pop()
                    # Unwrap
                    ts = ts.value if hasattr(ts, 'value') else ts
                    action = action.value if hasattr(action, 'value') else action
                    data = data.value if hasattr(data, 'value') else data
                    
                    entry = {"timestamp": ts, "action": action, "data": data}
                    self.env.setdefault("_audit_log", []).append(entry)
                    if self.debug: print(f"[AUDIT] {entry}")
    
                elif op_name == "RESTRICT_ACCESS":
                    restriction = stack.pop()
                    prop = stack.pop()
                    obj = stack.pop()
                    # Just store in a registry for now. 
                    # Real implementation would hook into PropertyAccess/Assign logic.
                    r_key = f"{obj}.{prop}" if prop else str(obj)
                    self.env.setdefault("_restrictions", {})[r_key] = restriction
    
                elif op_name == "LEDGER_APPEND":
                    entry = stack.pop() if stack else None
                    if isinstance(entry, dict) and "timestamp" not in entry:
                        entry["timestamp"] = time.time()
                    self.env.setdefault("_ledger", []).append(entry)
    
                else:
                    if debug: print(f"[VM] Unknown Opcode: {op}")
    
                # Record instruction timing (if profiling enabled)
                if instr_start_time is not None and self.profiler:
                    elapsed = time.perf_counter() - instr_start_time
                    self.profiler.measure_instruction(current_ip, elapsed)
            except Exception as e:
                if self.env.get("_continue_on_error", False):
                    # Error Recovery Mode
                    if debug: print(f"[VM ERROR RECOVERY] {e}")
                    self.env.setdefault("_errors", []).append(str(e))
                else:
                    raise

        if profile_ops and opcode_counts is not None:
            self._last_opcode_profile = sorted(opcode_counts.items(), key=lambda item: item[1], reverse=True)
        elif self._last_opcode_profile is not None:
            self._last_opcode_profile = None

        if not running:
            return return_value
        return stack[-1] if stack else None

    # ==================== Helpers ====================

    async def _invoke_callable_or_funcdesc(self, fn, args, is_constant=False):
        # 1. Function Descriptor (VM Bytecode Closure)
        if isinstance(fn, dict) and "bytecode" in fn:
            func_bc = fn["bytecode"]
            params = fn.get("params", [])
            is_async = fn.get("is_async", False)
            # Use captured parent_vm (closure), fallback to self
            parent_env = fn.get("parent_vm", self)
            
            local_env = {k: v for k, v in zip(params, args)}
            
            inner_vm = VM(
                builtins=self.builtins, 
                env=local_env, 
                parent_env=parent_env,
                # Inherit configuration
                use_jit=self.use_jit,
                use_memory_manager=self.use_memory_manager
            )
            snapshot = fn.get("closure_snapshot")
            if snapshot:
                for key, value in snapshot.items():
                    inner_vm._closure_cells[key] = Cell(value)
            return await inner_vm._run_stack_bytecode(func_bc, debug=False)
        
        # 2. Python Callable / Builtin Wrapper
        return await self._call_builtin_async_obj(fn, args)

    async def _call_builtin_async(self, name: str, args: List[Any], wrap_args: bool = True):
        target = self.builtins.get(name) or self.env.get(name)
        
        # Check Renderer Backend
        if _BACKEND_AVAILABLE and hasattr(_BACKEND, name):
            fn = getattr(_BACKEND, name)
            if asyncio.iscoroutinefunction(fn): return await fn(*args)
            return fn(*args)
        
        if target is None:
            return self._call_fallback_builtin(name, args)

        return await self._call_builtin_async_obj(target, args, wrap_args=wrap_args)

    async def _call_builtin_async_obj(self, fn_obj, args: List[Any], wrap_args: bool = True):
        try:
            if fn_obj is None: return None
            
            # Extract .fn if it's a wrapper
            real_fn = fn_obj.fn if hasattr(fn_obj, "fn") else fn_obj

            # Execute Zexus Action/LambdaFunction via evaluator
            try:
                from ..object import Action as ZAction, LambdaFunction as ZLambda
                if isinstance(real_fn, (ZAction, ZLambda)):
                    from ..evaluator.core import Evaluator
                    if self._action_evaluator is None:
                        self._action_evaluator = Evaluator(use_vm=False)
                    call_args = [self._wrap_for_builtin(arg) for arg in args] if wrap_args else list(args)
                    result = self._action_evaluator.apply_function(real_fn, call_args)
                    return self._unwrap_after_builtin(result)
            except Exception:
                pass
            
            if not callable(real_fn): return real_fn
            
            call_args = [self._wrap_for_builtin(arg) for arg in args] if wrap_args else list(args)
            verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
            verbose_active = verbose_flag and verbose_flag.lower() not in ("0", "false", "off")
            if verbose_active:
                fn_name = getattr(fn_obj, "name", getattr(real_fn, "__name__", "<callable>"))
                print(f"[VM DEBUG] calling builtin {fn_name} args={[type(a).__name__ for a in call_args]}")
            res = real_fn(*call_args)
            if verbose_active and res is None:
                fn_name = getattr(fn_obj, "name", getattr(real_fn, "__name__", "<callable>"))
                print(f"[VM DEBUG] builtin {fn_name} returned None args={call_args}")
            if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
                res = await res
            return self._unwrap_after_builtin(res)
        except Exception as e:
            return e

    def _to_coro(self, fn, args):
        if asyncio.iscoroutinefunction(fn):
            return fn(*args)
        async def _wrap():
            if callable(fn): return fn(*args)
            return None
        return _wrap()

    def _call_fallback_builtin(self, name: str, args: List[Any]):
        func = _FALLBACK_BUILTINS.get(name)
        if not func:
            return None
        try:
            return func(args)
        except Exception as exc:
            if self.debug:
                print(f"[VM] fallback builtin '{name}' failed: {exc}")
            return ZEvaluationError(f"Builtin '{name}' failed: {exc}")

    def profile_execution(self, bytecode, iterations: int = 1000) -> Dict[str, Any]:
        """Profile execution performance across available modes"""
        import timeit
        results = {'iterations': iterations, 'modes': {}}
        
        # Stack
        def run_stack(): return self._run_coroutine_sync(self._execute_stack(bytecode))
        t_stack = timeit.timeit(run_stack, number=iterations)
        stack_avg = t_stack / iterations if iterations else 0.0
        results['modes']['stack'] = {'total': t_stack, 'avg': stack_avg}
        
        # Register
        if self._register_vm:
            def run_reg(): return self._execute_register(bytecode)
            t_reg = timeit.timeit(run_reg, number=iterations)
            reg_avg = t_reg / iterations if iterations else 0.0
            reg_entry = {'total': t_reg, 'avg': reg_avg}
            if t_reg > 0:
                reg_entry['speedup'] = t_stack / t_reg if t_stack > 0 else float('inf')
            results['modes']['register'] = reg_entry
            
        return results
    
    # ==================== Profiler Interface ====================
    
    def start_profiling(self):
        """Start profiling session"""
        if self.profiler:
            self.profiler.start()
    
    def stop_profiling(self):
        """Stop profiling session"""
        if self.profiler:
            self.profiler.stop()
    
    def get_profiling_report(self, format: str = 'text', top_n: int = 20) -> str:
        """Get profiling report"""
        if self.profiler:
            return self.profiler.generate_report(format=format, top_n=top_n)
        return "Profiling not enabled"
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get profiling summary statistics"""
        if self.profiler:
            return self.profiler.get_summary()
        return {'error': 'Profiling not enabled'}
    
    def reset_profiler(self):
        """Reset profiler statistics"""
        if self.profiler:
            self.profiler.reset()
    
    # ==================== Memory Pool Interface ====================
    
    def allocate_integer(self, value: int) -> int:
        """Allocate an integer from the pool"""
        if self.integer_pool:
            return self.integer_pool.get(value)
        return value
    
    def release_integer(self, value: int):
        """Release an integer back to the pool (no-op for integers)"""
        # IntegerPool doesn't need explicit release
        pass
    
    def allocate_string(self, value: str) -> str:
        """Allocate a string from the pool"""
        if self.string_pool:
            return self.string_pool.get(value)
        return value
    
    def release_string(self, value: str):
        """Release a string back to the pool (no-op for strings)"""
        # StringPool doesn't need explicit release (uses interning)
        pass
    
    def allocate_list(self, initial_capacity: int = 0) -> list:
        """Allocate a list from the pool"""
        if self.list_pool:
            return self.list_pool.acquire(initial_capacity)
        return [None] * initial_capacity if initial_capacity > 0 else []
    
    def release_list(self, value: list):
        """Release a list back to the pool"""
        if self.list_pool:
            self.list_pool.release(value)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        if not self.enable_memory_pool:
            return {'error': 'Memory pooling not enabled'}
        
        stats = {}
        if self.integer_pool:
            stats['integer_pool'] = self.integer_pool.stats.to_dict()
        if self.string_pool:
            stats['string_pool'] = self.string_pool.stats.to_dict()
        if self.list_pool:
            stats['list_pool'] = self.list_pool.get_stats()
        
        return stats
    
    def reset_pools(self):
        """Reset all memory pools"""
        if self.integer_pool:
            self.integer_pool.clear()
        if self.string_pool:
            self.string_pool.clear()
        if self.list_pool:
            self.list_pool.clear()
    
    # ==================== Peephole Optimizer Interface ====================
    
    def optimize_bytecode(self, bytecode):
        """
        Optimize bytecode using peephole optimizer
        
        Args:
            bytecode: Bytecode object or list of instructions
            
        Returns:
            Optimized bytecode
        """
        if not self.peephole_optimizer:
            return bytecode
        
        return self.peephole_optimizer.optimize(bytecode)
    
    def get_optimizer_stats(self) -> Dict[str, Any]:
        """Get peephole optimizer statistics"""
        if not self.peephole_optimizer:
            return {'error': 'Peephole optimizer not enabled'}
        
        return self.peephole_optimizer.stats.to_dict()
    
    def reset_optimizer_stats(self):
        """Reset peephole optimizer statistics"""
        if self.peephole_optimizer:
            self.peephole_optimizer.reset_stats()
    
    # ==================== Async Optimizer Interface ====================
    
    def get_async_stats(self) -> Dict[str, Any]:
        """Get async optimizer statistics"""
        if not self.async_optimizer:
            return {'error': 'Async optimizer not enabled'}
        
        return self.async_optimizer.get_stats()
    
    def reset_async_stats(self):
        """Reset async optimizer statistics"""
        if self.async_optimizer:
            self.async_optimizer.reset_stats()
    
    # ==================== SSA & Register Allocator Interface ====================
    
    def convert_to_ssa(self, instructions: List[Tuple]) -> Optional['SSAProgram']:
        """
        Convert instructions to SSA form
        
        Args:
            instructions: List of bytecode instructions
            
        Returns:
            SSAProgram or None if SSA not enabled
        """
        if not self.ssa_converter:
            return None
        
        return self.ssa_converter.convert_to_ssa(instructions)
    
    def allocate_registers(
        self,
        instructions: List[Tuple]
    ) -> Optional['AllocationResult']:
        """
        Allocate registers for instructions
        
        Args:
            instructions: List of bytecode instructions
            
        Returns:
            AllocationResult or None if register allocator not enabled
        """
        if not self.register_allocator:
            return None
        
        # Compute live ranges
        live_ranges = compute_live_ranges(instructions)
        
        # Allocate registers
        return self.register_allocator.allocate(instructions, live_ranges)
    
    def get_ssa_stats(self) -> Dict[str, Any]:
        """Get SSA converter statistics"""
        if not self.ssa_converter:
            return {'error': 'SSA converter not enabled'}
        
        return self.ssa_converter.get_stats()
    
    def get_allocator_stats(self) -> Dict[str, Any]:
        """Get register allocator statistics"""
        if not self.register_allocator:
            return {'error': 'Register allocator not enabled'}
        
        return self.register_allocator.get_stats()
    
    def reset_ssa_stats(self):
        """Reset SSA converter statistics"""
        if self.ssa_converter:
            self.ssa_converter.reset_stats()
    
    def reset_allocator_stats(self):
        """Reset register allocator statistics"""
        if self.register_allocator:
            self.register_allocator.reset_stats()

# ==================== Factory Functions ====================

def create_vm(mode: str = "auto", use_jit: bool = True, **kwargs) -> VM:
    return VM(mode=VMMode(mode.lower()), use_jit=use_jit, **kwargs)

def create_high_performance_vm() -> VM:
    return create_vm(
        mode="auto",
        use_jit=True,
        use_memory_manager=True,
        enable_memory_pool=True,
        enable_peephole_optimizer=True,
        optimization_level="AGGRESSIVE",
        worker_count=4
    )