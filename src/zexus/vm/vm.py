"""
Integrated Extended VM for Zexus.

Capabilities:
 - Architecture: Stack, Register, and Parallel execution modes.
 - Compilation: Tiered compilation with JIT (Hot path detection).
 - Memory: Managed memory with Garbage Collection.
 - Formats: High-level ops list and Low-level Bytecode.
 - Features: Async primitives (SPAWN/AWAIT), Event System, Module Imports.
 - Blockchain: Ziver-Chain specific opcodes (Merkle, Hash, State, Gas).
"""

import sys
import time
import asyncio
import importlib
import hashlib
import types
from typing import List, Any, Dict, Tuple, Optional, Union, Callable
from enum import Enum

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

# Renderer Backend
try:
    from renderer import backend as _BACKEND
    _BACKEND_AVAILABLE = True
except Exception:
    _BACKEND_AVAILABLE = False
    _BACKEND = None


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
        mode: VMMode = VMMode.AUTO,
        worker_count: int = None,
        chunk_size: int = 50,
        num_registers: int = 16,
        hybrid_mode: bool = True,
        debug: bool = False,
        enable_profiling: bool = False,
        profiling_level: str = "DETAILED",
        enable_memory_pool: bool = True,
        pool_max_size: int = 1000,
        enable_peephole_optimizer: bool = True,
        optimization_level: str = "MODERATE",
        enable_async_optimizer: bool = True,
        async_optimization_level: str = "MODERATE",
        enable_ssa: bool = False,
        enable_register_allocation: bool = False,
        num_allocator_registers: int = 16
    ):
        """
        Initialize the enhanced VM.
        """
        # --- Environment Setup ---
        self.builtins = builtins or {}
        self.env = env or {}
        self._parent_env = parent_env
        self.debug = debug
        
        # --- State Tracking ---
        self._events: Dict[str, List[Any]] = {}  # Event registry
        self._tasks: Dict[str, asyncio.Task] = {} # Async tasks
        self._task_counter = 0
        self._closure_cells: Dict[str, Cell] = {} # Closure storage
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._mode_usage = {m.value: 0 for m in VMMode}
        
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
                gc_threshold=1000
            )

        # --- Profiler (Phase 8) ---
        self.enable_profiling = enable_profiling and _PROFILER_AVAILABLE
        self.profiler = None
        if self.enable_profiling:
            try:
                level = getattr(ProfilingLevel, profiling_level, ProfilingLevel.DETAILED)
                self.profiler = InstructionProfiler(level=level)
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
                if debug:
                    print(f"[VM] Register allocator enabled: {num_allocator_registers} registers")
            except Exception as e:
                if debug:
                    print(f"[VM] Failed to enable register allocator: {e}")
                self.enable_register_allocation = False

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

    def _get_cpu_count(self) -> int:
        import os
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1

    # ==================== Public Execution API ====================

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
                return asyncio.run(self._run_high_level_ops(code, debug or self.debug))
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
            
            # 3. Stack Mode (Standard/Fallback + Async Support)
            else:
                result = asyncio.run(self._execute_stack(code, debug))
            
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
            return asyncio.run(self._run_stack_bytecode(bytecode, debug))

    def _execute_parallel(self, bytecode, debug: bool = False):
        """Execute using parallel VM"""
        try:
            return self._parallel_vm.execute_parallel(
                bytecode,
                initial_state={"env": self.env.copy(), "builtins": self.builtins.copy(), "parent_env": self._parent_env}
            )
        except Exception as e:
            if debug: print(f"[VM Parallel] Failed: {e}, falling back to stack")
            return asyncio.run(self._run_stack_bytecode(bytecode, debug))

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

    def get_jit_stats(self) -> Dict[str, Any]:
        if self.use_jit and self.jit_compiler:
            stats = self.jit_compiler.get_stats()
            stats['vm_hot_paths_tracked'] = len(self._jit_execution_stats)
            stats['jit_enabled'] = True
            return stats
        return {'jit_enabled': False}

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
                    self.env.setdefault("enums", {})[name] = members
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

    # ==================== Core Execution: Stack Bytecode ====================

    async def _run_stack_bytecode(self, bytecode, debug=False):
        # 1. JIT Check (with thread safety)
        if self.use_jit and self.jit_compiler:
            with self._jit_lock:
                bytecode_hash = self.jit_compiler._hash_bytecode(bytecode)
                jit_function = self.jit_compiler.compilation_cache.get(bytecode_hash)
            
            if jit_function:
                try:
                    start_t = time.perf_counter()
                    stack = []
                    result = jit_function(self, stack, self.env)
                    with self._jit_lock:
                        self.jit_compiler.record_execution_time(bytecode_hash, time.perf_counter() - start_t, ExecutionTier.JIT_NATIVE)
                    if debug: print(f"[VM JIT] Executed cached function")
                    return result
                except Exception as e:
                    if debug: print(f"[VM JIT] Failed: {e}, falling back")

        # 2. Bytecode Execution Setup
        consts = list(getattr(bytecode, "constants", []))
        instrs = list(getattr(bytecode, "instructions", []))
        ip = 0
        stack: List[Any] = []

        def const(idx): return consts[idx] if 0 <= idx < len(consts) else None

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

        # 3. Execution Loop
        prev_ip = None
        while ip < len(instrs):
            op, operand = instrs[ip]
            if debug: print(f"[VM SL] ip={ip} op={op} operand={operand} stack={stack}")
            
            # Profile instruction (if enabled) - start timing
            instr_start_time = None
            if self.enable_profiling and self.profiler and self.profiler.enabled:
                if self.profiler.level in (ProfilingLevel.DETAILED, ProfilingLevel.FULL):
                    instr_start_time = time.perf_counter()
                # Record instruction (count only for BASIC level)
                self.profiler.record_instruction(ip, op, operand, prev_ip, len(stack))
            
            prev_ip = ip
            ip += 1

            # --- Basic Stack Ops ---
            if op == "LOAD_CONST":
                stack.append(const(operand))
            elif op == "LOAD_NAME":
                name = const(operand)
                stack.append(_resolve(name))
            elif op == "STORE_NAME":
                name = const(operand)
                val = stack.pop() if stack else None
                _store(name, val)
                if self.use_memory_manager and val is not None:
                    self._allocate_managed(val, name=name)
            elif op == "POP":
                if stack: stack.pop()
            elif op == "DUP":
                if stack: stack.append(stack[-1])
            elif op == "PRINT":
                val = stack.pop() if stack else None
                print(val)
            
            # --- Function/Closure Ops ---
            elif op == "STORE_FUNC":
                name_idx, func_idx = operand
                name = const(name_idx)
                func_desc = const(func_idx)
                # Create func descriptor, capturing current VM as parent
                func_desc_copy = dict(func_desc) if isinstance(func_desc, dict) else {"bytecode": func_desc}
                func_desc_copy["parent_vm"] = self
                self.env[name] = func_desc_copy
            
            elif op == "CALL_NAME":
                name_idx, arg_count = operand
                func_name = const(name_idx)
                args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
                fn = _resolve(func_name) or self.builtins.get(func_name)
                res = await self._invoke_callable_or_funcdesc(fn, args)
                stack.append(res)
            
            elif op == "CALL_TOP":
                arg_count = operand
                args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
                fn_obj = stack.pop() if stack else None
                res = await self._invoke_callable_or_funcdesc(fn_obj, args)
                stack.append(res)

            # --- Arithmetic & Logic ---
            elif op == "ADD":
                b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                # Auto-unwrap evaluator objects
                if hasattr(a, 'value'): a = a.value
                if hasattr(b, 'value'): b = b.value
                stack.append(a + b)
            elif op == "SUB":
                b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                if hasattr(a, 'value'): a = a.value
                if hasattr(b, 'value'): b = b.value
                stack.append(a - b)
            elif op == "MUL":
                b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                if hasattr(a, 'value'): a = a.value
                if hasattr(b, 'value'): b = b.value
                stack.append(a * b)
            elif op == "DIV":
                b = stack.pop() if stack else 1; a = stack.pop() if stack else 0
                if hasattr(a, 'value'): a = a.value
                if hasattr(b, 'value'): b = b.value
                stack.append(a / b if b != 0 else 0)
            elif op == "MOD":
                b = stack.pop() if stack else 1; a = stack.pop() if stack else 0
                stack.append(a % b if b != 0 else 0)
            elif op == "POW":
                b = stack.pop() if stack else 1; a = stack.pop() if stack else 0
                stack.append(a ** b)
            elif op == "NEG":
                a = stack.pop() if stack else 0
                stack.append(-a)
            elif op == "EQ":
                b = stack.pop() if stack else None; a = stack.pop() if stack else None
                stack.append(a == b)
            elif op == "NEQ":
                b = stack.pop() if stack else None; a = stack.pop() if stack else None
                stack.append(a != b)
            elif op == "LT":
                b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                stack.append(a < b)
            elif op == "GT":
                b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                stack.append(a > b)
            elif op == "LTE":
                b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                stack.append(a <= b)
            elif op == "GTE":
                b = stack.pop() if stack else 0; a = stack.pop() if stack else 0
                stack.append(a >= b)
            elif op == "NOT":
                a = stack.pop() if stack else False
                stack.append(not a)

            # --- Control Flow ---
            elif op == "JUMP":
                ip = operand
            elif op == "JUMP_IF_FALSE":
                cond = stack.pop() if stack else None
                if not cond: ip = operand
            elif op == "RETURN":
                return stack.pop() if stack else None

            # --- Collections ---
            elif op == "BUILD_LIST":
                count = operand if operand is not None else 0
                elements = [stack.pop() for _ in range(count)][::-1]
                stack.append(elements)
            elif op == "BUILD_MAP":
                count = operand if operand is not None else 0
                result = {}
                for _ in range(count):
                    val = stack.pop(); key = stack.pop()
                    result[key] = val
                stack.append(result)
            elif op == "INDEX":
                idx = stack.pop(); obj = stack.pop()
                try: stack.append(obj[idx] if obj is not None else None)
                except (IndexError, KeyError, TypeError): stack.append(None)
            elif op == "GET_LENGTH":
                obj = stack.pop()
                try:
                    if obj is None:
                        stack.append(0)
                    elif hasattr(obj, '__len__'):
                        stack.append(len(obj))
                    else:
                        stack.append(0)
                except (TypeError, AttributeError):
                    stack.append(0)

            # --- Async & Events ---
            elif op == "SPAWN":
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

            elif op == "AWAIT":
                # Keep popping until we find a task to await
                result_found = False
                temp_stack = []
                
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

            elif op == "REGISTER_EVENT":
                event_name = const(operand[0]) if isinstance(operand, (list,tuple)) else const(operand)
                handler = const(operand[1]) if isinstance(operand, (list,tuple)) else None
                self._events.setdefault(event_name, []).append(handler)

            elif op == "EMIT_EVENT":
                event_name = const(operand[0])
                payload = const(operand[1]) if isinstance(operand, (list,tuple)) and len(operand) > 1 else None
                handlers = self._events.get(event_name, [])
                for h in handlers:
                    fn = self.builtins.get(h) or self.env.get(h)
                    asyncio.create_task(self._call_builtin_async_obj(fn, [payload]))

            elif op == "IMPORT":
                mod_name = const(operand[0])
                alias = const(operand[1]) if isinstance(operand, (list,tuple)) and len(operand) > 1 else None
                try:
                    mod = importlib.import_module(mod_name)
                    self.env[alias or mod_name] = mod
                except Exception:
                    self.env[alias or mod_name] = None

            elif op == "DEFINE_ENUM":
                enum_name = const(operand[0])
                enum_map = const(operand[1])
                self.env[enum_name] = enum_map

            elif op == "ASSERT_PROTOCOL":
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
            
            elif op == "HASH_BLOCK":
                block_data = stack.pop() if stack else ""
                if isinstance(block_data, dict):
                    import json; block_data = json.dumps(block_data, sort_keys=True)
                if not isinstance(block_data, (bytes, str)): block_data = str(block_data)
                if isinstance(block_data, str): block_data = block_data.encode('utf-8')
                stack.append(hashlib.sha256(block_data).hexdigest())

            elif op == "VERIFY_SIGNATURE":
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

            elif op == "MERKLE_ROOT":
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

            elif op == "STATE_READ":
                key = const(operand)
                stack.append(self.env.setdefault("_blockchain_state", {}).get(key))

            elif op == "STATE_WRITE":
                key = const(operand)
                val = stack.pop() if stack else None
                if self.env.get("_in_transaction", False):
                    self.env.setdefault("_tx_pending_state", {})[key] = val
                else:
                    self.env.setdefault("_blockchain_state", {})[key] = val

            elif op == "TX_BEGIN":
                self.env["_in_transaction"] = True
                self.env["_tx_pending_state"] = {}
                self.env["_tx_snapshot"] = dict(self.env.get("_blockchain_state", {}))
                if self.use_memory_manager: self.env["_tx_memory_snapshot"] = dict(self._managed_objects)

            elif op == "TX_COMMIT":
                if self.env.get("_in_transaction", False):
                    self.env.setdefault("_blockchain_state", {}).update(self.env.get("_tx_pending_state", {}))
                    self.env["_in_transaction"] = False
                    self.env["_tx_pending_state"] = {}
                    if "_tx_memory_snapshot" in self.env: del self.env["_tx_memory_snapshot"]

            elif op == "TX_REVERT":
                if self.env.get("_in_transaction", False):
                    self.env["_blockchain_state"] = dict(self.env.get("_tx_snapshot", {}))
                    self.env["_in_transaction"] = False
                    self.env["_tx_pending_state"] = {}
                    if self.use_memory_manager and "_tx_memory_snapshot" in self.env:
                        self._managed_objects = dict(self.env["_tx_memory_snapshot"])

            elif op == "GAS_CHARGE":
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

            elif op == "LEDGER_APPEND":
                entry = stack.pop() if stack else None
                if isinstance(entry, dict) and "timestamp" not in entry:
                    entry["timestamp"] = time.time()
                self.env.setdefault("_ledger", []).append(entry)

            else:
                if debug: print(f"[VM] Unknown Opcode: {op}")

            # Record instruction timing (if profiling enabled)
            if instr_start_time is not None and self.profiler:
                elapsed = time.perf_counter() - instr_start_time
                self.profiler.measure_instruction(ip, elapsed)

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
            return await inner_vm._run_stack_bytecode(func_bc, debug=False)
        
        # 2. Python Callable / Builtin Wrapper
        return await self._call_builtin_async_obj(fn, args)

    async def _call_builtin_async(self, name: str, args: List[Any]):
        target = self.builtins.get(name) or self.env.get(name)
        
        # Check Renderer Backend
        if _BACKEND_AVAILABLE and hasattr(_BACKEND, name):
            fn = getattr(_BACKEND, name)
            if asyncio.iscoroutinefunction(fn): return await fn(*args)
            return fn(*args)
            
        return await self._call_builtin_async_obj(target, args)

    async def _call_builtin_async_obj(self, fn_obj, args: List[Any]):
        try:
            if fn_obj is None: return None
            
            # Extract .fn if it's a wrapper
            real_fn = fn_obj.fn if hasattr(fn_obj, "fn") else fn_obj
            
            if not callable(real_fn): return real_fn
            
            res = real_fn(*args)
            if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
                return await res
            return res
        except Exception as e:
            return e

    def _to_coro(self, fn, args):
        if asyncio.iscoroutinefunction(fn):
            return fn(*args)
        async def _wrap():
            if callable(fn): return fn(*args)
            return None
        return _wrap()

    def profile_execution(self, bytecode, iterations: int = 1000) -> Dict[str, Any]:
        """Profile execution performance across available modes"""
        import timeit
        results = {'iterations': iterations, 'modes': {}}
        
        # Stack
        def run_stack(): return asyncio.run(self._execute_stack(bytecode))
        t_stack = timeit.timeit(run_stack, number=iterations)
        results['modes']['stack'] = {'total': t_stack, 'avg': t_stack/iterations}
        
        # Register
        if self._register_vm:
            def run_reg(): return self._execute_register(bytecode)
            t_reg = timeit.timeit(run_reg, number=iterations)
            results['modes']['register'] = {'total': t_reg, 'speedup': t_stack/t_reg}
            
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