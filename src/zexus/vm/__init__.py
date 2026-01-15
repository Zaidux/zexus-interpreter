"""
Zexus Virtual Machine - Backend Execution Engine

This package provides a comprehensive VM system for Zexus with:
- Stack-based VM (standard execution)
- Register-based VM (optimized paths)
- Parallel VM (multi-core execution)
- JIT compilation (hot path native code)
- Peephole optimizer (bytecode optimization)
- Bytecode caching (persistent across runs)
- Memory pooling (efficient allocation)
- Gas metering (resource control)
"""

# Core VM
from .vm import VM as ZexusVM, VMMode
from .bytecode import Bytecode, Opcode, BytecodeBuilder

# Caching system (with file-based persistence)
from .cache import BytecodeCache, CacheStats, FileMetadata

# Compilers
from .compiler import BytecodeCompiler

# Optimizers
try:
    from .optimizer import BytecodeOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    BytecodeOptimizer = None
    OPTIMIZER_AVAILABLE = False

try:
    from .peephole_optimizer import PeepholeOptimizer, OptimizationLevel
    PEEPHOLE_AVAILABLE = True
except ImportError:
    PeepholeOptimizer = None
    OptimizationLevel = None
    PEEPHOLE_AVAILABLE = False

# JIT
try:
    from .jit import JITCompiler, ExecutionTier
    JIT_AVAILABLE = True
except ImportError:
    JITCompiler = None
    ExecutionTier = None
    JIT_AVAILABLE = False

# Register VM
try:
    from .register_vm import RegisterVM
    REGISTER_VM_AVAILABLE = True
except ImportError:
    RegisterVM = None
    REGISTER_VM_AVAILABLE = False

# Parallel VM
try:
    from .parallel_vm import ParallelVM, ExecutionMode
    PARALLEL_VM_AVAILABLE = True
except ImportError:
    ParallelVM = None
    ExecutionMode = None
    PARALLEL_VM_AVAILABLE = False

# Memory management
try:
    from .memory_pool import IntegerPool, StringPool, ListPool
    MEMORY_POOL_AVAILABLE = True
except ImportError:
    IntegerPool = None
    StringPool = None
    ListPool = None
    MEMORY_POOL_AVAILABLE = False

# Gas metering
try:
    from .gas_metering import GasMetering, OutOfGasError
    GAS_METERING_AVAILABLE = True
except ImportError:
    GasMetering = None
    OutOfGasError = None
    GAS_METERING_AVAILABLE = False

# Profiler
try:
    from .profiler import InstructionProfiler, ProfilingLevel
    PROFILER_AVAILABLE = True
except ImportError:
    InstructionProfiler = None
    ProfilingLevel = None
    PROFILER_AVAILABLE = False

__all__ = [
    # Core
    'ZexusVM', 'VMMode', 'Bytecode', 'Opcode', 'BytecodeBuilder',
    # Cache
    'BytecodeCache', 'CacheStats', 'FileMetadata',
    # Compiler
    'BytecodeCompiler',
    # Optimizers
    'BytecodeOptimizer', 'PeepholeOptimizer', 'OptimizationLevel',
    # JIT
    'JITCompiler', 'ExecutionTier',
    # Advanced VMs
    'RegisterVM', 'ParallelVM', 'ExecutionMode',
    # Memory
    'IntegerPool', 'StringPool', 'ListPool',
    # Gas
    'GasMetering', 'OutOfGasError',
    # Profiler
    'InstructionProfiler', 'ProfilingLevel',
    # Availability flags
    'OPTIMIZER_AVAILABLE', 'PEEPHOLE_AVAILABLE', 'JIT_AVAILABLE',
    'REGISTER_VM_AVAILABLE', 'PARALLEL_VM_AVAILABLE', 
    'MEMORY_POOL_AVAILABLE', 'GAS_METERING_AVAILABLE', 'PROFILER_AVAILABLE',
]
