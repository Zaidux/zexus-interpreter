"""
Internal optimization system for Zexus.

Provides constant folding, dead code elimination, inlining, and bytecode compilation.
"""

from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib


class OptimizationType(Enum):
    """Types of code optimizations."""
    CONSTANT_FOLDING = "constant_folding"
    DEAD_CODE_ELIMINATION = "dce"
    INLINING = "inlining"
    LOOP_OPTIMIZATION = "loop_opt"
    COMMON_SUBEXPR_ELIM = "cse"


@dataclass
class BytecodeOp:
    """Single bytecode operation."""
    opcode: str  # Operation code (LOAD, STORE, ADD, CALL, etc.)
    operands: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        """String representation."""
        if self.operands:
            ops_str = ", ".join(str(op) for op in self.operands)
            return f"{self.opcode}({ops_str})"
        return self.opcode


@dataclass
class CompiledFunction:
    """Compiled function with bytecode."""
    name: str
    bytecode: List[BytecodeOp]
    constants: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    optimization_passes: List[str] = field(default_factory=list)
    
    def get_bytecode_size(self) -> int:
        """Get size of bytecode in operations."""
        return len(self.bytecode)
    
    def add_pass(self, pass_name: str):
        """Record an optimization pass applied."""
        if pass_name not in self.optimization_passes:
            self.optimization_passes.append(pass_name)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CompiledFunction({self.name}, {len(self.bytecode)} ops)"


class OptimizationPass(ABC):
    """Base class for optimization passes."""
    
    @abstractmethod
    def optimize(self, bytecode: List[BytecodeOp]) -> List[BytecodeOp]:
        """Apply optimization pass to bytecode."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get name of optimization."""
        pass


class ConstantFoldingPass(OptimizationPass):
    """Fold constant expressions at compile time."""
    
    def optimize(self, bytecode: List[BytecodeOp]) -> List[BytecodeOp]:
        """Apply constant folding."""
        result = []
        i = 0
        while i < len(bytecode):
            # Pattern: LOAD_CONST, LOAD_CONST, BINOP -> LOAD_CONST
            if i + 2 < len(bytecode):
                op1 = bytecode[i]
                op2 = bytecode[i + 1]
                op3 = bytecode[i + 2]
                
                if (op1.opcode == "LOAD_CONST" and
                    op2.opcode == "LOAD_CONST" and
                    op3.opcode in ("ADD", "SUB", "MUL", "DIV", "MOD")):
                    
                    try:
                        val1 = op1.operands[0]
                        val2 = op2.operands[0]
                        
                        # Compute constant expression
                        if op3.opcode == "ADD":
                            folded = val1 + val2
                        elif op3.opcode == "SUB":
                            folded = val1 - val2
                        elif op3.opcode == "MUL":
                            folded = val1 * val2
                        elif op3.opcode == "DIV":
                            folded = val1 / val2 if val2 != 0 else None
                        else:  # MOD
                            folded = val1 % val2 if val2 != 0 else None
                        
                        if folded is not None:
                            result.append(BytecodeOp("LOAD_CONST", [folded]))
                            i += 3
                            continue
                    except Exception:
                        pass
            
            result.append(bytecode[i])
            i += 1
        
        return result
    
    def get_name(self) -> str:
        """Get name."""
        return "ConstantFolding"


class DeadCodeEliminationPass(OptimizationPass):
    """Remove unreachable code after return/break."""
    
    def optimize(self, bytecode: List[BytecodeOp]) -> List[BytecodeOp]:
        """Apply dead code elimination."""
        result = []
        
        for op in bytecode:
            result.append(op)
            
            # Stop adding operations after unconditional return/break
            if op.opcode in ("RETURN", "BREAK", "BREAK_LOOP"):
                break
        
        return result
    
    def get_name(self) -> str:
        """Get name."""
        return "DeadCodeElimination"


class InliningPass(OptimizationPass):
    """Inline small function calls."""
    
    def __init__(self, size_threshold: int = 5):
        """
        Initialize inlining pass.
        
        Args:
            size_threshold: Max bytecode size to inline
        """
        self.size_threshold = size_threshold
        self.function_bytecode: Dict[str, List[BytecodeOp]] = {}
    
    def register_function(self, name: str, bytecode: List[BytecodeOp]):
        """Register a function for potential inlining."""
        self.function_bytecode[name] = bytecode
    
    def optimize(self, bytecode: List[BytecodeOp]) -> List[BytecodeOp]:
        """Apply inlining."""
        result = []
        
        for op in bytecode:
            # Inline small function calls
            if op.opcode == "CALL" and len(op.operands) > 0:
                func_name = op.operands[0]
                
                if (func_name in self.function_bytecode and
                    len(self.function_bytecode[func_name]) <= self.size_threshold):
                    
                    # Inline the function body
                    result.extend(self.function_bytecode[func_name])
                    continue
            
            result.append(op)
        
        return result
    
    def get_name(self) -> str:
        """Get name."""
        return "Inlining"


class LoopOptimizationPass(OptimizationPass):
    """Optimize loop structures."""
    
    def optimize(self, bytecode: List[BytecodeOp]) -> List[BytecodeOp]:
        """Apply loop optimizations."""
        result = []
        
        i = 0
        while i < len(bytecode):
            # Detect LOOP pattern
            if bytecode[i].opcode == "LOOP":
                loop_start = i
                loop_size = 1
                
                # Find loop end
                i += 1
                while i < len(bytecode) and bytecode[i].opcode != "LOOP_END":
                    loop_size += 1
                    i += 1
                
                # Add loop with optimization metadata
                loop_op = BytecodeOp("LOOP", bytecode[loop_start].operands)
                loop_op.metadata["optimized"] = True
                loop_op.metadata["size"] = loop_size
                result.append(loop_op)
            else:
                result.append(bytecode[i])
                i += 1
        
        return result
    
    def get_name(self) -> str:
        """Get name."""
        return "LoopOptimization"


class CommonSubexprEliminationPass(OptimizationPass):
    """Eliminate redundant subexpressions."""
    
    def optimize(self, bytecode: List[BytecodeOp]) -> List[BytecodeOp]:
        """Apply common subexpression elimination."""
        result = []
        expr_map: Dict[str, str] = {}  # Expression hash -> variable
        
        for op in bytecode:
            # For binary operations, check if already computed
            if op.opcode in ("ADD", "SUB", "MUL", "DIV"):
                expr_key = f"{op.opcode}:{op.operands}"
                expr_hash = hashlib.md5(expr_key.encode()).hexdigest()[:8]
                
                if expr_hash in expr_map:
                    # Use previously computed result
                    result.append(BytecodeOp("LOAD_VAR", [expr_map[expr_hash]]))
                    continue
                else:
                    # First occurrence - store result
                    expr_map[expr_hash] = f"_cse_{expr_hash}"
                    result.append(op)
                    result.append(BytecodeOp("STORE_VAR", [f"_cse_{expr_hash}"]))
            else:
                result.append(op)
        
        return result
    
    def get_name(self) -> str:
        """Get name."""
        return "CommonSubexprElimination"


class OptimizationPipeline:
    """Pipeline for running multiple optimization passes."""
    
    def __init__(self):
        """Initialize optimization pipeline."""
        self.passes: List[OptimizationPass] = []
        self.enabled_passes: Set[OptimizationType] = set(OptimizationType)
    
    def add_pass(self, pass_obj: OptimizationPass) -> 'OptimizationPipeline':
        """Add optimization pass."""
        self.passes.append(pass_obj)
        return self
    
    def enable_pass(self, pass_type: OptimizationType) -> 'OptimizationPipeline':
        """Enable specific optimization type."""
        self.enabled_passes.add(pass_type)
        return self
    
    def disable_pass(self, pass_type: OptimizationType) -> 'OptimizationPipeline':
        """Disable specific optimization type."""
        self.enabled_passes.discard(pass_type)
        return self
    
    def optimize(self, bytecode: List[BytecodeOp]) -> List[BytecodeOp]:
        """Run all enabled optimization passes."""
        result = bytecode
        
        for pass_obj in self.passes:
            result = pass_obj.optimize(result)
        
        return result


class BytecodeCompiler:
    """Compile AST/functions to bytecode."""
    
    def __init__(self):
        """Initialize bytecode compiler."""
        self.compiled_functions: Dict[str, CompiledFunction] = {}
        self.optimization_pipeline = OptimizationPipeline()
        self._setup_default_passes()
    
    def _setup_default_passes(self):
        """Setup default optimization passes."""
        self.optimization_pipeline.add_pass(ConstantFoldingPass())
        self.optimization_pipeline.add_pass(DeadCodeEliminationPass())
        self.optimization_pipeline.add_pass(LoopOptimizationPass())
        self.optimization_pipeline.add_pass(CommonSubexprEliminationPass())
    
    def compile_function(self, name: str, bytecode: List[BytecodeOp],
                        optimize: bool = True) -> CompiledFunction:
        """Compile function with optional optimization."""
        initial_size = len(bytecode)
        
        if optimize:
            optimized = self.optimization_pipeline.optimize(bytecode)
        else:
            optimized = bytecode
        
        compiled = CompiledFunction(
            name=name,
            bytecode=optimized,
            metadata={
                "initial_size": initial_size,
                "final_size": len(optimized),
                "reduction": initial_size - len(optimized)
            }
        )
        
        self.compiled_functions[name] = compiled
        return compiled
    
    def get_compiled_function(self, name: str) -> Optional[CompiledFunction]:
        """Get compiled function by name."""
        return self.compiled_functions.get(name)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        total_initial = 0
        total_final = 0
        
        for func in self.compiled_functions.values():
            total_initial += func.metadata.get("initial_size", 0)
            total_final += func.metadata.get("final_size", 0)
        
        return {
            "total_functions": len(self.compiled_functions),
            "total_initial_size": total_initial,
            "total_final_size": total_final,
            "total_reduction": total_initial - total_final,
            "reduction_percentage": (
                (total_initial - total_final) / total_initial * 100
                if total_initial > 0 else 0
            )
        }


class ExecutionProfile:
    """Profile for tracking function execution."""
    
    def __init__(self, name: str):
        """Initialize execution profile."""
        self.name = name
        self.call_count = 0
        self.total_time = 0.0
        self.optimization_enabled = True
    
    def record_call(self, execution_time: float):
        """Record a function call."""
        self.call_count += 1
        self.total_time += execution_time
    
    def get_avg_time(self) -> float:
        """Get average execution time."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def should_optimize(self) -> bool:
        """Determine if function should be optimized."""
        # Optimize hot functions (called frequently or slow)
        return self.call_count > 10 or self.total_time > 1.0


class OptimizationFramework:
    """Overall optimization framework."""
    
    def __init__(self):
        """Initialize optimization framework."""
        self.compiler = BytecodeCompiler()
        self.profiles: Dict[str, ExecutionProfile] = {}
    
    def create_profile(self, name: str) -> ExecutionProfile:
        """Create execution profile for function."""
        profile = ExecutionProfile(name)
        self.profiles[name] = profile
        return profile
    
    def get_profile(self, name: str) -> Optional[ExecutionProfile]:
        """Get execution profile."""
        return self.profiles.get(name)
    
    def get_hot_functions(self, threshold: int = 10) -> List[str]:
        """Get frequently called functions."""
        return [
            name for name, profile in self.profiles.items()
            if profile.call_count >= threshold
        ]
    
    def get_slow_functions(self, threshold: float = 1.0) -> List[str]:
        """Get slow functions."""
        return [
            name for name, profile in self.profiles.items()
            if profile.total_time >= threshold
        ]


# Global optimization framework
_global_optimizer = OptimizationFramework()


def get_optimizer() -> OptimizationFramework:
    """Get global optimizer instance."""
    return _global_optimizer


def compile_function(name: str, bytecode: List[BytecodeOp]) -> CompiledFunction:
    """Compile function with optimizations."""
    return _global_optimizer.compiler.compile_function(name, bytecode)
