"""
Bytecode Optimizer for Zexus VM

Implements advanced optimization passes to reduce bytecode size and improve execution speed:
1. Constant Folding - Pre-compute constant expressions
2. Copy Propagation - Eliminate redundant copies
3. Common Subexpression Elimination (CSE) - Reuse computed values
4. Dead Code Elimination (DCE) - Remove unreachable code
5. Peephole Optimization - Local pattern matching
6. Instruction Combining - Merge adjacent instructions
7. Jump Threading - Optimize jump chains
8. Strength Reduction - Replace expensive ops with cheaper equivalents
"""

from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, field
import copy


@dataclass
class OptimizationStats:
    """Track optimization statistics"""
    constant_folds: int = 0
    copies_eliminated: int = 0
    common_subexpressions: int = 0
    dead_code_removed: int = 0
    peephole_opts: int = 0
    instructions_combined: int = 0
    jumps_threaded: int = 0
    strength_reductions: int = 0
    original_size: int = 0
    optimized_size: int = 0
    passes_applied: int = 0
    
    @property
    def size_reduction(self) -> float:
        """Calculate bytecode size reduction percentage"""
        if self.original_size == 0:
            return 0.0
        return ((self.original_size - self.optimized_size) / self.original_size) * 100
    
    @property
    def total_optimizations(self) -> int:
        """Total number of optimizations applied"""
        return (self.constant_folds + self.copies_eliminated + 
                self.common_subexpressions + self.dead_code_removed +
                self.peephole_opts + self.instructions_combined +
                self.jumps_threaded + self.strength_reductions)


class BytecodeOptimizer:
    """
    Advanced bytecode optimizer with multiple optimization passes
    
    Usage:
        optimizer = BytecodeOptimizer(level=2)
        optimized_instructions = optimizer.optimize(instructions, constants)
    """
    
    def __init__(self, level: int = 2, max_passes: int = 5, debug: bool = False):
        """
        Initialize optimizer
        
        Args:
            level: Optimization level (0=none, 1=basic, 2=aggressive, 3=experimental)
            max_passes: Maximum number of optimization passes
            debug: Print optimization details
        """
        self.level = level
        self.max_passes = max_passes
        self.debug = debug
        self.stats = OptimizationStats()
    
    def optimize(self, instructions: List[Tuple[str, Any]], constants: List[Any] = None) -> List[Tuple[str, Any]]:
        """
        Apply all optimization passes to bytecode
        
        Args:
            instructions: List of (opcode, operand) tuples
            constants: Constant pool (optional)
            
        Returns:
            Optimized instruction list
        """
        if self.level == 0:
            return instructions  # No optimization
        
        if constants is None:
            constants = []
        
        self.stats.original_size = len(instructions)
        optimized = list(instructions)
        
        # Apply optimization passes repeatedly until no more changes
        for pass_num in range(self.max_passes):
            prev_size = len(optimized)
            
            # Level 1: Basic optimizations
            if self.level >= 1:
                optimized = self._constant_folding(optimized, constants)
                optimized = self._dead_code_elimination(optimized)
                optimized = self._peephole_optimization(optimized)
            
            # Level 2: Aggressive optimizations
            if self.level >= 2:
                optimized = self._copy_propagation(optimized)
                optimized = self._instruction_combining(optimized, constants)
                optimized = self._jump_threading(optimized)
                optimized = self._strength_reduction(optimized)
            
            # Level 3: Experimental optimizations
            if self.level >= 3:
                optimized = self._common_subexpression_elimination(optimized)
                optimized = self._loop_invariant_code_motion(optimized)
            
            self.stats.passes_applied += 1
            
            # Stop if no changes made
            if len(optimized) == prev_size:
                break
            
            if self.debug:
                print(f"Pass {pass_num + 1}: {prev_size} → {len(optimized)} instructions")
        
        self.stats.optimized_size = len(optimized)
        
        if self.debug:
            print(f"Optimization complete: {self.stats.original_size} → {self.stats.optimized_size} "
                  f"({self.stats.size_reduction:.1f}% reduction)")
            print(f"Total optimizations: {self.stats.total_optimizations}")
        
        return optimized
    
    def _constant_folding(self, instructions: List[Tuple[str, Any]], constants: List[Any]) -> List[Tuple[str, Any]]:
        """
        Fold constant expressions at compile time
        
        Examples:
            LOAD_CONST 2, LOAD_CONST 3, ADD → LOAD_CONST 5
            LOAD_CONST 10, NEG → LOAD_CONST -10
        """
        result = []
        i = 0
        
        while i < len(instructions):
            # Binary operations on constants
            if i + 2 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                op3, operand3 = instructions[i + 2]
                
                if op1 == "LOAD_CONST" and op2 == "LOAD_CONST":
                    val1 = constants[operand1] if operand1 < len(constants) else operand1
                    val2 = constants[operand2] if operand2 < len(constants) else operand2
                    
                    folded_value = None
                    if op3 == "ADD" and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        folded_value = val1 + val2
                    elif op3 == "SUB" and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        folded_value = val1 - val2
                    elif op3 == "MUL" and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        folded_value = val1 * val2
                    elif op3 == "DIV" and isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and val2 != 0:
                        folded_value = val1 / val2
                    elif op3 == "MOD" and isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and val2 != 0:
                        folded_value = val1 % val2
                    elif op3 == "POW" and isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        try:
                            folded_value = val1 ** val2
                        except (OverflowError, ValueError):
                            pass
                    
                    if folded_value is not None:
                        # Add to constants and emit single LOAD_CONST
                        const_idx = len(constants)
                        constants.append(folded_value)
                        result.append(("LOAD_CONST", const_idx))
                        self.stats.constant_folds += 1
                        i += 3
                        continue
            
            # Unary operations on constants
            if i + 1 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                
                if op1 == "LOAD_CONST":
                    val = constants[operand1] if operand1 < len(constants) else operand1
                    
                    folded_value = None
                    if op2 == "NEG" and isinstance(val, (int, float)):
                        folded_value = -val
                    elif op2 == "NOT":
                        folded_value = not val
                    
                    if folded_value is not None:
                        const_idx = len(constants)
                        constants.append(folded_value)
                        result.append(("LOAD_CONST", const_idx))
                        self.stats.constant_folds += 1
                        i += 2
                        continue
            
            result.append(instructions[i])
            i += 1
        
        return result
    
    def _copy_propagation(self, instructions: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Eliminate redundant copies
        
        Examples:
            STORE_NAME x, LOAD_NAME x → STORE_NAME x, DUP
            x = y; use x → x = y; use y (if y not modified)
        """
        result = []
        i = 0
        
        while i < len(instructions):
            # Pattern: STORE_NAME immediately followed by LOAD_NAME of same variable
            if i + 1 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                
                if op1 == "STORE_NAME" and op2 == "LOAD_NAME" and operand1 == operand2:
                    # Replace LOAD_NAME with DUP
                    result.append(instructions[i])
                    result.append(("DUP", None))
                    self.stats.copies_eliminated += 1
                    i += 2
                    continue
            
            result.append(instructions[i])
            i += 1
        
        return result
    
    def _common_subexpression_elimination(self, instructions: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Eliminate common subexpressions
        
        If a+b is computed twice without modification, compute once and reuse
        """
        # This is a simplified version - full CSE requires dataflow analysis
        result = []
        seen_expressions: Dict[str, int] = {}  # expression -> temp variable index
        
        i = 0
        while i < len(instructions):
            # Look for repeated binary operations
            if i + 2 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                op3, operand3 = instructions[i + 2]
                
                # Simple pattern: LOAD, LOAD, OP
                if op1 == "LOAD_NAME" and op2 == "LOAD_NAME" and op3 in ("ADD", "SUB", "MUL", "DIV"):
                    expr_key = f"{operand1}_{op3}_{operand2}"
                    
                    if expr_key in seen_expressions:
                        # Reuse previous result
                        result.append(("LOAD_NAME", f"_cse_{seen_expressions[expr_key]}"))
                        self.stats.common_subexpressions += 1
                        i += 3
                        continue
                    else:
                        # First occurrence - compute and store
                        result.extend([instructions[i], instructions[i + 1], instructions[i + 2]])
                        temp_var = len(seen_expressions)
                        result.append(("STORE_NAME", f"_cse_{temp_var}"))
                        result.append(("LOAD_NAME", f"_cse_{temp_var}"))
                        seen_expressions[expr_key] = temp_var
                        i += 3
                        continue
            
            result.append(instructions[i])
            i += 1
        
        return result
    
    def _dead_code_elimination(self, instructions: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Remove unreachable code after RETURN, unconditional JUMP, etc.
        """
        result = []
        in_dead_code = False
        
        for op, operand in instructions:
            if in_dead_code:
                # Skip until we hit a jump target or label
                if op in ("LABEL", "JUMP_TARGET"):
                    in_dead_code = False
                    result.append((op, operand))
                else:
                    self.stats.dead_code_removed += 1
            else:
                result.append((op, operand))
                # Mark dead code after unconditional control flow
                if op in ("RETURN", "JUMP"):
                    in_dead_code = True
        
        return result
    
    def _peephole_optimization(self, instructions: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Local pattern-based optimizations
        
        Examples:
            LOAD_NAME x, POP → (removed)
            PUSH, POP → (removed)
            JUMP to next instruction → (removed)
        """
        result = []
        i = 0
        
        while i < len(instructions):
            # Pattern: LOAD followed by POP (useless load)
            if i + 1 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                
                if op1 in ("LOAD_CONST", "LOAD_NAME") and op2 == "POP":
                    self.stats.peephole_opts += 1
                    i += 2
                    continue
                
                # Pattern: DUP followed by POP (cancel out)
                if op1 == "DUP" and op2 == "POP":
                    self.stats.peephole_opts += 1
                    i += 2
                    continue
                
                # Pattern: JUMP to next instruction (useless jump)
                if op1 == "JUMP" and i + 1 == operand1:
                    self.stats.peephole_opts += 1
                    i += 1
                    continue
            
            result.append(instructions[i])
            i += 1
        
        return result
    
    def _instruction_combining(self, instructions: List[Tuple[str, Any]], constants: List[Any]) -> List[Tuple[str, Any]]:
        """
        Combine adjacent instructions into specialized instructions
        
        Examples:
            LOAD_CONST x, STORE_NAME y → STORE_CONST y, x
            LOAD_CONST 1, ADD → INC
            LOAD_CONST 1, SUB → DEC
        """
        result = []
        i = 0
        
        while i < len(instructions):
            # Pattern: LOAD_CONST followed by STORE_NAME
            if i + 1 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                
                if op1 == "LOAD_CONST" and op2 == "STORE_NAME":
                    result.append(("STORE_CONST", (operand2, operand1)))
                    self.stats.instructions_combined += 1
                    i += 2
                    continue
                
                # TODO: Pattern: x, LOAD_CONST 1, ADD → x, INC (needs better pattern matching)
                # Currently disabled - INC/DEC need value already on stack
                # if op1 == "LOAD_CONST" and op2 == "ADD":
                #     val = constants[operand1] if operand1 < len(constants) else operand1
                #     if val == 1:
                #         result.append(("INC", None))
                #         self.stats.instructions_combined += 1
                #         i += 2
                #         continue
            
            result.append(instructions[i])
            i += 1
        
        return result
    
    def _jump_threading(self, instructions: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Optimize jump chains
        
        Examples:
            JUMP label1 → label1: JUMP label2 becomes JUMP label2
        """
        # Build jump target map
        jump_targets: Dict[int, int] = {}
        
        for i, (op, operand) in enumerate(instructions):
            if op == "JUMP":
                target = operand
                # Follow jump chain
                visited = {i}
                while target not in visited and target < len(instructions):
                    if instructions[target][0] == "JUMP":
                        visited.add(target)
                        target = instructions[target][1]
                    else:
                        break
                
                if target != operand:
                    jump_targets[i] = target
                    self.stats.jumps_threaded += 1
        
        # Apply jump threading
        result = []
        for i, (op, operand) in enumerate(instructions):
            if i in jump_targets:
                result.append((op, jump_targets[i]))
            else:
                result.append((op, operand))
        
        return result
    
    def _strength_reduction(self, instructions: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Replace expensive operations with cheaper equivalents
        
        Examples:
            x * 2 → x + x (addition cheaper than multiplication)
            x ** 2 → x * x
            x / 2 → x * 0.5 (if floating point)
        """
        result = []
        i = 0
        
        while i < len(instructions):
            # Pattern: multiply by power of 2
            if i + 2 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                op3, operand3 = instructions[i + 2]
                
                # x * 2 → x + x (cheaper)
                if op1 == "LOAD_NAME" and op2 == "LOAD_CONST" and op3 == "MUL":
                    # This is simplified - would need constant value check
                    pass
            
            result.append(instructions[i])
            i += 1
        
        return result
    
    def _loop_invariant_code_motion(self, instructions: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """
        Move loop-invariant computations outside loops
        
        This is experimental and requires loop detection
        """
        # Placeholder for future implementation
        return instructions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'constant_folds': self.stats.constant_folds,
            'copies_eliminated': self.stats.copies_eliminated,
            'common_subexpressions': self.stats.common_subexpressions,
            'dead_code_removed': self.stats.dead_code_removed,
            'peephole_opts': self.stats.peephole_opts,
            'instructions_combined': self.stats.instructions_combined,
            'jumps_threaded': self.stats.jumps_threaded,
            'strength_reductions': self.stats.strength_reductions,
            'original_size': self.stats.original_size,
            'optimized_size': self.stats.optimized_size,
            'size_reduction_pct': self.stats.size_reduction,
            'total_optimizations': self.stats.total_optimizations,
            'passes_applied': self.stats.passes_applied
        }
    
    def reset_stats(self):
        """Reset optimization statistics"""
        self.stats = OptimizationStats()
