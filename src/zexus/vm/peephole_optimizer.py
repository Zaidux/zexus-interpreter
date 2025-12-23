"""
Bytecode Peephole Optimizer for Zexus VM

Implements peephole optimization patterns including:
- Constant folding
- Dead code elimination
- Strength reduction
- Instruction fusion
- Jump threading

Phase 8.3 of VM Optimization Project
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class OptimizationLevel(Enum):
    """Optimization levels"""
    NONE = 0      # No optimization
    BASIC = 1     # Safe, basic optimizations
    MODERATE = 2  # More aggressive optimizations
    AGGRESSIVE = 3  # Maximum optimization


@dataclass
class Instruction:
    """Bytecode instruction representation"""
    opcode: str
    arg: Optional[Any] = None
    lineno: int = 0
    
    def __repr__(self):
        if self.arg is not None:
            return f"{self.opcode}({self.arg})"
        return self.opcode


@dataclass
class OptimizationStats:
    """Statistics for optimizations performed"""
    constant_folds: int = 0
    dead_code_eliminated: int = 0
    strength_reductions: int = 0
    instruction_fusions: int = 0
    jump_threads: int = 0
    total_instructions_removed: int = 0
    original_size: int = 0
    optimized_size: int = 0
    
    def to_dict(self) -> dict:
        """Convert stats to dictionary"""
        return {
            'constant_folds': self.constant_folds,
            'dead_code_eliminated': self.dead_code_eliminated,
            'strength_reductions': self.strength_reductions,
            'instruction_fusions': self.instruction_fusions,
            'jump_threads': self.jump_threads,
            'total_instructions_removed': self.total_instructions_removed,
            'original_size': self.original_size,
            'optimized_size': self.optimized_size,
            'reduction_percent': self._reduction_percent(),
        }
    
    def _reduction_percent(self) -> float:
        """Calculate reduction percentage"""
        if self.original_size == 0:
            return 0.0
        removed = self.original_size - self.optimized_size
        return (removed / self.original_size) * 100.0


class PeepholeOptimizer:
    """
    Peephole optimizer for Zexus bytecode
    
    Analyzes small windows of instructions (typically 2-4) and applies
    local optimizations.
    """
    
    def __init__(self, level: OptimizationLevel = OptimizationLevel.BASIC):
        self.level = level
        self.stats = OptimizationStats()
        
        # Optimization pattern matchers
        self.patterns = {
            OptimizationLevel.BASIC: [
                self._fold_constants,
                self._eliminate_nops,
            ],
            OptimizationLevel.MODERATE: [
                self._fold_constants,
                self._eliminate_nops,
                self._eliminate_dead_stores,
                self._fuse_load_store,
            ],
            OptimizationLevel.AGGRESSIVE: [
                self._fold_constants,
                self._eliminate_nops,
                self._eliminate_dead_stores,
                self._fuse_load_store,
                self._strength_reduce,
                self._thread_jumps,
            ],
        }
    
    def optimize(self, instructions: List[Instruction]) -> List[Instruction]:
        """
        Optimize a list of instructions
        
        Args:
            instructions: List of bytecode instructions
            
        Returns:
            Optimized instruction list
        """
        if self.level == OptimizationLevel.NONE:
            return instructions
        
        # Reset stats
        self.stats = OptimizationStats()
        self.stats.original_size = len(instructions)
        
        # Apply optimization passes
        optimized = instructions[:]
        
        # Multiple passes until no more changes
        max_passes = 5
        for pass_num in range(max_passes):
            old_len = len(optimized)
            
            # Apply all patterns for this optimization level
            for pattern in self.patterns[self.level]:
                optimized = self._apply_pattern(optimized, pattern)
            
            # Stop if no changes
            if len(optimized) == old_len:
                break
        
        self.stats.optimized_size = len(optimized)
        self.stats.total_instructions_removed = (
            self.stats.original_size - self.stats.optimized_size
        )
        
        return optimized
    
    def _apply_pattern(
        self,
        instructions: List[Instruction],
        pattern_func
    ) -> List[Instruction]:
        """Apply a pattern to all instruction windows"""
        result = []
        i = 0
        
        while i < len(instructions):
            # Try to match pattern at current position
            matched, replacement, consumed = pattern_func(instructions, i)
            
            if matched:
                # Pattern matched - add replacement
                result.extend(replacement)
                i += consumed
            else:
                # No match - keep original
                result.append(instructions[i])
                i += 1
        
        return result
    
    # ========== Constant Folding ==========
    
    def _fold_constants(
        self,
        instructions: List[Instruction],
        pos: int
    ) -> Tuple[bool, List[Instruction], int]:
        """
        Fold constant arithmetic operations
        
        Patterns:
        - LOAD_CONST a, LOAD_CONST b, ADD -> LOAD_CONST (a+b)
        - LOAD_CONST a, LOAD_CONST b, SUB -> LOAD_CONST (a-b)
        - LOAD_CONST a, LOAD_CONST b, MUL -> LOAD_CONST (a*b)
        - LOAD_CONST a, LOAD_CONST b, DIV -> LOAD_CONST (a/b)
        """
        if pos + 2 >= len(instructions):
            return False, [], 0
        
        inst1 = instructions[pos]
        inst2 = instructions[pos + 1]
        inst3 = instructions[pos + 2]
        
        # Check for LOAD_CONST, LOAD_CONST, BINARY_OP pattern
        if (inst1.opcode == 'LOAD_CONST' and
            inst2.opcode == 'LOAD_CONST' and
            inst3.opcode in ('ADD', 'SUB', 'MUL', 'DIV', 'MOD',
                           'BINARY_ADD', 'BINARY_SUB', 'BINARY_MUL',
                           'BINARY_DIV', 'BINARY_MOD')):
            
            # Both operands must be constants
            a = inst1.arg
            b = inst2.arg
            
            # Only fold numeric constants
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                return False, [], 0
            
            try:
                # Compute result
                op = inst3.opcode
                if op in ('ADD', 'BINARY_ADD'):
                    result = a + b
                elif op in ('SUB', 'BINARY_SUB'):
                    result = a - b
                elif op in ('MUL', 'BINARY_MUL'):
                    result = a * b
                elif op in ('DIV', 'BINARY_DIV'):
                    if b == 0:
                        return False, [], 0
                    result = a / b
                elif op in ('MOD', 'BINARY_MOD'):
                    if b == 0:
                        return False, [], 0
                    result = a % b
                else:
                    return False, [], 0
                
                # Replace with single LOAD_CONST
                replacement = [Instruction('LOAD_CONST', result, inst1.lineno)]
                self.stats.constant_folds += 1
                return True, replacement, 3
                
            except (ZeroDivisionError, OverflowError, ValueError):
                # Can't fold if operation would fail
                return False, [], 0
        
        return False, [], 0
    
    # ========== Dead Code Elimination ==========
    
    def _eliminate_nops(
        self,
        instructions: List[Instruction],
        pos: int
    ) -> Tuple[bool, List[Instruction], int]:
        """
        Eliminate NOP instructions and useless operations
        
        Patterns:
        - NOP -> (removed)
        - LOAD_CONST x, POP_TOP -> (removed)
        - DUP_TOP, POP_TOP -> (removed)
        """
        inst = instructions[pos]
        
        # Remove NOPs
        if inst.opcode == 'NOP':
            self.stats.dead_code_eliminated += 1
            return True, [], 1
        
        # LOAD_CONST followed by POP_TOP
        if pos + 1 < len(instructions):
            next_inst = instructions[pos + 1]
            
            if inst.opcode == 'LOAD_CONST' and next_inst.opcode == 'POP_TOP':
                self.stats.dead_code_eliminated += 1
                return True, [], 2
            
            # DUP_TOP followed by POP_TOP
            if inst.opcode == 'DUP_TOP' and next_inst.opcode == 'POP_TOP':
                self.stats.dead_code_eliminated += 1
                return True, [], 2
        
        return False, [], 0
    
    def _eliminate_dead_stores(
        self,
        instructions: List[Instruction],
        pos: int
    ) -> Tuple[bool, List[Instruction], int]:
        """
        Eliminate stores to variables that are never read
        
        Patterns:
        - STORE_FAST x, LOAD_FAST y (x != y) -> STORE_FAST x (if x not used)
        - STORE_FAST x, STORE_FAST x -> STORE_FAST x
        """
        if pos + 1 >= len(instructions):
            return False, [], 0
        
        inst1 = instructions[pos]
        inst2 = instructions[pos + 1]
        
        # Double store to same variable
        if (inst1.opcode == 'STORE_FAST' and
            inst2.opcode == 'STORE_FAST' and
            inst1.arg == inst2.arg):
            # First store is overwritten - remove it
            # Keep second store but need the value for it
            self.stats.dead_code_eliminated += 1
            return True, [inst2], 2
        
        return False, [], 0
    
    # ========== Strength Reduction ==========
    
    def _strength_reduce(
        self,
        instructions: List[Instruction],
        pos: int
    ) -> Tuple[bool, List[Instruction], int]:
        """
        Replace expensive operations with cheaper equivalents
        
        Patterns:
        - x * 2 -> x << 1 (x + x)
        - x / 2 -> x >> 1
        - x * 1 -> x
        - x * 0 -> 0
        """
        if pos + 2 >= len(instructions):
            return False, [], 0
        
        inst1 = instructions[pos]
        inst2 = instructions[pos + 1]
        inst3 = instructions[pos + 2]
        
        # Check for LOAD_VAR, LOAD_CONST, MUL pattern
        if (inst1.opcode in ('LOAD_FAST', 'LOAD_NAME', 'LOAD_GLOBAL') and
            inst2.opcode == 'LOAD_CONST' and
            inst3.opcode in ('MUL', 'BINARY_MUL')):
            
            const = inst2.arg
            
            # x * 0 = 0
            if const == 0:
                replacement = [Instruction('LOAD_CONST', 0, inst1.lineno)]
                self.stats.strength_reductions += 1
                return True, replacement, 3
            
            # x * 1 = x
            if const == 1:
                replacement = [inst1]
                self.stats.strength_reductions += 1
                return True, replacement, 3
            
            # x * 2 = x + x
            if const == 2:
                replacement = [
                    inst1,
                    Instruction('DUP_TOP', None, inst1.lineno),
                    Instruction('ADD', None, inst1.lineno),
                ]
                self.stats.strength_reductions += 1
                return True, replacement, 3
            
            # x * power of 2 = x << n
            if isinstance(const, int) and const > 0 and (const & (const - 1)) == 0:
                shift = (const - 1).bit_length()
                replacement = [
                    inst1,
                    Instruction('LOAD_CONST', shift, inst1.lineno),
                    Instruction('LSHIFT', None, inst1.lineno),
                ]
                self.stats.strength_reductions += 1
                return True, replacement, 3
        
        return False, [], 0
    
    # ========== Instruction Fusion ==========
    
    def _fuse_load_store(
        self,
        instructions: List[Instruction],
        pos: int
    ) -> Tuple[bool, List[Instruction], int]:
        """
        Fuse common load/store patterns
        
        Patterns:
        - LOAD_FAST x, STORE_FAST y -> COPY_FAST x, y (if available)
        - LOAD_CONST x, STORE_FAST y -> (keep as is, already minimal)
        """
        if pos + 1 >= len(instructions):
            return False, [], 0
        
        inst1 = instructions[pos]
        inst2 = instructions[pos + 1]
        
        # LOAD_FAST x, STORE_FAST x (reload same variable)
        if (inst1.opcode == 'LOAD_FAST' and
            inst2.opcode == 'STORE_FAST' and
            inst1.arg == inst2.arg):
            # This is a no-op if nothing else happens
            # But we need to be careful - stack effect matters
            # Keep the LOAD_FAST, remove STORE_FAST
            # Actually, this is LOAD then immediately STORE to same var
            # which means we pop and push same value - effectively DUP_TOP, POP_TOP
            # Can be eliminated if next instruction doesn't need the value
            return False, [], 0
        
        return False, [], 0
    
    # ========== Jump Threading ==========
    
    def _thread_jumps(
        self,
        instructions: List[Instruction],
        pos: int
    ) -> Tuple[bool, List[Instruction], int]:
        """
        Thread jumps through other jumps
        
        Patterns:
        - JUMP_IF_TRUE L1, L1: JUMP_IF_TRUE L2 -> JUMP_IF_TRUE L2
        - JUMP L1, L1: JUMP L2 -> JUMP L2
        """
        inst = instructions[pos]
        
        # For now, skip jump threading as it requires label resolution
        # Would need full control flow analysis
        return False, [], 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = OptimizationStats()


def optimize_bytecode(
    instructions: List[Instruction],
    level: OptimizationLevel = OptimizationLevel.BASIC
) -> Tuple[List[Instruction], Dict[str, Any]]:
    """
    Convenience function to optimize bytecode
    
    Args:
        instructions: List of bytecode instructions
        level: Optimization level
        
    Returns:
        Tuple of (optimized instructions, statistics)
    """
    optimizer = PeepholeOptimizer(level)
    optimized = optimizer.optimize(instructions)
    stats = optimizer.get_stats()
    return optimized, stats
