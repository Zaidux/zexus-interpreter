"""
Just-In-Time Compiler for Zexus VM

Provides tiered compilation:
- Tier 0: Interpreted execution (slowest, most flexible)
- Tier 1: Bytecode VM execution (fast, portable)
- Tier 2: JIT-compiled native code (fastest, hot paths only)

Features:
- Hot path detection via execution counters
- Bytecode optimization passes (via BytecodeOptimizer)
- Native code generation via Python compile()
- JIT cache for compiled code
- Automatic tier promotion
"""

import hashlib
import time
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

# Import optimizer for advanced optimization
try:
    from .optimizer import BytecodeOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False


@dataclass
class HotPathInfo:
    """Information about a hot execution path"""
    bytecode_hash: str
    execution_count: int = 0
    total_time: float = 0.0
    last_execution: float = 0.0
    compiled_version: Optional[Callable] = None
    compilation_time: float = 0.0
    tier: int = 1  # 0=interpreted, 1=bytecode, 2=JIT


@dataclass
class JITStats:
    """JIT compilation statistics"""
    hot_paths_detected: int = 0
    compilations: int = 0
    compilation_time: float = 0.0
    jit_executions: int = 0
    jit_speedup: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    tier_promotions: int = 0


class JITCompiler:
    """
    Just-In-Time compiler for Zexus VM bytecode
    
    Implements a tiered compilation strategy:
    1. Detect hot paths via execution counting
    2. Optimize bytecode with peephole passes
    3. Compile to native Python code for maximum speed
    4. Cache compiled code for reuse
    """
    
    def __init__(self, hot_threshold: int = 100, optimization_level: int = 1, debug: bool = False):
        """
        Initialize JIT compiler
        
        Args:
            hot_threshold: Number of executions before JIT compilation
            optimization_level: 0=none, 1=basic (default), 2=aggressive, 3=experimental
            debug: Enable debug output
        """
        self.hot_threshold = hot_threshold
        self.optimization_level = optimization_level
        self.debug = debug
        
        # Hot path tracking
        self.hot_paths: Dict[str, HotPathInfo] = {}
        
        # Compilation cache: bytecode_hash -> compiled_function
        self.compilation_cache: Dict[str, Callable] = {}
        
        # Statistics
        self.stats = JITStats()
        
        # Use advanced optimizer if available
        if OPTIMIZER_AVAILABLE and optimization_level > 0:
            self.optimizer = BytecodeOptimizer(level=optimization_level, max_passes=3, debug=False)
            if self.debug:
                print(f"🔧 JIT: Using BytecodeOptimizer (level {optimization_level})")
        else:
            self.optimizer = None
            if self.debug and optimization_level > 0:
                print("⚠️  JIT: BytecodeOptimizer not available, using basic optimization")
        
        # Fallback optimization passes (used if optimizer not available)
        self.optimization_passes = [
            self._constant_folding,
            self._dead_code_elimination,
            self._peephole_optimization,
            self._instruction_combining,
        ]
    
    def should_compile(self, bytecode_hash: str) -> bool:
        """
        Determine if bytecode should be JIT compiled
        
        Args:
            bytecode_hash: Hash of the bytecode
            
        Returns:
            True if code is hot enough for JIT compilation
        """
        if bytecode_hash not in self.hot_paths:
            return False
        
        info = self.hot_paths[bytecode_hash]
        
        # Promote to JIT after threshold executions
        if info.execution_count >= self.hot_threshold and info.tier < 2:
            return True
        
        return False
    
    def track_execution(self, bytecode, execution_time: float = 0.0):
        """
        Track bytecode execution for hot path detection
        
        Args:
            bytecode: Bytecode object or instructions
            execution_time: Time taken to execute
        """
        # Hash the bytecode for identification
        bytecode_hash = self._hash_bytecode(bytecode)
        
        if bytecode_hash not in self.hot_paths:
            self.hot_paths[bytecode_hash] = HotPathInfo(bytecode_hash=bytecode_hash)
        
        info = self.hot_paths[bytecode_hash]
        info.execution_count += 1
        info.total_time += execution_time
        info.last_execution = time.time()
        
        # Check for tier promotion
        if info.execution_count == self.hot_threshold:
            self.stats.hot_paths_detected += 1
            if self.debug:
                print(f"🔥 JIT: Hot path detected! Executed {info.execution_count} times")
    
    def compile_hot_path(self, bytecode) -> Optional[Callable]:
        """
        Compile hot bytecode to optimized native code
        
        Args:
            bytecode: Bytecode object with instructions and constants
            
        Returns:
            Compiled function or None if compilation failed
        """
        bytecode_hash = self._hash_bytecode(bytecode)
        
        # Check cache first
        if bytecode_hash in self.compilation_cache:
            self.stats.cache_hits += 1
            return self.compilation_cache[bytecode_hash]
        
        self.stats.cache_misses += 1
        
        start_time = time.time()
        
        try:
            # Step 1: Optimize bytecode
            optimized_instructions, updated_constants = self._optimize_bytecode(bytecode)
            
            # Step 2: Generate Python source code with updated constants
            python_code = self._generate_python_code(optimized_instructions, updated_constants)
            
            # Step 3: Compile to native Python bytecode
            compiled = compile(python_code, '<jit>', 'exec')
            
            # Step 4: Create executable function
            namespace = {}
            exec(compiled, namespace)
            jit_function = namespace.get('jit_execute')
            
            if jit_function:
                # Cache the compiled function
                self.compilation_cache[bytecode_hash] = jit_function
                
                # Update stats
                compilation_time = time.time() - start_time
                self.stats.compilations += 1
                self.stats.compilation_time += compilation_time
                self.stats.tier_promotions += 1
                
                # Update hot path info
                if bytecode_hash in self.hot_paths:
                    info = self.hot_paths[bytecode_hash]
                    info.compiled_version = jit_function
                    info.compilation_time = compilation_time
                    info.tier = 2
                
                if self.debug:
                    print(f"✅ JIT: Compiled in {compilation_time:.4f}s")
                
                return jit_function
            
        except Exception as e:
            if self.debug:
                print(f"❌ JIT: Compilation failed: {e}")
            return None
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get JIT compilation statistics"""
        return {
            'hot_paths_detected': self.stats.hot_paths_detected,
            'compilations': self.stats.compilations,
            'compilation_time': round(self.stats.compilation_time, 4),
            'jit_executions': self.stats.jit_executions,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'tier_promotions': self.stats.tier_promotions,
            'cache_size': len(self.compilation_cache),
        }
    
    def clear_cache(self):
        """Clear compilation cache"""
        self.compilation_cache.clear()
        self.hot_paths.clear()
        if self.debug:
            print("🗑️ JIT: Cache cleared")
    
    # ==================== Private Methods ====================
    
    def _hash_bytecode(self, bytecode) -> str:
        """Generate hash for bytecode identification"""
        if hasattr(bytecode, 'instructions'):
            # Bytecode object
            data = str(bytecode.instructions) + str(bytecode.constants)
        else:
            # Raw instructions
            data = str(bytecode)
        
        return hashlib.md5(data.encode()).hexdigest()
    
    def _optimize_bytecode(self, bytecode):
        """
        Apply optimization passes to bytecode
        
        Returns: (optimized_instructions, updated_constants)
        """
        instructions = list(bytecode.instructions) if hasattr(bytecode, 'instructions') else list(bytecode)
        constants = list(bytecode.constants) if hasattr(bytecode, 'constants') else []
        
        # Use advanced optimizer if available
        if self.optimizer:
            optimized = self.optimizer.optimize(instructions, constants)
            if self.debug:
                stats = self.optimizer.get_stats()
                print(f"🔧 JIT Optimizer: {stats['original_size']} → {stats['optimized_size']} instructions "
                      f"({stats['size_reduction_pct']:.1f}% reduction)")
                print(f"   Optimizations: {stats['total_optimizations']} "
                      f"(folds={stats['constant_folds']}, dce={stats['dead_code_removed']}, "
                      f"peephole={stats['peephole_opts']}, combined={stats['instructions_combined']})")
            return optimized, constants  # Return updated constants
        
        # Fallback to basic optimization passes
        for opt_pass in self.optimization_passes:
            instructions = opt_pass(instructions)
        
        return instructions, constants
    
    def _constant_folding(self, instructions):
        """Fold constant expressions at compile time"""
        optimized = []
        i = 0
        
        while i < len(instructions):
            if i + 2 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                op3, operand3 = instructions[i + 2]
                
                # LOAD_CONST, LOAD_CONST, ADD/SUB/MUL/DIV -> LOAD_CONST (result)
                if op1 == 'LOAD_CONST' and op2 == 'LOAD_CONST':
                    if op3 in ['ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'POW']:
                        # Can't fold without constants access - skip for now
                        pass
            
            optimized.append(instructions[i])
            i += 1
        
        return optimized
    
    def _dead_code_elimination(self, instructions):
        """Remove unreachable code"""
        optimized = []
        reachable = True
        
        for opcode, operand in instructions:
            if not reachable:
                # Skip until we hit a label/marker
                if opcode == 'JUMP' or opcode == 'RETURN':
                    continue
            
            optimized.append((opcode, operand))
            
            if opcode == 'RETURN':
                reachable = False
            elif opcode == 'JUMP':
                reachable = True  # Jump target is reachable
        
        return optimized
    
    def _peephole_optimization(self, instructions):
        """Apply peephole optimizations"""
        optimized = []
        i = 0
        
        while i < len(instructions):
            if i + 1 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                
                # LOAD_NAME x, POP -> (remove both)
                if op1 == 'LOAD_NAME' and op2 == 'POP':
                    i += 2
                    continue
                
                # DUP, POP -> (remove both)
                if op1 == 'DUP' and op2 == 'POP':
                    i += 2
                    continue
                
                # JUMP to next instruction -> (remove)
                if op1 == 'JUMP' and operand1 == i + 1:
                    i += 1
                    continue
            
            optimized.append(instructions[i])
            i += 1
        
        return optimized
    
    def _instruction_combining(self, instructions):
        """Combine adjacent instructions"""
        optimized = []
        i = 0
        
        while i < len(instructions):
            if i + 1 < len(instructions):
                op1, operand1 = instructions[i]
                op2, operand2 = instructions[i + 1]
                
                # LOAD_CONST, STORE_NAME -> combined operation
                if op1 == 'LOAD_CONST' and op2 == 'STORE_NAME':
                    optimized.append(('STORE_CONST', (operand1, operand2)))
                    i += 2
                    continue
            
            optimized.append(instructions[i])
            i += 1
        
        return optimized
    
    def _generate_python_code(self, instructions, constants) -> str:
        """
        Generate Python source code from optimized bytecode
        
        This creates a Python function that executes the same logic
        as the bytecode but runs at native Python speed.
        """
        lines = [
            "def jit_execute(vm, stack, env):",
            "    # JIT-compiled native code",
        ]
        
        for opcode, operand in instructions:
            if opcode == 'LOAD_CONST':
                const_val = constants[operand] if operand < len(constants) else None
                lines.append(f"    stack.append({repr(const_val)})")
            
            elif opcode == 'LOAD_NAME':
                name = constants[operand] if operand < len(constants) else 'unknown'
                lines.append(f"    stack.append(env.get({repr(name)}, None))")
            
            elif opcode == 'STORE_NAME':
                name = constants[operand] if operand < len(constants) else 'unknown'
                lines.append(f"    env[{repr(name)}] = stack.pop() if stack else None")
            
            elif opcode == 'STORE_CONST':
                # Combined LOAD_CONST + STORE_NAME -> STORE_CONST (name_idx, const_idx)
                if isinstance(operand, tuple):
                    name_idx, const_idx = operand
                    name = constants[name_idx] if name_idx < len(constants) else 'unknown'
                    const_val = constants[const_idx] if const_idx < len(constants) else None
                    lines.append(f"    env[{repr(name)}] = {repr(const_val)}")
            
            elif opcode == 'INC':
                # Increment top of stack (optimized ADD 1)
                lines.append("    if stack:")
                lines.append("        stack[-1] = stack[-1] + 1")
                lines.append("    else:")
                lines.append("        stack.append(1)")
            
            elif opcode == 'DEC':
                # Decrement top of stack (optimized SUB 1)
                lines.append("    if stack:")
                lines.append("        stack[-1] = stack[-1] - 1")
                lines.append("    else:")
                lines.append("        stack.append(-1)")
            
            elif opcode == 'ADD':
                lines.append("    b = stack.pop() if stack else 0")
                lines.append("    a = stack.pop() if stack else 0")
                lines.append("    stack.append(a + b)")
            
            elif opcode == 'SUB':
                lines.append("    b = stack.pop() if stack else 0")
                lines.append("    a = stack.pop() if stack else 0")
                lines.append("    stack.append(a - b)")
            
            elif opcode == 'MUL':
                lines.append("    b = stack.pop() if stack else 0")
                lines.append("    a = stack.pop() if stack else 0")
                lines.append("    stack.append(a * b)")
            
            elif opcode == 'DIV':
                lines.append("    b = stack.pop() if stack else 1")
                lines.append("    a = stack.pop() if stack else 0")
                lines.append("    stack.append(a / b if b != 0 else 0)")
            
            elif opcode == 'MOD':
                lines.append("    b = stack.pop() if stack else 1")
                lines.append("    a = stack.pop() if stack else 0")
                lines.append("    stack.append(a % b if b != 0 else 0)")
            
            elif opcode == 'POW':
                lines.append("    b = stack.pop() if stack else 0")
                lines.append("    a = stack.pop() if stack else 0")
                lines.append("    stack.append(a ** b)")
            
            elif opcode == 'EQ':
                lines.append("    b = stack.pop() if stack else None")
                lines.append("    a = stack.pop() if stack else None")
                lines.append("    stack.append(a == b)")
            
            elif opcode == 'LT':
                lines.append("    b = stack.pop() if stack else 0")
                lines.append("    a = stack.pop() if stack else 0")
                lines.append("    stack.append(a < b)")
            
            elif opcode == 'GT':
                lines.append("    b = stack.pop() if stack else 0")
                lines.append("    a = stack.pop() if stack else 0")
                lines.append("    stack.append(a > b)")
            
            elif opcode == 'RETURN':
                lines.append("    return stack[-1] if stack else None")
        
        # Default return
        lines.append("    return stack[-1] if stack else None")
        
        return "\n".join(lines)