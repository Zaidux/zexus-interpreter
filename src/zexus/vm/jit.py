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
- Stack state simulation for accurate Python code generation
"""

import hashlib
import time
import dis
from typing import Dict, Any, Optional, Tuple, Callable, List, Set
from dataclasses import dataclass, field
from enum import Enum

# Import optimizer for advanced optimization
try:
    from .optimizer import BytecodeOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False


class ExecutionTier(Enum):
    """Execution tiers for tiered compilation"""
    INTERPRETED = 0  # AST interpretation (slowest)
    BYTECODE = 1     # Stack-based VM (fast)
    JIT_NATIVE = 2   # JIT-compiled native code (fastest)


@dataclass
class HotPathInfo:
    """Information about a hot execution path"""
    bytecode_hash: str
    execution_count: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    last_execution: float = 0.0
    compiled_version: Optional[Callable] = None
    compilation_time: float = 0.0
    tier: ExecutionTier = ExecutionTier.BYTECODE
    speedup_factor: float = 1.0


@dataclass
class JITStats:
    """JIT compilation statistics"""
    hot_paths_detected: int = 0
    compilations: int = 0
    compilation_time: float = 0.0
    average_compilation_time: float = 0.0
    jit_executions: int = 0
    total_jit_execution_time: float = 0.0
    total_bytecode_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    tier_promotions: int = 0
    compilation_failures: int = 0


@dataclass
class StackFrame:
    """Represents stack state at a point in bytecode"""
    stack: List[str] = field(default_factory=list)  # Variable names on stack
    variables: Dict[str, str] = field(default_factory=dict)  # Variable -> expression
    constants: Dict[int, Any] = field(default_factory=dict)  # Constant ID -> value


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
        self.verbose = debug  # Alias for compatibility

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

        # Supported opcodes for JIT compilation
        self.supported_opcodes = {
            'LOAD_CONST', 'LOAD_NAME', 'STORE_NAME', 'STORE_CONST',
            'ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'POW',
            'EQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE',
            'AND', 'OR', 'NOT',
            'RETURN', 'JUMP', 'JUMP_IF_FALSE',
        }

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

        # Check if already compiled
        if info.tier == ExecutionTier.JIT_NATIVE:
            return False

        # Promote to JIT after threshold executions
        if info.execution_count >= self.hot_threshold:
            return True

        return False

    def track_execution(self, bytecode, execution_time: float = 0.0) -> HotPathInfo:
        """
        Track bytecode execution for hot path detection
        
        Args:
            bytecode: Bytecode object or instructions
            execution_time: Time taken to execute
            
        Returns:
            HotPathInfo for the tracked bytecode
        """
        # Hash the bytecode for identification
        bytecode_hash = self._hash_bytecode(bytecode)

        if bytecode_hash not in self.hot_paths:
            self.hot_paths[bytecode_hash] = HotPathInfo(
                bytecode_hash=bytecode_hash,
                tier=ExecutionTier.BYTECODE
            )

        info = self.hot_paths[bytecode_hash]
        info.execution_count += 1
        info.total_time += execution_time
        info.average_time = info.total_time / info.execution_count
        info.last_execution = time.time()

        # Check for tier promotion
        if info.execution_count == self.hot_threshold:
            self.stats.hot_paths_detected += 1
            if self.debug:
                print(f"🔥 JIT: Hot path detected! Executed {info.execution_count} times "
                      f"(avg: {info.average_time*1000:.2f}ms)")

        return info

    def compile_hot_path(self, bytecode) -> Optional[Callable]:
        """
        Compile hot bytecode to optimized native code
        
        Args:
            bytecode: Bytecode object with instructions and constants
            
        Returns:
            Compiled function or None if compilation failed
        """
        # First check if bytecode is JIT-compatible
        if not self._is_jit_compatible(bytecode):
            if self.debug:
                print(f"⚠️  JIT: Bytecode contains unsupported opcodes, skipping JIT")
            return None

        bytecode_hash = self._hash_bytecode(bytecode)

        # Check cache first
        if bytecode_hash in self.compilation_cache:
            self.stats.cache_hits += 1
            if self.debug:
                print(f"✅ JIT: Cache hit for {bytecode_hash[:8]}")
            return self.compilation_cache[bytecode_hash]

        self.stats.cache_misses += 1

        start_time = time.time()

        try:
            # Step 1: Analyze stack behavior
            stack_frames = self._analyze_stack_behavior(bytecode)
            if not stack_frames:
                if self.debug:
                    print(f"❌ JIT: Failed to analyze stack behavior")
                return None

            # Step 2: Optimize bytecode
            optimized_instructions, updated_constants = self._optimize_bytecode(bytecode)

            # Step 3: Generate efficient Python source code
            python_code = self._generate_efficient_python_code(
                optimized_instructions, 
                updated_constants,
                stack_frames
            )

            if self.debug and self.verbose:
                print(f"📝 JIT Generated Python code:\n{python_code[:500]}...")

            # Step 4: Compile to native Python bytecode
            compiled = compile(python_code, f'<jit:{bytecode_hash[:8]}>', 'exec')

            # Step 5: Create executable function
            namespace = {'__builtins__': {}}
            exec(compiled, namespace)
            jit_function = namespace.get('jit_execute')

            if jit_function:
                # Verify the compiled function works
                verification_result = self._verify_compilation(bytecode, jit_function)
                if not verification_result:
                    if self.debug:
                        print(f"❌ JIT: Verification failed for {bytecode_hash[:8]}")
                    self.stats.compilation_failures += 1
                    return None

                # Cache the compiled function
                self.compilation_cache[bytecode_hash] = jit_function

                # Update stats
                compilation_time = time.time() - start_time
                self.stats.compilations += 1
                self.stats.compilation_time += compilation_time
                self.stats.average_compilation_time = (
                    self.stats.compilation_time / self.stats.compilations
                )
                self.stats.tier_promotions += 1

                # Update hot path info
                if bytecode_hash in self.hot_paths:
                    info = self.hot_paths[bytecode_hash]
                    info.compiled_version = jit_function
                    info.compilation_time = compilation_time
                    info.tier = ExecutionTier.JIT_NATIVE

                if self.debug:
                    print(f"✅ JIT: Compiled {bytecode_hash[:8]} in {compilation_time*1000:.2f}ms "
                          f"({len(optimized_instructions)} instructions)")

                return jit_function

        except Exception as e:
            if self.debug:
                print(f"❌ JIT: Compilation failed for {bytecode_hash[:8]}: {e}")
                import traceback
                traceback.print_exc()
            self.stats.compilation_failures += 1
            return None

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get JIT compilation statistics"""
        # Calculate speedup if we have enough data
        speedup = 0.0
        if (self.stats.total_jit_execution_time > 0 and 
            self.stats.total_bytecode_execution_time > 0 and
            self.stats.jit_executions > 0):
            avg_jit_time = self.stats.total_jit_execution_time / self.stats.jit_executions
            avg_bytecode_time = self.stats.total_bytecode_execution_time / max(self.stats.jit_executions, 1)
            speedup = avg_bytecode_time / avg_jit_time if avg_jit_time > 0 else 0

        return {
            'hot_paths_detected': self.stats.hot_paths_detected,
            'compilations': self.stats.compilations,
            'compilation_time': round(self.stats.compilation_time, 4),
            'average_compilation_time': round(self.stats.average_compilation_time, 6),
            'jit_executions': self.stats.jit_executions,
            'total_jit_execution_time': round(self.stats.total_jit_execution_time, 4),
            'total_bytecode_execution_time': round(self.stats.total_bytecode_execution_time, 4),
            'speedup_factor': round(speedup, 2),
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'cache_size': len(self.compilation_cache),
            'tier_promotions': self.stats.tier_promotions,
            'compilation_failures': self.stats.compilation_failures,
            'cache_hit_rate': round(
                self.stats.cache_hits / max(self.stats.cache_hits + self.stats.cache_misses, 1) * 100, 
                1
            ),
        }

    def clear_cache(self):
        """Clear compilation cache"""
        self.compilation_cache.clear()
        self.hot_paths.clear()
        self.stats = JITStats()  # Reset stats
        if self.debug:
            print("🗑️ JIT: Cache cleared and stats reset")

    def record_execution_time(self, bytecode_hash: str, execution_time: float, tier: ExecutionTier):
        """
        Record execution time for performance tracking
        
        Args:
            bytecode_hash: Hash of executed bytecode
            execution_time: Time taken for execution
            tier: Which tier was used for execution
        """
        if tier == ExecutionTier.JIT_NATIVE:
            self.stats.jit_executions += 1
            self.stats.total_jit_execution_time += execution_time
            
            # Update speedup calculation in hot path info
            if bytecode_hash in self.hot_paths:
                info = self.hot_paths[bytecode_hash]
                # Track for speedup calculation
                if info.average_time > 0:
                    info.speedup_factor = info.average_time / execution_time
        elif tier == ExecutionTier.BYTECODE:
            # Track bytecode execution time for comparison
            self.stats.total_bytecode_execution_time += execution_time

    # ==================== Private Methods ====================

    def _hash_bytecode(self, bytecode) -> str:
        """Generate hash for bytecode identification"""
        if hasattr(bytecode, 'instructions'):
            # Bytecode object - include both instructions and constants
            data = str(bytecode.instructions) + str(bytecode.constants)
        elif hasattr(bytecode, '__iter__'):
            # List of instructions
            data = str(list(bytecode))
        else:
            data = str(bytecode)

        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _is_jit_compatible(self, bytecode) -> bool:
        """Check if bytecode contains only JIT-supported opcodes"""
        instructions = bytecode.instructions if hasattr(bytecode, 'instructions') else bytecode
        
        for opcode, _ in instructions:
            if opcode not in self.supported_opcodes:
                if self.debug:
                    print(f"⚠️  JIT: Unsupported opcode: {opcode}")
                return False
        return True

    def _analyze_stack_behavior(self, bytecode) -> List[StackFrame]:
        """
        Analyze stack behavior to generate efficient Python code
        
        Returns:
            List of StackFrame objects representing stack state at each instruction
        """
        instructions = bytecode.instructions if hasattr(bytecode, 'instructions') else bytecode
        constants = bytecode.constants if hasattr(bytecode, 'constants') else []
        
        frames = []
        current_frame = StackFrame()
        
        for i, (opcode, operand) in enumerate(instructions):
            # Record current frame state
            frames.append(StackFrame(
                stack=list(current_frame.stack),
                variables=dict(current_frame.variables),
                constants=dict(current_frame.constants)
            ))
            
            # Update frame based on opcode
            if opcode == 'LOAD_CONST':
                # Create a unique variable name for this constant
                var_name = f"const_{operand}_{i}"
                current_frame.stack.append(var_name)
                current_frame.constants[operand] = constants[operand] if operand < len(constants) else None
                current_frame.variables[var_name] = repr(constants[operand]) if operand < len(constants) else 'None'
                
            elif opcode == 'LOAD_NAME':
                var_name = f"var_{operand}_{i}"
                current_frame.stack.append(var_name)
                # We don't know the value yet, will be looked up from env
                current_frame.variables[var_name] = f"env.get({repr(constants[operand])}, None)" if operand < len(constants) else 'None'
                
            elif opcode == 'STORE_NAME':
                if current_frame.stack:
                    value_expr = current_frame.stack.pop()
                    name = constants[operand] if operand < len(constants) else f'unknown_{operand}'
                    current_frame.variables[name] = current_frame.variables.get(value_expr, value_expr)
                    
            elif opcode in ['ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'POW']:
                if len(current_frame.stack) >= 2:
                    b_expr = current_frame.stack.pop()
                    a_expr = current_frame.stack.pop()
                    result_expr = f"result_{i}"
                    
                    a_val = current_frame.variables.get(a_expr, a_expr)
                    b_val = current_frame.variables.get(b_expr, b_expr)
                    
                    # Try to evaluate if both are constants
                    try:
                        if (a_expr.startswith('const_') and b_expr.startswith('const_')):
                            # Both are constants, can pre-compute
                            a_const = eval(a_val) if isinstance(a_val, str) and a_val[0].isdigit() else a_val
                            b_const = eval(b_val) if isinstance(b_val, str) and b_val[0].isdigit() else b_val
                            
                            if opcode == 'ADD':
                                result = a_const + b_const
                            elif opcode == 'SUB':
                                result = a_const - b_const
                            elif opcode == 'MUL':
                                result = a_const * b_const
                            elif opcode == 'DIV':
                                result = a_const / b_const if b_const != 0 else 0
                            elif opcode == 'MOD':
                                result = a_const % b_const if b_const != 0 else 0
                            elif opcode == 'POW':
                                result = a_const ** b_const
                                
                            current_frame.variables[result_expr] = repr(result)
                        else:
                            # Runtime computation needed
                            current_frame.variables[result_expr] = f"({a_val} {self._opcode_to_operator(opcode)} {b_val})"
                    except (TypeError, ValueError, KeyError, NameError):
                        # Fallback to runtime computation
                        current_frame.variables[result_expr] = f"({a_val} {self._opcode_to_operator(opcode)} {b_val})"
                    
                    current_frame.stack.append(result_expr)
                    
            elif opcode == 'RETURN':
                if current_frame.stack:
                    return_expr = current_frame.stack[-1]
                    current_frame.variables['__return__'] = current_frame.variables.get(return_expr, return_expr)
        
        return frames

    def _opcode_to_operator(self, opcode: str) -> str:
        """Convert opcode to Python operator"""
        return {
            'ADD': '+',
            'SUB': '-',
            'MUL': '*',
            'DIV': '/',
            'MOD': '%',
            'POW': '**',
            'EQ': '==',
            'NEQ': '!=',
            'LT': '<',
            'GT': '>',
            'LTE': '<=',
            'GTE': '>=',
            'AND': 'and',
            'OR': 'or',
        }.get(opcode, opcode)

    def _optimize_bytecode(self, bytecode) -> Tuple[List, List]:
        """
        Apply optimization passes to bytecode
        
        Returns: (optimized_instructions, updated_constants)
        """
        instructions = list(bytecode.instructions) if hasattr(bytecode, 'instructions') else list(bytecode)
        constants = list(bytecode.constants) if hasattr(bytecode, 'constants') else []

        if self.optimizer:
            try:
                optimized, updated_constants = self.optimizer.optimize(instructions, constants)
                if self.debug:
                    stats = self.optimizer.get_stats()
                    if stats['total_optimizations'] > 0:
                        print(f"🔧 JIT Optimizer: {stats['original_size']} → {stats['optimized_size']} instructions "
                              f"({stats['size_reduction_pct']:.1f}% reduction)")
                return optimized, updated_constants
            except Exception as e:
                if self.debug:
                    print(f"⚠️  JIT: Optimizer failed: {e}")
                return instructions, constants

        # Fallback: apply basic optimizations
        optimized = self._apply_basic_optimizations(instructions, constants)
        return optimized, constants

    def _apply_basic_optimizations(self, instructions: List, constants: List) -> List:
        """Apply basic peephole optimizations"""
        optimized = []
        i = 0
        
        while i < len(instructions):
            opcode, operand = instructions[i]
            
            # Skip useless patterns
            if i + 1 < len(instructions):
                next_op, next_operand = instructions[i + 1]
                
                # LOAD_CONST + POP -> skip both
                if opcode == 'LOAD_CONST' and next_op == 'POP':
                    i += 2
                    continue
                    
                # LOAD_NAME + POP -> skip both
                if opcode == 'LOAD_NAME' and next_op == 'POP':
                    i += 2
                    continue
            
            optimized.append((opcode, operand))
            i += 1
            
        return optimized

    def _generate_efficient_python_code(self, instructions: List, constants: List, stack_frames: List[StackFrame]) -> str:
        """
        Generate efficient Python source code from optimized bytecode
        using stack analysis to minimize runtime checks
        """
        lines = [
            "def jit_execute(vm, stack, env):",
            "    # JIT-compiled native code",
        ]
        
        # Track variable assignments for efficient code generation
        local_vars = set()
        stack_height = 0
        
        for i, (opcode, operand) in enumerate(instructions):
            if i >= len(stack_frames):
                break
                
            frame = stack_frames[i]
            
            if opcode == 'LOAD_CONST':
                if operand < len(constants):
                    const_val = constants[operand]
                    var_name = f"const_{operand}_{i}"
                    if var_name in frame.variables:
                        # Use pre-computed value if available
                        lines.append(f"    {var_name} = {frame.variables[var_name]}")
                        local_vars.add(var_name)
                        lines.append(f"    stack.append({var_name})")
                    else:
                        lines.append(f"    stack.append({repr(const_val)})")
                stack_height += 1
                
            elif opcode == 'LOAD_NAME':
                if operand < len(constants):
                    name = constants[operand]
                    var_name = f"var_{operand}_{i}"
                    if var_name in frame.variables:
                        # Use efficient lookup
                        lines.append(f"    {var_name} = env.get({repr(name)}, None)")
                        local_vars.add(var_name)
                        lines.append(f"    stack.append({var_name})")
                    else:
                        lines.append(f"    stack.append(env.get({repr(name)}, None))")
                stack_height += 1
                
            elif opcode == 'STORE_NAME':
                if operand < len(constants):
                    name = constants[operand]
                    if stack_height > 0:
                        lines.append(f"    env[{repr(name)}] = stack.pop()")
                        stack_height -= 1
                        
            elif opcode == 'STORE_CONST':
                if isinstance(operand, tuple) and len(operand) == 2:
                    name_idx, const_idx = operand
                    if name_idx < len(constants) and const_idx < len(constants):
                        name = constants[name_idx]
                        const_val = constants[const_idx]
                        lines.append(f"    env[{repr(name)}] = {repr(const_val)}")
                        
            elif opcode in ['ADD', 'SUB', 'MUL', 'DIV', 'MOD', 'POW']:
                if stack_height >= 2:
                    # Generate efficient arithmetic
                    operator = self._opcode_to_operator(opcode)
                    result_var = f"result_{i}"
                    
                    # Try to use pre-computed values from stack analysis
                    if i > 0 and i-1 < len(stack_frames):
                        prev_frame = stack_frames[i-1]
                        if len(prev_frame.stack) >= 2:
                            a_expr = prev_frame.stack[-2]
                            b_expr = prev_frame.stack[-1]
                            
                            if a_expr in frame.variables and b_expr in frame.variables:
                                # Both values are known, use pre-computation
                                a_val = frame.variables[a_expr]
                                b_val = frame.variables[b_expr]
                                
                                # Check if we can compute at compile time
                                try:
                                    if (a_expr.startswith('const_') and b_expr.startswith('const_')):
                                        # Already computed by optimizer
                                        computed = eval(f"{a_val} {operator} {b_val}")
                                        lines.append(f"    {result_var} = {repr(computed)}")
                                    else:
                                        lines.append(f"    {result_var} = {a_val} {operator} {b_val}")
                                except (TypeError, ValueError, NameError, SyntaxError):
                                    lines.append(f"    {result_var} = {a_val} {operator} {b_val}")
                            else:
                                lines.append(f"    b = stack.pop()")
                                lines.append(f"    a = stack.pop()")
                                lines.append(f"    {result_var} = a {operator} b")
                    else:
                        lines.append(f"    b = stack.pop()")
                        lines.append(f"    a = stack.pop()")
                        lines.append(f"    {result_var} = a {operator} b")
                    
                    local_vars.add(result_var)
                    lines.append(f"    stack.append({result_var})")
                    stack_height -= 1  # 2 popped, 1 pushed = net -1
                    
            elif opcode == 'RETURN':
                if stack_height > 0:
                    lines.append("    return stack[-1]")
                else:
                    lines.append("    return None")
                break
        
        # Default return if no RETURN encountered
        if not any(opcode == 'RETURN' for opcode, _ in instructions):
            if stack_height > 0:
                lines.append("    return stack[-1] if stack else None")
            else:
                lines.append("    return None")
        
        return "\n".join(lines)

    def _verify_compilation(self, original_bytecode, jit_function) -> bool:
        """
        Verify that JIT-compiled code produces same results as bytecode
        
        Args:
            original_bytecode: Original bytecode to verify against
            jit_function: JIT-compiled function to verify
            
        Returns:
            True if verification passes
        """
        try:
            # Create test environment
            from .vm import VM
            
            # Test with a few different inputs
            test_cases = [
                ({}, []),  # Empty
                ({'x': 5, 'y': 3}, []),  # Some variables
                ({'a': 10, 'b': 20, 'c': 30}, []),  # More variables
            ]
            
            for env_dict, stack in test_cases:
                # Run bytecode version
                vm_bytecode = VM(use_jit=False)
                vm_bytecode.env = env_dict.copy()
                vm_bytecode.stack = stack.copy()
                bytecode_result = vm_bytecode.execute(original_bytecode)
                
                # Run JIT version
                vm_jit = VM(use_jit=False)
                vm_jit.env = env_dict.copy()
                vm_jit.stack = stack.copy()
                jit_result = jit_function(vm_jit, vm_jit.stack, vm_jit.env)
                
                # Compare results
                if bytecode_result != jit_result:
                    if self.debug:
                        print(f"❌ JIT Verification failed:")
                        print(f"   Bytecode result: {bytecode_result}")
                        print(f"   JIT result: {jit_result}")
                        print(f"   Environment: {env_dict}")
                    return False
                    
            return True
            
        except Exception as e:
            if self.debug:
                print(f"❌ JIT Verification error: {e}")
            return False

    def get_hot_path_info(self, bytecode_hash: str) -> Optional[HotPathInfo]:
        """Get information about a hot path"""
        return self.hot_paths.get(bytecode_hash)

    def get_compilation_cache_info(self) -> Dict[str, Any]:
        """Get information about compilation cache"""
        return {
            'size': len(self.compilation_cache),
            'entries': list(self.compilation_cache.keys()),
            'hit_rate': self.stats.cache_hits / max(self.stats.cache_hits + self.stats.cache_misses, 1),
        }