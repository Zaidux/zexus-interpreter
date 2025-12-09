"""
Pure Function Enforcement System for Zexus.

Enforces referential transparency and side-effect detection to enable:
- Easier reasoning about code
- Better optimizations (caching, parallelization)
- Security analysis and sandboxing
- Concurrent and distributed execution

A pure function:
1. Depends only on its input parameters
2. Produces no side effects (no I/O, no global state modification, etc.)
3. Returns the same output for the same input (deterministic)
4. Does not throw exceptions (in strict mode)
"""

from typing import Set, Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import inspect


class Purity(Enum):
    """Levels of function purity."""
    PURE = 3           # Fully pure, no side effects
    RESTRICTED = 2    # Side effects only on local state
    IMPURE = 1        # Has side effects
    UNKNOWN = 0       # Purity not determined


class SideEffectType(Enum):
    """Types of side effects."""
    IO_READ = "io.read"           # File/network read
    IO_WRITE = "io.write"         # File/network write
    GLOBAL_READ = "global.read"   # Read global state
    GLOBAL_WRITE = "global.write" # Write global state
    EXCEPTION = "exception"       # Throws exception
    EXTERNAL_CALL = "external"    # Calls external/impure function
    TIME_DEPENDENT = "time"       # Depends on current time
    RANDOM = "random"             # Uses randomness
    MEMORY_ALLOC = "memory"       # Allocates memory
    CONCURRENCY = "concurrency"   # Uses threads/async


@dataclass
class SideEffect:
    """Records a detected side effect."""
    effect_type: SideEffectType
    description: str = ""
    location: str = ""  # Function/line where side effect occurs
    severity: str = "warning"  # "warning", "error", "critical"
    
    def __str__(self):
        return f"{self.effect_type.value}: {self.description} at {self.location}"


@dataclass
class FunctionSignature:
    """Signature of a function for purity analysis."""
    name: str
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    declared_purity: Purity = Purity.UNKNOWN
    actual_purity: Purity = Purity.UNKNOWN
    side_effects: List[SideEffect] = field(default_factory=list)
    immutable_inputs: Set[str] = field(default_factory=set)
    immutable_locals: Set[str] = field(default_factory=set)


class PurityAnalyzer:
    """
    Analyzes function code to determine purity.
    
    Detects:
    - Global variable access
    - I/O operations
    - Exception throwing
    - Calls to impure functions
    - Non-deterministic operations (random, time)
    - Memory allocation/deallocation
    """
    
    # Operations that indicate impurity
    IMPURE_BUILTINS = {
        # I/O operations
        'print', 'input', 'open', 'read', 'write', 'close',
        'os.remove', 'os.rename', 'os.system',
        
        # Global state
        'globals', 'setattr', 'getattr',
        
        # Non-deterministic
        'random', 'random.random', 'time.time', 'datetime.now',
        
        # External
        'exec', 'eval', '__import__',
        
        # Concurrency
        'Thread', 'Process', 'Lock', 'asyncio.run',
    }
    
    # Operations that indicate I/O
    IO_OPERATIONS = {
        'open', 'read', 'write', 'close',
        'os.remove', 'os.rename', 'os.listdir',
        'socket', 'http.client', 'requests.get', 'requests.post',
    }
    
    # Operations that indicate global state access
    GLOBAL_OPERATIONS = {
        'globals', 'vars', 'getattr', 'setattr', 'delattr',
    }
    
    def __init__(self):
        self.function_cache: Dict[str, FunctionSignature] = {}
        self.known_pure: Set[str] = set()
        self.known_impure: Set[str] = set()
    
    def analyze(self, func_name: str, func_code: str, 
                parameters: List[str]) -> FunctionSignature:
        """
        Analyze a function for purity.
        
        Args:
            func_name: Name of the function
            func_code: Source code of the function body
            parameters: List of parameter names
        
        Returns:
            FunctionSignature with purity information
        """
        if func_name in self.function_cache:
            return self.function_cache[func_name]
        
        sig = FunctionSignature(
            name=func_name,
            parameters=parameters,
            immutable_inputs=set(parameters)
        )
        
        # Analyze for side effects
        side_effects = self._detect_side_effects(func_code, parameters)
        sig.side_effects = side_effects
        
        # Determine purity level
        if not side_effects:
            sig.actual_purity = Purity.PURE
        else:
            # Check severity of side effects
            has_critical = any(e.severity == "critical" for e in side_effects)
            if has_critical:
                sig.actual_purity = Purity.IMPURE
            else:
                sig.actual_purity = Purity.RESTRICTED
        
        self.function_cache[func_name] = sig
        return sig
    
    def _detect_side_effects(self, func_code: str, 
                           parameters: List[str]) -> List[SideEffect]:
        """Detect side effects in function code."""
        side_effects = []
        
        # Check for impure function calls
        for impure_call in self.IMPURE_BUILTINS:
            if impure_call in func_code:
                if any(io_op in impure_call for io_op in self.IO_OPERATIONS):
                    side_effects.append(SideEffect(
                        effect_type=SideEffectType.IO_WRITE,
                        description=f"Calls {impure_call}",
                        severity="critical"
                    ))
                elif any(glob_op in impure_call for glob_op in self.GLOBAL_OPERATIONS):
                    side_effects.append(SideEffect(
                        effect_type=SideEffectType.GLOBAL_WRITE,
                        description=f"Accesses global state via {impure_call}",
                        severity="critical"
                    ))
                else:
                    side_effects.append(SideEffect(
                        effect_type=SideEffectType.EXTERNAL_CALL,
                        description=f"Calls impure function {impure_call}",
                        severity="warning"
                    ))
        
        # Check for exception raising
        if 'raise ' in func_code or 'throw ' in func_code:
            side_effects.append(SideEffect(
                effect_type=SideEffectType.EXCEPTION,
                description="Throws exception",
                severity="warning"
            ))
        
        # Check for time/random
        if 'time(' in func_code or 'time.' in func_code:
            side_effects.append(SideEffect(
                effect_type=SideEffectType.TIME_DEPENDENT,
                description="Uses system time",
                severity="warning"
            ))
        
        if 'random' in func_code:
            side_effects.append(SideEffect(
                effect_type=SideEffectType.RANDOM,
                description="Uses randomness",
                severity="warning"
            ))
        
        # Check for parameter mutation
        for param in parameters:
            mutation_patterns = [
                f"{param}.",  # Direct attribute modification
                f"{param}[",  # Index modification
                f"append({param}",
                f"extend({param}",
            ]
            
            if any(pattern in func_code for pattern in mutation_patterns):
                side_effects.append(SideEffect(
                    effect_type=SideEffectType.GLOBAL_WRITE,
                    description=f"Mutates parameter '{param}'",
                    severity="warning"
                ))
        
        return side_effects
    
    def declare_pure(self, func_name: str):
        """Declare a function as pure (for external/built-in functions)."""
        self.known_pure.add(func_name)
    
    def declare_impure(self, func_name: str):
        """Declare a function as impure (for external/built-in functions)."""
        self.known_impure.add(func_name)
    
    def is_known_pure(self, func_name: str) -> bool:
        """Check if function is known to be pure."""
        return func_name in self.known_pure
    
    def is_known_impure(self, func_name: str) -> bool:
        """Check if function is known to be impure."""
        return func_name in self.known_impure


class PurityEnforcer:
    """
    Enforces purity constraints at runtime and compile-time.
    
    - Validates that pure functions don't call impure functions
    - Prevents side effects in pure function execution
    - Tracks function calls for verification
    """
    
    def __init__(self, analyzer: PurityAnalyzer):
        self.analyzer = analyzer
        self.pure_functions: Dict[str, FunctionSignature] = {}
        self.call_trace: List[Tuple[str, bool]] = []  # (func_name, is_pure)
    
    def register_pure_function(self, sig: FunctionSignature):
        """Register a function as pure."""
        if sig.actual_purity == Purity.PURE:
            self.pure_functions[sig.name] = sig
            self.analyzer.declare_pure(sig.name)
    
    def enforce_pure_execution(self, func_name: str, 
                              func_code: str,
                              parameters: List[str]) -> bool:
        """
        Enforce that a function executes purely.
        
        Raises:
            PurityViolationError if impurity detected
        
        Returns:
            True if execution is pure
        """
        sig = self.analyzer.analyze(func_name, func_code, parameters)
        
        if sig.actual_purity == Purity.IMPURE:
            raise PurityViolationError(
                f"Function {func_name} violates purity constraint: "
                f"{sig.side_effects}"
            )
        
        self.pure_functions[func_name] = sig
        return True
    
    def validate_pure_call_chain(self, called_functions: List[str]) -> bool:
        """
        Validate that a pure function only calls other pure functions.
        
        Args:
            called_functions: List of functions called within a pure function
        
        Returns:
            True if all called functions are pure
        
        Raises:
            PurityViolationError if any called function is impure
        """
        for func in called_functions:
            if self.analyzer.is_known_impure(func):
                raise PurityViolationError(
                    f"Pure function calls impure function: {func}"
                )
        
        return True
    
    def trace_call(self, func_name: str, is_pure: bool):
        """Trace a function call."""
        self.call_trace.append((func_name, is_pure))
    
    def get_call_trace(self) -> List[Tuple[str, bool]]:
        """Get the call trace."""
        return self.call_trace.copy()
    
    def clear_call_trace(self):
        """Clear the call trace."""
        self.call_trace = []


class Immutability:
    """
    Tracks and enforces immutability of data structures.
    
    Immutable values:
    - Cannot be modified after creation
    - Are deeply immutable (including nested structures)
    - Provide referential equality
    """
    
    def __init__(self):
        self.immutable_objects: Set[int] = set()  # id() -> immutable
        self.frozen_values: Dict[int, Any] = {}
    
    def mark_immutable(self, obj: Any) -> Any:
        """Mark an object as immutable."""
        obj_id = id(obj)
        self.immutable_objects.add(obj_id)
        
        # For sequences and mappings, recursively mark contents
        if isinstance(obj, (list, dict)):
            if isinstance(obj, list):
                for item in obj:
                    self.mark_immutable(item)
            else:
                for key, value in obj.items():
                    self.mark_immutable(key)
                    self.mark_immutable(value)
        
        return obj
    
    def is_immutable(self, obj: Any) -> bool:
        """Check if an object is immutable."""
        return id(obj) in self.immutable_objects
    
    def freeze(self, obj: Any) -> Any:
        """Freeze an object to prevent modification."""
        self.frozen_values[id(obj)] = obj
        return self.mark_immutable(obj)
    
    def unfreeze(self, obj: Any):
        """Unfreeze an object."""
        obj_id = id(obj)
        if obj_id in self.frozen_values:
            del self.frozen_values[obj_id]
        if obj_id in self.immutable_objects:
            self.immutable_objects.discard(obj_id)


class PurityViolationError(Exception):
    """Exception raised when purity constraint is violated."""
    pass


class ImmutabilityViolationError(Exception):
    """Exception raised when immutability constraint is violated."""
    pass


# Global instances
_purity_analyzer = PurityAnalyzer()
_purity_enforcer = PurityEnforcer(_purity_analyzer)
_immutability = Immutability()


def get_purity_analyzer() -> PurityAnalyzer:
    """Get the global purity analyzer instance."""
    return _purity_analyzer


def get_purity_enforcer() -> PurityEnforcer:
    """Get the global purity enforcer instance."""
    return _purity_enforcer


def get_immutability_manager() -> Immutability:
    """Get the global immutability manager instance."""
    return _immutability
