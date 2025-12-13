"""
Bytecode Definitions for Zexus VM

This module provides comprehensive bytecode representation and manipulation
for both the compiler and interpreter (evaluator) paths.
"""
from typing import List, Any, Tuple, Dict, Optional
from enum import IntEnum


class Opcode(IntEnum):
    """
    Comprehensive opcode set for Zexus VM.
    Supports both high-level and low-level operations.
    """
    # Stack operations
    LOAD_CONST = 1      # Push constant onto stack
    LOAD_NAME = 2       # Load variable by name
    STORE_NAME = 3      # Store value to variable
    STORE_FUNC = 4      # Store function descriptor
    POP = 5             # Pop top of stack
    DUP = 6             # Duplicate top of stack
    
    # Arithmetic operations
    ADD = 10
    SUB = 11
    MUL = 12
    DIV = 13
    MOD = 14
    POW = 15
    NEG = 16            # Unary negation
    
    # Comparison operations
    EQ = 20             # ==
    NEQ = 21            # !=
    LT = 22             # <
    GT = 23             # >
    LTE = 24            # <=
    GTE = 25            # >=
    
    # Logical operations
    AND = 30            # &&
    OR = 31             # ||
    NOT = 32            # !
    
    # Control flow
    JUMP = 40           # Unconditional jump
    JUMP_IF_FALSE = 41  # Conditional jump
    JUMP_IF_TRUE = 42   # Conditional jump (true)
    RETURN = 43         # Return from function
    
    # Function/Action calls
    CALL_NAME = 50      # Call function by name
    CALL_FUNC_CONST = 51  # Call function by constant descriptor
    CALL_TOP = 52       # Call function on top of stack
    CALL_BUILTIN = 53   # Call builtin function
    
    # Collections
    BUILD_LIST = 60     # Build list from stack items
    BUILD_MAP = 61      # Build map from stack items
    BUILD_SET = 62      # Build set from stack items
    INDEX = 63          # Index into collection
    SLICE = 64          # Slice operation
    
    # Async/Concurrency
    SPAWN = 70          # Spawn coroutine/task
    AWAIT = 71          # Await coroutine
    SPAWN_CALL = 72     # Spawn function call as task
    
    # Events
    REGISTER_EVENT = 80
    EMIT_EVENT = 81
    
    # Modules
    IMPORT = 90
    EXPORT = 91
    
    # Advanced features
    DEFINE_ENUM = 100
    DEFINE_PROTOCOL = 101
    ASSERT_PROTOCOL = 102
    
    # I/O Operations
    PRINT = 110
    READ = 111
    WRITE = 112
    
    # High-level constructs (for evaluator)
    DEFINE_SCREEN = 120
    DEFINE_COMPONENT = 121
    DEFINE_THEME = 122
    
    # Special
    NOP = 255           # No operation


class Bytecode:
    """
    Bytecode container with instructions and constants pool.
    Used by both compiler and evaluator to represent compiled code.
    """
    def __init__(self):
        self.instructions: List[Tuple[str, Any]] = []
        self.constants: List[Any] = []
        self.metadata: Dict[str, Any] = {
            'source_file': None,
            'version': '1.0',
            'created_by': 'unknown'
        }
        
    def add_instruction(self, opcode: str, operand: Any = None) -> int:
        """Add an instruction and return its index"""
        idx = len(self.instructions)
        self.instructions.append((opcode, operand))
        return idx
    
    def add_constant(self, value: Any) -> int:
        """Add a constant to the pool and return its index"""
        # Check if constant already exists (optimization)
        for i, const in enumerate(self.constants):
            if const == value and type(const) == type(value):
                return i
        
        idx = len(self.constants)
        self.constants.append(value)
        return idx
    
    def update_instruction(self, idx: int, opcode: str, operand: Any = None):
        """Update an instruction at a specific index"""
        if 0 <= idx < len(self.instructions):
            self.instructions[idx] = (opcode, operand)
    
    def get_instruction(self, idx: int) -> Optional[Tuple[str, Any]]:
        """Get instruction at index"""
        if 0 <= idx < len(self.instructions):
            return self.instructions[idx]
        return None
    
    def get_constant(self, idx: int) -> Any:
        """Get constant at index"""
        if 0 <= idx < len(self.constants):
            return self.constants[idx]
        return None
    
    def set_metadata(self, key: str, value: Any):
        """Set metadata for this bytecode"""
        self.metadata[key] = value
    
    def disassemble(self) -> str:
        """
        Generate human-readable disassembly of bytecode.
        Useful for debugging.
        """
        lines = []
        lines.append(f"Bytecode Object ({len(self.instructions)} instructions, {len(self.constants)} constants)")
        lines.append("=" * 60)
        
        # Constants section
        if self.constants:
            lines.append("\nConstants:")
            for i, const in enumerate(self.constants):
                const_repr = repr(const)
                if len(const_repr) > 50:
                    const_repr = const_repr[:47] + "..."
                lines.append(f"  {i:3d}: {const_repr}")
        
        # Instructions section
        lines.append("\nInstructions:")
        for i, (opcode, operand) in enumerate(self.instructions):
            if operand is not None:
                lines.append(f"  {i:4d}  {opcode:20s} {operand}")
            else:
                lines.append(f"  {i:4d}  {opcode:20s}")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return f"Bytecode({len(self.instructions)} instructions, {len(self.constants)} constants)"
    
    def __len__(self):
        return len(self.instructions)


class BytecodeBuilder:
    """
    Helper class for building bytecode with higher-level constructs.
    Provides convenience methods for common patterns.
    """
    def __init__(self):
        self.bytecode = Bytecode()
        self._label_positions: Dict[str, int] = {}
        self._forward_refs: Dict[str, List[int]] = {}
    
    def emit(self, opcode: str, operand: Any = None) -> int:
        """Emit an instruction"""
        return self.bytecode.add_instruction(opcode, operand)
    
    def emit_constant(self, value: Any) -> int:
        """Emit LOAD_CONST instruction"""
        const_idx = self.bytecode.add_constant(value)
        return self.emit("LOAD_CONST", const_idx)
    
    def emit_load(self, name: str) -> int:
        """Emit LOAD_NAME instruction"""
        name_idx = self.bytecode.add_constant(name)
        return self.emit("LOAD_NAME", name_idx)
    
    def emit_store(self, name: str) -> int:
        """Emit STORE_NAME instruction"""
        name_idx = self.bytecode.add_constant(name)
        return self.emit("STORE_NAME", name_idx)
    
    def emit_call(self, name: str, arg_count: int) -> int:
        """Emit CALL_NAME instruction"""
        name_idx = self.bytecode.add_constant(name)
        return self.emit("CALL_NAME", (name_idx, arg_count))
    
    def mark_label(self, label: str):
        """Mark a position with a label for jumps"""
        self._label_positions[label] = len(self.bytecode.instructions)
    
    def emit_jump(self, label: str) -> int:
        """Emit a jump to a label (resolved later)"""
        idx = self.emit("JUMP", None)
        self._forward_refs.setdefault(label, []).append(idx)
        return idx
    
    def emit_jump_if_false(self, label: str) -> int:
        """Emit a conditional jump to a label"""
        idx = self.emit("JUMP_IF_FALSE", None)
        self._forward_refs.setdefault(label, []).append(idx)
        return idx
    
    def resolve_labels(self):
        """Resolve all forward label references"""
        for label, instr_indices in self._forward_refs.items():
            if label in self._label_positions:
                target = self._label_positions[label]
                for idx in instr_indices:
                    opcode, _ = self.bytecode.instructions[idx]
                    self.bytecode.instructions[idx] = (opcode, target)
    
    def build(self) -> Bytecode:
        """Finalize and return the bytecode"""
        self.resolve_labels()
        return self.bytecode