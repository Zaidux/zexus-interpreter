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
    
    # Blockchain-specific operations (110-119)
    # These opcodes provide native VM support for blockchain operations
    # dramatically improving smart contract execution performance
    HASH_BLOCK = 110        # Hash a block structure (SHA-256)
    VERIFY_SIGNATURE = 111  # Verify cryptographic signature
    MERKLE_ROOT = 112       # Calculate Merkle tree root
    STATE_READ = 113        # Read from blockchain state
    STATE_WRITE = 114       # Write to blockchain state (TX context only)
    TX_BEGIN = 115          # Begin transaction context
    TX_COMMIT = 116         # Commit transaction changes
    TX_REVERT = 117         # Revert transaction (rollback)
    GAS_CHARGE = 118        # Deduct gas from execution limit
    LEDGER_APPEND = 119     # Append entry to immutable ledger
    ASSERT_PROTOCOL = 102
    
    # Register-based operations (200-299) - Phase 5
    # These provide 1.5-3x faster arithmetic via register allocation
    LOAD_REG = 200          # Load constant to register: LOAD_REG r1, 42
    LOAD_VAR_REG = 201      # Load variable to register: LOAD_VAR_REG r1, "x"
    STORE_REG = 202         # Store register to variable: STORE_REG r1, "x"
    MOV_REG = 203           # Move between registers: MOV_REG r2, r1
    
    ADD_REG = 210           # r3 = r1 + r2
    SUB_REG = 211           # r3 = r1 - r2
    MUL_REG = 212           # r3 = r1 * r2
    DIV_REG = 213           # r3 = r1 / r2
    MOD_REG = 214           # r3 = r1 % r2
    POW_REG = 215           # r3 = r1 ** r2
    NEG_REG = 216           # r2 = -r1 (unary)
    
    EQ_REG = 220            # r3 = (r1 == r2)
    NEQ_REG = 221           # r3 = (r1 != r2)
    LT_REG = 222            # r3 = (r1 < r2)
    GT_REG = 223            # r3 = (r1 > r2)
    LTE_REG = 224           # r3 = (r1 <= r2)
    GTE_REG = 225           # r3 = (r1 >= r2)
    
    AND_REG = 230           # r3 = r1 && r2
    OR_REG = 231            # r3 = r1 || r2
    NOT_REG = 232           # r2 = !r1
    
    PUSH_REG = 240          # Push register value to stack (hybrid mode)
    POP_REG = 241           # Pop stack value to register (hybrid mode)
    
    # Parallel execution operations (300-399) - Phase 6
    # These enable multi-core parallel bytecode execution for 2-4x speedup
    PARALLEL_START = 300    # Mark start of parallelizable region
    PARALLEL_END = 301      # Mark end of parallelizable region
    BARRIER = 302           # Synchronization barrier (wait for all parallel tasks)
    SPAWN_TASK = 303        # Spawn a new parallel task
    TASK_JOIN = 304         # Wait for specific task to complete
    TASK_RESULT = 305       # Get result from completed task
    LOCK_ACQUIRE = 306      # Acquire shared lock
    LOCK_RELEASE = 307      # Release shared lock
    ATOMIC_ADD = 308        # Atomic addition to shared variable
    ATOMIC_CAS = 309        # Compare-and-swap operation
    
    # I/O Operations
    PRINT = 250
    READ = 251
    WRITE = 252
    
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
    
    # Blockchain-specific helper methods
    def emit_hash_block(self) -> int:
        """Emit HASH_BLOCK instruction - expects block data on stack"""
        return self.emit("HASH_BLOCK")
    
    def emit_verify_signature(self) -> int:
        """Emit VERIFY_SIGNATURE - expects signature, message, public_key on stack"""
        return self.emit("VERIFY_SIGNATURE")
    
    def emit_merkle_root(self, leaf_count: int) -> int:
        """Emit MERKLE_ROOT - expects leaf_count items on stack"""
        return self.emit("MERKLE_ROOT", leaf_count)
    
    def emit_state_read(self, key: str) -> int:
        """Emit STATE_READ instruction"""
        key_idx = self.bytecode.add_constant(key)
        return self.emit("STATE_READ", key_idx)
    
    def emit_state_write(self, key: str) -> int:
        """Emit STATE_WRITE instruction - expects value on stack"""
        key_idx = self.bytecode.add_constant(key)
        return self.emit("STATE_WRITE", key_idx)
    
    def emit_tx_begin(self) -> int:
        """Emit TX_BEGIN instruction"""
        return self.emit("TX_BEGIN")
    
    def emit_tx_commit(self) -> int:
        """Emit TX_COMMIT instruction"""
        return self.emit("TX_COMMIT")
    
    def emit_tx_revert(self, reason: str = None) -> int:
        """Emit TX_REVERT instruction with optional reason"""
        if reason:
            reason_idx = self.bytecode.add_constant(reason)
            return self.emit("TX_REVERT", reason_idx)
        return self.emit("TX_REVERT")
    
    def emit_gas_charge(self, amount: int) -> int:
        """Emit GAS_CHARGE instruction"""
        return self.emit("GAS_CHARGE", amount)
    
    def emit_ledger_append(self) -> int:
        """Emit LEDGER_APPEND - expects entry on stack"""
        return self.emit("LEDGER_APPEND")
    
    # Convenience methods for test compatibility
    def emit_load_const(self, value: Any) -> int:
        """Emit LOAD_CONST instruction (alias for emit_constant)"""
        return self.emit_constant(value)
    
    def emit_load_name(self, name: str) -> int:
        """Emit LOAD_NAME instruction (alias for emit_load)"""
        return self.emit_load(name)
    
    def emit_store_name(self, name: str) -> int:
        """Emit STORE_NAME instruction (alias for emit_store)"""
        return self.emit_store(name)
    
    def emit_add(self) -> int:
        """Emit ADD instruction"""
        return self.emit("ADD")
    
    def emit_sub(self) -> int:
        """Emit SUB instruction"""
        return self.emit("SUB")
    
    def emit_mul(self) -> int:
        """Emit MUL instruction"""
        return self.emit("MUL")
    
    def emit_div(self) -> int:
        """Emit DIV instruction"""
        return self.emit("DIV")
    
    def emit_pow(self) -> int:
        """Emit POW instruction"""
        return self.emit("POW")
    
    def emit_mod(self) -> int:
        """Emit MOD instruction"""
        return self.emit("MOD")
    
    def emit_eq(self) -> int:
        """Emit EQ instruction"""
        return self.emit("EQ")
    
    def emit_lt(self) -> int:
        """Emit LT instruction"""
        return self.emit("LT")
    
    def emit_gt(self) -> int:
        """Emit GT instruction"""
        return self.emit("GT")
    
    def emit_pop(self) -> int:
        """Emit POP instruction"""
        return self.emit("POP")
    
    def emit_return(self) -> int:
        """Emit RETURN instruction"""
        return self.emit("RETURN")
    
    def emit_spawn(self, operand: Any = None) -> int:
        """Emit SPAWN instruction"""
        return self.emit("SPAWN", operand)
    
    def emit_await(self) -> int:
        """Emit AWAIT instruction"""
        return self.emit("AWAIT")
    
    def emit_register_event(self, event_data: Tuple[str, str]) -> int:
        """Emit REGISTER_EVENT instruction"""
        event_name_idx = self.bytecode.add_constant(event_data[0])
        handler_idx = self.bytecode.add_constant(event_data[1])
        return self.emit("REGISTER_EVENT", (event_name_idx, handler_idx))
    
    def emit_emit_event(self, event_name: str) -> int:
        """Emit EMIT_EVENT instruction"""
        event_idx = self.bytecode.add_constant(event_name)
        return self.emit("EMIT_EVENT", (event_idx,))
    
    def emit_label(self, label: str):
        """Alias for mark_label - for compatibility"""
        return self.mark_label(label)
    
    def build(self) -> Bytecode:
        """Finalize and return the bytecode"""
        self.resolve_labels()
        return self.bytecode