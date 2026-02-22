#!/usr/bin/env python3
"""Minimal comparison: Python VM loop vs Rust VM."""
import sys, time
sys.path.insert(0, "src")
from zexus.vm.bytecode import Bytecode, Opcode
from zexus.vm.binary_bytecode import serialize
from zexus_core import RustVMExecutor

# Build bytecode: 1K iteration arithmetic loop
N = 1000
bc = Bytecode()
bc.constants = ["sum", 0, "i", 1, N, 2]
bc.instructions = [
    (Opcode.LOAD_CONST, 1), (Opcode.STORE_NAME, 0),
    (Opcode.LOAD_CONST, 1), (Opcode.STORE_NAME, 2),
    (Opcode.LOAD_NAME, 2), (Opcode.LOAD_CONST, 4), (Opcode.GTE, None),
    (Opcode.JUMP_IF_TRUE, 21),
    (Opcode.LOAD_NAME, 0), (Opcode.LOAD_NAME, 2),
    (Opcode.LOAD_CONST, 5), (Opcode.MUL, None),
    (Opcode.LOAD_CONST, 3), (Opcode.ADD, None), (Opcode.ADD, None),
    (Opcode.STORE_NAME, 0),
    (Opcode.LOAD_NAME, 2), (Opcode.LOAD_CONST, 3), (Opcode.ADD, None),
    (Opcode.STORE_NAME, 2),
    (Opcode.JUMP, 4),
    (Opcode.LOAD_NAME, 0), (Opcode.RETURN, None),
]
data = serialize(bc)

# Python: manual stack-based execution (simulate Python VM hot loop)
def python_run(bc_obj):
    """Minimal Python stack-machine interpreter (matches core of VM._run_stack_bytecode_sync)."""
    consts = bc_obj.constants
    instrs = bc_obj.instructions
    stack = []
    env = {}
    ip = 0
    n = len(instrs)
    while ip < n:
        op, operand = instrs[ip]
        ip += 1
        opv = op.value if hasattr(op, 'value') else op
        if opv == 1:  # LOAD_CONST
            stack.append(consts[operand])
        elif opv == 2:  # LOAD_NAME
            name = consts[operand]
            stack.append(env.get(name))
        elif opv == 3:  # STORE_NAME
            name = consts[operand]
            env[name] = stack.pop()
        elif opv == 10:  # ADD
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif opv == 11:  # SUB
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif opv == 12:  # MUL
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif opv == 25:  # GTE
            b, a = stack.pop(), stack.pop()
            stack.append(a >= b)
        elif opv == 40:  # JUMP
            ip = operand
        elif opv == 42:  # JUMP_IF_TRUE
            cond = stack.pop()
            if cond:
                ip = operand
        elif opv == 43:  # RETURN
            return stack.pop() if stack else None
    return stack.pop() if stack else None

# Python benchmark
ITERS = 100
start = time.perf_counter()
for _ in range(ITERS):
    python_run(bc)
py_elapsed = time.perf_counter() - start

# Rust benchmark
executor = RustVMExecutor()
stats = executor.benchmark(data, iterations=ITERS)

py_ms = py_elapsed * 1000
rust_ms = stats["elapsed_ms"]
speedup = py_ms / rust_ms if rust_ms > 0 else 0

print(f"Python: {py_ms:.1f} ms ({ITERS} iters)")
print(f"Rust:   {rust_ms:.1f} ms ({ITERS} iters)")
print(f"Speedup: {speedup:.1f}x")
print(f"Rust throughput: {stats['instructions_per_sec']/1e6:.1f} MIPS")
