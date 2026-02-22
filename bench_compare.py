#!/usr/bin/env python3
"""Compare Python VM vs Rust VM on same bytecode."""
import sys, time
sys.path.insert(0, "src")
from zexus.vm.bytecode import Bytecode, Opcode
from zexus.vm.binary_bytecode import serialize
from zexus.vm.vm import VM as ZexusVM
from zexus_core import RustVMExecutor

def make_bc(n):
    bc = Bytecode()
    bc.constants = ["sum", 0, "i", 1, n, 2]
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
    return bc

N = 1000
ITERS = 10

# Python VM
bc = make_bc(N)
vm = ZexusVM()
start = time.perf_counter()
for _ in range(ITERS):
    vm._run_stack_bytecode_sync(bc)
py_elapsed = time.perf_counter() - start
py_instrs = N * 17 * ITERS  # ~17 instrs per loop iteration
py_ips = py_instrs / py_elapsed

# Rust VM
data = serialize(bc)
executor = RustVMExecutor()
stats = executor.benchmark(data, iterations=ITERS)
rust_ips = stats["instructions_per_sec"]

speedup = rust_ips / py_ips if py_ips > 0 else 0
print(f"Python VM: {py_elapsed*1000:.1f} ms, ~{py_ips/1e6:.2f} MIPS")
print(f"Rust VM:   {stats['elapsed_ms']:.1f} ms, ~{rust_ips/1e6:.2f} MIPS")
print(f"Speedup:   {speedup:.1f}x")
