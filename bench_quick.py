#!/usr/bin/env python3
"""Quick benchmark for Phase 2 Rust VM."""
import sys, time
sys.path.insert(0, "src")
from zexus.vm.bytecode import Bytecode, Opcode
from zexus.vm.binary_bytecode import serialize
from zexus_core import RustVMExecutor

def mk(consts, instrs):
    bc = Bytecode()
    bc.constants = list(consts)
    bc.instructions = [(Opcode(op), op2) for (op, op2) in instrs]
    return serialize(bc)

def arith(n):
    c = ["sum", 0, "i", 1, n, 2]
    i = [
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
    return mk(c, i)

e = RustVMExecutor()
for n, it in [(1000, 100), (10000, 100), (100000, 10)]:
    d = arith(n)
    s = e.benchmark(d, iterations=it)
    mips = s["instructions_per_sec"] / 1e6
    print(f"  n={n:>6}, {it:>4} iters: {s['elapsed_ms']:>8.1f} ms, {mips:.1f} MIPS")

d = arith(10000)
s = e.benchmark(d, iterations=1000)
mips = s["instructions_per_sec"] / 1e6
print(f"  Peak: {mips:.1f} MIPS")
