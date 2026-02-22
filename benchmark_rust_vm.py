#!/usr/bin/env python3
"""
Phase 2 Benchmark — Rust VM vs Python VM.

Compares execution speed of the Rust bytecode interpreter (RustVMExecutor)
against the Python stack-based VM for various workloads.
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from zexus.vm.bytecode import Bytecode, Opcode
from zexus.vm.binary_bytecode import serialize

try:
    from zexus_core import RustVMExecutor
    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    print("WARNING: zexus_core not available – Rust benchmarks will be skipped")

# ── Workload helpers ──────────────────────────────────────────────────

def _make_zxc(constants, instructions):
    bc = Bytecode()
    bc.constants = list(constants)
    bc.instructions = [(Opcode(op), operand) for (op, operand) in instructions]
    return serialize(bc)


def make_arithmetic_loop(n):
    """Tight loop: sum = 0; for i in 0..n: sum += i*2+1."""
    consts = ["sum", 0, "i", 1, n, 2]
    # idx:     0     1   2   3  4  5
    instrs = [
        (Opcode.LOAD_CONST, 1), (Opcode.STORE_NAME, 0),  # sum=0
        (Opcode.LOAD_CONST, 1), (Opcode.STORE_NAME, 2),  # i=0
        # loop head at ip=4
        (Opcode.LOAD_NAME, 2), (Opcode.LOAD_CONST, 4), (Opcode.GTE, None),
        (Opcode.JUMP_IF_TRUE, 18),  # exit at 18
        # sum += i*2 + 1
        (Opcode.LOAD_NAME, 0),   # load sum
        (Opcode.LOAD_NAME, 2),   # load i
        (Opcode.LOAD_CONST, 5),  # 2
        (Opcode.MUL, None),
        (Opcode.LOAD_CONST, 3),  # 1
        (Opcode.ADD, None),       # i*2+1
        (Opcode.ADD, None),       # sum + (i*2+1)
        (Opcode.STORE_NAME, 0),  # sum = ...
        # i += 1
        (Opcode.LOAD_NAME, 2), (Opcode.LOAD_CONST, 3), (Opcode.ADD, None),
        (Opcode.STORE_NAME, 2),
        (Opcode.JUMP, 4),
        # exit at 18  (but we need 2 more instrs after JUMP)
    ]
    # Fix: JUMP is at index 17 (0-indexed after counting). Let me recount.
    #  0: LOAD_CONST 1
    #  1: STORE_NAME 0
    #  2: LOAD_CONST 1
    #  3: STORE_NAME 2
    #  4: LOAD_NAME 2
    #  5: LOAD_CONST 4
    #  6: GTE
    #  7: JUMP_IF_TRUE exit
    #  8: LOAD_NAME 0
    #  9: LOAD_NAME 2
    # 10: LOAD_CONST 5
    # 11: MUL
    # 12: LOAD_CONST 3
    # 13: ADD
    # 14: ADD
    # 15: STORE_NAME 0
    # 16: LOAD_NAME 2
    # 17: LOAD_CONST 3
    # 18: ADD         <-- oops, that's the exit target
    # Fix: exit at 21
    instrs = [
        (Opcode.LOAD_CONST, 1), (Opcode.STORE_NAME, 0),  # sum=0
        (Opcode.LOAD_CONST, 1), (Opcode.STORE_NAME, 2),  # i=0
        # loop head at ip=4
        (Opcode.LOAD_NAME, 2), (Opcode.LOAD_CONST, 4), (Opcode.GTE, None),
        (Opcode.JUMP_IF_TRUE, 21),  # exit at 21
        # body
        (Opcode.LOAD_NAME, 0),   # 8
        (Opcode.LOAD_NAME, 2),   # 9
        (Opcode.LOAD_CONST, 5),  # 10
        (Opcode.MUL, None),       # 11
        (Opcode.LOAD_CONST, 3),  # 12
        (Opcode.ADD, None),       # 13
        (Opcode.ADD, None),       # 14
        (Opcode.STORE_NAME, 0),  # 15
        # i += 1
        (Opcode.LOAD_NAME, 2),   # 16
        (Opcode.LOAD_CONST, 3),  # 17
        (Opcode.ADD, None),       # 18
        (Opcode.STORE_NAME, 2),  # 19
        (Opcode.JUMP, 4),         # 20: loop back
        # exit
        (Opcode.LOAD_NAME, 0),   # 21
        (Opcode.RETURN, None),    # 22
    ]
    return _make_zxc(consts, instrs)


def make_state_ops(n):
    """Write+read state in a loop n times."""
    consts = ["key", "val", "i", 0, n, 1, 42]
    #          0      1     2   3  4  5  6
    instrs = [
        (Opcode.LOAD_CONST, 3), (Opcode.STORE_NAME, 2),  # i=0
        # loop at 2
        (Opcode.LOAD_NAME, 2), (Opcode.LOAD_CONST, 4), (Opcode.GTE, None),
        (Opcode.JUMP_IF_TRUE, 14),
        # state_write + state_read
        (Opcode.LOAD_CONST, 6), (Opcode.STATE_WRITE, 0),  # state["key"]=42
        (Opcode.STATE_READ, 0), (Opcode.POP, None),        # pop
        # i += 1
        (Opcode.LOAD_NAME, 2), (Opcode.LOAD_CONST, 5), (Opcode.ADD, None),
        (Opcode.STORE_NAME, 2),
        (Opcode.JUMP, 2),
        # exit at 14 (but that's the JUMP target, need +1)
    ]
    # Recount:
    # 0: LOAD_CONST 3
    # 1: STORE_NAME 2     -> i=0
    # 2: LOAD_NAME 2      -> loop head
    # 3: LOAD_CONST 4
    # 4: GTE
    # 5: JUMP_IF_TRUE exit
    # 6: LOAD_CONST 6
    # 7: STATE_WRITE 0
    # 8: STATE_READ 0
    # 9: POP
    # 10: LOAD_NAME 2
    # 11: LOAD_CONST 5
    # 12: ADD
    # 13: STORE_NAME 2
    # 14: JUMP 2
    # 15: RETURN       exit
    instrs = [
        (Opcode.LOAD_CONST, 3), (Opcode.STORE_NAME, 2),  # i=0
        (Opcode.LOAD_NAME, 2), (Opcode.LOAD_CONST, 4), (Opcode.GTE, None),
        (Opcode.JUMP_IF_TRUE, 15),
        (Opcode.LOAD_CONST, 6), (Opcode.STATE_WRITE, 0),
        (Opcode.STATE_READ, 0), (Opcode.POP, None),
        (Opcode.LOAD_NAME, 2), (Opcode.LOAD_CONST, 5), (Opcode.ADD, None),
        (Opcode.STORE_NAME, 2),
        (Opcode.JUMP, 2),
        (Opcode.LOAD_CONST, 6), (Opcode.RETURN, None),  # 15, 16
    ]
    return _make_zxc(consts, instrs)


def make_hash_loop(n):
    """Hash strings in a loop n times."""
    consts = ["i", 0, n, 1, "data to hash"]
    #          0   1  2  3  4
    instrs = [
        (Opcode.LOAD_CONST, 1), (Opcode.STORE_NAME, 0),  # i=0
        (Opcode.LOAD_NAME, 0), (Opcode.LOAD_CONST, 2), (Opcode.GTE, None),
        (Opcode.JUMP_IF_TRUE, 13),
        (Opcode.LOAD_CONST, 4), (Opcode.HASH_BLOCK, None), (Opcode.POP, None),
        (Opcode.LOAD_NAME, 0), (Opcode.LOAD_CONST, 3), (Opcode.ADD, None),
        (Opcode.STORE_NAME, 0),
        (Opcode.JUMP, 2),
        (Opcode.LOAD_CONST, 1), (Opcode.RETURN, None),
    ]
    return _make_zxc(consts, instrs)


# ── Benchmark runner ─────────────────────────────────────────────────

def benchmark_rust(name, data, iterations=1000, gas_limit=0):
    executor = RustVMExecutor()
    stats = executor.benchmark(data, iterations=iterations, gas_limit=gas_limit)
    return stats


def benchmark_python_vm(name, constants, instructions, iterations=100):
    """Run in the Python VM for comparison."""
    from zexus.vm.vm import ZexusVM
    from zexus.vm.bytecode import Bytecode, Opcode
    bc = Bytecode()
    bc.constants = list(constants)
    bc.instructions = [(Opcode(op), operand) for (op, operand) in instructions]

    vm = ZexusVM()
    start = time.perf_counter()
    for _ in range(iterations):
        vm._run_stack_bytecode_sync(bc)
    elapsed = time.perf_counter() - start
    return {
        "iterations": iterations,
        "elapsed_ms": elapsed * 1000,
    }


def main():
    print("=" * 70)
    print("  Zexus Phase 2 — Rust VM Benchmark")
    print("=" * 70)
    print()

    workloads = [
        ("Arithmetic Loop (1K iters)", make_arithmetic_loop(1000)),
        ("Arithmetic Loop (10K iters)", make_arithmetic_loop(10000)),
        ("State Read/Write (1K ops)", make_state_ops(1000)),
        ("SHA-256 Hash (100 hashes)", make_hash_loop(100)),
        ("SHA-256 Hash (1K hashes)", make_hash_loop(1000)),
    ]

    for name, data in workloads:
        if HAS_RUST:
            stats = benchmark_rust(name, data, iterations=100)
            ips = stats["instructions_per_sec"]
            elapsed = stats["elapsed_ms"]
            total = stats["total_instructions"]
            print(f"  [Rust]  {name:<35s}  "
                  f"{elapsed:>8.2f} ms  "
                  f"{total:>10,} instrs  "
                  f"{ips:>12,.0f} IPS")
        else:
            print(f"  [Rust]  {name:<35s}  SKIPPED (no zexus_core)")

    print()
    print("-" * 70)
    print()

    # Quick Rust-only benchmark: throughput at peak
    if HAS_RUST:
        print("  Peak throughput (arithmetic loop 10K, 1000 iterations):")
        data = make_arithmetic_loop(10000)
        stats = benchmark_rust("peak", data, iterations=1000)
        ips = stats["instructions_per_sec"]
        elapsed = stats["elapsed_ms"]
        total = stats["total_instructions"]
        print(f"    {elapsed:.1f} ms total, {total:,} instructions, "
              f"{ips:,.0f} IPS")
        print(f"    {ips/1_000_000:.1f} MIPS (million instructions per second)")
        print()

    print("=" * 70)
    print("  Benchmark complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
