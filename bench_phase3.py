#!/usr/bin/env python3
"""
Phase 3 — Benchmark: Adaptive VM Routing Performance

Compares execution speed of:
  1. Python VM only (Rust VM disabled)
  2. Adaptive routing (automatic Rust/Python switching)
  3. Direct Rust VM execution

Also benchmarks the RustStateAdapter.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from zexus.vm.bytecode import Bytecode, Opcode
from zexus.vm.binary_bytecode import serialize
from zexus.vm.vm import VM


def make_program(n_ops):
    """Create a numeric computation: 0 + 1 + 1 + ... (n_ops additions)."""
    bc = Bytecode()
    bc.constants = [0, 1]
    instrs = [(Opcode.LOAD_CONST, 0)]
    for _ in range(n_ops):
        instrs.append((Opcode.LOAD_CONST, 1))
        instrs.append((Opcode.ADD, None))
    instrs.append((Opcode.RETURN, None))
    bc.instructions = instrs
    return bc


def bench_python_vm(bc, iterations=5):
    """Benchmark with Rust VM disabled."""
    vm = VM()
    vm._perf_fast_dispatch = True
    vm._rust_vm_enabled = False

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = vm.execute(bc)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_ms = sum(times) / len(times) * 1000
    return result, avg_ms


def bench_adaptive_vm(bc, iterations=5, threshold=10_000):
    """Benchmark with adaptive Rust/Python routing."""
    vm = VM()
    vm._perf_fast_dispatch = True
    vm._rust_vm_threshold = threshold

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = vm.execute(bc)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_ms = sum(times) / len(times) * 1000
    stats = vm.get_rust_vm_stats()
    return result, avg_ms, stats


def bench_direct_rust(bc, iterations=5):
    """Benchmark direct Rust VM execution."""
    try:
        from zexus_core import RustVMExecutor
    except ImportError:
        return None, 0.0

    executor = RustVMExecutor()
    zxc_data = serialize(bc)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = executor.execute(zxc_data, gas_limit=0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_ms = sum(times) / len(times) * 1000
    return result.get("result"), avg_ms


def bench_state_adapter():
    """Benchmark RustStateAdapter performance."""
    try:
        from zexus_core import RustStateAdapter
    except ImportError:
        print("  RustStateAdapter not available")
        return

    sizes = [1_000, 10_000, 100_000]
    for size in sizes:
        adapter = RustStateAdapter()
        data = {f"key_{i}": i for i in range(size)}

        start = time.perf_counter()
        adapter.load_from_dict(data)
        load_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        for i in range(min(size, 10_000)):
            adapter.get(f"key_{i}")
        read_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        for i in range(min(size, 10_000)):
            adapter.set(f"key_{i}", i * 2)
        write_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        dirty = adapter.flush_dirty()
        flush_ms = (time.perf_counter() - start) * 1000

        print(f"  {size:>7,} keys: load={load_ms:.1f}ms  "
              f"read={read_ms:.1f}ms  write={write_ms:.1f}ms  "
              f"flush={flush_ms:.1f}ms")


def main():
    print("=" * 70)
    print("Phase 3 — Adaptive VM Routing Benchmark")
    print("=" * 70)
    print()

    # Test various program sizes
    sizes = [100, 1_000, 5_000, 10_000, 50_000]
    iters = 5

    print(f"{'Size':>8} | {'Python VM':>12} | {'Adaptive':>12} | {'Direct Rust':>12} | {'Speedup':>8}")
    print("-" * 70)

    for size in sizes:
        bc = make_program(size)
        n_instrs = len(bc.instructions)

        py_result, py_ms = bench_python_vm(bc, iters)
        ada_result, ada_ms, ada_stats = bench_adaptive_vm(bc, iters)
        rust_result, rust_ms = bench_direct_rust(bc, iters)

        speedup = py_ms / ada_ms if ada_ms > 0 else 0
        rust_speedup = py_ms / rust_ms if rust_ms > 0 else 0
        rust_used = ada_stats.get("rust_executions", 0)
        route = "Rust" if rust_used > 0 else "Python"

        print(f"{n_instrs:>8,} | {py_ms:>9.2f} ms | {ada_ms:>9.2f} ms | {rust_ms:>9.2f} ms | {speedup:>5.1f}x/{rust_speedup:>5.1f}x [{route}]")

    print()
    print("  Speedup format: Adaptive/DirectRust vs Python VM")

    print()
    print("RustStateAdapter Benchmark:")
    print("-" * 70)
    bench_state_adapter()

    print()
    print("=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
