#!/usr/bin/env python3
"""Phase 6 Benchmark — Rust Builtins vs Python Fallback.

Measures throughput of builtin-heavy contracts executing entirely in Rust
versus the fallback path.
"""

import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.zexus.vm.bytecode import Bytecode, Opcode
from src.zexus.vm.binary_bytecode import serialize
from zexus_core import RustVMExecutor, RustBatchExecutor, RustHasher

def make_zxc(constants, instructions):
    bc = Bytecode(constants=constants, instructions=instructions)
    return serialize(bc, include_checksum=True)

def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

# ── 1. Single-invocation keccak256 ────────────────────────────────────

banner("1. Single keccak256 (Rust builtin vs Python hasher)")

bc_data = make_zxc(
    ['keccak256', 'benchmark_data_payload'],
    [(Opcode.LOAD_CONST, 1), (Opcode.CALL_BUILTIN, (0, 1)), (Opcode.RETURN, None)],
)

executor = RustVMExecutor()
N = 10_000

t0 = time.perf_counter()
for _ in range(N):
    executor.execute(bc_data)
rust_time = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N):
    RustHasher.keccak256_str('benchmark_data_payload')
raw_time = time.perf_counter() - t0

print(f"  Rust VM keccak256 x {N:,}: {rust_time:.3f}s  ({N/rust_time:,.0f} ops/s)")
print(f"  Raw RustHasher    x {N:,}: {raw_time:.3f}s  ({N/raw_time:,.0f} ops/s)")
print(f"  VM overhead per call: {(rust_time - raw_time) / N * 1e6:.1f} µs")

# ── 2. Batch keccak256 ───────────────────────────────────────────────

banner("2. Batch keccak256 (1000 txns, native executor)")

batch_exec = RustBatchExecutor(max_workers=4)
txs = [
    {"contract_address": f"c{i%50}", "caller": "alice", "bytecode": bc_data, "state": {}}
    for i in range(1000)
]

t0 = time.perf_counter()
result = batch_exec.execute_batch_native(txs)
batch_time = time.perf_counter() - t0

print(f"  Succeeded: {result['succeeded']}/{result['total']}")
print(f"  Fallbacks: {result['fallbacks']}")
print(f"  Time: {batch_time:.3f}s")
print(f"  Throughput: {result['throughput']:,.0f} TPS")

# ── 3. Batch with builtins mix ──────────────────────────────────────

banner("3. Mixed builtins batch (keccak + emit + transfer)")

bc_mix = make_zxc(
    ['keccak256', 'payload', 'emit', 'Hashed', 'transfer', 'bob', 100, 'get_balance', 'bob'],
    [
        (Opcode.LOAD_CONST, 1),
        (Opcode.CALL_BUILTIN, (0, 1)),  # keccak256('payload')
        (Opcode.POP, None),
        (Opcode.LOAD_CONST, 3),         # 'Hashed'
        (Opcode.LOAD_CONST, 1),         # 'payload'
        (Opcode.CALL_BUILTIN, (2, 2)),  # emit('Hashed', 'payload')
        (Opcode.POP, None),
        (Opcode.LOAD_CONST, 5),         # 'bob'
        (Opcode.LOAD_CONST, 6),         # 100
        (Opcode.CALL_BUILTIN, (4, 2)),  # transfer('bob', 100)
        (Opcode.POP, None),
        (Opcode.LOAD_CONST, 8),         # 'bob'
        (Opcode.CALL_BUILTIN, (7, 1)),  # get_balance('bob')
        (Opcode.RETURN, None),
    ],
)

txs_mix = [
    {
        "contract_address": f"c{i%20}",
        "caller": f"alice_{i}",
        "bytecode": bc_mix,
        "state": {f"_balance:alice_{i}": 10000, "_balance:bob": 0},
    }
    for i in range(1000)
]

t0 = time.perf_counter()
result_mix = batch_exec.execute_batch_native(txs_mix)
mix_time = time.perf_counter() - t0

print(f"  Succeeded: {result_mix['succeeded']}/{result_mix['total']}")
print(f"  Fallbacks: {result_mix['fallbacks']}")
print(f"  Events: {len(result_mix.get('events', []))}")
print(f"  Time: {mix_time:.3f}s")
print(f"  Throughput: {result_mix['throughput']:,.0f} TPS")

# ── 4. String builtins batch ────────────────────────────────────────

banner("4. String builtins batch (upper + replace + split)")

bc_str = make_zxc(
    ['upper', 'hello world', 'replace', 'HELLO WORLD', 'WORLD', 'RUST', 'split', 'a,b,c,d,e', ','],
    [
        (Opcode.LOAD_CONST, 1),
        (Opcode.CALL_BUILTIN, (0, 1)),  # upper('hello world')
        (Opcode.POP, None),
        (Opcode.LOAD_CONST, 3),
        (Opcode.LOAD_CONST, 4),
        (Opcode.LOAD_CONST, 5),
        (Opcode.CALL_BUILTIN, (2, 3)),  # replace('...', 'WORLD', 'RUST')
        (Opcode.POP, None),
        (Opcode.LOAD_CONST, 7),
        (Opcode.LOAD_CONST, 8),
        (Opcode.CALL_BUILTIN, (6, 2)),  # split('a,b,c,d,e', ',')
        (Opcode.RETURN, None),
    ],
)

txs_str = [
    {"contract_address": "str_test", "caller": "alice", "bytecode": bc_str, "state": {}}
    for _ in range(1000)
]

t0 = time.perf_counter()
result_str = batch_exec.execute_batch_native(txs_str)
str_time = time.perf_counter() - t0

print(f"  Succeeded: {result_str['succeeded']}/{result_str['total']}")
print(f"  Fallbacks: {result_str['fallbacks']}")
print(f"  Time: {str_time:.3f}s")
print(f"  Throughput: {result_str['throughput']:,.0f} TPS")

# ── 5. CALL_NAME dispatch ───────────────────────────────────────────

banner("5. CALL_NAME dispatch (keccak256 via CALL_NAME — Phase 6 optimization)")

bc_cn = make_zxc(
    ['keccak256', 'call_name_test'],
    [(Opcode.LOAD_CONST, 1), (Opcode.CALL_NAME, (0, 1)), (Opcode.RETURN, None)],
)

txs_cn = [
    {"contract_address": "cn_test", "caller": "alice", "bytecode": bc_cn, "state": {}}
    for _ in range(1000)
]

t0 = time.perf_counter()
result_cn = batch_exec.execute_batch_native(txs_cn)
cn_time = time.perf_counter() - t0

print(f"  Succeeded: {result_cn['succeeded']}/{result_cn['total']}")
print(f"  Fallbacks: {result_cn['fallbacks']}")
print(f"  Time: {cn_time:.3f}s")
print(f"  Throughput: {result_cn['throughput']:,.0f} TPS")

# ── Summary ──────────────────────────────────────────────────────────

banner("PHASE 6 BENCHMARK SUMMARY")
print(f"  Single keccak256:     {N/rust_time:>10,.0f} ops/s")
print(f"  Batch keccak256:      {result['throughput']:>10,.0f} TPS")
print(f"  Mixed builtins:       {result_mix['throughput']:>10,.0f} TPS")
print(f"  String builtins:      {result_str['throughput']:>10,.0f} TPS")
print(f"  CALL_NAME dispatch:   {result_cn['throughput']:>10,.0f} TPS")
print(f"  Zero fallbacks:       {result['fallbacks'] + result_mix['fallbacks'] + result_str['fallbacks'] + result_cn['fallbacks']} total")
print()
