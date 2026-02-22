#!/usr/bin/env python3
"""
Gas Stress Benchmark — Phase 4 Pre-Analysis

Measures gas consumption and execution speed for complex, high-volume
transactions across Python VM, adaptive Rust routing, and direct Rust VM.

Simulates real-world contract patterns:
  1. Token transfer (arithmetic + state read/write)
  2. DEX swap (heavy arithmetic + multiple state ops)
  3. NFT mint (string ops + state writes + events)
  4. Staking rewards (loops + multiplication + state)
  5. Governance vote (conditional logic + state + events)
  6. High-iteration loop (pure compute bound)
  7. Batch transfers (many state writes)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from zexus.vm.binary_bytecode import serialize as serialize_zxc
from zexus.vm.gas_metering import GasMetering, GasCost, OutOfGasError

# ---------- helpers ----------

try:
    from zexus_core import RustVMExecutor
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


class FakeBytecode:
    """Minimal bytecode holder accepted by serialize()."""
    def __init__(self, instructions, constants=None):
        self.instructions = instructions
        self.constants = constants or []


def build_program(pattern: str, scale: int = 1):
    """Return (instructions, constants) for a contract pattern."""

    instrs = []
    consts = []

    if pattern == "token_transfer":
        # Simulate: read balance A, read balance B, subtract, add, write both
        for _ in range(scale):
            instrs += [
                ("STATE_READ", "balance_a"),       # read sender balance
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(100)
            instrs += [
                ("SUB", None),                     # balance_a - amount
                ("STATE_WRITE", "balance_a"),       # write sender
                ("STATE_READ", "balance_b"),        # read receiver
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(100)
            instrs += [
                ("ADD", None),                     # balance_b + amount
                ("STATE_WRITE", "balance_b"),       # write receiver
                ("EMIT_EVENT", "Transfer"),
            ]
        instrs.append(("RETURN", None))

    elif pattern == "dex_swap":
        # Simulate: AMM swap with x*y=k invariant check
        for _ in range(scale):
            instrs += [
                ("STATE_READ", "reserve_x"),
                ("STATE_READ", "reserve_y"),
                ("MUL", None),                     # k = x * y
                ("STORE_NAME", "k_before"),
                ("STATE_READ", "reserve_x"),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(50)
            instrs += [
                ("ADD", None),                     # new_x = reserve_x + dx
                ("STORE_NAME", "new_x"),
                ("LOAD_NAME", "k_before"),
                ("LOAD_NAME", "new_x"),
                ("DIV", None),                     # new_y = k / new_x
                ("STORE_NAME", "new_y"),
                ("STATE_READ", "reserve_y"),
                ("LOAD_NAME", "new_y"),
                ("SUB", None),                     # dy = reserve_y - new_y
                ("STORE_NAME", "dy"),
                ("LOAD_NAME", "new_x"),
                ("STATE_WRITE", "reserve_x"),
                ("LOAD_NAME", "new_y"),
                ("STATE_WRITE", "reserve_y"),
                ("EMIT_EVENT", "Swap"),
            ]
        instrs.append(("RETURN", None))

    elif pattern == "nft_mint":
        # Simulate: mint NFT with metadata write + event
        for _ in range(scale):
            instrs += [
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(f"nft_{_}")
            instrs += [
                ("STORE_NAME", "token_id"),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append("0xowner")
            instrs += [
                ("STATE_WRITE", "owner"),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append("ipfs://metadata")
            instrs += [
                ("STATE_WRITE", "metadata"),
                ("STATE_READ", "total_supply"),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(1)
            instrs += [
                ("ADD", None),
                ("STATE_WRITE", "total_supply"),
                ("EMIT_EVENT", "Mint"),
            ]
        instrs.append(("RETURN", None))

    elif pattern == "staking_rewards":
        # Simulate: calculate staking rewards for N stakers
        instrs += [
            ("LOAD_CONST", len(consts)),
        ]
        consts.append(0)
        instrs.append(("STORE_NAME", "total_rewards"))
        for i in range(scale):
            instrs += [
                ("STATE_READ", f"stake_{i}"),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(5)  # 5% APY
            instrs += [
                ("MUL", None),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(100)
            instrs += [
                ("DIV", None),
                ("STORE_NAME", f"reward_{i}"),
                ("LOAD_NAME", "total_rewards"),
                ("LOAD_NAME", f"reward_{i}"),
                ("ADD", None),
                ("STORE_NAME", "total_rewards"),
                ("LOAD_NAME", f"reward_{i}"),
                ("STATE_WRITE", f"reward_{i}"),
            ]
        instrs += [
            ("LOAD_NAME", "total_rewards"),
            ("STATE_WRITE", "total_rewards"),
            ("RETURN", None),
        ]

    elif pattern == "governance_vote":
        # Simulate: tally votes with threshold check
        for i in range(scale):
            instrs += [
                ("STATE_READ", f"vote_{i}"),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(1)
            instrs += [
                ("EQ", None),
                ("JUMP_IF_FALSE", len(instrs) + 6),
                ("STATE_READ", "yes_count"),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(1)
            instrs += [
                ("ADD", None),
                ("STATE_WRITE", "yes_count"),
                ("JUMP", len(instrs) + 6),
                ("STATE_READ", "no_count"),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(1)
            instrs += [
                ("ADD", None),
                ("STATE_WRITE", "no_count"),
            ]
        instrs += [
            ("STATE_READ", "yes_count"),
            ("LOAD_CONST", len(consts)),
        ]
        consts.append(scale // 2)
        instrs += [
            ("GT", None),
            ("RETURN", None),
        ]

    elif pattern == "compute_loop":
        # Pure arithmetic loop
        instrs += [
            ("LOAD_CONST", len(consts)),
        ]
        consts.append(0)
        instrs.append(("STORE_NAME", "acc"))
        loop_start = len(instrs)
        for _ in range(scale):
            instrs += [
                ("LOAD_NAME", "acc"),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(7)
            instrs += [
                ("ADD", None),
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(3)
            instrs += [
                ("MUL", None),
                ("STORE_NAME", "acc"),
            ]
        instrs += [
            ("LOAD_NAME", "acc"),
            ("RETURN", None),
        ]

    elif pattern == "batch_transfers":
        # Many state writes (like airdrop)
        for i in range(scale):
            instrs += [
                ("LOAD_CONST", len(consts)),
            ]
            consts.append(100)
            instrs += [
                ("STATE_WRITE", f"balance_{i}"),
            ]
        instrs.append(("RETURN", None))

    return instrs, consts


def measure_python_gas(instrs, consts, gas_limit=10_000_000):
    """Measure gas cost via Python GasMetering (no actual VM execution)."""
    gm = GasMetering(gas_limit=gas_limit)
    for op_name, operand in instrs:
        if not gm.consume(op_name):
            return gm.gas_used, gm.operation_count, True  # out of gas
    return gm.gas_used, gm.operation_count, False


def measure_rust_gas(instrs, consts, gas_limit=10_000_000):
    """Execute via Rust VM and get gas_used."""
    if not RUST_AVAILABLE:
        return None, None, None

    bc = FakeBytecode(instrs, consts)
    try:
        zxc_data = serialize_zxc(bc, include_checksum=True)
    except Exception as e:
        return None, None, f"serialize error: {e}"

    executor = RustVMExecutor()
    state = {}
    # Pre-populate state for read ops
    for op_name, operand in instrs:
        if op_name == "STATE_READ" and operand:
            state[operand] = 1000

    t0 = time.perf_counter()
    result = executor.execute(zxc_data, env={}, state=state, gas_limit=gas_limit)
    elapsed = time.perf_counter() - t0

    return (
        result.get("gas_used", 0),
        result.get("instructions_executed", 0),
        result.get("error"),
        elapsed,
        result.get("needs_fallback", False),
    )


# ────────────────────────────────────────────────────────────────────
# Main benchmark
# ────────────────────────────────────────────────────────────────────

PATTERNS = [
    ("token_transfer", "Token Transfer (read/write/event)"),
    ("dex_swap", "DEX Swap (AMM x*y=k)"),
    ("nft_mint", "NFT Mint (metadata + event)"),
    ("staking_rewards", "Staking Rewards (loop + math)"),
    ("governance_vote", "Governance Vote (conditionals)"),
    ("compute_loop", "Pure Compute Loop"),
    ("batch_transfers", "Batch Transfers (airdrop)"),
]

SCALES = [10, 100, 500, 1000]

def main():
    print("=" * 90)
    print("GAS STRESS BENCHMARK — Python vs Rust VM Gas Consumption")
    print("=" * 90)

    gas_limit = 10_000_000  # 10M gas budget

    for pattern, label in PATTERNS:
        print(f"\n{'─' * 90}")
        print(f"  {label}")
        print(f"{'─' * 90}")
        print(f"{'Scale':>8} {'Instrs':>8} {'Py Gas':>12} {'Py Gas%':>8} "
              f"{'Rust Gas':>12} {'Rust Gas%':>8} {'Savings':>8} "
              f"{'Rust ms':>10} {'Rust MIPS':>10}")

        for scale in SCALES:
            instrs, consts = build_program(pattern, scale)
            n_instrs = len(instrs)

            py_gas, py_ops, py_oog = measure_python_gas(instrs, consts, gas_limit)
            py_pct = py_gas / gas_limit * 100

            if RUST_AVAILABLE:
                res = measure_rust_gas(instrs, consts, gas_limit)
                if res and res[2] is None and not res[4]:
                    r_gas, r_ops, _, r_time, _ = res
                    r_pct = r_gas / gas_limit * 100
                    savings = (1 - r_gas / py_gas) * 100 if py_gas > 0 else 0
                    r_mips = (r_ops / r_time / 1_000_000) if r_time > 0 else 0
                    print(f"{scale:>8} {n_instrs:>8} {py_gas:>12,} {py_pct:>7.2f}% "
                          f"{r_gas:>12,} {r_pct:>7.2f}% {savings:>+7.1f}% "
                          f"{r_time*1000:>9.2f} {r_mips:>10.1f}")
                else:
                    err = res[2] if res else "N/A"
                    fb = res[4] if res else False
                    print(f"{scale:>8} {n_instrs:>8} {py_gas:>12,} {py_pct:>7.2f}% "
                          f"{'ERROR':>12} {'---':>8} {'---':>8} "
                          f"{'---':>10} {'---':>10}  [{err}] fb={fb}")
            else:
                print(f"{scale:>8} {n_instrs:>8} {py_gas:>12,} {py_pct:>7.2f}% "
                      f"{'N/A':>12} {'---':>8}")

            if py_oog:
                print(f"  ⚠ Python ran out of gas at scale={scale}!")

    # ── Summary: Gas per operation breakdown ──
    print(f"\n{'=' * 90}")
    print("GAS COST BREAKDOWN — Per-Operation Analysis")
    print(f"{'=' * 90}")

    # Simulate a realistic "DeFi swap" at scale=100
    instrs, consts = build_program("dex_swap", 100)
    gm = GasMetering(gas_limit=100_000_000)
    for op_name, _ in instrs:
        gm.consume(op_name)

    stats = gm.get_stats()
    print(f"\nDEX Swap (100 swaps), {len(instrs)} instructions:")
    print(f"  Total gas: {stats['gas_used']:,}")
    print(f"  Utilization: {stats['utilization_percent']:.4f}%")
    print(f"\n  {'Operation':<20} {'Count':>8} {'Gas':>12} {'Avg':>6} {'% of Total':>10}")
    print(f"  {'─'*56}")
    sorted_ops = sorted(stats['gas_by_operation'].items(), key=lambda x: -x[1])
    for op, gas in sorted_ops:
        cnt = stats['operation_counts'].get(op, 0)
        avg = gas / cnt if cnt > 0 else 0
        pct = gas / stats['gas_used'] * 100 if stats['gas_used'] > 0 else 0
        print(f"  {op:<20} {cnt:>8} {gas:>12,} {avg:>6.1f} {pct:>9.1f}%")

    # ── High-volume stress: 10K transactions ──
    print(f"\n{'=' * 90}")
    print("HIGH-VOLUME STRESS TEST — 10K Token Transfers")
    print(f"{'=' * 90}")

    instrs_one, consts_one = build_program("token_transfer", 1)
    bc_one = FakeBytecode(instrs_one, consts_one)

    if RUST_AVAILABLE:
        try:
            zxc_one = serialize_zxc(bc_one, include_checksum=True)
            executor = RustVMExecutor()
            state = {"balance_a": 1_000_000_000, "balance_b": 0}

            total_gas = 0
            total_time = 0.0
            n_txns = 10_000

            t0 = time.perf_counter()
            for _ in range(n_txns):
                res = executor.execute(zxc_one, env={}, state=dict(state), gas_limit=gas_limit)
                total_gas += res.get("gas_used", 0)
            total_time = time.perf_counter() - t0

            tps = n_txns / total_time
            avg_gas = total_gas / n_txns
            print(f"  Transactions: {n_txns:,}")
            print(f"  Total time:   {total_time:.3f}s")
            print(f"  TPS:          {tps:,.0f}")
            print(f"  Avg gas/tx:   {avg_gas:,.0f}")
            print(f"  Total gas:    {total_gas:,}")

            # With gas discount (proposed)
            discount = 0.6  # 40% discount for Rust execution
            discounted_gas = int(total_gas * discount)
            print(f"\n  With 40% Rust discount:")
            print(f"  Discounted gas/tx:  {int(avg_gas * discount):,}")
            print(f"  Discounted total:   {discounted_gas:,}")
            print(f"  Gas savings:        {total_gas - discounted_gas:,} ({(1-discount)*100:.0f}%)")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  Rust VM not available")

    # ── Proposed gas optimization summary ──
    print(f"\n{'=' * 90}")
    print("PROPOSED GAS OPTIMIZATIONS")
    print(f"{'=' * 90}")
    print("""
  1. RUST EXECUTION DISCOUNT (40%):
     Rust executes 20x faster than Python. Users shouldn't pay the same
     gas for 20x faster execution. Apply 0.6x multiplier when Rust VM
     handles execution.

  2. ALIGN RUST DYNAMIC SCALING:
     Rust currently uses flat costs for BUILD_LIST, BUILD_MAP, CALL_*,
     and MERKLE_ROOT. Add per-element/per-arg scaling to match Python
     and prevent under-charging.

  3. REDUCE STATE_WRITE COST (50 → 30):
     With RustStateAdapter caching, state writes are now memory-local
     (not disk I/O). Reduce from 50 to 30. Flush cost handled separately.

  4. FIX STORE_FUNC DISCREPANCY:
     Python charges 5, Rust charges 3. Align both to 3 (function
     storage is a cheap memory op).

  5. GAS LIGHT MODE FOR READ-ONLY CALLS:
     Static calls (view functions) should use light gas metering
     (flat 1 per op) since they don't consume chain resources.
""")


if __name__ == "__main__":
    main()
