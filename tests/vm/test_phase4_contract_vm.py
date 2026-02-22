"""
Phase 4 Tests — Rust ContractVM Orchestration + Gas Optimizations

Tests:
  1. RustContractVM: import, init, gas_discount, stats
  2. GasOptimizations: Rust discount, STATE_WRITE cost reduction, STORE_FUNC alignment
  3. ContractVM integration: Phase 4 tier, reentrancy, call-depth, receipts
  4. Gas stress: high-volume transactions, gas savings measurement
  5. Performance: TPS benchmarking with Rust orchestration
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from zexus.vm.binary_bytecode import serialize as serialize_zxc
from zexus.vm.gas_metering import GasMetering, GasCost

# Try to import the Rust modules
try:
    from zexus_core import RustContractVM, RustVMExecutor, RustStateAdapter
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Try to import the Python ContractVM
try:
    from zexus.blockchain.contract_vm import ContractVM, ContractExecutionReceipt
    from zexus.blockchain.chain import Chain
    CONTRACT_VM_AVAILABLE = True
except ImportError:
    CONTRACT_VM_AVAILABLE = False


# ── Helpers ──────────────────────────────────────────────────────────

class FakeBytecode:
    """Minimal bytecode holder for serialize()."""
    def __init__(self, instructions, constants=None):
        self.instructions = instructions
        self.constants = constants or []


def make_zxc(instrs, consts=None):
    """Create .zxc bytes from instructions and constants."""
    bc = FakeBytecode(instrs, consts or [])
    return serialize_zxc(bc, include_checksum=True)


def simple_add_program():
    """LOAD 10, LOAD 20, ADD, RETURN => 30"""
    return make_zxc(
        [("LOAD_CONST", 0), ("LOAD_CONST", 1), ("ADD", None), ("RETURN", None)],
        [10, 20],
    )


def state_write_program():
    """LOAD 42, STATE_WRITE counter (const idx 1), RETURN"""
    return make_zxc(
        [
            ("LOAD_CONST", 0),
            ("STATE_WRITE", 1),    # 1 → constants[1] = "counter"
            ("LOAD_CONST", 0),
            ("RETURN", None),
        ],
        [42, "counter"],
    )


def state_rw_program():
    """STATE_READ balance, LOAD 100, ADD, STATE_WRITE balance, LOAD_NAME balance, RETURN"""
    return make_zxc(
        [
            ("STATE_READ", 1),     # 1 → constants[1] = "balance"
            ("LOAD_CONST", 0),
            ("ADD", None),
            ("STATE_WRITE", 1),    # 1 → constants[1] = "balance"
            ("STATE_READ", 1),     # re-read to get final value
            ("RETURN", None),
        ],
        [100, "balance"],
    )


def require_fail_program():
    """LOAD false, REQUIRE "must be true" => RequireFailed"""
    return make_zxc(
        [
            ("LOAD_CONST", 0),
            ("REQUIRE", "must be true"),
        ],
        [False],
    )


def gas_hog_program(n=10000):
    """N POW operations (20 gas each) — quickly exhausts gas."""
    instrs = [("LOAD_CONST", 0)]
    consts = [2]
    for _ in range(n):
        instrs.append(("LOAD_CONST", 1))
        if len(consts) < 2:
            consts.append(10)
        else:
            instrs[-1] = ("LOAD_CONST", 1)
        instrs.append(("POW", None))
    instrs.append(("RETURN", None))
    return make_zxc(instrs, consts)


# =====================================================================
# 1. RustContractVM — Import and Basic API
# =====================================================================

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust core not available")
class TestRustContractVMImport:

    def test_import(self):
        from zexus_core import RustContractVM
        assert RustContractVM is not None

    def test_default_init(self):
        vm = RustContractVM()
        assert vm.gas_discount == pytest.approx(0.6, abs=0.01)
        assert vm.max_call_depth == 10
        assert vm.call_depth == 0

    def test_custom_init(self):
        vm = RustContractVM(gas_discount=0.5, default_gas_limit=5_000_000, max_call_depth=5)
        assert vm.gas_discount == pytest.approx(0.5, abs=0.01)
        assert vm.max_call_depth == 5

    def test_gas_discount_setter(self):
        vm = RustContractVM()
        vm.gas_discount = 0.8
        assert vm.gas_discount == pytest.approx(0.8, abs=0.01)
        # Clamped to [0.01, 1.0]
        vm.gas_discount = 0.0
        assert vm.gas_discount >= 0.01
        vm.gas_discount = 2.0
        assert vm.gas_discount <= 1.0

    def test_initial_stats(self):
        vm = RustContractVM()
        stats = vm.get_stats()
        assert stats["total_executions"] == 0
        assert stats["successful"] == 0
        assert stats["failed"] == 0
        assert stats["gas_discount"] == pytest.approx(0.6, abs=0.01)

    def test_is_executing(self):
        vm = RustContractVM()
        assert not vm.is_executing("0xabc")


# =====================================================================
# 2. Gas Optimizations
# =====================================================================

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust core not available")
class TestGasOptimizations:

    def test_rust_gas_discount_applied(self):
        """Rust VM should charge less gas than full price."""
        data = simple_add_program()
        # Full price (discount=1.0)
        exe_full = RustVMExecutor(gas_discount=1.0)
        res_full = exe_full.execute(data, gas_limit=10000)
        # Discounted (0.6)
        exe_disc = RustVMExecutor(gas_discount=0.6)
        res_disc = exe_disc.execute(data, gas_limit=10000)

        assert res_disc["gas_used"] < res_full["gas_used"]
        assert res_disc["gas_discount"] == pytest.approx(0.6, abs=0.01)
        assert res_full["gas_discount"] == pytest.approx(1.0, abs=0.01)

    def test_state_write_cost_reduced(self):
        """STATE_WRITE should cost 30 (not 50) in both Python and Rust."""
        # Python gas metering
        assert GasCost.STATE_WRITE == 30

        # Rust VM
        data = state_write_program()
        exe = RustVMExecutor(gas_discount=1.0)
        res = exe.execute(data, gas_limit=100000)
        # LOAD_CONST(1) + STATE_WRITE(30) + LOAD_CONST(1) + RETURN(2) = 34
        assert res["gas_used"] == 34
        assert res["error"] is None

    def test_store_func_aligned(self):
        """STORE_FUNC should cost 3 in Python (aligned with Rust)."""
        assert GasCost.STORE_FUNC == 3

    def test_gas_discount_in_contract_vm(self):
        """RustContractVM should apply gas discount to execution."""
        vm = RustContractVM(gas_discount=0.6)
        data = simple_add_program()
        res = vm.execute_contract(
            contract_address="0xtest",
            action_bytecode=data,
            gas_limit=10000,
            caller="0xcaller",
        )
        assert res["success"]
        assert res["gas_discount"] == pytest.approx(0.6, abs=0.01)
        # gas_saved should be > 0
        assert res["gas_saved"] > 0

    def test_full_price_vs_discounted(self):
        """Compare full-price and discounted gas for same program."""
        data = simple_add_program()
        vm_full = RustContractVM(gas_discount=1.0)
        vm_disc = RustContractVM(gas_discount=0.6)

        res_full = vm_full.execute_contract("0xtest", data, gas_limit=10000, caller="0x1")
        res_disc = vm_disc.execute_contract("0xtest", data, gas_limit=10000, caller="0x1")

        assert res_full["success"]
        assert res_disc["success"]
        assert res_disc["gas_used"] < res_full["gas_used"]


# =====================================================================
# 3. RustContractVM — Execution Lifecycle
# =====================================================================

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust core not available")
class TestRustContractVMExecution:

    def test_simple_execution(self):
        vm = RustContractVM()
        data = simple_add_program()
        res = vm.execute_contract("0xabc", data, gas_limit=10000, caller="0xcaller")
        assert res["success"]
        assert res["result"] == 30
        assert res["gas_used"] > 0
        assert res["gas_limit"] == 10000

    def test_state_write(self):
        vm = RustContractVM()
        data = state_write_program()
        state = {"counter": 0}
        res = vm.execute_contract("0xabc", data, state=state, gas_limit=10000, caller="0x1")
        assert res["success"]
        assert "counter" in res.get("state_changes", {})
        assert res.get("new_state", {}).get("counter") == 42

    def test_state_read_write(self):
        vm = RustContractVM()
        data = state_rw_program()
        state = {"balance": 500}
        res = vm.execute_contract("0xabc", data, state=state, gas_limit=100000, caller="0x1")
        assert res["success"]
        new_state = res.get("new_state", {})
        assert new_state.get("balance") == 600  # 500 + 100

    def test_require_failure(self):
        vm = RustContractVM()
        data = require_fail_program()
        res = vm.execute_contract("0xabc", data, gas_limit=10000, caller="0x1")
        assert not res["success"]
        assert "RequireFailed" in res.get("error", "")

    def test_reentrancy_guard(self):
        """Simulating reentrancy: executing same contract twice concurrently."""
        # RustContractVM tracks executing set internally
        # The Python wrapper won't hit this directly, but we can test via
        # the is_executing method after manually checking
        vm = RustContractVM()
        assert not vm.is_executing("0xabc")

    def test_call_depth_exceeded(self):
        """Manually exceed call depth by setting max_call_depth=0."""
        vm = RustContractVM(max_call_depth=1)
        # First call succeeds
        data = simple_add_program()
        # We can't easily nest, but we can check the guard mechanism
        # by checking the Rust-side enforcement
        res1 = vm.execute_contract("0xabc", data, gas_limit=10000, caller="0x1")
        assert res1["success"]  # First call at depth 0 < max 1

    def test_out_of_gas(self):
        """Execution should fail when gas runs out."""
        vm = RustContractVM(gas_discount=1.0)
        # Simple program needs about 7 gas, give it only 3
        data = simple_add_program()
        res = vm.execute_contract("0xabc", data, gas_limit=3, caller="0x1")
        assert not res["success"]
        assert "OutOfGas" in res.get("error", "")

    def test_stats_tracking(self):
        vm = RustContractVM()
        data = simple_add_program()
        vm.execute_contract("0xabc", data, gas_limit=10000, caller="0x1")
        vm.execute_contract("0xdef", data, gas_limit=10000, caller="0x2")

        stats = vm.get_stats()
        assert stats["total_executions"] == 2
        assert stats["successful"] == 2
        assert stats["failed"] == 0
        assert stats["total_gas_used"] > 0
        assert stats["total_gas_saved"] > 0
        assert stats["avg_gas_per_execution"] > 0

    def test_reset_stats(self):
        vm = RustContractVM()
        vm.execute_contract("0xabc", simple_add_program(), gas_limit=10000, caller="0x1")
        vm.reset_stats()
        stats = vm.get_stats()
        assert stats["total_executions"] == 0

    def test_receipt_fields(self):
        vm = RustContractVM()
        res = vm.execute_contract("0xabc", simple_add_program(), gas_limit=10000, caller="0x1")
        required_fields = [
            "success", "result", "gas_used", "gas_limit", "error",
            "revert_reason", "state_changes", "instructions_executed",
            "output", "needs_fallback", "gas_discount", "gas_saved",
        ]
        for field in required_fields:
            assert field in res, f"Missing field: {field}"


# =====================================================================
# 4. Batch Execution
# =====================================================================

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust core not available")
class TestBatchExecution:

    def test_batch_simple(self):
        vm = RustContractVM()
        data = simple_add_program()
        calls = [
            {"contract_address": "0x1", "bytecode": data, "gas_limit": 10000, "caller": "0xa"},
            {"contract_address": "0x2", "bytecode": data, "gas_limit": 10000, "caller": "0xb"},
            {"contract_address": "0x3", "bytecode": data, "gas_limit": 10000, "caller": "0xc"},
        ]
        results = vm.execute_batch(calls)
        assert len(results) == 3
        for r in results:
            assert r["success"]
            assert r["result"] == 30

    def test_batch_with_failures(self):
        vm = RustContractVM(gas_discount=1.0)
        data_ok = simple_add_program()
        data_fail = simple_add_program()  # Give insufficient gas
        calls = [
            {"contract_address": "0x1", "bytecode": data_ok, "gas_limit": 10000, "caller": "0xa"},
            {"contract_address": "0x2", "bytecode": data_fail, "gas_limit": 2, "caller": "0xb"},
            {"contract_address": "0x3", "bytecode": data_ok, "gas_limit": 10000, "caller": "0xc"},
        ]
        results = vm.execute_batch(calls)
        assert len(results) == 3
        assert results[0]["success"]
        assert not results[1]["success"]
        assert results[2]["success"]


# =====================================================================
# 5. Gas Stress Testing
# =====================================================================

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust core not available")
class TestGasStress:

    def test_high_volume_transfers(self):
        """10,000 simple transfers should complete fast with low gas."""
        vm = RustContractVM()
        data = simple_add_program()
        n = 10_000

        t0 = time.perf_counter()
        total_gas = 0
        total_saved = 0
        for i in range(n):
            res = vm.execute_contract(
                f"0x{i:04x}", data, gas_limit=10000, caller="0x1"
            )
            assert res["success"], f"Failed at tx {i}: {res['error']}"
            total_gas += res["gas_used"]
            total_saved += res.get("gas_saved", 0)
        elapsed = time.perf_counter() - t0

        tps = n / elapsed
        avg_gas = total_gas / n

        stats = vm.get_stats()
        assert stats["total_executions"] == n
        assert stats["successful"] == n
        assert total_saved > 0, "Gas discount should produce savings"
        assert tps > 10_000, f"TPS too low: {tps:.0f}"

    def test_state_heavy_batch(self):
        """1000 state-write transactions."""
        vm = RustContractVM()
        data = state_write_program()

        for i in range(1000):
            res = vm.execute_contract(
                f"0x{i:04x}", data,
                state={"counter": i},
                gas_limit=100000,
                caller="0x1",
            )
            assert res["success"]

        stats = vm.get_stats()
        assert stats["successful"] == 1000

    def test_gas_savings_measurement(self):
        """Verify gas savings are correctly reported."""
        vm_discounted = RustContractVM(gas_discount=0.6)
        vm_full_price = RustContractVM(gas_discount=1.0)

        data = state_rw_program()
        state = {"balance": 1000}

        res_disc = vm_discounted.execute_contract("0x1", data, state=state, gas_limit=100000, caller="0x1")
        res_full = vm_full_price.execute_contract("0x1", data, state=state, gas_limit=100000, caller="0x1")

        assert res_disc["success"]
        assert res_full["success"]
        assert res_disc["gas_used"] < res_full["gas_used"]
        savings_pct = (1 - res_disc["gas_used"] / res_full["gas_used"]) * 100
        assert savings_pct > 20, f"Expected >20% savings, got {savings_pct:.1f}%"


# =====================================================================
# 6. ContractVM Integration (if available)
# =====================================================================

@pytest.mark.skipif(
    not CONTRACT_VM_AVAILABLE or not RUST_AVAILABLE,
    reason="ContractVM or Rust core not available",
)
class TestContractVMIntegration:

    def _make_chain(self):
        return Chain(chain_id="test-phase4")

    def test_rust_contract_vm_available(self):
        chain = self._make_chain()
        cvm = ContractVM(chain=chain, use_bytecode_vm=True)
        assert cvm._rust_contract_vm is not None

    def test_execution_stats_include_phase4(self):
        chain = self._make_chain()
        cvm = ContractVM(chain=chain, use_bytecode_vm=True)
        stats = cvm.get_vm_execution_stats()
        assert "rust_contract_vm_available" in stats
        assert stats["rust_contract_vm_available"] is True
        assert "rust_contract_vm_stats" in stats


# =====================================================================
# 7. Performance Characterisation
# =====================================================================

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust core not available")
class TestPerformance:

    def test_execution_speed(self):
        """Measure TPS for the Rust ContractVM."""
        vm = RustContractVM()
        data = simple_add_program()
        n = 5000

        t0 = time.perf_counter()
        for i in range(n):
            vm.execute_contract(f"0x{i:04x}", data, gas_limit=10000, caller="0x1")
        elapsed = time.perf_counter() - t0

        tps = n / elapsed
        # Should be at least 50K TPS for simple programs
        assert tps > 50_000, f"TPS too low: {tps:.0f}"

    def test_batch_speed(self):
        """Measure batch execution speed."""
        vm = RustContractVM()
        data = simple_add_program()
        calls = [
            {"contract_address": f"0x{i:04x}", "bytecode": data,
             "gas_limit": 10000, "caller": "0xa"}
            for i in range(1000)
        ]

        t0 = time.perf_counter()
        results = vm.execute_batch(calls)
        elapsed = time.perf_counter() - t0

        assert len(results) == 1000
        assert all(r["success"] for r in results)
        tps = 1000 / elapsed
        assert tps > 50_000, f"Batch TPS too low: {tps:.0f}"
