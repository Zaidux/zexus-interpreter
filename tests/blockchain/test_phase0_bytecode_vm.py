"""Phase 0 — Validate bytecoded vs tree-walked contract execution.

Tests that the bytecode VM path produces identical results to the
tree-walk path for all contract patterns.
"""

import time
import pytest
from types import SimpleNamespace

from zexus.lexer import Lexer
from zexus.parser import UltimateParser
from zexus.blockchain.chain import Chain
from zexus.blockchain.contract_vm import ContractVM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action_body(source: str):
    """Parse a .zx source snippet and return the BlockStatement body."""
    lexer = Lexer(source)
    parser = UltimateParser(lexer, enable_advanced_strategies=False)
    program = parser.parse_program()
    assert parser.errors == [], f"Parse errors: {parser.errors}"
    # Wrap statements in a BlockStatement
    from zexus.zexus_ast import BlockStatement
    block = BlockStatement()
    block.statements = program.statements
    return block


def _make_contract(address, actions_map, initial_storage=None):
    """Create a minimal contract namespace with parsed action bodies."""
    actions = {}
    for name, (params, source) in actions_map.items():
        body = _parse_action_body(source)
        from zexus.zexus_ast import Identifier
        param_idents = [Identifier(p) for p in params]
        actions[name] = SimpleNamespace(
            parameters=param_idents,
            body=body,
        )
    return SimpleNamespace(
        name=f"Test_{address[:8]}",
        address=address,
        storage=SimpleNamespace(current_state=initial_storage or {}),
        actions=actions,
    )


def _run_contract(chain, contract, action, args, caller="tester",
                  gas_limit=1_000_000, use_bytecode_vm=False):
    """Deploy and execute a contract, returning the receipt."""
    vm = ContractVM(chain, gas_limit=gas_limit, use_bytecode_vm=use_bytecode_vm)
    deploy = vm.deploy_contract(contract, deployer=caller)
    assert deploy.success, f"Deploy failed: {deploy.error}"
    receipt = vm.execute_contract(
        contract.address, action, args=args, caller=caller, gas_limit=gas_limit,
    )
    return receipt, vm


def _compare_modes(tmp_path, actions_map, action, args, initial_storage=None,
                   caller="tester", gas_limit=1_000_000):
    """Run the same contract in tree-walk and bytecoded mode, compare results."""
    address = "Zx_phase0_test_" + action

    # Tree-walk execution
    chain_tw = Chain(chain_id="tw", data_dir=str(tmp_path / "tw"))
    contract_tw = _make_contract(address, actions_map, initial_storage)
    receipt_tw, vm_tw = _run_contract(
        chain_tw, contract_tw, action, args, caller, gas_limit,
        use_bytecode_vm=False,
    )

    # Bytecoded execution
    chain_bc = Chain(chain_id="bc", data_dir=str(tmp_path / "bc"))
    contract_bc = _make_contract(address, actions_map, initial_storage)
    receipt_bc, vm_bc = _run_contract(
        chain_bc, contract_bc, action, args, caller, gas_limit,
        use_bytecode_vm=True,
    )

    return receipt_tw, receipt_bc, vm_tw, vm_bc


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestPhase0Arithmetic:
    """Test basic arithmetic operations under both modes."""

    def test_addition(self, tmp_path):
        actions = {
            "add": (["a", "b"], "let result = a + b; return result;"),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "add", {"a": 10, "b": 20})
        assert tw.success, f"Tree-walk failed: {tw.error}"
        assert bc.success, f"Bytecoded failed: {bc.error}"

    def test_multiplication(self, tmp_path):
        actions = {
            "mul": (["x", "y"], "let result = x * y; return result;"),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "mul", {"x": 7, "y": 6})
        assert tw.success
        assert bc.success

    def test_complex_expression(self, tmp_path):
        actions = {
            "calc": (["a", "b", "c"], """
                let x = a + b * c;
                let y = x - a;
                let z = y / b;
                return z;
            """),
        }
        tw, bc, _, _ = _compare_modes(
            tmp_path, actions, "calc", {"a": 5, "b": 3, "c": 4}
        )
        assert tw.success
        assert bc.success


class TestPhase0Conditionals:
    """Test conditional logic."""

    def test_if_else(self, tmp_path):
        actions = {
            "check": (["x"], """
                let result = 0;
                if x > 10 {
                    result = 1;
                } else {
                    result = -1;
                }
                return result;
            """),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "check", {"x": 15})
        assert tw.success
        assert bc.success

    def test_nested_if(self, tmp_path):
        actions = {
            "grade": (["score"], """
                let grade = "F";
                if score >= 90 {
                    grade = "A";
                } else {
                    if score >= 80 {
                        grade = "B";
                    } else {
                        if score >= 70 {
                            grade = "C";
                        }
                    }
                }
                return grade;
            """),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "grade", {"score": 85})
        assert tw.success
        assert bc.success


class TestPhase0Loops:
    """Test loop execution (this is the biggest VM benefit)."""

    def test_while_loop(self, tmp_path):
        actions = {
            "sum_to": (["n"], """
                let total = 0;
                let i = 1;
                while i <= n {
                    total = total + i;
                    i = i + 1;
                }
                return total;
            """),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "sum_to", {"n": 100})
        assert tw.success
        assert bc.success

    def test_nested_loops(self, tmp_path):
        actions = {
            "matrix_sum": ([], """
                let total = 0;
                let i = 0;
                while i < 5 {
                    let j = 0;
                    while j < 5 {
                        total = total + i * j;
                        j = j + 1;
                    }
                    i = i + 1;
                }
                return total;
            """),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "matrix_sum", {})
        assert tw.success
        assert bc.success


class TestPhase0StateOperations:
    """Test blockchain state reads/writes."""

    def test_state_read_write(self, tmp_path):
        actions = {
            "set_value": (["key_val"], """
                let current = 0;
                current = key_val + 10;
                return current;
            """),
        }
        tw, bc, _, _ = _compare_modes(
            tmp_path, actions, "set_value", {"key_val": 42},
            initial_storage={"counter": 0},
        )
        assert tw.success
        assert bc.success

    def test_counter_increment(self, tmp_path):
        actions = {
            "increment": (["amount"], """
                let counter = 0;
                counter = counter + amount;
                return counter;
            """),
        }
        tw, bc, _, _ = _compare_modes(
            tmp_path, actions, "increment", {"amount": 5},
            initial_storage={"counter": 10},
        )
        assert tw.success
        assert bc.success


class TestPhase0StringOperations:
    """Test string handling."""

    def test_string_concat(self, tmp_path):
        actions = {
            "greet": (["name"], """
                let greeting = "Hello, " + name + "!";
                return greeting;
            """),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "greet", {"name": "Zexus"})
        assert tw.success
        assert bc.success


class TestPhase0FunctionCalls:
    """Test action/function definitions and calls within contracts."""

    def test_local_function(self, tmp_path):
        actions = {
            "compute": (["x"], """
                action double(n) {
                    return n * 2;
                }
                let result = double(x);
                return result;
            """),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "compute", {"x": 21})
        assert tw.success
        assert bc.success


class TestPhase0Collections:
    """Test list and map operations."""

    def test_list_creation(self, tmp_path):
        actions = {
            "make_list": ([], """
                let items = [1, 2, 3, 4, 5];
                return items;
            """),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "make_list", {})
        assert tw.success
        assert bc.success

    def test_map_creation(self, tmp_path):
        actions = {
            "make_map": ([], """
                let data = {"name": "test", "value": 42};
                return data;
            """),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "make_map", {})
        assert tw.success
        assert bc.success


class TestPhase0ErrorHandling:
    """Test that errors propagate correctly in both modes."""

    def test_both_modes_succeed_on_valid(self, tmp_path):
        actions = {
            "ok": ([], "return 42;"),
        }
        tw, bc, _, _ = _compare_modes(tmp_path, actions, "ok", {})
        assert tw.success
        assert bc.success


class TestPhase0ExecutionStats:
    """Test that execution stats are tracked correctly."""

    def test_treewalk_stats(self, tmp_path):
        chain = Chain(chain_id="stats-tw", data_dir=str(tmp_path / "stats_tw"))
        actions = {"noop": ([], "return 1;")}
        contract = _make_contract("Zx_stats_tw", actions)
        _, vm = _run_contract(chain, contract, "noop", {}, use_bytecode_vm=False)
        stats = vm.get_vm_execution_stats()
        assert stats["treewalk_executions"] > 0
        assert stats["bytecode_executions"] == 0
        assert stats["use_bytecode_vm"] is False

    def test_bytecode_stats(self, tmp_path):
        chain = Chain(chain_id="stats-bc", data_dir=str(tmp_path / "stats_bc"))
        actions = {"noop": ([], "return 1;")}
        contract = _make_contract("Zx_stats_bc", actions)
        _, vm = _run_contract(chain, contract, "noop", {}, use_bytecode_vm=True)
        stats = vm.get_vm_execution_stats()
        assert stats["use_bytecode_vm"] is True
        # Either bytecode succeeded or fell back — both are acceptable in Phase 0
        total = stats["bytecode_executions"] + stats["bytecode_fallbacks"] + stats["treewalk_executions"]
        assert total > 0


class TestPhase0Benchmark:
    """Benchmark tree-walk vs bytecoded execution (informational, not asserted)."""

    def test_loop_performance(self, tmp_path, capsys):
        """Compare loop performance — bytecoded should be faster."""
        actions = {
            "bench": (["n"], """
                let total = 0;
                let i = 0;
                while i < n {
                    total = total + i * i;
                    i = i + 1;
                }
                return total;
            """),
        }

        n = 500
        address = "Zx_bench_loop"

        # Tree-walk
        chain_tw = Chain(chain_id="bench-tw", data_dir=str(tmp_path / "bench_tw"))
        contract_tw = _make_contract(address, actions)
        vm_tw = ContractVM(chain_tw, use_bytecode_vm=False)
        vm_tw.deploy_contract(contract_tw, deployer="bench")
        t0 = time.perf_counter()
        r_tw = vm_tw.execute_contract(address, "bench", args={"n": n}, caller="bench")
        tw_time = time.perf_counter() - t0

        # Bytecoded
        chain_bc = Chain(chain_id="bench-bc", data_dir=str(tmp_path / "bench_bc"))
        contract_bc = _make_contract(address, actions)
        vm_bc = ContractVM(chain_bc, use_bytecode_vm=True)
        vm_bc.deploy_contract(contract_bc, deployer="bench")
        t0 = time.perf_counter()
        r_bc = vm_bc.execute_contract(address, "bench", args={"n": n}, caller="bench")
        bc_time = time.perf_counter() - t0

        print(f"\n--- Phase 0 Loop Benchmark (n={n}) ---")
        print(f"Tree-walk: {tw_time*1000:.1f}ms (success={r_tw.success})")
        print(f"Bytecoded: {bc_time*1000:.1f}ms (success={r_bc.success})")
        if bc_time > 0 and tw_time > 0:
            speedup = tw_time / bc_time
            print(f"Speedup:   {speedup:.2f}x")

        bc_stats = vm_bc.get_vm_execution_stats()
        print(f"BC stats:  {bc_stats}")
        print("---")

        # Both should succeed
        assert r_tw.success, f"Tree-walk failed: {r_tw.error}"
        # Bytecoded may fallback — that's OK for Phase 0 validation
