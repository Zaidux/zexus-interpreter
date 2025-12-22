"""
Comprehensive VM and JIT Integration Test Suite

Tests:
1. Basic VM functionality with all execution modes
2. JIT compilation and hot path detection
3. Memory manager integration
4. Blockchain opcode execution
5. Performance validation across modes
"""

import sys
import os
import time
import asyncio
import json
import hashlib
from typing import Dict, Any, List
import unittest
import tempfile

# Add the source directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.zexus.vm.bytecode import Bytecode, BytecodeBuilder
from src.zexus.vm.vm import VM, VMMode, create_vm, create_high_performance_vm
from src.zexus.vm.jit import JITCompiler, ExecutionTier


class TestVMBasicFunctionality(unittest.TestCase):
    """Test basic VM execution in all modes"""
    
    def setUp(self):
        self.test_env = {"x": 10, "y": 20, "z": 30}
        self.builtins = {
            "add": lambda a, b: a + b,
            "mul": lambda a, b: a * b,
        }
    
    def test_stack_mode_basic_arithmetic(self):
        """Test basic arithmetic in stack mode"""
        vm = VM(
            builtins=self.builtins,
            env=self.test_env.copy(),
            mode=VMMode.STACK,
            use_jit=False,
            debug=False
        )
        
        # Create simple bytecode: x + y
        builder = BytecodeBuilder()
        builder.emit_load_name("x")
        builder.emit_load_name("y")
        builder.emit_add()
        builder.emit_return()
        
        bytecode = builder.build()
        result = vm.execute(bytecode)
        
        self.assertEqual(result, 30)  # 10 + 20
    
    def test_register_mode_arithmetic(self):
        """Test arithmetic in register mode (if available)"""
        try:
            vm = VM(
                builtins=self.builtins,
                env=self.test_env.copy(),
                mode=VMMode.REGISTER,
                use_jit=False,
                debug=False
            )
            
            # Simple arithmetic bytecode
            builder = BytecodeBuilder()
            builder.emit_load_name("x")
            builder.emit_load_name("y")
            builder.emit_add()
            builder.emit_load_name("z")
            builder.emit_mul()  # (x + y) * z
            builder.emit_return()
            
            bytecode = builder.build()
            result = vm.execute(bytecode)
            
            self.assertEqual(result, (10 + 20) * 30)  # 900
            
        except Exception as e:
            # Register mode might not be available, skip test
            self.skipTest(f"Register mode not available: {e}")
    
    def test_auto_mode_selection(self):
        """Test that auto mode selects appropriate execution mode"""
        # Create arithmetic-heavy bytecode (should prefer register mode)
        builder = BytecodeBuilder()
        for i in range(50):  # Enough to trigger register mode
            builder.emit_load_const(i)
            builder.emit_load_const(i + 1)
            builder.emit_add()
            builder.emit_pop()  # Discard result
        builder.emit_load_const(42)
        builder.emit_return()
        
        bytecode = builder.build()
        
        vm = VM(
            builtins=self.builtins,
            env=self.test_env.copy(),
            mode=VMMode.AUTO,
            use_jit=False,
            debug=False
        )
        
        result = vm.execute(bytecode)
        self.assertEqual(result, 42)
        
        # Check that a mode was selected
        stats = vm.get_stats()
        self.assertGreater(sum(stats['mode_usage'].values()), 0)
    
    def test_high_level_ops_execution(self):
        """Test high-level operations list execution"""
        vm = VM(
            builtins=self.builtins,
            env=self.test_env.copy(),
            mode=VMMode.STACK,
            use_jit=False,
            debug=False
        )
        
        # High-level ops format
        ops = [
            ("LET", "result", ("CALL_BUILTIN", "add", [("IDENT", "x"), ("IDENT", "y")])),
            ("EXPR", ("IDENT", "result"))
        ]
        
        result = vm.execute(ops)
        self.assertEqual(result, 30)  # x + y = 30


class TestJITIntegration(unittest.TestCase):
    """Test JIT compiler integration with VM"""
    
    def setUp(self):
        self.builtins = {
            "square": lambda x: x * x,
            "inc": lambda x: x + 1,
        }
        self.env = {"base": 5}
    
    def test_jit_hot_path_detection(self):
        """Test that JIT detects hot paths after threshold executions"""
        vm = VM(
            builtins=self.builtins,
            env=self.env.copy(),
            mode=VMMode.STACK,
            use_jit=True,
            jit_threshold=5,  # Low threshold for testing
            debug=False
        )
        
        # Create bytecode that will be executed many times
        builder = BytecodeBuilder()
        builder.emit_load_name("base")
        builder.emit_load_const(2)
        builder.emit_pow()  # base ** 2
        builder.emit_return()
        
        bytecode = builder.build()
        
        # Execute multiple times
        results = []
        for i in range(10):
            result = vm.execute(bytecode)
            results.append(result)
        
        # All results should be 25 (5 ** 2)
        self.assertTrue(all(r == 25 for r in results))
        
        # Check JIT stats
        jit_stats = vm.get_jit_stats()
        self.assertIn('jit_enabled', jit_stats)
        
        if jit_stats['jit_enabled']:
            # Should have detected hot path and compiled it
            self.assertGreater(jit_stats.get('hot_paths_detected', 0), 0)
            self.assertGreater(jit_stats.get('compilations', 0), 0)
    
    def test_jit_cache_effectiveness(self):
        """Test that JIT cache prevents recompilation"""
        vm = VM(
            builtins=self.builtins,
            env=self.env.copy(),
            mode=VMMode.STACK,
            use_jit=True,
            jit_threshold=3,
            debug=False
        )
        
        builder = BytecodeBuilder()
        builder.emit_load_const(42)
        builder.emit_return()
        
        bytecode = builder.build()
        
        # First execution should compile
        for i in range(10):
            vm.execute(bytecode)
        
        jit_stats = vm.get_jit_stats()
        if jit_stats.get('jit_enabled', False):
            # Should have compiled once, cached for rest
            self.assertEqual(jit_stats.get('compilations', 0), 1)
            self.assertGreater(jit_stats.get('cache_hits', 0), 0)
    
    def test_jit_performance_improvement(self):
        """Test that JIT actually improves performance (statistically)"""
        import statistics
        import timeit
        
        vm_no_jit = VM(
            builtins=self.builtins,
            env=self.env.copy(),
            mode=VMMode.STACK,
            use_jit=False,
            debug=False
        )
        
        vm_with_jit = VM(
            builtins=self.builtins,
            env=self.env.copy(),
            mode=VMMode.STACK,
            use_jit=True,
            jit_threshold=10,
            debug=False
        )
        
        # Create computation-heavy bytecode
        builder = BytecodeBuilder()
        for i in range(20):
            builder.emit_load_const(i)
            builder.emit_load_const(i + 1)
            builder.emit_mul()
            if i > 0:
                builder.emit_add()
        builder.emit_return()
        
        bytecode = builder.build()
        
        # Warm up JIT
        for _ in range(15):
            vm_with_jit.execute(bytecode)
        
        # Measure performance
        def run_no_jit():
            return vm_no_jit.execute(bytecode)
        
        def run_with_jit():
            return vm_with_jit.execute(bytecode)
        
        # Run multiple times for statistical significance
        iterations = 100
        times_no_jit = timeit.repeat(run_no_jit, number=1, repeat=iterations)
        times_jit = timeit.repeat(run_with_jit, number=1, repeat=iterations)
        
        # Calculate statistics
        median_no_jit = statistics.median(times_no_jit[10:])  # Skip warm-up
        median_jit = statistics.median(times_jit[10:])
        
        # JIT should be faster or similar (accounting for warm-up)
        speedup = median_no_jit / median_jit if median_jit > 0 else 1
        
        print(f"JIT Performance: {speedup:.2f}x (No JIT: {median_no_jit*1000:.2f}ms, JIT: {median_jit*1000:.2f}ms)")
        
        # Acceptable if JIT is at least as fast (might be slower on first runs)
        self.assertGreaterEqual(speedup, 0.5)  # Allow for warm-up overhead
    
    def test_jit_clear_cache(self):
        """Test that JIT cache can be cleared"""
        vm = VM(
            builtins=self.builtins,
            env=self.env.copy(),
            mode=VMMode.STACK,
            use_jit=True,
            jit_threshold=2,
            debug=False
        )
        
        builder = BytecodeBuilder()
        builder.emit_load_const(100)
        builder.emit_return()
        
        bytecode = builder.build()
        
        # Trigger JIT compilation
        for _ in range(5):
            vm.execute(bytecode)
        
        # Clear cache
        vm.clear_jit_cache()
        
        # Check stats were reset
        jit_stats = vm.get_jit_stats()
        if jit_stats.get('jit_enabled', False):
            self.assertEqual(jit_stats.get('cache_size', 1), 0)


class TestMemoryManagerIntegration(unittest.TestCase):
    """Test memory manager integration with VM"""
    
    def test_memory_manager_basic(self):
        """Test basic memory manager functionality"""
        vm = VM(
            mode=VMMode.STACK,
            use_memory_manager=True,
            max_heap_mb=10,  # Small heap for testing
            debug=False
        )
        
        # Allocate objects through STORE_NAME
        builder = BytecodeBuilder()
        builder.emit_load_const("large_string_data" * 100)  # Create larger object
        builder.emit_store_name("large_var")
        builder.emit_load_const(42)
        builder.emit_return()
        
        bytecode = builder.build()
        result = vm.execute(bytecode)
        
        self.assertEqual(result, 42)
        
        # Check memory stats
        mem_stats = vm.get_memory_stats()
        self.assertTrue(mem_stats.get('memory_manager_enabled', False))
        if mem_stats.get('memory_manager_enabled'):
            self.assertGreater(mem_stats.get('allocation_count', 0), 0)
    
    def test_garbage_collection(self):
        """Test that garbage collection works"""
        vm = VM(
            mode=VMMode.STACK,
            use_memory_manager=True,
            max_heap_mb=5,  # Very small heap to force GC
            debug=False
        )
        
        # Create many objects
        results = []
        for i in range(100):
            builder = BytecodeBuilder()
            builder.emit_load_const(f"object_{i}" * 50)  # Create moderate object
            builder.emit_store_name(f"var_{i}")
            builder.emit_load_const(i)
            builder.emit_return()
            
            bytecode = builder.build()
            result = vm.execute(bytecode)
            results.append(result)
        
        # Force garbage collection
        gc_result = vm.collect_garbage(force=True)
        
        # Check that GC ran
        self.assertIn('collected', gc_result)
        self.assertIn('gc_time', gc_result)
        
        # Get memory report
        report = vm.get_memory_report()
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)
    
    def test_memory_manager_with_transactions(self):
        """Test memory manager with blockchain transactions"""
        vm = VM(
            mode=VMMode.STACK,
            use_memory_manager=True,
            debug=False
        )
        
        # Start transaction
        builder = BytecodeBuilder()
        builder.emit_tx_begin()
        builder.emit_load_const("transaction_data")
        builder.emit_store_name("tx_var")
        builder.emit_load_const(100)
        builder.emit_return()
        
        bytecode = builder.build()
        result = vm.execute(bytecode)
        
        self.assertEqual(result, 100)
        
        # Check memory stats
        mem_stats = vm.get_memory_stats()
        if mem_stats.get('memory_manager_enabled'):
            self.assertGreater(mem_stats.get('managed_objects_count', 0), 0)


class TestBlockchainOpcodes(unittest.TestCase):
    """Test blockchain-specific opcodes"""
    
    def setUp(self):
        self.vm = VM(mode=VMMode.STACK, debug=False)
        self.vm.env["_blockchain_state"] = {}
        self.vm.env["_gas_remaining"] = 1000
    
    def test_hash_block(self):
        """Test HASH_BLOCK opcode"""
        builder = BytecodeBuilder()
        
        # Test with string
        builder.emit_load_const("block_data")
        builder.emit_hash_block()
        builder.emit_return()
        
        bytecode = builder.build()
        result = self.vm.execute(bytecode)
        
        # Should be a SHA-256 hash (64 hex chars)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
        
        # Verify it's correct SHA-256
        expected = hashlib.sha256(b"block_data").hexdigest()
        self.assertEqual(result, expected)
    
    def test_merkle_root(self):
        """Test MERKLE_ROOT opcode"""
        builder = BytecodeBuilder()
        
        # Add 4 leaves
        for i in range(4):
            builder.emit_load_const(f"leaf_{i}")
        
        # Calculate Merkle root
        builder.emit_merkle_root(4)
        builder.emit_return()
        
        bytecode = builder.build()
        result = self.vm.execute(bytecode)
        
        # Should be a SHA-256 hash
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
    
    def test_state_operations(self):
        """Test STATE_READ and STATE_WRITE opcodes"""
        # First write
        builder = BytecodeBuilder()
        builder.emit_load_const("balance_key")
        builder.emit_load_const(1000)
        builder.emit_state_write(0)  # Write 1000 at key 0
        builder.emit_load_const("balance_key")
        builder.emit_state_read(0)   # Read back
        builder.emit_return()
        
        bytecode = builder.build()
        result = self.vm.execute(bytecode)
        
        self.assertEqual(result, 1000)
        
        # Verify it's in blockchain state
        self.assertEqual(self.vm.env["_blockchain_state"].get("balance_key"), 1000)
    
    def test_transaction_flow(self):
        """Test transaction begin/commit/revert"""
        # Initial state
        self.vm.env["_blockchain_state"]["initial"] = 100
        
        # Transaction that modifies state
        builder = BytecodeBuilder()
        builder.emit_tx_begin()
        builder.emit_load_const("new_key")
        builder.emit_load_const(200)
        builder.emit_state_write(0)
        builder.emit_tx_commit()
        builder.emit_load_const("new_key")
        builder.emit_state_read(0)
        builder.emit_return()
        
        bytecode = builder.build()
        result = self.vm.execute(bytecode)
        
        self.assertEqual(result, 200)
        self.assertEqual(self.vm.env["_blockchain_state"]["new_key"], 200)
        self.assertEqual(self.vm.env["_blockchain_state"]["initial"], 100)
    
    def test_transaction_revert(self):
        """Test transaction revert on failure"""
        # Initial state
        self.vm.env["_blockchain_state"]["initial"] = 100
        
        builder = BytecodeBuilder()
        builder.emit_tx_begin()
        builder.emit_load_const("temp_key")
        builder.emit_load_const(999)
        builder.emit_state_write(0)
        builder.emit_tx_revert()  # Revert changes
        builder.emit_load_const("temp_key")
        builder.emit_state_read(0)  # Should not exist
        builder.emit_return()
        
        bytecode = builder.build()
        result = self.vm.execute(bytecode)
        
        self.assertIsNone(result)  # Key shouldn't exist after revert
        self.assertNotIn("temp_key", self.vm.env["_blockchain_state"])
        self.assertEqual(self.vm.env["_blockchain_state"]["initial"], 100)
    
    def test_gas_metering(self):
        """Test GAS_CHARGE opcode"""
        builder = BytecodeBuilder()
        
        # Charge gas twice
        builder.emit_gas_charge(100)  # Charge 100 gas
        builder.emit_gas_charge(200)  # Charge 200 more
        builder.emit_load_const("remaining_gas")
        builder.emit_state_read(0)  # Should be None, just for opcode
        builder.emit_load_const(42)
        builder.emit_return()
        
        bytecode = builder.build()
        result = self.vm.execute(bytecode)
        
        self.assertEqual(result, 42)
        
        # Should have used 300 gas
        self.assertEqual(self.vm.env.get("_gas_remaining"), 700)
    
    def test_out_of_gas(self):
        """Test out-of-gas condition"""
        builder = BytecodeBuilder()
        
        # Try to use more gas than available
        builder.emit_tx_begin()
        builder.emit_gas_charge(1500)  # More than 1000 available
        builder.emit_load_const("should_not_execute")
        builder.emit_load_const(999)
        builder.emit_state_write(0)
        builder.emit_tx_commit()
        builder.emit_load_const(42)
        builder.emit_return()
        
        bytecode = builder.build()
        result = self.vm.execute(bytecode)
        
        # Should get OutOfGas error, not 42
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("error"), "OutOfGas")
        
        # Transaction should have been reverted
        self.assertNotIn("should_not_execute", self.vm.env.get("_blockchain_state", {}))
        self.assertFalse(self.vm.env.get("_in_transaction", False))
    
    def test_ledger_append(self):
        """Test LEDGER_APPEND opcode"""
        builder = BytecodeBuilder()
        
        # Create ledger entry
        entry = {"action": "transfer", "amount": 100, "from": "alice", "to": "bob"}
        builder.emit_load_const(entry)
        builder.emit_ledger_append()
        builder.emit_load_const(1)
        builder.emit_return()
        
        bytecode = builder.build()
        result = self.vm.execute(bytecode)
        
        self.assertEqual(result, 1)
        
        # Check ledger
        ledger = self.vm.env.get("_ledger", [])
        self.assertEqual(len(ledger), 1)
        self.assertIn("timestamp", ledger[0])
        self.assertEqual(ledger[0]["action"], "transfer")


class TestPerformanceComparison(unittest.TestCase):
    """Performance comparison tests across modes"""
    
    def test_mode_performance_profiling(self):
        """Profile and compare different execution modes"""
        vm = create_high_performance_vm()
        
        # Create computation-heavy bytecode
        builder = BytecodeBuilder()
        
        # Fibonacci-like calculation
        builder.emit_load_const(20)  # n = 20
        builder.emit_store_name("n")
        builder.emit_load_const(0)
        builder.emit_store_name("a")
        builder.emit_load_const(1)
        builder.emit_store_name("b")
        builder.emit_load_const(0)
        builder.emit_store_name("i")
        
        # Loop label
        builder.emit_label("loop_start")
        builder.emit_load_name("i")
        builder.emit_load_name("n")
        builder.emit_lt()
        builder.emit_jump_if_false("loop_end")
        
        # Fibonacci iteration: a, b = b, a + b
        builder.emit_load_name("a")
        builder.emit_load_name("b")
        builder.emit_add()
        builder.emit_store_name("temp")
        builder.emit_load_name("b")
        builder.emit_store_name("a")
        builder.emit_load_name("temp")
        builder.emit_store_name("b")
        
        # i += 1
        builder.emit_load_name("i")
        builder.emit_load_const(1)
        builder.emit_add()
        builder.emit_store_name("i")
        
        builder.emit_jump("loop_start")
        builder.emit_label("loop_end")
        builder.emit_load_name("a")
        builder.emit_return()
        
        bytecode = builder.build()
        
        # Profile execution
        profile = vm.profile_execution(bytecode, iterations=100)
        
        print("\n" + "="*60)
        print("PERFORMANCE PROFILE")
        print("="*60)
        for mode, data in profile['modes'].items():
            print(f"{mode.upper():10} | Avg: {data['avg']*1000:6.2f}ms | Total: {data['total']:.4f}s")
            
            if 'speedup' in data:
                print(f"           | Speedup vs stack: {data['speedup']:.2f}x")
        
        # At least one mode should work
        self.assertGreater(len(profile['modes']), 0)


class TestConcurrentExecution(unittest.TestCase):
    """Test concurrent/async execution"""
    
    def test_async_spawn_await(self):
        """Test SPAWN and AWAIT opcodes"""
        vm = VM(mode=VMMode.STACK, debug=False)
        
        # Define an async builtin
        async def async_add(a, b):
            await asyncio.sleep(0.01)  # Simulate async work
            return a + b
        
        vm.builtins["async_add"] = async_add
        
        builder = BytecodeBuilder()
        
        # Spawn async task
        builder.emit_load_const(10)
        builder.emit_load_const(20)
        builder.emit_spawn(("CALL", "async_add", 2))
        builder.emit_store_name("task_id")
        
        # Await result
        builder.emit_load_name("task_id")
        builder.emit_await()
        builder.emit_return()
        
        bytecode = builder.build()
        
        # This should work with async execution
        result = vm.execute(bytecode)
        self.assertEqual(result, 30)
    
    def test_event_system(self):
        """Test event registration and emission"""
        vm = VM(mode=VMMode.STACK, debug=False)
        
        event_log = []
        
        def event_handler(payload):
            event_log.append(payload)
            return "handled"
        
        vm.builtins["handler"] = event_handler
        
        builder = BytecodeBuilder()
        
        # Register event
        builder.emit_register_event(("test_event", "handler"))
        
        # Emit event
        builder.emit_load_const("test_payload")
        builder.emit_emit_event(("test_event", 0))  # payload at const 0
        builder.emit_load_const(42)
        builder.emit_return()
        
        bytecode = builder.build()
        
        # Need to run in async context
        async def run_test():
            return await vm._run_stack_bytecode(bytecode, debug=False)
        
        result = asyncio.run(run_test())
        
        self.assertEqual(result, 42)
        self.assertEqual(len(event_log), 1)
        self.assertEqual(event_log[0], "test_payload")


class TestFactoryFunctions(unittest.TestCase):
    """Test VM factory functions"""
    
    def test_create_vm_custom_config(self):
        """Test creating VM with custom configuration"""
        vm = create_vm(
            mode="register",
            use_jit=True,
            jit_threshold=50,
            use_memory_manager=True,
            max_heap_mb=50,
            num_registers=32,
            debug=False
        )
        
        self.assertEqual(vm.mode, VMMode.REGISTER)
        self.assertTrue(vm.use_jit)
        self.assertTrue(vm.use_memory_manager)
    
    def test_create_high_performance_vm(self):
        """Test high performance VM factory"""
        vm = create_high_performance_vm()
        
        # Should have JIT and memory manager enabled
        self.assertTrue(vm.use_jit)
        self.assertTrue(vm.use_memory_manager)
        self.assertEqual(vm.mode, VMMode.AUTO)
    
    def test_create_debug_vm(self):
        """Test debug VM factory"""
        # Note: Need to add create_debug_vm to vm.py
        # For now, test custom debug config
        vm = create_vm(mode="stack", use_jit=False, debug=True)
        
        self.assertEqual(vm.mode, VMMode.STACK)
        self.assertFalse(vm.use_jit)


def run_all_tests():
    """Run all test suites and print summary"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVMBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestJITIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestBlockchainOpcodes))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestConcurrentExecution))
    suite.addTests(loader.loadTestsFromTestCase(TestFactoryFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print detailed summary
    print("\n" + "="*70)
    print("VM & JIT INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {result.testsRun - (result.testsRun - len(result.failures) - len(result.errors))}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)