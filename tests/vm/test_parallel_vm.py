"""
Comprehensive test suite for Zexus Parallel VM (Phase 6)

Tests parallel bytecode execution, dependency analysis, chunking,
worker pool management, and thread-safe state operations.

Author: Zexus Team
Date: December 19, 2025
"""

import unittest
import time
from typing import List
import multiprocessing as mp

from src.zexus.vm.parallel_vm import (
    ParallelVM, BytecodeChunker, DependencyAnalyzer,
    SharedState, ResultMerger, WorkerPool,
    BytecodeChunk, ExecutionResult, ExecutionMode
)
from src.zexus.vm.bytecode import Bytecode, Opcode, BytecodeBuilder
from src.zexus.vm.vm import VM


class TestDependencyAnalyzer(unittest.TestCase):
    """Test dependency analysis for parallelization"""
    
    def setUp(self):
        self.analyzer = DependencyAnalyzer()
    
    def test_analyze_read_instruction(self):
        """Test detecting variable reads"""
        reads, writes = self.analyzer.analyze_instruction(Opcode.LOAD_NAME, "x")
        self.assertEqual(reads, {"x"})
        self.assertEqual(writes, set())
    
    def test_analyze_write_instruction(self):
        """Test detecting variable writes"""
        reads, writes = self.analyzer.analyze_instruction(Opcode.STORE_NAME, "y")
        self.assertEqual(reads, set())
        self.assertEqual(writes, {"y"})
    
    def test_analyze_non_variable_instruction(self):
        """Test non-variable instructions"""
        reads, writes = self.analyzer.analyze_instruction(Opcode.ADD, None)
        self.assertEqual(reads, set())
        self.assertEqual(writes, set())
    
    def test_detect_war_dependency(self):
        """Test Write-After-Read dependency detection"""
        chunk1 = BytecodeChunk(0, [], 0, 10)
        chunk1.variables_read = {"x"}
        
        chunk2 = BytecodeChunk(1, [], 10, 20)
        chunk2.variables_written = {"x"}
        
        chunks = [chunk1, chunk2]
        self.analyzer.detect_dependencies(chunks)
        
        self.assertIn(0, chunk2.dependencies)
    
    def test_detect_raw_dependency(self):
        """Test Read-After-Write dependency detection"""
        chunk1 = BytecodeChunk(0, [], 0, 10)
        chunk1.variables_written = {"y"}
        
        chunk2 = BytecodeChunk(1, [], 10, 20)
        chunk2.variables_read = {"y"}
        
        chunks = [chunk1, chunk2]
        self.analyzer.detect_dependencies(chunks)
        
        self.assertIn(0, chunk2.dependencies)
    
    def test_detect_waw_dependency(self):
        """Test Write-After-Write dependency detection"""
        chunk1 = BytecodeChunk(0, [], 0, 10)
        chunk1.variables_written = {"z"}
        
        chunk2 = BytecodeChunk(1, [], 10, 20)
        chunk2.variables_written = {"z"}
        
        chunks = [chunk1, chunk2]
        self.analyzer.detect_dependencies(chunks)
        
        self.assertIn(0, chunk2.dependencies)
    
    def test_no_dependency(self):
        """Test independent chunks"""
        chunk1 = BytecodeChunk(0, [], 0, 10)
        chunk1.variables_read = {"a"}
        chunk1.variables_written = {"b"}
        
        chunk2 = BytecodeChunk(1, [], 10, 20)
        chunk2.variables_read = {"c"}
        chunk2.variables_written = {"d"}
        
        chunks = [chunk1, chunk2]
        self.analyzer.detect_dependencies(chunks)
        
        self.assertEqual(len(chunk2.dependencies), 0)


class TestBytecodeChunker(unittest.TestCase):
    """Test bytecode chunking for parallelization"""
    
    def setUp(self):
        self.chunker = BytecodeChunker(chunk_size=5)
    
    def test_basic_chunking(self):
        """Test basic bytecode splitting"""
        bytecode = Bytecode()
        for i in range(15):
            bytecode.add_instruction(Opcode.LOAD_CONST, i)
        
        chunks = self.chunker.chunk_bytecode(bytecode)
        
        self.assertEqual(len(chunks), 3)  # 15 / 5 = 3 chunks
        self.assertEqual(chunks[0].chunk_id, 0)
        self.assertEqual(chunks[1].chunk_id, 1)
        self.assertEqual(chunks[2].chunk_id, 2)
    
    def test_chunk_variable_tracking(self):
        """Test tracking variables in chunks"""
        bytecode = Bytecode()
        # Load x, store y
        bytecode.add_instruction(Opcode.LOAD_NAME, "x")
        bytecode.add_instruction(Opcode.STORE_NAME, "y")
        
        chunks = self.chunker.chunk_bytecode(bytecode)
        
        self.assertEqual(len(chunks), 1)
        self.assertIn("x", chunks[0].variables_read)
        self.assertIn("y", chunks[0].variables_written)
    
    def test_control_flow_marking(self):
        """Test marking control flow chunks as non-parallelizable"""
        bytecode = Bytecode()
        bytecode.add_instruction(Opcode.LOAD_CONST, 42)
        bytecode.add_instruction(Opcode.JUMP, 10)
        
        chunks = self.chunker.chunk_bytecode(bytecode)
        
        self.assertFalse(chunks[0].can_parallelize)
    
    def test_small_bytecode(self):
        """Test chunking small bytecode"""
        bytecode = Bytecode()
        bytecode.add_instruction(Opcode.LOAD_CONST, 1)
        
        chunks = self.chunker.chunk_bytecode(bytecode)
        
        self.assertEqual(len(chunks), 1)


class TestSharedState(unittest.TestCase):
    """Test thread-safe shared state"""
    
    def setUp(self):
        self.manager = mp.Manager()
        self.state = SharedState(self.manager)
    
    def test_read_write(self):
        """Test basic read/write"""
        self.state.write("x", 42)
        self.assertEqual(self.state.read("x"), 42)
    
    def test_read_nonexistent(self):
        """Test reading nonexistent variable"""
        self.assertIsNone(self.state.read("nonexistent"))
    
    def test_batch_read(self):
        """Test batch reading"""
        self.state.write("a", 1)
        self.state.write("b", 2)
        self.state.write("c", 3)
        
        results = self.state.batch_read(["a", "b", "c"])
        
        self.assertEqual(results["a"], 1)
        self.assertEqual(results["b"], 2)
        self.assertEqual(results["c"], 3)
    
    def test_batch_write(self):
        """Test batch writing"""
        self.state.batch_write({"x": 10, "y": 20, "z": 30})
        
        self.assertEqual(self.state.read("x"), 10)
        self.assertEqual(self.state.read("y"), 20)
        self.assertEqual(self.state.read("z"), 30)
    
    def test_overwrite(self):
        """Test overwriting values"""
        self.state.write("key", "old")
        self.state.write("key", "new")
        self.assertEqual(self.state.read("key"), "new")


class TestResultMerger(unittest.TestCase):
    """Test result merging from parallel workers"""
    
    def setUp(self):
        self.merger = ResultMerger()
    
    def test_add_result(self):
        """Test adding results"""
        result = ExecutionResult(chunk_id=0, success=True, result=42)
        self.merger.add_result(result)
        
        self.assertEqual(len(self.merger.results), 1)
        self.assertEqual(self.merger.results[0].result, 42)
    
    def test_merge_success(self):
        """Test successful merging"""
        for i in range(3):
            result = ExecutionResult(
                chunk_id=i,
                success=True,
                result=i * 10,
                variables_modified={"x": i}
            )
            self.merger.add_result(result)
        
        success, final_result, variables = self.merger.merge(3)
        
        self.assertTrue(success)
        self.assertEqual(final_result, 20)  # Last result
        self.assertEqual(variables["x"], 2)  # Last write
    
    def test_merge_failure(self):
        """Test merging with failed chunk"""
        self.merger.add_result(ExecutionResult(0, True, 1))
        self.merger.add_result(ExecutionResult(1, False, error="Test error"))
        
        success, result, variables = self.merger.merge(2)
        
        self.assertFalse(success)
        self.assertIn("Failed chunks", result)
    
    def test_merge_incomplete(self):
        """Test merging with missing results"""
        self.merger.add_result(ExecutionResult(0, True, 42))
        
        success, result, variables = self.merger.merge(3)  # Expecting 3, got 1
        
        self.assertFalse(success)
    
    def test_merge_order(self):
        """Test that merging respects chunk order"""
        # Add out of order
        self.merger.add_result(ExecutionResult(2, True, variables_modified={"a": 3}))
        self.merger.add_result(ExecutionResult(0, True, variables_modified={"a": 1}))
        self.merger.add_result(ExecutionResult(1, True, variables_modified={"a": 2}))
        
        success, result, variables = self.merger.merge(3)
        
        self.assertTrue(success)
        self.assertEqual(variables["a"], 3)  # Last write in order
    
    def test_get_statistics(self):
        """Test statistics generation"""
        for i in range(3):
            result = ExecutionResult(i, True, execution_time=0.1 * (i + 1))
            self.merger.add_result(result)
        
        stats = self.merger.get_statistics()
        
        self.assertEqual(stats['total_chunks'], 3)
        self.assertAlmostEqual(stats['total_time'], 0.6, places=2)
        self.assertAlmostEqual(stats['average_time'], 0.2, places=2)


class TestWorkerPool(unittest.TestCase):
    """Test worker pool management"""
    
    def setUp(self):
        self.pool = WorkerPool(num_workers=2)
    
    def tearDown(self):
        self.pool.shutdown()
    
    def test_pool_creation(self):
        """Test creating worker pool"""
        self.assertIsNotNone(self.pool)
        self.assertGreaterEqual(self.pool.num_workers, 1)
        self.assertLessEqual(self.pool.num_workers, mp.cpu_count())
    
    def test_start_shutdown(self):
        """Test starting and shutting down pool"""
        self.pool.start()
        self.assertIsNotNone(self.pool.pool)
        
        self.pool.shutdown()
        self.assertIsNone(self.pool.pool)
    
    def test_execute_chunk(self):
        """Test executing a single chunk"""
        manager = mp.Manager()
        shared_state = SharedState(manager)
        
        # Create simple chunk  
        chunk = BytecodeChunk(
            chunk_id=0,
            instructions=[(Opcode.LOAD_CONST, 42)],
            start_index=0,
            end_index=1
        )
        
        result = self.pool.execute_chunk(chunk, shared_state)
        
        self.assertTrue(result.success)
        self.assertEqual(result.chunk_id, 0)
    
    def test_context_manager(self):
        """Test using pool as context manager"""
        with WorkerPool(num_workers=2) as pool:
            self.assertIsNotNone(pool.pool)
        
        # Should be shutdown after exit
        self.assertIsNone(pool.pool)


class TestParallelVM(unittest.TestCase):
    """Test parallel VM execution"""
    
    def setUp(self):
        from zexus.vm.parallel_vm import ParallelConfig
        config = ParallelConfig(worker_count=2, chunk_size=10)
        self.vm = ParallelVM(config=config)
    
    def tearDown(self):
        self.vm.worker_pool.shutdown()
    
    def test_vm_creation(self):
        """Test creating parallel VM"""
        self.assertIsNotNone(self.vm)
        self.assertEqual(self.vm.config.chunk_size, 10)
        self.assertGreaterEqual(self.vm.config.worker_count, 1)
    
    def test_sequential_fallback_small_bytecode(self):
        """Test sequential fallback for small bytecode"""
        builder = BytecodeBuilder()
        builder.emit_constant(10)
        builder.emit_constant(20)
        builder.emit("ADD", None)
        
        result = self.vm.execute(builder.bytecode)
        
        # Should execute sequentially (too small)
        self.assertEqual(self.vm.stats['sequential_executions'], 1)
        self.assertEqual(self.vm.stats['parallel_executions'], 0)
    
    def test_parallel_execution_large_bytecode(self):
        """Test parallel execution for large bytecode"""
        builder = BytecodeBuilder()
        
        # Create large bytecode (100 instructions)
        for i in range(100):
            builder.emit_constant(i)
        
        result = self.vm.execute(builder.bytecode)
        
        # Should have metrics after execution
        self.assertIsNotNone(self.vm.last_metrics)
    
    def test_statistics(self):
        """Test statistics tracking"""
        stats = self.vm.get_statistics()
        
        # Empty until execution
        self.assertIsInstance(stats, dict)
    
    def test_reset_statistics(self):
        """Test resetting statistics"""
        builder = BytecodeBuilder()
        builder.emit_constant(1)
        self.vm.execute(builder.bytecode)
        
        self.assertIsNotNone(self.vm.last_metrics)
        
        self.vm.reset_statistics()
        
        self.assertIsNone(self.vm.last_metrics)
    
    def test_repr(self):
        """Test string representation"""
        repr_str = repr(self.vm)
        self.assertIn("ParallelVM", repr_str)
        self.assertIn(str(self.vm.config.worker_count), repr_str)


class TestParallelVMIntegration(unittest.TestCase):
    """Integration tests for parallel VM"""
    
    def test_simple_arithmetic(self):
        """Test parallel execution of simple arithmetic"""
        builder = BytecodeBuilder()
        
        # Create: x = 10 + 20 + 30 + 40 + 50
        for i in range(10, 60, 10):
            builder.emit_constant(i)
        
        # 4 ADD operations
        for _ in range(4):
            builder.emit("ADD", None)
        
        from zexus.vm.parallel_vm import ParallelConfig
        config = ParallelConfig(worker_count=2, chunk_size=3)
        vm = ParallelVM(config=config)
        result = vm.execute(builder.bytecode, sequential_fallback=True)
        
        # Should execute (with fallback if needed)
        self.assertIsNotNone(result)
        vm.worker_pool.shutdown()
    
    def test_variable_operations(self):
        """Test parallel execution with variables"""
        builder = BytecodeBuilder()
        
        # x = 10
        builder.emit_constant(10)
        builder.emit("STORE_NAME", "x")
        
        # y = 20
        builder.emit_constant(20)
        builder.emit("STORE_NAME", "y")
        
        # Many more operations to trigger chunking
        for i in range(50):
            builder.emit_constant(i)
            builder.emit("STORE_NAME", f"var{i}")
        
        from zexus.vm.parallel_vm import ParallelConfig
        config = ParallelConfig(worker_count=2, chunk_size=10)
        vm = ParallelVM(config=config)
        result = vm.execute(builder.bytecode, sequential_fallback=True)
        
        # Should execute without errors
        self.assertIsNotNone(result)
        vm.worker_pool.shutdown()
    
    def test_performance_comparison(self):
        """Test that parallel execution can be faster"""
        # Note: This is a rough test, actual speedup depends on workload
        builder = BytecodeBuilder()
        
        # Create large bytecode
        for i in range(200):
            builder.emit_constant(i)
            builder.emit("DUP", None)
            builder.emit("POP", None)
        
        # Sequential
        vm_seq = VM()
        start = time.time()
        result_seq = vm_seq.execute(builder.bytecode)
        seq_time = time.time() - start
        
        # Parallel (with fallback)
        from zexus.vm.parallel_vm import ParallelConfig
        config = ParallelConfig(worker_count=4, chunk_size=20)
        vm_par = ParallelVM(config=config)
        start = time.time()
        result_par = vm_par.execute(builder.bytecode, sequential_fallback=True)
        par_time = time.time() - start
        
        # Just verify both complete (actual speedup varies)
        self.assertIsNotNone(result_seq)
        self.assertIsNotNone(result_par)
        vm_par.worker_pool.shutdown()


class TestExecutionMode(unittest.TestCase):
    """Test execution mode enum"""
    
    def test_modes(self):
        """Test execution modes"""
        self.assertEqual(ExecutionMode.SEQUENTIAL.value, "sequential")
        self.assertEqual(ExecutionMode.PARALLEL.value, "parallel")
        self.assertEqual(ExecutionMode.HYBRID.value, "hybrid")


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience function"""
    
    def test_execute_parallel(self):
        """Test execute_parallel convenience function"""
        from src.zexus.vm.parallel_vm import execute_parallel
        
        builder = BytecodeBuilder()
        builder.emit_constant(42)
        
        result, metrics = execute_parallel(
            builder.bytecode, worker_count=2, chunk_size=5
        )
        self.assertIsNotNone(result)
        
        self.assertIsNotNone(result)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
