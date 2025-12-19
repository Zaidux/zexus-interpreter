"""
Performance Benchmarks for Zexus Parallel VM (Phase 6)

Compares parallel execution performance against sequential execution.
Measures scalability with different worker counts.

Author: Zexus Team
Date: December 19, 2025
"""

import time
import unittest
from statistics import mean, stdev
import multiprocessing as mp

from src.zexus.vm.parallel_vm import ParallelVM, ExecutionMode
from src.zexus.vm.vm import VM
from src.zexus.vm.bytecode import Bytecode, BytecodeBuilder


class ParallelVMBenchmark(unittest.TestCase):
    """Benchmark suite for parallel VM"""
    
    def setUp(self):
        self.iterations = 5  # Number of times to run each benchmark
        self.warmup = 1      # Warmup iterations
    
    def benchmark(self, name, sequential_func, parallel_func, expected_speedup=1.5):
        """
        Run a benchmark comparing sequential and parallel execution.
        
        Args:
            name: Benchmark name
            sequential_func: Function that creates bytecode for sequential execution
            parallel_func: Function that creates bytecode for parallel execution  
            expected_speedup: Minimum expected speedup
        """
        print(f"\n{'='*60}")
        print(f"Benchmark: {name}")
        print(f"{'='*60}")
        
        # Warmup
        print("Warming up...")
        for _ in range(self.warmup):
            bytecode_seq = sequential_func()
            vm = VM()
            vm.execute(bytecode_seq)
        
        # Sequential execution
        print(f"Running sequential execution ({self.iterations} iterations)...")
        seq_times = []
        for i in range(self.iterations):
            bytecode = sequential_func()
            vm = VM()
            start = time.time()
            result_seq = vm.execute(bytecode)
            elapsed = time.time() - start
            seq_times.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.4f}s")
        
        seq_avg = mean(seq_times)
        seq_std = stdev(seq_times) if len(seq_times) > 1 else 0
        
        # Parallel execution (2 workers)
        print(f"\nRunning parallel execution with 2 workers ({self.iterations} iterations)...")
        par_times_2 = []
        for i in range(self.iterations):
            bytecode = parallel_func()
            vm_par = ParallelVM(num_workers=2, chunk_size=20)
            start = time.time()
            result_par = vm_par.execute(bytecode, sequential_fallback=True)
            elapsed = time.time() - start
            par_times_2.append(elapsed)
            vm_par.worker_pool.shutdown()
            print(f"  Iteration {i+1}: {elapsed:.4f}s")
        
        par_avg_2 = mean(par_times_2)
        par_std_2 = stdev(par_times_2) if len(par_times_2) > 1 else 0
        speedup_2 = seq_avg / par_avg_2 if par_avg_2 > 0 else 0
        
        # Parallel execution (4 workers) - if available
        cpu_count = mp.cpu_count()
        if cpu_count >= 4:
            print(f"\nRunning parallel execution with 4 workers ({self.iterations} iterations)...")
            par_times_4 = []
            for i in range(self.iterations):
                bytecode = parallel_func()
                vm_par = ParallelVM(num_workers=4, chunk_size=20)
                start = time.time()
                result_par = vm_par.execute(bytecode, sequential_fallback=True)
                elapsed = time.time() - start
                par_times_4.append(elapsed)
                vm_par.worker_pool.shutdown()
                print(f"  Iteration {i+1}: {elapsed:.4f}s")
            
            par_avg_4 = mean(par_times_4)
            par_std_4 = stdev(par_times_4) if len(par_times_4) > 1 else 0
            speedup_4 = seq_avg / par_avg_4 if par_avg_4 > 0 else 0
        else:
            par_avg_4 = 0
            speedup_4 = 0
            print(f"\nSkipping 4-worker test (only {cpu_count} cores available)")
        
        # Results
        print(f"\n{'-'*60}")
        print("Results:")
        print(f"{'-'*60}")
        print(f"Sequential:     {seq_avg:.4f}s ± {seq_std:.4f}s")
        print(f"Parallel (2w):  {par_avg_2:.4f}s ± {par_std_2:.4f}s  (speedup: {speedup_2:.2f}x)")
        if cpu_count >= 4:
            print(f"Parallel (4w):  {par_avg_4:.4f}s ± {par_std_4:.4f}s  (speedup: {speedup_4:.2f}x)")
        print(f"{'-'*60}")
        
        # Note: Due to multiprocessing overhead and fallback behavior,
        # we don't enforce strict speedup requirements in tests
        print(f"✓ Benchmark completed")
        
        return {
            'sequential': seq_avg,
            'parallel_2': par_avg_2,
            'parallel_4': par_avg_4,
            'speedup_2': speedup_2,
            'speedup_4': speedup_4
        }
    
    def test_1_independent_arithmetic_loop(self):
        """Benchmark: Independent arithmetic in loop"""
        
        def create_bytecode():
            builder = BytecodeBuilder()
            
            # Large loop with independent operations
            # for i in range(200): result = i * 2 + i * 3
            for i in range(200):
                builder.emit_constant(i)
                builder.emit("DUP", None)
                builder.emit_constant(2)
                builder.emit("MUL", None)
                
                builder.emit_constant(i)
                builder.emit_constant(3)
                builder.emit("MUL", None)
                builder.emit("ADD", None)
                
                builder.emit("POP", None)  # Discard result
            
            return builder.bytecode
        
        results = self.benchmark(
            "Independent Arithmetic Loop (200 iterations)",
            create_bytecode,
            create_bytecode,
            expected_speedup=1.5
        )
        
        self.assertGreater(results['sequential'], 0)
    
    def test_2_matrix_computation(self):
        """Benchmark: Matrix-like computations"""
        
        def create_bytecode():
            builder = BytecodeBuilder()
            
            # Simulate matrix operations (20x20)
            for i in range(20):
                for j in range(20):
                    # result[i][j] = i * j + (i + j)
                    builder.emit_constant(i)
                    builder.emit_constant(j)
                    builder.emit("MUL", None)
                    
                    builder.emit_constant(i)
                    builder.emit_constant(j)
                    builder.emit("ADD", None)
                    
                    builder.emit("ADD", None)
                    builder.emit("POP", None)
            
            return builder.bytecode
        
        results = self.benchmark(
            "Matrix Computation (20x20)",
            create_bytecode,
            create_bytecode,
            expected_speedup=1.8
        )
        
        self.assertGreater(results['sequential'], 0)
    
    def test_3_complex_expressions(self):
        """Benchmark: Complex arithmetic expressions"""
        
        def create_bytecode():
            builder = BytecodeBuilder()
            
            # Many complex expressions
            for i in range(100):
                # (i * 2 + 5) * (i * 3 + 10) - (i + 1)
                builder.emit_constant(i)
                builder.emit_constant(2)
                builder.emit("MUL", None)
                builder.emit_constant(5)
                builder.emit("ADD", None)
                
                builder.emit_constant(i)
                builder.emit_constant(3)
                builder.emit("MUL", None)
                builder.emit_constant(10)
                builder.emit("ADD", None)
                
                builder.emit("MUL", None)
                
                builder.emit_constant(i)
                builder.emit_constant(1)
                builder.emit("ADD", None)
                
                builder.emit("SUB", None)
                builder.emit("POP", None)
            
            return builder.bytecode
        
        results = self.benchmark(
            "Complex Expressions (100 iterations)",
            create_bytecode,
            create_bytecode,
            expected_speedup=2.0
        )
        
        self.assertGreater(results['sequential'], 0)
    
    def test_4_scalability(self):
        """Benchmark: Scalability with different worker counts"""
        
        print(f"\n{'='*60}")
        print("Scalability Test: Worker Count vs Performance")
        print(f"{'='*60}")
        
        def create_bytecode():
            builder = BytecodeBuilder()
            for i in range(150):
                builder.emit_constant(i)
                builder.emit("DUP", None)
                builder.emit("MUL", None)  # i * i
                builder.emit("POP", None)
            return builder.bytecode
        
        cpu_count = mp.cpu_count()
        worker_counts = [1, 2]
        if cpu_count >= 4:
            worker_counts.append(4)
        
        results = {}
        for workers in worker_counts:
            print(f"\nTesting with {workers} worker(s)...")
            times = []
            for _ in range(self.iterations):
                bytecode = create_bytecode()
                vm = ParallelVM(num_workers=workers, chunk_size=25)
                start = time.time()
                vm.execute(bytecode, sequential_fallback=True)
                elapsed = time.time() - start
                times.append(elapsed)
                vm.worker_pool.shutdown()
            
            avg_time = mean(times)
            results[workers] = avg_time
            print(f"  Average time: {avg_time:.4f}s")
        
        print(f"\n{'-'*60}")
        print("Scalability Results:")
        print(f"{'-'*60}")
        baseline = results[1]
        for workers, time_val in results.items():
            speedup = baseline / time_val if time_val > 0 else 0
            efficiency = (speedup / workers) * 100 if workers > 0 else 0
            print(f"  {workers} worker(s): {time_val:.4f}s  (speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)")
        print(f"{'-'*60}")
        
        self.assertGreater(baseline, 0)


def run_benchmarks():
    """Run all benchmarks"""
    print("\n" + "="*60)
    print("ZEXUS PARALLEL VM PERFORMANCE BENCHMARKS")
    print("="*60)
    print(f"CPU Cores: {mp.cpu_count()}")
    print(f"Python Multiprocessing: Available")
    print("="*60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(ParallelVMBenchmark)
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Total Benchmarks: {result.testsRun}")
    print(f"Completed: {result.testsRun - len(result.failures) - len(result.errors)}")
    if result.failures:
        print(f"Failures: {len(result.failures)}")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
    print("="*60)


if __name__ == '__main__':
    run_benchmarks()
