import unittest
import threading
import concurrent.futures
from zexus.vm.vm import VM
from zexus.vm.bytecode import BytecodeBuilder

# Check if JIT is available
try:
    JIT_AVAILABLE = True
except (ImportError, ModuleNotFoundError, AttributeError):
    JIT_AVAILABLE = False

class TestConcurrentExecution(unittest.TestCase):
    """Test VM in concurrent scenarios"""
    
    def test_thread_safety(self):
        """Test VM can be used safely from multiple threads"""
        results = []
        errors = []
        
        def worker(worker_id, vm_instance, bytecode):
            try:
                for i in range(100):
                    result = vm_instance.execute(bytecode)
                    if result != 42:
                        errors.append(f"Worker {worker_id}: Wrong result {result}")
                results.append((worker_id, "success"))
            except Exception as e:
                errors.append(f"Worker {worker_id} failed: {e}")
                results.append((worker_id, "failed"))
        
        # Create simple bytecode
        builder = BytecodeBuilder()
        builder.emit_load_const(42)
        builder.emit_return()
        bytecode = builder.build()
        
        # Create multiple VM instances (thread-local is safer)
        vms = [VM(debug=False) for _ in range(4)]
        
        # Run in threads
        threads = []
        for i, vm in enumerate(vms):
            t = threading.Thread(
                target=worker,
                args=(i, vm, bytecode),
                name=f"VM-Worker-{i}"
            )
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        print(f"\n🧵 CONCURRENT EXECUTION:")
        print(f"   Workers: {len(vms)}")
        print(f"   Successes: {len([r for r in results if r[1] == 'success'])}")
        print(f"   Errors: {len(errors)}")
        
        # All threads should complete successfully
        self.assertEqual(len(errors), 0, f"Thread safety issues: {errors}")
        self.assertEqual(len(results), len(vms), "All workers should complete")
    
    def test_concurrent_jit_compilation(self):
        """Test JIT compilation doesn't have race conditions"""
        if not JIT_AVAILABLE:
            self.skipTest("JIT not available")
        
        from concurrent.futures import ThreadPoolExecutor
        
        # Create hot code that will trigger JIT
        builder = BytecodeBuilder()
        for i in range(50):
            builder.emit_load_const(i)
            builder.emit_load_const(2)
            builder.emit_mul()
            builder.emit_pop()
        builder.emit_load_const(999)
        builder.emit_return()
        
        bytecode = builder.build()
        
        def compile_and_execute(thread_id):
            vm = VM(use_jit=True, jit_threshold=5, debug=False)
            results = []
            for i in range(20):  # Enough to trigger JIT
                result = vm.execute(bytecode)
                results.append(result)
            return thread_id, results[-1] == 999
        
        # Run concurrent compilations
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(compile_and_execute, i) for i in range(8)]
            results = [f.result() for f in futures]
        
        # All should complete without errors
        success_count = sum(1 for _, success in results if success)
        
        print(f"\n⚡ CONCURRENT JIT COMPILATION:")
        print(f"   Threads: {len(results)}")
        print(f"   Successful: {success_count}")
        
        self.assertEqual(success_count, len(results), 
                         "All concurrent JIT compilations should succeed")
    
    def test_concurrent_memory_access(self):
        """Test concurrent access to shared VM memory is safe"""
        shared_results = []
        errors = []
        
        def memory_worker(worker_id):
            try:
                vm = VM(debug=False)
                for i in range(50):
                    builder = BytecodeBuilder()
                    builder.emit_load_const(worker_id * 100 + i)
                    builder.emit_store_name(f"worker_{worker_id}_var_{i}")
                    builder.emit_load_name(f"worker_{worker_id}_var_{i}")
                    builder.emit_return()
                    
                    bytecode = builder.build()
                    result = vm.execute(bytecode)
                    expected = worker_id * 100 + i
                    if result != expected:
                        errors.append(f"Worker {worker_id} iteration {i}: got {result}, expected {expected}")
                
                shared_results.append((worker_id, "success"))
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
                shared_results.append((worker_id, "failed"))
        
        # Run multiple workers
        import threading
        threads = []
        for i in range(8):
            t = threading.Thread(target=memory_worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        print(f"\n[MEMORY] CONCURRENT MEMORY ACCESS:")
        print(f"   Workers: {len(threads)}")
        print(f"   Successes: {len([r for r in shared_results if r[1] == 'success'])}")
        print(f"   Errors: {len(errors)}")
        
        self.assertEqual(len(errors), 0, f"Memory safety issues: {errors[:5]}")
        self.assertEqual(len(shared_results), len(threads))
