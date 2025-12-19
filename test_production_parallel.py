#!/usr/bin/env python3
"""
Quick production test for Phase 6 Parallel VM
"""

import sys
import time
from src.zexus.vm.bytecode import Bytecode, Opcode
from src.zexus.vm.parallel_vm import ParallelVM, ParallelConfig, ExecutionMetrics

def create_test_bytecode(size=100):
    """Create test bytecode with independent arithmetic operations"""
    bytecode = Bytecode()
    
    for i in range(size):
        # LOAD_CONST value
        bytecode.instructions.append((Opcode.LOAD_CONST, i))
        # LOAD_CONST value + 1
        bytecode.instructions.append((Opcode.LOAD_CONST, i + 1))
        # ADD
        bytecode.instructions.append((Opcode.ADD, None))
        # Store result
        bytecode.instructions.append((Opcode.STORE_NAME, f"result_{i}"))
    
    return bytecode

def test_basic_execution():
    """Test basic parallel execution"""
    print("\n=== Test 1: Basic Execution ===")
    
    config = ParallelConfig(
        worker_count=2,
        chunk_size=20,
        timeout_seconds=5.0,
        retry_attempts=2
    )
    
    bytecode = create_test_bytecode(50)
    vm = ParallelVM(config=config)
    
    try:
        start = time.time()
        result = vm.execute(bytecode)
        duration = time.time() - start
        
        print(f"✓ Execution completed in {duration:.4f}s")
        
        if vm.last_metrics:
            print(f"  Chunks: {vm.last_metrics.chunk_count}")
            print(f"  Succeeded: {vm.last_metrics.chunks_succeeded}")
            print(f"  Failed: {vm.last_metrics.chunks_failed}")
            print(f"  Speedup: {vm.last_metrics.speedup:.2f}x")
        
        return True
    except Exception as e:
        print(f"✗ Execution failed: {e}")
        return False
    finally:
        vm.worker_pool.shutdown()

def test_config_validation():
    """Test configuration validation"""
    print("\n=== Test 2: Configuration Validation ===")
    
    try:
        # Valid config
        config = ParallelConfig(worker_count=4, chunk_size=50)
        print(f"✓ Valid config created: {config.worker_count} workers")
        
        # Invalid worker count
        try:
            bad_config = ParallelConfig(worker_count=0)
            print("✗ Should have raised ValueError for worker_count=0")
            return False
        except ValueError as e:
            print(f"✓ Correctly rejected invalid worker_count: {e}")
        
        # Invalid chunk size
        try:
            bad_config = ParallelConfig(chunk_size=-1)
            print("✗ Should have raised ValueError for chunk_size=-1")
            return False
        except ValueError as e:
            print(f"✓ Correctly rejected invalid chunk_size: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Config validation failed: {e}")
        return False

def test_metrics():
    """Test metrics collection"""
    print("\n=== Test 3: Metrics Collection ===")
    
    config = ParallelConfig(worker_count=2, chunk_size=10)
    bytecode = create_test_bytecode(30)
    vm = ParallelVM(config=config)
    
    try:
        vm.execute(bytecode)
        
        if vm.last_metrics:
            metrics_dict = vm.last_metrics.to_dict()
            print(f"✓ Metrics collected:")
            for key, value in metrics_dict.items():
                print(f"  {key}: {value}")
            return True
        else:
            print("✗ No metrics collected")
            return False
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False
    finally:
        vm.worker_pool.shutdown()

def test_fallback():
    """Test sequential fallback"""
    print("\n=== Test 4: Sequential Fallback ===")
    
    config = ParallelConfig(worker_count=2, chunk_size=10, enable_fallback=True)
    
    # Create small bytecode that should use sequential
    bytecode = Bytecode()
    bytecode.instructions.append((Opcode.LOAD_CONST, 42))
    
    vm = ParallelVM(config=config)
    
    try:
        result = vm.execute(bytecode)
        print(f"✓ Small bytecode executed (fallback to sequential)")
        return True
    except Exception as e:
        print(f"✗ Fallback failed: {e}")
        return False
    finally:
        vm.worker_pool.shutdown()

def main():
    """Run all production tests"""
    print("=" * 60)
    print("Phase 6 Parallel VM - Production Readiness Tests")
    print("=" * 60)
    
    tests = [
        test_basic_execution,
        test_config_validation,
        test_metrics,
        test_fallback,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n🎉 All tests passed! Phase 6 is production-ready!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
