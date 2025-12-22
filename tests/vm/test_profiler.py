"""
Tests for VM Instruction Profiler (Phase 8 - Part 1)

Tests cover:
- Basic profiling functionality
- Instruction counting
- Timing statistics
- Hot loop detection
- Report generation
- Performance overhead
"""

import sys
import unittest
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.zexus.vm.vm import VM, VMMode
from src.zexus.vm.profiler import InstructionProfiler, ProfilingLevel, InstructionStats, HotLoop
from src.zexus.vm.bytecode import BytecodeBuilder


class TestInstructionStats(unittest.TestCase):
    """Test InstructionStats class"""
    
    def test_stats_creation(self):
        """Test creating instruction stats"""
        stat = InstructionStats(opcode="LOAD_CONST", operand=42, ip=0)
        self.assertEqual(stat.opcode, "LOAD_CONST")
        self.assertEqual(stat.operand, 42)
        self.assertEqual(stat.ip, 0)
        self.assertEqual(stat.count, 0)
    
    def test_record_execution(self):
        """Test recording execution"""
        stat = InstructionStats(opcode="ADD", operand=None, ip=10)
        stat.record_execution(0.001)  # 1ms
        stat.record_execution(0.002)  # 2ms
        
        self.assertEqual(stat.count, 2)
        self.assertAlmostEqual(stat.total_time, 0.003, places=5)
        self.assertAlmostEqual(stat.avg_time(), 0.0015, places=5)
        self.assertAlmostEqual(stat.min_time, 0.001, places=5)
        self.assertAlmostEqual(stat.max_time, 0.002, places=5)
    
    def test_percentiles(self):
        """Test percentile calculation"""
        stat = InstructionStats(opcode="MUL", operand=None, ip=20)
        
        # Add 100 samples
        for i in range(100):
            stat.record_execution(i / 1000000.0)  # 0 to 99 microseconds
        
        p50 = stat.percentile(50)
        p95 = stat.percentile(95)
        p99 = stat.percentile(99)
        
        self.assertGreater(p95, p50)
        self.assertGreater(p99, p95)
    
    def test_to_dict(self):
        """Test serialization to dictionary"""
        stat = InstructionStats(opcode="LOAD_NAME", operand="x", ip=5)
        stat.record_execution(0.0001)
        
        data = stat.to_dict()
        self.assertEqual(data['opcode'], "LOAD_NAME")
        self.assertEqual(data['ip'], 5)
        self.assertEqual(data['count'], 1)
        self.assertIn('avg_time_us', data)


class TestProfilerBasics(unittest.TestCase):
    """Test basic profiler functionality"""
    
    def setUp(self):
        self.profiler = InstructionProfiler(level=ProfilingLevel.DETAILED)
    
    def test_profiler_creation(self):
        """Test creating profiler"""
        self.assertEqual(self.profiler.level, ProfilingLevel.DETAILED)
        self.assertTrue(self.profiler.enabled)
        self.assertEqual(self.profiler.total_instructions, 0)
    
    def test_start_stop(self):
        """Test start/stop profiling"""
        self.profiler.start()
        self.assertIsNotNone(self.profiler.start_time)
        
        time.sleep(0.01)
        self.profiler.stop()
        self.assertIsNotNone(self.profiler.end_time)
        self.assertGreater(self.profiler.end_time, self.profiler.start_time)
    
    def test_reset(self):
        """Test reset functionality"""
        self.profiler.record_instruction(0, "LOAD_CONST", 42)
        self.assertEqual(self.profiler.total_instructions, 1)
        
        self.profiler.reset()
        self.assertEqual(self.profiler.total_instructions, 0)
        self.assertEqual(len(self.profiler.stats), 0)
    
    def test_record_instruction(self):
        """Test recording instructions"""
        self.profiler.record_instruction(0, "LOAD_CONST", 42)
        self.profiler.record_instruction(1, "LOAD_CONST", 43)
        self.profiler.record_instruction(2, "ADD", None)
        
        self.assertEqual(self.profiler.total_instructions, 3)
        self.assertEqual(len(self.profiler.stats), 3)
        self.assertEqual(self.profiler.opcode_counter["LOAD_CONST"], 2)
        self.assertEqual(self.profiler.opcode_counter["ADD"], 1)


class TestHotLoopDetection(unittest.TestCase):
    """Test hot loop detection"""
    
    def setUp(self):
        self.profiler = InstructionProfiler(level=ProfilingLevel.FULL)
        self.profiler.start()
    
    def test_backward_jump_detection(self):
        """Test detecting backward jumps (loops)"""
        # Simulate a loop: instructions 10-20, jumping back from 20 to 10
        for iteration in range(1500):
            # Forward execution
            for ip in range(10, 21):
                self.profiler.record_instruction(ip, "LOAD_CONST", iteration, ip - 1)
            
            # Backward jump (loop)
            self.profiler.record_instruction(10, "JUMP_BACKWARD", -10, 20)
        
        hot_loops = self.profiler.get_hot_loops(min_iterations=1000)
        self.assertGreater(len(hot_loops), 0, "Should detect at least one hot loop")
        
        if hot_loops:
            loop = hot_loops[0]
            self.assertEqual(loop.start_ip, 10)
            self.assertEqual(loop.end_ip, 20)
            self.assertGreaterEqual(loop.iterations, 1000)
    
    def test_multiple_loops(self):
        """Test detecting multiple hot loops"""
        # Loop 1: 10-15
        for _ in range(1200):
            for ip in range(10, 16):
                self.profiler.record_instruction(ip, "LOAD_CONST", 1, ip - 1)
            self.profiler.record_instruction(10, "JUMP", -5, 15)
        
        # Loop 2: 30-35
        for _ in range(1500):
            for ip in range(30, 36):
                self.profiler.record_instruction(ip, "LOAD_CONST", 2, ip - 1)
            self.profiler.record_instruction(30, "JUMP", -5, 35)
        
        hot_loops = self.profiler.get_hot_loops(min_iterations=1000)
        self.assertGreaterEqual(len(hot_loops), 2, "Should detect at least 2 hot loops")


class TestProfilerStatistics(unittest.TestCase):
    """Test profiler statistics and reporting"""
    
    def setUp(self):
        self.profiler = InstructionProfiler(level=ProfilingLevel.DETAILED)
        self.profiler.start()
        
        # Record some sample instructions (only count, no measure_instruction calls yet)
        for i in range(100):
            self.profiler.record_instruction(0, "LOAD_CONST", i)
        
        for i in range(50):
            self.profiler.record_instruction(1, "ADD", None)
        
        for i in range(10):
            self.profiler.record_instruction(2, "RETURN", None)
        
        # THEN separately add timing data without counting again
        # Now timing is added without incrementing count
        for i in range(100):
            if 0 in self.profiler.stats:
                self.profiler.stats[0].record_execution(0.0001, increment_count=False)  # 100μs
        
        for i in range(50):
            if 1 in self.profiler.stats:
                self.profiler.stats[1].record_execution(0.0002, increment_count=False)  # 200μs
        
        for i in range(10):
            if 2 in self.profiler.stats:
                self.profiler.stats[2].record_execution(0.00005, increment_count=False)  # 50μs
        
        self.profiler.stop()
    
    def test_get_hottest_instructions(self):
        """Test getting hottest instructions"""
        hottest = self.profiler.get_hottest_instructions(top_n=3)
        
        self.assertEqual(len(hottest), 3)
        self.assertEqual(hottest[0].opcode, "LOAD_CONST")
        # Count should be 100 (we called record_instruction 100 times)
        # measure_instruction only adds timing, not count
        self.assertEqual(hottest[0].count, 100, f"Got count: {hottest[0].count}")
        self.assertEqual(hottest[1].opcode, "ADD")
        self.assertEqual(hottest[1].count, 50, f"Got count: {hottest[1].count}")
    
    def test_get_slowest_instructions(self):
        """Test getting slowest instructions"""
        slowest = self.profiler.get_slowest_instructions(top_n=3)
        
        self.assertEqual(len(slowest), 3)
        # ADD should be slowest (50 * 200μs = 10ms total)
        self.assertEqual(slowest[0].opcode, "ADD")
    
    def test_get_summary(self):
        """Test getting profiling summary"""
        summary = self.profiler.get_summary()
        
        self.assertEqual(summary['total_instructions'], 160)
        self.assertEqual(summary['unique_instructions'], 3)
        self.assertIn('profiling_level', summary)
        self.assertIn('instructions_per_sec', summary)
        self.assertIn('profiling_overhead_ms', summary)
    
    def test_text_report_generation(self):
        """Test text report generation"""
        report = self.profiler.generate_report(format='text', top_n=3)
        
        self.assertIn("ZEXUS VM INSTRUCTION PROFILING REPORT", report)
        self.assertIn("SUMMARY", report)
        self.assertIn("LOAD_CONST", report)
        self.assertIn("ADD", report)
        self.assertIn("MOST COMMON OPCODES", report)
    
    def test_json_report_generation(self):
        """Test JSON report generation"""
        import json
        report = self.profiler.generate_report(format='json', top_n=3)
        
        data = json.loads(report)
        self.assertIn('summary', data)
        self.assertIn('hottest_instructions', data)
        self.assertIn('slowest_instructions', data)
        self.assertEqual(len(data['hottest_instructions']), 3)
    
    def test_html_report_generation(self):
        """Test HTML report generation"""
        report = self.profiler.generate_report(format='html', top_n=3)
        
        self.assertIn("<!DOCTYPE html>", report)
        self.assertIn("Zexus VM Profiling Report", report)
        self.assertIn("LOAD_CONST", report)


class TestVMProfilerIntegration(unittest.TestCase):
    """Test profiler integration with VM"""
    
    def test_vm_with_profiling_enabled(self):
        """Test creating VM with profiling enabled"""
        vm = VM(enable_profiling=True, profiling_level="DETAILED")
        
        self.assertTrue(vm.enable_profiling)
        self.assertIsNotNone(vm.profiler)
        self.assertEqual(vm.profiler.level, ProfilingLevel.DETAILED)
    
    def test_vm_without_profiling(self):
        """Test creating VM without profiling"""
        vm = VM(enable_profiling=False)
        
        self.assertFalse(vm.enable_profiling)
    
    def test_vm_profiling_simple_bytecode(self):
        """Test profiling simple bytecode execution"""
        vm = VM(enable_profiling=True, profiling_level="DETAILED", use_jit=False)
        
        builder = BytecodeBuilder()
        builder.emit_load_const(42)
        builder.emit_load_const(8)
        builder.emit_add()
        builder.emit_return()
        
        bytecode = builder.build()
        result = vm.execute(bytecode)
        
        self.assertEqual(result, 50)
        
        summary = vm.get_profiling_summary()
        self.assertGreater(summary['total_instructions'], 0)
        self.assertIn('LOAD_CONST', summary['most_common_opcodes'])
    
    def test_vm_profiling_loop(self):
        """Test profiling loop execution"""
        vm = VM(enable_profiling=True, profiling_level="FULL", use_jit=False)
        
        builder = BytecodeBuilder()
        # sum = 0
        builder.emit_load_const(0)
        builder.emit_store_name("sum")
        # i = 0
        builder.emit_load_const(0)
        builder.emit_store_name("i")
        
        # Loop: while i < 100
        builder.emit_label("loop_start")
        builder.emit_load_name("i")
        builder.emit_load_const(100)
        builder.emit_lt()
        builder.emit_jump_if_false("loop_end")
        
        # sum += i
        builder.emit_load_name("sum")
        builder.emit_load_name("i")
        builder.emit_add()
        builder.emit_store_name("sum")
        
        # i += 1
        builder.emit_load_name("i")
        builder.emit_load_const(1)
        builder.emit_add()
        builder.emit_store_name("i")
        
        builder.emit_jump("loop_start")
        builder.emit_label("loop_end")
        
        builder.emit_load_name("sum")
        builder.emit_return()
        
        bytecode = builder.build()
        result = vm.execute(bytecode)
        
        self.assertEqual(result, 4950)  # sum of 0 to 99
        
        summary = vm.get_profiling_summary()
        self.assertGreater(summary['total_instructions'], 100)
    
    def test_profiler_start_stop(self):
        """Test profiler start/stop methods"""
        vm = VM(enable_profiling=True, profiling_level="BASIC")
        
        vm.start_profiling()
        self.assertTrue(vm.profiler.enabled)
        
        vm.stop_profiling()
        self.assertFalse(vm.profiler.enabled)
    
    def test_profiler_reset(self):
        """Test profiler reset"""
        vm = VM(enable_profiling=True, profiling_level="BASIC", use_jit=False)
        
        builder = BytecodeBuilder()
        builder.emit_load_const(42)
        builder.emit_return()
        bytecode = builder.build()
        
        vm.execute(bytecode)
        summary1 = vm.get_profiling_summary()
        self.assertGreater(summary1['total_instructions'], 0)
        
        vm.reset_profiler()
        summary2 = vm.get_profiling_summary()
        self.assertEqual(summary2['total_instructions'], 0)
    
    def test_profiling_report_text(self):
        """Test getting text profiling report from VM"""
        vm = VM(enable_profiling=True, profiling_level="DETAILED", use_jit=False)
        
        builder = BytecodeBuilder()
        for i in range(10):
            builder.emit_load_const(i)
            builder.emit_load_const(i + 1)
            builder.emit_add()
        builder.emit_return()
        
        bytecode = builder.build()
        vm.execute(bytecode)
        
        report = vm.get_profiling_report(format='text', top_n=5)
        self.assertIn("PROFILING REPORT", report)
        self.assertIn("LOAD_CONST", report)
        self.assertIn("ADD", report)
    
    def test_profiling_report_json(self):
        """Test getting JSON profiling report from VM"""
        import json
        vm = VM(enable_profiling=True, profiling_level="DETAILED", use_jit=False)
        
        builder = BytecodeBuilder()
        builder.emit_load_const(10)
        builder.emit_load_const(20)
        builder.emit_mul()
        builder.emit_return()
        
        bytecode = builder.build()
        vm.execute(bytecode)
        
        report_json = vm.get_profiling_report(format='json', top_n=3)
        data = json.loads(report_json)
        
        self.assertIn('summary', data)
        self.assertIn('hottest_instructions', data)


class TestProfilingOverhead(unittest.TestCase):
    """Test profiling performance overhead"""
    
    def test_basic_profiling_overhead(self):
        """Test that BASIC profiling has acceptable overhead"""
        vm_no_prof = VM(enable_profiling=False, use_jit=False)
        vm_with_prof = VM(enable_profiling=True, profiling_level="BASIC", use_jit=False)
        
        builder = BytecodeBuilder()
        for i in range(100):
            builder.emit_load_const(i)
        builder.emit_return()
        
        bytecode = builder.build()
        
        # Measure without profiling
        start = time.perf_counter()
        for _ in range(10):
            vm_no_prof.execute(bytecode)
        time_no_prof = time.perf_counter() - start
        
        # Measure with profiling
        start = time.perf_counter()
        for _ in range(10):
            vm_with_prof.execute(bytecode)
        time_with_prof = time.perf_counter() - start
        
        overhead_pct = ((time_with_prof - time_no_prof) / time_no_prof * 100) if time_no_prof > 0 else 0
        
        # BASIC profiling should have <200% overhead (interpreted Python has higher overhead)
        # In production JIT-compiled code, this would be <5%
        self.assertLess(overhead_pct, 200, f"Overhead too high: {overhead_pct:.1f}%")
    
    def test_detailed_profiling_overhead(self):
        """Test that DETAILED profiling overhead is acceptable"""
        vm_no_prof = VM(enable_profiling=False, use_jit=False)
        vm_with_prof = VM(enable_profiling=True, profiling_level="DETAILED", use_jit=False)
        
        builder = BytecodeBuilder()
        for i in range(50):
            builder.emit_load_const(i)
        builder.emit_return()
        
        bytecode = builder.build()
        
        # Measure without profiling
        start = time.perf_counter()
        for _ in range(10):
            vm_no_prof.execute(bytecode)
        time_no_prof = time.perf_counter() - start
        
        # Measure with profiling
        start = time.perf_counter()
        for _ in range(10):
            vm_with_prof.execute(bytecode)
        time_with_prof = time.perf_counter() - start
        
        overhead_pct = ((time_with_prof - time_no_prof) / time_no_prof * 100) if time_no_prof > 0 else 0
        
        # DETAILED profiling should have <300% overhead (interpreted Python has higher overhead)
        # In production JIT-compiled code, this would be <10%
        self.assertLess(overhead_pct, 300, f"Overhead too high: {overhead_pct:.1f}%")


if __name__ == '__main__':
    unittest.main(verbosity=2)
