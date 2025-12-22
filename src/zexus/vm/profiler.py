"""
Instruction-Level Profiler for Zexus VM

Provides detailed execution profiling including:
- Per-instruction execution counts
- Timing statistics (min/max/avg/p95/p99)
- Hot loop detection
- Memory access patterns
- Branch prediction analysis
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum


class ProfilingLevel(Enum):
    """Profiling detail levels"""
    NONE = 0        # No profiling
    BASIC = 1       # Count only
    DETAILED = 2    # Count + timing
    FULL = 3        # Count + timing + memory + branches


@dataclass
class InstructionStats:
    """Statistics for a single instruction"""
    opcode: str
    operand: Any
    ip: int
    
    # Execution statistics
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    
    # Timing samples (for percentile calculation)
    timing_samples: List[float] = field(default_factory=list)
    
    # Memory tracking
    memory_reads: int = 0
    memory_writes: int = 0
    
    # Branch tracking (for jumps)
    branch_taken: int = 0
    branch_not_taken: int = 0
    
    def record_execution(self, execution_time: float = 0.0, increment_count: bool = True):
        """Record a single execution"""
        if increment_count:
            self.count += 1
        if execution_time > 0:
            self.total_time += execution_time
            self.min_time = min(self.min_time, execution_time)
            self.max_time = max(self.max_time, execution_time)
            self.timing_samples.append(execution_time)
            
            # Keep sample size manageable (max 10000 samples)
            if len(self.timing_samples) > 10000:
                # Keep percentiles and random samples
                self.timing_samples = sorted(self.timing_samples)
                keep = [self.timing_samples[i] for i in 
                       [0, len(self.timing_samples)//4, len(self.timing_samples)//2,
                        3*len(self.timing_samples)//4, -1]]
                self.timing_samples = keep
    
    def avg_time(self) -> float:
        """Average execution time"""
        return self.total_time / self.count if self.count > 0 else 0.0
    
    def percentile(self, p: int) -> float:
        """Calculate percentile (0-100)"""
        if not self.timing_samples:
            return 0.0
        sorted_samples = sorted(self.timing_samples)
        idx = int(len(sorted_samples) * p / 100)
        idx = min(idx, len(sorted_samples) - 1)
        return sorted_samples[idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'opcode': self.opcode,
            'operand': str(self.operand),
            'ip': self.ip,
            'count': self.count,
            'total_time_ms': self.total_time * 1000,
            'avg_time_us': self.avg_time() * 1_000_000,
            'min_time_us': self.min_time * 1_000_000 if self.min_time != float('inf') else 0,
            'max_time_us': self.max_time * 1_000_000,
            'p50_us': self.percentile(50) * 1_000_000,
            'p95_us': self.percentile(95) * 1_000_000,
            'p99_us': self.percentile(99) * 1_000_000,
            'memory_reads': self.memory_reads,
            'memory_writes': self.memory_writes,
            'branch_taken': self.branch_taken,
            'branch_not_taken': self.branch_not_taken,
            'branch_prediction_rate': self._branch_prediction_rate()
        }
    
    def _branch_prediction_rate(self) -> float:
        """Calculate branch prediction success rate (always taken heuristic)"""
        total = self.branch_taken + self.branch_not_taken
        if total == 0:
            return 0.0
        # Simple heuristic: predict always taken
        return (self.branch_taken / total) * 100


@dataclass
class HotLoop:
    """Detected hot loop"""
    start_ip: int
    end_ip: int
    iterations: int
    total_time: float = 0.0
    instructions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_ip': self.start_ip,
            'end_ip': self.end_ip,
            'iterations': self.iterations,
            'total_time_ms': self.total_time * 1000,
            'avg_iteration_us': (self.total_time / self.iterations * 1_000_000) if self.iterations > 0 else 0,
            'instruction_count': len(self.instructions),
            'instructions': self.instructions[:10]  # First 10 instructions
        }


class InstructionProfiler:
    """
    Profiler for VM instruction execution
    
    Tracks execution statistics at instruction level with minimal overhead.
    """
    
    def __init__(self, level: ProfilingLevel = ProfilingLevel.DETAILED):
        self.level = level
        self.enabled = level != ProfilingLevel.NONE
        
        # Per-instruction statistics (keyed by instruction pointer)
        self.stats: Dict[int, InstructionStats] = {}
        
        # Opcode frequency counter
        self.opcode_counter: Counter = Counter()
        
        # Hot loop detection
        self.loops: List[HotLoop] = []
        self._backward_jumps: Dict[Tuple[int, int], int] = defaultdict(int)  # (from_ip, to_ip) -> count
        self._loop_start_times: Dict[Tuple[int, int], float] = {}
        
        # Global statistics
        self.total_instructions: int = 0
        self.profiling_overhead: float = 0.0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Memory access tracking
        self.memory_operations: Set[str] = {
            'LOAD_NAME', 'STORE_NAME', 'LOAD_CONST', 'LOAD_ATTR', 'STORE_ATTR',
            'LOAD_INDEX', 'STORE_INDEX', 'STATE_READ', 'STATE_WRITE'
        }
        
        # Branch instructions
        self.branch_operations: Set[str] = {
            'JUMP', 'JUMP_IF_FALSE', 'JUMP_IF_TRUE', 'JUMP_FORWARD', 'JUMP_BACKWARD'
        }
    
    def start(self):
        """Start profiling session"""
        self.start_time = time.perf_counter()
        self.enabled = True
    
    def stop(self):
        """Stop profiling session"""
        self.end_time = time.perf_counter()
        self.enabled = False
    
    def reset(self):
        """Reset all profiling data"""
        self.stats.clear()
        self.opcode_counter.clear()
        self.loops.clear()
        self._backward_jumps.clear()
        self._loop_start_times.clear()
        self.total_instructions = 0
        self.profiling_overhead = 0.0
        self.start_time = None
        self.end_time = None
    
    def record_instruction(
        self,
        ip: int,
        opcode: str,
        operand: Any,
        prev_ip: Optional[int] = None,
        stack_size: int = 0
    ):
        """
        Record execution of an instruction
        
        Args:
            ip: Instruction pointer
            opcode: Operation code
            operand: Operand value
            prev_ip: Previous instruction pointer (for jump detection)
            stack_size: Current stack size
        """
        if not self.enabled:
            return
        
        overhead_start = time.perf_counter()
        
        # Update counters
        self.total_instructions += 1
        self.opcode_counter[opcode] += 1
        
        # Get or create stats for this instruction
        if ip not in self.stats:
            self.stats[ip] = InstructionStats(opcode=opcode, operand=operand, ip=ip)
        
        stat = self.stats[ip]
        
        # Only increment count (timing done separately by measure_instruction)
        stat.count += 1
        
        # Track memory operations (FULL level only)
        if self.level == ProfilingLevel.FULL:
            if opcode in self.memory_operations:
                if opcode in ('LOAD_NAME', 'LOAD_CONST', 'LOAD_ATTR', 'LOAD_INDEX', 'STATE_READ'):
                    stat.memory_reads += 1
                else:
                    stat.memory_writes += 1
            
            # Track branches and detect hot loops
            if prev_ip is not None and opcode in self.branch_operations:
                # Detect backward jump (potential loop)
                if ip < prev_ip:
                    jump_key = (prev_ip, ip)
                    self._backward_jumps[jump_key] += 1
                    
                    # Hot loop threshold: >1000 iterations
                    if self._backward_jumps[jump_key] == 1:
                        self._loop_start_times[jump_key] = time.perf_counter()
                    elif self._backward_jumps[jump_key] % 1000 == 0:
                        # Record as hot loop
                        loop_time = time.perf_counter() - self._loop_start_times.get(jump_key, 0)
                        instructions = [self.stats[i].opcode for i in range(ip, prev_ip + 1) if i in self.stats]
                        loop = HotLoop(
                            start_ip=ip,
                            end_ip=prev_ip,
                            iterations=self._backward_jumps[jump_key],
                            total_time=loop_time,
                            instructions=instructions
                        )
                        # Update or add loop
                        existing = next((l for l in self.loops if l.start_ip == ip and l.end_ip == prev_ip), None)
                        if existing:
                            existing.iterations = loop.iterations
                            existing.total_time = loop_time
                        else:
                            self.loops.append(loop)
                    
                    stat.branch_taken += 1
                else:
                    stat.branch_not_taken += 1
        
        # Minimal overhead tracking for non-FULL levels
        self.profiling_overhead += (time.perf_counter() - overhead_start)
    
    def measure_instruction(self, ip: int, execution_time: float):
        """
        Record execution time for an instruction
        
        Args:
            ip: Instruction pointer
            execution_time: Time taken to execute (seconds)
        """
        if not self.enabled or self.level == ProfilingLevel.BASIC:
            return
        
        if ip in self.stats:
            overhead_start = time.perf_counter()
            # Don't increment count - that was already done by record_instruction
            self.stats[ip].record_execution(execution_time, increment_count=False)
            self.profiling_overhead += (time.perf_counter() - overhead_start)
    
    def get_hottest_instructions(self, top_n: int = 10) -> List[InstructionStats]:
        """Get top N hottest instructions by execution count"""
        return sorted(self.stats.values(), key=lambda s: s.count, reverse=True)[:top_n]
    
    def get_slowest_instructions(self, top_n: int = 10) -> List[InstructionStats]:
        """Get top N slowest instructions by total time"""
        return sorted(self.stats.values(), key=lambda s: s.total_time, reverse=True)[:top_n]
    
    def get_hot_loops(self, min_iterations: int = 1000) -> List[HotLoop]:
        """Get hot loops (loops executed many times)"""
        return [loop for loop in self.loops if loop.iterations >= min_iterations]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary statistics"""
        if not self.stats:
            return {
                'profiling_level': self.level.name,
                'total_instructions': 0,
                'unique_instructions': 0,
                'total_time_sec': 0,
                'instructions_per_sec': 0,
                'profiling_overhead_ms': 0,
                'overhead_percentage': 0,
                'hot_loops_detected': 0,
                'most_common_opcodes': {}
            }
        
        total_time = (self.end_time or time.perf_counter()) - (self.start_time or 0)
        
        return {
            'profiling_level': self.level.name,
            'total_instructions': self.total_instructions,
            'unique_instructions': len(self.stats),
            'total_time_sec': total_time,
            'instructions_per_sec': self.total_instructions / total_time if total_time > 0 else 0,
            'profiling_overhead_ms': self.profiling_overhead * 1000,
            'overhead_percentage': (self.profiling_overhead / total_time * 100) if total_time > 0 else 0,
            'hot_loops_detected': len(self.loops),
            'most_common_opcodes': dict(self.opcode_counter.most_common(10))
        }
    
    def generate_report(self, format: str = 'text', top_n: int = 20) -> str:
        """
        Generate profiling report
        
        Args:
            format: Output format ('text', 'json', 'html')
            top_n: Number of top items to include
            
        Returns:
            Formatted report string
        """
        if format == 'json':
            return self._generate_json_report(top_n)
        elif format == 'html':
            return self._generate_html_report(top_n)
        else:
            return self._generate_text_report(top_n)
    
    def _generate_text_report(self, top_n: int) -> str:
        """Generate text format report"""
        summary = self.get_summary()
        hottest = self.get_hottest_instructions(top_n)
        slowest = self.get_slowest_instructions(top_n)
        hot_loops = self.get_hot_loops()
        
        lines = [
            "=" * 80,
            "ZEXUS VM INSTRUCTION PROFILING REPORT",
            "=" * 80,
            "",
            "SUMMARY",
            "-" * 80,
            f"Profiling Level:       {summary['profiling_level']}",
            f"Total Instructions:    {summary['total_instructions']:,}",
            f"Unique Instructions:   {summary['unique_instructions']:,}",
            f"Total Time:            {summary['total_time_sec']:.4f} seconds",
            f"Instructions/Second:   {summary['instructions_per_sec']:,.0f}",
            f"Profiling Overhead:    {summary['overhead_percentage']:.2f}% ({summary['profiling_overhead_ms']:.2f}ms)",
            f"Hot Loops Detected:    {summary['hot_loops_detected']}",
            "",
            "MOST COMMON OPCODES",
            "-" * 80,
        ]
        
        for opcode, count in summary['most_common_opcodes'].items():
            pct = (count / summary['total_instructions'] * 100) if summary['total_instructions'] > 0 else 0
            lines.append(f"  {opcode:20} {count:10,} ({pct:5.1f}%)")
        
        lines.extend([
            "",
            f"TOP {top_n} HOTTEST INSTRUCTIONS (by count)",
            "-" * 80,
            f"{'IP':>6} {'Opcode':20} {'Count':>12} {'Avg (μs)':>12} {'Total (ms)':>12}",
            "-" * 80,
        ])
        
        for stat in hottest:
            lines.append(
                f"{stat.ip:6} {stat.opcode:20} {stat.count:12,} "
                f"{stat.avg_time() * 1_000_000:12.2f} {stat.total_time * 1000:12.2f}"
            )
        
        if self.level in (ProfilingLevel.DETAILED, ProfilingLevel.FULL):
            lines.extend([
                "",
                f"TOP {top_n} SLOWEST INSTRUCTIONS (by total time)",
                "-" * 80,
                f"{'IP':>6} {'Opcode':20} {'Total (ms)':>12} {'Count':>12} {'Avg (μs)':>12}",
                "-" * 80,
            ])
            
            for stat in slowest:
                lines.append(
                    f"{stat.ip:6} {stat.opcode:20} {stat.total_time * 1000:12.2f} "
                    f"{stat.count:12,} {stat.avg_time() * 1_000_000:12.2f}"
                )
        
        if hot_loops:
            lines.extend([
                "",
                "HOT LOOPS (>1000 iterations)",
                "-" * 80,
                f"{'Start':>6} {'End':>6} {'Iterations':>12} {'Total (ms)':>12} {'Avg/iter (μs)':>15}",
                "-" * 80,
            ])
            
            for loop in sorted(hot_loops, key=lambda l: l.iterations, reverse=True):
                avg_iter = (loop.total_time / loop.iterations * 1_000_000) if loop.iterations > 0 else 0
                lines.append(
                    f"{loop.start_ip:6} {loop.end_ip:6} {loop.iterations:12,} "
                    f"{loop.total_time * 1000:12.2f} {avg_iter:15.2f}"
                )
        
        lines.extend([
            "",
            "=" * 80,
        ])
        
        return "\n".join(lines)
    
    def _generate_json_report(self, top_n: int) -> str:
        """Generate JSON format report"""
        return json.dumps({
            'summary': self.get_summary(),
            'hottest_instructions': [s.to_dict() for s in self.get_hottest_instructions(top_n)],
            'slowest_instructions': [s.to_dict() for s in self.get_slowest_instructions(top_n)],
            'hot_loops': [l.to_dict() for l in self.get_hot_loops()],
            'opcode_distribution': dict(self.opcode_counter)
        }, indent=2)
    
    def _generate_html_report(self, top_n: int) -> str:
        """Generate HTML format report"""
        summary = self.get_summary()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Zexus VM Profiling Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .summary-item {{ background: #f9f9f9; padding: 15px; border-left: 4px solid #4CAF50; }}
        .summary-item strong {{ display: block; color: #333; margin-bottom: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Zexus VM Instruction Profiling Report</h1>
        
        <h2>Summary</h2>
        <div class="summary">
            <div class="summary-item">
                <strong>Total Instructions</strong>
                {summary['total_instructions']:,}
            </div>
            <div class="summary-item">
                <strong>Unique Instructions</strong>
                {summary['unique_instructions']:,}
            </div>
            <div class="summary-item">
                <strong>Total Time</strong>
                {summary['total_time_sec']:.4f} seconds
            </div>
            <div class="summary-item">
                <strong>Instructions/Second</strong>
                {summary['instructions_per_sec']:,.0f}
            </div>
            <div class="summary-item">
                <strong>Profiling Overhead</strong>
                {summary['overhead_percentage']:.2f}%
            </div>
            <div class="summary-item">
                <strong>Hot Loops Detected</strong>
                {summary['hot_loops_detected']}
            </div>
        </div>
        
        <h2>Top {top_n} Hottest Instructions</h2>
        <table>
            <tr>
                <th>IP</th>
                <th>Opcode</th>
                <th>Count</th>
                <th>Avg (μs)</th>
                <th>Total (ms)</th>
            </tr>
"""
        
        for stat in self.get_hottest_instructions(top_n):
            html += f"""
            <tr>
                <td>{stat.ip}</td>
                <td>{stat.opcode}</td>
                <td>{stat.count:,}</td>
                <td>{stat.avg_time() * 1_000_000:.2f}</td>
                <td>{stat.total_time * 1000:.2f}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
</body>
</html>
"""
        return html
