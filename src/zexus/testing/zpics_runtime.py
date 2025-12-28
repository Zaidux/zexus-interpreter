"""
ZPICS Evaluator - Runtime Behavior Validation
==============================================

Extension of ZPICS that validates runtime behavior (evaluator) in addition to parse trees.
Ensures that code changes don't break expected runtime results.

Author: Zexus Team  
Date: December 2025
"""

import json
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from io import StringIO


@dataclass
class ExecutionSnapshot:
    """Represents a snapshot of code execution"""
    source_code: str
    source_hash: str
    stdout_output: str
    stderr_output: str
    exit_code: int
    execution_metadata: Dict[str, Any]
    variables_final: Dict[str, str]  # Final variable values (as strings)
    execution_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionSnapshot':
        """Create from dictionary"""
        return cls(**data)
    
    def fingerprint(self) -> str:
        """Generate unique fingerprint for this execution result"""
        fp_data = {
            'source_hash': self.source_hash,
            'stdout': self.stdout_output,
            'exit_code': self.exit_code,
            'variables': sorted(self.variables_final.items())
        }
        fp_str = json.dumps(fp_data, sort_keys=True)
        return hashlib.sha256(fp_str.encode()).hexdigest()


@dataclass
class RuntimeViolation:
    """Represents a detected runtime behavior violation"""
    test_name: str
    violation_type: str
    expected: Any
    actual: Any
    severity: str  # 'critical', 'warning', 'info'
    description: str


class ZPICSEvaluatorValidator:
    """Validator for runtime behavior invariants"""
    
    def __init__(self, snapshot_dir: str = ".zpics_runtime"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.violations: List[RuntimeViolation] = []
    
    def create_snapshot(self, test_name: str, source_code: str) -> ExecutionSnapshot:
        """
        Create an execution snapshot by running the code
        
        Args:
            test_name: Name of the test case
            source_code: The Zexus code to execute
            
        Returns:
            ExecutionSnapshot object
        """
        import time
        from zexus.lexer import Lexer
        from zexus.parser import Parser
        from zexus.evaluator import Evaluator
        from zexus.object import Environment
        
        # Hash the source code
        source_hash = hashlib.md5(source_code.encode()).hexdigest()
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        exit_code = 0
        variables_final = {}
        execution_time = 0.0
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            start_time = time.time()
            
            # Parse and evaluate
            lexer = Lexer(source_code)
            tokens = lexer.tokenize()
            parser = Parser()
            ast = parser.parse(tokens)
            
            env = Environment()
            evaluator = Evaluator()
            evaluator.eval(ast, env)
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Extract final variable states
            if hasattr(env, 'store'):
                for var_name, var_value in env.store.items():
                    # Skip internal/builtin variables
                    if not var_name.startswith('_'):
                        variables_final[var_name] = str(var_value)
            
        except SystemExit as e:
            exit_code = e.code if e.code else 0
        except Exception as e:
            exit_code = 1
            stderr_capture.write(f"Error: {str(e)}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        return ExecutionSnapshot(
            source_code=source_code,
            source_hash=source_hash,
            stdout_output=stdout_output,
            stderr_output=stderr_output,
            exit_code=exit_code,
            execution_metadata={
                'test_name': test_name
            },
            variables_final=variables_final,
            execution_time_ms=execution_time
        )
    
    def save_snapshot(self, test_name: str, snapshot: ExecutionSnapshot):
        """Save snapshot to disk"""
        snapshot_file = self.snapshot_dir / f"{test_name}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)
    
    def load_snapshot(self, test_name: str) -> Optional[ExecutionSnapshot]:
        """Load snapshot from disk"""
        snapshot_file = self.snapshot_dir / f"{test_name}.json"
        if not snapshot_file.exists():
            return None
        
        with open(snapshot_file, 'r') as f:
            data = json.load(f)
            return ExecutionSnapshot.from_dict(data)
    
    def validate_snapshot(self, test_name: str, current: ExecutionSnapshot,
                         baseline: ExecutionSnapshot) -> List[RuntimeViolation]:
        """
        Validate current execution against baseline snapshot
        
        Returns list of violations found
        """
        violations = []
        
        # Check stdout output
        if current.stdout_output.strip() != baseline.stdout_output.strip():
            violations.append(RuntimeViolation(
                test_name=test_name,
                violation_type='stdout_changed',
                expected=baseline.stdout_output,
                actual=current.stdout_output,
                severity='critical',
                description='Program output has changed'
            ))
        
        # Check exit code
        if current.exit_code != baseline.exit_code:
            violations.append(RuntimeViolation(
                test_name=test_name,
                violation_type='exit_code_changed',
                expected=baseline.exit_code,
                actual=current.exit_code,
                severity='critical',
                description=f'Exit code changed from {baseline.exit_code} to {current.exit_code}'
            ))
        
        # Check final variables
        baseline_vars = set(baseline.variables_final.keys())
        current_vars = set(current.variables_final.keys())
        
        missing_vars = baseline_vars - current_vars
        extra_vars = current_vars - baseline_vars
        
        if missing_vars:
            violations.append(RuntimeViolation(
                test_name=test_name,
                violation_type='missing_variables',
                expected=list(baseline_vars),
                actual=list(current_vars),
                severity='critical',
                description=f'Variables missing after execution: {missing_vars}'
            ))
        
        if extra_vars:
            violations.append(RuntimeViolation(
                test_name=test_name,
                violation_type='extra_variables',
                expected=list(baseline_vars),
                actual=list(current_vars),
                severity='warning',
                description=f'Extra variables after execution: {extra_vars}'
            ))
        
        # Check variable values for common variables
        common_vars = baseline_vars & current_vars
        for var_name in common_vars:
            if baseline.variables_final[var_name] != current.variables_final[var_name]:
                violations.append(RuntimeViolation(
                    test_name=test_name,
                    violation_type='variable_value_changed',
                    expected=f"{var_name}={baseline.variables_final[var_name]}",
                    actual=f"{var_name}={current.variables_final[var_name]}",
                    severity='critical',
                    description=f'Variable {var_name} value changed'
                ))
        
        # Check execution fingerprint
        if current.fingerprint() != baseline.fingerprint():
            violations.append(RuntimeViolation(
                test_name=test_name,
                violation_type='execution_fingerprint_mismatch',
                expected=baseline.fingerprint(),
                actual=current.fingerprint(),
                severity='critical',
                description='Execution behavior fingerprint has changed'
            ))
        
        # Performance regression check (warning only)
        if current.execution_time_ms > baseline.execution_time_ms * 2:
            violations.append(RuntimeViolation(
                test_name=test_name,
                violation_type='performance_regression',
                expected=f"{baseline.execution_time_ms}ms",
                actual=f"{current.execution_time_ms}ms",
                severity='warning',
                description='Execution time increased significantly'
            ))
        
        return violations
    
    def generate_report(self, violations: List[RuntimeViolation]) -> str:
        """Generate human-readable report of violations"""
        if not violations:
            return "✅ All runtime invariants validated successfully!\n"
        
        report = []
        report.append("=" * 70)
        report.append("ZPICS RUNTIME VALIDATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Group by severity
        critical = [v for v in violations if v.severity == 'critical']
        warnings = [v for v in violations if v.severity == 'warning']
        info = [v for v in violations if v.severity == 'info']
        
        if critical:
            report.append(f"❌ CRITICAL VIOLATIONS: {len(critical)}")
            report.append("-" * 70)
            for v in critical:
                report.append(f"  Test: {v.test_name}")
                report.append(f"  Type: {v.violation_type}")
                report.append(f"  {v.description}")
                if v.violation_type in ['stdout_changed']:
                    report.append(f"  Expected output:")
                    for line in str(v.expected).split('\n')[:5]:
                        report.append(f"    {line}")
                    report.append(f"  Actual output:")
                    for line in str(v.actual).split('\n')[:5]:
                        report.append(f"    {line}")
                else:
                    report.append(f"  Expected: {v.expected}")
                    report.append(f"  Actual: {v.actual}")
                report.append("")
        
        if warnings:
            report.append(f"⚠️  WARNINGS: {len(warnings)}")
            report.append("-" * 70)
            for v in warnings:
                report.append(f"  Test: {v.test_name}")
                report.append(f"  {v.description}")
                report.append("")
        
        if info:
            report.append(f"ℹ️  INFO: {len(info)}")
            report.append("-" * 70)
            for v in info:
                report.append(f"  Test: {v.test_name}")
                report.append(f"  {v.description}")
                report.append("")
        
        report.append("=" * 70)
        report.append(f"Total violations: {len(violations)}")
        report.append(f"Critical: {len(critical)} | Warnings: {len(warnings)} | Info: {len(info)}")
        report.append("=" * 70)
        
        return "\n".join(report)


def validate_runtime_changes(golden_tests_dir: str = "tests/golden") -> bool:
    """
    Main entry point for validating runtime behavior changes
    
    Args:
        golden_tests_dir: Directory containing golden test files
        
    Returns:
        True if all validations pass, False otherwise
    """
    validator = ZPICSEvaluatorValidator()
    all_violations = []
    
    golden_dir = Path(golden_tests_dir)
    if not golden_dir.exists():
        print(f"⚠️  Golden tests directory not found: {golden_tests_dir}")
        return True
    
    # Process each golden test file
    for test_file in golden_dir.glob("*.zx"):
        test_name = test_file.stem
        source_code = test_file.read_text()
        
        # Create current snapshot
        current = validator.create_snapshot(test_name, source_code)
        
        # Load baseline
        baseline = validator.load_snapshot(test_name)
        
        if baseline is None:
            # First time running - create baseline
            print(f"📸 Creating runtime baseline for: {test_name}")
            validator.save_snapshot(test_name, current)
            continue
        
        # Validate against baseline
        violations = validator.validate_snapshot(test_name, current, baseline)
        all_violations.extend(violations)
    
    # Generate and print report
    if all_violations:
        print(validator.generate_report(all_violations))
        return False
    else:
        print("✅ All runtime invariants validated successfully!")
        print(f"   Tested {len(list(golden_dir.glob('*.zx')))} golden test cases")
        return True
