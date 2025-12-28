"""
Zexus Parser Invariant Checking System (ZPICS)
==============================================

A comprehensive regression prevention system that ensures parser changes
don't break existing functionality by maintaining parse tree fingerprints
and validating statement boundaries.

Author: Zexus Team
Date: December 2025
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ParseSnapshot:
    """Represents a snapshot of how a piece of code parses"""
    source_code: str
    source_hash: str
    statements_count: int
    token_boundaries: List[Tuple[int, int]]  # (start, end) for each statement
    ast_structure: Dict[str, Any]
    variable_declarations: List[str]
    statement_types: List[str]
    parse_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParseSnapshot':
        """Create from dictionary"""
        return cls(**data)
    
    def fingerprint(self) -> str:
        """Generate unique fingerprint for this parse result"""
        # Create deterministic string representation
        fp_data = {
            'source_hash': self.source_hash,
            'statements_count': self.statements_count,
            'token_boundaries': self.token_boundaries,
            'variable_declarations': sorted(self.variable_declarations),
            'statement_types': self.statement_types
        }
        fp_str = json.dumps(fp_data, sort_keys=True)
        return hashlib.sha256(fp_str.encode()).hexdigest()


@dataclass
class InvariantViolation:
    """Represents a detected invariant violation"""
    test_name: str
    violation_type: str
    expected: Any
    actual: Any
    severity: str  # 'critical', 'warning', 'info'
    description: str


class ZPICSValidator:
    """Main validator for parser invariants"""
    
    def __init__(self, snapshot_dir: str = ".zpics_snapshots"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.violations: List[InvariantViolation] = []
    
    def create_snapshot(self, test_name: str, source_code: str) -> ParseSnapshot:
        """
        Create a parse snapshot for given source code
        
        Args:
            test_name: Name of the test case
            source_code: The Zexus code to parse
            
        Returns:
            ParseSnapshot object
        """
        from zexus.lexer import Lexer
        from zexus.parser import Parser
        
        # Hash the source code
        source_hash = hashlib.md5(source_code.encode()).hexdigest()
        
        # Tokenize and parse
        try:
            lexer = Lexer(source_code)
            tokens = lexer.tokenize()
            
            parser = Parser()
            ast = parser.parse(tokens)
        except Exception as e:
            # If parsing fails, record it
            return ParseSnapshot(
                source_code=source_code,
                source_hash=source_hash,
                statements_count=0,
                token_boundaries=[],
                ast_structure={'error': str(e)},
                variable_declarations=[],
                statement_types=[],
                parse_metadata={'parse_failed': True, 'error': str(e)}
            )
        
        # Extract metadata from AST
        statements_count, token_boundaries, statement_types = self._analyze_ast(ast)
        variable_declarations = self._extract_variables(ast)
        ast_structure = self._ast_to_dict(ast)
        
        return ParseSnapshot(
            source_code=source_code,
            source_hash=source_hash,
            statements_count=statements_count,
            token_boundaries=token_boundaries,
            ast_structure=ast_structure,
            variable_declarations=variable_declarations,
            statement_types=statement_types,
            parse_metadata={
                'test_name': test_name,
                'created_at': str(Path(__file__).stat().st_mtime)
            }
        )
    
    def save_snapshot(self, test_name: str, snapshot: ParseSnapshot):
        """Save snapshot to disk"""
        snapshot_file = self.snapshot_dir / f"{test_name}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)
    
    def load_snapshot(self, test_name: str) -> Optional[ParseSnapshot]:
        """Load snapshot from disk"""
        snapshot_file = self.snapshot_dir / f"{test_name}.json"
        if not snapshot_file.exists():
            return None
        
        with open(snapshot_file, 'r') as f:
            data = json.load(f)
            return ParseSnapshot.from_dict(data)
    
    def validate_snapshot(self, test_name: str, current: ParseSnapshot, 
                         baseline: ParseSnapshot) -> List[InvariantViolation]:
        """
        Validate current parse against baseline snapshot
        
        Returns list of violations found
        """
        violations = []
        
        # Check if source code changed
        if current.source_hash != baseline.source_hash:
            violations.append(InvariantViolation(
                test_name=test_name,
                violation_type='source_changed',
                expected=baseline.source_hash,
                actual=current.source_hash,
                severity='info',
                description='Source code has changed'
            ))
        
        # Check statement count
        if current.statements_count != baseline.statements_count:
            violations.append(InvariantViolation(
                test_name=test_name,
                violation_type='statement_count_mismatch',
                expected=baseline.statements_count,
                actual=current.statements_count,
                severity='critical',
                description=f'Statement count changed from {baseline.statements_count} to {current.statements_count}'
            ))
        
        # Check token boundaries
        if current.token_boundaries != baseline.token_boundaries:
            violations.append(InvariantViolation(
                test_name=test_name,
                violation_type='token_boundaries_changed',
                expected=baseline.token_boundaries,
                actual=current.token_boundaries,
                severity='critical',
                description='Token collection boundaries have changed'
            ))
        
        # Check variable declarations
        baseline_vars = set(baseline.variable_declarations)
        current_vars = set(current.variable_declarations)
        
        missing_vars = baseline_vars - current_vars
        extra_vars = current_vars - baseline_vars
        
        if missing_vars:
            violations.append(InvariantViolation(
                test_name=test_name,
                violation_type='missing_variables',
                expected=list(baseline_vars),
                actual=list(current_vars),
                severity='critical',
                description=f'Variables missing: {missing_vars}'
            ))
        
        if extra_vars:
            violations.append(InvariantViolation(
                test_name=test_name,
                violation_type='extra_variables',
                expected=list(baseline_vars),
                actual=list(current_vars),
                severity='warning',
                description=f'Extra variables declared: {extra_vars}'
            ))
        
        # Check statement types
        if current.statement_types != baseline.statement_types:
            violations.append(InvariantViolation(
                test_name=test_name,
                violation_type='statement_types_changed',
                expected=baseline.statement_types,
                actual=current.statement_types,
                severity='critical',
                description='Statement types sequence has changed'
            ))
        
        # Check fingerprints
        if current.fingerprint() != baseline.fingerprint():
            violations.append(InvariantViolation(
                test_name=test_name,
                violation_type='fingerprint_mismatch',
                expected=baseline.fingerprint(),
                actual=current.fingerprint(),
                severity='critical',
                description='Parse tree fingerprint has changed'
            ))
        
        return violations
    
    def _analyze_ast(self, ast) -> Tuple[int, List[Tuple[int, int]], List[str]]:
        """Extract statement count, token boundaries, and types from AST"""
        if not hasattr(ast, 'statements'):
            return 0, [], []
        
        statements = ast.statements if hasattr(ast, 'statements') else []
        count = len(statements)
        boundaries = []
        types = []
        
        for stmt in statements:
            stmt_type = type(stmt).__name__
            types.append(stmt_type)
            
            # Try to get token boundaries if available
            if hasattr(stmt, 'start_pos') and hasattr(stmt, 'end_pos'):
                boundaries.append((stmt.start_pos, stmt.end_pos))
            else:
                boundaries.append((0, 0))  # Placeholder
        
        return count, boundaries, types
    
    def _extract_variables(self, ast) -> List[str]:
        """Extract all variable declarations from AST"""
        variables = []
        
        def walk(node):
            if node is None:
                return
            
            # Check for variable declarations
            node_type = type(node).__name__
            if node_type in ['LetStatement', 'ConstStatement']:
                if hasattr(node, 'name') and hasattr(node.name, 'value'):
                    variables.append(node.name.value)
            
            # Recursively walk children
            if hasattr(node, 'statements'):
                for stmt in node.statements:
                    walk(stmt)
            if hasattr(node, 'body'):
                walk(node.body)
        
        walk(ast)
        return variables
    
    def _ast_to_dict(self, node, max_depth: int = 5) -> Dict[str, Any]:
        """Convert AST node to dictionary representation"""
        if max_depth <= 0 or node is None:
            return {'type': 'truncated'}
        
        result = {
            'type': type(node).__name__
        }
        
        # Add relevant attributes
        if hasattr(node, 'value'):
            result['value'] = str(node.value)
        if hasattr(node, 'name') and hasattr(node.name, 'value'):
            result['name'] = node.name.value
        
        # Handle collections
        if hasattr(node, 'statements'):
            result['statements'] = [
                self._ast_to_dict(stmt, max_depth - 1) 
                for stmt in (node.statements or [])
            ]
        
        return result
    
    def generate_report(self, violations: List[InvariantViolation]) -> str:
        """Generate human-readable report of violations"""
        if not violations:
            return "✅ All parser invariants validated successfully!\n"
        
        report = []
        report.append("=" * 70)
        report.append("ZPICS VALIDATION REPORT")
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


def validate_parser_changes(golden_tests_dir: str = "tests/golden") -> bool:
    """
    Main entry point for validating parser changes
    
    Args:
        golden_tests_dir: Directory containing golden test files
        
    Returns:
        True if all validations pass, False otherwise
    """
    validator = ZPICSValidator()
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
            print(f"📸 Creating baseline snapshot for: {test_name}")
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
        print("✅ All parser invariants validated successfully!")
        print(f"   Tested {len(list(golden_dir.glob('*.zx')))} golden test cases")
        return True
