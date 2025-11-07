"""
Minimal/Resilient Semantic Analyzer for the compiler frontend.

Provides:
 - `SemanticAnalyzer` with a simple `environment` mapping
 - `register_builtins(builtins)` to accept interpreter builtins
 - `analyze(ast)` returning a list of semantic errors (empty = ok)

This implementation is intentionally permissive so the compiler pipeline can run
while you incrementally implement full semantic checks (name resolution, types).
"""

from typing import List, Dict, Any

class SemanticAnalyzer:
	def __init__(self):
		# Simple environment used by the compiler frontend (name -> value)
		self.environment: Dict[str, Any] = {}
		self._builtins_registered = False

	def register_builtins(self, builtins: Dict[str, Any]):
		"""Register builtin functions/values into the analyzer environment.

		Best-effort: won't raise on unexpected builtin shapes.
		"""
		if not builtins:
			return
		try:
			for name, val in builtins.items():
				if name not in self.environment:
					self.environment[name] = val
			self._builtins_registered = True
		except Exception:
			# Non-fatal: leave environment as-is
			self._builtins_registered = False

	def analyze(self, ast) -> List[str]:
		"""Perform minimal semantic analysis and return a list of error messages.

		Currently lightweight: verifies AST structure and allows compilation to proceed.
		Extend this with full name resolution, type checks, export checks, etc.
		"""
		errors: List[str] = []

		try:
			if ast is None:
				errors.append("No AST provided")
				return errors

			stmts = getattr(ast, "statements", None)
			if stmts is None:
				errors.append("Invalid AST: missing 'statements' list")
				return errors

			# Placeholder for additional semantic checks:
			# - verify top-level declarations
			# - detect duplicate symbols
			# - simple builtin usage checks (optional)

		except Exception as e:
			errors.append(f"Semantic analyzer internal error: {e}")

		return errors