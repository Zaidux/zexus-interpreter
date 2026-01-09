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
# Import compiler AST node classes we need to inspect
from .zexus_ast import Program, ActionStatement, AwaitExpression, ProtocolDeclaration, EventDeclaration, MapLiteral, BlockStatement

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

			visited = set()

			def walk(node, in_async=False):
				if node is None:
					return
				node_id = id(node)
				if node_id in visited:
					return
				visited.add(node_id)

				if isinstance(node, AwaitExpression):
					if not in_async:
						errors.append("Semantic error: 'await' used outside an async function")
						return
					expr = getattr(node, "expression", None)
					if hasattr(expr, "__dict__"):
						walk(expr, in_async=True)
					return

				if isinstance(node, ActionStatement):
					body = getattr(node, "body", None)
					async_flag = getattr(node, "is_async", False)
					if body:
						for s in getattr(body, "statements", []):
							walk(s, in_async=async_flag)
					return

				if isinstance(node, ProtocolDeclaration):
					spec = getattr(node, "spec", {})
					methods = spec.get("methods") if isinstance(spec, dict) else None
					if not isinstance(methods, list):
						errors.append(f"Protocol '{node.name.value}' spec invalid: 'methods' list missing")
					else:
						for m in methods:
							if not isinstance(m, str):
								errors.append(f"Protocol '{node.name.value}' has non-string method name: {m}")
					return

				if isinstance(node, EventDeclaration):
					props = getattr(node, "properties", None)
					if not isinstance(props, (MapLiteral, BlockStatement)):
						errors.append(f"Event '{node.name.value}' properties should be a map or block")
					return

				for attr, val in vars(node).items():
					if attr.startswith("_") or attr in ("token", "token_literal"):
						continue
					if isinstance(val, list):
						for item in val:
							if hasattr(item, "__dict__"):
								walk(item, in_async=in_async)
					elif hasattr(val, "__dict__"):
						walk(val, in_async=in_async)

			for s in stmts:
				walk(s, in_async=False)

		except Exception as e:
			errors.append(f"Semantic analyzer internal error: {e}")

		return errors