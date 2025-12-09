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

			# Run new checks: await usage and protocol/event validation
			# Walk AST with context
			def walk(node, in_async=False):
				# Protect against cycles by tracking visited node ids
				if not hasattr(walk, '_visited'):
					walk._visited = set()
				if node is None:
					return
				node_id = id(node)
				if node_id in walk._visited:
					return
				walk._visited.add(node_id)
				# Quick type checks for relevant nodes
				if isinstance(node, AwaitExpression):
					if not in_async:
						errors.append("Semantic error: 'await' used outside an async function")
				# ActionStatement may have is_async flag
				if isinstance(node, ActionStatement):
					body = getattr(node, "body", None)
					async_flag = getattr(node, "is_async", False)
					if body:
						# walk body with in_async = async_flag
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
					# further checks can be added
					return

				# Generic traversal
				for attr in dir(node):
					if attr.startswith("_") or attr in ("token_literal", "__repr__"):
						continue
					val = getattr(node, attr)
					if isinstance(val, list):
						for item in val:
							if hasattr(item, "__class__"):
								walk(item, in_async=in_async)
					elif hasattr(val, "__class__"):
						walk(val, in_async=in_async)

			# Walk top-level statements
			for s in stmts:
				walk(s, in_async=False)

		except Exception as e:
			errors.append(f"Semantic analyzer internal error: {e}")

		return errors