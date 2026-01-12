import sys
import os
import asyncio
import inspect
import importlib
import types
import pytest

# Ensure src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# ...existing code... (imports and helpers)
from zexus.lexer import Lexer
from zexus.parser import Parser as IntParser
from zexus.evaluator import evaluate, builtins as evaluator_builtins
from zexus.object import Environment, EvaluationError
from zexus.compiler import ZexusCompiler, BUILTINS as COMPILER_BUILTINS

# Helper: run interpreter and return a normalized Python value
def run_interpreter_and_get_result(src):
	deflex = Lexer(src)
	parser = IntParser(deflex)
	prog = parser.parse_program()
	if getattr(parser, "errors", None):
		raise AssertionError(f"Interpreter parse errors: {parser.errors}")
	env = Environment()
	res = evaluate(prog, env, debug_mode=False)

	def _drain_coroutine(coro, limit=10000):
		pending = None
		for _ in range(limit):
			done, value = coro.resume(pending)
			if done:
				error = getattr(coro, "error", None)
				if error is not None:
					return error
				if value is not None:
					return value
				return getattr(coro, "result", None)
			pending = None
		return coro

	def _normalize(value):
		visited = 0
		while True:
			if value is None:
				return None
			if isinstance(value, EvaluationError):
				return value
			if isinstance(value, str) and value.startswith("❌"):
				return value
			value_type = value.type() if hasattr(value, "type") else None
			if value_type == "COROUTINE":
				value = _drain_coroutine(value)
				visited += 1
				if visited > 100:
					return value
				continue
			if value_type == "PROMISE":
				if getattr(value, "is_resolved", lambda: False)():
					try:
						value = value.get_value()
						visited += 1
						if visited > 100:
							return value
						continue
					except Exception as exc:
						return f"Promise rejected: {exc}"
				return value
			if hasattr(value, "value"):
				return value.value
			if hasattr(value, "inspect"):
				return value.inspect()
			return value

	if isinstance(res, EvaluationError):
		return res
	if isinstance(res, str) and res.startswith("❌"):
		return res

	try:
		val = env.get("r")
		if val is not None:
			normalized = _normalize(val)
			if normalized is not None:
				return normalized
	except Exception:
		pass

	return _normalize(res)

# Helper: run compiler pipeline and return normalized result
def run_compiler_and_get_result(src):
	compiler = ZexusCompiler(src, enable_optimizations=False)
	bytecode = compiler.compile()
	if compiler.errors:
		raise AssertionError(f"Compiler errors: {compiler.errors}")
	# run bytecode via VM; ensure final expression in source returns the desired value (see tests)
	result = compiler.run_bytecode(debug=False)
	# result might be Python primitive or VM-specific; normalize similarly
	if hasattr(result, "value"):
		return result.value
	if isinstance(result, (int, float, str, bool)):
		return result
	if hasattr(result, "inspect"):
		return result.inspect()
	return result

# Register an async builtin for tests so both interpreter and compiler paths can call it.
# This builtin returns its argument after an asyncio.sleep(0) to simulate async work.
async def _async_identity(x):
	await asyncio.sleep(0)
	# Try to unwrap a Zexus object (Integer/String) by checking .value
	if hasattr(x, "value"):
		return x.value
	return x

# Install into evaluator builtins and compiler BUILTINS proxy (best-effort)
evaluator_builtins["async_identity"] = _async_identity
try:
	if isinstance(COMPILER_BUILTINS, dict):
		COMPILER_BUILTINS["async_identity"] = _async_identity
except Exception:
	# ignore if COMPILER_BUILTINS isn't a dict
	pass

# Test 1: Closure capture parity
def test_closure_capture_parity():
	src = """
let x = 1
action f() {
    return x
}
let x = 2
let r = f()
r
"""
	int_res = run_interpreter_and_get_result(src)
	comp_res = run_compiler_and_get_result(src)
	assert int_res == comp_res, f"Closure parity mismatch: interpreter={int_res!r} compiler={comp_res!r}"

# Test 2: Async/await parity (uses async_identity builtin)
def test_async_await_parity():
	src = """
action async test_async() {
    let v = await async_identity(42)
    return v
}
test_async()
"""
	int_res = run_interpreter_and_get_result(src)
	comp_res = run_compiler_and_get_result(src)
	assert int_res == comp_res, f"Async parity mismatch: interpreter={int_res!r} compiler={comp_res!r}"

# Test 3: Renderer builtin 'mix' parity (simple builtin check)
def test_renderer_mix_builtin_parity():
	src = '''
let m = mix("blue", "white", 0.2)
m
'''
	int_res = run_interpreter_and_get_result(src)
	comp_res = run_compiler_and_get_result(src)
	# normalize to string
	int_str = str(int_res)
	comp_str = str(comp_res)
	assert int_str == comp_str, f"mix builtin mismatch: interpreter={int_str!r} compiler={comp_str!r}"

# A final catch-all that runs a combined scenario exercising closure+async+mix in one source
def test_combined_scenario_parity():
	src = '''
let base = "ok"
action async actor(x) {
    let local = base
    let r = await async_identity(x)
    return r
}
let val = actor(7)
val
'''
	int_res = run_interpreter_and_get_result(src)
	comp_res = run_compiler_and_get_result(src)
	assert int_res == comp_res, f"Combined parity mismatch: interpreter={int_res!r} compiler={comp_res!r}"