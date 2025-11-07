#!/usr/bin/env python3
"""
Quick integration checks for interpreter vs compiler.
Run: python3 scripts/verify_integration.py
"""
import sys
import traceback
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(base_dir, "src")
sys.path.insert(0, src_path)

def run_interpreter_test(code, desc):
	print(f"\n--- Interpreter Test: {desc} ---")
	try:
		from zexus.lexer import Lexer as IntLexer
		from zexus.parser import Parser as IntParser
		from zexus.evaluator import evaluate
		from zexus.object import Environment

		lexer = IntLexer(code)
		parser = IntParser(lexer)
		program = parser.parse_program()
		if getattr(parser, "errors", None):
			print("Interpreter parse errors:", parser.errors)
		env = Environment()
		result = evaluate(program, env, debug_mode=False)
		print("Interpreter result:", result)
		return True
	except Exception as e:
		print("Interpreter test failed:", e)
		traceback.print_exc()
		return False

def run_compiler_test(code, desc):
	print(f"\n--- Compiler Test: {desc} ---")
	try:
		from zexus.compiler import ZexusCompiler, BUILTINS, Parser as CompParser

		# If Parser is not available (None), provide clear diagnostic and fail test
		if CompParser is None:
			print("❌ Compiler Parser is not available (zexus.compiler.Parser is None).")
			print("   This indicates an import-time problem in src/zexus/compiler/__init__.py or src/zexus/compiler/parser.py.")
			return False

		has_string = False
		try:
			# BUILTINS may be a dict or similar mapping
			if isinstance(BUILTINS, dict):
				has_string = "string" in BUILTINS
			else:
				has_string = getattr(BUILTINS, "__contains__", lambda k: False)("string")
		except Exception:
			has_string = False

		print("Compiler BUILTINS includes 'string':", has_string)

		compiler = ZexusCompiler(code, enable_optimizations=False)
		bytecode = compiler.compile()
		if compiler.errors:
			print("Compiler errors:", compiler.errors)
			return False
		print("Compiler compiled successfully. Bytecode (repr):", repr(bytecode)[:200])
		return True
	except Exception as e:
		print("Compiler test failed:", e)
		traceback.print_exc()
		return False

def main():
	tests = [
		{
			"code": 'print(string(42));',
			"desc": "builtin string() and semicolon handling"
		},
		{
			"code": '''
try {
    let x = 10 / 0
} catch((error)) {
    print("Error: " + string(error))
}
''',
			"desc": "try/catch parsing and string(error) in catch"
		},
		{
			"code": 'let m = { "a": 1, b: 2 }; print(string(m));',
			"desc": "map/object literal handling and string(map)"
		},
	]

	all_ok = True
	for t in tests:
		code = t["code"]
		desc = t["desc"]
		print("\n\n===== Test:", desc, "=====")
		ok_int = run_interpreter_test(code, desc)
		ok_comp = run_compiler_test(code, desc)
		if not (ok_int and ok_comp):
			all_ok = False
			print(f"❌ {desc} : FAILED (interpreter={ok_int}, compiler={ok_comp})")
		else:
			print(f"✅ {desc} : OK (interpreter and compiler)")

	print("\n\nSummary:", "ALL OK" if all_ok else "SOME TESTS FAILED")
	return 0 if all_ok else 2

if __name__ == "__main__":
	sys.exit(main())
