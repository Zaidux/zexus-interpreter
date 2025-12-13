"""
Extended VM for Zexus.

Capabilities:
 - Execute two bytecode formats:
    * high-level ops list (("DEFINE_SCREEN",...), etc.)
    * low-level Bytecode object with .instructions and .constants (stack-machine)
 - Low-level opcodes: LOAD_CONST, LOAD, STORE, CALL, PRINT, JUMP, JUMP_IF_FALSE, RETURN
 - Async primitives: SPAWN (start coroutine), AWAIT (await coroutine/task)
 - Event system: REGISTER_EVENT, EMIT_EVENT
 - Module import: IMPORT (importlib)
 - Enums: DEFINE_ENUM (store mapping in env)
 - Protocol/assert: ASSERT_PROTOCOL (runtime shape check)
"""
from typing import List, Any, Dict, Tuple, Optional, Union
import asyncio
import importlib
import types

# Try to use renderer backend
try:
	from renderer import backend as _BACKEND
	_BACKEND_AVAILABLE = True
except Exception:
	_BACKEND_AVAILABLE = False
	_BACKEND = None

# NEW: mutable cell used for proper closure capture semantics
class Cell:
	def __init__(self, value):
		self.value = value
	def __repr__(self):
		return f"<Cell {self.value!r}>"

class VM:
	def __init__(self, builtins: Dict[str, Any] = None, env: Dict[str, Any] = None, parent_env: Dict[str, Any] = None):
		# builtins: mapping name -> Builtin wrapper or callable
		self.builtins = builtins or {}
		# env: runtime environment for variables, modules, screens, etc.
		self.env = env or {}
		# parent env for lexical fallback (closure parent)
		self._parent_env = parent_env
		# event registry: event_name -> list of callables or identifiers
		self._events: Dict[str, List[Any]] = {}
		# spawned tasks tracking
		self._tasks: Dict[str, asyncio.Task] = {}
		self._task_counter = 0
		# closure cell mapping (name -> Cell) for this VM frame (used when VM itself is used as closure store)
		self._closure_cells: Dict[str, Cell] = {}

	# Public entry: accept either high-level ops list or low-level Bytecode object
	def execute(self, code: Union[List[Tuple], Any], debug: bool = False):
		# If Bytecode-like object with instructions and constants -> run stack VM
		if hasattr(code, "instructions") and hasattr(code, "constants"):
			if debug:
				print("[VM] Running low-level Bytecode (stack VM)")
			return asyncio.run(self._run_stack_bytecode(code, debug))
		# Otherwise treat as high-level ops list
		if isinstance(code, list):
			if debug:
				print("[VM] Running high-level ops list")
			return asyncio.run(self._run_high_level_ops(code, debug))
		return None

	# -----------------------
	# High-level ops executor
	# -----------------------
	async def _run_high_level_ops(self, ops: List[Tuple], debug: bool = False):
		last = None
		for i, op in enumerate(ops):
			if not isinstance(op, (list, tuple)) or len(op) == 0:
				continue
			code = op[0]
			if debug:
				print(f"[VM HL] op#{i}: {op}")
			try:
				if code == "DEFINE_SCREEN":
					_, name, props = op
					if _BACKEND_AVAILABLE:
						_BACKEND.define_screen(name, props)
					else:
						self.env.setdefault("screens", {})[name] = props
					last = None
				elif code == "DEFINE_COMPONENT":
					_, name, props = op
					if _BACKEND_AVAILABLE:
						_BACKEND.define_component(name, props)
					else:
						self.env.setdefault("components", {})[name] = props
					last = None
				elif code == "DEFINE_THEME":
					_, name, props = op
					self.env.setdefault("themes", {})[name] = props
					last = None
				elif code == "CALL_BUILTIN":
					_, name, arg_ops = op
					args = [self._eval_hl_op(a) for a in arg_ops]
					last = await self._call_builtin_async(name, args)
				elif code == "LET":
					_, name, val_op = op
					val = self._eval_hl_op(val_op)
					self.env[name] = val
					last = None
				elif code == "EXPR":
					_, expr_op = op
					last = self._eval_hl_op(expr_op)
				# NEW: high-level event registration
				elif code == "REGISTER_EVENT":
					_, name, props = op
					# register an event name with an empty handler list (handlers may be added later)
					self._events.setdefault(name, [])
					last = None
				elif code == "EMIT_EVENT":
					_, name, payload_op = op
					payload = self._eval_hl_op(payload_op)
					handlers = self._events.get(name, [])
					for h in handlers:
						# call handler asynchronously
						await self._call_builtin_async(h, [payload])
					last = None
				# NEW: IMPORT
				elif code == "IMPORT":
					_, module_path, alias = op
					try:
						mod = importlib.import_module(module_path)
						self.env[alias or module_path] = mod
					except Exception as e:
						self.env[alias or module_path] = None
					last = None
				elif code == "DEFINE_ENUM":
					_, name, members = op
					self.env.setdefault("enums", {})[name] = members
					last = None
				elif code == "DEFINE_PROTOCOL":
					_, name, spec = op
					self.env.setdefault("protocols", {})[name] = spec
					last = None
				elif code == "AWAIT":
					_, inner_op = op
					# evaluate inner op, which may be CALL_BUILTIN or a coroutine-producing op
					evaluated = self._eval_hl_op(inner_op)
					# if coroutine or future, await
					if asyncio.iscoroutine(evaluated) or isinstance(evaluated, asyncio.Future):
						last = await evaluated
					else:
						last = evaluated
				else:
					# nop or unknown
					last = None
			except Exception as e:
				last = e
		return last

	def _eval_hl_op(self, op):
		if not isinstance(op, tuple):
			return op
		tag = op[0]
		if tag == "LITERAL": return op[1]
		if tag == "IDENT":
			name = op[1]
			if name in self.env: return self.env[name]
			if name in self.builtins: return self.builtins[name]
			return None
		if tag == "CALL_BUILTIN":
			name = op[1]; args = [self._eval_hl_op(a) for a in op[2]]
			# call sync or async through _call_builtin_async
			return asyncio.run(self._call_builtin_async(name, args))
		if tag == "MAP": return {k: self._eval_hl_op(v) for k, v in op[1].items()}
		if tag == "LIST": return [self._eval_hl_op(e) for e in op[1]]
		return None

	# -----------------------
	# Low-level stack VM
	# -----------------------
	async def _run_stack_bytecode(self, bytecode, debug=False):
		consts = list(getattr(bytecode, "constants", []))
		instrs = list(getattr(bytecode, "instructions", []))
		ip = 0
		stack: List[Any] = []
		# helper to resolve constant operand (index)
		def const(idx): return consts[idx] if 0 <= idx < len(consts) else None

		# Helper for lexical name resolution (check local env, closure cells, parent chain)
		def _resolve(name):
			# local env
			if name in self.env:
				val = self.env[name]
				# if value is a Cell, return its content
				if isinstance(val, Cell):
					return val.value
				return val
			# closure cells attached to this VM
			if name in self._closure_cells:
				return self._closure_cells[name].value
			# parent env chain (if parent_env is VM instance or plain dict)
			p = self._parent_env
			while p is not None:
				# if parent is another VM, check its env and its closure
				if isinstance(p, VM):
					if name in p.env:
						val = p.env[name]
						if isinstance(val, Cell):
							return val.value
						return val
					if name in p._closure_cells:
						return p._closure_cells[name].value
					p = p._parent_env
				else:
					# plain dict
					if name in p:
						return p[name]
					# cannot walk further
					p = None
			return None

		# Helper to store into appropriate place (local -> closure -> parent)
		def _store(name, value):
			# If name exists in local env and is a Cell -> update cell
			if name in self.env and isinstance(self.env[name], Cell):
				self.env[name].value = value
				return
			# If name exists in local env (non-cell) -> assign local
			if name in self.env:
				self.env[name] = value
				return
			# If name exists as closure cell -> update there
			if name in self._closure_cells:
				self._closure_cells[name].value = value
				return
			# Walk parent chain: if a closure cell exists in ancestor, update it
			p = self._parent_env
			while p is not None:
				if isinstance(p, VM):
					if name in p._closure_cells:
						p._closure_cells[name].value = value
						return
					if name in p.env:
						p.env[name] = value
						return
					p = p._parent_env
				else:
					if name in p:
						p[name] = value
						return
					p = None
			# otherwise create local binding
			self.env[name] = value

		while ip < len(instrs):
			op, operand = instrs[ip]
			if debug:
				print(f"[VM SL] ip={ip} op={op} operand={operand} stack={stack}")
			ip += 1

			# Stack ops
			if op == "LOAD_CONST":
				stack.append(const(operand))
			elif op == "LOAD_NAME":
				# operand: const index for name
				name = const(operand)
				# Use _resolve to handle closure semantics properly
				val = _resolve(name)
				stack.append(val)
			elif op == "STORE_NAME":
				name = const(operand)
				val = stack.pop() if stack else None
				# Respect closure cells and lexical semantics
				_store(name, val)
			elif op == "POP":
				if stack:
					stack.pop()
			elif op == "DUP":
				if stack:
					stack.append(stack[-1])
			elif op == "STORE_FUNC":
				# operand: (name_idx, func_const_idx)
				name_idx, func_idx = operand
				name = const(name_idx)
				func_desc = const(func_idx)
				# Prepare descriptor copy
				func_desc_copy = dict(func_desc) if isinstance(func_desc, dict) else {"bytecode": func_desc}
				# DO NOT capture closure cells at function definition time.
				# Instead, attach a reference to the CURRENT VM as parent_env.
				# This allows the function to dynamically look up variables from the enclosing scope,
				# ensuring it sees live updates if parent variables are reassigned.
				func_desc_copy["parent_vm"] = self
				# store function descriptor with closure reference into environment
				self.env[name] = func_desc_copy
			elif op == "CALL_NAME":
				# operand: (name_const_idx, arg_count)
				name_idx, arg_count = operand
				func_name = const(name_idx)
				# pop args
				args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
				# resolve function by checking local env (_resolve) then builtins
				fn = _resolve(func_name)
				if fn is None:
					fn = self.builtins.get(func_name)
				res = await self._invoke_callable_or_funcdesc(fn, args)
				stack.append(res)
			elif op == "CALL_FUNC_CONST":
				# operand: (func_const_idx, arg_count)
				func_idx, arg_count = operand
				func_desc = const(func_idx)
				args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
				res = await self._invoke_callable_or_funcdesc(func_desc, args, is_constant=True)
				stack.append(res)
			elif op == "CALL_TOP":
				# operand: arg_count; top of stack is function object/callable
				arg_count = operand
				args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
				fn_obj = stack.pop() if stack else None
				res = await self._invoke_callable_or_funcdesc(fn_obj, args)
				stack.append(res)
			elif op == "PRINT":
				val = stack.pop() if stack else None
				print(val)
			# Arithmetic operations
			elif op == "ADD":
				b = stack.pop() if stack else 0
				a = stack.pop() if stack else 0
				stack.append(a + b)
			elif op == "SUB":
				b = stack.pop() if stack else 0
				a = stack.pop() if stack else 0
				stack.append(a - b)
			elif op == "MUL":
				b = stack.pop() if stack else 0
				a = stack.pop() if stack else 0
				stack.append(a * b)
			elif op == "DIV":
				b = stack.pop() if stack else 1
				a = stack.pop() if stack else 0
				stack.append(a / b if b != 0 else 0)
			elif op == "MOD":
				b = stack.pop() if stack else 1
				a = stack.pop() if stack else 0
				stack.append(a % b if b != 0 else 0)
			elif op == "POW":
				b = stack.pop() if stack else 0
				a = stack.pop() if stack else 0
				stack.append(a ** b)
			elif op == "NEG":
				a = stack.pop() if stack else 0
				stack.append(-a)
			# Comparison operations
			elif op == "EQ":
				b = stack.pop() if stack else None
				a = stack.pop() if stack else None
				stack.append(a == b)
			elif op == "NEQ":
				b = stack.pop() if stack else None
				a = stack.pop() if stack else None
				stack.append(a != b)
			elif op == "LT":
				b = stack.pop() if stack else 0
				a = stack.pop() if stack else 0
				stack.append(a < b)
			elif op == "GT":
				b = stack.pop() if stack else 0
				a = stack.pop() if stack else 0
				stack.append(a > b)
			elif op == "LTE":
				b = stack.pop() if stack else 0
				a = stack.pop() if stack else 0
				stack.append(a <= b)
			elif op == "GTE":
				b = stack.pop() if stack else 0
				a = stack.pop() if stack else 0
				stack.append(a >= b)
			# Logical operations
			elif op == "AND":
				b = stack.pop() if stack else False
				a = stack.pop() if stack else False
				stack.append(a and b)
			elif op == "OR":
				b = stack.pop() if stack else False
				a = stack.pop() if stack else False
				stack.append(a or b)
			elif op == "NOT":
				a = stack.pop() if stack else False
				stack.append(not a)
			# Collection operations
			elif op == "BUILD_LIST":
				count = operand if operand is not None else 0
				elements = [stack.pop() for _ in range(count)][::-1] if count > 0 and len(stack) >= count else []
				stack.append(elements)
			elif op == "BUILD_MAP":
				count = operand if operand is not None else 0
				pairs = count  # number of key-value pairs
				result = {}
				for _ in range(pairs):
					if len(stack) >= 2:
						value = stack.pop()
						key = stack.pop()
						result[key] = value
				stack.append(result)
			elif op == "INDEX":
				index = stack.pop() if stack else 0
				obj = stack.pop() if stack else None
				try:
					if obj is not None:
						stack.append(obj[index])
					else:
						stack.append(None)
				except (KeyError, IndexError, TypeError):
					stack.append(None)
			elif op == "JUMP":
				ip = operand
			elif op == "JUMP_IF_FALSE":
				cond = stack.pop() if stack else None
				falsy = (cond is None) or (cond is False)
				if falsy:
					ip = operand
			elif op == "RETURN":
				return stack.pop() if stack else None
			# Async/Task ops
			elif op == "SPAWN":
				# operand: tuple ("CALL", func_name, arg_count) OR index to function const
				task_handle = None
				if isinstance(operand, tuple) and operand[0] == "CALL":
					fn_name = operand[1]; arg_count = operand[2]
					args = [stack.pop() for _ in range(arg_count)][::-1]
					fn = self.builtins.get(fn_name) or self.env.get(fn_name)
					coro = self._to_coro(fn, args)
					task = asyncio.create_task(coro)
					self._task_counter += 1
					tid = f"task_{self._task_counter}"
					self._tasks[tid] = task
					task_handle = tid
				stack.append(task_handle)
			elif op == "AWAIT":
				# operand: None; await top of stack (task or coroutine)
				top = stack.pop() if stack else None
				if isinstance(top, str) and top in self._tasks:
					task = self._tasks[top]
					res = await task
					stack.append(res)
				elif asyncio.iscoroutine(top) or isinstance(top, asyncio.Future):
					res = await top
					stack.append(res)
				else:
					# not awaitable -> push back as-is
					stack.append(top)
			# Events
			elif op == "REGISTER_EVENT":
				# operand: (event_name_const_idx, handler_name_const_idx)
				event_name = const(operand[0]) if isinstance(operand, (list,tuple)) else const(operand)
				handler = const(operand[1]) if isinstance(operand, (list,tuple)) else None
				self._events.setdefault(event_name, []).append(handler)
			elif op == "EMIT_EVENT":
				# operand: (event_name_const_idx, payload_op)
				event_name = const(operand[0])
				payload = const(operand[1]) if len(operand) > 1 else None
				handlers = self._events.get(event_name, [])
				for h in handlers:
					fn = self.builtins.get(h) or self.env.get(h)
					# dispatch handler; allow async
					asyncio.create_task(self._call_builtin_async_obj(fn, [payload]))
			# Modules
			elif op == "IMPORT":
				# operand: (module_name_const_idx, alias_const_idx_or_none)
				mod_name = const(operand[0])
				alias = const(operand[1]) if len(operand) > 1 else None
				try:
					mod = importlib.import_module(mod_name)
					name = alias or mod_name
					self.env[name] = mod
				except Exception as e:
					self.env[alias or mod_name] = None
			# Enums
			elif op == "DEFINE_ENUM":
				# operand: (enum_name_const, map_const or list)
				enum_name = const(operand[0])
				enum_map = const(operand[1])
				self.env[enum_name] = enum_map
			# Protocol assertion
			elif op == "ASSERT_PROTOCOL":
				# operand: (obj_name_const, protocol_spec_const)
				obj_name = const(operand[0])
				spec = const(operand[1])  # expected dict: method_name->callable-signature
				obj = self.env.get(obj_name)
				ok = True
				missing = []
				for m in spec.get("methods", []):
					if not hasattr(obj, m):
						ok = False; missing.append(m)
				stack.append((ok, missing))
			else:
				# unknown opcode: ignore
				if debug:
					print(f"[VM] Unknown opcode: {op}")
		# end loop
		return stack[-1] if stack else None

	# new helper to execute function descriptor or callable
	async def _invoke_callable_or_funcdesc(self, fn, args, is_constant=False):
		# fn may be:
		# - a descriptor dict: {"bytecode": Bytecode, "params": [...], "is_async": bool, "parent_vm": VM}
		# - a Builtin wrapper or Python callable
		if fn is None:
			return None
		# descriptor case
		if isinstance(fn, dict) and "bytecode" in fn:
			func_bc = fn["bytecode"]
			params = fn.get("params", [])
			is_async = fn.get("is_async", False)
			# Get parent VM if available (ensures closures see live parent env)
			parent_env = fn.get("parent_vm", self)
			# Build local env with parameter bindings
			local_env = {}
			for name, val in zip(params, args):
				local_env[name] = val
			# Create inner VM with parent_env set to the captured parent VM (for closure lookups)
			inner_vm = VM(builtins=self.builtins, env=local_env, parent_env=parent_env)
			# Execute
			if is_async:
				return await inner_vm._run_stack_bytecode(func_bc, debug=False)
			else:
				return await inner_vm._run_stack_bytecode(func_bc, debug=False)
		# Builtin or callable
		if hasattr(fn, "fn") and callable(fn.fn):
			res = fn.fn(*args)
			if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
				return await res
			return res
		if callable(fn):
			res = fn(*args)
			if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
				return await res
			return res
		# unknown
		return fn

	# -----------------------
	# Helpers for async calls
	# -----------------------
	async def _call_builtin_async(self, name: str, args: List[Any]):
		# Resolve target
		target = self.builtins.get(name) if self.builtins else None
		if target is None and name in self.env:
			target = self.env[name]
		# prefer renderer backend mapping
		if _BACKEND_AVAILABLE and hasattr(_BACKEND, name):
			fn = getattr(_BACKEND, name)
			if asyncio.iscoroutinefunction(fn):
				return await fn(*args)
			else:
				return fn(*args)
		# if target is Builtin wrapper with .fn
		if hasattr(target, "fn") and callable(target.fn):
			res = target.fn(*args)
			# if coroutine -> await
			if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
				return await res
			return res
		# plain callable
		if callable(target):
			res = target(*args)
			if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
				return await res
			return res
		# unknown target
		return None

	async def _call_builtin_async_obj(self, fn_obj, args: List[Any]):
		# Accept callables, Builtin wrappers, module functions, or Python callables
		try:
			if fn_obj is None:
				return None
			if hasattr(fn_obj, "fn") and callable(fn_obj.fn):
				res = fn_obj.fn(*args)
			elif callable(fn_obj):
				res = fn_obj(*args)
			else:
				# not callable: return as-is
				return fn_obj
			if asyncio.iscoroutine(res) or isinstance(res, asyncio.Future):
				return await res
			return res
		except Exception as e:
			return e

	def _to_coro(self, fn, args):
		# Ensure we return a coroutine for spawn; wrap sync call in coroutine
		if asyncio.iscoroutinefunction(fn):
			return fn(*args)
		async def _wrap():
			if callable(fn):
				return fn(*args)
			return None
		return _wrap()

	# Utility: inspect current VM env
	def inspect_env(self):
		return dict(self.env)