# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
Cython-accelerated hot-path VM execution.

Implements a fast dispatch loop for common bytecode ops used in benchmarks.
Falls back to Python execution for unsupported ops by raising NotImplementedError.
"""

from zexus.object import List as ZList
from zexus.object import Map as ZMap
from zexus.object import String as ZString


def _resolve(name, dict env, dict closure_cells):
    if name in env:
        val = env[name]
        if hasattr(val, "value"):
            return val.value
        return val
    if closure_cells is not None and name in closure_cells:
        cell = closure_cells[name]
        if hasattr(cell, "value"):
            return cell.value
        return cell
    return None


def _store(name, value, dict env, dict closure_cells):
    if name in env:
        current = env[name]
        if hasattr(current, "value"):
            current.value = value
            return
        env[name] = value
        return
    if closure_cells is not None and name in closure_cells:
        cell = closure_cells[name]
        if hasattr(cell, "value"):
            cell.value = value
            return
        closure_cells[name] = value
        return
    env[name] = value


def _call_method(target, method_name, args):
    if target is None:
        return None
    try:
        if method_name == "set":
            if isinstance(target, ZMap) and len(args) >= 2:
                key = args[0]
                if isinstance(key, str):
                    key = ZString(key)
                return target.set(key, args[1])
            if isinstance(target, ZList) and len(args) >= 2:
                return target.set(args[0], args[1])
            if isinstance(target, (dict, list)) and len(args) >= 2:
                target[args[0]] = args[1]
                return args[1]
        if hasattr(target, "call_method"):
            return target.call_method(method_name, args)
        attr = getattr(target, method_name, None)
        if callable(attr):
            return attr(*args)
        if isinstance(target, dict) and method_name in target:
            candidate = target[method_name]
            return candidate(*args) if callable(candidate) else candidate
        return attr
    except Exception:
        return None


def _call_top(fn_obj, args):
    if fn_obj is None:
        return None
    if isinstance(fn_obj, dict) and "bytecode" in fn_obj:
        raise NotImplementedError("CALL_TOP requires VM execution")
    try:
        if callable(fn_obj):
            return fn_obj(*args)
        return None
    except Exception:
        return None


def execute(list instrs, list consts, dict env, dict builtins, dict closure_cells=None):
    cdef Py_ssize_t ip = 0
    cdef Py_ssize_t instr_count = len(instrs)
    cdef list stack = []
    cdef object op_name
    cdef object operand

    while ip < instr_count:
        op_name, operand = instrs[ip]
        ip += 1

        if op_name == "LOAD_CONST":
            if isinstance(operand, int) and 0 <= operand < len(consts):
                stack.append(consts[operand])
            else:
                stack.append(operand)
        elif op_name == "LOAD_NAME":
            name = consts[operand] if isinstance(operand, int) and 0 <= operand < len(consts) else operand
            stack.append(_resolve(name, env, closure_cells))
        elif op_name == "STORE_NAME":
            name = consts[operand] if isinstance(operand, int) and 0 <= operand < len(consts) else operand
            val = stack.pop() if stack else None
            _store(name, val, env, closure_cells)
        elif op_name == "POP":
            if stack:
                stack.pop()
        elif op_name == "DUP":
            if stack:
                stack.append(stack[-1])
        elif op_name == "ADD":
            b = stack.pop() if stack else 0
            a = stack.pop() if stack else 0
            if hasattr(a, "value"):
                a = a.value
            if hasattr(b, "value"):
                b = b.value
            stack.append(a + b)
        elif op_name == "SUB":
            b = stack.pop() if stack else 0
            a = stack.pop() if stack else 0
            if hasattr(a, "value"):
                a = a.value
            if hasattr(b, "value"):
                b = b.value
            stack.append(a - b)
        elif op_name == "MUL":
            b = stack.pop() if stack else 0
            a = stack.pop() if stack else 0
            if hasattr(a, "value"):
                a = a.value
            if hasattr(b, "value"):
                b = b.value
            stack.append(a * b)
        elif op_name == "DIV":
            b = stack.pop() if stack else 1
            a = stack.pop() if stack else 0
            if hasattr(a, "value"):
                a = a.value
            if hasattr(b, "value"):
                b = b.value
            stack.append(a / b if b != 0 else 0)
        elif op_name == "MOD":
            b = stack.pop() if stack else 1
            a = stack.pop() if stack else 0
            stack.append(a % b if b != 0 else 0)
        elif op_name == "EQ":
            b = stack.pop() if stack else None
            a = stack.pop() if stack else None
            stack.append(a == b)
        elif op_name == "NEQ":
            b = stack.pop() if stack else None
            a = stack.pop() if stack else None
            stack.append(a != b)
        elif op_name == "LT":
            b = stack.pop() if stack else 0
            a = stack.pop() if stack else 0
            stack.append(a < b)
        elif op_name == "GT":
            b = stack.pop() if stack else 0
            a = stack.pop() if stack else 0
            stack.append(a > b)
        elif op_name == "LTE":
            b = stack.pop() if stack else 0
            a = stack.pop() if stack else 0
            stack.append(a <= b)
        elif op_name == "GTE":
            b = stack.pop() if stack else 0
            a = stack.pop() if stack else 0
            stack.append(a >= b)
        elif op_name == "NOT":
            a = stack.pop() if stack else False
            stack.append(not a)
        elif op_name == "NEG":
            a = stack.pop() if stack else 0
            stack.append(-a)
        elif op_name == "JUMP":
            ip = operand
        elif op_name == "JUMP_IF_FALSE":
            cond = stack.pop() if stack else None
            if not cond:
                ip = operand
        elif op_name == "RETURN":
            return stack.pop() if stack else None
        elif op_name == "BUILD_LIST":
            count = operand if operand is not None else 0
            elements = [stack.pop() for _ in range(count)][::-1]
            stack.append(elements)
        elif op_name == "BUILD_MAP":
            count = operand if operand is not None else 0
            result = {}
            for _ in range(count):
                val = stack.pop(); key = stack.pop()
                result[key] = val
            stack.append(result)
        elif op_name == "BUILD_SET":
            count = operand if operand is not None else 0
            elements = [stack.pop() for _ in range(count)][::-1]
            stack.append(set(elements))
        elif op_name == "INDEX":
            idx = stack.pop()
            obj = stack.pop()
            try:
                if isinstance(obj, ZList):
                    stack.append(obj.get(idx))
                elif isinstance(obj, ZMap):
                    stack.append(obj.get(idx))
                else:
                    stack.append(obj[idx] if obj is not None else None)
            except Exception:
                stack.append(None)
        elif op_name == "SLICE":
            end = stack.pop() if stack else None
            start = stack.pop() if stack else None
            obj = stack.pop() if stack else None
            if hasattr(start, "value"):
                start = start.value
            if hasattr(end, "value"):
                end = end.value
            try:
                if isinstance(obj, ZList):
                    stack.append(ZList(obj.elements[start:end]))
                elif isinstance(obj, ZString):
                    stack.append(ZString(obj.value[start:end]))
                else:
                    stack.append(obj[start:end] if obj is not None else None)
            except Exception:
                stack.append(None)
        elif op_name == "GET_LENGTH":
            obj = stack.pop() if stack else None
            try:
                if obj is None:
                    stack.append(0)
                elif isinstance(obj, ZList):
                    stack.append(len(obj.elements))
                elif isinstance(obj, ZMap):
                    stack.append(len(obj.pairs))
                elif isinstance(obj, ZString):
                    stack.append(len(obj.value))
                else:
                    stack.append(len(obj))
            except Exception:
                stack.append(0)
        elif op_name == "CALL_NAME":
            name_idx, arg_count = operand
            func_name = consts[name_idx] if isinstance(name_idx, int) and 0 <= name_idx < len(consts) else name_idx
            args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
            fn = _resolve(func_name, env, closure_cells) or builtins.get(func_name)
            if fn is None:
                stack.append(None)
            else:
                stack.append(fn(*args))
        elif op_name == "CALL_BUILTIN":
            name_idx, arg_count = operand
            func_name = consts[name_idx] if isinstance(name_idx, int) and 0 <= name_idx < len(consts) else name_idx
            args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
            fn = builtins.get(func_name)
            if fn is None:
                stack.append(None)
            else:
                try:
                    stack.append(fn(*args))
                except Exception:
                    stack.append(None)
        elif op_name == "CALL_METHOD":
            method_idx, arg_count = operand
            args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
            target = stack.pop() if stack else None
            method_name = consts[method_idx] if isinstance(method_idx, int) and 0 <= method_idx < len(consts) else method_idx
            stack.append(_call_method(target, method_name, args))
        elif op_name == "CALL_TOP":
            arg_count = operand
            args = [stack.pop() for _ in range(arg_count)][::-1] if arg_count else []
            fn_obj = stack.pop() if stack else None
            stack.append(_call_top(fn_obj, args))
        else:
            raise NotImplementedError(f"opcode not supported: {op_name}")

    return stack.pop() if stack else None
