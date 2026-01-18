"""
Native JIT backend using LLVM (llvmlite).

Compiles a restricted subset of numeric bytecode into native machine code.
"""

from typing import Optional, Callable, List, Tuple, Dict, Any

try:
    from llvmlite import ir
    from llvmlite import binding as llvm
    _LLVM_AVAILABLE = True
except Exception:
    _LLVM_AVAILABLE = False
    ir = None
    llvm = None


class NativeJITBackend:
    def __init__(self, debug: bool = False):
        if not _LLVM_AVAILABLE:
            raise ImportError("llvmlite is required for native JIT")
        self.debug = debug
        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        self.target = llvm.Target.from_default_triple()
        self.target_machine = self.target.create_target_machine()
        self.engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), self.target_machine)
        self._register_cabi_symbols()

    def _register_cabi_symbols(self) -> None:
        symbols = None
        try:
            from . import native_runtime
            symbols = native_runtime.get_symbols()
        except Exception:
            symbols = None
        if symbols is None:
            try:
                from . import cabi
                symbols = cabi.get_symbols()
            except Exception:
                symbols = None

        if isinstance(symbols, dict):
            for name, addr in symbols.items():
                try:
                    llvm.add_symbol(name, int(addr))
                except Exception:
                    continue

    def compile(self, bytecode) -> Optional[Callable]:
        instrs = list(getattr(bytecode, "instructions", []))
        consts = list(getattr(bytecode, "constants", []))

        normalized: List[Tuple[str, Any]] = []
        names: List[str] = []
        name_index: Dict[str, int] = {}

        for instr in instrs:
            if instr is None:
                continue
            if isinstance(instr, tuple) and len(instr) >= 2:
                op = instr[0]
                operand = instr[1] if len(instr) == 2 else tuple(instr[1:])
                op_name = op.name if hasattr(op, "name") else op
                normalized.append((op_name, operand))
                if op_name in ("LOAD_NAME", "STORE_NAME"):
                    name = consts[operand] if isinstance(operand, int) and 0 <= operand < len(consts) else operand
                    if isinstance(name, str) and name not in name_index:
                        name_index[name] = len(names)
                        names.append(name)
            else:
                return None

        supported = {
            "LOAD_CONST",
            "LOAD_NAME",
            "STORE_NAME",
            "POP",
            "DUP",
            "ADD",
            "SUB",
            "MUL",
            "DIV",
            "MOD",
            "NEG",
            "EQ",
            "NEQ",
            "LT",
            "GT",
            "LTE",
            "GTE",
            "AND",
            "OR",
            "NOT",
            "RETURN",
            "JUMP",
            "JUMP_IF_FALSE",
        }
        object_ops = {
            "CALL_NAME",
            "CALL_METHOD",
            "CALL_TOP",
            "BUILD_LIST",
            "BUILD_MAP",
            "INDEX",
            "GET_ATTR",
            "GET_LENGTH",
        }
        use_object_mode = False
        for op_name, _ in normalized:
            if op_name in object_ops:
                use_object_mode = True
            if op_name not in supported and op_name not in object_ops:
                return None

        if use_object_mode:
            func_ptr = self._build_object_function(normalized, consts)
        else:
            func_ptr = self._build_function(normalized, consts, names)
        if func_ptr is None:
            return None

        from ctypes import CFUNCTYPE, c_double, POINTER, py_object

        if use_object_mode:
            cfunc = CFUNCTYPE(py_object, py_object, py_object, POINTER(py_object))(func_ptr)
            const_objs = [c for c in consts]

            def jit_execute(vm, stack, env):
                try:
                    const_buf = (py_object * max(len(const_objs), 1))(*const_objs)
                    return cfunc(env, vm.builtins, const_buf)
                except Exception:
                    return None

            return jit_execute
        else:
            cfunc = CFUNCTYPE(c_double, POINTER(c_double), POINTER(c_double))(func_ptr)
            const_values = [float(c) if isinstance(c, (int, float)) else 0.0 for c in consts]

            def jit_execute(vm, stack, env):
                try:
                    locals_buf = (c_double * max(len(names), 1))()
                    for idx, name in enumerate(names):
                        val = env.get(name)
                        if isinstance(val, bool):
                            val = 1.0 if val else 0.0
                        if not isinstance(val, (int, float)):
                            return None
                        locals_buf[idx] = float(val)
                    const_buf = (c_double * max(len(const_values), 1))(*const_values)
                    result = cfunc(locals_buf, const_buf)
                    # write-back locals to env
                    for idx, name in enumerate(names):
                        env[name] = float(locals_buf[idx])
                    return result
                except Exception:
                    return None

            return jit_execute

    def _build_function(self, instrs: List[Tuple[str, Any]], consts: List[Any], names: List[str]) -> Optional[int]:
        module = ir.Module(name="zexus_native_jit")
        double = ir.DoubleType()
        double_ptr = double.as_pointer()
        func_type = ir.FunctionType(double, [double_ptr, double_ptr])
        func = ir.Function(module, func_type, name="jit_fn")

        blocks = [func.append_basic_block(name=f"b{i}") for i in range(len(instrs) + 1)]
        builder = ir.IRBuilder(blocks[0])

        stack_size = max(8, len(instrs) + 4)
        stack_arr = builder.alloca(ir.ArrayType(double, stack_size))
        stack_base = builder.gep(stack_arr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        sp = builder.alloca(ir.IntType(32))
        builder.store(ir.Constant(ir.IntType(32), 0), sp)

        locals_ptr, const_ptr = func.args

        def push(val):
            cur_sp = builder.load(sp)
            ptr = builder.gep(stack_base, [cur_sp])
            builder.store(val, ptr)
            builder.store(builder.add(cur_sp, ir.Constant(ir.IntType(32), 1)), sp)

        def pop():
            cur_sp = builder.load(sp)
            new_sp = builder.sub(cur_sp, ir.Constant(ir.IntType(32), 1))
            builder.store(new_sp, sp)
            ptr = builder.gep(stack_base, [new_sp])
            return builder.load(ptr)

        for idx, (op_name, operand) in enumerate(instrs):
            builder = ir.IRBuilder(blocks[idx])

            if op_name == "LOAD_CONST":
                if isinstance(operand, int) and 0 <= operand < len(consts):
                    val = consts[operand]
                    val = float(val) if isinstance(val, (int, float)) else 0.0
                else:
                    val = float(operand) if isinstance(operand, (int, float)) else 0.0
                push(ir.Constant(double, val))
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "LOAD_NAME":
                if isinstance(operand, int) and 0 <= operand < len(consts):
                    name = consts[operand]
                else:
                    name = operand
                if name in names:
                    name_idx = names.index(name)
                    ptr = builder.gep(locals_ptr, [ir.Constant(ir.IntType(32), name_idx)])
                    push(builder.load(ptr))
                else:
                    push(ir.Constant(double, 0.0))
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "STORE_NAME":
                if isinstance(operand, int) and 0 <= operand < len(consts):
                    name = consts[operand]
                else:
                    name = operand
                val = pop()
                if name in names:
                    name_idx = names.index(name)
                    ptr = builder.gep(locals_ptr, [ir.Constant(ir.IntType(32), name_idx)])
                    builder.store(val, ptr)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "POP":
                _ = pop()
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DUP":
                val = pop()
                push(val)
                push(val)
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("ADD", "SUB", "MUL", "DIV", "MOD"):
                b = pop()
                a = pop()
                if op_name == "ADD":
                    res = builder.fadd(a, b)
                elif op_name == "SUB":
                    res = builder.fsub(a, b)
                elif op_name == "MUL":
                    res = builder.fmul(a, b)
                elif op_name == "DIV":
                    res = builder.fdiv(a, b)
                else:
                    res = builder.frem(a, b)
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "NEG":
                a = pop()
                res = builder.fsub(ir.Constant(double, 0.0), a)
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("EQ", "NEQ", "LT", "GT", "LTE", "GTE"):
                b = pop()
                a = pop()
                if op_name == "EQ":
                    cmp = builder.fcmp_ordered("==", a, b)
                elif op_name == "NEQ":
                    cmp = builder.fcmp_ordered("!=", a, b)
                elif op_name == "LT":
                    cmp = builder.fcmp_ordered("<", a, b)
                elif op_name == "GT":
                    cmp = builder.fcmp_ordered(">", a, b)
                elif op_name == "LTE":
                    cmp = builder.fcmp_ordered("<=", a, b)
                else:
                    cmp = builder.fcmp_ordered(">=", a, b)
                # bool -> double
                res = builder.uitofp(cmp, double)
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("AND", "OR"):
                b = pop()
                a = pop()
                zero = ir.Constant(double, 0.0)
                a_true = builder.fcmp_ordered("!=", a, zero)
                b_true = builder.fcmp_ordered("!=", b, zero)
                if op_name == "AND":
                    res_bool = builder.and_(a_true, b_true)
                else:
                    res_bool = builder.or_(a_true, b_true)
                res = builder.uitofp(res_bool, double)
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "NOT":
                a = pop()
                zero = ir.Constant(double, 0.0)
                is_false = builder.fcmp_ordered("==", a, zero)
                res = builder.uitofp(is_false, double)
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "JUMP":
                target = int(operand) if operand is not None else idx + 1
                builder.branch(blocks[target])
                continue

            if op_name == "JUMP_IF_FALSE":
                cond = pop()
                zero = ir.Constant(double, 0.0)
                is_false = builder.fcmp_ordered("==", cond, zero)
                target = int(operand) if operand is not None else idx + 1
                builder.cbranch(is_false, blocks[target], blocks[idx + 1])
                continue

            if op_name == "RETURN":
                res = pop()
                builder.ret(res)
                continue

            builder.branch(blocks[idx + 1])

        builder = ir.IRBuilder(blocks[-1])
        builder.ret(ir.Constant(double, 0.0))

        llvm_ir = str(module)
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()

        func_ptr = self.engine.get_function_address("jit_fn")
        return func_ptr

    def _build_object_function(self, instrs: List[Tuple[str, Any]], consts: List[Any]) -> Optional[int]:
        ptr = ir.IntType(8).as_pointer()  # PyObject*
        ptr_ptr = ptr.as_pointer()       # PyObject**

        func_type = ir.FunctionType(ptr, [ptr, ptr, ptr_ptr])
        func = ir.Function(ir.Module(name="zexus_native_jit_obj"), func_type, name="jit_obj_fn")
        env_ptr, builtins_ptr, consts_ptr = func.args

        module = func.module

        # Declare C ABI functions
        def decl(name, ret, args):
            return ir.Function(module, ir.FunctionType(ret, args), name=name)

        cabi_env_get = decl("zexus_env_get", ptr, [ptr, ptr])
        cabi_env_set = decl("zexus_env_set", ir.IntType(32), [ptr, ptr, ptr])
        cabi_call_name = decl("zexus_call_name", ptr, [ptr, ptr, ptr, ptr])
        cabi_call_method = decl("zexus_call_method", ptr, [ptr, ptr, ptr])
        cabi_call_callable = decl("zexus_call_callable", ptr, [ptr, ptr])
        cabi_build_list = decl("zexus_build_list_from_array", ptr, [ptr, ir.IntType(64)])
        cabi_build_map = decl("zexus_build_map_from_array", ptr, [ptr, ir.IntType(64)])
        cabi_build_tuple = decl("zexus_build_tuple_from_array", ptr, [ptr, ir.IntType(64)])
        cabi_add = decl("zexus_number_add", ptr, [ptr, ptr])
        cabi_sub = decl("zexus_number_sub", ptr, [ptr, ptr])
        cabi_mul = decl("zexus_number_mul", ptr, [ptr, ptr])
        cabi_div = decl("zexus_number_div", ptr, [ptr, ptr])
        cabi_mod = decl("zexus_number_mod", ptr, [ptr, ptr])
        cabi_neg = decl("zexus_number_neg", ptr, [ptr])
        cabi_eq = decl("zexus_compare_eq", ptr, [ptr, ptr])
        cabi_neq = decl("zexus_compare_ne", ptr, [ptr, ptr])
        cabi_lt = decl("zexus_compare_lt", ptr, [ptr, ptr])
        cabi_gt = decl("zexus_compare_gt", ptr, [ptr, ptr])
        cabi_lte = decl("zexus_compare_lte", ptr, [ptr, ptr])
        cabi_gte = decl("zexus_compare_gte", ptr, [ptr, ptr])
        cabi_truthy = decl("zexus_truthy_int", ir.IntType(32), [ptr])
        cabi_not = decl("zexus_not", ptr, [ptr])
        cabi_bool_and = decl("zexus_bool_and", ptr, [ptr, ptr])
        cabi_bool_or = decl("zexus_bool_or", ptr, [ptr, ptr])
        cabi_index = decl("zexus_index", ptr, [ptr, ptr])
        cabi_get_attr = decl("zexus_get_attr", ptr, [ptr, ptr])
        cabi_get_length = decl("zexus_get_length", ptr, [ptr])

        blocks = [func.append_basic_block(name=f"b{i}") for i in range(len(instrs) + 1)]
        builder = ir.IRBuilder(blocks[0])

        stack_size = max(16, len(instrs) + 4)
        stack_arr = builder.alloca(ir.ArrayType(ptr, stack_size))
        stack_base = builder.gep(stack_arr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        sp = builder.alloca(ir.IntType(32))
        builder.store(ir.Constant(ir.IntType(32), 0), sp)

        def push(val):
            cur_sp = builder.load(sp)
            ptr_slot = builder.gep(stack_base, [cur_sp])
            builder.store(val, ptr_slot)
            builder.store(builder.add(cur_sp, ir.Constant(ir.IntType(32), 1)), sp)

        def pop():
            cur_sp = builder.load(sp)
            new_sp = builder.sub(cur_sp, ir.Constant(ir.IntType(32), 1))
            builder.store(new_sp, sp)
            ptr_slot = builder.gep(stack_base, [new_sp])
            return builder.load(ptr_slot)

        def const_obj(idx):
            return builder.load(builder.gep(consts_ptr, [ir.Constant(ir.IntType(32), idx)]))

        for idx, (op_name, operand) in enumerate(instrs):
            builder = ir.IRBuilder(blocks[idx])

            if op_name == "LOAD_CONST":
                if isinstance(operand, int):
                    push(const_obj(operand))
                else:
                    push(ir.Constant(ptr, 0))
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "LOAD_NAME":
                name_obj = const_obj(int(operand)) if isinstance(operand, int) else ir.Constant(ptr, 0)
                val = builder.call(cabi_env_get, [env_ptr, name_obj])
                push(val)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "STORE_NAME":
                name_obj = const_obj(int(operand)) if isinstance(operand, int) else ir.Constant(ptr, 0)
                val = pop()
                builder.call(cabi_env_set, [env_ptr, name_obj, val])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "POP":
                _ = pop()
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DUP":
                val = pop()
                push(val)
                push(val)
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("ADD", "SUB", "MUL", "DIV", "MOD"):
                b = pop()
                a = pop()
                if op_name == "ADD":
                    res = builder.call(cabi_add, [a, b])
                elif op_name == "SUB":
                    res = builder.call(cabi_sub, [a, b])
                elif op_name == "MUL":
                    res = builder.call(cabi_mul, [a, b])
                elif op_name == "DIV":
                    res = builder.call(cabi_div, [a, b])
                else:
                    res = builder.call(cabi_mod, [a, b])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "NEG":
                a = pop()
                res = builder.call(cabi_neg, [a])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("EQ", "NEQ", "LT", "GT", "LTE", "GTE"):
                b = pop()
                a = pop()
                if op_name == "EQ":
                    res = builder.call(cabi_eq, [a, b])
                elif op_name == "NEQ":
                    res = builder.call(cabi_neq, [a, b])
                elif op_name == "LT":
                    res = builder.call(cabi_lt, [a, b])
                elif op_name == "GT":
                    res = builder.call(cabi_gt, [a, b])
                elif op_name == "LTE":
                    res = builder.call(cabi_lte, [a, b])
                else:
                    res = builder.call(cabi_gte, [a, b])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "AND":
                b = pop()
                a = pop()
                res = builder.call(cabi_bool_and, [a, b])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "OR":
                b = pop()
                a = pop()
                res = builder.call(cabi_bool_or, [a, b])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "NOT":
                a = pop()
                res = builder.call(cabi_not, [a])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "JUMP":
                target = int(operand) if operand is not None else idx + 1
                builder.branch(blocks[target])
                continue

            if op_name == "JUMP_IF_FALSE":
                cond = pop()
                truth = builder.call(cabi_truthy, [cond])
                zero = ir.Constant(ir.IntType(32), 0)
                is_false = builder.icmp_signed("==", truth, zero)
                target = int(operand) if operand is not None else idx + 1
                builder.cbranch(is_false, blocks[target], blocks[idx + 1])
                continue

            if op_name == "CALL_NAME":
                name_idx, arg_count = operand
                name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                # build args tuple from stack
                count_val = ir.Constant(ir.IntType(64), int(arg_count))
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), int(arg_count)))
                args_ptr = builder.gep(stack_base, [base])
                args_ptr_cast = builder.bitcast(args_ptr, ptr)
                args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                # pop args
                builder.store(base, sp)
                res = builder.call(cabi_call_name, [env_ptr, builtins_ptr, name_obj, args_tuple])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "CALL_METHOD":
                method_idx, arg_count = operand
                count_val = ir.Constant(ir.IntType(64), int(arg_count))
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), int(arg_count)))
                args_ptr = builder.gep(stack_base, [base])
                args_ptr_cast = builder.bitcast(args_ptr, ptr)
                args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                builder.store(base, sp)
                target = pop()
                method_obj = const_obj(int(method_idx)) if isinstance(method_idx, int) else ir.Constant(ptr, 0)
                res = builder.call(cabi_call_method, [target, method_obj, args_tuple])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "CALL_TOP":
                arg_count = int(operand) if operand is not None else 0
                count_val = ir.Constant(ir.IntType(64), arg_count)
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                args_ptr = builder.gep(stack_base, [base])
                args_ptr_cast = builder.bitcast(args_ptr, ptr)
                args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                builder.store(base, sp)
                callable_obj = pop()
                res = builder.call(cabi_call_callable, [callable_obj, args_tuple])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "BUILD_LIST":
                count = int(operand) if operand is not None else 0
                count_val = ir.Constant(ir.IntType(64), count)
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), count))
                items_ptr = builder.gep(stack_base, [base])
                items_ptr_cast = builder.bitcast(items_ptr, ptr)
                lst = builder.call(cabi_build_list, [items_ptr_cast, count_val])
                builder.store(base, sp)
                push(lst)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "BUILD_MAP":
                count = int(operand) if operand is not None else 0
                count_val = ir.Constant(ir.IntType(64), count * 2)
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), count * 2))
                items_ptr = builder.gep(stack_base, [base])
                items_ptr_cast = builder.bitcast(items_ptr, ptr)
                mp = builder.call(cabi_build_map, [items_ptr_cast, count_val])
                builder.store(base, sp)
                push(mp)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "INDEX":
                idx_val = pop()
                obj_val = pop()
                res = builder.call(cabi_index, [obj_val, idx_val])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "GET_ATTR":
                attr = pop()
                obj = pop()
                res = builder.call(cabi_get_attr, [obj, attr])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "GET_LENGTH":
                obj = pop()
                res = builder.call(cabi_get_length, [obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "RETURN":
                res = pop()
                builder.ret(res)
                continue

            builder.branch(blocks[idx + 1])

        builder = ir.IRBuilder(blocks[-1])
        builder.ret(ir.Constant(ptr, 0))

        llvm_ir = str(module)
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        return self.engine.get_function_address("jit_obj_fn")
