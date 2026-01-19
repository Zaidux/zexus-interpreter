g"""
Native JIT backend using LLVM (llvmlite).

Compiles a restricted subset of numeric bytecode into native machine code.
"""

from typing import Optional, Callable, List, Tuple, Dict, Any
import os

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
        trace_enabled = self.debug or os.environ.get("ZEXUS_NATIVE_JIT_TRACE", "0") in ("1", "true", "yes")
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
            "JUMP_IF_TRUE",
        }
        object_ops = {
            "CALL_NAME",
            "CALL_FUNC_CONST",
            "CALL_METHOD",
            "CALL_TOP",
            "CALL_BUILTIN",
            "LOAD_REG",
            "LOAD_VAR_REG",
            "STORE_REG",
            "MOV_REG",
            "ADD_REG",
            "SUB_REG",
            "MUL_REG",
            "DIV_REG",
            "MOD_REG",
            "POW_REG",
            "NEG_REG",
            "EQ_REG",
            "NEQ_REG",
            "LT_REG",
            "GT_REG",
            "LTE_REG",
            "GTE_REG",
            "AND_REG",
            "OR_REG",
            "NOT_REG",
            "PUSH_REG",
            "POP_REG",
            "BUILD_LIST",
            "BUILD_MAP",
            "BUILD_SET",
            "INDEX",
            "SLICE",
            "GET_ATTR",
            "GET_LENGTH",
            "PRINT",
            "READ",
            "WRITE",
            "IMPORT",
            "EXPORT",
            "STORE_FUNC",
            "STORE_CONST",
            "POW",
            "SPAWN",
            "AWAIT",
            "SPAWN_CALL",
            "REGISTER_EVENT",
            "EMIT_EVENT",
            "HASH_BLOCK",
            "VERIFY_SIGNATURE",
            "MERKLE_ROOT",
            "STATE_READ",
            "STATE_WRITE",
            "TX_BEGIN",
            "TX_COMMIT",
            "TX_REVERT",
            "GAS_CHARGE",
            "REQUIRE",
            "LEDGER_APPEND",
            "DEFINE_ENUM",
            "DEFINE_PROTOCOL",
            "ASSERT_PROTOCOL",
            "DEFINE_CAPABILITY",
            "GRANT_CAPABILITY",
            "REVOKE_CAPABILITY",
            "AUDIT_LOG",
            "DEFINE_SCREEN",
            "DEFINE_COMPONENT",
            "DEFINE_THEME",
            "DEFINE_CONTRACT",
            "DEFINE_ENTITY",
            "RESTRICT_ACCESS",
            "ENABLE_ERROR_MODE",
            "NOP",
            "PARALLEL_START",
            "PARALLEL_END",
            "SPAWN_TASK",
            "TASK_JOIN",
            "TASK_RESULT",
            "LOCK_ACQUIRE",
            "LOCK_RELEASE",
            "ATOMIC_ADD",
            "ATOMIC_CAS",
            "BARRIER",
            "SETUP_TRY",
            "POP_TRY",
            "THROW",
            "FOR_ITER",
        }
        use_object_mode = False
        for op_name, _ in normalized:
            if op_name in object_ops:
                use_object_mode = True
            if op_name not in supported and op_name not in object_ops:
                return None

        if trace_enabled:
            try:
                mode = "object" if use_object_mode else "numeric"
                print(f"[NATIVE JIT] compile mode={mode} instrs={len(normalized)} consts={len(consts)}")
            except Exception:
                pass

        if use_object_mode:
            func_ptr = self._build_object_function(normalized, consts)
        else:
            func_ptr = self._build_function(normalized, consts, names)
        if func_ptr is None:
            return None

        from ctypes import CFUNCTYPE, c_double, POINTER, py_object

        if use_object_mode:
            cfunc = CFUNCTYPE(py_object, py_object, py_object, py_object, POINTER(py_object))(func_ptr)
            const_objs = [c for c in consts]

            def jit_execute(vm, stack, env):
                try:
                    const_buf = (py_object * max(len(const_objs), 1))(*const_objs)
                    return cfunc(vm, env, vm.builtins, const_buf)
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

            if op_name == "PUSH_REG":
                reg_idx = int(operand) if not isinstance(operand, (list, tuple)) else int(operand[0])
                reg_slot = reg_ptr(ir.Constant(ir.IntType(32), reg_idx))
                val = builder.load(reg_slot)
                push(val)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "POP_REG":
                reg_idx = int(operand) if not isinstance(operand, (list, tuple)) else int(operand[0])
                val = pop()
                reg_slot = reg_ptr(ir.Constant(ir.IntType(32), reg_idx))
                builder.store(val, reg_slot)
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

            if op_name == "JUMP_IF_TRUE":
                cond = pop()
                zero = ir.Constant(double, 0.0)
                is_true = builder.fcmp_ordered("!=", cond, zero)
                target = int(operand) if operand is not None else idx + 1
                builder.cbranch(is_true, blocks[target], blocks[idx + 1])
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

        # Determine register file size for register ops
        max_reg = -1
        register_ops = {
            "LOAD_REG", "LOAD_VAR_REG", "STORE_REG", "MOV_REG",
            "ADD_REG", "SUB_REG", "MUL_REG", "DIV_REG", "MOD_REG", "POW_REG",
            "NEG_REG", "EQ_REG", "NEQ_REG", "LT_REG", "GT_REG", "LTE_REG", "GTE_REG",
            "AND_REG", "OR_REG", "NOT_REG", "PUSH_REG", "POP_REG",
        }
        for op_name, operand in instrs:
            if op_name in register_ops:
                if op_name in ("LOAD_REG", "LOAD_VAR_REG", "STORE_REG"):
                    if isinstance(operand, (list, tuple)) and operand and isinstance(operand[0], int):
                        max_reg = max(max_reg, operand[0])
                elif op_name == "MOV_REG":
                    if isinstance(operand, (list, tuple)) and len(operand) >= 2:
                        if isinstance(operand[0], int):
                            max_reg = max(max_reg, operand[0])
                        if isinstance(operand[1], int):
                            max_reg = max(max_reg, operand[1])
                elif op_name in ("NEG_REG", "NOT_REG"):
                    if isinstance(operand, (list, tuple)) and len(operand) >= 2:
                        if isinstance(operand[0], int):
                            max_reg = max(max_reg, operand[0])
                        if isinstance(operand[1], int):
                            max_reg = max(max_reg, operand[1])
                elif op_name in ("PUSH_REG", "POP_REG"):
                    if isinstance(operand, int):
                        max_reg = max(max_reg, operand)
                    elif isinstance(operand, (list, tuple)) and operand and isinstance(operand[0], int):
                        max_reg = max(max_reg, operand[0])
                else:
                    if isinstance(operand, (list, tuple)):
                        for item in operand:
                            if isinstance(item, int):
                                max_reg = max(max_reg, item)
        reg_count = max(1, max_reg + 1)

        func_type = ir.FunctionType(ptr, [ptr, ptr, ptr, ptr_ptr])
        func = ir.Function(ir.Module(name="zexus_native_jit_obj"), func_type, name="jit_obj_fn")
        vm_ptr, env_ptr, builtins_ptr, consts_ptr = func.args

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
        cabi_build_set = decl("zexus_build_set_from_array", ptr, [ptr, ir.IntType(64)])
        cabi_build_tuple = decl("zexus_build_tuple_from_array", ptr, [ptr, ir.IntType(64)])
        cabi_iter_next = decl("zexus_iter_next", ptr, [ptr])
        cabi_add = decl("zexus_number_add", ptr, [ptr, ptr])
        cabi_sub = decl("zexus_number_sub", ptr, [ptr, ptr])
        cabi_mul = decl("zexus_number_mul", ptr, [ptr, ptr])
        cabi_div = decl("zexus_number_div", ptr, [ptr, ptr])
        cabi_mod = decl("zexus_number_mod", ptr, [ptr, ptr])
        cabi_pow = decl("zexus_number_pow", ptr, [ptr, ptr])
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
        cabi_slice = decl("zexus_slice", ptr, [ptr, ptr, ptr])
        cabi_get_attr = decl("zexus_get_attr", ptr, [ptr, ptr])
        cabi_get_length = decl("zexus_get_length", ptr, [ptr])
        cabi_print = decl("zexus_print", ptr, [ptr])
        cabi_read = decl("zexus_read", ptr, [ptr])
        cabi_write = decl("zexus_write", ptr, [ptr, ptr])
        cabi_import = decl("zexus_import", ptr, [ptr])
        cabi_export = decl("zexus_export", ptr, [ptr, ptr, ptr])
        cabi_int_from_long = decl("zexus_int_from_long", ptr, [ir.IntType(64)])
        cabi_hash_block = decl("zexus_hash_block", ptr, [ptr])
        cabi_merkle_root = decl("zexus_merkle_root", ptr, [ptr, ir.IntType(64)])
        cabi_verify_signature = decl("zexus_verify_signature", ptr, [ptr, ptr, ptr, ptr, ptr])
        cabi_state_read = decl("zexus_state_read", ptr, [ptr, ptr])
        cabi_state_write = decl("zexus_state_write", ptr, [ptr, ptr, ptr])
        cabi_tx_begin = decl("zexus_tx_begin", ptr, [ptr])
        cabi_tx_commit = decl("zexus_tx_commit", ptr, [ptr])
        cabi_tx_revert = decl("zexus_tx_revert", ptr, [ptr])
        cabi_gas_charge = decl("zexus_gas_charge", ptr, [ptr, ptr])
        cabi_require = decl("zexus_require", ptr, [ptr, ptr, ptr])
        cabi_ledger_append = decl("zexus_ledger_append", ptr, [ptr, ptr])
        cabi_register_event = decl("zexus_register_event", ptr, [ptr, ptr, ptr])
        cabi_emit_event = decl("zexus_emit_event", ptr, [ptr, ptr, ptr, ptr, ptr])
        cabi_spawn_name = decl("zexus_spawn_name", ptr, [ptr, ptr, ptr, ptr, ptr])
        cabi_spawn_call = decl("zexus_spawn_call", ptr, [ptr, ptr, ptr])
        cabi_await = decl("zexus_await", ptr, [ptr, ptr])
        cabi_lock_acquire = decl("zexus_lock_acquire", ptr, [ptr, ptr])
        cabi_lock_release = decl("zexus_lock_release", ptr, [ptr, ptr])
        cabi_barrier_wait = decl("zexus_barrier_wait", ptr, [ptr, ptr])
        cabi_atomic_add = decl("zexus_atomic_add", ptr, [ptr, ptr, ptr])
        cabi_atomic_cas = decl("zexus_atomic_cas", ptr, [ptr, ptr, ptr, ptr])
        cabi_get_iter = decl("zexus_get_iter", ptr, [ptr])
        cabi_iter_next_pair = decl("zexus_iter_next_pair", ptr, [ptr])
        cabi_atomic_add = decl("zexus_atomic_add", ptr, [ptr, ptr, ptr])
        cabi_atomic_cas = decl("zexus_atomic_cas", ptr, [ptr, ptr, ptr, ptr])
        cabi_barrier_wait = decl("zexus_barrier_wait", ptr, [ptr, ptr])
        cabi_define_enum = decl("zexus_define_enum", ptr, [ptr, ptr, ptr])
        cabi_define_protocol = decl("zexus_define_protocol", ptr, [ptr, ptr, ptr])
        cabi_assert_protocol = decl("zexus_assert_protocol", ptr, [ptr, ptr, ptr])
        cabi_define_capability = decl("zexus_define_capability", ptr, [ptr, ptr, ptr])
        cabi_define_screen = decl("zexus_define_screen", ptr, [ptr, ptr, ptr])
        cabi_define_component = decl("zexus_define_component", ptr, [ptr, ptr, ptr])
        cabi_define_theme = decl("zexus_define_theme", ptr, [ptr, ptr, ptr])
        cabi_grant_capability = decl("zexus_grant_capability", ptr, [ptr, ptr, ptr, ir.IntType(64)])
        cabi_revoke_capability = decl("zexus_revoke_capability", ptr, [ptr, ptr, ptr, ir.IntType(64)])
        cabi_audit_log = decl("zexus_audit_log", ptr, [ptr, ptr, ptr, ptr])
        cabi_define_contract = decl("zexus_define_contract", ptr, [ptr, ir.IntType(64), ptr])
        cabi_define_entity = decl("zexus_define_entity", ptr, [ptr, ir.IntType(64), ptr])
        cabi_restrict_access = decl("zexus_restrict_access", ptr, [ptr, ptr, ptr, ptr])
        cabi_enable_error_mode = decl("zexus_enable_error_mode", ptr, [ptr])

        blocks = [func.append_basic_block(name=f"b{i}") for i in range(len(instrs) + 1)]
        builder = ir.IRBuilder(blocks[0])

        stack_size = max(16, len(instrs) + 4)
        stack_arr = builder.alloca(ir.ArrayType(ptr, stack_size))
        stack_base = builder.gep(stack_arr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        sp = builder.alloca(ir.IntType(32))
        builder.store(ir.Constant(ir.IntType(32), 0), sp)

        try_stack_size = max(4, len(instrs) + 2)
        try_stack_arr = builder.alloca(ir.ArrayType(ir.IntType(32), try_stack_size))
        try_stack_base = builder.gep(try_stack_arr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        try_sp = builder.alloca(ir.IntType(32))
        builder.store(ir.Constant(ir.IntType(32), 0), try_sp)

        reg_arr = builder.alloca(ir.ArrayType(ptr, reg_count))
        reg_base = builder.gep(reg_arr, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)])
        null_ptr = ir.Constant(ptr, 0)
        for r in range(reg_count):
            reg_slot = builder.gep(reg_base, [ir.Constant(ir.IntType(32), r)])
            builder.store(null_ptr, reg_slot)

        def push(val):
            cur_sp = builder.load(sp)
            ptr_slot = builder.gep(stack_base, [cur_sp])
            builder.store(val, ptr_slot)
            builder.store(builder.add(cur_sp, ir.Constant(ir.IntType(32), 1)), sp)

        def push_with(bld, val):
            cur_sp = bld.load(sp)
            ptr_slot = bld.gep(stack_base, [cur_sp])
            bld.store(val, ptr_slot)
            bld.store(bld.add(cur_sp, ir.Constant(ir.IntType(32), 1)), sp)

        def pop():
            cur_sp = builder.load(sp)
            new_sp = builder.sub(cur_sp, ir.Constant(ir.IntType(32), 1))
            builder.store(new_sp, sp)
            ptr_slot = builder.gep(stack_base, [new_sp])
            return builder.load(ptr_slot)

        def const_obj(idx):
            return builder.load(builder.gep(consts_ptr, [ir.Constant(ir.IntType(32), idx)]))

        def reg_ptr(idx_val):
            return builder.gep(reg_base, [idx_val])

        def try_push(handler_idx):
            cur_sp = builder.load(try_sp)
            slot = builder.gep(try_stack_base, [cur_sp])
            builder.store(handler_idx, slot)
            builder.store(builder.add(cur_sp, ir.Constant(ir.IntType(32), 1)), try_sp)

        def try_pop():
            cur_sp = builder.load(try_sp)
            new_sp = builder.sub(cur_sp, ir.Constant(ir.IntType(32), 1))
            builder.store(new_sp, try_sp)
            slot = builder.gep(try_stack_base, [new_sp])
            return builder.load(slot)

        def try_pop_with(bld):
            cur_sp = bld.load(try_sp)
            new_sp = bld.sub(cur_sp, ir.Constant(ir.IntType(32), 1))
            bld.store(new_sp, try_sp)
            slot = bld.gep(try_stack_base, [new_sp])
            return bld.load(slot)

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

            if op_name == "STORE_CONST":
                if isinstance(operand, (tuple, list)) and len(operand) == 2:
                    name_idx = operand[0]
                    value_idx = operand[1]
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    value_obj = const_obj(int(value_idx)) if isinstance(value_idx, int) else ir.Constant(ptr, 0)
                    builder.call(cabi_env_set, [env_ptr, name_obj, value_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "STORE_FUNC":
                name_idx, func_idx = operand
                name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                func_obj = const_obj(int(func_idx)) if isinstance(func_idx, int) else ir.Constant(ptr, 0)
                builder.call(cabi_env_set, [env_ptr, name_obj, func_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "LOAD_REG":
                reg_idx, const_idx = operand
                reg_slot = reg_ptr(ir.Constant(ir.IntType(32), int(reg_idx)))
                value_obj = const_obj(int(const_idx)) if isinstance(const_idx, int) else ir.Constant(ptr, 0)
                builder.store(value_obj, reg_slot)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "LOAD_VAR_REG":
                reg_idx, name_idx = operand
                name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                val = builder.call(cabi_env_get, [env_ptr, name_obj])
                reg_slot = reg_ptr(ir.Constant(ir.IntType(32), int(reg_idx)))
                builder.store(val, reg_slot)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "STORE_REG":
                reg_idx, name_idx = operand
                reg_slot = reg_ptr(ir.Constant(ir.IntType(32), int(reg_idx)))
                val = builder.load(reg_slot)
                name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                builder.call(cabi_env_set, [env_ptr, name_obj, val])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "MOV_REG":
                dest_idx, src_idx = operand
                src_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src_idx)))
                dest_slot = reg_ptr(ir.Constant(ir.IntType(32), int(dest_idx)))
                val = builder.load(src_slot)
                builder.store(val, dest_slot)
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("ADD_REG", "SUB_REG", "MUL_REG", "DIV_REG", "MOD_REG"):
                dest_idx, src1_idx, src2_idx = operand
                src1_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src1_idx)))
                src2_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src2_idx)))
                dest_slot = reg_ptr(ir.Constant(ir.IntType(32), int(dest_idx)))
                v1 = builder.load(src1_slot)
                v2 = builder.load(src2_slot)
                if op_name == "ADD_REG":
                    res = builder.call(cabi_add, [v1, v2])
                elif op_name == "SUB_REG":
                    res = builder.call(cabi_sub, [v1, v2])
                elif op_name == "MUL_REG":
                    res = builder.call(cabi_mul, [v1, v2])
                elif op_name == "DIV_REG":
                    res = builder.call(cabi_div, [v1, v2])
                else:
                    res = builder.call(cabi_mod, [v1, v2])
                builder.store(res, dest_slot)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "POW_REG":
                dest_idx, src1_idx, src2_idx = operand
                src1_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src1_idx)))
                src2_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src2_idx)))
                dest_slot = reg_ptr(ir.Constant(ir.IntType(32), int(dest_idx)))
                v1 = builder.load(src1_slot)
                v2 = builder.load(src2_slot)
                res = builder.call(cabi_pow, [v1, v2])
                builder.store(res, dest_slot)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "NEG_REG":
                dest_idx, src_idx = operand
                src_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src_idx)))
                dest_slot = reg_ptr(ir.Constant(ir.IntType(32), int(dest_idx)))
                v1 = builder.load(src_slot)
                res = builder.call(cabi_neg, [v1])
                builder.store(res, dest_slot)
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("EQ_REG", "NEQ_REG", "LT_REG"):
                dest_idx, src1_idx, src2_idx = operand
                src1_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src1_idx)))
                src2_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src2_idx)))
                dest_slot = reg_ptr(ir.Constant(ir.IntType(32), int(dest_idx)))
                v1 = builder.load(src1_slot)
                v2 = builder.load(src2_slot)
                if op_name == "EQ_REG":
                    res = builder.call(cabi_eq, [v1, v2])
                elif op_name == "NEQ_REG":
                    res = builder.call(cabi_neq, [v1, v2])
                else:
                    res = builder.call(cabi_lt, [v1, v2])
                builder.store(res, dest_slot)
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("GT_REG", "LTE_REG", "GTE_REG"):
                dest_idx, src1_idx, src2_idx = operand
                src1_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src1_idx)))
                src2_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src2_idx)))
                dest_slot = reg_ptr(ir.Constant(ir.IntType(32), int(dest_idx)))
                v1 = builder.load(src1_slot)
                v2 = builder.load(src2_slot)
                if op_name == "GT_REG":
                    res = builder.call(cabi_gt, [v1, v2])
                elif op_name == "LTE_REG":
                    res = builder.call(cabi_lte, [v1, v2])
                else:
                    res = builder.call(cabi_gte, [v1, v2])
                builder.store(res, dest_slot)
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("AND_REG", "OR_REG"):
                dest_idx, src1_idx, src2_idx = operand
                src1_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src1_idx)))
                src2_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src2_idx)))
                dest_slot = reg_ptr(ir.Constant(ir.IntType(32), int(dest_idx)))
                v1 = builder.load(src1_slot)
                v2 = builder.load(src2_slot)
                if op_name == "AND_REG":
                    res = builder.call(cabi_bool_and, [v1, v2])
                else:
                    res = builder.call(cabi_bool_or, [v1, v2])
                builder.store(res, dest_slot)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "NOT_REG":
                dest_idx, src_idx = operand
                src_slot = reg_ptr(ir.Constant(ir.IntType(32), int(src_idx)))
                dest_slot = reg_ptr(ir.Constant(ir.IntType(32), int(dest_idx)))
                v1 = builder.load(src_slot)
                res = builder.call(cabi_not, [v1])
                builder.store(res, dest_slot)
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

            if op_name == "POW":
                b = pop()
                a = pop()
                res = builder.call(cabi_pow, [a, b])
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

            if op_name == "JUMP_IF_TRUE":
                cond = pop()
                truth = builder.call(cabi_truthy, [cond])
                zero = ir.Constant(ir.IntType(32), 0)
                is_true = builder.icmp_signed("!=", truth, zero)
                target = int(operand) if operand is not None else idx + 1
                builder.cbranch(is_true, blocks[target], blocks[idx + 1])
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

            if op_name == "CALL_FUNC_CONST":
                func_idx, arg_count = operand
                func_obj = const_obj(int(func_idx)) if isinstance(func_idx, int) else ir.Constant(ptr, 0)
                count_val = ir.Constant(ir.IntType(64), int(arg_count))
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), int(arg_count)))
                args_ptr = builder.gep(stack_base, [base])
                args_ptr_cast = builder.bitcast(args_ptr, ptr)
                args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                builder.store(base, sp)
                res = builder.call(cabi_call_callable, [func_obj, args_tuple])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "CALL_BUILTIN":
                name_idx, arg_count = operand
                name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                count_val = ir.Constant(ir.IntType(64), int(arg_count))
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), int(arg_count)))
                args_ptr = builder.gep(stack_base, [base])
                args_ptr_cast = builder.bitcast(args_ptr, ptr)
                args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                builder.store(base, sp)
                null_env = ir.Constant(ptr, 0)
                res = builder.call(cabi_call_name, [null_env, builtins_ptr, name_obj, args_tuple])
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

            if op_name == "SETUP_TRY":
                handler = int(operand) if operand is not None else idx + 1
                try_push(ir.Constant(ir.IntType(32), handler))
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "POP_TRY":
                _ = try_pop()
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "THROW":
                exc = pop()
                cur_sp = builder.load(try_sp)
                has_handler = builder.icmp_unsigned("!=", cur_sp, ir.Constant(ir.IntType(32), 0))
                handler_block = func.append_basic_block(name=f"throw_handler_{idx}")
                no_handler_block = func.append_basic_block(name=f"throw_unhandled_{idx}")
                builder.cbranch(has_handler, handler_block, no_handler_block)

                handler_builder = ir.IRBuilder(handler_block)
                handler_idx = try_pop_with(handler_builder)
                push_with(handler_builder, exc)
                switch_inst = handler_builder.switch(handler_idx, blocks[idx + 1])
                for case_idx, block in enumerate(blocks):
                    switch_inst.add_case(ir.Constant(ir.IntType(32), case_idx), block)

                no_handler_builder = ir.IRBuilder(no_handler_block)
                no_handler_builder.ret(exc)
                continue

            if op_name == "FOR_ITER":
                target = int(operand) if operand is not None else idx + 1
                iter_obj = pop()
                iter_obj = builder.call(cabi_get_iter, [iter_obj])
                pair = builder.call(cabi_iter_next_pair, [iter_obj])
                one_obj = builder.call(cabi_int_from_long, [ir.Constant(ir.IntType(64), 1)])
                zero_obj = builder.call(cabi_int_from_long, [ir.Constant(ir.IntType(64), 0)])
                has_flag = builder.call(cabi_index, [pair, one_obj])
                truth = builder.call(cabi_truthy, [has_flag])
                is_true = builder.icmp_signed("!=", truth, ir.Constant(ir.IntType(32), 0))
                has_block = func.append_basic_block(name=f"for_has_{idx}")
                done_block = func.append_basic_block(name=f"for_done_{idx}")
                builder.cbranch(is_true, has_block, done_block)

                has_builder = ir.IRBuilder(has_block)
                val = has_builder.call(cabi_index, [pair, zero_obj])
                push_with(has_builder, iter_obj)
                push_with(has_builder, val)
                has_builder.branch(blocks[idx + 1])

                done_builder = ir.IRBuilder(done_block)
                done_builder.branch(blocks[target])
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

            if op_name == "BUILD_SET":
                count = int(operand) if operand is not None else 0
                count_val = ir.Constant(ir.IntType(64), count)
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), count))
                items_ptr = builder.gep(stack_base, [base])
                items_ptr_cast = builder.bitcast(items_ptr, ptr)
                st = builder.call(cabi_build_set, [items_ptr_cast, count_val])
                builder.store(base, sp)
                push(st)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "INDEX":
                idx_val = pop()
                obj_val = pop()
                res = builder.call(cabi_index, [obj_val, idx_val])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "SLICE":
                count = int(operand) if operand is not None else 2
                end_val = pop() if count >= 3 else ir.Constant(ptr, None)
                start_val = pop()
                obj_val = pop()
                res = builder.call(cabi_slice, [obj_val, start_val, end_val])
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

            if op_name == "PRINT":
                val = pop()
                _ = builder.call(cabi_print, [val])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "READ":
                path = pop()
                res = builder.call(cabi_read, [path])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "WRITE":
                content = pop()
                path = pop()
                _ = builder.call(cabi_write, [path, content])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "IMPORT":
                if isinstance(operand, (list, tuple)) and operand:
                    name_idx = operand[0]
                    alias_idx = operand[1] if len(operand) > 1 else None
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    alias_obj = const_obj(int(alias_idx)) if isinstance(alias_idx, int) else name_obj
                    res = builder.call(cabi_import, [name_obj])
                    builder.call(cabi_env_set, [env_ptr, alias_obj, res])
                else:
                    name = pop()
                    res = builder.call(cabi_import, [name])
                    push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "EXPORT":
                if isinstance(operand, (list, tuple)) and operand:
                    name_idx = operand[0]
                    value_idx = operand[1] if len(operand) > 1 else None
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    if isinstance(value_idx, int):
                        value_obj = const_obj(int(value_idx))
                    else:
                        value_obj = pop()
                    builder.call(cabi_export, [env_ptr, name_obj, value_obj])
                elif isinstance(operand, int):
                    name_obj = const_obj(int(operand))
                    builder.call(cabi_export, [env_ptr, name_obj, ir.Constant(ptr, 0)])
                else:
                    value_obj = pop()
                    name_obj = pop()
                    builder.call(cabi_export, [env_ptr, name_obj, value_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "SPAWN":
                if isinstance(operand, tuple) and len(operand) >= 3 and operand[0] == "CALL":
                    name_idx = operand[1]
                    arg_count = int(operand[2])
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    res = builder.call(cabi_spawn_name, [vm_ptr, env_ptr, builtins_ptr, name_obj, args_tuple])
                    push(res)
                else:
                    arg_count = int(operand) if operand is not None else 0
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    callable_obj = pop()
                    res = builder.call(cabi_spawn_call, [vm_ptr, callable_obj, args_tuple])
                    push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "SPAWN_CALL":
                if isinstance(operand, tuple) and len(operand) >= 2 and operand[0] == "CALL_NAME":
                    name_idx = operand[1]
                    arg_count = int(operand[2]) if len(operand) > 2 else 0
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    res = builder.call(cabi_spawn_name, [vm_ptr, env_ptr, builtins_ptr, name_obj, args_tuple])
                    push(res)
                elif isinstance(operand, tuple) and len(operand) >= 2 and operand[0] == "CALL_BUILTIN":
                    name_idx = operand[1]
                    arg_count = int(operand[2]) if len(operand) > 2 else 0
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    res = builder.call(cabi_spawn_name, [vm_ptr, env_ptr, builtins_ptr, name_obj, args_tuple])
                    push(res)
                elif isinstance(operand, tuple) and len(operand) >= 2 and operand[0] == "CALL_FUNC_CONST":
                    func_idx = operand[1]
                    arg_count = int(operand[2]) if len(operand) > 2 else 0
                    func_obj = const_obj(int(func_idx)) if isinstance(func_idx, int) else ir.Constant(ptr, 0)
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    res = builder.call(cabi_spawn_call, [vm_ptr, func_obj, args_tuple])
                    push(res)
                elif isinstance(operand, tuple) and len(operand) >= 2 and operand[0] == "CALL_TOP":
                    arg_count = int(operand[1]) if len(operand) > 1 else 0
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    callable_obj = pop()
                    res = builder.call(cabi_spawn_call, [vm_ptr, callable_obj, args_tuple])
                    push(res)
                else:
                    arg_count = int(operand) if operand is not None else 0
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    callable_obj = pop()
                    res = builder.call(cabi_spawn_call, [vm_ptr, callable_obj, args_tuple])
                    push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "SPAWN_TASK":
                if isinstance(operand, tuple) and len(operand) >= 2 and operand[0] == "CALL_NAME":
                    name_idx = operand[1]
                    arg_count = int(operand[2]) if len(operand) > 2 else 0
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    res = builder.call(cabi_spawn_name, [vm_ptr, env_ptr, builtins_ptr, name_obj, args_tuple])
                    push(res)
                elif isinstance(operand, tuple) and len(operand) >= 2 and operand[0] == "CALL_BUILTIN":
                    name_idx = operand[1]
                    arg_count = int(operand[2]) if len(operand) > 2 else 0
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    res = builder.call(cabi_spawn_name, [vm_ptr, env_ptr, builtins_ptr, name_obj, args_tuple])
                    push(res)
                elif isinstance(operand, tuple) and len(operand) >= 2 and operand[0] == "CALL_FUNC_CONST":
                    func_idx = operand[1]
                    arg_count = int(operand[2]) if len(operand) > 2 else 0
                    func_obj = const_obj(int(func_idx)) if isinstance(func_idx, int) else ir.Constant(ptr, 0)
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    res = builder.call(cabi_spawn_call, [vm_ptr, func_obj, args_tuple])
                    push(res)
                elif isinstance(operand, tuple) and len(operand) >= 2 and operand[0] == "CALL_TOP":
                    arg_count = int(operand[1]) if len(operand) > 1 else 0
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    callable_obj = pop()
                    res = builder.call(cabi_spawn_call, [vm_ptr, callable_obj, args_tuple])
                    push(res)
                else:
                    arg_count = int(operand) if operand is not None else 0
                    count_val = ir.Constant(ir.IntType(64), arg_count)
                    cur_sp = builder.load(sp)
                    base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), arg_count))
                    args_ptr = builder.gep(stack_base, [base])
                    args_ptr_cast = builder.bitcast(args_ptr, ptr)
                    args_tuple = builder.call(cabi_build_tuple, [args_ptr_cast, count_val])
                    builder.store(base, sp)
                    callable_obj = pop()
                    res = builder.call(cabi_spawn_call, [vm_ptr, callable_obj, args_tuple])
                    push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "TASK_JOIN":
                task_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else pop())
                _ = builder.call(cabi_await, [vm_ptr, task_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "TASK_RESULT":
                task_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else pop())
                res = builder.call(cabi_await, [vm_ptr, task_obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "AWAIT":
                obj = pop()
                res = builder.call(cabi_await, [vm_ptr, obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "REGISTER_EVENT":
                if isinstance(operand, (list, tuple)):
                    name_idx = operand[0] if len(operand) > 0 else None
                    handler_idx = operand[1] if len(operand) > 1 else None
                else:
                    name_idx = operand
                    handler_idx = None
                name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                handler_obj = const_obj(int(handler_idx)) if isinstance(handler_idx, int) else ir.Constant(ptr, 0)
                builder.call(cabi_register_event, [vm_ptr, name_obj, handler_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "EMIT_EVENT":
                if isinstance(operand, (list, tuple)):
                    name_idx = operand[0] if len(operand) > 0 else None
                    payload_idx = operand[1] if len(operand) > 1 else None
                else:
                    name_idx = operand
                    payload_idx = None
                name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                if isinstance(payload_idx, int):
                    payload_obj = const_obj(int(payload_idx))
                    builder.call(cabi_emit_event, [vm_ptr, env_ptr, builtins_ptr, name_obj, payload_obj])
                    builder.branch(blocks[idx + 1])
                    continue

                cur_sp = builder.load(sp)
                has_payload = builder.icmp_signed("!=", cur_sp, ir.Constant(ir.IntType(32), 0))
                has_block = func.append_basic_block(name=f"emit_has_payload_{idx}")
                none_block = func.append_basic_block(name=f"emit_no_payload_{idx}")
                merge_block = func.append_basic_block(name=f"emit_merge_{idx}")
                builder.cbranch(has_payload, has_block, none_block)

                has_builder = ir.IRBuilder(has_block)
                payload_val = None
                # pop in has payload path
                has_cur_sp = has_builder.load(sp)
                has_new_sp = has_builder.sub(has_cur_sp, ir.Constant(ir.IntType(32), 1))
                has_builder.store(has_new_sp, sp)
                has_ptr_slot = has_builder.gep(stack_base, [has_new_sp])
                payload_val = has_builder.load(has_ptr_slot)
                has_builder.branch(merge_block)

                none_builder = ir.IRBuilder(none_block)
                none_builder.branch(merge_block)

                merge_builder = ir.IRBuilder(merge_block)
                payload_phi = merge_builder.phi(ptr)
                payload_phi.add_incoming(payload_val, has_block)
                payload_phi.add_incoming(ir.Constant(ptr, 0), none_block)
                merge_builder.call(cabi_emit_event, [vm_ptr, env_ptr, builtins_ptr, name_obj, payload_phi])
                merge_builder.branch(blocks[idx + 1])
                continue

            if op_name == "HASH_BLOCK":
                obj = pop()
                res = builder.call(cabi_hash_block, [obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "VERIFY_SIGNATURE":
                pk = pop()
                msg = pop()
                sig = pop()
                res = builder.call(cabi_verify_signature, [env_ptr, builtins_ptr, sig, msg, pk])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "MERKLE_ROOT":
                count = int(operand) if operand is not None else 0
                count_val = ir.Constant(ir.IntType(64), count)
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), count))
                items_ptr = builder.gep(stack_base, [base])
                items_ptr_cast = builder.bitcast(items_ptr, ptr)
                res = builder.call(cabi_merkle_root, [items_ptr_cast, count_val])
                builder.store(base, sp)
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "STATE_READ":
                key_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else ir.Constant(ptr, 0))
                res = builder.call(cabi_state_read, [env_ptr, key_obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "STATE_WRITE":
                val = pop()
                key_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else ir.Constant(ptr, 0))
                builder.call(cabi_state_write, [env_ptr, key_obj, val])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "TX_BEGIN":
                builder.call(cabi_tx_begin, [env_ptr])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "TX_COMMIT":
                builder.call(cabi_tx_commit, [env_ptr])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "TX_REVERT":
                builder.call(cabi_tx_revert, [env_ptr])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "GAS_CHARGE":
                if isinstance(operand, int):
                    amount_obj = builder.call(cabi_int_from_long, [ir.Constant(ir.IntType(64), int(operand))])
                else:
                    amount_obj = pop()
                res = builder.call(cabi_gas_charge, [env_ptr, amount_obj])
                null_ptr = ir.Constant(ptr, None)
                has_err = builder.icmp_unsigned("!=", res, null_ptr)
                err_block = func.append_basic_block(name=f"gas_err_{idx}")
                builder.cbranch(has_err, err_block, blocks[idx + 1])
                err_builder = ir.IRBuilder(err_block)
                err_builder.ret(res)
                continue

            if op_name == "REQUIRE":
                msg_obj = None
                if isinstance(operand, int):
                    msg_obj = const_obj(int(operand))
                else:
                    msg_obj = pop()
                cond_obj = pop()
                res = builder.call(cabi_require, [env_ptr, cond_obj, msg_obj])
                null_ptr = ir.Constant(ptr, None)
                has_err = builder.icmp_unsigned("!=", res, null_ptr)
                err_block = func.append_basic_block(name=f"req_err_{idx}")
                builder.cbranch(has_err, err_block, blocks[idx + 1])
                err_builder = ir.IRBuilder(err_block)
                err_builder.ret(res)
                continue

            if op_name == "LEDGER_APPEND":
                entry = pop()
                builder.call(cabi_ledger_append, [env_ptr, entry])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DEFINE_ENUM":
                if isinstance(operand, (list, tuple)) and len(operand) >= 2:
                    name_idx = operand[0]
                    spec_idx = operand[1]
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    spec_obj = const_obj(int(spec_idx)) if isinstance(spec_idx, int) else ir.Constant(ptr, 0)
                    builder.call(cabi_define_enum, [env_ptr, name_obj, spec_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DEFINE_SCREEN":
                if isinstance(operand, (list, tuple)) and len(operand) >= 2:
                    name_idx = operand[0]
                    props_idx = operand[1]
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    props_obj = const_obj(int(props_idx)) if isinstance(props_idx, int) else ir.Constant(ptr, 0)
                else:
                    props_obj = pop()
                    name_obj = pop()
                builder.call(cabi_define_screen, [env_ptr, name_obj, props_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DEFINE_COMPONENT":
                if isinstance(operand, (list, tuple)) and len(operand) >= 2:
                    name_idx = operand[0]
                    props_idx = operand[1]
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    props_obj = const_obj(int(props_idx)) if isinstance(props_idx, int) else ir.Constant(ptr, 0)
                else:
                    props_obj = pop()
                    name_obj = pop()
                builder.call(cabi_define_component, [env_ptr, name_obj, props_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DEFINE_THEME":
                if isinstance(operand, (list, tuple)) and len(operand) >= 2:
                    name_idx = operand[0]
                    props_idx = operand[1]
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    props_obj = const_obj(int(props_idx)) if isinstance(props_idx, int) else ir.Constant(ptr, 0)
                else:
                    props_obj = pop()
                    name_obj = pop()
                builder.call(cabi_define_theme, [env_ptr, name_obj, props_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DEFINE_PROTOCOL":
                if isinstance(operand, (list, tuple)) and len(operand) >= 2:
                    name_idx = operand[0]
                    spec_idx = operand[1]
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    spec_obj = const_obj(int(spec_idx)) if isinstance(spec_idx, int) else ir.Constant(ptr, 0)
                    builder.call(cabi_define_protocol, [env_ptr, name_obj, spec_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "ASSERT_PROTOCOL":
                if isinstance(operand, (list, tuple)) and len(operand) >= 2:
                    name_idx = operand[0]
                    spec_idx = operand[1]
                    name_obj = const_obj(int(name_idx)) if isinstance(name_idx, int) else ir.Constant(ptr, 0)
                    spec_obj = const_obj(int(spec_idx)) if isinstance(spec_idx, int) else ir.Constant(ptr, 0)
                    res = builder.call(cabi_assert_protocol, [env_ptr, name_obj, spec_obj])
                    push(res)
                else:
                    push(ir.Constant(ptr, 0))
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DEFINE_CAPABILITY":
                name_obj = pop()
                definition = pop()
                builder.call(cabi_define_capability, [env_ptr, name_obj, definition])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "GRANT_CAPABILITY":
                count = int(operand) if operand is not None else 0
                count_val = ir.Constant(ir.IntType(64), count)
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), count))
                items_ptr = builder.gep(stack_base, [base])
                items_ptr_cast = builder.bitcast(items_ptr, ptr)
                builder.store(base, sp)
                entity_name = pop()
                builder.call(cabi_grant_capability, [env_ptr, entity_name, items_ptr_cast, count_val])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "REVOKE_CAPABILITY":
                count = int(operand) if operand is not None else 0
                count_val = ir.Constant(ir.IntType(64), count)
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, ir.Constant(ir.IntType(32), count))
                items_ptr = builder.gep(stack_base, [base])
                items_ptr_cast = builder.bitcast(items_ptr, ptr)
                builder.store(base, sp)
                entity_name = pop()
                builder.call(cabi_revoke_capability, [env_ptr, entity_name, items_ptr_cast, count_val])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "AUDIT_LOG":
                ts = pop()
                action = pop()
                data = pop()
                builder.call(cabi_audit_log, [env_ptr, ts, action, data])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DEFINE_CONTRACT":
                count = int(operand) if operand is not None else 0
                count_val = ir.Constant(ir.IntType(64), count)
                pair_count = ir.Constant(ir.IntType(32), count * 2)
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, pair_count)
                items_ptr = builder.gep(stack_base, [base])
                items_ptr_cast = builder.bitcast(items_ptr, ptr)
                builder.store(base, sp)
                name_obj = pop()
                res = builder.call(cabi_define_contract, [items_ptr_cast, count_val, name_obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "DEFINE_ENTITY":
                count = int(operand) if operand is not None else 0
                count_val = ir.Constant(ir.IntType(64), count)
                pair_count = ir.Constant(ir.IntType(32), count * 2)
                cur_sp = builder.load(sp)
                base = builder.sub(cur_sp, pair_count)
                items_ptr = builder.gep(stack_base, [base])
                items_ptr_cast = builder.bitcast(items_ptr, ptr)
                builder.store(base, sp)
                name_obj = pop()
                res = builder.call(cabi_define_entity, [items_ptr_cast, count_val, name_obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "RESTRICT_ACCESS":
                restriction = pop()
                prop = pop()
                obj = pop()
                builder.call(cabi_restrict_access, [env_ptr, obj, prop, restriction])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "ENABLE_ERROR_MODE":
                builder.call(cabi_enable_error_mode, [env_ptr])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "NOP":
                builder.branch(blocks[idx + 1])
                continue

            if op_name in ("PARALLEL_START", "PARALLEL_END"):
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "LOCK_ACQUIRE":
                key_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else pop())
                builder.call(cabi_lock_acquire, [env_ptr, key_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "LOCK_RELEASE":
                key_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else pop())
                builder.call(cabi_lock_release, [env_ptr, key_obj])
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "BARRIER":
                barrier_obj = pop()
                timeout_obj = None
                if operand is not None and isinstance(operand, int):
                    timeout_obj = const_obj(int(operand))
                else:
                    timeout_obj = ir.Constant(ptr, 0)
                res = builder.call(cabi_barrier_wait, [barrier_obj, timeout_obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "ATOMIC_ADD":
                delta = pop()
                key_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else pop())
                res = builder.call(cabi_atomic_add, [env_ptr, key_obj, delta])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "ATOMIC_CAS":
                new_val = pop()
                expected = pop()
                key_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else pop())
                res = builder.call(cabi_atomic_cas, [env_ptr, key_obj, expected, new_val])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "ATOMIC_ADD":
                delta_obj = pop()
                key_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else pop())
                res = builder.call(cabi_atomic_add, [env_ptr, key_obj, delta_obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "ATOMIC_CAS":
                new_obj = pop()
                expected_obj = pop()
                key_obj = pop() if operand is None else (const_obj(int(operand)) if isinstance(operand, int) else pop())
                res = builder.call(cabi_atomic_cas, [env_ptr, key_obj, expected_obj, new_obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "BARRIER":
                timeout_obj = ir.Constant(ptr, 0)
                if operand is not None and isinstance(operand, int):
                    timeout_obj = const_obj(int(operand))
                barrier_obj = pop()
                res = builder.call(cabi_barrier_wait, [barrier_obj, timeout_obj])
                push(res)
                builder.branch(blocks[idx + 1])
                continue

            if op_name == "FOR_ITER":
                cur_sp = builder.load(sp)
                top_idx = builder.sub(cur_sp, ir.Constant(ir.IntType(32), 1))
                iter_ptr = builder.gep(stack_base, [top_idx])
                iter_obj = builder.load(iter_ptr)
                res = builder.call(cabi_iter_next, [iter_obj])
                null_ptr = ir.Constant(ptr, 0)
                is_done = builder.icmp_unsigned("==", res, null_ptr)
                done_block = func.append_basic_block(name=f"for_done_{idx}")
                cont_block = func.append_basic_block(name=f"for_cont_{idx}")
                builder.cbranch(is_done, done_block, cont_block)

                done_builder = ir.IRBuilder(done_block)
                done_builder.store(top_idx, sp)
                target = int(operand) if operand is not None else idx + 1
                done_builder.branch(blocks[target])

                cont_builder = ir.IRBuilder(cont_block)
                push_with(cont_builder, res)
                cont_builder.branch(blocks[idx + 1])
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
