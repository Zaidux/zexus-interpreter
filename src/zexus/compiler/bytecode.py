"""
Low-level Bytecode Generator for Zexus compiler frontend.

Generates a stack-machine Bytecode object:
 - bytecode.instructions: list of (opcode, operand)
 - bytecode.constants: list of python literals / nested Bytecode function descriptors

New opcodes introduced/used:
 - LOAD_CONST (idx)     ; push constant
 - LOAD_NAME (idx)      ; push env[name] at runtime (name stored in constants)
 - STORE_NAME (idx)     ; pop and store into env[name]
 - CALL_NAME (name_idx, arg_count)   ; call function by name (env or builtins)
 - CALL_FUNC_CONST (func_const_idx, arg_count) ; call function by constant descriptor
 - RETURN
 - SPAWN_CALL (call_operand) ; spawn a call as task (call_operand is same structure as CALL_*)
 - AWAIT
 - Other control ops: JUMP, JUMP_IF_FALSE, etc.

This generator focuses on action/function lowering and call sites.
"""
from typing import List, Any, Dict, Tuple
from .zexus_ast import (
    Program, LetStatement, ExpressionStatement, PrintStatement, ReturnStatement,
    IfStatement, WhileStatement, Identifier, IntegerLiteral, StringLiteral,
    Boolean as AST_Boolean, InfixExpression, PrefixExpression, CallExpression,
    ActionStatement, BlockStatement, MapLiteral, ListLiteral, AwaitExpression,
    EnumDeclaration, ImportStatement, EventDeclaration, ProtocolDeclaration,
    EmitStatement
)

# --- Bytecode representation ---
class Bytecode:
    def __init__(self):
        self.instructions: List[Tuple[str, Any]] = []
        self.constants: List[Any] = []

    def add_instruction(self, opcode: str, operand: Any = None):
        self.instructions.append((opcode, operand))

    def add_constant(self, value: Any) -> int:
        idx = len(self.constants)
        self.constants.append(value)
        return idx

# --- Generator ---
class BytecodeGenerator:
    def __init__(self):
        self.bytecode = Bytecode()

    def generate(self, program: Program) -> Bytecode:
        self.bytecode = Bytecode()
        statements = list(getattr(program, "statements", []) or [])
        total = len(statements)
        index = 0
        while index < total:
            stmt = statements[index]

            # Pattern-match legacy import syntax: import "module" [as alias]
            if (
                isinstance(stmt, ExpressionStatement)
                and isinstance(stmt.expression, Identifier)
                and stmt.expression.value == "import"
            ):
                module_value = None
                alias_value = None
                consumed = 1

                if index + 1 < total:
                    module_stmt = statements[index + 1]
                    if isinstance(module_stmt, ExpressionStatement):
                        module_expr = getattr(module_stmt, "expression", None)
                        if isinstance(module_expr, StringLiteral):
                            module_value = module_expr.value
                            consumed = 2
                        elif isinstance(module_expr, Identifier):
                            module_value = module_expr.value
                            consumed = 2

                if module_value is not None and index + consumed < total:
                    next_stmt = statements[index + consumed]
                    if (
                        isinstance(next_stmt, ExpressionStatement)
                        and isinstance(next_stmt.expression, Identifier)
                        and next_stmt.expression.value == "as"
                        and index + consumed + 1 < total
                    ):
                        alias_stmt = statements[index + consumed + 1]
                        if isinstance(alias_stmt, ExpressionStatement) and isinstance(alias_stmt.expression, Identifier):
                            alias_value = alias_stmt.expression.value
                            consumed += 2

                if module_value is not None:
                    import_node = ImportStatement(module_path=module_value, alias=alias_value)
                    self._emit_statement(
                        import_node,
                        self.bytecode,
                        is_top_level=True,
                        is_last=(index + consumed - 1 == total - 1),
                    )
                    index += consumed
                    continue

            self._emit_statement(
                stmt,
                self.bytecode,
                is_top_level=True,
                is_last=(index == total - 1),
            )
            index += 1
        return self.bytecode

    # Statement lowering
    def _emit_statement(self, stmt, bc: Bytecode, *, is_top_level: bool = False, is_last: bool = False):
        t = type(stmt).__name__

        if t == "LetStatement":
            # Evaluate value -> push result, then STORE_NAME
            self._emit_expression(stmt.value, bc)
            name_idx = bc.add_constant(stmt.name.value)
            bc.add_instruction("STORE_NAME", name_idx)
            return

        if t == "ExpressionStatement":
            self._emit_expression(stmt.expression, bc)
            if not (is_top_level and is_last):
                bc.add_instruction("POP", None)
            return

        if t == "PrintStatement":
            self._emit_expression(stmt.value, bc)
            bc.add_instruction("PRINT", None)
            return

        if t == "ReturnStatement":
            self._emit_expression(stmt.return_value, bc)
            bc.add_instruction("RETURN", None)
            return

        if t == "ActionStatement":
            # Compile action body into a nested Bytecode; store as function descriptor constant
            func_bc = Bytecode()
            # compile body: we expect BlockStatement
            for s in getattr(stmt.body, "statements", []):
                self._emit_statement(s, func_bc)
            # ensure function returns (implicit)
            func_bc.add_instruction("RETURN", None)
            # function descriptor: dict with bytecode, params list, is_async flag
            params = [p.value for p in getattr(stmt, "parameters", [])]
            func_desc = {"bytecode": func_bc, "params": params, "is_async": getattr(stmt, "is_async", False)}
            func_const_idx = bc.add_constant(func_desc)
            # store function descriptor into environment under name
            name_idx = bc.add_constant(stmt.name.value)
            # STORE_FUNC: operand (name_idx, func_const_idx)
            bc.add_instruction("STORE_FUNC", (name_idx, func_const_idx))
            return

        if t == "IfStatement":
            # Very basic lowering: condition, JUMP_IF_FALSE to else/start, consequence, [else], ...
            self._emit_expression(stmt.condition, bc)
            # placeholder jump; compute positions
            jump_pos = len(bc.instructions)
            bc.add_instruction("JUMP_IF_FALSE", None)
            # consequence
            for s in getattr(stmt.consequence, "statements", []):
                self._emit_statement(s, bc)
            # update jump to after consequence
            end_pos = len(bc.instructions)
            bc.instructions[jump_pos] = ("JUMP_IF_FALSE", end_pos)
            return

        if t == "WhileStatement":
            start_pos = len(bc.instructions)
            self._emit_expression(stmt.condition, bc)
            # placeholder jump
            jump_pos = len(bc.instructions)
            bc.add_instruction("JUMP_IF_FALSE", None)
            # body
            for s in getattr(stmt.body, "statements", []):
                self._emit_statement(s, bc)
            # loop back
            bc.add_instruction("JUMP", start_pos)
            end_pos = len(bc.instructions)
            bc.instructions[jump_pos] = ("JUMP_IF_FALSE", end_pos)
            return

        if t == "EnumDeclaration":
            enum_name = getattr(stmt.name, "value", stmt.name)
            members = getattr(stmt, "members", {}) or {}
            resolved_members = {}
            next_auto = 0
            for key, explicit in members.items():
                value = explicit if explicit is not None else next_auto
                resolved_members[key] = value
                next_auto = (value + 1) if explicit is not None else (next_auto + 1)

            name_idx = bc.add_constant(enum_name)
            members_idx = bc.add_constant(resolved_members)
            bc.add_instruction("DEFINE_ENUM", (name_idx, members_idx))
            return

        if t == "ImportStatement":
            module_idx = bc.add_constant(getattr(stmt, "module_path", None))
            alias = getattr(stmt, "alias", None)
            if alias is not None:
                alias_idx = bc.add_constant(alias)
                bc.add_instruction("IMPORT", (module_idx, alias_idx))
            else:
                bc.add_instruction("IMPORT", (module_idx,))
            return

        if t == "EventDeclaration":
            event_name = getattr(stmt.name, "value", stmt.name)
            name_idx = bc.add_constant(event_name)
            bc.add_instruction("REGISTER_EVENT", name_idx)
            return

        if t == "ProtocolDeclaration":
            proto_name = getattr(stmt.name, "value", stmt.name)
            spec = getattr(stmt, "spec", {}) or {}
            name_idx = bc.add_constant(proto_name)
            spec_idx = bc.add_constant(spec)
            bc.add_instruction("DEFINE_PROTOCOL", (name_idx, spec_idx))
            return

        if t == "EmitStatement":
            event_name = getattr(stmt.name, "value", stmt.name)
            name_idx = bc.add_constant(event_name)
            if getattr(stmt, "payload", None) is not None:
                self._emit_expression(stmt.payload, bc)
            bc.add_instruction("EMIT_EVENT", (name_idx,))
            return

        if t == "StreamStatement":
            # Streams are currently handled at runtime only; compiler no-op.
            return

        if t == "WatchStatement":
            # Watch statements are runtime constructs; compiler emits no bytecode.
            return

        return

    # Expression lowering
    def _emit_expression(self, expr, bc: Bytecode):
        if expr is None:
            bc.add_instruction("LOAD_CONST", bc.add_constant(None))
            return

        typ = type(expr).__name__

        if typ == "IntegerLiteral":
            const_idx = bc.add_constant(expr.value)
            bc.add_instruction("LOAD_CONST", const_idx)
            return

        if typ == "FloatLiteral":
            const_idx = bc.add_constant(float(expr.value))
            bc.add_instruction("LOAD_CONST", const_idx)
            return

        if typ == "StringLiteral":
            const_idx = bc.add_constant(expr.value)
            bc.add_instruction("LOAD_CONST", const_idx)
            return

        if typ == "Identifier":
            # push variable value at runtime
            name_idx = bc.add_constant(expr.value)
            bc.add_instruction("LOAD_NAME", name_idx)
            return

        if typ == "CallExpression":
            # Evaluate arguments first (push in order)
            for arg in expr.arguments:
                self._emit_expression(arg, bc)

            # If function is an Identifier -> CALL_NAME (by name lookup at runtime)
            if isinstance(expr.function, Identifier):
                name_idx = bc.add_constant(expr.function.value)
                # operand: (name_const_idx, arg_count)
                bc.add_instruction("CALL_NAME", (name_idx, len(expr.arguments)))
                return

            # If function is a literal function descriptor (constant), emit CALL_FUNC_CONST
            if isinstance(expr.function, ActionLiteral):
                # compile inline action literal into nested func bytecode and store as constant
                # compile nested action body into func_bc
                func_bc = Bytecode()
                # lower the action literal's body statements (best-effort)
                for s in getattr(expr.function.body, "statements", []):
                    self._emit_statement(s, func_bc)
                func_bc.add_instruction("RETURN", None)
                func_desc = {"bytecode": func_bc, "params": [p.value for p in expr.function.parameters], "is_async": getattr(expr.function, "is_async", False)}
                func_const_idx = bc.add_constant(func_desc)
                bc.add_instruction("CALL_FUNC_CONST", (func_const_idx, len(expr.arguments)))
                return

            # Otherwise function expression evaluated to value on stack, call with CALL_TOP
            self._emit_expression(expr.function, bc)
            bc.add_instruction("CALL_TOP", len(expr.arguments))
            return

        # NEW: AwaitExpression lowering to CALL_* followed by AWAIT
        if typ == "AwaitExpression":
            inner = expr.expression
            # If inner is call by name, emit CALL_NAME then AWAIT
            if isinstance(inner, CallExpression) and isinstance(inner.function, Identifier):
                for arg in inner.arguments:
                    self._emit_expression(arg, bc)
                name_idx = bc.add_constant(inner.function.value)
                bc.add_instruction("CALL_NAME", (name_idx, len(inner.arguments)))
                bc.add_instruction("AWAIT", None)
                return
            # If inner is call to function expression, evaluate function then CALL_TOP then AWAIT
            if isinstance(inner, CallExpression):
                for arg in inner.arguments:
                    self._emit_expression(arg, bc)
                self._emit_expression(inner.function, bc)
                bc.add_instruction("CALL_TOP", len(inner.arguments))
                bc.add_instruction("AWAIT", None)
                return
            # generic: emit inner then AWAIT
            self._emit_expression(inner, bc)
            bc.add_instruction("AWAIT", None)
            return

        if typ == "InfixExpression":
            # Evaluate left then right then op
            self._emit_expression(expr.left, bc)
            self._emit_expression(expr.right, bc)
            op_map = {
                '+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV',
                '==': 'EQ', '!=': 'NEQ', '<': 'LT', '>': 'GT',
                '<=': 'LTE', '>=': 'GTE', '&&': 'AND', '||': 'OR'
            }
            bc.add_instruction(op_map.get(expr.operator, "UNKNOWN_OP"), None)
            return

        if typ == "PrefixExpression":
            self._emit_expression(expr.right, bc)
            if expr.operator == "!":
                bc.add_instruction("NOT", None)
            elif expr.operator == "-":
                bc.add_instruction("NEG", None)
            return

        if typ == "ListLiteral":
            # emit each element and then BUILD_LIST with count
            for el in expr.elements:
                self._emit_expression(el, bc)
            bc.add_instruction("BUILD_LIST", len(expr.elements))
            return

        if typ == "MapLiteral":
            # emit k,v pairs as literals (best-effort)
            items = {}
            for k_expr, v_expr in expr.pairs:
                # assume keys are string or identifier
                if isinstance(k_expr, StringLiteral):
                    key = k_expr.value
                elif isinstance(k_expr, Identifier):
                    key = k_expr.value
                else:
                    key = str(k_expr)
                # lower value to constant if possible
                if isinstance(v_expr, (StringLiteral, IntegerLiteral)):
                    items[key] = v_expr.value
                else:
                    # fallback: emit value and store under a temp constant (not ideal)
                    self._emit_expression(v_expr, bc)
                    # pop and store into a constant slot? Simplify: mark as None
                    items[key] = None
            const_idx = bc.add_constant(items)
            bc.add_instruction("LOAD_CONST", const_idx)
            return

        # fallback: push None
        const_idx = bc.add_constant(None)
        bc.add_instruction("LOAD_CONST", const_idx)
        return

# Backwards compatibility
Generator = BytecodeGenerator