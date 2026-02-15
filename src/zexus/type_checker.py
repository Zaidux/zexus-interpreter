"""
Static Type Checker for Zexus
=============================

Performs a single AST-walking pass **before** evaluation to detect type
errors at authoring time.  The checker intentionally operates on
*annotations only* — it will never execute user code.

Usage::

    from zexus.type_checker import StaticTypeChecker
    checker = StaticTypeChecker()
    diagnostics = checker.check(program_ast)
    for d in diagnostics:
        print(d)

Each diagnostic is a ``TypeDiagnostic`` named-tuple with ``level``
(``"error"`` or ``"warning"``), ``message``, and an optional ``node``
reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .type_system import BaseType, TypeSpec, STANDARD_TYPES, parse_type_annotation
from . import zexus_ast as ast


# ---------------------------------------------------------------------------
# Diagnostic container
# ---------------------------------------------------------------------------

@dataclass
class TypeDiagnostic:
    level: str  # "error" | "warning"
    message: str
    node: Optional[ast.Node] = None

    def __str__(self) -> str:
        prefix = "TypeError" if self.level == "error" else "TypeWarning"
        return f"[{prefix}] {self.message}"


# ---------------------------------------------------------------------------
# Scope (symbol table for one lexical scope)
# ---------------------------------------------------------------------------

@dataclass
class _Scope:
    """Tracks declared names -> TypeSpec in a single lexical scope."""
    parent: Optional["_Scope"] = None
    symbols: Dict[str, TypeSpec] = field(default_factory=dict)
    # Actions/functions: name -> (param_types, return_type)
    callables: Dict[str, Tuple[List[Tuple[str, Optional[TypeSpec]]], Optional[TypeSpec]]] = field(
        default_factory=dict
    )

    def define(self, name: str, ts: Optional[TypeSpec]):
        self.symbols[name] = ts  # type: ignore[assignment]

    def lookup(self, name: str) -> Optional[TypeSpec]:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def define_callable(
        self,
        name: str,
        params: List[Tuple[str, Optional[TypeSpec]]],
        return_type: Optional[TypeSpec],
    ):
        self.callables[name] = (params, return_type)

    def lookup_callable(
        self, name: str
    ) -> Optional[Tuple[List[Tuple[str, Optional[TypeSpec]]], Optional[TypeSpec]]]:
        if name in self.callables:
            return self.callables[name]
        if self.parent:
            return self.parent.lookup_callable(name)
        return None


# ---------------------------------------------------------------------------
# Type inference helpers
# ---------------------------------------------------------------------------

_LITERAL_TYPE_MAP = {
    ast.IntegerLiteral: BaseType.INT,
    ast.FloatLiteral: BaseType.FLOAT,
    ast.StringLiteral: BaseType.STRING,
    ast.Boolean: BaseType.BOOL,
}


def _resolve_annotation(annotation) -> Optional[TypeSpec]:
    """Convert a string or ``Identifier`` annotation to ``TypeSpec``,
    returning *None* when the annotation is absent or un-parseable."""
    if annotation is None:
        return None
    # Handle both raw strings and Identifier AST nodes
    if isinstance(annotation, ast.Identifier):
        annotation = annotation.value
    if not isinstance(annotation, str):
        annotation = str(annotation)
    return parse_type_annotation(annotation)


def _infer_expr_type(node: ast.Expression, scope: _Scope) -> Optional[TypeSpec]:
    """Best-effort inference of the static type of *node*.

    Returns ``None`` when the type cannot be determined statically.
    """
    node_cls = type(node)

    # Literals
    base = _LITERAL_TYPE_MAP.get(node_cls)
    if base is not None:
        return TypeSpec(base)

    # Array/List literal
    if isinstance(node, ast.ListLiteral):
        if node.elements:
            elem_type = _infer_expr_type(node.elements[0], scope)
            return TypeSpec(BaseType.ARRAY, array_of=elem_type)
        return TypeSpec(BaseType.ARRAY, array_of=TypeSpec(BaseType.ANY))

    # Map literal
    if isinstance(node, ast.MapLiteral):
        return TypeSpec(BaseType.OBJECT)

    # Identifier — look up in scope
    if isinstance(node, ast.Identifier):
        return scope.lookup(node.value)

    # Infix — numeric ops yield numeric types
    if isinstance(node, ast.InfixExpression):
        if node.operator in ("+", "-", "*", "/", "%", "**"):
            left = _infer_expr_type(node.left, scope)
            right = _infer_expr_type(node.right, scope)
            if left and right:
                if left.base_type == BaseType.FLOAT or right.base_type == BaseType.FLOAT:
                    return TypeSpec(BaseType.FLOAT)
                if left.base_type == BaseType.INT and right.base_type == BaseType.INT:
                    if node.operator == "/":
                        return TypeSpec(BaseType.FLOAT)
                    return TypeSpec(BaseType.INT)
                # String concatenation
                if node.operator == "+" and left.base_type == BaseType.STRING:
                    return TypeSpec(BaseType.STRING)
            return None
        if node.operator in ("==", "!=", "<", ">", "<=", ">=", "&&", "||", "and", "or"):
            return TypeSpec(BaseType.BOOL)

    # Prefix — `!` => bool, `-` => numeric
    if isinstance(node, ast.PrefixExpression):
        if node.operator == "!":
            return TypeSpec(BaseType.BOOL)
        if node.operator == "-":
            return _infer_expr_type(node.right, scope)

    # Call expression — look up return type
    if isinstance(node, ast.CallExpression):
        fn_name = None
        if isinstance(node.function, ast.Identifier):
            fn_name = node.function.value
        if fn_name:
            sig = scope.lookup_callable(fn_name)
            if sig:
                _, ret = sig
                return ret

    return None


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def _is_compatible(declared: TypeSpec, actual: TypeSpec) -> bool:
    """Check whether *actual* can be assigned to a slot of type *declared*."""
    if declared.base_type == BaseType.ANY or actual.base_type == BaseType.ANY:
        return True
    # number promotion: int -> float
    if declared.base_type == BaseType.FLOAT and actual.base_type == BaseType.INT:
        return True
    if declared.base_type != actual.base_type:
        return False
    # Array element types
    if declared.array_of and actual.array_of:
        return _is_compatible(declared.array_of, actual.array_of)
    return True


# ---------------------------------------------------------------------------
# Main checker
# ---------------------------------------------------------------------------

class StaticTypeChecker:
    """Walk an AST ``Program`` and collect type diagnostics."""

    def __init__(self) -> None:
        self.diagnostics: List[TypeDiagnostic] = []
        self._scope: _Scope = _Scope()  # global scope

    # -- public API ---------------------------------------------------------

    def check(self, program: ast.Program) -> List[TypeDiagnostic]:
        """Run the checker on *program* and return all diagnostics."""
        self.diagnostics = []
        self._scope = _Scope()
        for stmt in program.statements:
            self._check_statement(stmt)
        return list(self.diagnostics)

    # -- helpers ------------------------------------------------------------

    def _error(self, msg: str, node: Optional[ast.Node] = None):
        self.diagnostics.append(TypeDiagnostic("error", msg, node))

    def _warning(self, msg: str, node: Optional[ast.Node] = None):
        self.diagnostics.append(TypeDiagnostic("warning", msg, node))

    def _push_scope(self) -> _Scope:
        self._scope = _Scope(parent=self._scope)
        return self._scope

    def _pop_scope(self):
        if self._scope.parent:
            self._scope = self._scope.parent

    # -- statement visitors -------------------------------------------------

    def _check_statement(self, stmt: ast.Statement):
        handler = getattr(self, f"_check_{type(stmt).__name__}", None)
        if handler:
            handler(stmt)

    def _check_LetStatement(self, stmt: ast.LetStatement):
        ann = _resolve_annotation(getattr(stmt, "type_annotation", None))
        if stmt.value is not None:
            self._check_expression(stmt.value)
            actual = _infer_expr_type(stmt.value, self._scope)
            if ann and actual and not _is_compatible(ann, actual):
                self._error(
                    f"Cannot assign {actual.base_type.value} to "
                    f"'{stmt.name}' declared as {ann.base_type.value}",
                    stmt,
                )
        name = stmt.name.value if isinstance(stmt.name, ast.Identifier) else None
        if name:
            self._scope.define(name, ann)

    def _check_ConstStatement(self, stmt: ast.ConstStatement):
        ann = _resolve_annotation(getattr(stmt, "type_annotation", None))
        if stmt.value is not None:
            self._check_expression(stmt.value)
            actual = _infer_expr_type(stmt.value, self._scope)
            if ann and actual and not _is_compatible(ann, actual):
                self._error(
                    f"Cannot assign {actual.base_type.value} to "
                    f"const '{stmt.name}' declared as {ann.base_type.value}",
                    stmt,
                )
        name = stmt.name.value if isinstance(stmt.name, ast.Identifier) else None
        if name:
            self._scope.define(name, ann)

    def _check_ActionStatement(self, stmt: ast.ActionStatement):
        name = stmt.name if isinstance(stmt.name, str) else getattr(stmt.name, "value", None)
        param_types: List[Tuple[str, Optional[TypeSpec]]] = []
        for p in stmt.parameters or []:
            ann_str = getattr(p, "type_annotation", None)
            ts = _resolve_annotation(ann_str)
            param_types.append((p.value, ts))
        ret_ts = _resolve_annotation(stmt.return_type)
        if name:
            self._scope.define_callable(name, param_types, ret_ts)
            self._scope.define(name, TypeSpec(BaseType.ACTION))

        # Check body in a nested scope
        self._push_scope()
        for pname, pts in param_types:
            self._scope.define(pname, pts)
        if stmt.body:
            self._check_block(stmt.body, ret_ts)
        self._pop_scope()

    def _check_FunctionStatement(self, stmt: ast.FunctionStatement):
        # Reuse action logic
        name = stmt.name if isinstance(stmt.name, str) else getattr(stmt.name, "value", None)
        param_types: List[Tuple[str, Optional[TypeSpec]]] = []
        for p in stmt.parameters or []:
            ann_str = getattr(p, "type_annotation", None)
            ts = _resolve_annotation(ann_str)
            param_types.append((p.value, ts))
        ret_ts = _resolve_annotation(getattr(stmt, "return_type", None))
        if name:
            self._scope.define_callable(name, param_types, ret_ts)
            self._scope.define(name, TypeSpec(BaseType.ACTION))

        self._push_scope()
        for pname, pts in param_types:
            self._scope.define(pname, pts)
        if stmt.body:
            self._check_block(stmt.body, ret_ts)
        self._pop_scope()

    def _check_ReturnStatement(self, stmt: ast.ReturnStatement):
        # Return type validation is handled during block check
        pass

    def _check_ExpressionStatement(self, stmt: ast.ExpressionStatement):
        if stmt.expression:
            self._check_expression(stmt.expression)

    def _check_IfStatement(self, stmt):
        if hasattr(stmt, "condition") and stmt.condition:
            self._check_expression(stmt.condition)
        if hasattr(stmt, "consequence") and stmt.consequence:
            self._check_block(stmt.consequence)
        if hasattr(stmt, "alternative") and stmt.alternative:
            self._check_block(stmt.alternative)

    def _check_WhileStatement(self, stmt):
        if hasattr(stmt, "condition") and stmt.condition:
            self._check_expression(stmt.condition)
        if hasattr(stmt, "body") and stmt.body:
            self._check_block(stmt.body)

    def _check_ForStatement(self, stmt):
        if hasattr(stmt, "body") and stmt.body:
            self._check_block(stmt.body)

    # -- block / body visitor -----------------------------------------------

    def _check_block(self, block, expected_return: Optional[TypeSpec] = None):
        stmts = getattr(block, "statements", None)
        if not stmts:
            return
        for s in stmts:
            self._check_statement(s)
            # Check return type
            if expected_return and isinstance(s, ast.ReturnStatement):
                if s.return_value:
                    actual = _infer_expr_type(s.return_value, self._scope)
                    if actual and not _is_compatible(expected_return, actual):
                        self._error(
                            f"Return type {actual.base_type.value} is incompatible "
                            f"with declared return type {expected_return.base_type.value}",
                            s,
                        )

    # -- expression visitors ------------------------------------------------

    def _check_expression(self, expr: ast.Expression):
        if isinstance(expr, ast.CallExpression):
            self._check_call(expr)
        elif isinstance(expr, ast.InfixExpression):
            if expr.left:
                self._check_expression(expr.left)
            if expr.right:
                self._check_expression(expr.right)
            self._check_infix_types(expr)
        elif isinstance(expr, ast.PrefixExpression):
            if expr.right:
                self._check_expression(expr.right)

    def _check_call(self, call: ast.CallExpression):
        fn_name = None
        if isinstance(call.function, ast.Identifier):
            fn_name = call.function.value
        if not fn_name:
            return
        sig = self._scope.lookup_callable(fn_name)
        if sig is None:
            return  # unknown function — skip
        params, _ = sig
        args = call.arguments or []

        # Arity check
        expected = len(params)
        got = len(args)
        if got != expected:
            self._error(
                f"'{fn_name}' expects {expected} argument(s) but got {got}",
                call,
            )

        # Per-argument type check
        for i, (pname, pts) in enumerate(params):
            if i >= len(args):
                break
            if pts is None:
                continue
            actual = _infer_expr_type(args[i], self._scope)
            if actual and not _is_compatible(pts, actual):
                self._error(
                    f"Argument '{pname}' of '{fn_name}' expects "
                    f"{pts.base_type.value} but got {actual.base_type.value}",
                    call,
                )

    def _check_infix_types(self, expr: ast.InfixExpression):
        """Warn on obviously wrong infix operations."""
        if expr.operator in ("+", "-", "*", "/", "%", "**"):
            left = _infer_expr_type(expr.left, self._scope)
            right = _infer_expr_type(expr.right, self._scope)
            if left and right:
                # String + Number is fine (concatenation). But Number - String isn't.
                if expr.operator != "+" and (
                    (left.base_type == BaseType.STRING and right.base_type in (BaseType.INT, BaseType.FLOAT))
                    or (right.base_type == BaseType.STRING and left.base_type in (BaseType.INT, BaseType.FLOAT))
                ):
                    self._warning(
                        f"Suspicious operation: {left.base_type.value} "
                        f"{expr.operator} {right.base_type.value}",
                        expr,
                    )
