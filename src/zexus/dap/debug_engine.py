"""
Debug Engine — breakpoint / stepping state machine.

The engine is attached to the evaluator and consulted before every AST
node is dispatched.  When a stop condition is met the engine parks the
execution thread on an ``Event`` and waits for the DAP front-end to
resume it.

Thread model
~~~~~~~~~~~~
* **Execution thread** — runs the Zexus program via the evaluator.
  Calls ``DebugEngine.check()`` before each node.
* **DAP I/O thread** — reads DAP JSON messages, mutates engine state,
  and signals the execution thread to continue.

Locking: a single ``threading.Condition`` guards all mutable state.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class StopReason(Enum):
    BREAKPOINT = auto()
    STEP = auto()
    PAUSE = auto()
    ENTRY = auto()
    EXCEPTION = auto()


class StepMode(Enum):
    CONTINUE = auto()
    STEP_OVER = auto()
    STEP_INTO = auto()
    STEP_OUT = auto()


@dataclass
class Breakpoint:
    id: int
    file: str
    line: int
    condition: Optional[str] = None
    hit_count: int = 0
    enabled: bool = True


@dataclass
class StackFrame:
    id: int
    name: str
    file: str
    line: int
    column: int = 0
    variables: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DebugEngine:
    """Breakpoint / stepping state machine consulted by the evaluator."""

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._paused = False
        self._stop_reason: Optional[StopReason] = None
        self._step_mode: StepMode = StepMode.CONTINUE
        self._step_depth: int = 0  # call depth at step start
        self._stop_on_entry: bool = False

        # Breakpoints: file -> {line -> Breakpoint}
        self._breakpoints: Dict[str, Dict[int, Breakpoint]] = {}
        self._conditional_breakpoints: Dict[str, Dict[int, ConditionalBreakpoint]] = {}
        self._bp_id_counter: int = 0

        # Current execution state (updated by evaluator hook)
        self._current_file: str = ""
        self._current_line: int = 0
        self._call_depth: int = 0
        self._call_stack: List[StackFrame] = []
        self._frame_id_counter: int = 0

        # Callback fired when execution stops (DAP server sends stopped event)
        self.on_stopped: Optional[Callable[[StopReason, str], None]] = None
        # Callback fired when execution completes
        self.on_terminated: Optional[Callable[[], None]] = None

        self._terminated = False

    # -- Evaluator hook (called from eval_node) ----------------------------

    def check(self, file: str, line: int, env: Any = None) -> None:
        """Called by the evaluator **before** dispatching each AST node.

        If a stop condition is met the calling thread parks here until the
        DAP server calls ``continue_execution()`` or a step variant.
        """
        if self._terminated:
            return

        with self._lock:
            self._current_file = file
            self._current_line = line

            # Update top frame
            if self._call_stack:
                self._call_stack[-1].line = line
                if env:
                    self._call_stack[-1].variables = self._extract_vars(env)

            reason = self._should_stop(file, line)
            if reason is None:
                return

            self._paused = True
            self._stop_reason = reason

        # Notify DAP server (outside lock)
        if self.on_stopped:
            self.on_stopped(reason, f"{file}:{line}")

        # Park execution thread
        with self._lock:
            while self._paused and not self._terminated:
                self._lock.wait(timeout=0.5)

    def notify_call(self, name: str, file: str, line: int, env: Any = None):
        """Push a new frame when entering a function/action."""
        with self._lock:
            self._call_depth += 1
            self._frame_id_counter += 1
            frame = StackFrame(
                id=self._frame_id_counter,
                name=name,
                file=file,
                line=line,
                variables=self._extract_vars(env) if env else {},
            )
            self._call_stack.append(frame)

    def notify_return(self):
        """Pop a frame when leaving a function/action."""
        with self._lock:
            self._call_depth -= 1
            if self._call_stack:
                self._call_stack.pop()

    def notify_terminated(self):
        """Signal that program execution finished."""
        with self._lock:
            self._terminated = True
            self._paused = False
            self._lock.notify_all()
        if self.on_terminated:
            self.on_terminated()

    # -- DAP server control (called from I/O thread) -----------------------

    def continue_execution(self):
        with self._lock:
            self._step_mode = StepMode.CONTINUE
            self._paused = False
            self._lock.notify_all()

    def step_over(self):
        with self._lock:
            self._step_mode = StepMode.STEP_OVER
            self._step_depth = self._call_depth
            self._paused = False
            self._lock.notify_all()

    def step_into(self):
        with self._lock:
            self._step_mode = StepMode.STEP_INTO
            self._paused = False
            self._lock.notify_all()

    def step_out(self):
        with self._lock:
            self._step_mode = StepMode.STEP_OUT
            self._step_depth = self._call_depth
            self._paused = False
            self._lock.notify_all()

    def pause(self):
        """Request the execution to pause at the next node."""
        with self._lock:
            self._step_mode = StepMode.STEP_INTO  # stop on next node

    def terminate(self):
        with self._lock:
            self._terminated = True
            self._paused = False
            self._lock.notify_all()

    # -- Breakpoint management ---------------------------------------------

    def set_breakpoints(self, file: str, lines: List[int]) -> List[Breakpoint]:
        """Replace breakpoints for *file* with the given *lines*."""
        bps: Dict[int, Breakpoint] = {}
        result = []
        for ln in lines:
            self._bp_id_counter += 1
            bp = Breakpoint(id=self._bp_id_counter, file=file, line=ln)
            bps[ln] = bp
            result.append(bp)
        with self._lock:
            self._breakpoints[file] = bps
        return result

    def clear_breakpoints(self, file: str):
        with self._lock:
            self._breakpoints.pop(file, None)
            self._conditional_breakpoints.pop(file, None)

    def set_conditional_breakpoint(self, file: str, line: int,
                                   condition: Optional[str] = None,
                                   hit_count: Optional[int] = None,
                                   log_message: Optional[str] = None) -> Breakpoint:
        """Add a single conditional breakpoint and return it."""
        self._bp_id_counter += 1
        bp = Breakpoint(id=self._bp_id_counter, file=file, line=line,
                        condition=condition)
        cond_bp = ConditionalBreakpoint(file, line, condition, hit_count, log_message)
        with self._lock:
            self._breakpoints.setdefault(file, {})[line] = bp
            self._conditional_breakpoints.setdefault(file, {})[line] = cond_bp
        return bp

    def evaluate_expression(self, expression: str, frame: Optional[StackFrame] = None) -> str:
        """Evaluate *expression* against the given (or topmost) stack frame.

        Returns a string representation of the result.
        """
        variables: Dict[str, Any] = {}
        with self._lock:
            if frame is not None:
                variables = dict(frame.variables)
            elif self._call_stack:
                variables = dict(self._call_stack[-1].variables)
        try:
            result = _safe_eval(expression, variables)
            return str(result)
        except Exception as exc:
            return f"<error: {exc}>"

    # -- Inspection --------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    def get_stack_trace(self) -> List[StackFrame]:
        with self._lock:
            return list(reversed(self._call_stack))

    def get_variables(self, frame_id: int) -> Dict[str, Any]:
        with self._lock:
            for f in self._call_stack:
                if f.id == frame_id:
                    return dict(f.variables)
        return {}

    def set_stop_on_entry(self, stop: bool):
        self._stop_on_entry = stop

    # -- Internal ----------------------------------------------------------

    def _should_stop(self, file: str, line: int) -> Optional[StopReason]:
        """Determine if execution should stop at this location.

        Called with ``_lock`` held.
        """
        # Stop on entry (first node)
        if self._stop_on_entry:
            self._stop_on_entry = False
            return StopReason.ENTRY

        # Breakpoint hit
        file_bps = self._breakpoints.get(file)
        if file_bps:
            bp = file_bps.get(line)
            if bp and bp.enabled:
                bp.hit_count += 1
                # Check conditional breakpoint if one exists
                cond_bps = self._conditional_breakpoints.get(file)
                if cond_bps and line in cond_bps:
                    ctx = {}
                    if self._call_stack:
                        ctx = dict(self._call_stack[-1].variables)
                    if not cond_bps[line].should_break(ctx):
                        return None
                return StopReason.BREAKPOINT

        # Stepping logic
        if self._step_mode == StepMode.STEP_INTO:
            return StopReason.STEP
        if self._step_mode == StepMode.STEP_OVER:
            if self._call_depth <= self._step_depth:
                return StopReason.STEP
        if self._step_mode == StepMode.STEP_OUT:
            if self._call_depth < self._step_depth:
                return StopReason.STEP

        return None

    @staticmethod
    def _extract_vars(env: Any) -> Dict[str, Any]:
        """Pull variable names from a Zexus Environment."""
        result: Dict[str, Any] = {}
        store = getattr(env, "store", None) or getattr(env, "_store", None)
        if isinstance(store, dict):
            for k, v in store.items():
                try:
                    result[k] = _format_value(v)
                except Exception:
                    result[k] = str(v)
        return result


class ConditionalBreakpoint:
    """A breakpoint that fires only when a condition is satisfied or
    after a specified number of hits."""

    def __init__(self, file: str, line: int, condition: Optional[str] = None,
                 hit_count: Optional[int] = None, log_message: Optional[str] = None):
        self.file = file
        self.line = line
        self.condition = condition
        self.hit_count_threshold = hit_count
        self.log_message = log_message
        self._hits = 0

    def should_break(self, context: Dict[str, Any]) -> bool:
        """Return True if execution should pause at this breakpoint.

        *context* is a mapping of variable-name → value from the current
        stack frame.
        """
        self._hits += 1

        if self.hit_count_threshold is not None:
            if self._hits < self.hit_count_threshold:
                return False

        if self.condition is not None:
            try:
                result = _safe_eval(self.condition, context)
                if not result:
                    return False
            except Exception:
                return False

        return True


def _safe_eval(expression: str, variables: Dict[str, Any]) -> Any:
    """Evaluate a simple expression against *variables*.

    Supports: variable lookup, comparisons (==, !=, <, >, <=, >=),
    arithmetic (+, -, *, /, %), boolean logic (and, or, not), and
    string literals.
    """
    import ast as _ast

    allowed_names = dict(variables)
    allowed_names.update({"True": True, "False": False, "None": None,
                          "true": True, "false": False, "null": None})

    # AST-level validation: reject function calls, attribute access,
    # imports, comprehensions, and lambdas.
    try:
        tree = _ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc

    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Call, _ast.Attribute, _ast.Import,
                             _ast.ImportFrom, _ast.Lambda,
                             _ast.ListComp, _ast.SetComp,
                             _ast.DictComp, _ast.GeneratorExp)):
            raise NameError(
                f"Disallowed expression construct: {type(node).__name__}"
            )

    # Compile with restricted builtins to prevent code execution
    code = compile(tree, "<expr>", "eval")
    # Disallow any names that are not in our whitelist
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(f"Use of '{name}' is not allowed")
    return eval(code, {"__builtins__": {}}, allowed_names)  # noqa: S307


def _format_value(v: Any) -> str:
    """Format a Zexus runtime value for display in the debugger."""
    if v is None:
        return "null"
    if hasattr(v, "inspect"):
        return v.inspect()
    if hasattr(v, "value"):
        return str(v.value)
    return str(v)
