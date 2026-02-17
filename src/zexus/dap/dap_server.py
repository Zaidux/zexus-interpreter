"""
DAP (Debug Adapter Protocol) Server for Zexus.

Implements the DAP JSON wire protocol over stdin/stdout so that VS Code
(or any DAP client) can drive step-through debugging of Zexus programs.

The server spawns the Zexus evaluator on a worker thread and controls it
via the ``DebugEngine``.

Reference: https://microsoft.github.io/debug-adapter-protocol/specification
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional

from .debug_engine import DebugEngine, StopReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_message(stream) -> Optional[Dict]:
    """Read one DAP message (Content-Length header + JSON body)."""
    headers: Dict[str, str] = {}
    while True:
        line = stream.readline()
        if not line:
            return None  # EOF
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = line.strip()
        if line == "":
            break
        if ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip()] = val.strip()
    length = int(headers.get("Content-Length", 0))
    if length == 0:
        return None
    body = stream.read(length)
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


def _send_message(stream, msg: Dict):
    """Write one DAP message."""
    body = json.dumps(msg)
    header = f"Content-Length: {len(body)}\r\n\r\n"
    data = header + body
    if hasattr(stream, "buffer"):
        stream.buffer.write(data.encode("utf-8"))
        stream.buffer.flush()
    else:
        stream.write(data.encode("utf-8") if isinstance(data, str) else data)
        stream.flush()


# ---------------------------------------------------------------------------
# DAP Server
# ---------------------------------------------------------------------------

_SEQ = 0


def _next_seq() -> int:
    global _SEQ
    _SEQ += 1
    return _SEQ


class DAPServer:
    """Minimal DAP server that wraps the Zexus evaluator."""

    def __init__(self, input_stream=None, output_stream=None):
        self._in = input_stream or sys.stdin
        self._out = output_stream or sys.stdout
        self.engine = DebugEngine()
        self.engine.on_stopped = self._on_stopped
        self.engine.on_terminated = self._on_terminated
        self._program_path: Optional[str] = None
        self._worker: Optional[threading.Thread] = None
        self._launched = False
        self._variable_refs: Dict[int, Dict[str, Any]] = {}
        self._var_ref_counter = 0

    # -- Main loop ---------------------------------------------------------

    def run(self):
        """Read DAP messages in a loop and dispatch them."""
        while True:
            try:
                msg = _read_message(self._in)
            except Exception:
                break
            if msg is None:
                break
            msg_type = msg.get("type")
            if msg_type == "request":
                self._handle_request(msg)
            # (events and responses from client are ignored)

    # -- Response / event helpers ------------------------------------------

    def _respond(self, request: Dict, body: Optional[Dict] = None,
                 success: bool = True, message: str = ""):
        resp: Dict[str, Any] = {
            "seq": _next_seq(),
            "type": "response",
            "request_seq": request["seq"],
            "command": request["command"],
            "success": success,
        }
        if body:
            resp["body"] = body
        if message:
            resp["message"] = message
        _send_message(self._out, resp)

    def _event(self, event: str, body: Optional[Dict] = None):
        msg: Dict[str, Any] = {
            "seq": _next_seq(),
            "type": "event",
            "event": event,
        }
        if body:
            msg["body"] = body
        _send_message(self._out, msg)

    # -- Request dispatch --------------------------------------------------

    def _handle_request(self, req: Dict):
        cmd = req.get("command", "")
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler:
            try:
                handler(req)
            except Exception as exc:
                self._respond(req, success=False, message=str(exc))
        else:
            # Unknown command — respond with success (DAP spec recommends this)
            self._respond(req)

    # -- DAP commands ------------------------------------------------------

    def _cmd_initialize(self, req: Dict):
        capabilities = {
            "supportsConfigurationDoneRequest": True,
            "supportsFunctionBreakpoints": False,
            "supportsConditionalBreakpoints": False,
            "supportsEvaluateForHovers": True,
            "supportsStepBack": False,
            "supportsSetVariable": False,
            "supportsTerminateRequest": True,
            "supportsSingleThreadExecutionRequests": False,
        }
        self._respond(req, body=capabilities)
        self._event("initialized")

    def _cmd_configurationDone(self, req: Dict):
        self._respond(req)

    def _cmd_launch(self, req: Dict):
        args = req.get("arguments", {})
        self._program_path = args.get("program", "")
        stop_on_entry = args.get("stopOnEntry", False)
        self.engine.set_stop_on_entry(stop_on_entry)
        self._respond(req)
        # Start execution on worker thread
        self._worker = threading.Thread(
            target=self._run_program, daemon=True
        )
        self._launched = True
        self._worker.start()

    def _cmd_disconnect(self, req: Dict):
        self.engine.terminate()
        self._respond(req)

    def _cmd_terminate(self, req: Dict):
        self.engine.terminate()
        self._respond(req)

    def _cmd_setBreakpoints(self, req: Dict):
        args = req.get("arguments", {})
        source = args.get("source", {})
        path = source.get("path", "")
        bp_args = args.get("breakpoints", [])
        lines = [b["line"] for b in bp_args]
        bps = self.engine.set_breakpoints(path, lines)
        body = {
            "breakpoints": [
                {"id": bp.id, "verified": True, "line": bp.line}
                for bp in bps
            ]
        }
        self._respond(req, body=body)

    def _cmd_threads(self, req: Dict):
        self._respond(req, body={
            "threads": [{"id": 1, "name": "Main Thread"}]
        })

    def _cmd_stackTrace(self, req: Dict):
        frames = self.engine.get_stack_trace()
        dap_frames = []
        for f in frames:
            self._var_ref_counter += 1
            ref = self._var_ref_counter
            self._variable_refs[ref] = f.variables
            dap_frames.append({
                "id": f.id,
                "name": f.name,
                "source": {"name": os.path.basename(f.file), "path": f.file},
                "line": f.line,
                "column": f.column,
                "scopes": ref,  # we'll use this in scopes request
            })
        self._respond(req, body={
            "stackFrames": dap_frames,
            "totalFrames": len(dap_frames),
        })

    def _cmd_scopes(self, req: Dict):
        args = req.get("arguments", {})
        frame_id = args.get("frameId", 0)
        variables = self.engine.get_variables(frame_id)
        self._var_ref_counter += 1
        ref = self._var_ref_counter
        self._variable_refs[ref] = variables
        self._respond(req, body={
            "scopes": [{
                "name": "Locals",
                "variablesReference": ref,
                "expensive": False,
            }]
        })

    def _cmd_variables(self, req: Dict):
        args = req.get("arguments", {})
        ref = args.get("variablesReference", 0)
        variables = self._variable_refs.get(ref, {})
        dap_vars = []
        for name, val in variables.items():
            if name.startswith("__"):
                continue  # hide internal vars
            dap_vars.append({
                "name": name,
                "value": str(val),
                "variablesReference": 0,
            })
        self._respond(req, body={"variables": dap_vars})

    def _cmd_continue(self, req: Dict):
        self.engine.continue_execution()
        self._respond(req, body={"allThreadsContinued": True})

    def _cmd_next(self, req: Dict):
        self.engine.step_over()
        self._respond(req)

    def _cmd_stepIn(self, req: Dict):
        self.engine.step_into()
        self._respond(req)

    def _cmd_stepOut(self, req: Dict):
        self.engine.step_out()
        self._respond(req)

    def _cmd_pause(self, req: Dict):
        self.engine.pause()
        self._respond(req)

    def _cmd_evaluate(self, req: Dict):
        args = req.get("arguments", {})
        expression = args.get("expression", "")
        # Simple variable lookup from current frame
        frames = self.engine.get_stack_trace()
        result = "undefined"
        if frames:
            vs = frames[0].variables
            if expression in vs:
                result = str(vs[expression])
        self._respond(req, body={"result": result, "variablesReference": 0})

    # -- Execution ---------------------------------------------------------

    def _run_program(self):
        """Execute the Zexus program on this thread."""
        try:
            if not self._program_path:
                self._event("output", {
                    "category": "stderr",
                    "output": "No program specified\n",
                })
                return

            # Import Zexus machinery
            from ..lexer import Lexer
            from ..parser.parser import UltimateParser as Parser
            from ..evaluator import evaluate, Evaluator
            from ..environment import Environment
            from ..object import String

            with open(self._program_path, "r") as f:
                source = f.read()

            # Parse
            lexer = Lexer(source, filename=self._program_path)
            parser = Parser(lexer, "auto", enable_advanced_strategies=True)
            program = parser.parse_program()

            if parser.errors:
                for err in parser.errors:
                    self._event("output", {
                        "category": "stderr",
                        "output": f"Parse error: {err}\n",
                    })
                return

            # Set up environment
            env = Environment()
            abs_path = os.path.abspath(self._program_path)
            env.set("__file__", String(abs_path))
            env.set("__FILE__", String(abs_path))
            env.set("__MODULE__", String("__main__"))
            env.set("__DIR__", String(os.path.dirname(abs_path)))

            # Push initial frame
            self.engine.notify_call(
                "<module>", abs_path, 1, env
            )

            # Evaluate with debug engine attached
            evaluator = Evaluator(use_vm=False)
            evaluator._debug_engine = self.engine
            evaluator._debug_file = abs_path

            # Merge builtins
            for name, val in evaluator.builtins.items():
                env.set(name, val)

            evaluator.eval_node(program, env)

        except Exception as exc:
            self._event("output", {
                "category": "stderr",
                "output": f"Runtime error: {exc}\n{traceback.format_exc()}",
            })
        finally:
            self.engine.notify_terminated()

    # -- Engine callbacks --------------------------------------------------

    def _on_stopped(self, reason: StopReason, description: str):
        reason_map = {
            StopReason.BREAKPOINT: "breakpoint",
            StopReason.STEP: "step",
            StopReason.PAUSE: "pause",
            StopReason.ENTRY: "entry",
            StopReason.EXCEPTION: "exception",
        }
        self._event("stopped", {
            "reason": reason_map.get(reason, "unknown"),
            "threadId": 1,
            "description": description,
            "allThreadsStopped": True,
        })

    def _on_terminated(self):
        self._event("terminated")


# ---------------------------------------------------------------------------
# Entry point  (python -m zexus.dap)
# ---------------------------------------------------------------------------

def main():
    server = DAPServer()
    server.run()


if __name__ == "__main__":
    main()
