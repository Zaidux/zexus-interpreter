"""
Tk GUI Backend for the Zexus Renderer.

Provides a real windowed GUI using Python's built-in ``tkinter``.
The backend translates the existing ``Screen`` / ``Canvas`` / ``Component``
model into Tk widgets.

Usage from Zexus::

    @zexus gui_backend = "tk"

    screen main {
        title: "My App"
        width: 400
        height: 300
    }

    component greeting label {
        text: "Hello from Zexus!"
        x: 50
        y: 30
    }

    add greeting to main
    show main
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import font as tkfont
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

from .layout import Screen, ScreenComponent, ScreenRegistry, Button, Label, TextBox
from .canvas import Canvas, CanvasRegistry
from .color_system import RGBColor


def _rgb_hex(color: Optional[RGBColor]) -> str:
    """Convert ``RGBColor`` to Tk hex string."""
    if color is None:
        return "#000000"
    return f"#{color.r:02x}{color.g:02x}{color.b:02x}"


class TkBackend:
    """Windowed GUI backend backed by tkinter.

    The Tk event loop runs on a **dedicated thread** so the Zexus
    evaluator thread isn't blocked.
    """

    def __init__(self, screen_registry: Optional[ScreenRegistry] = None,
                 canvas_registry: Optional[CanvasRegistry] = None) -> None:
        if not TK_AVAILABLE:
            raise RuntimeError(
                "tkinter is not available. On Ubuntu: sudo apt install python3-tk"
            )
        self.screen_registry = screen_registry or ScreenRegistry()
        self.canvas_registry = canvas_registry or CanvasRegistry()
        self._root: Optional[tk.Tk] = None
        self._windows: Dict[str, tk.Toplevel] = {}
        self._tk_canvases: Dict[str, tk.Canvas] = {}
        self._event_handlers: Dict[str, Callable] = {}
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    # -- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Launch the Tk main loop on a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_mainloop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def stop(self) -> None:
        """Destroy all windows and shut down the Tk loop."""
        if self._root:
            self._root.after(0, self._root.destroy)

    def _run_mainloop(self) -> None:
        self._root = tk.Tk()
        self._root.withdraw()  # hide root window
        self._ready.set()
        self._root.mainloop()

    # -- Screen rendering --------------------------------------------------

    def show_screen(self, name: str) -> None:
        """Render a registered ``Screen`` as a Tk window."""
        screen = self.screen_registry.get_screen(name)
        width = int(screen.get("width", 400))
        height = int(screen.get("height", 300))
        title = str(screen.get("title", name))
        bg = str(screen.get("background", "#ffffff"))

        def _create():
            win = tk.Toplevel(self._root)
            win.title(title)
            win.geometry(f"{width}x{height}")
            win.configure(bg=bg)
            self._windows[name] = win

            for child in screen.children:
                self._render_component(win, child)

        self._root.after(0, _create)

    def close_screen(self, name: str) -> None:
        win = self._windows.pop(name, None)
        if win:
            self._root.after(0, win.destroy)

    # -- Component rendering -----------------------------------------------

    def _render_component(self, parent: tk.Toplevel, comp: ScreenComponent) -> None:
        x = int(comp.get("x", 10))
        y = int(comp.get("y", 10))
        fg = str(comp.get("color", "black"))
        text = str(comp.get("text", comp.name))

        if isinstance(comp, Button):
            handler = self._event_handlers.get(f"{comp.name}:click")
            btn = tk.Button(parent, text=text, fg=fg,
                            command=handler if handler else lambda: None)
            btn.place(x=x, y=y)

        elif isinstance(comp, TextBox):
            w = int(comp.get("width", 20))
            entry = tk.Entry(parent, width=w, fg=fg)
            entry.insert(0, text)
            entry.place(x=x, y=y)

        elif isinstance(comp, Label):
            size = int(comp.get("fontSize", 12))
            lbl = tk.Label(parent, text=text, fg=fg,
                           font=("TkDefaultFont", size))
            lbl.place(x=x, y=y)

        else:
            # Generic component → label
            lbl = tk.Label(parent, text=text)
            lbl.place(x=x, y=y)

    # -- Canvas rendering --------------------------------------------------

    def show_canvas(self, canvas_id: str) -> None:
        """Render a ``Canvas`` in a new Tk window."""
        canvas = self.canvas_registry.get(canvas_id)

        def _create():
            win = tk.Toplevel(self._root)
            win.title(f"Canvas: {canvas_id}")
            tk_c = tk.Canvas(win, width=canvas.width * 4,
                             height=canvas.height * 4, bg="white")
            tk_c.pack()
            self._tk_canvases[canvas_id] = tk_c
            self._draw_canvas(tk_c, canvas)

        self._root.after(0, _create)

    def _draw_canvas(self, tk_c: tk.Canvas, canvas: Canvas) -> None:
        scale = 4
        for op_type, args in canvas.operations:
            if op_type == "line":
                x1, y1, x2, y2 = args
                tk_c.create_line(
                    x1 * scale, y1 * scale,
                    x2 * scale, y2 * scale,
                    fill="black", width=2,
                )
            elif op_type == "text":
                x, y, text = args
                tk_c.create_text(x * scale, y * scale, text=text, anchor="nw")

    # -- Event binding -----------------------------------------------------

    def on(self, component_name: str, event: str, handler: Callable) -> None:
        """Register an event handler for a component.

        Example: ``backend.on("my_button", "click", my_handler)``
        """
        self._event_handlers[f"{component_name}:{event}"] = handler

    # -- Utility -----------------------------------------------------------

    def wait(self) -> None:
        """Block until the Tk loop terminates (all windows closed)."""
        if self._thread:
            self._thread.join()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_tk_backend(**kwargs) -> TkBackend:
    backend = TkBackend(**kwargs)
    backend.start()
    return backend
