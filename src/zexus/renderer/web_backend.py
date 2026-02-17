"""
Web GUI Backend for the Zexus Renderer.

Serves the rendered UI as HTML over a local HTTP server.  The evaluator
creates screens/components normally and then calls ``show_screen()`` which
opens a browser tab to ``http://localhost:<port>/``.

The server uses only the Python standard library (``http.server`` +
``json``).  No Flask/Django/etc. dependency.

Usage from Zexus::

    @zexus gui_backend = "web"

    screen dashboard {
        title: "Dashboard"
        width: 600
        height: 400
    }

    component counter label {
        text: "Count: 0"
        x: 20
        y: 30
    }

    add counter to dashboard
    show dashboard
"""

from __future__ import annotations

import json
import os
import socket
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

from .layout import Screen, ScreenComponent, ScreenRegistry, Button, Label, TextBox
from .canvas import Canvas, CanvasRegistry
from .color_system import RGBColor


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: {bg}; }}
  .zx-screen {{ position: relative; width: {width}px; height: {height}px;
                margin: 20px auto; border: 1px solid #ddd; border-radius: 8px;
                background: {bg}; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.12); }}
  .zx-label {{ position: absolute; }}
  .zx-button {{ position: absolute; padding: 6px 16px; border: 1px solid #888;
                border-radius: 4px; cursor: pointer; background: #f4f4f4; }}
  .zx-button:hover {{ background: #e0e0e0; }}
  .zx-textbox {{ position: absolute; padding: 4px 8px; border: 1px solid #aaa;
                 border-radius: 3px; }}
  .zx-canvas {{ position: relative; }}
</style>
</head>
<body>
<div class="zx-screen">
{components}
</div>
{canvas_section}
<script>
  // Event relay — POST click events back to /event
  document.querySelectorAll('.zx-button').forEach(btn => {{
    btn.addEventListener('click', () => {{
      fetch('/event', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{component: btn.dataset.name, event: 'click'}})
      }});
    }});
  }});
</script>
</body>
</html>
"""


def _component_html(comp: ScreenComponent) -> str:
    x = int(comp.get("x", 10))
    y = int(comp.get("y", 10))
    color = comp.get("color", "black")
    text = comp.get("text", comp.name)

    if isinstance(comp, Button):
        return (f'<button class="zx-button" data-name="{comp.name}" '
                f'style="left:{x}px;top:{y}px;color:{color}">{text}</button>')
    if isinstance(comp, TextBox):
        w = int(comp.get("width", 20))
        return (f'<input class="zx-textbox" '
                f'style="left:{x}px;top:{y}px;width:{w*8}px;color:{color}" '
                f'value="{text}" />')
    # Default: Label
    size = int(comp.get("fontSize", 14))
    return (f'<span class="zx-label" '
            f'style="left:{x}px;top:{y}px;font-size:{size}px;color:{color}">{text}</span>')


def _canvas_svg(canvas: Canvas) -> str:
    """Render a ``Canvas`` as an inline SVG element."""
    scale = 4
    w, h = canvas.width * scale, canvas.height * scale
    lines: List[str] = [f'<svg class="zx-canvas" width="{w}" height="{h}" '
                        f'style="margin:20px auto;display:block;border:1px solid #ddd">']
    for op_type, args in canvas.operations:
        if op_type == "line":
            x1, y1, x2, y2 = args
            lines.append(f'<line x1="{x1*scale}" y1="{y1*scale}" '
                         f'x2="{x2*scale}" y2="{y2*scale}" stroke="black" stroke-width="2"/>')
        elif op_type == "text":
            x, y, text = args
            lines.append(f'<text x="{x*scale}" y="{y*scale+12}">{text}</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def build_html(screen: Screen, canvases: Optional[Dict[str, Canvas]] = None) -> str:
    title = str(screen.get("title", screen.name))
    width = int(screen.get("width", 600))
    height = int(screen.get("height", 400))
    bg = str(screen.get("background", "#ffffff"))

    components = "\n".join(_component_html(c) for c in screen.children)
    canvas_section = ""
    if canvases:
        canvas_section = "\n".join(_canvas_svg(c) for c in canvases.values())

    return _HTML_TEMPLATE.format(
        title=title, width=width, height=height, bg=bg,
        components=components, canvas_section=canvas_section,
    )


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Serves the screen HTML and accepts event POSTs."""

    backend: "WebBackend"  # set on class before server starts

    def do_GET(self):
        html = self.backend._current_html or "<h1>No screen loaded</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            comp_name = data.get("component", "")
            event = data.get("event", "")
            handler = self.backend._event_handlers.get(f"{comp_name}:{event}")
            if handler:
                handler()
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, format, *args):
        pass  # silence request logs


# ---------------------------------------------------------------------------
# WebBackend
# ---------------------------------------------------------------------------

class WebBackend:
    """Serves Zexus UI as HTML on a local HTTP port."""

    def __init__(self, screen_registry: Optional[ScreenRegistry] = None,
                 canvas_registry: Optional[CanvasRegistry] = None,
                 port: int = 0) -> None:
        self.screen_registry = screen_registry or ScreenRegistry()
        self.canvas_registry = canvas_registry or CanvasRegistry()
        self._port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._current_html: Optional[str] = None
        self._event_handlers: Dict[str, Any] = {}

    @property
    def url(self) -> str:
        if self._server:
            return f"http://localhost:{self._server.server_address[1]}"
        return ""

    # -- Lifecycle ---------------------------------------------------------

    def start(self) -> None:
        handler_cls = type("H", (_Handler,), {"backend": self})
        self._server = HTTPServer(("127.0.0.1", self._port), handler_cls)
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    # -- Screen rendering --------------------------------------------------

    def show_screen(self, name: str, open_browser: bool = True) -> str:
        """Build HTML from a registered screen and serve it.

        Returns the URL.
        """
        if self._server is None:
            self.start()

        screen = self.screen_registry.get_screen(name)
        canvases = self.canvas_registry.snapshot() if self.canvas_registry else None
        # Convert snapshot dicts back to Canvas objects for rendering
        canvas_objs: Dict[str, Canvas] = {}
        if self.canvas_registry:
            canvas_objs = dict(self.canvas_registry.canvases)
        self._current_html = build_html(screen, canvas_objs or None)

        url = self.url
        if open_browser:
            webbrowser.open(url)
        return url

    # -- Event binding -----------------------------------------------------

    def on(self, component_name: str, event: str, handler) -> None:
        self._event_handlers[f"{component_name}:{event}"] = handler


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_web_backend(**kwargs) -> WebBackend:
    backend = WebBackend(**kwargs)
    backend.start()
    return backend
