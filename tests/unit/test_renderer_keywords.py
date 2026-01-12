import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.unit.test_find_load_keywords import runtime_runner  # noqa: E402
from src.zexus.renderer import inspect_registry  # noqa: E402
from src.zexus.renderer import backend as renderer_backend  # noqa: E402


@pytest.fixture(autouse=True)
def reset_renderer_backend():
    renderer_backend._BACKEND = renderer_backend.RendererBackend()
    try:
        yield
    finally:
        renderer_backend._BACKEND = renderer_backend.RendererBackend()


def test_color_statement_registers_palette(runtime_runner):
    code = 'color Primary = "#3366FF";'
    env, _ = runtime_runner(code)

    color_value = env.get("Primary")
    assert color_value is not None
    assert getattr(color_value, "value", None) == "#3366ff"

    registry = inspect_registry()
    assert registry["colours"].get("Primary") == "#3366ff"


def test_canvas_statement_creates_named_canvas(runtime_runner):
    code = (
        "canvas Chart(20, 5) {\n"
        "    draw_line(Chart, 0, 0, 3, 0);\n"
        "    draw_text(Chart, 1, 1, \"Hi\");\n"
        "}\n"
    )
    env, _ = runtime_runner(code)

    chart = env.get("Chart")
    assert chart is not None
    canvas_id = getattr(chart, "value", None)

    registry = inspect_registry()
    assert canvas_id in registry["canvases"], "Canvas identifier should exist"
    alias_map = registry["canvas_aliases"]
    assert alias_map.get("Chart") == canvas_id

    draw_ops = registry["canvases"][canvas_id]["draw_ops"]
    assert any(op[0] == "line" for op in draw_ops)
    assert any(op[0] == "text" for op in draw_ops)


def test_graphics_animation_clock_statements(runtime_runner):
    code = (
        "graphics Overlay {\n"
        "    let layers = [\"###\", \"...\"];\n"
        "}\n"
        "animation Pulse(250, true) {\n"
        "    let frames = [\"a\", \"b\"];\n"
        "}\n"
        "clock UTCClock = { timezone: \"UTC\", format: \"%H:%M\" };\n"
    )
    env, _ = runtime_runner(code)

    assert env.get("Overlay") is not None
    assert env.get("Pulse") is not None
    assert env.get("UTCClock") is not None

    registry = inspect_registry()
    graphics_entry = registry["graphics"].get("Overlay")
    assert graphics_entry is not None
    assert graphics_entry.get("layers") == ["###", "..."]

    animation_entry = registry["animations"].get("Pulse")
    assert animation_entry is not None
    assert animation_entry.get("properties") == [250, True]
    assert animation_entry.get("frames") == ["a", "b"]

    clock_entry = registry["clocks"].get("UTCClock")
    assert clock_entry == {"timezone": "UTC", "format": "%H:%M"}
