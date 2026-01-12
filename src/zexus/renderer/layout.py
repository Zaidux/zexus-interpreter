"""Screen/component hierarchy used by the renderer backend.

The API mirrors the original ``renderer.layout`` module but embraces dataclasses
and type hints.  Rendering still produces ASCII output so existing snapshot
based tests continue to work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class ScreenComponent:
    name: str
    properties: Dict[str, object] = field(default_factory=dict)
    children: List["ScreenComponent"] = field(default_factory=list)
    parent: Optional["ScreenComponent"] = None
    type: str = "component"

    def add_child(self, component: "ScreenComponent") -> None:
        component.parent = self
        self.children.append(component)

    # ------------------------------------------------------------------
    def get(self, key: str, default: object | None = None) -> object | None:
        if key in self.properties:
            return self.properties[key]
        if self.parent:
            return self.parent.get(key, default)
        return default

    def clone(self, name: Optional[str] = None) -> "ScreenComponent":
        copied = type(self)(name or self.name, **dict(self.properties))
        for child in self.children:
            copied.add_child(child.clone())
        return copied

    # Rendering ---------------------------------------------------------
    def render(self, width: int, height: int) -> List[str]:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class Screen(ScreenComponent):
    type: str = "screen"

    def __post_init__(self) -> None:
        self.properties.setdefault("width", 80)
        self.properties.setdefault("height", 24)
        self.properties.setdefault("border", True)
        self.properties.setdefault("title", "")

    def render(self, width: int, height: int) -> List[str]:
        screen_width = int(self.get("width", width))
        screen_height = int(self.get("height", height))
        show_border = bool(self.get("border", True))

        canvas = [[" " for _ in range(screen_width)] for _ in range(screen_height)]

        if show_border:
            _draw_border(canvas)

        title = str(self.get("title", ""))
        if title:
            _draw_title(canvas, title)

        for child in self.children:
            child_canvas = child.render(screen_width - 4, screen_height - 4)
            origin_x = int(child.get("x", 2))
            origin_y = int(child.get("y", 2))
            _blit(canvas, child_canvas, origin_x, origin_y)

        return ["".join(row) for row in canvas]


@dataclass
class Button(ScreenComponent):
    type: str = "button"

    def __post_init__(self) -> None:
        defaults = {
            "text": "Button",
            "width": 14,
            "height": 3,
            "enabled": True,
            "x": 0,
            "y": 0,
        }
        for key, value in defaults.items():
            self.properties.setdefault(key, value)

    def render(self, width: int, height: int) -> List[str]:
        w = max(3, int(self.get("width", 14)))
        h = max(3, int(self.get("height", 3)))
        text = str(self.get("text", "Button"))[: w - 2]
        enabled = bool(self.get("enabled", True))
        border = ("┌", "┐", "└", "┘", "─", "│") if enabled else ("+", "+", "+", "+", "-", "|")

        lines = [border[0] + border[4] * (w - 2) + border[1]]
        lines.append(border[5] + text.center(w - 2) + border[5])
        lines.append(border[2] + border[4] * (w - 2) + border[3])
        while len(lines) < h:
            lines.append(" " * w)
        return lines


@dataclass
class TextBox(ScreenComponent):
    type: str = "textbox"

    def __post_init__(self) -> None:
        defaults = {
            "text": "",
            "placeholder": "",
            "width": 20,
            "height": 3,
            "x": 0,
            "y": 0,
        }
        for key, value in defaults.items():
            self.properties.setdefault(key, value)

    def render(self, width: int, height: int) -> List[str]:
        w = max(4, int(self.get("width", 20)))
        h = max(3, int(self.get("height", 3)))
        text = str(self.get("text", "")) or str(self.get("placeholder", ""))
        if len(text) > w - 4:
            text = f"{text[: w - 7]}..."

        lines = ["┌" + "─" * (w - 2) + "┐"]
        lines.append("│ " + text.ljust(w - 4) + " │")
        lines.append("└" + "─" * (w - 2) + "┘")
        while len(lines) < h:
            lines.append(" " * w)
        return lines


@dataclass
class Label(ScreenComponent):
    type: str = "label"

    def __post_init__(self) -> None:
        self.properties.setdefault("text", "")
        self.properties.setdefault("x", 0)
        self.properties.setdefault("y", 0)

    def render(self, width: int, height: int) -> List[str]:
        return [str(self.get("text", ""))]


class ScreenRegistry:
    def __init__(self) -> None:
        self._screens: Dict[str, Screen] = {}
        self._components: Dict[str, ScreenComponent] = {}

    # ------------------------------------------------------------------
    def register_screen(self, screen: Screen) -> None:
        self._screens[screen.name] = screen

    def register_component(self, component: ScreenComponent) -> None:
        self._components[component.name] = component

    def get_screen(self, name: str) -> Screen:
        if name not in self._screens:
            raise KeyError(f"screen '{name}' not defined")
        return self._screens[name]

    def get_component(self, name: str) -> ScreenComponent:
        if name not in self._components:
            raise KeyError(f"component '{name}' not defined")
        return self._components[name]

    def clone_screen(self, name: str) -> Screen:
        return self.get_screen(name).clone()

    def list_screens(self) -> Iterable[str]:
        return list(self._screens.keys())

    def list_components(self) -> Iterable[str]:
        return list(self._components.keys())

    def snapshot(self) -> Dict[str, Screen]:
        return dict(self._screens)

    def component_snapshot(self) -> Dict[str, ScreenComponent]:
        return dict(self._components)


# Internal helpers ------------------------------------------------------

def _draw_border(canvas: List[List[str]]) -> None:
    width = len(canvas[0])
    height = len(canvas)
    canvas[0][0] = "╭"
    canvas[0][width - 1] = "╮"
    canvas[height - 1][0] = "╰"
    canvas[height - 1][width - 1] = "╯"
    for x in range(1, width - 1):
        canvas[0][x] = "─"
        canvas[height - 1][x] = "─"
    for y in range(1, height - 1):
        canvas[y][0] = "│"
        canvas[y][width - 1] = "│"


def _draw_title(canvas: List[List[str]], title: str) -> None:
    width = len(canvas[0])
    title = f" {title} "
    title = title if len(title) < width - 2 else title[: width - 5] + "..."
    offset = max(1, (width - len(title)) // 2)
    for i, ch in enumerate(title):
        canvas[0][offset + i] = ch


def _blit(canvas: List[List[str]], child: List[str], x: int, y: int) -> None:
    for row, line in enumerate(child):
        for col, ch in enumerate(line):
            target_y = y + row
            target_x = x + col
            if 0 <= target_y < len(canvas) and 0 <= target_x < len(canvas[0]) and ch != " ":
                canvas[target_y][target_x] = ch
