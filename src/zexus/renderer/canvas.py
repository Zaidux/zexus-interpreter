"""Lightweight drawing canvas used for renderer CANVAS helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .color_system import RGBColor


@dataclass
class Canvas:
    width: int
    height: int
    pixels: List[List[str]] = field(init=False)
    colours: List[List[RGBColor | None]] = field(init=False)
    operations: List[Tuple[str, Tuple[object, ...]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pixels = [[" " for _ in range(self.width)] for _ in range(self.height)]
        self.colours = [[None for _ in range(self.width)] for _ in range(self.height)]

    # ------------------------------------------------------------------
    def draw_line(self, x1: int, y1: int, x2: int, y2: int, *, char: str = "█", colour: RGBColor | None = None) -> None:
        orig = (x1, y1, x2, y2)
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy
        while True:
            self._plot(x1, y1, char, colour)
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy
        self.operations.append(("line", orig))

    def draw_text(self, x: int, y: int, text: str, *, colour: RGBColor | None = None) -> None:
        for offset, ch in enumerate(text):
            self._plot(x + offset, y, ch, colour)
        self.operations.append(("text", (x, y, text)))

    def snapshot(self) -> Dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "pixels": ["".join(row) for row in self.pixels],
            "draw_ops": list(self.operations),
        }

    # ------------------------------------------------------------------
    def _plot(self, x: int, y: int, char: str, colour: RGBColor | None) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = char
            self.colours[y][x] = colour


@dataclass
class CanvasRegistry:
    canvases: Dict[str, Canvas] = field(default_factory=dict)

    def create(self, *, width: int, height: int) -> str:
        identifier = f"canvas_{len(self.canvases) + 1}"
        self.canvases[identifier] = Canvas(width, height)
        return identifier

    def get(self, identifier: str) -> Canvas:
        if identifier not in self.canvases:
            raise KeyError(f"canvas '{identifier}' not found")
        return self.canvases[identifier]

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        return {name: canvas.snapshot() for name, canvas in self.canvases.items()}
