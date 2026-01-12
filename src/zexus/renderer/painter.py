"""ASCII painter used by ``main_renderer``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AsciiPainter:
    width: int = 0
    height: int = 0
    buffer: List[List[str]] = field(default_factory=list)

    def init_screen(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.buffer = [[" " for _ in range(width)] for _ in range(height)]

    def draw_lines(self, lines: List[str], x: int, y: int) -> None:
        for row, line in enumerate(lines):
            target_y = y + row
            if not (0 <= target_y < self.height):
                continue
            for col, char in enumerate(line):
                target_x = x + col
                if 0 <= target_x < self.width and char != " ":
                    self.buffer[target_y][target_x] = char

    def render(self) -> str:
        return "\n".join("".join(row) for row in self.buffer)
