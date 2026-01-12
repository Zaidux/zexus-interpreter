"""Colour utilities for the Zexus renderer.

The previous implementation lived at ``renderer/color_system.py``.  The new
version adopts dataclasses, immutability and richer typing to make the colour
pipeline easier to reason about and safer to share between the interpreter and
VM backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class RGBColor:
    """Simple immutable RGB colour representation."""

    r: int
    g: int
    b: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "r", int(_clamp(self.r, 0, 255)))
        object.__setattr__(self, "g", int(_clamp(self.g, 0, 255)))
        object.__setattr__(self, "b", int(_clamp(self.b, 0, 255)))

    # ------------------------------------------------------------------
    # Blending helpers
    # ------------------------------------------------------------------
    def mix(self, other: "RGBColor", ratio: float = 0.5) -> "RGBColor":
        ratio = _clamp(ratio, 0.0, 1.0)
        inv = 1.0 - ratio
        return RGBColor(
            int(self.r * inv + other.r * ratio),
            int(self.g * inv + other.g * ratio),
            int(self.b * inv + other.b * ratio),
        )

    def lighten(self, amount: float = 0.1) -> "RGBColor":
        return self.mix(RGBColor(255, 255, 255), amount)

    def darken(self, amount: float = 0.1) -> "RGBColor":
        return self.mix(RGBColor(0, 0, 0), amount)

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    def to_hsv(self) -> Tuple[float, float, float]:
        r, g, b = self.r / 255.0, self.g / 255.0, self.b / 255.0
        c_max = max(r, g, b)
        c_min = min(r, g, b)
        delta = c_max - c_min

        if delta == 0:
            hue = 0.0
        elif c_max == r:
            hue = (60 * ((g - b) / delta)) % 360
        elif c_max == g:
            hue = 60 * (((b - r) / delta) + 2)
        else:
            hue = 60 * (((r - g) / delta) + 4)

        saturation = 0.0 if c_max == 0 else delta / c_max
        value = c_max
        return hue, saturation, value

    @staticmethod
    def from_hsv(h: float, s: float, v: float) -> "RGBColor":
        h = h % 360
        s = _clamp(s, 0.0, 1.0)
        v = _clamp(v, 0.0, 1.0)

        c = v * s
        x = c * (1 - abs(((h / 60.0) % 2) - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return RGBColor(int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

    # ------------------------------------------------------------------
    def to_ansi(self, *, background: bool = False) -> str:
        code = 48 if background else 38
        return f"\033[{code};2;{self.r};{self.g};{self.b}m"

    def __str__(self) -> str:  # pragma: no cover - human-friendly repr
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


class ColorPalette:
    """Palette of named colours with simple mixing helpers."""

    def __init__(self) -> None:
        self._base = self._build_base_palette()
        self._custom: Dict[str, RGBColor] = {}
        self._gradients: Dict[str, List[RGBColor]] = {}

    @staticmethod
    def _build_base_palette() -> Dict[str, RGBColor]:
        values = {
            "red": (255, 59, 48),
            "orange": (255, 149, 0),
            "yellow": (255, 204, 0),
            "green": (52, 199, 89),
            "mint": (0, 199, 190),
            "teal": (48, 176, 199),
            "cyan": (50, 173, 230),
            "blue": (0, 122, 255),
            "indigo": (88, 86, 214),
            "purple": (175, 82, 222),
            "pink": (255, 45, 85),
            "brown": (162, 132, 94),
            "white": (255, 255, 255),
            "gray": (142, 142, 147),
            "gray2": (174, 174, 178),
            "gray3": (199, 199, 204),
            "gray4": (209, 209, 214),
            "gray5": (229, 229, 234),
            "gray6": (242, 242, 247),
            "black": (0, 0, 0),
        }
        base = {name: RGBColor(*rgb) for name, rgb in values.items()}
        for name, colour in list(base.items()):
            base[f"light_{name}"] = colour.lighten(0.3)
            base[f"dark_{name}"] = colour.darken(0.3)
        return base

    # ------------------------------------------------------------------
    def get(self, name: str) -> RGBColor:
        if name in self._base:
            return self._base[name]
        if name in self._custom:
            return self._custom[name]
        raise KeyError(f"unknown colour '{name}'")

    def define(self, name: str, colour: RGBColor) -> None:
        self._custom[name] = colour

    def mix(self, colour_a: str, colour_b: str, ratio: float = 0.5) -> RGBColor:
        colour = self.get(colour_a).mix(self.get(colour_b), ratio)
        return colour

    def gradient(self, start: str, end: str, steps: int, *, name: str | None = None) -> List[RGBColor]:
        steps = max(2, steps)
        start_colour = self.get(start)
        end_colour = self.get(end)
        values = [start_colour.mix(end_colour, i / (steps - 1)) for i in range(steps)]
        if name:
            self._gradients[name] = values
        return values

    def list_names(self) -> Iterable[str]:
        yield from self._base.keys()
        yield from self._custom.keys()


class Theme:
    """Collection of palette-derived colours."""

    def __init__(self, name: str, palette: ColorPalette) -> None:
        self.name = name
        self._palette = palette
        self._colours: Dict[str, RGBColor] = {}

    def set_colour(self, key: str, colour_name: str) -> None:
        self._colours[key] = self._palette.get(colour_name)

    def derive_from(self, colour_name: str) -> None:
        base = self._palette.get(colour_name)
        self._colours.update(
            {
                "primary": base,
                "primary_light": base.lighten(0.2),
                "primary_dark": base.darken(0.2),
            }
        )

    def get(self, key: str, default: str | None = None) -> RGBColor:
        if key in self._colours:
            return self._colours[key]
        if default:
            return self._palette.get(default)
        return self._palette.get("black")

    def snapshot(self) -> Dict[str, str]:
        return {key: str(colour) for key, colour in self._colours.items()}
