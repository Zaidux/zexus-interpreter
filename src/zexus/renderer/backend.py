"""Renderer backend exposed to the evaluator and VM implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, Optional

from .canvas import CanvasRegistry
from .color_system import ColorPalette, RGBColor, Theme
from .layout import Screen, ScreenRegistry
from .main_renderer import ZexusScreenRenderer


@dataclass
class RendererState:
    screens: ScreenRegistry = field(default_factory=ScreenRegistry)
    canvases: CanvasRegistry = field(default_factory=CanvasRegistry)
    themes: Dict[str, Theme] = field(default_factory=dict)
    current_theme: Optional[str] = None
    colours: Dict[str, str] = field(default_factory=dict)
    graphics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    animations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    clocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    canvas_aliases: Dict[str, str] = field(default_factory=dict)


class RendererBackend:
    def __init__(self) -> None:
        self.palette = ColorPalette()
        self.state = RendererState()
        self.renderer = ZexusScreenRenderer(self.state.screens)

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------
    def mix(self, colour_a: str, colour_b: str, ratio: float = 0.5) -> RGBColor:
        return self.palette.mix(colour_a, colour_b, ratio)

    def define_color(self, name: str, spec: object) -> RGBColor:
        colour = self._coerce_colour(spec)
        self.palette.define(name, colour)
        self.state.colours[name] = str(colour)
        return colour

    # ------------------------------------------------------------------
    # Screen/component management
    # ------------------------------------------------------------------
    def define_screen(self, name: str, properties: Dict[str, Any] | None = None) -> None:
        self.renderer.define_screen(name, **(properties or {}))

    def define_component(self, name: str, properties: Dict[str, Any] | None = None) -> None:
        props = dict(properties or {})
        component_type = props.pop("type", None)
        self.renderer.define_component(name, component_type, **props)

    def add_to_screen(self, screen_name: str, component_name: str, overrides: Dict[str, Any] | None = None) -> None:
        self.renderer.add_to_screen(screen_name, component_name, **(overrides or {}))

    def render_screen(self, name: str, overrides: Dict[str, Any] | None = None) -> str:
        return self.renderer.render_screen(name, **(overrides or {}))

    # ------------------------------------------------------------------
    # Theme handling
    # ------------------------------------------------------------------
    def set_theme(self, target_or_theme: str, theme_name: str | None = None) -> None:
        if theme_name is None:
            self.state.current_theme = target_or_theme
            if target_or_theme not in self.state.themes:
                self.state.themes[target_or_theme] = Theme(target_or_theme, self.palette)
            return

        if target_or_theme in self.state.screens.list_screens():
            screen = self.state.screens.get_screen(target_or_theme)
            screen.properties["theme"] = theme_name
            if theme_name not in self.state.themes:
                self.state.themes[theme_name] = Theme(theme_name, self.palette)
        else:
            if theme_name not in self.state.themes:
                self.state.themes[theme_name] = Theme(theme_name, self.palette)

    # ------------------------------------------------------------------
    # Canvas helpers
    # ------------------------------------------------------------------
    def create_canvas(self, width: int, height: int) -> str:
        return self.state.canvases.create(width=width, height=height)

    def create_named_canvas(self, name: str, width: int, height: int) -> str:
        identifier = self.create_canvas(width, height)
        self.state.canvas_aliases[name] = identifier
        return identifier

    def draw_line(self, canvas_id: str, x1: int, y1: int, x2: int, y2: int) -> None:
        canvas = self.state.canvases.get(self._resolve_canvas_identifier(canvas_id))
        canvas.draw_line(x1, y1, x2, y2)

    def draw_text(self, canvas_id: str, x: int, y: int, text: str) -> None:
        canvas = self.state.canvases.get(self._resolve_canvas_identifier(canvas_id))
        canvas.draw_text(x, y, text)

    # ------------------------------------------------------------------
    # Higher-level registries
    # ------------------------------------------------------------------
    def register_graphics(self, name: str, layers: Iterable[str] | Dict[str, Any]) -> None:
        if isinstance(layers, dict):
            payload = dict(layers)
        else:
            payload = {f"layer_{idx}": layer for idx, layer in enumerate(layers)}
        self.state.graphics[name] = payload

    def register_animation(self, name: str, frames: Dict[str, Any]) -> None:
        self.state.animations[name] = dict(frames)

    def register_clock(self, name: str, config: Dict[str, Any]) -> None:
        self.state.clocks[name] = dict(config)

    # ------------------------------------------------------------------
    def inspect_registry(self) -> Dict[str, Any]:
        return {
            "screens": {
                name: {
                    "properties": dict(screen.properties),
                    "children": [child.name for child in screen.children],
                    "theme": screen.properties.get("theme"),
                }
                for name, screen in self.state.screens.snapshot().items()
            },
            "components": {
                name: dict(component.properties)
                for name, component in self.state.screens.component_snapshot().items()
            },
            "themes": {name: theme.snapshot() for name, theme in self.state.themes.items()},
            "canvases": self.state.canvases.snapshot(),
            "canvas_aliases": dict(self.state.canvas_aliases),
            "colours": dict(self.state.colours),
            "graphics": {name: dict(data) for name, data in self.state.graphics.items()},
            "animations": {name: dict(data) for name, data in self.state.animations.items()},
            "clocks": {name: dict(data) for name, data in self.state.clocks.items()},
            "current_theme": self.state.current_theme,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_canvas_identifier(self, identifier: str) -> str:
        if identifier in self.state.canvases.canvases:
            return identifier
        return self.state.canvas_aliases.get(identifier, identifier)

    def _coerce_colour(self, spec: object) -> RGBColor:
        if isinstance(spec, RGBColor):
            return spec

        if isinstance(spec, str):
            text = spec.strip()
            match = re.fullmatch(r"#?([0-9a-fA-F]{6})", text)
            if match:
                raw = match.group(1)
                return RGBColor(int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
            try:
                return self.palette.get(text)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc

        if isinstance(spec, dict):
            if {"r", "g", "b"}.issubset(spec):
                return RGBColor(int(spec["r"]), int(spec["g"]), int(spec["b"]))
            if {"red", "green", "blue"}.issubset(spec):
                return RGBColor(int(spec["red"]), int(spec["green"]), int(spec["blue"]))
            if "hex" in spec:
                return self._coerce_colour(str(spec["hex"]))
            raise ValueError("Colour map must include r/g/b values or hex")

        if isinstance(spec, (list, tuple)) and len(spec) == 3:
            r, g, b = spec
            return RGBColor(int(r), int(g), int(b))

        raise ValueError(f"Unsupported colour specification: {spec}")


# Module level helper mirroring the legacy API -------------------------
_BACKEND = RendererBackend()


def mix(colour_a: str, colour_b: str, ratio: float = 0.5) -> RGBColor:
    return _BACKEND.mix(colour_a, colour_b, ratio)


def define_color(name: str, spec: object) -> RGBColor:
    return _BACKEND.define_color(name, spec)


def define_screen(name: str, properties: Dict[str, Any] | None = None) -> None:
    _BACKEND.define_screen(name, properties)


def define_component(name: str, properties: Dict[str, Any] | None = None) -> None:
    _BACKEND.define_component(name, properties)


def add_to_screen(screen_name: str, component_name: str, overrides: Dict[str, Any] | None = None) -> None:
    _BACKEND.add_to_screen(screen_name, component_name, overrides)


def render_screen(name: str, overrides: Dict[str, Any] | None = None) -> str:
    return _BACKEND.render_screen(name, overrides)


def set_theme(target_or_theme: str, theme_name: str | None = None) -> None:
    _BACKEND.set_theme(target_or_theme, theme_name)


def create_canvas(width: int, height: int) -> str:
    return _BACKEND.create_canvas(width, height)


def create_named_canvas(name: str, width: int, height: int) -> str:
    return _BACKEND.create_named_canvas(name, width, height)


def draw_line(canvas_id: str, x1: int, y1: int, x2: int, y2: int) -> None:
    _BACKEND.draw_line(canvas_id, x1, y1, x2, y2)


def draw_text(canvas_id: str, x: int, y: int, text: str) -> None:
    _BACKEND.draw_text(canvas_id, x, y, text)


def register_graphics(name: str, data: Dict[str, Any] | Iterable[str]) -> None:
    _BACKEND.register_graphics(name, data)


def register_animation(name: str, data: Dict[str, Any]) -> None:
    _BACKEND.register_animation(name, data)


def register_clock(name: str, data: Dict[str, Any]) -> None:
    _BACKEND.register_clock(name, data)


def inspect_registry() -> Dict[str, Any]:
    return _BACKEND.inspect_registry()


__all__ = [
    "RendererBackend",
    "add_to_screen",
    "create_canvas",
    "create_named_canvas",
    "define_color",
    "define_component",
    "define_screen",
    "draw_line",
    "draw_text",
    "inspect_registry",
    "mix",
    "register_animation",
    "register_clock",
    "register_graphics",
    "render_screen",
    "set_theme",
]
