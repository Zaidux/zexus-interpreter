"""High-level renderer facade used by the backend."""
from __future__ import annotations

from typing import Dict

from .layout import Button, Label, Screen, ScreenComponent, ScreenRegistry, TextBox
from .painter import AsciiPainter


_COMPONENT_TYPES: Dict[str, type[ScreenComponent]] = {
    "button": Button,
    "textbox": TextBox,
    "label": Label,
    "screen": Screen,
}


class ZexusScreenRenderer:
    def __init__(self, registry: ScreenRegistry | None = None) -> None:
        self.registry = registry or ScreenRegistry()
        self.painter = AsciiPainter()
        self._install_default_components()

    def _install_default_components(self) -> None:
        if "primary_button" not in self.registry.list_components():
            self.define_component("primary_button", "button", text="Submit", color="blue")
        if "secondary_button" not in self.registry.list_components():
            self.define_component("secondary_button", "button", text="Cancel", color="gray")

    # ------------------------------------------------------------------
    def define_screen(self, name: str, **properties) -> Screen:
        screen = Screen(name)
        if properties:
            screen.properties.update(properties)
        self.registry.register_screen(screen)
        return screen

    def define_component(self, name: str, component_type: str | None = None, **properties) -> ScreenComponent:
        if component_type is None:
            component_type = str(properties.pop("type", "label"))
        if component_type not in _COMPONENT_TYPES:
            raise ValueError(f"unknown component type '{component_type}'")
        component = _COMPONENT_TYPES[component_type](name)
        if properties:
            component.properties.update(properties)
        self.registry.register_component(component)
        return component

    def add_to_screen(self, screen_name: str, component_name: str, **overrides) -> None:
        screen = self.registry.get_screen(screen_name)
        component = self.registry.get_component(component_name)
        clone = component.clone()
        clone.properties.update(overrides)
        screen.add_child(clone)

    def render_screen(self, screen_name: str, **overrides) -> str:
        screen = self.registry.clone_screen(screen_name)
        screen.properties.update(overrides)

        width = int(screen.get("width", 80))
        height = int(screen.get("height", 24))
        self.painter.init_screen(width, height)

        rendered_lines = screen.render(width, height)
        self.painter.draw_lines(rendered_lines, 0, 0)
        return self.painter.render()
