"""Zexus renderer package.

This package exposes a cohesive rendering subsystem that can be consumed by
both the interpreter and the VM/compiler paths.  The public API mirrors the
previous top-level ``renderer`` module so existing call sites remain
compatible while the implementation now lives alongside the core runtime.
"""
from __future__ import annotations

from .backend import (
    RendererBackend,
    add_to_screen,
    create_canvas,
    create_named_canvas,
    define_color,
    define_component,
    define_screen,
    draw_line,
    draw_text,
    inspect_registry,
    mix,
    register_animation,
    register_clock,
    register_graphics,
    render_screen,
    set_theme,
)

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
