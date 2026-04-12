"""Tests for stdlib TemplateModule."""

import pytest
from src.zexus.stdlib.template import TemplateModule


class TestVariableInterpolation:
    def test_simple_variable(self):
        result = TemplateModule.render("Hello, {{name}}!", {"name": "World"})
        assert result == "Hello, World!"

    def test_multiple_variables(self):
        result = TemplateModule.render("{{a}} + {{b}}", {"a": "1", "b": "2"})
        assert result == "1 + 2"

    def test_missing_variable(self):
        result = TemplateModule.render("{{missing}}", {})
        assert result == ""

    def test_nested_access(self):
        result = TemplateModule.render("{{user.name}}", {"user": {"name": "Alice"}})
        assert result == "Alice"


class TestFilters:
    def test_upper_filter(self):
        result = TemplateModule.render("{{name|upper}}", {"name": "hello"})
        assert result == "HELLO"

    def test_lower_filter(self):
        result = TemplateModule.render("{{name|lower}}", {"name": "HELLO"})
        assert result == "hello"

    def test_capitalize_filter(self):
        result = TemplateModule.render("{{name|capitalize}}", {"name": "hello"})
        assert result == "Hello"

    def test_escape_filter(self):
        result = TemplateModule.render("{{html|escape}}", {"html": "<b>bold</b>"})
        assert "<b>" not in result
        assert "&lt;" in result


class TestConditionals:
    def test_if_true(self):
        tpl = "{% if show %}visible{% endif %}"
        result = TemplateModule.render(tpl, {"show": True})
        assert "visible" in result

    def test_if_false(self):
        tpl = "{% if show %}visible{% endif %}"
        result = TemplateModule.render(tpl, {"show": False})
        assert "visible" not in result

    def test_if_else(self):
        tpl = "{% if admin %}Admin{% else %}User{% endif %}"
        assert "Admin" in TemplateModule.render(tpl, {"admin": True})
        assert "User" in TemplateModule.render(tpl, {"admin": False})


class TestLoops:
    def test_for_loop(self):
        tpl = "{% for item in items %}{{item}} {% endfor %}"
        result = TemplateModule.render(tpl, {"items": ["a", "b", "c"]})
        assert "a" in result
        assert "b" in result
        assert "c" in result


class TestSafeRendering:
    def test_render_safe_escapes_html(self):
        result = TemplateModule.render_safe("{{user}}", {"user": "<script>alert(1)</script>"})
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


class TestRenderFile:
    def test_render_file(self, tmp_path):
        tpl_file = tmp_path / "test.html"
        tpl_file.write_text("Hello, {{name}}!")
        result = TemplateModule.render_file(str(tpl_file), {"name": "World"})
        assert result == "Hello, World!"
