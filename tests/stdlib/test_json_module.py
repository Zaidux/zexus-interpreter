"""Tests for stdlib JsonModule."""

import json
import pytest
from src.zexus.stdlib.json_module import JsonModule


@pytest.fixture
def jm():
    return JsonModule()


# ── parse ────────────────────────────────────────────────────────────────
class TestParse:
    def test_parse_object(self, jm):
        assert jm.parse('{"a": 1, "b": 2}') == {"a": 1, "b": 2}

    def test_parse_array(self, jm):
        assert jm.parse("[1, 2, 3]") == [1, 2, 3]

    def test_parse_string(self, jm):
        assert jm.parse('"hello"') == "hello"

    def test_parse_number(self, jm):
        assert jm.parse("42") == 42

    def test_parse_invalid(self, jm):
        with pytest.raises(Exception):
            jm.parse("{bad json")


# ── stringify ────────────────────────────────────────────────────────────
class TestStringify:
    def test_stringify_dict(self, jm):
        result = jm.stringify({"x": 1})
        assert json.loads(result) == {"x": 1}

    def test_stringify_with_indent(self, jm):
        result = jm.stringify({"a": 1}, indent=2)
        assert "\n" in result

    def test_stringify_list(self, jm):
        result = jm.stringify([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]


# ── validate ─────────────────────────────────────────────────────────────
class TestValidate:
    def test_valid_json(self, jm):
        assert jm.validate('{"a": 1}') is True

    def test_invalid_json(self, jm):
        assert jm.validate("{bad") is False


# ── merge ────────────────────────────────────────────────────────────────
class TestMerge:
    def test_merge_dicts(self, jm):
        assert jm.merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_merge_overwrite(self, jm):
        assert jm.merge({"a": 1}, {"a": 99}) == {"a": 99}


# ── pretty_print ─────────────────────────────────────────────────────────
class TestPrettyPrint:
    def test_pretty_print(self, jm):
        result = jm.pretty_print({"z": 1, "a": 2})
        assert "a" in result
        assert "\n" in result
        assert "z" in result


# ── load / save ──────────────────────────────────────────────────────────
class TestFileIO:
    def test_save_and_load(self, jm, tmp_path):
        fp = str(tmp_path / "test.json")
        jm.save(fp, {"hello": "world"})
        assert jm.load(fp) == {"hello": "world"}

    def test_load_nonexistent(self, jm):
        with pytest.raises(Exception):
            jm.load("/nonexistent/file.json")
