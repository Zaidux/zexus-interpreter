"""
Unit tests for src/zexus/config.py — Config class.

Tests the configuration manager: debug levels, runtime properties,
merge logic, should_log helper, and the singleton pattern.
"""

import pytest
import json
import os
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers — create isolated Config instances that don't touch the real
# ~/.zexus directory.
# ---------------------------------------------------------------------------

def _make_config(tmp_dir, initial_data=None):
    """Create a Config instance pointing at a temporary directory."""
    from src.zexus.config import Config, DEFAULT_CONFIG

    cfg = Config.__new__(Config)
    cfg.config_dir = Path(tmp_dir)
    cfg.config_file = Path(tmp_dir) / "config.json"
    cfg.fast_debug_enabled = False
    cfg._data = copy.deepcopy(DEFAULT_CONFIG)

    if initial_data:
        cfg._data = cfg._merge_dicts(DEFAULT_CONFIG, initial_data)
        cfg._write()

    # Ensure runtime defaults
    from src.zexus.config import DEFAULT_RUNTIME
    cfg._data.setdefault("runtime", {})
    for k, v in DEFAULT_RUNTIME.items():
        cfg._data["runtime"].setdefault(k, v)
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDebugLevel:
    def test_default_debug_level_is_none(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.debug_level == "none"

    def test_set_debug_level_full(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.debug_level = "full"
        assert cfg.debug_level == "full"
        assert cfg.fast_debug_enabled is True

    def test_set_debug_level_minimal(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.debug_level = "minimal"
        assert cfg.debug_level == "minimal"
        assert cfg.fast_debug_enabled is True

    def test_set_debug_level_none_disables(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.debug_level = "full"
        cfg.debug_level = "none"
        assert cfg.fast_debug_enabled is False

    def test_invalid_debug_level_raises(self, tmp_path):
        cfg = _make_config(tmp_path)
        with pytest.raises(ValueError, match="Invalid debug level"):
            cfg.debug_level = "verbose"

    def test_enable_debug_defaults_to_full(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.enable_debug()
        assert cfg.debug_level == "full"

    def test_enable_debug_with_level(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.enable_debug("minimal")
        assert cfg.debug_level == "minimal"

    def test_disable_debug(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.enable_debug("full")
        cfg.disable_debug()
        assert cfg.debug_level == "none"

    def test_is_debug_full(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.is_debug_full() is False
        cfg.debug_level = "full"
        assert cfg.is_debug_full() is True

    def test_is_debug_minimal(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.is_debug_minimal() is False
        cfg.debug_level = "minimal"
        assert cfg.is_debug_minimal() is True

    def test_is_debug_none(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.is_debug_none() is True
        cfg.debug_level = "full"
        assert cfg.is_debug_none() is False


class TestShouldLog:
    def test_level_none_only_logs_errors(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.should_log("error") is True
        assert cfg.should_log("warn") is False
        assert cfg.should_log("info") is False
        assert cfg.should_log("debug") is False

    def test_level_minimal_logs_info_warn_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.debug_level = "minimal"
        assert cfg.should_log("error") is True
        assert cfg.should_log("warn") is True
        assert cfg.should_log("info") is True
        assert cfg.should_log("debug") is False

    def test_level_full_logs_everything(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.debug_level = "full"
        assert cfg.should_log("error") is True
        assert cfg.should_log("warn") is True
        assert cfg.should_log("info") is True
        assert cfg.should_log("debug") is True


class TestRuntimeProperties:
    def test_syntax_style_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.syntax_style == "auto"

    def test_syntax_style_setter(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.syntax_style = "modern"
        assert cfg.syntax_style == "modern"

    def test_enable_advanced_parsing_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.enable_advanced_parsing is True

    def test_enable_advanced_parsing_setter(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.enable_advanced_parsing = False
        assert cfg.enable_advanced_parsing is False

    def test_enable_debug_logs_maps_to_debug_level(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.enable_debug_logs is False
        cfg.enable_debug_logs = True
        assert cfg.debug_level == "minimal"

    def test_disable_debug_logs(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.enable_debug("full")
        cfg.enable_debug_logs = False
        assert cfg.debug_level == "none"

    def test_enable_parser_debug_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.enable_parser_debug is False

    def test_enable_parser_debug_setter(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.enable_parser_debug = True
        assert cfg.enable_parser_debug is True

    def test_advanced_parsing_max_lines_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.advanced_parsing_max_lines == 2000

    def test_advanced_parsing_max_lines_setter(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.advanced_parsing_max_lines = 5000
        assert cfg.advanced_parsing_max_lines == 5000

    def test_advanced_parsing_max_tokens_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.advanced_parsing_max_tokens == 50000

    def test_advanced_parsing_max_tokens_setter(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.advanced_parsing_max_tokens = 100000
        assert cfg.advanced_parsing_max_tokens == 100000

    def test_use_hybrid_compiler_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.use_hybrid_compiler is True

    def test_fallback_to_interpreter_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.fallback_to_interpreter is True

    def test_compiler_line_threshold_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.compiler_line_threshold == 100

    def test_compiler_line_threshold_setter(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.compiler_line_threshold = 200
        assert cfg.compiler_line_threshold == 200

    def test_enable_execution_stats_default(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert cfg.enable_execution_stats is False

    def test_enable_execution_stats_setter(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.enable_execution_stats = True
        assert cfg.enable_execution_stats is True


class TestMergeDicts:
    def test_shallow_merge(self, tmp_path):
        cfg = _make_config(tmp_path)
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = cfg._merge_dicts(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_deep_merge(self, tmp_path):
        cfg = _make_config(tmp_path)
        base = {"debug": {"enabled": False, "level": "none"}}
        override = {"debug": {"level": "full"}}
        result = cfg._merge_dicts(base, override)
        assert result == {"debug": {"enabled": False, "level": "full"}}

    def test_base_not_mutated(self, tmp_path):
        cfg = _make_config(tmp_path)
        base = {"a": 1}
        override = {"a": 2}
        cfg._merge_dicts(base, override)
        assert base["a"] == 1


class TestPersistence:
    def test_write_creates_file(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg._write()
        assert (tmp_path / "config.json").exists()

    def test_written_file_is_valid_json(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg._write()
        with open(tmp_path / "config.json") as f:
            data = json.load(f)
        assert "debug" in data

    def test_last_updated_set_on_write(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg._write()
        with open(tmp_path / "config.json") as f:
            data = json.load(f)
        assert data["debug"]["last_updated"] is not None

    def test_setting_property_persists(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.debug_level = "full"
        with open(tmp_path / "config.json") as f:
            data = json.load(f)
        assert data["debug"]["level"] == "full"


class TestSingleton:
    def test_module_exports_singleton(self):
        from src.zexus.config import config
        assert config is not None
        assert hasattr(config, "debug_level")
