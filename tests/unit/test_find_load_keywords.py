import json
import sys
from pathlib import Path

import pytest

# Ensure repository src directory is importable when tests run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zexus.lexer import Lexer
from src.zexus.parser.parser import UltimateParser
from src.zexus.evaluator.core import Evaluator
from src.zexus.object import Environment
from src.zexus.runtime import get_load_manager, clear_load_caches
from src.zexus import module_manager


@pytest.fixture
def runtime_runner(tmp_path):
    default_manager = module_manager._default_manager
    old_base = default_manager.base_path
    old_search_paths = list(default_manager.search_paths)
    old_cache = dict(default_manager.module_cache)

    module_manager.clear_cache()

    default_manager.base_path = tmp_path
    default_manager.search_paths = [
        tmp_path,
        tmp_path / "zpm_modules",
        tmp_path / "modules",
        tmp_path / "lib",
    ]
    default_manager.module_cache = {}

    load_manager = get_load_manager()
    old_project_root = load_manager.project_root
    load_manager.project_root = tmp_path
    clear_load_caches()

    module_file = tmp_path / "main.zx"
    module_file.write_text("// main\n", encoding="utf-8")

    def run(code, *, env_overrides=None):
        env = Environment()
        env.set("__file__", str(module_file))
        env.set("__FILE__", str(module_file))
        env.set("__DIR__", str(tmp_path))
        if env_overrides:
            for key, value in env_overrides.items():
                env.set(key, value)

        lexer = Lexer(code)
        parser = UltimateParser(lexer, enable_advanced_strategies=True)
        program = parser.parse_program()
        assert not parser.errors, parser.errors

        evaluator = Evaluator(trusted=True, use_vm=False)
        result = evaluator.eval(program, env)
        return env, result

    try:
        yield run
    finally:
        module_manager.clear_cache()
        default_manager.base_path = old_base
        default_manager.search_paths = old_search_paths
        default_manager.module_cache = dict(old_cache)
        load_manager.project_root = old_project_root
        clear_load_caches()


def test_find_returns_absolute_path(runtime_runner, tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    target_file = pkg_dir / "module.zx"
    target_file.write_text("// module\n", encoding="utf-8")

    env, _ = runtime_runner('let found = find "pkg/module.zx";')
    found = env.get("found")
    assert found is not None
    assert hasattr(found, "value")
    expected = str(target_file.resolve()).replace("\\", "/")
    assert found.value == expected


def test_load_env_provider(runtime_runner, monkeypatch):
    monkeypatch.setenv("APP_KEY", "super-secret")

    env, _ = runtime_runner('let value = load env.APP_KEY;')
    loaded = env.get("value")
    assert loaded is not None
    assert getattr(loaded, "value", None) == "super-secret"
    assert getattr(loaded, "is_trusted", True) is False


def test_load_json_source(runtime_runner, tmp_path):
    config = {"api": {"key": "from-file"}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    env, _ = runtime_runner('let value = load api.key from "config.json";')
    loaded = env.get("value")
    assert loaded is not None
    assert getattr(loaded, "value", None) == "from-file"
    assert getattr(loaded, "is_trusted", True) is False


@pytest.fixture
def vm_runner(tmp_path):
    default_manager = module_manager._default_manager
    old_base = default_manager.base_path
    old_search_paths = list(default_manager.search_paths)
    old_cache = dict(default_manager.module_cache)

    module_manager.clear_cache()

    default_manager.base_path = tmp_path
    default_manager.search_paths = [
        tmp_path,
        tmp_path / "zpm_modules",
        tmp_path / "modules",
        tmp_path / "lib",
    ]
    default_manager.module_cache = {}

    load_manager = get_load_manager()
    old_project_root = load_manager.project_root
    load_manager.project_root = tmp_path
    clear_load_caches()

    module_file = tmp_path / "main.zx"
    module_file.write_text("// main\n", encoding="utf-8")

    def run(code, *, env_overrides=None):
        env = Environment()
        env.set("__file__", str(module_file))
        env.set("__FILE__", str(module_file))
        env.set("__DIR__", str(tmp_path))
        if env_overrides:
            for key, value in env_overrides.items():
                env.set(key, value)

        lexer = Lexer(code)
        parser = UltimateParser(lexer, enable_advanced_strategies=True)
        program = parser.parse_program()
        assert not parser.errors, parser.errors

        evaluator = Evaluator(trusted=True, use_vm=True)
        previous_executions = evaluator.vm_stats['vm_executions']
        result = evaluator._execute_via_vm(program, env)
        assert result is not None, "VM execution fell back to direct evaluation"
        assert evaluator.vm_stats['vm_executions'] == previous_executions + 1
        return env, result

    try:
        yield run
    finally:
        module_manager.clear_cache()
        default_manager.base_path = old_base
        default_manager.search_paths = old_search_paths
        default_manager.module_cache = dict(old_cache)
        load_manager.project_root = old_project_root
        clear_load_caches()


def test_find_executes_via_vm(vm_runner, tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    target_file = pkg_dir / "module.zx"
    target_file.write_text("// module\n", encoding="utf-8")

    env, _ = vm_runner('let found = find "pkg/module.zx";')
    found = env.get("found")
    assert found is not None
    expected = str(target_file.resolve()).replace("\\", "/")
    assert getattr(found, "value", None) == expected


def test_load_env_provider_executes_via_vm(vm_runner, monkeypatch):
    monkeypatch.setenv("APP_KEY", "super-secret")

    env, _ = vm_runner('let value = load env.APP_KEY;')
    loaded = env.get("value")
    assert loaded is not None
    assert getattr(loaded, "value", None) == "super-secret"
    assert getattr(loaded, "is_trusted", True) is False


def test_load_json_source_executes_via_vm(vm_runner, tmp_path):
    config = {"api": {"key": "from-file"}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    env, _ = vm_runner('let value = load api.key from "config.json";')
    loaded = env.get("value")
    assert loaded is not None
    assert getattr(loaded, "value", None) == "from-file"
    assert getattr(loaded, "is_trusted", True) is False
