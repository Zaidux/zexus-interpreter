"""
Configuration manager for Zexus interpreter.
Provides per-user persistent config stored at ~/.zexus/config.json
Exports a `config` singleton with convenient helpers.
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone

DEFAULT_CONFIG = {
    "debug": {
        "enabled": False,
        "level": "none",  # none, minimal, full
        "last_updated": None
    },
    "user_preferences": {
        "show_warnings": True,
        "color_output": True,
        "max_output_lines": 1000
    }
}

# Backwards-compatible runtime settings expected by earlier modules
DEFAULT_RUNTIME = {
    'syntax_style': 'auto',
    'enable_advanced_parsing': True,
    'enable_debug_logs': False,
    'enable_parser_debug': False,  # OPTIMIZATION: Disable parser debug output for speed
    # Legacy runtime flags expected by older modules
    'use_hybrid_compiler': True,
    'fallback_to_interpreter': True,
    'compiler_line_threshold': 100,
    'enable_execution_stats': False,
}


class Config:
    def __init__(self):
        self.config_dir = Path.home() / ".zexus"
        self.config_file = self.config_dir / "config.json"
        self._data = DEFAULT_CONFIG.copy()
        self._ensure_loaded()

        # ensure runtime defaults exist for backward compatibility
        self._data.setdefault('runtime', {})
        for k, v in DEFAULT_RUNTIME.items():
            self._data['runtime'].setdefault(k, v)

    def _ensure_loaded(self):
        try:
            self.config_dir.mkdir(mode=0o700, exist_ok=True)
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    # Merge with defaults to keep compatibility
                    self._data = self._merge_dicts(DEFAULT_CONFIG, content)
            else:
                self._data = DEFAULT_CONFIG.copy()
                self._write()
        except Exception:
            # If anything goes wrong, fall back to defaults in-memory
            self._data = DEFAULT_CONFIG.copy()

    def _merge_dicts(self, base, override):
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._merge_dicts(result[k], v)
            else:
                result[k] = v
        return result

    def _write(self):
        try:
            self.config_dir.mkdir(mode=0o700, exist_ok=True)
            self._data['debug']['last_updated'] = datetime.now(timezone.utc).isoformat()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=4)
        except Exception:
            # Fail silently; we do not want to crash the interpreter for config issues
            pass

    # Public API
    @property
    def debug_level(self):
        return self._data.get('debug', {}).get('level', 'none')

    @debug_level.setter
    def debug_level(self, value):
        if value not in ('none', 'minimal', 'full'):
            raise ValueError('Invalid debug level')
        self._data.setdefault('debug', {})['level'] = value
        self._data['debug']['enabled'] = (value != 'none')
        self._write()

    def enable_debug(self, level='full'):
        self.debug_level = level

    def disable_debug(self):
        self.debug_level = 'none'

    def is_debug_full(self):
        return self.debug_level == 'full'

    def is_debug_minimal(self):
        return self.debug_level == 'minimal'

    def is_debug_none(self):
        return self.debug_level == 'none'

    # Backwards-compatible properties
    @property
    def syntax_style(self):
        return self._data.get('runtime', {}).get('syntax_style', 'auto')

    @syntax_style.setter
    def syntax_style(self, value):
        self._data.setdefault('runtime', {})['syntax_style'] = value
        self._write()

    @property
    def enable_advanced_parsing(self):
        return self._data.get('runtime', {}).get('enable_advanced_parsing', True)

    @enable_advanced_parsing.setter
    def enable_advanced_parsing(self, value):
        self._data.setdefault('runtime', {})['enable_advanced_parsing'] = bool(value)
        self._write()

    @property
    def enable_debug_logs(self):
        # Map legacy flag to debug level
        return self.debug_level != 'none'

    @enable_debug_logs.setter
    def enable_debug_logs(self, value):
        if value:
            if self.debug_level == 'none':
                self.debug_level = 'minimal'
        else:
            self.debug_level = 'none'

    # Legacy runtime properties
    @property
    def use_hybrid_compiler(self):
        return self._data.get('runtime', {}).get('use_hybrid_compiler', True)

    @use_hybrid_compiler.setter
    def use_hybrid_compiler(self, value):
        self._data.setdefault('runtime', {})['use_hybrid_compiler'] = bool(value)
        self._write()

    @property
    def fallback_to_interpreter(self):
        return self._data.get('runtime', {}).get('fallback_to_interpreter', True)

    @fallback_to_interpreter.setter
    def fallback_to_interpreter(self, value):
        self._data.setdefault('runtime', {})['fallback_to_interpreter'] = bool(value)
        self._write()

    @property
    def compiler_line_threshold(self):
        return int(self._data.get('runtime', {}).get('compiler_line_threshold', 100))

    @compiler_line_threshold.setter
    def compiler_line_threshold(self, value):
        try:
            v = int(value)
        except Exception:
            v = 100
        self._data.setdefault('runtime', {})['compiler_line_threshold'] = v
        self._write()

    @property
    def enable_execution_stats(self):
        return bool(self._data.get('runtime', {}).get('enable_execution_stats', False))

    @enable_execution_stats.setter
    def enable_execution_stats(self, value):
        self._data.setdefault('runtime', {})['enable_execution_stats'] = bool(value)
        self._write()

    # Helper logging function used by modules
    def should_log(self, level='debug'):
        """Decide whether to emit a log of a particular level.
        Levels: 'debug' (very verbose), 'info' (useful info), 'warn', 'error'
        """
        dl = self.debug_level
        if dl == 'full':
            return True
        if dl == 'minimal':
            return level in ('error', 'warn', 'info')
        # none
        return level in ('error',)


# Singleton
config = Config()