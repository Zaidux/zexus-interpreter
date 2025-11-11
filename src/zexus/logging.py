"""
Centralized logging for Zexus.
Provides consistent debug/info/warn/error logging with level control and formatting.
"""
import sys
from . import config

# ANSI color codes
COLORS = {
    'reset': '\033[0m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m'
}

# Icons for different log types
ICONS = {
    'debug': '🔍',
    'info': 'ℹ️ ',
    'warn': '⚠️ ',
    'error': '❌',
    'success': '✅',
    'parser': '🔧',
    'eval': '🚀',
}

class Logger:
    def __init__(self, module_name):
        self.module_name = module_name
        self.config = config.config  # Use the singleton

    def _should_log(self, level):
        try:
            return self.config.should_log(level)
        except Exception:
            # If config check fails, only allow errors
            return level == 'error'
            
    def _format(self, level, message, data=None, color=None):
        icon = ICONS.get(level, '')
        prefix = f"{icon} [{self.module_name.upper()}]"
        if color and sys.stdout.isatty():  # Only use colors if terminal supports it
            prefix = f"{COLORS[color]}{prefix}{COLORS['reset']}"
        
        if data is not None:
            if isinstance(data, (list, tuple)) and len(data) > 0:
                # Format sequences nicely
                items = [str(x) for x in data]
                data_str = f"[{', '.join(items)}]"
            else:
                data_str = str(data)
            return f"{prefix} {message}: {data_str}"
        return f"{prefix} {message}"

    def debug(self, message, data=None):
        if self._should_log('debug'):
            print(self._format('debug', message, data, color='cyan'))

    def info(self, message, data=None):
        if self._should_log('info'):
            print(self._format('info', message, data, color='blue'))

    def warn(self, message, data=None):
        if self._should_log('warn'):
            print(self._format('warn', message, data, color='yellow'))

    def error(self, message, data=None):
        if self._should_log('error'):
            print(self._format('error', message, data, color='red'), file=sys.stderr)

    def success(self, message, data=None):
        if self._should_log('info'):
            print(self._format('success', message, data, color='green'))

    def parser(self, message, data=None):
        if self._should_log('debug'):
            print(self._format('parser', message, data, color='magenta'))
            
    def eval(self, message, data=None):
        if self._should_log('debug'):
            print(self._format('eval', message, data, color='blue'))

# Pre-configured loggers for main subsystems
parser_log = Logger('parser')
eval_log = Logger('eval')
runtime_log = Logger('runtime')