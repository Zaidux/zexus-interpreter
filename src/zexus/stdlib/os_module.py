"""OS module for Zexus standard library."""

import os
import sys
import platform
import subprocess
from typing import Dict, List, Any


class OSModule:
    """Provides OS-specific operations."""

    @staticmethod
    def name() -> str:
        """Get OS name."""
        return os.name

    @staticmethod
    def platform_system() -> str:
        """Get platform system (Linux, Windows, Darwin, etc.)."""
        return platform.system()

    @staticmethod
    def platform_release() -> str:
        """Get platform release version."""
        return platform.release()

    @staticmethod
    def platform_version() -> str:
        """Get platform version."""
        return platform.version()

    @staticmethod
    def platform_machine() -> str:
        """Get machine type."""
        return platform.machine()

    @staticmethod
    def platform_processor() -> str:
        """Get processor name."""
        return platform.processor()

    @staticmethod
    def python_version() -> str:
        """Get Python version."""
        return platform.python_version()

    @staticmethod
    def getcwd() -> str:
        """Get current working directory."""
        return os.getcwd()

    @staticmethod
    def chdir(path: str) -> None:
        """Change current working directory."""
        os.chdir(path)

    @staticmethod
    def getenv(name: str, default: str = "") -> str:
        """Get environment variable."""
        return os.getenv(name, default)

    @staticmethod
    def setenv(name: str, value: str) -> None:
        """Set environment variable."""
        os.environ[name] = value

    @staticmethod
    def unsetenv(name: str) -> None:
        """Unset environment variable."""
        if name in os.environ:
            del os.environ[name]

    @staticmethod
    def listenv() -> Dict[str, str]:
        """List all environment variables."""
        return dict(os.environ)

    @staticmethod
    def get_home_dir() -> str:
        """Get user home directory."""
        return os.path.expanduser("~")

    @staticmethod
    def get_temp_dir() -> str:
        """Get temporary directory."""
        import tempfile
        return tempfile.gettempdir()

    @staticmethod
    def get_username() -> str:
        """Get current username."""
        return os.getlogin() if hasattr(os, 'getlogin') else os.getenv('USER', os.getenv('USERNAME', 'unknown'))

    @staticmethod
    def get_hostname() -> str:
        """Get hostname."""
        return platform.node()

    @staticmethod
    def get_pid() -> int:
        """Get current process ID."""
        return os.getpid()

    @staticmethod
    def get_ppid() -> int:
        """Get parent process ID."""
        return os.getppid()

    @staticmethod
    def cpu_count() -> int:
        """Get number of CPUs."""
        return os.cpu_count() or 1

    @staticmethod
    def execute(command: str, shell: bool = True, capture_output: bool = True) -> Dict[str, Any]:
        """Execute shell command."""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                timeout=30
            )
            return {
                'returncode': result.returncode,
                'stdout': result.stdout if capture_output else '',
                'stderr': result.stderr if capture_output else '',
                'success': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Command timed out',
                'success': False
            }
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'success': False
            }

    @staticmethod
    def path_join(*parts: str) -> str:
        """Join path components."""
        return os.path.join(*parts)

    @staticmethod
    def path_split(path: str) -> tuple:
        """Split path into directory and file."""
        return os.path.split(path)

    @staticmethod
    def path_dirname(path: str) -> str:
        """Get directory name from path."""
        return os.path.dirname(path)

    @staticmethod
    def path_basename(path: str) -> str:
        """Get base name from path."""
        return os.path.basename(path)

    @staticmethod
    def path_exists(path: str) -> bool:
        """Check if path exists."""
        return os.path.exists(path)

    @staticmethod
    def path_isfile(path: str) -> bool:
        """Check if path is a file."""
        return os.path.isfile(path)

    @staticmethod
    def path_isdir(path: str) -> bool:
        """Check if path is a directory."""
        return os.path.isdir(path)

    @staticmethod
    def path_abspath(path: str) -> str:
        """Get absolute path."""
        return os.path.abspath(path)

    @staticmethod
    def path_realpath(path: str) -> str:
        """Get real path (resolving symlinks)."""
        return os.path.realpath(path)

    @staticmethod
    def get_file_size(path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(path)

    @staticmethod
    def get_file_mtime(path: str) -> float:
        """Get file modification time."""
        return os.path.getmtime(path)

    @staticmethod
    def get_file_atime(path: str) -> float:
        """Get file access time."""
        return os.path.getatime(path)

    @staticmethod
    def get_file_ctime(path: str) -> float:
        """Get file creation time."""
        return os.path.getctime(path)

    @staticmethod
    def get_path_separator() -> str:
        """Get path separator for current OS."""
        return os.sep

    @staticmethod
    def get_path_delimiter() -> str:
        """Get path delimiter for current OS."""
        return os.pathsep

    @staticmethod
    def get_line_separator() -> str:
        """Get line separator for current OS."""
        return os.linesep


# Export functions for easy access
name = OSModule.name
platform_system = OSModule.platform_system
platform_release = OSModule.platform_release
platform_version = OSModule.platform_version
platform_machine = OSModule.platform_machine
platform_processor = OSModule.platform_processor
python_version = OSModule.python_version
getcwd = OSModule.getcwd
chdir = OSModule.chdir
getenv = OSModule.getenv
setenv = OSModule.setenv
unsetenv = OSModule.unsetenv
listenv = OSModule.listenv
get_home_dir = OSModule.get_home_dir
get_temp_dir = OSModule.get_temp_dir
get_username = OSModule.get_username
get_hostname = OSModule.get_hostname
get_pid = OSModule.get_pid
get_ppid = OSModule.get_ppid
cpu_count = OSModule.cpu_count
execute = OSModule.execute
path_join = OSModule.path_join
path_split = OSModule.path_split
path_dirname = OSModule.path_dirname
path_basename = OSModule.path_basename
path_exists = OSModule.path_exists
path_isfile = OSModule.path_isfile
path_isdir = OSModule.path_isdir
path_abspath = OSModule.path_abspath
path_realpath = OSModule.path_realpath
get_file_size = OSModule.get_file_size
get_file_mtime = OSModule.get_file_mtime
get_file_atime = OSModule.get_file_atime
get_file_ctime = OSModule.get_file_ctime
get_path_separator = OSModule.get_path_separator
get_path_delimiter = OSModule.get_path_delimiter
get_line_separator = OSModule.get_line_separator
