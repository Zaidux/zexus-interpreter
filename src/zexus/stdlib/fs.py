"""File System module for Zexus standard library."""

import os
import shutil
import glob as glob_module
from pathlib import Path
from typing import List, Dict, Any


class FileSystemModule:
    """Provides file system operations."""

    @staticmethod
    def read_file(path: str, encoding: str = 'utf-8') -> str:
        """Read entire file as text."""
        with open(path, 'r', encoding=encoding) as f:
            return f.read()

    @staticmethod
    def write_file(path: str, content: str, encoding: str = 'utf-8') -> None:
        """Write text to file."""
        # Create parent directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)

    @staticmethod
    def append_file(path: str, content: str, encoding: str = 'utf-8') -> None:
        """Append text to file."""
        # Create parent directory if it doesn't exist (for consistency with write_file)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding=encoding) as f:
            f.write(content)

    @staticmethod
    def read_binary(path: str) -> bytes:
        """Read file as binary."""
        with open(path, 'rb') as f:
            return f.read()

    @staticmethod
    def write_binary(path: str, data: bytes) -> None:
        """Write binary data to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    @staticmethod
    def exists(path: str) -> bool:
        """Check if file or directory exists."""
        return os.path.exists(path)

    @staticmethod
    def is_file(path: str) -> bool:
        """Check if path is a file."""
        return os.path.isfile(path)

    @staticmethod
    def is_dir(path: str) -> bool:
        """Check if path is a directory."""
        return os.path.isdir(path)

    @staticmethod
    def mkdir(path: str, parents: bool = True) -> None:
        """Create directory."""
        Path(path).mkdir(parents=parents, exist_ok=True)

    @staticmethod
    def rmdir(path: str, recursive: bool = False) -> None:
        """Remove directory."""
        if recursive:
            shutil.rmtree(path)
        else:
            os.rmdir(path)

    @staticmethod
    def remove(path: str) -> None:
        """Remove file."""
        os.remove(path)

    @staticmethod
    def rename(old_path: str, new_path: str) -> None:
        """Rename/move file or directory."""
        os.rename(old_path, new_path)

    @staticmethod
    def copy_file(src: str, dst: str) -> None:
        """Copy file."""
        shutil.copy2(src, dst)

    @staticmethod
    def copy_dir(src: str, dst: str) -> None:
        """Copy directory recursively."""
        shutil.copytree(src, dst)

    @staticmethod
    def list_dir(path: str = '.') -> List[str]:
        """List directory contents."""
        return os.listdir(path)

    @staticmethod
    def walk(path: str) -> List[Dict[str, Any]]:
        """Walk directory tree."""
        result = []
        for root, dirs, files in os.walk(path):
            result.append({
                'root': root,
                'dirs': dirs,
                'files': files
            })
        return result

    @staticmethod
    def glob(pattern: str, recursive: bool = False) -> List[str]:
        """Find files matching pattern."""
        return glob_module.glob(pattern, recursive=recursive)

    @staticmethod
    def get_size(path: str) -> int:
        """Get file size in bytes."""
        return os.path.getsize(path)

    @staticmethod
    def get_cwd() -> str:
        """Get current working directory."""
        return os.getcwd()

    @staticmethod
    def chdir(path: str) -> None:
        """Change current working directory."""
        os.chdir(path)

    @staticmethod
    def abs_path(path: str) -> str:
        """Get absolute path."""
        return os.path.abspath(path)

    @staticmethod
    def join(*paths: str) -> str:
        """Join path components."""
        return os.path.join(*paths)

    @staticmethod
    def basename(path: str) -> str:
        """Get file name from path."""
        return os.path.basename(path)

    @staticmethod
    def dirname(path: str) -> str:
        """Get directory name from path."""
        return os.path.dirname(path)

    @staticmethod
    def splitext(path: str) -> tuple:
        """Split file extension."""
        return os.path.splitext(path)

    @staticmethod
    def get_stat(path: str) -> Dict[str, Any]:
        """Get file statistics."""
        stat = os.stat(path)
        return {
            'size': stat.st_size,
            'atime': stat.st_atime,
            'mtime': stat.st_mtime,
            'ctime': stat.st_ctime,
            'mode': stat.st_mode,
        }


# Export functions for easy access
read_file = FileSystemModule.read_file
write_file = FileSystemModule.write_file
append_file = FileSystemModule.append_file
exists = FileSystemModule.exists
is_file = FileSystemModule.is_file
is_dir = FileSystemModule.is_dir
mkdir = FileSystemModule.mkdir
rmdir = FileSystemModule.rmdir
remove = FileSystemModule.remove
rename = FileSystemModule.rename
copy_file = FileSystemModule.copy_file
copy_dir = FileSystemModule.copy_dir
list_dir = FileSystemModule.list_dir
walk = FileSystemModule.walk
glob = FileSystemModule.glob
get_size = FileSystemModule.get_size
get_cwd = FileSystemModule.get_cwd
chdir = FileSystemModule.chdir
abs_path = FileSystemModule.abs_path
join = FileSystemModule.join
basename = FileSystemModule.basename
dirname = FileSystemModule.dirname
splitext = FileSystemModule.splitext
get_stat = FileSystemModule.get_stat
