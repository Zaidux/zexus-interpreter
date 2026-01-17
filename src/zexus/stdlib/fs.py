"""File System module for Zexus standard library."""

import os
import shutil
import glob as glob_module
from pathlib import Path
from typing import List, Dict, Any, Optional


class PathTraversalError(Exception):
    """Raised when path traversal attack is detected."""
    pass


class FileSystemModule:
    """Provides file system operations with path traversal protection."""
    
    # Allowed base directories for file operations
    # If None, uses current working directory
    _allowed_base_dirs: Optional[List[str]] = None
    _strict_mode: bool = True  # Enable path validation by default
    
    @classmethod
    def configure_security(cls, allowed_dirs: Optional[List[str]] = None, strict: bool = True):
        """
        Configure file system security settings.
        
        Args:
            allowed_dirs: List of allowed base directories. None = use CWD only.
            strict: Enable strict path validation
        """
        cls._allowed_base_dirs = allowed_dirs
        cls._strict_mode = strict
    
    @classmethod
    def _validate_path(cls, path: str, operation: str = "access") -> str:
        """
        Validate path to prevent traversal attacks.
        
        Args:
            path: User-provided path
            operation: Type of operation (for error messages)
            
        Returns:
            Validated absolute path
            
        Raises:
            PathTraversalError: If path traversal detected
        """
        if not cls._strict_mode:
            return path
        
        # Convert to absolute path
        abs_path = Path(path).resolve()
        
        # Check for common traversal patterns
        path_str = str(path)
        if '..' in path_str:
            # Allow .. only if it doesn't escape allowed directories
            pass  # Will be checked below
        
        # Determine allowed base directories
        if cls._allowed_base_dirs is None:
            # Default: only allow access within CWD
            allowed_bases = [Path.cwd().resolve()]
        else:
            allowed_bases = [Path(d).resolve() for d in cls._allowed_base_dirs]
        
        # Check if resolved path is within allowed directories
        is_allowed = False
        for base in allowed_bases:
            try:
                abs_path.relative_to(base)
                is_allowed = True
                break
            except ValueError:
                continue
        
        if not is_allowed:
            raise PathTraversalError(
                f"Path traversal detected: '{path}' resolves to '{abs_path}' "
                f"which is outside allowed directories. "
                f"Allowed: {[str(b) for b in allowed_bases]}"
            )
        
        return str(abs_path)

    @staticmethod
    def read_file(path: str, encoding: str = 'utf-8') -> str:
        """Read entire file as text."""
        validated_path = FileSystemModule._validate_path(path, "read")
        with open(validated_path, 'r', encoding=encoding) as f:
            return f.read()

    @staticmethod
    def write_file(path: str, content: str, encoding: str = 'utf-8') -> None:
        """Write text to file."""
        validated_path = FileSystemModule._validate_path(path, "write")
        # Create parent directory if it doesn't exist
        Path(validated_path).parent.mkdir(parents=True, exist_ok=True)
        with open(validated_path, 'w', encoding=encoding) as f:
            f.write(content)

    @staticmethod
    def append_file(path: str, content: str, encoding: str = 'utf-8') -> None:
        """Append text to file."""
        validated_path = FileSystemModule._validate_path(path, "append")
        # Create parent directory if it doesn't exist (for consistency with write_file)
        Path(validated_path).parent.mkdir(parents=True, exist_ok=True)
        with open(validated_path, 'a', encoding=encoding) as f:
            f.write(content)

    @staticmethod
    def read_binary(path: str) -> bytes:
        """Read file as binary."""
        validated_path = FileSystemModule._validate_path(path, "read_binary")
        with open(validated_path, 'rb') as f:
            return f.read()

    @staticmethod
    def write_binary(path: str, data: bytes) -> None:
        """Write binary data to file."""
        validated_path = FileSystemModule._validate_path(path, "write_binary")
        Path(validated_path).parent.mkdir(parents=True, exist_ok=True)
        with open(validated_path, 'wb') as f:
            f.write(data)

    @staticmethod
    def exists(path: str) -> bool:
        """Check if file or directory exists."""
        try:
            validated_path = FileSystemModule._validate_path(path, "exists")
            return os.path.exists(validated_path)
        except PathTraversalError:
            return False  # Return False for invalid paths instead of error

    @staticmethod
    def is_file(path: str) -> bool:
        """Check if path is a file."""
        try:
            validated_path = FileSystemModule._validate_path(path, "is_file")
            return os.path.isfile(validated_path)
        except PathTraversalError:
            return False

    @staticmethod
    def is_dir(path: str) -> bool:
        """Check if path is a directory."""
        try:
            validated_path = FileSystemModule._validate_path(path, "is_dir")
            return os.path.isdir(validated_path)
        except PathTraversalError:
            return False

    @staticmethod
    def mkdir(path: str, parents: bool = True) -> None:
        validated_path = FileSystemModule._validate_path(path, "remove")
        os.remove(validated_path)

    @staticmethod
    def mkdir(path: str, parents: bool = True) -> None:
        """Create directory."""
        validated_path = FileSystemModule._validate_path(path, "mkdir")
        Path(validated_path).mkdir(parents=parents, exist_ok=True)

    @staticmethod
    def rmdir(path: str, recursive: bool = False) -> None:
        """Remove directory."""
        validated_path = FileSystemModule._validate_path(path, "rmdir")
        if recursive:
            shutil.rmtree(validated_path)
        else:
            os.rmdir(validated_path)

    @staticmethod
    def remove(path: str) -> None:
        """Remove file."""
        validated_path = FileSystemModule._validate_path(path, "remove")
        os.remove(validated_path)

    @staticmethod
    def rename(old_path: str, new_path: str) -> None:
        """Rename/move file or directory."""
        validated_old = FileSystemModule._validate_path(old_path, "rename_source")
        validated_new = FileSystemModule._validate_path(new_path, "rename_dest")
        os.rename(validated_old, validated_new)

    @staticmethod
    def copy_file(src: str, dst: str) -> None:
        """Copy file."""
        validated_src = FileSystemModule._validate_path(src, "copy_source")
        validated_dst = FileSystemModule._validate_path(dst, "copy_dest")
        shutil.copy2(validated_src, validated_dst)

    @staticmethod
    def copy_dir(src: str, dst: str) -> None:
        """Copy directory recursively."""
        validated_src = FileSystemModule._validate_path(src, "copy_dir_source")
        validated_dst = FileSystemModule._validate_path(dst, "copy_dir_dest")
        shutil.copytree(validated_src, validated_dst, dirs_exist_ok=True)

    @staticmethod
    def list_dir(path: str = '.') -> List[str]:
        """List directory contents."""
        validated_path = FileSystemModule._validate_path(path, "list_dir")
        return os.listdir(validated_path)


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
