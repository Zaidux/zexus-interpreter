"""
Virtual filesystem and memory layer for sandboxed execution.

Provides isolated file access and memory quotas for plugins.
Each plugin operates in a restricted filesystem namespace.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import os
import sys


class FileAccessMode(Enum):
    """File access permission levels."""
    NONE = 0
    READ = 1
    WRITE = 2
    READ_WRITE = 3
    EXECUTE = 4


@dataclass
class MemoryQuota:
    """Memory usage quota for a plugin/sandbox."""
    max_bytes: int
    warning_threshold: float = 0.8  # Warn at 80%
    current_usage: int = 0
    
    def is_over_quota(self) -> bool:
        """Check if quota exceeded."""
        return self.current_usage > self.max_bytes
    
    def is_over_warning(self) -> bool:
        """Check if warning threshold exceeded."""
        return self.current_usage > (self.max_bytes * self.warning_threshold)
    
    def get_available(self) -> int:
        """Get available memory."""
        return max(0, self.max_bytes - self.current_usage)
    
    def allocate(self, size: int) -> bool:
        """Try to allocate memory."""
        if self.current_usage + size > self.max_bytes:
            return False
        self.current_usage += size
        return True
    
    def deallocate(self, size: int):
        """Deallocate memory."""
        self.current_usage = max(0, self.current_usage - size)


@dataclass
class FileSystemPath:
    """Represents a path in the virtual filesystem."""
    real_path: str  # Actual filesystem path
    virtual_path: str  # How it appears to sandbox
    access_mode: FileAccessMode = FileAccessMode.READ


class SandboxFileSystem:
    """
    Virtual filesystem for a sandbox/plugin.
    
    Provides isolated file access with path restrictions.
    Each plugin sees only paths it's granted access to.
    """
    
    def __init__(self, sandbox_id: str):
        """Initialize sandbox filesystem."""
        self.sandbox_id = sandbox_id
        self.mounts: Dict[str, FileSystemPath] = {}  # virtual_path -> FileSystemPath
        self.access_log: List[Dict] = []
        
    def mount(self, real_path: str, virtual_path: str, 
             access_mode: FileAccessMode = FileAccessMode.READ) -> bool:
        """
        Mount a real filesystem path into the sandbox.
        
        Args:
            real_path: Actual path on filesystem
            virtual_path: Path as seen by sandbox
            access_mode: READ, WRITE, READ_WRITE, or EXECUTE
            
        Returns:
            True if mounted successfully
        """
        if virtual_path in self.mounts:
            return False  # Already mounted
        
        # Normalize paths
        real_path = os.path.normpath(real_path)
        virtual_path = os.path.normpath(virtual_path)
        
        if not os.path.exists(real_path):
            return False
        
        mount = FileSystemPath(real_path, virtual_path, access_mode)
        self.mounts[virtual_path] = mount
        return True
    
    def unmount(self, virtual_path: str) -> bool:
        """Unmount a path from the sandbox."""
        if virtual_path in self.mounts:
            del self.mounts[virtual_path]
            return True
        return False
    
    def resolve_path(self, virtual_path: str) -> Optional[Tuple[str, FileAccessMode]]:
        """
        Resolve a virtual path to a real path.
        
        Returns:
            (real_path, access_mode) if accessible, None otherwise
        """
        virtual_path = os.path.normpath(virtual_path)
        
        # Check exact mount
        if virtual_path in self.mounts:
            mount = self.mounts[virtual_path]
            return mount.real_path, mount.access_mode
        
        # Check if under a mounted directory
        for mount_point, mount in self.mounts.items():
            if virtual_path.startswith(mount_point + os.sep):
                # Get relative path
                rel_path = virtual_path[len(mount_point)+1:]
                real_path = os.path.normpath(os.path.join(mount.real_path, rel_path))
                
                # Ensure real path is still under mounted root
                if real_path.startswith(mount.real_path):
                    return real_path, mount.access_mode
        
        return None
    
    def can_read(self, virtual_path: str) -> bool:
        """Check if path can be read."""
        result = self.resolve_path(virtual_path)
        if result is None:
            return False
        _, mode = result
        return mode in (FileAccessMode.READ, FileAccessMode.READ_WRITE)
    
    def can_write(self, virtual_path: str) -> bool:
        """Check if path can be written."""
        result = self.resolve_path(virtual_path)
        if result is None:
            return False
        _, mode = result
        return mode in (FileAccessMode.WRITE, FileAccessMode.READ_WRITE)
    
    def log_access(self, operation: str, virtual_path: str, allowed: bool, reason: str = ""):
        """Log a filesystem access attempt."""
        entry = {
            "operation": operation,
            "virtual_path": virtual_path,
            "allowed": allowed,
            "reason": reason
        }
        self.access_log.append(entry)
    
    def get_access_log(self) -> List[Dict]:
        """Get filesystem access log."""
        return self.access_log.copy()


class VirtualFileSystemManager:
    """
    Manages virtual filesystems for multiple sandboxes.
    
    Coordinates sandbox creation, isolation, and resource cleanup.
    """
    
    def __init__(self):
        """Initialize filesystem manager."""
        self.sandboxes: Dict[str, SandboxFileSystem] = {}
        self.memory_quotas: Dict[str, MemoryQuota] = {}
        self.default_memory_quota = 1024 * 1024 * 100  # 100MB default
    
    def create_sandbox(self, sandbox_id: str, memory_quota_mb: int = 100) -> SandboxFileSystem:
        """
        Create a new sandbox.
        
        Args:
            sandbox_id: Unique identifier for sandbox
            memory_quota_mb: Memory limit in MB
            
        Returns:
            SandboxFileSystem instance
        """
        if sandbox_id in self.sandboxes:
            return self.sandboxes[sandbox_id]
        
        sandbox = SandboxFileSystem(sandbox_id)
        self.sandboxes[sandbox_id] = sandbox
        
        # Set memory quota
        quota = MemoryQuota(max_bytes=memory_quota_mb * 1024 * 1024)
        self.memory_quotas[sandbox_id] = quota
        
        return sandbox
    
    def get_sandbox(self, sandbox_id: str) -> Optional[SandboxFileSystem]:
        """Get existing sandbox."""
        return self.sandboxes.get(sandbox_id)
    
    def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete a sandbox and cleanup."""
        if sandbox_id not in self.sandboxes:
            return False
        
        del self.sandboxes[sandbox_id]
        if sandbox_id in self.memory_quotas:
            del self.memory_quotas[sandbox_id]
        
        return True
    
    def allocate_memory(self, sandbox_id: str, size: int) -> bool:
        """
        Allocate memory for a sandbox.
        
        Returns:
            True if allocated, False if quota exceeded
        """
        if sandbox_id not in self.memory_quotas:
            return False
        
        quota = self.memory_quotas[sandbox_id]
        return quota.allocate(size)
    
    def deallocate_memory(self, sandbox_id: str, size: int):
        """Deallocate memory from a sandbox."""
        if sandbox_id in self.memory_quotas:
            quota = self.memory_quotas[sandbox_id]
            quota.deallocate(size)
    
    def get_memory_quota(self, sandbox_id: str) -> Optional[MemoryQuota]:
        """Get memory quota for a sandbox."""
        return self.memory_quotas.get(sandbox_id)
    
    def list_sandboxes(self) -> List[str]:
        """List all active sandboxes."""
        return list(self.sandboxes.keys())


class StandardMounts:
    """Standard filesystem mount configurations."""
    
    @staticmethod
    def read_only_home() -> Dict[str, Tuple[str, FileAccessMode]]:
        """Read-only home directory access."""
        return {
            "/home": (os.path.expanduser("~"), FileAccessMode.READ),
        }
    
    @staticmethod
    def temp_directory() -> Dict[str, Tuple[str, FileAccessMode]]:
        """Temporary directory with read-write access."""
        return {
            "/tmp": ("/tmp", FileAccessMode.READ_WRITE),
        }
    
    @staticmethod
    def app_data() -> Dict[str, Tuple[str, FileAccessMode]]:
        """Application data directory."""
        return {
            "/app": (os.path.expanduser("~/.app"), FileAccessMode.READ_WRITE),
        }
    
    @staticmethod
    def system_readonly() -> Dict[str, Tuple[str, FileAccessMode]]:
        """System paths (read-only)."""
        return {
            "/etc": ("/etc", FileAccessMode.READ),
            "/usr/share": ("/usr/share", FileAccessMode.READ),
        }


class SandboxBuilder:
    """Builder for configuring sandboxes with standard mounts."""
    
    def __init__(self, manager: VirtualFileSystemManager, sandbox_id: str):
        """Initialize builder."""
        self.manager = manager
        self.sandbox_id = sandbox_id
        self.sandbox = manager.create_sandbox(sandbox_id)
        self.mounts: Dict[str, Tuple[str, FileAccessMode]] = {}
    
    def add_mount(self, virtual_path: str, real_path: str, 
                 access_mode: FileAccessMode = FileAccessMode.READ) -> 'SandboxBuilder':
        """Add a mount point."""
        self.mounts[virtual_path] = (real_path, access_mode)
        return self
    
    def with_temp_access(self) -> 'SandboxBuilder':
        """Add temporary directory access."""
        self.mounts.update(StandardMounts.temp_directory())
        return self
    
    def with_home_readonly(self) -> 'SandboxBuilder':
        """Add read-only home directory."""
        self.mounts.update(StandardMounts.read_only_home())
        return self
    
    def with_app_data(self) -> 'SandboxBuilder':
        """Add application data directory."""
        self.mounts.update(StandardMounts.app_data())
        return self
    
    def build(self) -> SandboxFileSystem:
        """Build the sandbox with configured mounts."""
        for virtual_path, (real_path, access_mode) in self.mounts.items():
            self.sandbox.mount(real_path, virtual_path, access_mode)
        return self.sandbox


# Predefined sandbox configurations

SANDBOX_PRESETS = {
    "read_only": {
        "description": "Read-only file access",
        "mounts": {
            "/data": ("./data", FileAccessMode.READ),
        },
        "memory_quota_mb": 50
    },
    
    "trusted": {
        "description": "Trusted code with full access",
        "mounts": {
            "/": ("/", FileAccessMode.READ_WRITE),
        },
        "memory_quota_mb": 500
    },
    
    "isolated": {
        "description": "Isolated with only temp directory",
        "mounts": {
            "/tmp": ("/tmp", FileAccessMode.READ_WRITE),
        },
        "memory_quota_mb": 100
    },
    
    "plugin": {
        "description": "Plugin with app data and temp",
        "mounts": {
            "/app": (os.path.expanduser("~/.zexus/plugins"), FileAccessMode.READ_WRITE),
            "/tmp": ("/tmp", FileAccessMode.READ_WRITE),
        },
        "memory_quota_mb": 200
    },
}
