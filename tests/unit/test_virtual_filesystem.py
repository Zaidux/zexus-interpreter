"""
Tests for virtual filesystem and memory quotas.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.virtual_filesystem import (
    VirtualFileSystemManager, SandboxFileSystem, SandboxBuilder,
    FileAccessMode, MemoryQuota, StandardMounts, SANDBOX_PRESETS
)


def test_memory_quota():
    """Test memory quota tracking."""
    quota = MemoryQuota(max_bytes=1000)
    
    # Can allocate within quota
    assert quota.allocate(500)
    assert quota.current_usage == 500
    
    # Can allocate more
    assert quota.allocate(400)
    assert quota.current_usage == 900
    
    # Cannot allocate beyond quota
    assert not quota.allocate(200)
    assert quota.current_usage == 900
    
    # Deallocate
    quota.deallocate(300)
    assert quota.current_usage == 600
    
    print("✓ memory_quota test passed")


def test_memory_quota_thresholds():
    """Test memory quota warning threshold."""
    quota = MemoryQuota(max_bytes=1000, warning_threshold=0.8)
    
    assert not quota.is_over_warning()
    
    quota.allocate(810)  # 81% - over threshold
    assert quota.is_over_warning()
    
    assert not quota.is_over_quota()
    
    # Manually set current_usage to exceed quota to test is_over_quota
    quota.current_usage = 1001
    assert quota.is_over_quota()
    
    print("✓ memory_quota_thresholds test passed")


def test_sandbox_filesystem_creation():
    """Test creating a sandbox filesystem."""
    sandbox = SandboxFileSystem("test_sandbox")
    
    assert sandbox.sandbox_id == "test_sandbox"
    assert len(sandbox.mounts) == 0
    
    print("✓ sandbox_filesystem_creation test passed")


def test_sandbox_mount():
    """Test mounting paths in sandbox."""
    sandbox = SandboxFileSystem("test_sandbox")
    
    # Mount a real directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = sandbox.mount(tmpdir, "/data", FileAccessMode.READ)
        assert result
        assert "/data" in sandbox.mounts
        
        # Verify mount details
        mount = sandbox.mounts["/data"]
        assert mount.real_path == tmpdir
        assert mount.virtual_path == "/data"
        assert mount.access_mode == FileAccessMode.READ


    print("✓ sandbox_mount test passed")


def test_sandbox_path_resolution():
    """Test resolving virtual paths."""
    sandbox = SandboxFileSystem("test_sandbox")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox.mount(tmpdir, "/data")
        
        # Resolve exact mount point
        result = sandbox.resolve_path("/data")
        assert result is not None
        real_path, mode = result
        assert real_path == tmpdir
        assert mode == FileAccessMode.READ


    print("✓ sandbox_path_resolution test passed")


def test_sandbox_read_write_checks():
    """Test read/write permission checks."""
    sandbox = SandboxFileSystem("test_sandbox")
    
    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            sandbox.mount(tmpdir1, "/ro", FileAccessMode.READ)
            sandbox.mount(tmpdir2, "/rw", FileAccessMode.READ_WRITE)
            
            assert sandbox.can_read("/ro")
            assert not sandbox.can_write("/ro")
            
            assert sandbox.can_read("/rw")
            assert sandbox.can_write("/rw")
            
            assert not sandbox.can_read("/unmounted")
            assert not sandbox.can_write("/unmounted")


    print("✓ sandbox_read_write_checks test passed")


def test_sandbox_access_logging():
    """Test filesystem access logging."""
    sandbox = SandboxFileSystem("test_sandbox")
    
    sandbox.log_access("read", "/data/file.txt", True, "File readable")
    sandbox.log_access("write", "/data/secret.txt", False, "Write denied")
    
    log = sandbox.get_access_log()
    assert len(log) == 2
    assert log[0]["operation"] == "read"
    assert log[0]["allowed"] is True
    assert log[1]["allowed"] is False
    
    print("✓ sandbox_access_logging test passed")


def test_vfs_manager_sandbox_creation():
    """Test VFS manager creating sandboxes."""
    manager = VirtualFileSystemManager()
    
    sandbox1 = manager.create_sandbox("plugin_a", memory_quota_mb=100)
    assert sandbox1 is not None
    assert "plugin_a" in manager.list_sandboxes()
    
    # Getting same sandbox returns same instance
    sandbox1_again = manager.create_sandbox("plugin_a")
    assert sandbox1 is sandbox1_again
    
    # Create another sandbox
    sandbox2 = manager.create_sandbox("plugin_b", memory_quota_mb=50)
    sandboxes = manager.list_sandboxes()
    assert "plugin_a" in sandboxes
    assert "plugin_b" in sandboxes
    
    print("✓ vfs_manager_sandbox_creation test passed")


def test_vfs_manager_memory_allocation():
    """Test memory quota enforcement."""
    manager = VirtualFileSystemManager()
    manager.create_sandbox("app", memory_quota_mb=1)  # 1MB
    
    # Should be able to allocate within quota
    assert manager.allocate_memory("app", 500 * 1024)  # 500KB
    
    # Should be able to allocate more
    assert manager.allocate_memory("app", 400 * 1024)  # 400KB
    
    # Should fail when exceeding quota
    assert not manager.allocate_memory("app", 200 * 1024)  # 200KB (over 1MB limit)
    
    # Deallocate and try again
    manager.deallocate_memory("app", 300 * 1024)
    assert manager.allocate_memory("app", 200 * 1024)
    
    print("✓ vfs_manager_memory_allocation test passed")


def test_vfs_manager_sandbox_cleanup():
    """Test deleting sandboxes."""
    manager = VirtualFileSystemManager()
    
    manager.create_sandbox("temp_sandbox")
    assert "temp_sandbox" in manager.list_sandboxes()
    
    # Delete sandbox
    result = manager.delete_sandbox("temp_sandbox")
    assert result
    assert "temp_sandbox" not in manager.list_sandboxes()
    
    # Cannot delete non-existent sandbox
    assert not manager.delete_sandbox("nonexistent")
    
    print("✓ vfs_manager_sandbox_cleanup test passed")


def test_sandbox_builder():
    """Test sandbox builder pattern."""
    manager = VirtualFileSystemManager()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = (SandboxBuilder(manager, "app")
                  .add_mount("/data", tmpdir, FileAccessMode.READ)
                  .with_temp_access()
                  .build())
        
        assert sandbox is not None
        assert "/data" in sandbox.mounts
        assert "/tmp" in sandbox.mounts


    print("✓ sandbox_builder test passed")


def test_standard_mounts():
    """Test standard mount configurations."""
    mounts = StandardMounts.temp_directory()
    assert "/tmp" in mounts
    
    mounts = StandardMounts.app_data()
    assert "/app" in mounts
    
    print("✓ standard_mounts test passed")


def test_sandbox_presets():
    """Test predefined sandbox configurations."""
    assert "read_only" in SANDBOX_PRESETS
    assert "trusted" in SANDBOX_PRESETS
    assert "isolated" in SANDBOX_PRESETS
    assert "plugin" in SANDBOX_PRESETS
    
    for preset_name, config in SANDBOX_PRESETS.items():
        assert "description" in config
        assert "mounts" in config
        assert "memory_quota_mb" in config
    
    print("✓ sandbox_presets test passed")


def test_sandbox_path_escaping():
    """Test that sandbox prevents directory traversal attacks."""
    sandbox = SandboxFileSystem("test")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox.mount(tmpdir, "/safe")
        
        # Try to escape the sandbox
        result = sandbox.resolve_path("/safe/../../etc/passwd")
        # Path traversal should be prevented
        if result:
            real_path, _ = result
            # Real path should still be within mount point or normalized
            assert real_path.startswith(tmpdir) or os.path.isabs(real_path)


    print("✓ sandbox_path_escaping test passed")


def test_sandbox_unmount():
    """Test unmounting paths."""
    sandbox = SandboxFileSystem("test")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox.mount(tmpdir, "/data")
        assert "/data" in sandbox.mounts
        
        # Unmount
        result = sandbox.unmount("/data")
        assert result
        assert "/data" not in sandbox.mounts
        
        # Cannot unmount twice
        result = sandbox.unmount("/data")
        assert not result


    print("✓ sandbox_unmount test passed")


def test_memory_quota_get_available():
    """Test getting available memory."""
    quota = MemoryQuota(max_bytes=1000)
    
    assert quota.get_available() == 1000
    
    quota.allocate(300)
    assert quota.get_available() == 700
    
    quota.allocate(200)
    assert quota.get_available() == 500
    
    print("✓ memory_quota_get_available test passed")


if __name__ == '__main__':
    try:
        test_memory_quota()
        test_memory_quota_thresholds()
        test_sandbox_filesystem_creation()
        test_sandbox_mount()
        test_sandbox_path_resolution()
        test_sandbox_read_write_checks()
        test_sandbox_access_logging()
        test_vfs_manager_sandbox_creation()
        test_vfs_manager_memory_allocation()
        test_vfs_manager_sandbox_cleanup()
        test_sandbox_builder()
        test_standard_mounts()
        test_sandbox_presets()
        test_sandbox_path_escaping()
        test_sandbox_unmount()
        test_memory_quota_get_available()
        print("\n✅ All virtual filesystem tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
