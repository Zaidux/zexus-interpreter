"""
Tests for capability-based security system.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zexus.capability_system import (
    CapabilityManager, Capability, CapabilityPolicy, AllowAllPolicy,
    DenyAllPolicy, SelectivePolicy, CapabilityLevel, CapabilityAuditLog,
    CAPABILITY_SETS
)


def test_base_capabilities():
    """Test that base capabilities are always available."""
    manager = CapabilityManager()
    
    # Base capabilities should be available
    assert manager.has_capability("app", "core.language")
    assert manager.has_capability("app", "core.math")
    assert manager.has_capability("app", "core.strings")
    print("✓ base_capabilities test passed")


def test_grant_capability():
    """Test granting individual capabilities."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    manager.grant_capability("plugin_a", "io.read")
    assert manager.has_capability("plugin_a", "io.read")
    assert not manager.has_capability("plugin_b", "io.read")
    print("✓ grant_capability test passed")


def test_grant_multiple_capabilities():
    """Test granting multiple capabilities at once."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    manager.grant_capabilities("plugin_a", ["io.read", "io.write", "network.http"])
    
    assert manager.has_capability("plugin_a", "io.read")
    assert manager.has_capability("plugin_a", "io.write")
    assert manager.has_capability("plugin_a", "network.http")
    assert not manager.has_capability("plugin_a", "exec.shell")
    print("✓ grant_multiple_capabilities test passed")


def test_check_capability_with_reason():
    """Test capability check returns reason."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    # Denied capability
    allowed, reason = manager.check_capability("plugin_a", "io.read")
    assert not allowed
    assert "not granted" in reason
    
    # Granted capability
    manager.grant_capability("plugin_a", "io.read")
    allowed, reason = manager.check_capability("plugin_a", "io.read")
    assert allowed
    assert "granted" in reason
    print("✓ check_capability_with_reason test passed")


def test_require_capability_raises():
    """Test that require_capability raises PermissionError if denied."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    try:
        manager.require_capability("plugin_a", "io.read")
        assert False, "Should have raised PermissionError"
    except PermissionError as e:
        assert "cannot access" in str(e)
    
    # After granting, should not raise
    manager.grant_capability("plugin_a", "io.read")
    manager.require_capability("plugin_a", "io.read")  # Should not raise
    print("✓ require_capability_raises test passed")


def test_allow_all_policy():
    """Test AllowAllPolicy grants all capabilities."""
    manager = CapabilityManager(default_policy=AllowAllPolicy())
    
    # Any capability should be allowed
    assert manager.has_capability("plugin", "io.read")
    assert manager.has_capability("plugin", "exec.shell")
    assert manager.has_capability("plugin", "custom.capability")
    print("✓ allow_all_policy test passed")


def test_deny_all_policy():
    """Test DenyAllPolicy denies non-base capabilities."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    # Base capabilities always available
    assert manager.has_capability("plugin", "core.language")
    # Privileged capabilities denied
    assert not manager.has_capability("plugin", "io.read")
    assert not manager.has_capability("plugin", "exec.shell")
    assert not manager.has_capability("plugin", "custom.capability")
    print("✓ deny_all_policy test passed")


def test_selective_policy():
    """Test SelectivePolicy allows only specified capabilities."""
    allowed = ["io.read", "core.math"]
    manager = CapabilityManager(default_policy=SelectivePolicy(allowed))
    
    assert manager.has_capability("plugin", "io.read")
    assert manager.has_capability("plugin", "core.math")
    assert not manager.has_capability("plugin", "io.write")
    assert not manager.has_capability("plugin", "exec.shell")
    print("✓ selective_policy test passed")


def test_declare_requirements():
    """Test declaring required capabilities."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    manager.declare_required_capabilities("my_plugin", ["io.read", "network.http"])
    
    required = manager.get_required_capabilities("my_plugin")
    assert "io.read" in required
    assert "network.http" in required
    assert len(required) == 2
    print("✓ declare_requirements test passed")


def test_validate_requirements():
    """Test validating that requirements are met."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    manager.declare_required_capabilities("my_plugin", ["io.read", "network.http"])
    
    # Requirements not met initially
    valid, missing = manager.validate_requirements("my_plugin")
    assert not valid
    assert "io.read" in missing
    assert "network.http" in missing
    
    # Grant requirements
    manager.grant_capabilities("my_plugin", ["io.read", "network.http"])
    valid, missing = manager.validate_requirements("my_plugin")
    assert valid
    assert len(missing) == 0
    print("✓ validate_requirements test passed")


def test_get_granted_capabilities():
    """Test getting list of granted capabilities."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    manager.grant_capabilities("plugin_a", ["io.read", "io.write"])
    
    caps = manager.get_granted_capabilities("plugin_a")
    assert "io.read" in caps
    assert "io.write" in caps
    assert "exec.shell" not in caps
    print("✓ get_granted_capabilities test passed")


def test_audit_logging():
    """Test that capability checks are logged."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    # Make some checks
    manager.check_capability("plugin_a", "io.read")
    manager.grant_capability("plugin_a", "io.write")
    manager.check_capability("plugin_a", "io.write")
    
    log = manager.get_audit_log()
    assert len(log) > 0
    
    # Check log entries
    io_read_entries = [e for e in log if e["capability"] == "io.read"]
    assert len(io_read_entries) > 0
    assert io_read_entries[0]["granted"] is False
    
    io_write_entries = [e for e in log if e["capability"] == "io.write"]
    assert len(io_write_entries) > 0
    assert io_write_entries[0]["granted"] is True
    print("✓ audit_logging test passed")


def test_audit_statistics():
    """Test audit statistics."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    manager.grant_capability("plugin_a", "io.read")
    manager.check_capability("plugin_a", "io.read")
    manager.check_capability("plugin_a", "io.write")
    
    stats = manager.get_audit_statistics()
    assert stats.get("io.read:granted", 0) > 0
    assert stats.get("io.write:denied", 0) > 0
    print("✓ audit_statistics test passed")


def test_set_policy():
    """Test changing policies at runtime."""
    manager = CapabilityManager()
    
    # Start with DenyAll
    manager.set_policy(DenyAllPolicy())
    assert not manager.has_capability("plugin", "io.read")
    
    # Switch to AllowAll
    manager.set_policy(AllowAllPolicy())
    assert manager.has_capability("plugin", "io.read")
    
    # Switch to Selective
    manager.set_policy(SelectivePolicy(["io.write"]))
    assert not manager.has_capability("plugin", "io.read")
    assert manager.has_capability("plugin", "io.write")
    print("✓ set_policy test passed")


def test_capability_sets():
    """Test predefined capability sets."""
    # Check that standard sets exist
    assert "read_only" in CAPABILITY_SETS
    assert "io_full" in CAPABILITY_SETS
    assert "network" in CAPABILITY_SETS
    assert "crypto" in CAPABILITY_SETS
    assert "untrusted" in CAPABILITY_SETS
    assert "trusted" in CAPABILITY_SETS
    assert "system" in CAPABILITY_SETS
    
    # Check set structure
    for set_name, set_def in CAPABILITY_SETS.items():
        assert "capabilities" in set_def
        assert "description" in set_def
        assert isinstance(set_def["capabilities"], list)
    
    # Verify capabilities in sets are strings
    for set_name, set_def in CAPABILITY_SETS.items():
        for cap in set_def["capabilities"]:
            assert isinstance(cap, str)
    
    print("✓ capability_sets test passed")


def test_capability_policy_grant_deny():
    """Test CapabilityPolicy grant and deny."""
    policy = CapabilityPolicy("test_policy")
    
    policy.grant("io.read", CapabilityLevel.ALLOWED)
    assert policy.check("io.read") == CapabilityLevel.ALLOWED
    
    policy.deny("io.read")
    assert policy.check("io.read") == CapabilityLevel.DENY
    
    # Denying removes from grants
    policy.grant("io.write", CapabilityLevel.ALLOWED)
    assert policy.check("io.write") == CapabilityLevel.ALLOWED
    policy.deny("io.write")
    assert policy.check("io.write") == CapabilityLevel.DENY
    print("✓ capability_policy_grant_deny test passed")


def test_audit_log_filtering():
    """Test filtering audit log by capability and requester."""
    manager = CapabilityManager(default_policy=DenyAllPolicy())
    
    manager.grant_capability("plugin_a", "io.read")
    manager.check_capability("plugin_a", "io.read")
    manager.check_capability("plugin_b", "network.http")
    manager.grant_capability("plugin_c", "io.write")
    manager.check_capability("plugin_c", "io.write")
    
    # Filter by capability
    io_read_entries = manager.audit_log.get_entries(capability="io.read")
    assert all(e["capability"] == "io.read" for e in io_read_entries)
    
    # Filter by requester
    plugin_a_entries = manager.audit_log.get_entries(requester="plugin_a")
    assert all(e["requester"] == "plugin_a" for e in plugin_a_entries)
    
    print("✓ audit_log_filtering test passed")


if __name__ == '__main__':
    try:
        test_base_capabilities()
        test_grant_capability()
        test_grant_multiple_capabilities()
        test_check_capability_with_reason()
        test_require_capability_raises()
        test_allow_all_policy()
        test_deny_all_policy()
        test_selective_policy()
        test_declare_requirements()
        test_validate_requirements()
        test_get_granted_capabilities()
        test_audit_logging()
        test_audit_statistics()
        test_set_policy()
        test_capability_sets()
        test_capability_policy_grant_deny()
        test_audit_log_filtering()
        print("\n✅ All capability system tests passed!")
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
