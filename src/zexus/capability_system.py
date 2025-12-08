"""
Capability-based security system for Zexus.

Implements fine-grained access control through capability tokens.
Plugins declare required capabilities, and the evaluator enforces access.
"""

from typing import Set, Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class CapabilityLevel(Enum):
    """Capability access levels."""
    DENY = 0
    RESTRICTED = 1
    ALLOWED = 2
    UNRESTRICTED = 3


@dataclass
class Capability:
    """A capability token representing an action or resource access."""
    name: str
    level: CapabilityLevel = CapabilityLevel.ALLOWED
    reason: str = ""  # Why this capability is granted/denied
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Which plugin/policy granted this


@dataclass
class CapabilityRequest:
    """Request for capability access."""
    capability: str
    requester: str  # Plugin or code requesting access
    context: Dict = field(default_factory=dict)  # Request context
    timestamp: float = field(default_factory=time.time)


class CapabilityPolicy:
    """Base class for capability policies."""
    
    def __init__(self, name: str):
        self.name = name
        self.grants: Dict[str, CapabilityLevel] = {}
        self.denials: Set[str] = set()
    
    def grant(self, capability: str, level: CapabilityLevel = CapabilityLevel.ALLOWED):
        """Grant a capability at specified level."""
        self.grants[capability] = level
    
    def deny(self, capability: str):
        """Deny a capability."""
        self.denials.add(capability)
        if capability in self.grants:
            del self.grants[capability]
    
    def check(self, capability: str) -> CapabilityLevel:
        """Check capability status."""
        if capability in self.denials:
            return CapabilityLevel.DENY
        return self.grants.get(capability, CapabilityLevel.DENY)


class AllowAllPolicy(CapabilityPolicy):
    """Allow all capabilities (development mode)."""
    
    def __init__(self):
        super().__init__("allow_all")
    
    def check(self, capability: str) -> CapabilityLevel:
        """All capabilities allowed."""
        return CapabilityLevel.ALLOWED


class DenyAllPolicy(CapabilityPolicy):
    """Deny all capabilities (secure sandbox)."""
    
    def __init__(self):
        super().__init__("deny_all")
    
    def check(self, capability: str) -> CapabilityLevel:
        """All capabilities denied."""
        return CapabilityLevel.DENY


class SelectivePolicy(CapabilityPolicy):
    """Allow only specific capabilities."""
    
    def __init__(self, allowed_capabilities: List[str]):
        super().__init__("selective")
        for cap in allowed_capabilities:
            self.grant(cap, CapabilityLevel.ALLOWED)


class CapabilityAuditLog:
    """Log capability access for audit purposes."""
    
    def __init__(self):
        self.entries: List[Dict] = []
        self.statistics: Dict[str, int] = {}
    
    def log_request(self, request: CapabilityRequest, granted: bool, reason: str = ""):
        """Log a capability request."""
        entry = {
            "timestamp": request.timestamp,
            "capability": request.capability,
            "requester": request.requester,
            "granted": granted,
            "reason": reason,
            "context": request.context
        }
        self.entries.append(entry)
        
        # Update statistics
        key = f"{request.capability}:{'granted' if granted else 'denied'}"
        self.statistics[key] = self.statistics.get(key, 0) + 1
    
    def get_entries(self, capability: Optional[str] = None, requester: Optional[str] = None) -> List[Dict]:
        """Get audit log entries, optionally filtered."""
        entries = self.entries
        
        if capability:
            entries = [e for e in entries if e["capability"] == capability]
        if requester:
            entries = [e for e in entries if e["requester"] == requester]
        
        return entries
    
    def get_statistics(self) -> Dict[str, int]:
        """Get access statistics."""
        return self.statistics.copy()


class CapabilityManager:
    """
    Manages capabilities, policies, and access control.
    
    Enforces that code only accesses capabilities it has declared.
    Supports both compile-time declarations and runtime checks.
    """
    
    # Base capabilities always available
    BASE_CAPABILITIES = {
        "core.language",      # Core language features
        "core.control",       # Control flow
        "core.math",          # Math operations
        "core.strings",       # String operations
        "core.arrays",        # Array operations
        "core.objects",       # Object operations
    }
    
    # Privileged capabilities
    PRIVILEGED_CAPABILITIES = {
        "io.read",            # File reading
        "io.write",           # File writing
        "io.delete",          # File deletion
        "network.tcp",        # TCP connections
        "network.http",       # HTTP requests
        "crypto.keygen",      # Key generation
        "crypto.sign",        # Signing
        "exec.shell",         # Shell execution
        "exec.spawn",         # Process spawning
        "sys.env",            # Environment variables
        "sys.time",           # System time
        "sys.exit",           # Process exit
    }
    
    def __init__(self, default_policy: Optional[CapabilityPolicy] = None):
        """Initialize capability manager."""
        self.policy = default_policy or AllowAllPolicy()
        self.audit_log = CapabilityAuditLog()
        self.granted_capabilities: Dict[str, Set[str]] = {}  # requester -> capabilities
        self.required_capabilities: Dict[str, Set[str]] = {}  # requester -> required caps
        
        # Initialize with base capabilities
        for cap in self.BASE_CAPABILITIES:
            self.policy.grant(cap, CapabilityLevel.ALLOWED)
    
    def set_policy(self, policy: CapabilityPolicy):
        """Set the capability policy."""
        self.policy = policy
    
    def declare_required_capabilities(self, requester: str, capabilities: List[str]):
        """Declare that a module/plugin requires certain capabilities."""
        self.required_capabilities[requester] = set(capabilities)
    
    def grant_capability(self, requester: str, capability: str):
        """Grant a capability to a requester."""
        if requester not in self.granted_capabilities:
            self.granted_capabilities[requester] = set()
        self.granted_capabilities[requester].add(capability)
    
    def grant_capabilities(self, requester: str, capabilities: List[str]):
        """Grant multiple capabilities to a requester."""
        for cap in capabilities:
            self.grant_capability(requester, cap)
    
    def check_capability(self, requester: str, capability: str, 
                        context: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Check if a requester can access a capability.
        
        Returns:
            (allowed: bool, reason: str)
        """
        request = CapabilityRequest(
            capability=capability,
            requester=requester,
            context=context or {}
        )
        
        # 1. Check if policy allows all (AllowAllPolicy)
        if isinstance(self.policy, AllowAllPolicy):
            reason = f"Capability {capability} allowed by policy (allow-all)"
            self.audit_log.log_request(request, True, reason)
            return True, reason
        
        # 2. Check if capability is base capability (always available)
        if capability in self.BASE_CAPABILITIES:
            reason = f"Base capability {capability} available"
            self.audit_log.log_request(request, True, reason)
            return True, reason
        
        # 3. Check if requester has been explicitly granted this capability
        if requester in self.granted_capabilities:
            if capability in self.granted_capabilities[requester]:
                reason = f"Capability {capability} granted to {requester}"
                self.audit_log.log_request(request, True, reason)
                return True, reason
        
        # 4. Check if capability is allowed by policy
        policy_level = self.policy.check(capability)
        if policy_level == CapabilityLevel.ALLOWED:
            reason = f"Capability {capability} allowed by policy"
            self.audit_log.log_request(request, True, reason)
            return True, reason
        
        # Denied by default
        reason = f"Capability {capability} not granted to {requester}"
        self.audit_log.log_request(request, False, reason)
        return False, reason
    
    def require_capability(self, requester: str, capability: str, 
                          context: Optional[Dict] = None) -> bool:
        """
        Require a capability. Raises error if not available.
        
        This is the hook that evaluator calls.
        """
        allowed, reason = self.check_capability(requester, capability, context)
        if not allowed:
            raise PermissionError(f"{requester} cannot access {capability}: {reason}")
        return True
    
    def has_capability(self, requester: str, capability: str) -> bool:
        """Check if a requester has a capability (without error)."""
        allowed, _ = self.check_capability(requester, capability)
        return allowed
    
    def get_granted_capabilities(self, requester: str) -> List[str]:
        """Get all capabilities granted to a requester."""
        caps = list(self.BASE_CAPABILITIES)
        if requester in self.granted_capabilities:
            caps.extend(list(self.granted_capabilities[requester]))
        return sorted(list(set(caps)))
    
    def get_required_capabilities(self, requester: str) -> List[str]:
        """Get capabilities required by a requester."""
        return sorted(list(self.required_capabilities.get(requester, set())))
    
    def validate_requirements(self, requester: str) -> Tuple[bool, List[str]]:
        """
        Validate that all required capabilities are granted.
        
        Returns:
            (valid: bool, missing_capabilities: [str])
        """
        required = self.get_required_capabilities(requester)
        granted = set(self.get_granted_capabilities(requester))
        
        missing = [cap for cap in required if cap not in granted]
        return len(missing) == 0, missing
    
    def get_audit_log(self) -> List[Dict]:
        """Get audit log entries."""
        return self.audit_log.get_entries()
    
    def get_audit_statistics(self) -> Dict[str, int]:
        """Get audit statistics."""
        return self.audit_log.get_statistics()


class CapabilityError(Exception):
    """Exception raised when capability check fails."""
    pass


# Standard capability sets for common scenarios

CAPABILITY_SETS = {
    "read_only": {
        "capabilities": [
            "io.read",
            "sys.time"
        ],
        "description": "Read-only access to files and time"
    },
    
    "io_full": {
        "capabilities": [
            "io.read",
            "io.write",
            "io.delete",
            "sys.time"
        ],
        "description": "Full file I/O access"
    },
    
    "network": {
        "capabilities": [
            "network.tcp",
            "network.http",
            "sys.time"
        ],
        "description": "Network access (TCP and HTTP)"
    },
    
    "crypto": {
        "capabilities": [
            "crypto.keygen",
            "crypto.sign",
            "crypto.hash"  # Assumed in crypto plugin
        ],
        "description": "Cryptographic operations"
    },
    
    "untrusted": {
        "capabilities": [],
        "description": "Minimal capabilities for untrusted code"
    },
    
    "trusted": {
        "capabilities": [
            "io.read",
            "io.write",
            "network.http",
            "crypto.hash",
            "sys.time"
        ],
        "description": "Standard trusted code capabilities"
    },
    
    "system": {
        "capabilities": [
            "io.read",
            "io.write",
            "io.delete",
            "network.tcp",
            "network.http",
            "crypto.keygen",
            "crypto.sign",
            "exec.shell",
            "exec.spawn",
            "sys.env",
            "sys.time",
            "sys.exit"
        ],
        "description": "Full system access (privileged code)"
    }
}
