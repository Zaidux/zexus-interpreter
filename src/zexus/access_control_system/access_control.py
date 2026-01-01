"""
Zexus Access Control System

Security Fix #9: Contract Access Control
Implements role-based access control (RBAC) and ownership patterns for smart contracts.
"""

from typing import Dict, Set, List as PyList, Optional
from ..object import String, Integer, Boolean, Map, List, NULL, EvaluationError


class AccessControlManager:
    """
    Global access control manager for contracts
    
    Provides:
    - Owner tracking and validation
    - Role-based access control (RBAC)
    - Permission management
    - Access validation helpers
    """
    
    def __init__(self):
        # contract_id -> owner_address
        self._owners: Dict[str, str] = {}
        
        # contract_id -> role -> set of addresses
        self._roles: Dict[str, Dict[str, Set[str]]] = {}
        
        # contract_id -> address -> set of permissions
        self._permissions: Dict[str, Dict[str, Set[str]]] = {}
        
        # Access attempt audit log
        self._access_log: PyList[Dict] = []
    
    def set_owner(self, contract_id: str, owner: str):
        """Set the owner of a contract"""
        self._owners[contract_id] = owner
        
        # Owner automatically has 'owner' role
        self.grant_role(contract_id, owner, "owner")
    
    def get_owner(self, contract_id: str) -> Optional[str]:
        """Get the owner of a contract"""
        return self._owners.get(contract_id)
    
    def is_owner(self, contract_id: str, address: str) -> bool:
        """Check if address is the owner of the contract"""
        owner = self._owners.get(contract_id)
        return owner == address if owner else False
    
    def grant_role(self, contract_id: str, address: str, role: str):
        """Grant a role to an address for a contract"""
        if contract_id not in self._roles:
            self._roles[contract_id] = {}
        if role not in self._roles[contract_id]:
            self._roles[contract_id][role] = set()
        
        self._roles[contract_id][role].add(address)
        
        # Log the grant
        self._access_log.append({
            "action": "grant_role",
            "contract": contract_id,
            "address": address,
            "role": role
        })
    
    def revoke_role(self, contract_id: str, address: str, role: str):
        """Revoke a role from an address"""
        if (contract_id in self._roles and 
            role in self._roles[contract_id] and
            address in self._roles[contract_id][role]):
            
            self._roles[contract_id][role].remove(address)
            
            # Log the revoke
            self._access_log.append({
                "action": "revoke_role",
                "contract": contract_id,
                "address": address,
                "role": role
            })
    
    def has_role(self, contract_id: str, address: str, role: str) -> bool:
        """Check if an address has a specific role"""
        if contract_id not in self._roles:
            return False
        if role not in self._roles[contract_id]:
            return False
        
        # Log access check
        self._access_log.append({
            "action": "check_role",
            "contract": contract_id,
            "address": address,
            "role": role
        })
        
        return address in self._roles[contract_id][role]
    
    def get_roles(self, contract_id: str, address: str) -> Set[str]:
        """Get all roles for an address in a contract"""
        if contract_id not in self._roles:
            return set()
        
        roles = set()
        for role, addresses in self._roles[contract_id].items():
            if address in addresses:
                roles.add(role)
        
        return roles
    
    def grant_permission(self, contract_id: str, address: str, permission: str):
        """Grant a specific permission to an address"""
        if contract_id not in self._permissions:
            self._permissions[contract_id] = {}
        if address not in self._permissions[contract_id]:
            self._permissions[contract_id][address] = set()
        
        self._permissions[contract_id][address].add(permission)
        
        # Log the grant
        self._access_log.append({
            "action": "grant_permission",
            "contract": contract_id,
            "address": address,
            "permission": permission
        })
    
    def revoke_permission(self, contract_id: str, address: str, permission: str):
        """Revoke a specific permission from an address"""
        if (contract_id in self._permissions and
            address in self._permissions[contract_id] and
            permission in self._permissions[contract_id][address]):
            
            self._permissions[contract_id][address].remove(permission)
            
            # Log the revoke
            self._access_log.append({
                "action": "revoke_permission",
                "contract": contract_id,
                "address": address,
                "permission": permission
            })
    
    def has_permission(self, contract_id: str, address: str, permission: str) -> bool:
        """Check if an address has a specific permission"""
        if contract_id not in self._permissions:
            return False
        if address not in self._permissions[contract_id]:
            return False
        
        # Log access check
        self._access_log.append({
            "action": "check_permission",
            "contract": contract_id,
            "address": address,
            "permission": permission
        })
        
        return permission in self._permissions[contract_id][address]
    
    def require_owner(self, contract_id: str, caller: str, message: str = "Only owner can perform this action"):
        """
        Require that caller is the owner of the contract
        Raises EvaluationError if not owner
        """
        if not self.is_owner(contract_id, caller):
            raise EvaluationError(f"Access denied: {message}")
    
    def require_role(self, contract_id: str, caller: str, role: str, message: str = None):
        """
        Require that caller has a specific role
        Raises EvaluationError if role not granted
        """
        if not self.has_role(contract_id, caller, role):
            msg = message or f"Caller must have role: {role}"
            raise EvaluationError(f"Access denied: {msg}")
    
    def require_permission(self, contract_id: str, caller: str, permission: str, message: str = None):
        """
        Require that caller has a specific permission
        Raises EvaluationError if permission not granted
        """
        if not self.has_permission(contract_id, caller, permission):
            msg = message or f"Caller must have permission: {permission}"
            raise EvaluationError(f"Access denied: {msg}")
    
    def require_any_role(self, contract_id: str, caller: str, roles: PyList[str], message: str = None):
        """
        Require that caller has at least one of the specified roles
        Raises EvaluationError if no matching role
        """
        for role in roles:
            if self.has_role(contract_id, caller, role):
                return
        
        msg = message or f"Caller must have one of: {', '.join(roles)}"
        raise EvaluationError(f"Access denied: {msg}")
    
    def get_access_log(self, contract_id: Optional[str] = None, limit: int = 100) -> PyList[Dict]:
        """Get access control audit log"""
        if contract_id:
            return [entry for entry in self._access_log[-limit:] if entry.get("contract") == contract_id]
        return self._access_log[-limit:]
    
    def clear_contract(self, contract_id: str):
        """Clear all access control data for a contract"""
        self._owners.pop(contract_id, None)
        self._roles.pop(contract_id, None)
        self._permissions.pop(contract_id, None)


# Global access control manager
_access_control = AccessControlManager()


def get_access_control() -> AccessControlManager:
    """Get the global access control manager"""
    return _access_control


# Predefined standard roles
ROLE_OWNER = "owner"
ROLE_ADMIN = "admin"
ROLE_MODERATOR = "moderator"
ROLE_USER = "user"

# Common permission constants
PERMISSION_READ = "read"
PERMISSION_WRITE = "write"
PERMISSION_EXECUTE = "execute"
PERMISSION_DELETE = "delete"
PERMISSION_TRANSFER = "transfer"
PERMISSION_MINT = "mint"
PERMISSION_BURN = "burn"
