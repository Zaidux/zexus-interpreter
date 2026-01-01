"""
Zexus Security Module

Security Fix #9: Contract Access Control
Provides role-based access control and permission management for smart contracts.
"""

from .access_control import (
    AccessControlManager,
    get_access_control,
    ROLE_OWNER,
    ROLE_ADMIN,
    ROLE_MODERATOR,
    ROLE_USER,
    PERMISSION_READ,
    PERMISSION_WRITE,
    PERMISSION_EXECUTE,
    PERMISSION_DELETE,
    PERMISSION_TRANSFER,
    PERMISSION_MINT,
    PERMISSION_BURN,
)

__all__ = [
    'AccessControlManager',
    'get_access_control',
    'ROLE_OWNER',
    'ROLE_ADMIN',
    'ROLE_MODERATOR',
    'ROLE_USER',
    'PERMISSION_READ',
    'PERMISSION_WRITE',
    'PERMISSION_EXECUTE',
    'PERMISSION_DELETE',
    'PERMISSION_TRANSFER',
    'PERMISSION_MINT',
    'PERMISSION_BURN',
]
