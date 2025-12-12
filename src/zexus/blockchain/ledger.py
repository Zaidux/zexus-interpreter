"""
Zexus Blockchain Ledger System

Implements immutable, versioned state storage for blockchain and smart contract features.
"""

import json
import time
import hashlib
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


class LedgerEntry:
    """A single versioned entry in the ledger"""
    
    def __init__(self, key: str, value: Any, version: int, timestamp: float, tx_hash: str):
        self.key = key
        self.value = value
        self.version = version
        self.timestamp = timestamp
        self.tx_hash = tx_hash
        self.prev_hash = None
    
    def to_dict(self) -> Dict:
        """Convert entry to dictionary for hashing"""
        return {
            'key': self.key,
            'value': str(self.value),
            'version': self.version,
            'timestamp': self.timestamp,
            'tx_hash': self.tx_hash,
            'prev_hash': self.prev_hash
        }
    
    def hash(self) -> str:
        """Calculate hash of this entry"""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


class Ledger:
    """
    Immutable, versioned ledger for blockchain state.
    
    Features:
    - Immutability: Old values are never modified, only new versions created
    - Versioning: Every write creates a new version
    - Cryptographic integrity: Each entry is hashed and linked
    - Audit trail: Complete history of all changes
    """
    
    def __init__(self, name: str):
        self.name = name
        self.entries: List[LedgerEntry] = []
        self.current_state: Dict[str, Any] = {}
        self.version_index: Dict[str, List[LedgerEntry]] = {}  # key -> list of versions
        self.locked = False
    
    def write(self, key: str, value: Any, tx_hash: str) -> LedgerEntry:
        """
        Write a new value to the ledger (creates new version, doesn't modify old)
        
        Args:
            key: The variable name
            value: The new value
            tx_hash: Hash of the transaction making this write
            
        Returns:
            The new ledger entry
        """
        if self.locked:
            raise RuntimeError(f"Ledger '{self.name}' is locked (immutable)")
        
        # Get current version number
        version = len(self.version_index.get(key, [])) + 1
        
        # Create new entry
        entry = LedgerEntry(
            key=key,
            value=deepcopy(value),  # Deep copy to prevent external modification
            version=version,
            timestamp=time.time(),
            tx_hash=tx_hash
        )
        
        # Link to previous entry
        if key in self.version_index and self.version_index[key]:
            prev_entry = self.version_index[key][-1]
            entry.prev_hash = prev_entry.hash()
        
        # Add to ledger
        self.entries.append(entry)
        
        # Update indices
        if key not in self.version_index:
            self.version_index[key] = []
        self.version_index[key].append(entry)
        
        # Update current state
        self.current_state[key] = value
        
        return entry
    
    def read(self, key: str, version: Optional[int] = None) -> Any:
        """
        Read a value from the ledger
        
        Args:
            key: The variable name
            version: Optional version number (defaults to latest)
            
        Returns:
            The value at the specified version
        """
        if key not in self.version_index:
            raise KeyError(f"Key '{key}' not found in ledger '{self.name}'")
        
        if version is None:
            # Return current version
            return deepcopy(self.current_state[key])
        
        # Return specific version
        versions = self.version_index[key]
        if version < 1 or version > len(versions):
            raise ValueError(f"Invalid version {version} for key '{key}' (1-{len(versions)} available)")
        
        return deepcopy(versions[version - 1].value)
    
    def get_history(self, key: str) -> List[Tuple[int, Any, float, str]]:
        """
        Get complete history for a key
        
        Returns:
            List of (version, value, timestamp, tx_hash) tuples
        """
        if key not in self.version_index:
            return []
        
        return [
            (entry.version, entry.value, entry.timestamp, entry.tx_hash)
            for entry in self.version_index[key]
        ]
    
    def verify_integrity(self) -> bool:
        """
        Verify the cryptographic integrity of the ledger
        
        Returns:
            True if all hashes are valid and chain is intact
        """
        for key, versions in self.version_index.items():
            prev_hash = None
            for entry in versions:
                # Verify hash chain
                if entry.prev_hash != prev_hash:
                    return False
                prev_hash = entry.hash()
        return True
    
    def seal(self):
        """Make the ledger immutable (no more writes allowed)"""
        self.locked = True
    
    def get_state_root(self) -> str:
        """
        Calculate the merkle root hash of the current state
        
        Returns:
            SHA256 hash representing the entire current state
        """
        state_data = json.dumps(self.current_state, sort_keys=True)
        return hashlib.sha256(state_data.encode()).hexdigest()
    
    def export_audit_trail(self) -> List[Dict]:
        """Export complete audit trail as JSON-serializable data"""
        return [entry.to_dict() for entry in self.entries]


class LedgerManager:
    """
    Global ledger manager
    
    Manages all ledgers in the system and provides transaction isolation.
    """
    
    def __init__(self):
        self.ledgers: Dict[str, Ledger] = {}
        self.transaction_stack: List[Dict[str, Any]] = []
    
    def create_ledger(self, name: str) -> Ledger:
        """Create a new ledger"""
        if name in self.ledgers:
            raise ValueError(f"Ledger '{name}' already exists")
        
        ledger = Ledger(name)
        self.ledgers[name] = ledger
        return ledger
    
    def get_ledger(self, name: str) -> Ledger:
        """Get existing ledger"""
        if name not in self.ledgers:
            raise KeyError(f"Ledger '{name}' not found")
        return self.ledgers[name]
    
    def begin_transaction(self, tx_hash: str, caller: str, timestamp: float):
        """Begin a new transaction scope"""
        tx_context = {
            'tx_hash': tx_hash,
            'caller': caller,
            'timestamp': timestamp,
            'writes': [],  # List of (ledger_name, key, old_value)
            'gas_used': 0,
            'gas_limit': None
        }
        self.transaction_stack.append(tx_context)
    
    def commit_transaction(self):
        """Commit current transaction"""
        if not self.transaction_stack:
            raise RuntimeError("No active transaction to commit")
        
        # Remove transaction from stack
        self.transaction_stack.pop()
    
    def revert_transaction(self):
        """
        Revert current transaction
        
        Note: For ledgers, we can't truly "revert" since they're immutable.
        Instead, we write compensating entries to restore old values.
        """
        if not self.transaction_stack:
            raise RuntimeError("No active transaction to revert")
        
        tx = self.transaction_stack.pop()
        
        # Write compensating entries
        for ledger_name, key, old_value in reversed(tx['writes']):
            ledger = self.ledgers[ledger_name]
            # Create a revert entry
            ledger.write(key, old_value, f"REVERT:{tx['tx_hash']}")
    
    def get_current_tx(self) -> Optional[Dict]:
        """Get current transaction context"""
        return self.transaction_stack[-1] if self.transaction_stack else None


# Global ledger manager instance
_ledger_manager = LedgerManager()


def get_ledger_manager() -> LedgerManager:
    """Get the global ledger manager"""
    return _ledger_manager
