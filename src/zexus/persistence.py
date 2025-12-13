# src/zexus/persistence.py
"""
Persistent Memory Management for Zexus
Extends persistent storage beyond contracts to all storage keywords (LET, CONST, ENTITY, etc.)
"""

import os
import json
import sqlite3
import weakref
from threading import Lock
from typing import Dict, Any, Optional, Set
from .object import (
    Object, Integer, Float, String, Boolean as BooleanObj, 
    Null, NULL, List, Map, EntityInstance
)

# Storage directory for persistent data
PERSISTENCE_DIR = os.path.expanduser("~/.zexus/persistence")
os.makedirs(PERSISTENCE_DIR, exist_ok=True)


# ===============================================
# MEMORY LEAK TRACKING
# ===============================================

class MemoryTracker:
    """Track object allocations and detect potential memory leaks"""
    
    def __init__(self):
        self.allocations: Dict[int, Dict[str, Any]] = {}
        self.weak_refs: Dict[int, weakref.ref] = {}
        self.lock = Lock()
        self.enabled = True
        self.allocation_count = 0
        self.max_allocations = 100000  # Alert threshold
        
    def track(self, obj: Object, context: str = "unknown"):
        """Track an object allocation"""
        if not self.enabled:
            return
            
        with self.lock:
            obj_id = id(obj)
            self.allocation_count += 1
            
            # Create weak reference to detect when object is garbage collected
            def cleanup(ref):
                with self.lock:
                    if obj_id in self.allocations:
                        del self.allocations[obj_id]
                    if obj_id in self.weak_refs:
                        del self.weak_refs[obj_id]
            
            self.weak_refs[obj_id] = weakref.ref(obj, cleanup)
            self.allocations[obj_id] = {
                'type': obj.type() if hasattr(obj, 'type') else type(obj).__name__,
                'context': context,
                'allocation_number': self.allocation_count
            }
            
            # Check for potential memory leak
            if len(self.allocations) > self.max_allocations:
                self._report_potential_leak()
    
    def untrack(self, obj: Object):
        """Manually untrack an object"""
        with self.lock:
            obj_id = id(obj)
            if obj_id in self.allocations:
                del self.allocations[obj_id]
            if obj_id in self.weak_refs:
                del self.weak_refs[obj_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        with self.lock:
            type_counts = {}
            for alloc in self.allocations.values():
                obj_type = alloc['type']
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            
            return {
                'total_tracked': len(self.allocations),
                'total_allocated': self.allocation_count,
                'by_type': type_counts
            }
    
    def _report_potential_leak(self):
        """Report potential memory leak"""
        stats = self.get_stats()
        print(f"⚠️  MEMORY WARNING: {stats['total_tracked']} objects tracked (threshold: {self.max_allocations})")
        print(f"   Breakdown: {stats['by_type']}")
    
    def clear(self):
        """Clear all tracking data"""
        with self.lock:
            self.allocations.clear()
            self.weak_refs.clear()
            self.allocation_count = 0


# Global memory tracker instance
_memory_tracker = MemoryTracker()


def track_allocation(obj: Object, context: str = "unknown"):
    """Track an object allocation"""
    _memory_tracker.track(obj, context)


def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics"""
    return _memory_tracker.get_stats()


def enable_memory_tracking():
    """Enable memory tracking"""
    _memory_tracker.enabled = True


def disable_memory_tracking():
    """Disable memory tracking"""
    _memory_tracker.enabled = False


# ===============================================
# PERSISTENT STORAGE BACKEND
# ===============================================

class PersistentStorage:
    """Persistent storage for variables using SQLite"""
    
    def __init__(self, scope_id: str, storage_dir: str = PERSISTENCE_DIR):
        self.scope_id = scope_id
        self.db_path = os.path.join(storage_dir, f"{scope_id}.sqlite")
        self.conn = None
        self.lock = Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS variables (
                name TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                value TEXT NOT NULL,
                is_const INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_name ON variables(name)
        ''')
        self.conn.commit()
    
    def set(self, name: str, value: Object, is_const: bool = False):
        """Persist a variable"""
        with self.lock:
            serialized = self._serialize(value)
            cursor = self.conn.cursor()
            
            import time
            timestamp = time.time()
            
            cursor.execute('''
                INSERT OR REPLACE INTO variables (name, type, value, is_const, created_at, updated_at)
                VALUES (?, ?, ?, ?, 
                    COALESCE((SELECT created_at FROM variables WHERE name = ?), ?),
                    ?)
            ''', (name, serialized['type'], serialized['value'], 1 if is_const else 0, 
                  name, timestamp, timestamp))
            
            self.conn.commit()
    
    def get(self, name: str) -> Optional[Object]:
        """Retrieve a persisted variable"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT type, value FROM variables WHERE name = ?', (name,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._deserialize({'type': row[0], 'value': row[1]})
    
    def delete(self, name: str):
        """Delete a persisted variable"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM variables WHERE name = ?', (name,))
            self.conn.commit()
    
    def is_const(self, name: str) -> bool:
        """Check if a variable is const"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT is_const FROM variables WHERE name = ?', (name,))
            row = cursor.fetchone()
            return bool(row[0]) if row else False
    
    def list_variables(self) -> list:
        """List all persisted variables"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT name, type, is_const FROM variables')
            return [{'name': row[0], 'type': row[1], 'const': bool(row[2])} for row in cursor.fetchall()]
    
    def clear(self):
        """Clear all persisted variables"""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM variables')
            self.conn.commit()
    
    def _serialize(self, obj: Object) -> Dict[str, str]:
        """Serialize Zexus object to JSON"""
        if isinstance(obj, String):
            return {'type': 'string', 'value': json.dumps(obj.value)}
        elif isinstance(obj, Integer):
            return {'type': 'integer', 'value': json.dumps(obj.value)}
        elif isinstance(obj, Float):
            return {'type': 'float', 'value': json.dumps(obj.value)}
        elif isinstance(obj, BooleanObj):
            return {'type': 'boolean', 'value': json.dumps(obj.value)}
        elif isinstance(obj, List):
            serialized = [self._serialize(e) for e in obj.elements]
            return {'type': 'list', 'value': json.dumps(serialized)}
        elif isinstance(obj, Map):
            serialized = {k: self._serialize(v) for k, v in obj.pairs.items()}
            return {'type': 'map', 'value': json.dumps(serialized)}
        elif obj is Null or obj is NULL:
            return {'type': 'null', 'value': json.dumps(None)}
        elif isinstance(obj, EntityInstance):
            serialized_values = {k: self._serialize(v) for k, v in obj.values.items()}
            return {
                'type': 'entity_instance',
                'value': json.dumps({
                    'entity_name': obj.entity_def.name,
                    'values': serialized_values
                })
            }
        else:
            # Fallback: convert to string
            return {'type': 'string', 'value': json.dumps(str(obj.inspect() if hasattr(obj, 'inspect') else obj))}
    
    def _deserialize(self, data: Dict[str, str]) -> Object:
        """Deserialize JSON to Zexus object"""
        obj_type = data['type']
        value = json.loads(data['value'])
        
        if obj_type == 'string':
            return String(value)
        elif obj_type == 'integer':
            return Integer(value)
        elif obj_type == 'float':
            return Float(value)
        elif obj_type == 'boolean':
            return BooleanObj(value)
        elif obj_type == 'null':
            return NULL
        elif obj_type == 'list':
            elements = [self._deserialize(e) for e in value]
            return List(elements)
        elif obj_type == 'map':
            pairs = {k: self._deserialize(v) for k, v in value.items()}
            return Map(pairs)
        elif obj_type == 'entity_instance':
            # Note: This creates a basic EntityInstance without full EntityDefinition
            # For production, you'd need to store/restore the entity definition
            from .object import EntityDefinition
            entity_name = value['entity_name']
            serialized_values = value['values']
            deserialized_values = {k: self._deserialize(v) for k, v in serialized_values.items()}
            
            # Create minimal entity definition
            entity_def = EntityDefinition(entity_name, [])
            return EntityInstance(entity_def, deserialized_values)
        else:
            return String(str(value))
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# ===============================================
# PERSISTENT ENVIRONMENT MIXIN
# ===============================================

class PersistentEnvironmentMixin:
    """Mixin to add persistence to Environment class"""
    
    def __init__(self, *args, persistence_scope: Optional[str] = None, enable_persistence: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistence_enabled = enable_persistence
        self.persistence_scope = persistence_scope
        self.persistent_storage = None
        self.persisted_vars: Set[str] = set()  # Track which vars are persisted
        
        if enable_persistence and persistence_scope:
            self.persistent_storage = PersistentStorage(persistence_scope)
            self._load_persisted_vars()
    
    def _load_persisted_vars(self):
        """Load persisted variables into environment"""
        if not self.persistent_storage:
            return
        
        for var_info in self.persistent_storage.list_variables():
            name = var_info['name']
            value = self.persistent_storage.get(name)
            if value is not None:
                self.store[name] = value
                self.persisted_vars.add(name)
                if var_info['const']:
                    self.const_vars.add(name)
    
    def set_persistent(self, name: str, val: Object, is_const: bool = False):
        """Set a variable with persistence"""
        # Set in memory
        if is_const:
            self.set_const(name, val)
        else:
            self.set(name, val)
        
        # Persist to storage
        if self.persistence_enabled and self.persistent_storage:
            self.persistent_storage.set(name, val, is_const)
            self.persisted_vars.add(name)
            track_allocation(val, f"persistent:{name}")
        
        return val
    
    def get_persistent(self, name: str) -> Optional[Object]:
        """Get a variable, checking persistence if not in memory"""
        # Check memory first
        val = self.get(name)
        if val is not None:
            return val
        
        # Check persistent storage
        if self.persistence_enabled and self.persistent_storage:
            val = self.persistent_storage.get(name)
            if val is not None:
                self.store[name] = val
                self.persisted_vars.add(name)
                return val
        
        return None
    
    def clear_persistence(self):
        """Clear all persisted variables"""
        if self.persistent_storage:
            self.persistent_storage.clear()
            self.persisted_vars.clear()


# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def create_persistent_scope(scope_name: str) -> PersistentStorage:
    """Create a new persistent storage scope"""
    return PersistentStorage(scope_name)


def list_persistent_scopes() -> list:
    """List all persistent storage scopes"""
    if not os.path.exists(PERSISTENCE_DIR):
        return []
    
    scopes = []
    for filename in os.listdir(PERSISTENCE_DIR):
        if filename.endswith('.sqlite'):
            scope_name = filename[:-7]  # Remove .sqlite extension
            scopes.append(scope_name)
    
    return scopes


def delete_persistent_scope(scope_name: str):
    """Delete a persistent storage scope"""
    db_path = os.path.join(PERSISTENCE_DIR, f"{scope_name}.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
