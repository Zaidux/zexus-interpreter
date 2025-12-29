# src/zexus/security.py

"""
Advanced Security and Contract Features for Zexus

This module implements entity, verify, contract, and protect statements
providing a powerful security framework for Zexus programs.
"""

import os
import json
import uuid
import sqlite3
import time

# Try importing advanced database drivers
try:
    import plyvel # For LevelDB
    _LEVELDB_AVAILABLE = True
except ImportError:
    _LEVELDB_AVAILABLE = False

try:
    import rocksdb # For RocksDB
    _ROCKSDB_AVAILABLE = True
except ImportError:
    _ROCKSDB_AVAILABLE = False

from .object import (
    Environment, Map, String, Integer, Float, Boolean as BooleanObj, 
    Builtin, List, Null, EvaluationError as ObjectEvaluationError
)

# Ensure storage directory exists
STORAGE_DIR = "chain_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# Audit logging directory
AUDIT_DIR = os.path.join(STORAGE_DIR, "audit_logs")
if not os.path.exists(AUDIT_DIR):
    os.makedirs(AUDIT_DIR)


class AuditLog:
    """Comprehensive audit logging system for compliance tracking
    
    Maintains audit trail of data access, modifications, and sensitive operations
    for regulatory compliance (GDPR, HIPAA, SOC2, etc.)
    """
    
    def __init__(self, max_entries=10000, persist_to_file=False):
        self.entries = []  # In-memory audit log
        self.max_entries = max_entries
        self.persist_to_file = persist_to_file
        self.audit_file = os.path.join(AUDIT_DIR, f"audit_{uuid.uuid4().hex[:8]}.jsonl")
    
    def log(self, data_name, action, data_type, timestamp=None, additional_context=None):
        """Log a single audit entry
        
        Args:
            data_name: Name of the data being audited
            action: Action type (access, modification, deletion, etc.)
            data_type: Type of data (STRING, MAP, ARRAY, FUNCTION, etc.)
            timestamp: Optional ISO 8601 timestamp (auto-generated if None)
            additional_context: Optional dict with additional audit context
        
        Returns:
            Audit entry dict
        """
        import datetime
        
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        entry = {
            "id": str(uuid.uuid4()),
            "data_name": data_name,
            "action": action,
            "data_type": data_type,
            "timestamp": timestamp,
            "context": additional_context or {}
        }
        
        self.entries.append(entry)
        
        # Enforce max entries limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Optionally persist to file
        if self.persist_to_file:
            self._write_entry_to_file(entry)
        
        return entry
    
    def _write_entry_to_file(self, entry):
        """Write audit entry to JSONL file"""
        try:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except IOError as e:
            print(f"Warning: Could not write audit log to file: {e}")
    
    def get_entries(self, data_name=None, action=None, limit=None):
        """Query audit log entries
        
        Args:
            data_name: Filter by data name (optional)
            action: Filter by action type (optional)
            limit: Limit number of results (optional)
        
        Returns:
            List of matching audit entries
        """
        results = self.entries
        
        if data_name:
            results = [e for e in results if e['data_name'] == data_name]
        
        if action:
            results = [e for e in results if e['action'] == action]
        
        if limit:
            results = results[-limit:]
        
        return results
    
    def clear(self):
        """Clear in-memory audit log"""
        self.entries = []
    
    def export_to_file(self, filename):
        """Export entire audit log to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.entries, f, indent=2)
            return True
        except IOError as e:
            print(f"Warning: Could not export audit log: {e}")
            return False
    
    def __repr__(self):
        return f"AuditLog(entries={len(self.entries)}, max={self.max_entries})"


class SecurityContext:
    """Global security context for enforcement"""
    def __init__(self):
        self.verify_checks = {}      # Registered verification checks
        self.protections = {}        # Active protection rules
        self.contracts = {}          # Deployed contracts
        self.middlewares = {}        # Registered middleware
        self.auth_config = None      # Global auth configuration
        self.cache_store = {}        # Caching store
        self.audit_log = AuditLog()  # Audit logging system
        # Registries for new commands/integration
        self.restrictions = {}       # id -> restriction entry
        self._restrictions_index = {}# target.field -> id
        self.trails = {}             # id -> trail config
        self.sandbox_runs = {}       # id -> sandbox run metadata
        # Sandbox policy store: name -> {allowed_builtins: set(...)}
        self.sandbox_policies = {}
        # Trail sinks: list of sink configs (type: 'file'|'stdout'|'callback')
        self.trail_sinks = []
    
    def log_audit(self, data_name, action, data_type, timestamp=None, context=None):
        """Log an audit entry through the security context
        
        Args:
            data_name: Name of data being audited
            action: Action type (access, modification, deletion, etc.)
            data_type: Type of data
            timestamp: Optional ISO 8601 timestamp
            context: Optional additional audit context
        
        Returns:
            Audit entry dict
        """
        return self.audit_log.log(data_name, action, data_type, timestamp, context)

    # -------------------------------
    # Restriction registry
    # -------------------------------
    def register_restriction(self, target, field, restriction_type, author=None, timestamp=None):
        """Register a field-level restriction.

        Args:
            target: full target string (e.g. 'user.email')
            field: property name (e.g. 'email')
            restriction_type: string describing rule (e.g. 'read-only')
            author: optional actor applying the restriction
            timestamp: ISO timestamp (auto-generated if None)

        Returns:
            restriction entry dict
        """
        import datetime
        rid = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        entry = {
            'id': rid,
            'target': target,
            'field': field,
            'restriction': restriction_type,
            'author': author,
            'timestamp': timestamp
        }

        self.restrictions[rid] = entry
        # index by full path for quick lookup (store latest)
        self._restrictions_index[f"{target}"] = rid
        return entry

    def get_restriction(self, target, field=None):
        """Lookup a restriction by target (and optional field). Returns entry or None."""
        key = f"{target}"
        rid = self._restrictions_index.get(key)
        if not rid:
            return None
        return self.restrictions.get(rid)

    def list_restrictions(self):
        return list(self.restrictions.values())

    def remove_restriction(self, rid):
        entry = self.restrictions.pop(rid, None)
        if entry:
            k = entry.get('target')
            if k and self._restrictions_index.get(k) == rid:
                del self._restrictions_index[k]
            return True
        return False

    # -------------------------------
    # Trail registry
    # -------------------------------
    def register_trail(self, event_type, filter_key=None, author=None, timestamp=None):
        import datetime
        tid = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = {
            'id': tid,
            'type': event_type,
            'filter': filter_key,
            'author': author,
            'timestamp': timestamp,
            'enabled': True
        }
        self.trails[tid] = entry
        return entry

    def list_trails(self):
        return list(self.trails.values())

    def remove_trail(self, tid):
        if tid in self.trails:
            del self.trails[tid]
            return True
        return False

    # -------------------------------
    # Sandbox run registry
    # -------------------------------
    def register_sandbox_run(self, parent_context=None, policy=None, result_summary=None, timestamp=None):
        import datetime
        sid = str(uuid.uuid4())
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry = {
            'id': sid,
            'parent': parent_context,
            'policy': policy,
            'result': result_summary,
            'timestamp': timestamp
        }
        self.sandbox_runs[sid] = entry
        return entry

    def list_sandbox_runs(self):
        return list(self.sandbox_runs.values())

    # -------------------------------
    # Sandbox policy management
    # -------------------------------
    def register_sandbox_policy(self, name, allowed_builtins=None):
        """Register a sandbox policy by name. `allowed_builtins` is an iterable of builtin names allowed inside the sandbox.

        If `allowed_builtins` is None, the policy is permissive (allows all).
        """
        if allowed_builtins is None:
            allowed_set = None
        else:
            allowed_set = set(allowed_builtins)
        self.sandbox_policies[name] = {'allowed_builtins': allowed_set}
        return self.sandbox_policies[name]

    def get_sandbox_policy(self, name):
        return self.sandbox_policies.get(name)

    # -------------------------------
    # Trail sink management
    # -------------------------------
    def register_trail_sink(self, sink_type, **kwargs):
        """Register a trail sink.

        sink_type: 'stdout' | 'file' | 'callback' | 'sqlite'
        kwargs: for 'file' provide `path`; for 'callback' provide `callback` callable.
        """
        sink = {'type': sink_type}
        sink.update(kwargs)
        self.trail_sinks.append(sink)
        return sink


    # -------------------------------
    # Event dispatcher (Trail integration)
    # -------------------------------
    def emit_event(self, event_type, payload):
        """Emit an event through the trail registry.

        - Matches active trails by `type` or `*`.
        - Applies simple substring filter matching against stringified payload.
        - For matching trails, record a derived audit entry and also print to stdout.
        """
        try:
            # simple stringify payload for filtering
            payload_str = json.dumps(payload) if not isinstance(payload, str) else payload
        except Exception:
            try:
                payload_str = str(payload)
            except Exception:
                payload_str = "<unserializable>"

        for tid, trail in list(self.trails.items()):
            ttype = trail.get('type')
            flt = trail.get('filter')
            # type match or wildcard
            if ttype != '*' and ttype != event_type:
                continue

            # filter match if provided — support substring, key:value, and regex (prefix 're:')
            if flt and flt != '*':
                matched = False
                try:
                    if isinstance(flt, str) and flt.startswith('re:'):
                        import re
                        pattern = flt[3:]
                        if re.search(pattern, payload_str):
                            matched = True
                    elif isinstance(flt, str) and ':' in flt:
                        # key:value pattern
                        k, v = flt.split(':', 1)
                        try:
                            # if payload is JSON object, check key
                            p_obj = json.loads(payload_str)
                            if isinstance(p_obj, dict) and k in p_obj and v in str(p_obj.get(k)):
                                matched = True
                        except Exception:
                            if k in payload_str and v in payload_str:
                                matched = True
                    else:
                        if flt in payload_str:
                            matched = True
                except Exception:
                    matched = False

                if not matched:
                    continue

            # Create audit-like entry for the trail event
            entry = {
                'id': str(uuid.uuid4()),
                'trail_id': tid,
                'event_type': event_type,
                'payload': payload_str,
                'timestamp': time.time()
            }
            # Persist to audit log if enabled
            try:
                self.audit_log.entries.append(entry)
            except Exception:
                pass

            # Deliver to configured sinks
            for sink in list(self.trail_sinks):
                try:
                    stype = sink.get('type')
                    if stype == 'stdout':
                        print(f"[TRAIL:{tid}] {event_type} -> {payload_str}")
                    elif stype == 'file':
                        path = sink.get('path') or os.path.join(AUDIT_DIR, 'trails.jsonl')
                        try:
                            with open(path, 'a', encoding='utf-8') as sf:
                                sf.write(json.dumps(entry) + '\n')
                        except Exception:
                            pass
                    elif stype == 'sqlite':
                        db_path = sink.get('db_path') or os.path.join(STORAGE_DIR, 'trails.db')
                        try:
                            conn = sqlite3.connect(db_path, check_same_thread=False)
                            cur = conn.cursor()
                            cur.execute('''CREATE TABLE IF NOT EXISTS trails (
                                id TEXT PRIMARY KEY,
                                trail_id TEXT,
                                event_type TEXT,
                                payload TEXT,
                                timestamp REAL
                            )''')
                            cur.execute('INSERT OR REPLACE INTO trails (id, trail_id, event_type, payload, timestamp) VALUES (?,?,?,?,?)', (
                                entry['id'], entry['trail_id'], entry['event_type'], entry['payload'], entry['timestamp']
                            ))
                            conn.commit()
                            conn.close()
                        except Exception:
                            pass
                    elif stype == 'callback':
                        cb = sink.get('callback')
                        try:
                            if callable(cb):
                                cb(entry)
                        except Exception:
                            pass
                except Exception:
                    pass


    def register_verify_check(self, name, check_func):
        """Register a verification check function"""
        self.verify_checks[name] = check_func

    def register_protection(self, name, rules):
        """Register a protection rule set"""
        self.protections[name] = rules

    def register_contract(self, name, contract):
        """Register a smart contract"""
        self.contracts[name] = contract

    def check_protection(self, target_name, context_data):
        """Check if target access is protected"""
        if target_name not in self.protections:
            return True  # No protection = allowed

        rules = self.protections[target_name]

        # Check authentication requirement
        if rules.get("auth_required", False):
            if not context_data.get("authenticated"):
                return False

        # Check rate limiting
        rate_limit = rules.get("rate_limit")
        if rate_limit:
            # Simple rate limit check (production would use timestamp tracking)
            if context_data.get("request_count", 0) > rate_limit:
                return False

        # Check IP restrictions
        client_ip = context_data.get("client_ip")
        if client_ip:
            blocked_ips = rules.get("blocked_ips", [])
            if _is_ip_in_list(client_ip, blocked_ips):
                return False

            allowed_ips = rules.get("allowed_ips")
            if allowed_ips and not _is_ip_in_list(client_ip, allowed_ips):
                return False

        return True


# Global security context
_security_context = SecurityContext()


def get_security_context():
    """Get the global security context"""
    return _security_context


def _is_ip_in_list(ip, ip_list):
    """Check if IP matches CIDR or exact match in list"""
    for pattern in ip_list:
        if "/" in pattern:  # CIDR notation
            # Simplified CIDR check (would need proper IP math for production)
            network_part = pattern.split("/")[0]
            if ip.startswith(network_part.rsplit(".", 1)[0]):
                return True
        elif ip == pattern:  # Exact match
            return True
    return False


# ===============================================
# ENTITY SYSTEM - Object-Oriented Data Structures
# ===============================================

class EntityDefinition:
    """Represents an entity definition with properties and methods"""

    def __init__(self, name, properties, methods=None, parent=None):
        self.name = name
        self.properties = properties  # {prop_name: {type, default_value}}
        self.methods = methods or {}   # {method_name: Action}
        self.parent = parent          # Parent entity (inheritance)

    def create_instance(self, values=None):
        """Create an instance of this entity with dependency injection support"""
        # Perform dependency injection for marked properties
        injected_values = values or {}
        
        # Check if this entity has injected dependencies
        if hasattr(self, 'injected_deps') and self.injected_deps:
            from zexus.dependency_injection import get_di_registry
            
            registry = get_di_registry()
            # Use __main__ as default module context
            container = registry.get_container("__main__")
            
            for dep_name in self.injected_deps:
                if dep_name not in injected_values:
                    # Try to inject from DI container
                    try:
                        injected_value = container.get(dep_name)
                        injected_values[dep_name] = injected_value
                    except BaseException as e:
                        # Dependency not available - use NULL placeholder
                        from zexus.object import NULL
                        injected_values[dep_name] = NULL
        
        instance = EntityInstance(self, injected_values)
        return instance

    def get_all_properties(self):
        """Get all properties including inherited ones, in correct order (parent first, then child)"""
        props = {}
        # First add parent properties
        if self.parent:
            parent_props = self.parent.get_all_properties()
            props.update(parent_props)
        # Then add/override with child properties
        props.update(self.properties)
        return props


class EntityInstance:
    """Represents an instance of an entity"""

    def __init__(self, entity_def, values):
        self.entity_def = entity_def
        self.data = values or {}
        self._validate_properties()

    def _validate_properties(self):
        """Validate that all required properties are present and inject dependencies"""
        all_props = self.entity_def.get_all_properties()
        for prop_name, prop_config in all_props.items():
            if prop_name not in self.data:
                # Check if this is an injected dependency
                if prop_config.get("injected", False):
                    # Try to inject from DI registry
                    try:
                        from zexus.dependency_injection import get_di_registry
                        from zexus.object import NULL
                        registry = get_di_registry()
                        container = registry.get_container("__main__")
                        if container:
                            injected_value = container.get(prop_name)
                            self.data[prop_name] = injected_value
                        else:
                            # No container, set to NULL
                            self.data[prop_name] = NULL
                    except Exception:
                        # If injection fails, set to NULL
                        from zexus.object import NULL
                        self.data[prop_name] = NULL
                elif "default_value" in prop_config:
                    self.data[prop_name] = prop_config["default_value"]

    def get(self, property_name):
        """Get property value"""
        return self.data.get(property_name)

    def set(self, property_name, value):
        """Set property value (prevent modification if property is sealed)"""
        if property_name not in self.entity_def.get_all_properties():
            raise ValueError(f"Unknown property: {property_name}")
        existing = self.data.get(property_name)
        # Avoid importing SealedObject here to prevent circular imports; use name-based check
        if existing is not None and existing.__class__.__name__ == 'SealedObject':
            raise ValueError(f"Cannot modify sealed property: {property_name}")
        self.data[property_name] = value

    def to_dict(self):
        """Convert to dictionary"""
        return self.data
    
    def call_method(self, method_name, args):
        """Call a method on this entity instance"""
        if method_name not in self.entity_def.methods:
            from zexus.object import EvaluationError
            return EvaluationError(f"Method '{method_name}' not supported for ENTITY_INSTANCE")
        
        # Get the method (Action or Function)
        method = self.entity_def.methods[method_name]
        
        # Create a new environment for the method execution
        from zexus.environment import Environment
        method_env = Environment(outer=method.env if hasattr(method, 'env') else None)
        
        # Bind 'this' to the current instance in the method environment
        method_env.set('this', self)
        
        # Bind method parameters to arguments
        if hasattr(method, 'parameters'):
            for i, param in enumerate(method.parameters):
                if i < len(args):
                    # Handle both Identifier objects and ParameterNode objects
                    if hasattr(param, 'name'):
                        # It's a ParameterNode with name and type
                        param_name = param.name.value if hasattr(param.name, 'value') else str(param.name)
                    elif hasattr(param, 'value'):
                        # It's an Identifier
                        param_name = param.value
                    else:
                        # Fallback to string representation
                        param_name = str(param)
                    method_env.set(param_name, args[i])
        
        # Import evaluator to execute the method body
        # Avoid circular import by importing here
        from zexus.evaluator.core import Evaluator
        evaluator = Evaluator()
        
        # Execute the method body with stack trace
        result = evaluator.eval_node(method.body, method_env, stack_trace=[])
        
        # Unwrap return values
        from zexus.object import ReturnValue
        if isinstance(result, ReturnValue):
            return result.value
        
        return result


# ===============================================
# VERIFICATION SYSTEM - Security Checks
# ===============================================

class VerificationCheck:
    """Represents a single verification condition"""

    def __init__(self, name, condition_func, error_message=""):
        self.name = name
        self.condition_func = condition_func
        self.error_message = error_message or f"Verification check '{name}' failed"

    def verify(self, context_data):
        """Execute verification check"""
        try:
            result = self.condition_func(context_data)
            return (result, None) if result else (False, self.error_message)
        except Exception as e:
            return (False, str(e))


class VerifyWrapper:
    """Wraps a function with verification checks"""

    def __init__(self, target_func, checks, error_handler=None):
        self.target_func = target_func
        self.checks = checks  # List of VerificationCheck
        self.error_handler = error_handler

    def execute(self, args, context_data=None, env=None):
        """Execute target function with verification"""
        context_data = context_data or {}

        # Run all verification checks
        for check in self.checks:
            is_valid, error_msg = check.verify(context_data)
            if not is_valid:
                if self.error_handler:
                    return self.error_handler(error_msg, context_data, env)
                else:
                    return ObjectEvaluationError(error_msg)

        # All checks passed, execute target
        return self.target_func(args, env)


# ===============================================
# CONTRACT PERSISTENCE BACKENDS
# ===============================================

class StorageBackend:
    """Interface for storage backends"""
    def set(self, key, value): pass
    def get(self, key): pass
    def delete(self, key): pass
    def close(self): pass

class InMemoryBackend(StorageBackend):
    def __init__(self):
        self.data = {}
    def set(self, key, value): self.data[key] = value
    def get(self, key): return self.data.get(key)
    def delete(self, key): 
        if key in self.data: del self.data[key]

class SQLiteBackend(StorageBackend):
    def __init__(self, db_path):
        import sqlite3
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)")
        self.conn.commit()

    def set(self, key, value):
        self.cursor.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def get(self, key):
        self.cursor.execute("SELECT value FROM kv_store WHERE key=?", (key,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def delete(self, key):
        self.cursor.execute("DELETE FROM kv_store WHERE key=?", (key,))
        self.conn.commit()

    def close(self):
        self.conn.close()

class LevelDBBackend(StorageBackend):
    def __init__(self, db_path):
        if not _LEVELDB_AVAILABLE: raise ImportError("plyvel not installed")
        self.db = plyvel.DB(db_path, create_if_missing=True)

    def set(self, key, value):
        self.db.put(key.encode('utf-8'), value.encode('utf-8'))

    def get(self, key):
        res = self.db.get(key.encode('utf-8'))
        return res.decode('utf-8') if res else None

    def delete(self, key):
        self.db.delete(key.encode('utf-8'))

    def close(self):
        self.db.close()

class RocksDBBackend(StorageBackend):
    def __init__(self, db_path):
        if not _ROCKSDB_AVAILABLE: raise ImportError("rocksdb not installed")
        self.db = rocksdb.DB(db_path, rocksdb.Options(create_if_missing=True))

    def set(self, key, value):
        self.db.put(key.encode('utf-8'), value.encode('utf-8'))

    def get(self, key):
        res = self.db.get(key.encode('utf-8'))
        return res.decode('utf-8') if res else None

    def delete(self, key):
        self.db.delete(key.encode('utf-8'))

# ===============================================
# CONTRACT SYSTEM - Blockchain State & Logic
# ===============================================

class ContractStorage:
    """Persistent storage for contract state with DB selection"""

    def __init__(self, contract_id, db_type="sqlite"):
        self.transaction_log = []
        self.db_type = db_type
        
        # Determine strict path
        base_path = os.path.join(STORAGE_DIR, f"{contract_id}")
        
        # Initialize Backend
        if db_type == "leveldb" and _LEVELDB_AVAILABLE:
            self.backend = LevelDBBackend(base_path)
        elif db_type == "rocksdb" and _ROCKSDB_AVAILABLE:
            self.backend = RocksDBBackend(f"{base_path}.rdb")
        elif db_type == "sqlite":
            self.backend = SQLiteBackend(f"{base_path}.sqlite")
        else:
            print(f"   ⚠️ Storage Warning: '{db_type}' unavailable or unknown. Falling back to In-Memory.")
            self.backend = InMemoryBackend()

    def get(self, key):
        """Get value from storage and deserialize from JSON"""
        raw_val = self.backend.get(key)
        if raw_val is None:
            return None
        return self._deserialize(raw_val)

    def set(self, key, value):
        """Serialize to JSON and set value in storage"""
        serialized = self._serialize(value)
        self.backend.set(key, serialized)
        self._log_transaction("SET", key, serialized)

    def delete(self, key):
        """Delete value from storage"""
        self.backend.delete(key)
        self._log_transaction("DELETE", key, None)

    def _log_transaction(self, op, key, value):
        """Log transaction for audit trail"""
        self.transaction_log.append({
            "operation": op,
            "key": key,
            "value": value,
            "timestamp": _get_timestamp()
        })

    def _serialize(self, obj):
        """Convert Zexus Object -> JSON String"""
        if isinstance(obj, String):
            return json.dumps({"type": "string", "val": obj.value})
        elif isinstance(obj, Integer):
            return json.dumps({"type": "integer", "val": obj.value})
        elif isinstance(obj, Float):
            return json.dumps({"type": "float", "val": obj.value})
        elif isinstance(obj, BooleanObj):
            return json.dumps({"type": "boolean", "val": obj.value})
        elif isinstance(obj, List):
            # Recursively serialize list elements
            serialized_list = [self._serialize_val_recursive(e) for e in obj.elements]
            return json.dumps({"type": "list", "val": serialized_list})
        elif isinstance(obj, Map):
            # Recursively serialize map elements
            serialized_map = {k: self._serialize_val_recursive(v) for k, v in obj.pairs.items()}
            return json.dumps({"type": "map", "val": serialized_map})
        elif obj is Null or obj is None:
            return json.dumps({"type": "null", "val": None})
        else:
            # Fallback for complex objects or strings
            return json.dumps({"type": "string", "val": str(obj)})

    def _serialize_val_recursive(self, obj):
        """Helper for nested structures (returns dict, not json string)"""
        # This mirrors _serialize but returns the inner dict structure directly
        if isinstance(obj, String): return {"type": "string", "val": obj.value}
        elif isinstance(obj, Integer): return {"type": "integer", "val": obj.value}
        elif isinstance(obj, BooleanObj): return {"type": "boolean", "val": obj.value}
        elif isinstance(obj, List): 
            return {"type": "list", "val": [self._serialize_val_recursive(e) for e in obj.elements]}
        elif isinstance(obj, Map): 
            return {"type": "map", "val": {k: self._serialize_val_recursive(v) for k, v in obj.pairs.items()}}
        elif obj is Null: return {"type": "null", "val": None}
        return {"type": "string", "val": str(obj)}

    def _deserialize(self, json_str):
        """Convert JSON String -> Zexus Object"""
        try:
            data = json.loads(json_str)
            return self._deserialize_recursive(data)
        except Exception as e:
            print(f"Settings corruption error: {e}")
            return Null

    def _deserialize_recursive(self, data):
        """Helper to reconstruct objects"""
        dtype = data.get("type")
        val = data.get("val")

        if dtype == "string": return String(val)
        elif dtype == "integer": return Integer(val)
        elif dtype == "float": return Float(val)
        elif dtype == "boolean": return BooleanObj(val)
        elif dtype == "null": return Null
        elif dtype == "list":
            # Reconstruct list
            elements = [self._deserialize_recursive(item) for item in val]
            return List(elements)
        elif dtype == "map":
            # Reconstruct map
            pairs = {k: self._deserialize_recursive(v) for k, v in val.items()}
            return Map(pairs)
        return String(str(val)) # Fallback


class SmartContract:
    """Represents a smart contract with persistent storage"""

    def __init__(self, name, storage_vars, actions, blockchain_config=None, address=None):
        self.name = name
        self.storage_vars = storage_vars or []
        self.actions = actions or {}
        self.blockchain_config = blockchain_config or {}
        
        # Generate a unique address/ID for this specific instance if not provided
        self.address = address or str(uuid.uuid4())[:8]
        
        # Default to SQLite, can be configured via blockchain_config
        db_pref = (blockchain_config or {}).get("storage_engine", "sqlite")
        
        # Initialize storage linked to unique address
        # The unique ID ensures multiple "ZiverWallet()" calls don't overwrite each other
        contract_id = f"{self.name}_{self.address}"
        self.storage = ContractStorage(contract_id, db_type=db_pref)
        self.is_deployed = False

    def instantiate(self, args=None):
        """Create a new instance of this contract when called like ZiverWallet()."""
        print(f"📄 SmartContract.instantiate() called for: {self.name}")
        
        # Generate new unique address for the instance
        new_address = str(uuid.uuid4())[:16]
        
        # Create instance with clean storage connection
        instance = SmartContract(
            name=self.name,
            storage_vars=self.storage_vars,
            actions=self.actions,
            blockchain_config=self.blockchain_config,
            address=new_address
        )
        
        print(f"   🔗 Contract Address: {new_address}")

        # Deploy the instance (initialize storage)
        instance.deploy()
        instance.parent_contract = self
        
        print(f"   Available actions: {list(self.actions.keys())}")
        return instance

    def __call__(self, *args):
        return self.instantiate(args)

    def deploy(self):
        """Deploy the contract and initialize persistent storage"""
        # Checks if we should reset storage or strictly load existing
        # For simplicity in this VM, subsequent runs act like "loading" if DB exists
        self.is_deployed = True
        
        # Initialize storage only if key doesn't exist (preserve persistence)
        for var_node in self.storage_vars:
            var_name = None
            default_value = None

            if hasattr(var_node, 'initial_value'):
                var_name = var_node.name.value if hasattr(var_node.name, 'value') else var_node.name
                default_value = var_node.initial_value
            elif isinstance(var_node, dict) and "initial_value" in var_node:
                var_name = var_node.get("name")
                default_value = var_node["initial_value"]

            if var_name:
                # ONLY set if not already in DB (Persistence Logic)
                if self.storage.get(var_name) is None:
                    if default_value is not None:
                        self.storage.set(var_name, default_value)
                    else:
                        # Set reasonable defaults for types if null
                        self.storage.set(var_name, Null)

    def execute_action(self, action_name, args, context, env=None):
        """Execute a contract action"""
        if not self.is_deployed:
            return ObjectEvaluationError(f"Contract {self.name} not deployed")

        if action_name not in self.actions:
            return ObjectEvaluationError(f"Unknown action: {action_name}")

        return self.actions[action_name]

    def get_state(self):
        return self.storage.backend.data if isinstance(self.storage.backend, InMemoryBackend) else {}

    def get_balance(self, account=None):
        val = self.storage.get(f"balance_{account}") if account else self.storage.get("balance")
        return val or Integer(0)


# ===============================================
# PROTECTION SYSTEM - Security Guardrails
# ===============================================

class ProtectionRule:
    """Represents a single protection rule"""

    def __init__(self, name, rule_config):
        self.name = name
        self.config = rule_config

    def evaluate(self, context_data):
        """Evaluate if protection allows access"""
        # Rate limiting
        if self.config.get("rate_limit"):
            if context_data.get("request_count", 0) > self.config["rate_limit"]:
                return False, "Rate limit exceeded"

        # Authentication requirement
        if self.config.get("auth_required", False):
            if not context_data.get("user_authenticated"):
                return False, "Authentication required"

        # Password strength
        if self.config.get("min_password_strength"):
            strength = context_data.get("password_strength", "weak")
            required = self.config["min_password_strength"]
            strength_levels = {"weak": 0, "medium": 1, "strong": 2, "very_strong": 3}
            if strength_levels.get(strength, 0) < strength_levels.get(required, 0):
                return False, f"Password must be {required}"

        # Session timeout
        if self.config.get("session_timeout"):
            session_age = context_data.get("session_age_seconds", 0)
            if session_age > self.config["session_timeout"]:
                return False, "Session expired"

        # HTTPS requirement
        if self.config.get("require_https", False):
            if not context_data.get("is_https", False):
                return False, "HTTPS required"

        return True, None


class ProtectionPolicy:
    """Represents a set of protection rules for a target"""

    def __init__(self, target_name, rules, enforcement_level="strict"):
        self.target_name = target_name
        self.rules = {}  # {rule_name: ProtectionRule}
        self.enforcement_level = enforcement_level  # strict, warn, audit

        if isinstance(rules, dict):
            for rule_name, rule_config in rules.items():
                self.add_rule(rule_name, rule_config)

    def add_rule(self, rule_name, rule_config):
        """Add a protection rule"""
        self.rules[rule_name] = ProtectionRule(rule_name, rule_config)

    def check_access(self, context_data):
        """Check if access is allowed"""
        violations = []

        for rule_name, rule in self.rules.items():
            allowed, error_msg = rule.evaluate(context_data)
            if not allowed:
                violations.append((rule_name, error_msg))

        if violations:
            if self.enforcement_level == "strict":
                return False, violations[0][1]
            elif self.enforcement_level == "warn":
                return True, violations  # Allow but warn
            elif self.enforcement_level == "audit":
                return True, violations  # Allow but log

        return True, None


# ===============================================
# MIDDLEWARE SYSTEM - Request/Response Processing
# ===============================================

class Middleware:
    """Represents a middleware handler"""

    def __init__(self, name, handler_func):
        self.name = name
        self.handler_func = handler_func

    def execute(self, request, response, env=None):
        """Execute middleware"""
        try:
            return self.handler_func((request, response), env)
        except Exception as e:
            return ObjectEvaluationError(f"Middleware error: {str(e)}")


class MiddlewareChain:
    """Executes a chain of middleware"""

    def __init__(self):
        self.middlewares = []

    def add_middleware(self, middleware):
        """Add middleware to chain"""
        self.middlewares.append(middleware)

    def execute(self, request, response, env=None):
        """Execute all middleware in order"""
        for middleware in self.middlewares:
            result = middleware.execute(request, response, env)
            if isinstance(result, ObjectEvaluationError):
                return result
            # Check if middleware set response to stop chain
            if response.get("_stop_chain"):
                break
        return response


# ===============================================
# AUTHENTICATION & AUTHORIZATION
# ===============================================

class AuthConfig:
    """Authentication configuration"""

    def __init__(self, config_data=None):
        self.provider = "oauth2"
        self.scopes = ["read", "write"]
        self.token_expiry = 3600
        self.refresh_enabled = True

        if config_data:
            self.provider = config_data.get("provider", self.provider)
            self.scopes = config_data.get("scopes", self.scopes)
            self.token_expiry = config_data.get("token_expiry", self.token_expiry)
            self.refresh_enabled = config_data.get("refresh_enabled", self.refresh_enabled)

    def validate_token(self, token):
        """Validate a token"""
        # In production, this would validate with OAuth provider
        return True

    def is_token_expired(self, token_data):
        """Check if token is expired"""
        import time
        if "issued_at" not in token_data:
            return True
        age = time.time() - token_data["issued_at"]
        return age > self.token_expiry


# ===============================================
# CACHING SYSTEM
# ===============================================

class CachePolicy:
    """Cache policy for a function"""

    def __init__(self, ttl=3600, key_func=None, invalidate_on=None):
        self.ttl = ttl  # Time to live in seconds
        self.key_func = key_func or (lambda x: str(x))  # Function to generate cache key
        self.invalidate_on = invalidate_on or []  # Events that invalidate cache
        self.cache = {}
        self.timestamps = {}

    def get(self, key):
        """Get cached value"""
        import time
        if key not in self.cache:
            return None

        # Check if expired
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None

        return self.cache[key]

    def set(self, key, value):
        """Cache a value"""
        import time
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def invalidate(self, key=None):
        """Invalidate cache entry or entire cache"""
        if key is None:
            self.cache.clear()
            self.timestamps.clear()
        elif key in self.cache:
            del self.cache[key]
            del self.timestamps[key]


# ===============================================
# SEALING / IMMUTABILITY
# ===============================================

class SealedObject:
    """Wraps an object and prevents mutation (assignments or property writes).

    This is a lightweight runtime wrapper. The evaluator enforces immutability by
    checking for instances of SealedObject before allowing assignments.
    """

    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def inspect(self):
        # Delegate to inner object's inspect if available
        if hasattr(self._value, 'inspect'):
            return self._value.inspect()
        return str(self._value)

    def type(self):
        # Delegate to inner object's type() if available, otherwise use its class name
        try:
            inner_type = self._value.type() if hasattr(self._value, 'type') else type(self._value).__name__
        except Exception:
            inner_type = type(self._value).__name__
        return f"Sealed<{inner_type}>"

    def __repr__(self):
        return f"SealedObject({repr(self._value)})"


# ===============================================
# RATE LIMITING
# ===============================================

class RateLimiter:
    """Rate limiter for throttling"""

    def __init__(self, requests_per_minute=100, burst_size=10, per_user=False):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.per_user = per_user
        self.request_counts = {}  # {user_id: count}
        self.burst_counts = {}    # {user_id: burst_count}

    def allow_request(self, user_id=None):
        """Check if request is allowed"""
        if not self.per_user:
            user_id = "global"

        current_count = self.request_counts.get(user_id, 0)
        burst_count = self.burst_counts.get(user_id, 0)

        # Check rate limit
        if current_count >= self.requests_per_minute:
            return False, "Rate limit exceeded"

        # Check burst limit
        if burst_count >= self.burst_size:
            return False, "Burst limit exceeded"

        self.request_counts[user_id] = current_count + 1
        self.burst_counts[user_id] = burst_count + 1

        return True, None

    def reset(self, user_id=None):
        """Reset rate limit counters"""
        if user_id:
            if user_id in self.request_counts:
                del self.request_counts[user_id]
            if user_id in self.burst_counts:
                del self.burst_counts[user_id]
        else:
            self.request_counts.clear()
            self.burst_counts.clear()


# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def _get_timestamp():
    """Get current timestamp"""
    import time
    return int(time.time() * 1000)


def export_security_to_environment(env):
    """Export security functions to environment"""
    # Entity creation
    def make_entity(entity_def, values=None):
        if isinstance(entity_def, EntityDefinition):
            return entity_def.create_instance(values)
        return ObjectEvaluationError("Invalid entity definition")

    # Verification
    def make_verify(target, checks, error_handler=None):
        return VerifyWrapper(target, checks, error_handler)

    # Contract deployment
    def deploy_contract(contract):
        if isinstance(contract, SmartContract):
            contract.deploy()
            return contract
        return ObjectEvaluationError("Invalid contract")

    env.set("entity", Builtin(make_entity, "entity"))
    env.set("verify", Builtin(make_verify, "verify"))
    env.set("contract", Builtin(deploy_contract, "contract"))
    # sealing: make a variable/object immutable
    def make_seal(value):
        return SealedObject(value)

    env.set("seal", Builtin(make_seal, "seal"))