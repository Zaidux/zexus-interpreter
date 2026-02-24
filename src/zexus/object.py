# src/zexus/object.py
import time
import random
import json
import os
import re
import sys
import threading
import atexit
from threading import Lock


# ===============================================
# SECURITY: Path & Identifier Sanitization (v1.8.1)
# ===============================================

def _safe_resolve_path(path: str, sandbox: str = None) -> str:
    """Resolve *path* and ensure it stays within *sandbox* (default: cwd).

    Raises ``PermissionError`` if the resolved path escapes the sandbox.
    Rejects paths that contain null bytes.
    """
    if "\x00" in path:
        raise PermissionError("Path contains null bytes")
    if sandbox is None:
        sandbox = os.getcwd()
    sandbox = os.path.realpath(sandbox)
    resolved = os.path.realpath(os.path.join(sandbox, path))
    # Ensure the resolved path is still within the sandbox
    if not (resolved == sandbox or resolved.startswith(sandbox + os.sep)):
        raise PermissionError(
            f"Path traversal denied: '{path}' resolves outside the project directory"
        )
    return resolved


# Regex: only alphanumeric, underscore, hyphen, dot allowed in identifiers
_SAFE_ID_RE = re.compile(r'^[A-Za-z0-9_.\-]+$')


def _sanitize_identifier(name: str, label: str = "identifier") -> str:
    """Ensure *name* is a safe filesystem identifier (no path separators, no traversal).

    Raises ``ValueError`` on invalid input.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"Invalid {label}: must be a non-empty string")
    if not _SAFE_ID_RE.match(name):
        raise ValueError(
            f"Invalid {label} '{name}': only alphanumeric, underscore, hyphen, "
            f"and dot characters are allowed"
        )
    # Extra guard: reject hidden-file tricks and double-dot
    if name.startswith('.') or '..' in name:
        raise ValueError(f"Invalid {label} '{name}': leading dot or '..' not allowed")
    return name


# Env vars that must never be modified by Zexus programs
BLOCKED_ENV_VARS = frozenset({
    'PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES',
    'DYLD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONHOME',
    'NODE_PATH', 'NODE_OPTIONS', 'PERL5LIB', 'RUBYLIB',
    'CLASSPATH', 'HOME', 'USER', 'SHELL', 'LOGNAME',
    'SSH_AUTH_SOCK', 'GPG_AGENT_INFO',
})


class Object:
    def inspect(self):
        raise NotImplementedError("Subclasses must implement this method")

# === EXISTING TYPES ===
class Integer(Object):
    def __init__(self, value): self.value = value
    def inspect(self): return str(self.value)
    def type(self): return "INTEGER"

class Float(Object):
    def __init__(self, value): self.value = value
    def inspect(self): return str(self.value)
    def type(self): return "FLOAT"

class Boolean(Object):
    def __init__(self, value): self.value = value
    def inspect(self): return "true" if self.value else "false"
    def type(self): return "BOOLEAN"

class Null(Object):
    def inspect(self): return "null"
    def type(self): return "NULL"

class String(Object):
    def __init__(self, value, sanitized_for=None, is_trusted=False):
        self.value = value
        # Track sanitization status for security enforcement
        self.sanitized_for = sanitized_for  # None, 'sql', 'html', 'url', 'shell', etc.
        self.is_trusted = is_trusted  # True for literals, False for external input
        
    def inspect(self): return self.value
    def type(self): return "STRING"
    def __str__(self): return self.value
    def __eq__(self, other):
        """Enable String objects to be used as dict keys"""
        if isinstance(other, String):
            return self.value == other.value
        return False
    def __hash__(self):
        """Enable String objects to be used as dict keys"""
        return hash(self.value)
    
    def __getitem__(self, index):
        """Enable string indexing: str[i] returns character at index i"""
        try:
            # Handle Integer object or raw int
            idx = index.value if hasattr(index, 'value') else index
            idx = int(idx)
            if 0 <= idx < len(self.value):
                # Return a new String object containing the character
                return String(self.value[idx], sanitized_for=self.sanitized_for, is_trusted=self.is_trusted)
            return NULL
        except (ValueError, TypeError, AttributeError):
            return NULL
    
    def get(self, index):
        """Get character by index (for evaluator compatibility)"""
        return self.__getitem__(index)
    
    def mark_sanitized(self, context):
        """Mark this string as sanitized for a specific context"""
        self.sanitized_for = context
        return self
    
    def is_safe_for(self, context):
        """Check if string is safe to use in given context"""
        # Trusted strings (literals) are always safe
        if self.is_trusted:
            return True
        # Check if sanitized for this specific context
        return self.sanitized_for == context

class List(Object):
    def __init__(self, elements): self.elements = elements
    def inspect(self):
        elements_str = ", ".join([el.inspect() for el in self.elements])
        return f"[{elements_str}]"
    def type(self): return "LIST"
    
    def get(self, index):
        """Get element by index"""
        try:
            # Handle Integer object or raw int
            idx = index.value if hasattr(index, 'value') else index
            idx = int(idx)
            if 0 <= idx < len(self.elements):
                return self.elements[idx]
            return NULL
        except Exception:
            return NULL
    
    def set(self, index, value):
        """Set element at index, mirroring array-style assignment."""
        try:
            idx = index.value if hasattr(index, 'value') else index
            idx = int(idx)
        except Exception as exc:
            raise EvaluationError(f"Invalid index for assignment: {index}") from exc

        if idx < 0 or idx >= len(self.elements):
            raise EvaluationError(f"Index out of range for List assignment: {idx}")

        self.elements[idx] = value
        return value

    def append(self, item):
        """Append item to list in-place (mutating operation)"""
        self.elements.append(item)
        return self  # Return self for method chaining
    
    def extend(self, other_list):
        """Extend list with another list in-place"""
        if isinstance(other_list, List):
            self.elements.extend(other_list.elements)
        return self

class Map(Object):
    def __init__(self, pairs):
        self.pairs = pairs

    def type(self): return "MAP"
    def inspect(self):
        pairs = []
        for key, value in self.pairs.items():
            key_str = key.inspect() if hasattr(key, 'inspect') else str(key)
            value_str = value.inspect() if hasattr(value, 'inspect') else str(value)
            pairs.append(f"{key_str}: {value_str}")
        return "{" + ", ".join(pairs) + "}"

    def _normalize_key(self, key):
        if hasattr(key, 'inspect'):
            return key.inspect()
        return str(key)

    def get(self, key):
        """Get value by key (compatible with both string and String keys)"""
        # Try direct lookup first (works for String-keyed and plain-keyed maps)
        val = self.pairs.get(key)
        if val is not None:
            return val
        # Try with String object key (for maps created by _python_to_zexus)
        if isinstance(key, str):
            str_key = String(key)
            val = self.pairs.get(str_key)
            if val is not None:
                return val
        # Try with normalized plain string key (for maps with plain string keys)
        norm_key = self._normalize_key(key)
        return self.pairs.get(norm_key)

    def set(self, key, value):
        """Set value for key, blocking modification if key is sealed."""
        norm_key = self._normalize_key(key)
        existing = self.pairs.get(norm_key)
        if existing is not None and existing.__class__.__name__ == 'SealedObject':
            raise EvaluationError(f"Cannot modify sealed map key: {key}")
        self.pairs[norm_key] = value
        return value

    def keys(self):
        """Return array of map keys"""
        from .object import Array
        return Array([k for k in self.pairs.keys()])

    def values(self):
        """Return array of map values"""
        from .object import Array
        return Array([v for v in self.pairs.values()])

    def entries(self):
        """Return array of [key, value] pairs"""
        from .object import Array
        return Array([Array([k, v]) for k, v in self.pairs.items()])

    def has(self, key):
        """Check if key exists in map"""
        from .object import Boolean
        return Boolean(key in self.pairs)

    def size(self):
        """Return number of entries in map"""
        from .object import Integer
        return Integer(len(self.pairs))

class EmbeddedCode(Object):
    def __init__(self, name, language, code):
        self.name = name
        self.language = language
        self.code = code
    def inspect(self): return f"<embedded {self.language} code: {self.name}>"
    def type(self): return "EMBEDDED_CODE"

class ReturnValue(Object):
    def __init__(self, value): self.value = value
    def inspect(self): return self.value.inspect()
    def type(self): return "RETURN_VALUE"

class Action(Object):
    def __init__(self, parameters, body, env):
        self.parameters, self.body, self.env = parameters, body, env
    def inspect(self):
        params = ", ".join([p.value for p in self.parameters])
        return f"action({params}) {{\n  ...\n}}"
    def type(self): return "ACTION"

class LambdaFunction(Object):
    def __init__(self, parameters, body, env):
        self.parameters = parameters
        self.body = body
        self.env = env
    def inspect(self):
        params = ", ".join([p.value for p in self.parameters])
        return f"lambda({params})"
    def type(self): return "LAMBDA_FUNCTION"
class Modifier(Object):
    """Function modifier for access control and validation"""
    def __init__(self, name, parameters, body, env):
        self.name = name
        self.parameters = parameters
        self.body = body
        self.env = env
    def inspect(self):
        params = ", ".join([p.value for p in self.parameters]) if self.parameters else ""
        return f"modifier {self.name}({params})"
    def type(self): return "MODIFIER"


class ContractReference(Object):
    """Reference to a deployed contract instance by address
    
    Used for storing contract references in contract state. Instead of storing
    the contract instance directly (which can't be serialized), we store a
    reference that can be resolved via the contract registry.
    """
    def __init__(self, address, contract_name=None):
        self.address = address
        self.contract_name = contract_name  # Optional, for display purposes
        
    def inspect(self):
        if self.contract_name:
            return f"<ContractRef: {self.contract_name}@{self.address}>"
        return f"<ContractRef: {self.address}>"
    
    def type(self):
        return "CONTRACT_REFERENCE"
    
    def resolve(self, registry=None):
        """Resolve this reference to the actual contract instance
        
        Args:
            registry: Contract registry to look up the contract (if not provided, uses global)
            
        Returns:
            SmartContract instance or None if not found
        """
        if registry is None:
            # Use global contract registry
            from . import security
            registry = security.get_contract_registry()
        
        return registry.get(self.address)
    
    def call_method(self, method_name, args):
        """Transparently delegate method calls to the resolved contract
        
        Args:
            method_name: Name of the action to call
            args: List of arguments to pass to the action
            
        Returns:
            Result of the action execution
        """
        # Resolve the contract
        contract = self.resolve()
        
        if contract is None:
            from . import object as obj_module
            return obj_module.EvaluationError(f"Contract at address {self.address} not found in registry")
        
        # Delegate to the contract's call_method
        return contract.call_method(method_name, args)
    
    def get_attr(self, attr_name):
        """Transparently delegate attribute access to the resolved contract
        
        Args:
            attr_name: Name of the attribute to access
            
        Returns:
            The attribute value from the contract
        """
        # Resolve the contract
        contract = self.resolve()
        
        if contract is None:
            from . import object as obj_module
            return obj_module.EvaluationError(f"Contract at address {self.address} not found in registry")
        
        # Allow direct access to the contract's attributes
        if hasattr(contract, attr_name):
            attr_value = getattr(contract, attr_name)
            # Wrap special types if needed
            if isinstance(attr_value, str):
                return String(attr_value)
            return attr_value
        
        from . import object as obj_module
        return obj_module.EvaluationError(f"Contract has no attribute '{attr_name}'")


class Builtin(Object):
    def __init__(self, fn, name=""):
        self.fn = fn
        self.name = name
    def inspect(self): return f"<built-in function: {self.name}>"
    def type(self): return "BUILTIN"

# === ASYNC/AWAIT OBJECTS ===

class Promise(Object):
    """
    Promise object representing an async operation
    States: PENDING, FULFILLED, REJECTED
    """
    PENDING = "PENDING"
    FULFILLED = "FULFILLED"
    REJECTED = "REJECTED"
    
    def __init__(self, executor=None, env=None, stack_trace=None):
        self.state = Promise.PENDING
        self.value = None
        self.error = None
        self.then_callbacks = []
        self.catch_callbacks = []
        self.finally_callbacks = []
        
        # Async context propagation
        self.env = env  # Environment at promise creation
        self.stack_trace = stack_trace or []  # Stack trace context
        
        # If executor provided, run it immediately
        if executor:
            try:
                executor(self._resolve, self._reject)
            except Exception as e:
                self._reject(e)
    
    def _resolve(self, value):
        """Resolve the promise with a value"""
        if self.state != Promise.PENDING:
            return
        self.state = Promise.FULFILLED
        self.value = value
        
        # Execute then callbacks
        for callback in self.then_callbacks:
            try:
                callback(value)
            except Exception:
                pass
        
        # Execute finally callbacks
        for callback in self.finally_callbacks:
            try:
                callback()
            except Exception:
                pass
    
    def _reject(self, error):
        """Reject the promise with an error"""
        if self.state != Promise.PENDING:
            return
        self.state = Promise.REJECTED
        self.error = error
        
        # Execute catch callbacks
        for callback in self.catch_callbacks:
            try:
                callback(error)
            except Exception:
                pass
        
        # Execute finally callbacks
        for callback in self.finally_callbacks:
            try:
                callback()
            except Exception:
                pass
    
    def then(self, callback):
        """Add a success callback"""
        if self.state == Promise.FULFILLED:
            callback(self.value)
        elif self.state == Promise.PENDING:
            self.then_callbacks.append(callback)
        return self
    
    def catch(self, callback):
        """Add an error callback"""
        if self.state == Promise.REJECTED:
            callback(self.error)
        elif self.state == Promise.PENDING:
            self.catch_callbacks.append(callback)
        return self
    
    def finally_callback(self, callback):
        """Add a finally callback (runs regardless of outcome)"""
        if self.state != Promise.PENDING:
            callback()
        else:
            self.finally_callbacks.append(callback)
        return self
    
    def is_resolved(self):
        """Check if promise is resolved (fulfilled or rejected)"""
        return self.state != Promise.PENDING
    
    def get_value(self):
        """Get the promise value (blocks if pending)"""
        if self.state == Promise.FULFILLED:
            return self.value
        elif self.state == Promise.REJECTED:
            raise Exception(f"Promise rejected: {self.error}")
        else:
            raise Exception("Promise is still pending")
    
    def inspect(self):
        if self.state == Promise.PENDING:
            return "Promise { <pending> }"
        elif self.state == Promise.FULFILLED:
            value_str = self.value.inspect() if hasattr(self.value, 'inspect') else str(self.value)
            return f"Promise {{ <fulfilled>: {value_str} }}"
        else:
            return f"Promise {{ <rejected>: {self.error} }}"
    
    def type(self):
        return "PROMISE"


class Coroutine(Object):
    """
    Coroutine object representing an async function execution
    Wraps a generator/iterator for suspension and resumption
    """
    def __init__(self, generator, action):
        self.generator = generator
        self.action = action  # The async action that created this coroutine
        self.is_complete = False
        self.result = None
        self.error = None
    
    def resume(self, value=None):
        """Resume coroutine execution, returns (is_done, value/error)"""
        try:
            if self.is_complete:
                return (True, self.result)
            
            # Send value to generator and get next yielded value
            next_value = self.generator.send(value)
            return (False, next_value)
        except StopIteration as e:
            self.is_complete = True
            self.result = e.value if hasattr(e, 'value') else None
            return (True, self.result)
        except Exception as e:
            self.is_complete = True
            self.error = e
            return (True, e)
    
    def inspect(self):
        if self.is_complete:
            if self.result is not None:
                result_str = self.result.inspect() if hasattr(self.result, 'inspect') else str(self.result)
                return f"Coroutine {{ <complete>: {result_str} }}"
            return "Coroutine { <complete>: null }"
        return "Coroutine { <running> }"
    
    def type(self):
        return "COROUTINE"

# === ENTITY OBJECTS ===

# Thread-local evaluator cache for entity/contract method execution.
# Kept lazy to avoid circular imports at module import time.
_entity_evaluator_cache = threading.local()


def _get_cached_evaluator():
    evaluator = getattr(_entity_evaluator_cache, "evaluator", None)
    if evaluator is None:
        from .evaluator.core import Evaluator
        evaluator = Evaluator()
        _entity_evaluator_cache.evaluator = evaluator
    return evaluator


def _clear_cached_evaluator():
    if hasattr(_entity_evaluator_cache, "evaluator"):
        del _entity_evaluator_cache.evaluator


# Best-effort cleanup for the current thread at interpreter shutdown.
atexit.register(_clear_cached_evaluator)

class EntityDefinition(Object):
    """Entity definition.

    MEDIUM (M5): This is the canonical EntityDefinition implementation.
    It supports:
    - properties as either a dict (new format) or list (legacy format)
    - optional methods (Action objects)
    - optional parent entity (inheritance)
    - optional dependency injection for marked properties
    """

    def __init__(self, name, properties, methods=None, parent=None):
        self.name = name
        self.properties = properties  # dict or list
        self.methods = methods or {}
        self.parent = parent  # Optional parent entity for inheritance
        
    def type(self):
        return "ENTITY_DEF"
        
    def inspect(self):
        # Handle both dict format {prop_name: {type: ..., default_value: ...}}
        # and list format [{name: ..., type: ...}]
        if isinstance(self.properties, dict):
            props_str = ", ".join([f"{name}: {info['type']}" for name, info in self.properties.items()])
        else:
            props_str = ", ".join([f"{prop['name']}: {prop['type']}" for prop in self.properties])
        return f"entity {self.name} {{ {props_str} }}"

    def get_all_properties(self):
        """Get all properties including inherited ones (parent first)."""
        props = {}
        if self.parent:
            try:
                props.update(self.parent.get_all_properties())
            except Exception:
                pass
        if isinstance(self.properties, dict):
            props.update(self.properties)
        else:
            # Legacy list format: [{name,type,default_value?}, ...]
            for prop in self.properties:
                try:
                    props[prop['name']] = prop
                except Exception:
                    continue
        return props
        
    def create_instance(self, initial_values=None):
        """Create an instance of this entity with optional initial values."""
        injected_values = dict(initial_values or {})

        # Dependency injection support: fill missing injected deps with container values.
        injected_deps = getattr(self, 'injected_deps', None)
        if injected_deps:
            try:
                from .dependency_injection import get_di_registry
                registry = get_di_registry()
                container = registry.get_container("__main__")
            except Exception:
                container = None

            for dep_name in injected_deps:
                if dep_name in injected_values:
                    continue
                if container is None:
                    injected_values[dep_name] = NULL
                    continue
                try:
                    injected_values[dep_name] = container.get(dep_name)
                except BaseException:
                    injected_values[dep_name] = NULL

        return EntityInstance(self, injected_values)

class EntityInstance(Object):
    def __init__(self, entity_def, values):
        self.entity_def = entity_def
        self.data = values or {}
        # Backward compatibility alias
        self.values = self.data
        self._validate_properties()
        
    def type(self):
        return "ENTITY_INSTANCE"
        
    def inspect(self):
        values_str = ", ".join([
            f"{k}: {v.inspect() if hasattr(v, 'inspect') else v}" for k, v in self.data.items()
        ])
        return f"{self.entity_def.name} {{ {values_str} }}"

    def __str__(self):
        return self.inspect()

    def __repr__(self):
        return self.__str__()

    def _validate_properties(self):
        """Populate missing injected/default values for declared properties."""
        try:
            all_props = self.entity_def.get_all_properties()
        except Exception:
            all_props = {}

        for prop_name, prop_config in all_props.items():
            if prop_name in self.data:
                continue

            # New-format dict config
            if isinstance(prop_config, dict):
                if prop_config.get("injected", False):
                    try:
                        from .dependency_injection import get_di_registry
                        registry = get_di_registry()
                        container = registry.get_container("__main__")
                        if container is not None:
                            self.data[prop_name] = container.get(prop_name)
                        else:
                            self.data[prop_name] = NULL
                    except Exception:
                        self.data[prop_name] = NULL
                    continue

                if "default_value" in prop_config:
                    self.data[prop_name] = prop_config.get("default_value")
                    continue

            # Legacy list config may include default_value
            if isinstance(prop_config, dict) and "default_value" in prop_config:
                self.data[prop_name] = prop_config.get("default_value")
        
    def get(self, property_name):
        return self.data.get(property_name, NULL)
        
    def set(self, property_name, value):
        # Check if property exists in entity definition (including inherited props)
        try:
            props = self.entity_def.get_all_properties()
            if property_name not in props:
                return FALSE
        except Exception:
            # Legacy list-only entity definitions
            try:
                prop_def = next(
                    (prop for prop in self.entity_def.properties if prop.get('name') == property_name),
                    None,
                )
                if not prop_def:
                    return FALSE
            except Exception:
                return FALSE

        existing = self.data.get(property_name)
        if existing is not None and existing.__class__.__name__ == 'SealedObject':
            return EvaluationError(f"Cannot modify sealed property: {property_name}")

        self.data[property_name] = value
        return TRUE

    def call_method(self, method_name, args):
        """Call an entity method (Action) defined on the entity definition."""
        if not hasattr(self.entity_def, 'methods') or method_name not in self.entity_def.methods:
            return EvaluationError(f"Method '{method_name}' not supported for ENTITY_INSTANCE")

        method = self.entity_def.methods[method_name]

        # Create a new environment for method execution
        from .environment import Environment as ExecEnvironment
        method_env = ExecEnvironment(outer=method.env if hasattr(method, 'env') else None)
        method_env.set('this', self)

        # Bind method parameters
        if hasattr(method, 'parameters'):
            for i, param in enumerate(method.parameters):
                if i >= len(args):
                    break
                if hasattr(param, 'name'):
                    param_name = param.name.value if hasattr(param.name, 'value') else str(param.name)
                elif hasattr(param, 'value'):
                    param_name = param.value
                else:
                    param_name = str(param)
                method_env.set(param_name, args[i])

        evaluator = _get_cached_evaluator()

        result = evaluator.eval_node(method.body, method_env, stack_trace=[])
        if isinstance(result, ReturnValue):
            return result.value
        return result

# === UTILITY CLASSES ===

class DateTime(Object):
    def __init__(self, timestamp=None):
        self.timestamp = timestamp or time.time()

    def inspect(self):
        return f"<DateTime: {self.timestamp}>"

    def type(self):
        return "DATETIME"

    @staticmethod
    def now():
        return DateTime(time.time())

    def to_timestamp(self):
        return Integer(int(self.timestamp))

    def __str__(self):
        return str(self.timestamp)

class Math(Object):
    def type(self):
        return "MATH_UTILITY"

    def inspect(self):
        return "<Math utilities>"

    @staticmethod
    def random_int(min_val, max_val):
        return Integer(random.randint(min_val, max_val))

    @staticmethod
    def to_hex_string(number):
        if isinstance(number, Integer):
            return String(hex(number.value))
        return String(hex(number))

    @staticmethod
    def hex_to_int(hex_string):
        if isinstance(hex_string, String):
            return Integer(int(hex_string.value, 16))
        return Integer(int(hex_string, 16))

    @staticmethod 
    def sqrt(number):
        if isinstance(number, Integer):
            return Float(number.value ** 0.5)
        elif isinstance(number, Float):
            return Float(number.value ** 0.5)
        return Null()

class File(Object):
    def type(self):
        return "FILE_UTILITY"

    def inspect(self):
        return "<File I/O utilities>"

    # === BASIC TIER ===
    @staticmethod
    def read_text(path):
        try:
            if isinstance(path, String):
                path = path.value
            resolved = _safe_resolve_path(path)
            with open(resolved, 'r', encoding='utf-8') as f:
                # Files are external data sources - return untrusted strings
                return String(f.read(), is_trusted=False)
        except PermissionError as e:
            return EvaluationError(f"File read denied: {str(e)}")
        except Exception as e:
            return EvaluationError(f"File read error: {str(e)}")

    @staticmethod
    def write_text(path, content):
        try:
            if isinstance(path, String):
                path = path.value
            if isinstance(content, String):
                content = content.value
            resolved = _safe_resolve_path(path)
            with open(resolved, 'w', encoding='utf-8') as f:
                f.write(content)
            return Boolean(True)
        except PermissionError as e:
            return EvaluationError(f"File write denied: {str(e)}")
        except Exception as e:
            return EvaluationError(f"File write error: {str(e)}")

    @staticmethod
    def exists(path):
        if isinstance(path, String):
            path = path.value
        try:
            resolved = _safe_resolve_path(path)
            return Boolean(os.path.exists(resolved))
        except PermissionError:
            return Boolean(False)

    # === MEDIUM TIER ===
    @staticmethod
    def read_json(path):
        try:
            content = File.read_text(path)
            if isinstance(content, EvaluationError):
                return content
            data = json.loads(content.value)
            # Convert Python data to Zexus objects
            return File._python_to_zexus(data)
        except Exception as e:
            return EvaluationError(f"JSON read error: {str(e)}")

    @staticmethod
    def write_json(path, data):
        try:
            # Always convert Zexus objects to Python objects
            python_data = File._zexus_to_python(data)
            json_str = json.dumps(python_data, indent=2)
            return File.write_text(path, String(json_str))
        except Exception as e:
            return EvaluationError(f"JSON write error: {str(e)}")

    @staticmethod
    def append_text(path, content):
        try:
            if isinstance(path, String):
                path = path.value
            if isinstance(content, String):
                content = content.value
            with open(path, 'a', encoding='utf-8') as f:
                f.write(content + '\n')
            return Boolean(True)
        except Exception as e:
            return EvaluationError(f"File append error: {str(e)}")

    @staticmethod
    def list_directory(path):
        try:
            if isinstance(path, String):
                path = path.value
            files = os.listdir(path)
            return List([String(f) for f in files])
        except Exception as e:
            return EvaluationError(f"Directory list error: {str(e)}")

    # === ADVANCED TIER ===
    @staticmethod
    def read_chunk(path, offset, length):
        try:
            if isinstance(path, String):
                path = path.value
            if isinstance(offset, Integer):
                offset = offset.value
            if isinstance(length, Integer):
                length = length.value

            with open(path, 'rb') as f:
                f.seek(offset)
                data = f.read(length)
                return String(data.hex())  # Return as hex string
        except Exception as e:
            return EvaluationError(f"File chunk read error: {str(e)}")

    @staticmethod
    def write_chunk(path, offset, data):
        try:
            if isinstance(path, String):
                path = path.value
            if isinstance(offset, Integer):
                offset = offset.value
            if isinstance(data, String):
                data = bytes.fromhex(data.value)

            with open(path, 'r+b') as f:
                f.seek(offset)
                f.write(data)
            return Boolean(True)
        except Exception as e:
            return EvaluationError(f"File chunk write error: {str(e)}")

    @staticmethod
    def atomic_write(path, data):
        """Atomic write to prevent corruption"""
        try:
            if isinstance(path, String):
                path = path.value

            # Write to temporary file first
            temp_path = path + '.tmp'
            result = File.write_text(temp_path, data)
            if result == Boolean(True):
                # Atomic rename
                os.replace(temp_path, path)
                return Boolean(True)
            return result
        except Exception as e:
            return EvaluationError(f"Atomic write error: {str(e)}")

    # File locking for concurrent access
    _file_locks = {}
    _lock = Lock()

    @staticmethod
    def lock_file(path, timeout=None):
        """Lock file for exclusive access.

        SECURITY (M9): Previously this could block forever and hang the interpreter.
        A default timeout is applied (configurable via env var).

        Args:
            path: File path (string or String)
            timeout: Optional seconds (Integer/Float/str/number). If omitted, uses
                $ZEXUS_FILE_LOCK_TIMEOUT_SECONDS (default: 30).
        """
        try:
            if isinstance(path, String):
                path = path.value

            # Normalize key to reduce accidental duplicate lock objects
            try:
                lock_key = os.path.realpath(str(path))
            except Exception:
                lock_key = str(path)

            timeout_seconds = None
            if timeout is None:
                env_timeout = os.environ.get("ZEXUS_FILE_LOCK_TIMEOUT_SECONDS", "30")
                try:
                    timeout_seconds = float(env_timeout)
                except Exception:
                    timeout_seconds = 30.0
            else:
                if isinstance(timeout, Integer):
                    timeout_seconds = float(timeout.value)
                elif isinstance(timeout, Float):
                    timeout_seconds = float(timeout.value)
                elif isinstance(timeout, String):
                    timeout_seconds = float(timeout.value)
                else:
                    timeout_seconds = float(timeout)

            if timeout_seconds is not None and timeout_seconds < 0:
                timeout_seconds = 0.0

            with File._lock:
                lock_obj = File._file_locks.get(lock_key)
                if lock_obj is None:
                    lock_obj = Lock()
                    File._file_locks[lock_key] = lock_obj

            # Acquire without holding the global map lock
            acquired = False
            if timeout_seconds is None:
                # Shouldn't happen, but keep backward-compatible behavior
                lock_obj.acquire()
                acquired = True
            else:
                acquired = lock_obj.acquire(timeout=timeout_seconds)

            if acquired:
                return Boolean(True)
            return EvaluationError(f"File lock timeout after {timeout_seconds:.3f}s")
        except Exception as e:
            return EvaluationError(f"File lock error: {str(e)}")

    @staticmethod
    def unlock_file(path):
        """Unlock file"""
        try:
            if isinstance(path, String):
                path = path.value

            try:
                lock_key = os.path.realpath(str(path))
            except Exception:
                lock_key = str(path)

            with File._lock:
                lock_obj = File._file_locks.get(lock_key)
                if lock_obj is not None:
                    try:
                        if hasattr(lock_obj, "locked") and not lock_obj.locked():
                            return Boolean(False)
                        lock_obj.release()
                        return Boolean(True)
                    except RuntimeError:
                        # Release on an unlocked lock
                        return Boolean(False)
                return Boolean(False)
        except Exception as e:
            return EvaluationError(f"File unlock error: {str(e)}")

    # Helper methods for data conversion
    @staticmethod
    def _python_to_zexus(value):
        if isinstance(value, dict):
            pairs = {}
            for k, v in value.items():
                pairs[k] = File._python_to_zexus(v)
            return Map(pairs)
        elif isinstance(value, list):
            return List([File._python_to_zexus(item) for item in value])
        elif isinstance(value, str):
            return String(value)
        elif isinstance(value, int):
            return Integer(value)
        elif isinstance(value, float):
            return Float(value)
        elif isinstance(value, bool):
            return Boolean(value)
        else:
            return String(str(value))

    @staticmethod
    def _zexus_to_python(value):
        if isinstance(value, Map):
            # Handle both String keys and regular string keys
            result = {}
            for k, v in value.pairs.items():
                key = k.value if isinstance(k, String) else str(k)
                result[key] = File._zexus_to_python(v)
            return result
        elif isinstance(value, List):
            return [File._zexus_to_python(item) for item in value.elements]
        elif isinstance(value, String):
            return value.value
        elif isinstance(value, Integer):
            return value.value
        elif isinstance(value, Float):
            return value.value
        elif isinstance(value, Boolean):
            return value.value
        elif value == Null() or value is None:
            return None
        elif isinstance(value, dict):
            # Handle plain Python dicts (fallback)
            return {str(k): File._zexus_to_python(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Handle plain Python lists (fallback)
            return [File._zexus_to_python(item) for item in value]
        else:
            return str(value)

# Debug utility for enhanced error tracking
class Debug(Object):
    def type(self):
        return "DEBUG_UTILITY"

    def inspect(self):
        return "<Debug utilities>"

    @staticmethod
    def log(message, value=None):
        """Log debug information with optional value"""
        if isinstance(message, String):
            message = message.value

        debug_msg = f"🔍 DEBUG: {message}"
        if value is not None:
            debug_msg += f" → {value.inspect() if hasattr(value, 'inspect') else value}"

        print(debug_msg)
        return value if value is not None else Boolean(True)

    @staticmethod
    def trace(message):
        """Add stack trace to debug output"""
        import traceback
        if isinstance(message, String):
            message = message.value

        print(f"🔍 TRACE: {message}")
        print("Stack trace:")
        for line in traceback.format_stack()[:-1]:
            print(f"  {line.strip()}")
        return Boolean(True)

# Global dependency collector stack for WATCH feature
_dependency_collector_stack = []

def start_collecting_dependencies():
    _dependency_collector_stack.append(set())

def stop_collecting_dependencies():
    if _dependency_collector_stack:
        return _dependency_collector_stack.pop()
    return set()

def record_dependency(env, name):
    if _dependency_collector_stack:
        _dependency_collector_stack[-1].add((env, name))

class Environment:
    def __init__(self, outer=None, persistence_scope=None):
        self.store = {}
        self.const_vars = set()  # Track const variables
        self.outer = outer
        self.exports = {}
        # Debug tracking
        self.debug_mode = False
        # Reactive watchers: name -> list of (callback_fn, context_env)
        self.watchers = {}
        # Persistence support
        self.persistence_scope = persistence_scope
        self._persistent_storage = None
        self._memory_tracker = None
        self._init_persistence()

    def get(self, name):
        val = self.store.get(name)
        if val is not None:
            record_dependency(self, name)
            return val
            
        if self.outer is not None:
            return self.outer.get(name)
        return None

    def set(self, name, val):
        # Check if trying to reassign a const variable
        if name in self.const_vars:
            from .object import EvaluationError
            raise ValueError(f"Cannot reassign const variable '{name}'")
        self.store[name] = val
        self.notify_watchers(name, val)
        return val

    def clone_for_closure(self):
        """Create a shallow copy of the environment for closure capture."""
        cloned = Environment(outer=self.outer, persistence_scope=self.persistence_scope)
        cloned.store = dict(self.store)
        cloned.const_vars = set(self.const_vars)
        cloned.exports = dict(self.exports)
        cloned.watchers = {k: list(v) for k, v in self.watchers.items()}
        cloned.debug_mode = self.debug_mode
        cloned._persistent_storage = self._persistent_storage
        cloned._memory_tracker = self._memory_tracker
        return cloned
    
    def assign(self, name, val):
        """Assign to an existing variable in the scope chain, or create in current if not found."""
        # 1. Check current scope
        if name in self.store:
            if hasattr(self, 'const_vars') and name in self.const_vars:
                raise ValueError(f"Cannot reassign const variable '{name}'")
            self.store[name] = val
            if hasattr(self, 'notify_watchers'):
                self.notify_watchers(name, val)
            return val
        
        # 2. Check outer scope
        if self.outer is not None:
            # Let's try to find where it is defined first.
            scope = self.outer
            while scope is not None:
                if name in scope.store:
                    # Check for const (defensive - some envs might not have const_vars)
                    if hasattr(scope, 'const_vars') and name in scope.const_vars:
                        raise ValueError(f"Cannot reassign const variable '{name}'")
                    scope.store[name] = val
                    if hasattr(scope, 'notify_watchers'):
                        scope.notify_watchers(name, val)
                    return val
                scope = scope.outer
            
            # Not found anywhere -> Create in CURRENT scope (local)
            self.store[name] = val
            if hasattr(self, 'notify_watchers'):
                self.notify_watchers(name, val)
            return val
        
        # 3. No outer scope (Global) -> Create here
        self.store[name] = val
        if hasattr(self, 'notify_watchers'):
            self.notify_watchers(name, val)
        return val
    
    def _has_variable(self, name):
        """Check if a variable name exists in this scope or any outer scope."""
        if name in self.store:
            return True
        if self.outer and hasattr(self.outer, '_has_variable'):
            return self.outer._has_variable(name)
        return False

    def set_const(self, name, val):
        """Set a constant (immutable) variable"""
        # Only check CURRENT scope for const shadowing - allow shadowing in nested scopes
        if name in self.store and name in self.const_vars:
            from .object import EvaluationError
            raise ValueError(f"Cannot reassign const variable '{name}'")
        self.store[name] = val
        self.const_vars.add(name)
        return val

    def export(self, name, value):
        self.exports[name] = value
        return value

    def get_exports(self):
        return self.exports

    def enable_debug(self):
        self.debug_mode = True

    def disable_debug(self):
        self.debug_mode = False

    def add_watcher(self, name, callback):
        if name not in self.watchers:
            self.watchers[name] = []
        self.watchers[name].append(callback)

    def notify_watchers(self, name, new_val):
        if name in self.watchers:
            # Copy list to avoid modification during iteration issues
            callbacks = self.watchers[name][:]
            for cb in callbacks:
                try:
                    cb(new_val)
                except Exception as e:
                    print(f"Error in watcher for {name}: {e}")
    
    # === PERSISTENCE METHODS ===
    
    def _init_persistence(self):
        """Initialize persistence system if scope is provided"""
        if self.persistence_scope:
            try:
                from .persistence import PersistentStorage, MemoryTracker
                self._persistent_storage = PersistentStorage(self.persistence_scope)
                self._memory_tracker = MemoryTracker()
                self._memory_tracker.start_tracking()
            except (ImportError, Exception):
                # Persistence module not available or error - continue without it
                pass
    
    def set_persistent(self, name, val):
        """Set a persistent variable that survives program restarts"""
        if self._persistent_storage:
            self._persistent_storage.set(name, val)
        # Also store in regular memory for access
        self.store[name] = val
        self.notify_watchers(name, val)
        return val
    
    def get_persistent(self, name, default=None):
        """Get a persistent variable"""
        if self._persistent_storage:
            val = self._persistent_storage.get(name)
            if val is not None:
                return val
        # Fallback to regular store
        return self.store.get(name, default)
    
    def delete_persistent(self, name):
        """Delete a persistent variable"""
        if self._persistent_storage:
            self._persistent_storage.delete(name)
        if name in self.store:
            del self.store[name]
    
    def get_memory_stats(self):
        """Get current memory tracking statistics"""
        if self._memory_tracker:
            return self._memory_tracker.get_stats()
        return {"tracked_objects": 0, "message": "Memory tracking not enabled"}
    
    def enable_memory_tracking(self):
        """Enable memory leak detection"""
        if not self._memory_tracker:
            try:
                from .persistence import MemoryTracker
                self._memory_tracker = MemoryTracker()
                self._memory_tracker.start_tracking()
            except (ImportError, Exception):
                pass
    
    def cleanup_persistence(self):
        """Clean up persistence resources"""
        if self._memory_tracker:
            self._memory_tracker.stop_tracking()
        if self._persistent_storage:
            self._persistent_storage.close()

# Global constants
NULL = Null()
TRUE = Boolean(True)
FALSE = Boolean(False)

# File object for RAII pattern (using statement)
class FileHandle(Object):
    """File object that supports cleanup via close() method"""
    def __init__(self, path, mode='r'):
        self.path = path
        self.mode = mode
        self.handle = None
        self.closed = False
        
    def open(self):
        """Open the file"""
        if not self.handle:
            try:
                self.handle = open(self.path, self.mode)
            except Exception as e:
                raise Exception(f"Failed to open file {self.path}: {e}")
        return self
    
    def close(self):
        """Close the file (called by using statement cleanup)"""
        if self.handle and not self.closed:
            self.handle.close()
            self.closed = True
    
    def read(self):
        """Read file contents"""
        if not self.handle:
            self.open()
        return String(self.handle.read())
    
    def write(self, content):
        """Write content to file"""
        if not self.handle:
            self.open()
        self.handle.write(content)
        return NULL
    
    def inspect(self):
        status = "closed" if self.closed else "open"
        return f"File({self.path}, {status})"
    
    def type(self):
        return "FILE"

# EvaluationError class for error handling
class EvaluationError(Object):
    def __init__(self, message, line=None, column=None, stack_trace=None, filename=None, suggestion=None):
        self.message = message
        self.line = line
        self.column = column
        self.stack_trace = stack_trace or []
        self.filename = filename
        self.suggestion = suggestion
        self._is_error = True

    def inspect(self):
        return f"❌ Error: {self.message}"

    def type(self):
        return "ERROR"

    def __str__(self):
        """Format as a nice error message"""
        # Try to use error reporter if available
        try:
            from .error_reporter import get_error_reporter, ZexusError, ErrorCategory
            
            error_reporter = get_error_reporter()
            
            # Create a formatted error
            # We use a temporary ZexusError for formatting
            temp_error = ZexusError(
                message=self.message,
                category=ErrorCategory.USER_CODE,
                filename=self.filename or "<runtime>",
                line=self.line,
                column=self.column,
                source_line=error_reporter.get_source_line(self.filename, self.line) if self.line else None,
                suggestion=self.suggestion
            )
            
            # Add stack trace if available
            if self.stack_trace:
                formatted_trace = []
                for frame in self.stack_trace[-5:]:
                    if isinstance(frame, str):
                        formatted_trace.append(frame)
                    elif isinstance(frame, tuple) and len(frame) == 2:
                        node_type, line = frame
                        item = f"  at {node_type.__name__}"
                        if line:
                            item += f" (line {line})"
                        formatted_trace.append(item)
                
                trace = "\n".join(formatted_trace)
                temp_error.message += f"\n\nStack trace:\n{trace}"
            
            return temp_error.format_error()
        except Exception:
            # Fallback to simple format if error reporter not available
            location = f"Line {self.line}:{self.column}" if self.line and self.column else "Unknown location"
            
            # Format simple trace
            formatted_trace = []
            if self.stack_trace:
                for frame in self.stack_trace[-3:]:
                    if isinstance(frame, str):
                        formatted_trace.append(frame)
                    elif isinstance(frame, tuple) and len(frame) == 2:
                        node_type, line = frame
                        item = f"  at {node_type.__name__}"
                        if line:
                            item += f" (line {line})"
                        formatted_trace.append(item)
            
            trace = "\n".join(formatted_trace)
            trace_section = f"\n   Stack:\n{trace}" if trace else ""
            suggestion_section = f"\n   💡 Suggestion: {self.suggestion}" if self.suggestion else ""
            return f"❌ Runtime Error at {location}\n   {self.message}{suggestion_section}{trace_section}"

    def __len__(self):
        """Support len() on errors to prevent secondary failures"""
        return len(self.message)