# object.py (COMPLETE PHASE 1 UPDATE)
import time
import random
import json
import os
import sys
from threading import Lock

class Object:
    def inspect(self):
        raise NotImplementedError("Subclasses must implement this method")

# === EXISTING TYPES (unchanged) ===
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
    def __init__(self, value): self.value = value
    def inspect(self): return self.value
    def type(self): return "STRING"
    def __str__(self): return self.value

class List(Object):
    def __init__(self, elements): self.elements = elements
    def inspect(self):
        elements_str = ", ".join([el.inspect() for el in self.elements])
        return f"[{elements_str}]"
    def type(self): return "LIST"

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

    def get(self, key):
        """Get value by key (compatible with string keys)"""
        return self.pairs.get(key)

    def set(self, key, value):
        """Set value for key, blocking modification if key is sealed.

        We avoid importing SealedObject to prevent circular imports; instead use
        a runtime name-check of the wrapper class.
        """
        existing = self.pairs.get(key)
        if existing is not None and existing.__class__.__name__ == 'SealedObject':
            # Raise EvaluationError (defined later in this module) to signal runtime error
            raise EvaluationError(f"Cannot modify sealed map key: {key}")
        self.pairs[key] = value

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

class Builtin(Object):
    def __init__(self, fn, name=""):
        self.fn = fn
        self.name = name
    def inspect(self): return f"<built-in function: {self.name}>"
    def type(self): return "BUILTIN"

# === NEW: PHASE 1 UTILITY CLASSES ===

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
            with open(path, 'r', encoding='utf-8') as f:
                return String(f.read())
        except Exception as e:
            return EvaluationError(f"File read error: {str(e)}")

    @staticmethod
    def write_text(path, content):
        try:
            if isinstance(path, String):
                path = path.value
            if isinstance(content, String):
                content = content.value
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return Boolean(True)
        except Exception as e:
            return EvaluationError(f"File write error: {str(e)}")

    @staticmethod
    def exists(path):
        if isinstance(path, String):
            path = path.value
        return Boolean(os.path.exists(path))

    # === MEDIUM TIER ===
    @staticmethod
    def read_json(path):
        try:
            content = File.read_text(path)
            if isinstance(content, EvaluationError):
                return content
            data = json.loads(content.value)
            # Convert to Zexus Map
            pairs = {}
            for key, value in data.items():
                pairs[key] = File._python_to_zexus(value)
            return Map(pairs)
        except Exception as e:
            return EvaluationError(f"JSON read error: {str(e)}")

    @staticmethod
    def write_json(path, data):
        try:
            if isinstance(data, Map):
                python_data = File._zexus_to_python(data)
            else:
                python_data = data
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
    def lock_file(path):
        """Lock file for exclusive access"""
        try:
            if isinstance(path, String):
                path = path.value

            with File._lock:
                if path not in File._file_locks:
                    File._file_locks[path] = Lock()

                File._file_locks[path].acquire()
                return Boolean(True)
        except Exception as e:
            return EvaluationError(f"File lock error: {str(e)}")

    @staticmethod
    def unlock_file(path):
        """Unlock file"""
        try:
            if isinstance(path, String):
                path = path.value

            with File._lock:
                if path in File._file_locks:
                    File._file_locks[path].release()
                    return Boolean(True)
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
            return {k: File._zexus_to_python(v) for k, v in value.pairs.items()}
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
        elif value == Null():
            return None
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

class Environment:
    def __init__(self, outer=None):
        self.store = {}
        self.outer = outer
        self.exports = {}
        # Debug tracking
        self.debug_mode = False

    def get(self, name):
        val = self.store.get(name)
        if val is None and self.outer is not None:
            return self.outer.get(name)
        return val

    def set(self, name, val):
        self.store[name] = val
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

# Global constants
NULL = Null()
TRUE = Boolean(True)
FALSE = Boolean(False)

# EvaluationError class for error handling
class EvaluationError(Object):
    def __init__(self, message, line=None, column=None, stack_trace=None):
        self.message = message
        self.line = line
        self.column = column
        self.stack_trace = stack_trace or []

    def inspect(self):
        return f"❌ Error: {self.message}"

    def type(self):
        return "ERROR"

    def __str__(self):
        location = f"Line {self.line}:{self.column}" if self.line and self.column else "Unknown location"
        trace = "\n".join(self.stack_trace[-3:]) if self.stack_trace else ""
        trace_section = f"\n   Stack:\n{trace}" if trace else ""
        return f"❌ Runtime Error at {location}\n   {self.message}{trace_section}"

    def __len__(self):
        """Support len() on errors to prevent secondary failures"""
        return len(self.message)
