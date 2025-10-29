# object.py (Updated for Built-ins)

class Object:
    def inspect(self):
        raise NotImplementedError("Subclasses must implement this method")

# ... (Integer, Boolean, Null, String, List, ReturnValue, Action classes are unchanged) ...

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
        self.pairs = pairs  # dict of key -> value
    
    def type(self): return "MAP"
    
    def inspect(self):
        pairs = []
        for key, value in self.pairs.items():
            # Handle string keys (they don't have .inspect())
            key_str = key.inspect() if hasattr(key, 'inspect') else str(key)
            value_str = value.inspect() if hasattr(value, 'inspect') else str(value)
            pairs.append(f"{key_str}: {value_str}")
        return "{" + ", ".join(pairs) + "}"

# Add to object.py

class EmbeddedCode(Object):
    def __init__(self, name, language, code):
        self.name = name
        self.language = language
        self.code = code
        
    def inspect(self):
        return f"<embedded {self.language} code: {self.name}>"
        
    def type(self):
        return "EMBEDDED_CODE"

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

# NEW: The object to represent a built-in function
class Builtin(Object):
    def __init__(self, fn):
        self.fn = fn # Stores the native Python function

    def inspect(self):
        return "<built-in function>"
    
    def type(self):
        return "BUILTIN"

class Environment:
    def __init__(self, outer=None):
        self.store = {}
        self.outer = outer

    def get(self, name):
        val = self.store.get(name)
        if val is None and self.outer is not None:
            return self.outer.get(name)
        return val

    def set(self, name, val):
        self.store[name] = val
        return val
