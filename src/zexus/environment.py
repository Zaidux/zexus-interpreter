# environment.py

class Environment:
    def __init__(self, outer=None):
        self.store = {}
        self.outer = outer
        self.exports = {}
        self.modules = {}
        self._debug = False

    # ---- Mapping protocol helpers -------------------------------------------------

    def __contains__(self, name):
        if name in self.store:
            return True

        if "." in name:
            module_name, var_name = name.split(".", 1)
            module = self.modules.get(module_name)
            if module and var_name in module:
                return True

        if self.outer is not None and hasattr(self.outer, '__contains__'):
            try:
                return name in self.outer
            except TypeError:
                pass
        return False

    def __getitem__(self, name):
        return self.get(name)

    def __setitem__(self, name, value):
        self.set(name, value)

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def keys(self):
        return self.store.keys()

    def items(self):
        return self.store.items()

    def values(self):
        return self.store.values()

    def copy(self):
        return dict(self.store)

    def update(self, other):
        if other is None:
            return
        if hasattr(other, 'items'):
            for key, value in other.items():
                self.set(key, value)
        else:
            for key in other:
                self.set(key, other[key])

    def setdefault(self, name, default=None):
        if name in self:
            return self.get(name)
        self.set(name, default)
        return default

    # ---- Core environment operations ---------------------------------------------

    def get(self, name, default=None):
        """Get a value from the environment"""
        # Check local store (allow storing None explicitly)
        if name in self.store:
            return self.store[name]
            
        # Check modules
        if "." in name:
            module_name, var_name = name.split(".", 1)
            module = self.modules.get(module_name)
            if module:
                return module.get(var_name)
                
        # Check outer scope
        if self.outer:
            getter = getattr(self.outer, 'get', None)
            if callable(getter):
                return getter(name, default)
            try:
                return self.outer[name]
            except (KeyError, TypeError, AttributeError):
                pass
        
        return default

    def set(self, name, value):
        """Set a value in the environment (creates new variable)"""
        if "." in name:
            module_name, var_name = name.split(".", 1)
            module = self.modules.get(module_name)
            if module:
                module.set(var_name, value)
            else:
                # Create new module environment
                module = Environment(self)
                module.set(var_name, value)
                self.modules[module_name] = module
        else:
            self.store[name] = value
    
    def assign(self, name, value):
        """Assign to an existing variable or create if doesn't exist.
        
        This is used for reassignment (like in loops). It will:
        1. Update the variable in the scope where it was first defined
        2. Create a new variable in current scope if it doesn't exist anywhere
        """
        # Check if variable exists in current scope
        if name in self.store:
            self.store[name] = value
            return
        
        # Check if exists in outer scopes by checking the store directly
        if self.outer:
            # Recursively check if the name exists in any outer scope
            if self._has_variable(name):
                # Try to assign in outer scope
                self.outer.assign(name, value)
                return
        
        # Variable doesn't exist anywhere, create it in current scope
        self.store[name] = value
    
    def _has_variable(self, name):
        """Check if a variable name exists in this scope or any outer scope."""
        if name in self.store:
            return True
        if self.outer and hasattr(self.outer, '_has_variable'):
            return self.outer._has_variable(name)
        return False

    def export(self, name, value):
        """Export a value"""
        self.exports[name] = value
        self.store[name] = value

    def get_exports(self):
        """Get all exported values"""
        return self.exports.copy()

    def import_module(self, name, module_env):
        """Import a module environment"""
        self.modules[name] = module_env

    def enable_debug(self):
        """Enable debug logging"""
        self._debug = True

    def disable_debug(self):
        """Disable debug logging"""
        self._debug = False

    def debug_log(self, message):
        """Log debug message if debug is enabled"""
        if self._debug:
            print(f"[ENV] {message}")