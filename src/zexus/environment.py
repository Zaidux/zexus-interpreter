# environment.py

class Environment:
    def __init__(self, outer=None):
        self.store = {}
        self.outer = outer
        self.exports = {}
        self.modules = {}
        self._debug = False

    def get(self, name):
        """Get a value from the environment"""
        # Check local store
        value = self.store.get(name)
        if value is not None:
            return value
            
        # Check modules
        if "." in name:
            module_name, var_name = name.split(".", 1)
            module = self.modules.get(module_name)
            if module:
                return module.get(var_name)
                
        # Check outer scope
        if self.outer:
            return self.outer.get(name)
            
        return None

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
        if self.outer:
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