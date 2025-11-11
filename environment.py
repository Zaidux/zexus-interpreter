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
        """Set a value in the environment"""
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