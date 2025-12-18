# src/zexus/dependency_injection.py
"""
Dependency Injection and Module Mocking System for Zexus
Implements IoC (Inversion of Control) with EXPORT/EXTERNAL keywords
"""

from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from .object import Object, String, Integer, NULL, EvaluationError, Builtin, Action


# ===============================================
# EXECUTION MODES
# ===============================================

class ExecutionMode(Enum):
    """Execution modes for dependency injection"""
    PRODUCTION = "production"
    DEBUG = "debug"
    TEST = "test"
    SANDBOX = "sandbox"


# Current execution mode (global state)
_current_mode = ExecutionMode.PRODUCTION


def set_execution_mode(mode: ExecutionMode):
    """Set the current execution mode"""
    global _current_mode
    _current_mode = mode


def get_execution_mode() -> ExecutionMode:
    """Get the current execution mode"""
    return _current_mode


# ===============================================
# DEPENDENCY CONTRACTS
# ===============================================

class DependencyContract:
    """Defines a required dependency"""
    
    def __init__(self, name: str, dep_type: str = "any", required: bool = True, default_value: Any = None):
        self.name = name
        self.type = dep_type  # Type hint: "function", "object", "const", "any"
        self.required = required
        self.default_value = default_value
        self.description = ""
    
    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate that a provided value meets the contract"""
        if value is None or value is NULL:
            if self.required and self.default_value is None:
                return False, f"Required dependency '{self.name}' not provided"
            return True, ""
        
        # Type validation
        if self.type == "function":
            if not isinstance(value, (Action, Builtin, Callable)):
                return False, f"Dependency '{self.name}' must be a function, got {type(value)}"
        elif self.type == "const":
            # For const, we just check it's not mutable (no validation currently)
            pass
        elif self.type == "object":
            if not isinstance(value, Object):
                return False, f"Dependency '{self.name}' must be an object, got {type(value)}"
        # "any" type accepts anything
        
        return True, ""


# ===============================================
# MODULE DEPENDENCY CONTAINER
# ===============================================

class DependencyContainer:
    """Container for a module's dependencies"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.contracts: Dict[str, DependencyContract] = {}
        self.provided: Dict[str, Any] = {}
        self.mocks: Dict[str, Any] = {}  # For testing
    
    def declare_dependency(self, name: str, dep_type: str = "any", required: bool = True, default_value: Any = None):
        """Declare a dependency contract"""
        contract = DependencyContract(name, dep_type, required, default_value)
        self.contracts[name] = contract
    
    def provide(self, name: str, value: Any):
        """Provide a dependency value"""
        if name not in self.contracts:
            raise ValueError(f"No contract declared for dependency '{name}'")
        
        # Validate against contract
        contract = self.contracts[name]
        valid, error_msg = contract.validate(value)
        if not valid:
            raise ValueError(f"Dependency validation failed: {error_msg}")
        
        self.provided[name] = value
    
    def mock(self, name: str, mock_value: Any):
        """Provide a mock for testing"""
        if name not in self.contracts:
            raise ValueError(f"No contract declared for dependency '{name}'")
        
        self.mocks[name] = mock_value
    
    def get(self, name: str) -> Any:
        """Get a dependency value (mock in TEST mode, real otherwise)"""
        mode = get_execution_mode()
        
        # In TEST/SANDBOX mode, prefer mocks
        if mode in (ExecutionMode.TEST, ExecutionMode.SANDBOX):
            if name in self.mocks:
                return self.mocks[name]
        
        # Check provided dependencies
        if name in self.provided:
            return self.provided[name]
        
        # Check for default value
        if name in self.contracts:
            contract = self.contracts[name]
            if contract.default_value is not None:
                return contract.default_value
            
            if not contract.required:
                return NULL
        
        # Dependency not satisfied
        raise EvaluationError(f"Unsatisfied dependency: '{name}' in module '{self.module_name}'")
    
    def validate_all(self) -> tuple[bool, List[str]]:
        """Validate that all required dependencies are satisfied"""
        errors = []
        
        for name, contract in self.contracts.items():
            if contract.required:
                try:
                    self.get(name)
                except EvaluationError as e:
                    errors.append(str(e))
        
        return len(errors) == 0, errors
    
    def list_dependencies(self) -> List[Dict[str, Any]]:
        """List all declared dependencies"""
        return [
            {
                'name': name,
                'type': contract.type,
                'required': contract.required,
                'satisfied': name in self.provided or name in self.mocks or contract.default_value is not None
            }
            for name, contract in self.contracts.items()
        ]
    
    def clear_mocks(self):
        """Clear all mocks"""
        self.mocks.clear()


# ===============================================
# DEPENDENCY INJECTION REGISTRY
# ===============================================

class DIRegistry:
    """Global registry for dependency injection containers"""
    
    def __init__(self):
        self.containers: Dict[str, DependencyContainer] = {}
        self.module_factories: Dict[str, Callable] = {}
    
    def register_module(self, module_name: str) -> DependencyContainer:
        """Register a module and return its dependency container"""
        if module_name not in self.containers:
            self.containers[module_name] = DependencyContainer(module_name)
        return self.containers[module_name]
    
    def get_container(self, module_name: str) -> Optional[DependencyContainer]:
        """Get dependency container for a module, creating it if it doesn't exist"""
        if module_name not in self.containers:
            self.containers[module_name] = DependencyContainer(module_name)
        return self.containers[module_name]
    
    def provide_dependency(self, module_name: str, dep_name: str, value: Any):
        """Provide a dependency to a module"""
        container = self.get_container(module_name)
        if not container:
            raise ValueError(f"Module '{module_name}' not registered")
        container.provide(dep_name, value)
    
    def mock_dependency(self, module_name: str, dep_name: str, mock_value: Any):
        """Mock a dependency for testing"""
        container = self.get_container(module_name)
        if not container:
            raise ValueError(f"Module '{module_name}' not registered")
        container.mock(dep_name, mock_value)
    
    def validate_module(self, module_name: str) -> tuple[bool, List[str]]:
        """Validate all dependencies for a module"""
        container = self.get_container(module_name)
        if not container:
            return False, [f"Module '{module_name}' not registered"]
        return container.validate_all()
    
    def clear_all_mocks(self):
        """Clear all mocks from all modules"""
        for container in self.containers.values():
            container.clear_mocks()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get DI statistics"""
        total_deps = 0
        satisfied_deps = 0
        mocked_deps = 0
        
        for container in self.containers.values():
            deps = container.list_dependencies()
            total_deps += len(deps)
            satisfied_deps += sum(1 for d in deps if d['satisfied'])
            mocked_deps += len(container.mocks)
        
        return {
            'registered_modules': len(self.containers),
            'total_dependencies': total_deps,
            'satisfied_dependencies': satisfied_deps,
            'mocked_dependencies': mocked_deps,
            'execution_mode': get_execution_mode().value
        }


# Global DI registry
_di_registry = DIRegistry()


def get_di_registry() -> DIRegistry:
    """Get the global DI registry"""
    return _di_registry


def get_dependency_manager() -> DIRegistry:
    """Get the dependency manager (alias for get_di_registry)"""
    return _di_registry


# ===============================================
# MODULE BUILDER (For EXPORT blocks)
# ===============================================

class ModuleBuilder:
    """Builder for creating modules with dependency contracts"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.container = get_di_registry().register_module(module_name)
    
    def require_external(self, name: str, dep_type: str = "any", required: bool = True):
        """Declare an EXTERNAL dependency"""
        self.container.declare_dependency(name, dep_type, required)
        return self
    
    def require_const(self, name: str, default_value: Any = None):
        """Declare an EXTERNAL CONST dependency"""
        self.container.declare_dependency(name, "const", required=(default_value is None), default_value=default_value)
        return self
    
    def provide(self, name: str, value: Any):
        """Provide a dependency (for setup/testing)"""
        self.container.provide(name, value)
        return self
    
    def build(self) -> DependencyContainer:
        """Build and return the container"""
        return self.container


# ===============================================
# MOCK FACTORY
# ===============================================

class MockFactory:
    """Factory for creating mock objects for testing"""
    
    @staticmethod
    def create_function_mock(name: str, return_value: Any = NULL) -> Builtin:
        """Create a mock function"""
        def mock_fn(*args):
            return return_value
        return Builtin(mock_fn, name=f"mock_{name}")
    
    @staticmethod
    def create_object_mock(properties: Dict[str, Any]) -> Object:
        """Create a mock object with properties"""
        from .object import Map
        return Map(properties)
    
    @staticmethod
    def create_api_mock(endpoints: Dict[str, Any]) -> Object:
        """Create a mock API with endpoints"""
        from .object import Map
        
        mock_endpoints = {}
        for endpoint, response in endpoints.items():
            def make_endpoint_fn(resp):
                def endpoint_fn(*args):
                    return resp
                return endpoint_fn
            mock_endpoints[endpoint] = Builtin(make_endpoint_fn(response), name=f"mock_{endpoint}")
        
        return Map(mock_endpoints)
    
    @staticmethod
    def create_spy_function(name: str, original_fn: Optional[Callable] = None) -> tuple[Builtin, List]:
        """Create a spy function that records calls"""
        call_log = []
        
        def spy_fn(*args):
            call_log.append({
                'args': args,
                'timestamp': __import__('time').time()
            })
            if original_fn:
                return original_fn(*args)
            return NULL
        
        return Builtin(spy_fn, name=f"spy_{name}"), call_log


# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def create_module(module_name: str) -> ModuleBuilder:
    """Create a new module with dependency injection"""
    return ModuleBuilder(module_name)


def inject_dependency(module_name: str, dep_name: str, value: Any):
    """Inject a dependency into a module"""
    registry = get_di_registry()
    registry.provide_dependency(module_name, dep_name, value)


def mock_dependency(module_name: str, dep_name: str, mock_value: Any):
    """Mock a dependency for testing"""
    registry = get_di_registry()
    registry.mock_dependency(module_name, dep_name, mock_value)


def get_dependency(module_name: str, dep_name: str) -> Any:
    """Get a dependency from a module"""
    registry = get_di_registry()
    container = registry.get_container(module_name)
    if not container:
        raise ValueError(f"Module '{module_name}' not registered")
    return container.get(dep_name)


def validate_dependencies(module_name: str) -> tuple[bool, List[str]]:
    """Validate all dependencies for a module"""
    registry = get_di_registry()
    return registry.validate_module(module_name)


def clear_all_mocks():
    """Clear all mocks"""
    registry = get_di_registry()
    registry.clear_all_mocks()


def get_di_stats() -> Dict[str, Any]:
    """Get dependency injection statistics"""
    registry = get_di_registry()
    return registry.get_stats()


# ===============================================
# TESTING UTILITIES
# ===============================================

class TestContext:
    """Context manager for testing with mocks"""
    
    def __init__(self, module_name: str, mocks: Dict[str, Any]):
        self.module_name = module_name
        self.mocks = mocks
        self.original_mode = None
    
    def __enter__(self):
        """Enter test context"""
        self.original_mode = get_execution_mode()
        set_execution_mode(ExecutionMode.TEST)
        
        # Apply mocks
        for dep_name, mock_value in self.mocks.items():
            mock_dependency(self.module_name, dep_name, mock_value)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit test context"""
        # Restore original mode
        if self.original_mode:
            set_execution_mode(self.original_mode)
        
        # Clear mocks
        registry = get_di_registry()
        container = registry.get_container(self.module_name)
        if container:
            container.clear_mocks()
        
        return False


def test_with_mocks(module_name: str, mocks: Dict[str, Any]) -> TestContext:
    """Create a test context with mocks"""
    return TestContext(module_name, mocks)
