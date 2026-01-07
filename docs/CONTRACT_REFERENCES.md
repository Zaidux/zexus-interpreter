# Contract-to-Contract References

## Overview

Zexus 1.6.7 introduces a powerful contract reference system that allows smart contracts to store and interact with references to other contract instances. References are fully transparent, serializable, and persist across state changes.

## Features

- **Transparent Delegation**: Contract references act exactly like the underlying contract instance
- **Method Calls**: Call actions on referenced contracts naturally
- **Attribute Access**: Access contract properties (address, name, etc.) through references
- **Persistence**: References survive serialization/deserialization
- **Type Safety**: References resolve at runtime with proper error handling

## How It Works

### Contract Registry

All deployed contracts are automatically registered in a global `CONTRACT_REGISTRY` that maps addresses to contract instances:

```python
CONTRACT_REGISTRY = {}  # address -> SmartContract
```

When a contract is created, it automatically registers itself:

```python
def __init__(self, ...):
    # ... contract initialization ...
    CONTRACT_REGISTRY[self.address] = self
```

### ContractReference Class

The `ContractReference` class stores a reference to a contract by its address:

```python
class ContractReference:
    def __init__(self, address, contract_name=None):
        self.address = address
        self.contract_name = contract_name
    
    def resolve(self):
        """Look up the actual contract instance"""
        return CONTRACT_REGISTRY.get(self.address)
    
    def call_method(self, method_name, args):
        """Delegate method calls to the contract"""
        contract = self.resolve()
        return contract.call_action(method_name, args, ...)
    
    def get_attr(self, attr_name):
        """Delegate attribute access to the contract"""
        contract = self.resolve()
        return getattr(contract, attr_name)
```

### Serialization

When a contract stores another contract instance, it's automatically converted to a `ContractReference` during serialization:

```python
# Serialization
SmartContract instance → {"__type__": "ContractReference", "address": "...", "name": "..."}

# Deserialization  
{"__type__": "ContractReference", ...} → ContractReference instance
```

## Usage Examples

### Basic Reference Storage

```zexus
contract Storage {
    state data: map<str, str> = {};
    
    action set(key: str, value: str) {
        data[key] = value;
    }
    
    action get(key: str) -> str {
        return data.get(key, "");
    }
}

contract Controller {
    state storage_ref: any = null;
    
    action init(storage_contract: any) {
        storage_ref = storage_contract;  # Stores reference
    }
    
    action store_data(key: str, value: str) {
        storage_ref.set(key, value);  # Method call through reference
    }
    
    action retrieve_data(key: str) -> str {
        return storage_ref.get(key);  # Method call through reference
    }
    
    action get_storage_address() -> str {
        return storage_ref.address;  # Attribute access through reference
    }
}
```

### Using the Contracts

```zexus
# Deploy storage contract
let storage = deploy Storage();
print("Storage deployed at: " + storage.address);

# Deploy controller contract
let controller = deploy Controller();

# Initialize controller with storage reference
controller.init(storage);

# Use controller to interact with storage
controller.store_data("test_key", "test_value");
let retrieved = controller.retrieve_data("test_key");
print("Retrieved: " + retrieved);  # Output: Retrieved: test_value

# Access storage address through controller
let addr = controller.get_storage_address();
print("Storage address: " + addr);
```

## Transparency

Contract references are completely transparent to Zexus code:

```zexus
# These all work the same whether you have a direct reference or ContractReference:
storage.set("key", "value")          # Direct call
storage_ref.set("key", "value")      # Through ContractReference - identical!

let addr = storage.address           # Direct access
let addr = storage_ref.address       # Through ContractReference - identical!
```

## Persistence

References persist across state changes:

```zexus
contract Manager {
    state contracts: map<str, any> = {};
    
    action register(name: str, contract: any) {
        contracts[name] = contract;  # Stored as ContractReference
    }
    
    action call_registered(name: str) {
        let contract = contracts.get(name, null);
        if (contract != null) {
            contract.some_action();  # Works after deserialization!
        }
    }
}
```

When the `Manager` contract's state is saved:
1. The contract instance is serialized to a `ContractReference`
2. The reference (with address) is stored in JSON
3. On load, the reference is deserialized back to a `ContractReference`
4. When accessed, it resolves to the actual contract via `CONTRACT_REGISTRY`

## Error Handling

If a referenced contract is not found:

```zexus
# If contract at address doesn't exist:
let result = invalid_ref.some_action();  
# Raises: EvaluationError: "Contract at address <addr> not found"
```

## Advanced Patterns

### Contract Factory

```zexus
contract Factory {
    state instances: list<any> = [];
    
    action create_instance() -> any {
        let instance = deploy SomeContract();
        instances.push(instance);
        return instance;
    }
    
    action call_all(method: str) {
        for (let inst in instances) {
            inst[method]();  # Dynamic method call on references
        }
    }
}
```

### Multi-Contract System

```zexus
contract System {
    state database: any = null;
    state logger: any = null;
    state auth: any = null;
    
    action init() {
        database = deploy Database();
        logger = deploy Logger();
        auth = deploy Auth();
    }
    
    action process(user: str, data: str) {
        if (auth.verify(user)) {
            database.store(data);
            logger.log("Stored data for " + user);
        }
    }
}
```

## Implementation Details

### Files Modified

- **src/zexus/object.py** (lines 162-245): `ContractReference` class
- **src/zexus/security.py**: 
  - `CONTRACT_REGISTRY` global dictionary
  - `SmartContract.__init__`: Auto-registration
  - `SmartContract.get()`: Property access support (lines 1193-1218)
  - Serialization support (lines 856-902)
  - Deserialization support (lines 904-975)
- **src/zexus/evaluator/core.py** (lines 643-658): `get_attr` delegation support

### Test Coverage

See [test_contract_refs.zx](../test_contract_refs.zx) for comprehensive test coverage including:
- Method calls through references
- Attribute access through references  
- Persistence across state changes
- Multi-contract interactions

## Best Practices

1. **Type Annotations**: Use `any` for contract references (type system doesn't have contract types yet)
2. **Null Checks**: Always check if a reference is `null` before using it
3. **Error Handling**: Wrap contract calls in try/catch if the contract might not exist
4. **Documentation**: Document which contracts your contract depends on

## Limitations

- Contract references are stored by address only
- If a contract is undeployed/destroyed, its references become invalid
- No type checking for contract references (use `any` type)
- References don't prevent garbage collection of contracts

## Future Enhancements

Potential improvements for future versions:
- Strong typing for contract references
- Reference counting for automatic cleanup
- Cross-chain contract references
- Contract upgrade/migration support with reference updates
