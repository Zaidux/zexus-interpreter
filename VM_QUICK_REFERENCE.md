# VM Integration - Quick Reference

## For Users

### Basic Usage
```python
# Automatic VM usage (recommended)
result = evaluate(program, env)  # VM used when beneficial

# Force direct evaluation
result = evaluate(program, env, use_vm=False)

# Enable debugging
result = evaluate(program, env, debug_mode=True, use_vm=True)
```

### Get Statistics
```python
from zexus.evaluator.core import Evaluator

evaluator = Evaluator(use_vm=True)
result = evaluator.eval_with_vm_support(program, env)
stats = evaluator.get_vm_stats()

print(f"Bytecode compiles: {stats['bytecode_compiles']}")
print(f"VM executions: {stats['vm_executions']}")
print(f"Fallbacks: {stats['vm_fallbacks']}")
print(f"Direct evals: {stats['direct_evals']}")
```

## For Developers

### Check if Node Can Use VM
```python
from zexus.evaluator.bytecode_compiler import should_use_vm_for_node

if should_use_vm_for_node(ast_node):
    # This node will benefit from VM execution
    pass
```

### Compile to Bytecode
```python
from zexus.evaluator.bytecode_compiler import EvaluatorBytecodeCompiler

compiler = EvaluatorBytecodeCompiler()
bytecode = compiler.compile(ast_node, optimize=True)

if compiler.errors:
    print(f"Errors: {compiler.errors}")
else:
    print(bytecode.disassemble())
```

### Execute Bytecode Directly
```python
from zexus.vm.vm import VM

vm = VM(builtins=my_builtins, env=my_env)
result = vm.execute(bytecode, debug=True)
```

### Build Bytecode Manually
```python
from zexus.vm.bytecode import BytecodeBuilder

builder = BytecodeBuilder()

# Load constant and store to variable
builder.emit_constant(42)
builder.emit_store('x')

# Load variables and add
builder.emit_load('x')
builder.emit_load('y')
builder.emit('ADD')

# Return result
builder.emit('RETURN')

bytecode = builder.build()
```

## Opcodes Reference

### Stack Operations
- `LOAD_CONST idx` - Push constant[idx] onto stack
- `LOAD_NAME idx` - Push variable value onto stack  
- `STORE_NAME idx` - Pop and store to variable
- `POP` - Pop and discard top of stack
- `DUP` - Duplicate top of stack

### Arithmetic
- `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `POW` - Binary operations
- `NEG` - Unary negation

### Comparison
- `EQ`, `NEQ`, `LT`, `GT`, `LTE`, `GTE` - Comparison operations

### Logical
- `AND`, `OR` - Boolean operations
- `NOT` - Boolean negation

### Control Flow
- `JUMP addr` - Unconditional jump
- `JUMP_IF_FALSE addr` - Conditional jump
- `RETURN` - Return from function

### Functions
- `CALL_NAME (name_idx, arg_count)` - Call function by name
- `CALL_FUNC_CONST (func_idx, arg_count)` - Call function descriptor
- `CALL_TOP arg_count` - Call function on stack
- `STORE_FUNC (name_idx, func_idx)` - Store function definition

### Collections
- `BUILD_LIST count` - Build list from stack items
- `BUILD_MAP count` - Build map from key-value pairs
- `INDEX` - Index into collection

### Async
- `SPAWN` - Spawn coroutine
- `AWAIT` - Await coroutine result

## Performance Tips

1. **Use VM for loops** - Loops automatically use VM for better performance
2. **Batch operations** - Multiple statements in a program benefit from VM
3. **Profile your code** - Check VM stats to see what's being optimized
4. **Trust the heuristics** - The evaluator knows when VM helps

## Troubleshooting

### VM Not Being Used
Check statistics:
```python
stats = evaluator.get_vm_stats()
if stats['bytecode_compiles'] == 0:
    # VM isn't being triggered
    # Possible reasons:
    # - Code too simple
    # - use_vm=False
    # - Unsupported features
```

### Compilation Errors
```python
compiler = EvaluatorBytecodeCompiler()
bytecode = compiler.compile(node)
if compiler.errors:
    for error in compiler.errors:
        print(f"Error: {error}")
```

### Unexpected Fallbacks
Monitor fallback rate:
```python
stats = evaluator.get_vm_stats()
fallback_rate = stats['vm_fallbacks'] / max(1, stats['vm_fallbacks'] + stats['vm_executions'])
if fallback_rate > 0.5:
    print("High fallback rate - may have unsupported features")
```

## When VM is Used

✅ **Automatically Used For:**
- While loops
- For-each loops
- Programs with >10 statements
- Functions with >5 statements
- Math-heavy computations

❌ **Not Used For:**
- Very simple expressions (< 3 operations)
- Features not yet supported in bytecode
- When `use_vm=False` is specified
- When compilation fails (falls back automatically)

## API Summary

### evaluate()
```python
def evaluate(program, env, debug_mode=False, use_vm=True)
```

### Evaluator class
```python
class Evaluator:
    def __init__(self, trusted: bool = False, use_vm: bool = True)
    def eval_with_vm_support(self, node, env, stack_trace=None, debug_mode=False)
    def get_vm_stats(self) -> dict
```

### EvaluatorBytecodeCompiler
```python
class EvaluatorBytecodeCompiler:
    def compile(self, node, optimize: bool = True) -> Optional[Bytecode]
    def can_compile(self, node) -> bool
```

### VM
```python
class VM:
    def __init__(self, builtins: Dict = None, env: Dict = None, parent_env: Dict = None)
    def execute(self, code, debug: bool = False)
```

### Bytecode
```python
class Bytecode:
    def add_instruction(self, opcode: str, operand: Any = None) -> int
    def add_constant(self, value: Any) -> int
    def disassemble(self) -> str
```

## Examples

### Example 1: Simple Arithmetic
```python
code = "let x = 10 + 5; x * 2"
# VM automatically used for arithmetic
```

### Example 2: Loop Optimization
```python
code = """
let sum = 0;
let i = 0;
while (i < 100) {
    sum = sum + i;
    i = i + 1;
}
sum
"""
# VM used for efficient loop execution
```

### Example 3: Function Calls
```python
code = """
action calculate(a, b) {
    return a * b + 10;
}
calculate(5, 3)
"""
# VM can compile and execute functions
```

## Testing

Run tests:
```bash
python test_vm_integration.py
```

Check specific functionality:
```bash
python -c "from zexus.vm.bytecode import Bytecode; print(Bytecode())"
```

## Resources

- Full documentation: [VM_INTEGRATION_SUMMARY.md](VM_INTEGRATION_SUMMARY.md)
- Source code: [src/zexus/vm/](src/zexus/vm/), [src/zexus/evaluator/](src/zexus/evaluator/)
- Tests: [test_vm_integration.py](test_vm_integration.py)
