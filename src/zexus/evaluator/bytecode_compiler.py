"""
Bytecode Compiler for Evaluator

This module allows the evaluator to compile AST nodes to bytecode
for VM execution when performance is critical.
"""
import os
from typing import Dict, List, Optional
from .. import zexus_ast
from ..vm.bytecode import Bytecode, BytecodeBuilder

# Try to import cache (optional dependency)
try:
    from ..vm.cache import BytecodeCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False


class EvaluatorBytecodeCompiler:
    """
    Compiles Zexus AST nodes to bytecode for VM execution.
    Designed to work seamlessly with the evaluator's execution model.
    
    Features:
    - AST to bytecode compilation
    - Bytecode caching for repeated code
    - Optimization support
    - Error tracking
    """
    
    def __init__(self, use_cache: bool = True, cache_size: int = 1000):
        """
        Initialize bytecode compiler
        
        Args:
            use_cache: Enable bytecode caching
            cache_size: Maximum cache entries
        """
        self.builder: Optional[BytecodeBuilder] = None
        self.errors: List[str] = []
        self._loop_stack: List[Dict[str, str]] = []  # Stack of loop labels
        
        # Bytecode cache
        self.cache: Optional[BytecodeCache] = None
        if use_cache and CACHE_AVAILABLE:
            self.cache = BytecodeCache(
                max_size=cache_size,
                max_memory_mb=50,
                persistent=False,
                debug=False
            )
    
    def compile(self, node, optimize: bool = True, use_cache: bool = True) -> Optional[Bytecode]:
        """
        Compile an AST node to bytecode.
        
        Process:
        1. Check cache for existing bytecode
        2. If cache miss, compile AST to bytecode
        3. Apply optimizations if requested
        4. Store in cache for future use
        
        Args:
            node: AST node to compile
            optimize: Whether to apply optimizations
            use_cache: Whether to use cache (default True)
            
        Returns:
            Bytecode object or None if compilation failed
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(node)
            if cached:
                return cached
        
        # Cache miss - compile
        self.builder = BytecodeBuilder()
        self.errors = []
        
        try:
            self._compile_node(node)
            
            if self.errors:
                return None
            
            bytecode = self.builder.build()
            
            if optimize:
                bytecode = self._optimize(bytecode)
            
            bytecode.set_metadata('created_by', 'evaluator')
            
            # Store in cache
            if use_cache and self.cache:
                self.cache.put(node, bytecode)
            
            return bytecode
            
        except Exception as e:
            self.errors.append(f"Compilation error: {e}")
            return None
    
    def _compile_node(self, node):
        """Dispatch to appropriate compilation method based on node type"""
        if self.builder is None:
            self.errors.append("Bytecode builder not initialized")
            return
        
        if node is None:
            self.builder.emit_constant(None)
            return
        
        node_type = type(node).__name__
        
        # Dispatch to appropriate handler
        method_name = f'_compile_{node_type}'
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            method(node)
        else:
            # Unsupported node type - emit error
            error_msg = f"Unsupported node type for bytecode: {node_type}"
            self.errors.append(error_msg)
            self.builder.emit_constant(None)
    
    # === Statement Compilation ===
    
    def _compile_Program(self, node: zexus_ast.Program):
        """Compile a program (top-level)"""
        for stmt in node.statements:
            self._compile_node(stmt)
        # Ensure we return something
        if not node.statements:
            self.builder.emit_constant(None)
    
    def _compile_ExpressionStatement(self, node: zexus_ast.ExpressionStatement):
        """Compile expression statement"""
        self._compile_node(node.expression)
        # Pop result unless it's the last statement
        # self.builder.emit("POP")
    
    def _compile_LetStatement(self, node: zexus_ast.LetStatement):
        """Compile let statement"""
        # Compile the value expression
        self._compile_node(node.value)
        # Store it
        name = str(node.name.value).strip()
        self.builder.emit_store(name)
    
    def _compile_ConstStatement(self, node: zexus_ast.ConstStatement):
        """Compile const statement"""
        # Similar to let for now
        self._compile_node(node.value)
        name = str(node.name.value).strip()
        self.builder.emit_store(name)
    
    def _compile_ReturnStatement(self, node: zexus_ast.ReturnStatement):
        """Compile return statement"""
        if node.return_value:
            self._compile_node(node.return_value)
        else:
            self.builder.emit_constant(None)
        self.builder.emit("RETURN")
    
    def _compile_ContinueStatement(self, node: zexus_ast.ContinueStatement):
        """Compile continue statement - enables error recovery mode"""
        # Emit a special instruction or constant to signal continue mode
        # For now, emit a CONTINUE instruction that the VM can handle
        self.builder.emit("CONTINUE")
    
    def _compile_IfStatement(self, node: zexus_ast.IfStatement):
        """Compile if statement with elif and else branches"""
        else_label = f"else_{id(node)}"
        end_label = f"end_if_{id(node)}"
        
        # Compile condition
        self._compile_node(node.condition)
        self.builder.emit_jump_if_false(else_label)
        
        # Compile consequence
        self._compile_node(node.consequence)
        self.builder.emit_jump(end_label)
        
        # Compile alternative (elif/else)
        self.builder.mark_label(else_label)
        if node.alternative:
            self._compile_node(node.alternative)
        else:
            self.builder.emit_constant(None)
        
        self.builder.mark_label(end_label)
    
    def _compile_WhileStatement(self, node: zexus_ast.WhileStatement):
        """Compile while loop"""
        start_label = f"while_start_{id(node)}"
        end_label = f"while_end_{id(node)}"
        
        # Track loop labels for break/continue
        self._loop_stack.append({'start': start_label, 'end': end_label})
        
        # Loop start
        self.builder.mark_label(start_label)
        
        # Condition
        self._compile_node(node.condition)
        self.builder.emit_jump_if_false(end_label)
        
        # Body
        self._compile_node(node.body)
        
        # Jump back to start
        self.builder.emit_jump(start_label)
        
        # Loop end
        self.builder.mark_label(end_label)
        self._loop_stack.pop()

        # While statements don't return values by default
        self.builder.emit_constant(None)
    
    def _compile_ForEachStatement(self, node: zexus_ast.ForEachStatement):
        """Compile for-each loop"""
        start_label = f"foreach_start_{id(node)}"
        end_label = f"foreach_end_{id(node)}"
        
        # Track loop labels for break/continue
        self._loop_stack.append({'start': start_label, 'end': end_label})
        
        # Compile the iterable
        self._compile_node(node.iterable)
        
        # Get iterator (for now, assume it's already iterable)
        # Store iterator in a temp variable
        iter_var = f"_iter_{id(node)}"
        self.builder.emit_store(iter_var)
        
        # Get length/count for iteration
        self.builder.emit_load(iter_var)
        self.builder.emit("GET_LENGTH")  # Should push length onto stack
        index_var = f"_index_{id(node)}"
        self.builder.emit_constant(0)
        self.builder.emit_store(index_var)
        
        # Loop start
        self.builder.mark_label(start_label)
        
        # Check if index < length
        self.builder.emit_load(index_var)
        self.builder.emit_load(iter_var)
        self.builder.emit("GET_LENGTH")
        self.builder.emit("LT")  # index < length
        self.builder.emit_jump_if_false(end_label)
        
        # Get current element
        self.builder.emit_load(iter_var)
        self.builder.emit_load(index_var)
        self.builder.emit("INDEX")  # Get element at index
        
        # Store in loop variable
        item_name = node.item.value if hasattr(node.item, 'value') else str(node.item)
        self.builder.emit_store(item_name)
        
        # Compile body
        self._compile_node(node.body)
        
        # Increment index
        self.builder.emit_load(index_var)
        self.builder.emit_constant(1)
        self.builder.emit("ADD")
        self.builder.emit_store(index_var)
        
        # Jump back to start
        self.builder.emit_jump(start_label)
        
        # Loop end
        self.builder.mark_label(end_label)
        self._loop_stack.pop()
        
        # For-each doesn't return a value
        self.builder.emit_constant(None)
    
    def _compile_BlockStatement(self, node: zexus_ast.BlockStatement):
        """Compile block statement"""
        if not node.statements:
            self.builder.emit_constant(None)
            return
        
        for i, stmt in enumerate(node.statements):
            self._compile_node(stmt)
            # Keep the last value on stack, pop others
            if i < len(node.statements) - 1:
                # Check if stmt has return - if so, don't pop
                if type(stmt).__name__ != 'ReturnStatement':
                    # Only pop if not a declaration
                    if type(stmt).__name__ not in ['LetStatement', 'ConstStatement', 'ActionStatement']:
                        pass  # Keep value for now
    
    def _compile_ActionStatement(self, node: zexus_ast.ActionStatement):
        """Compile action definition"""
        # Create nested bytecode for function body
        inner_compiler = EvaluatorBytecodeCompiler()
        func_bytecode = inner_compiler.compile(node.body, optimize=False)
        
        if inner_compiler.errors:
            self.errors.extend(inner_compiler.errors)
            return
        
        # Build function descriptor
        params = [p.value for p in node.parameters] if hasattr(node, 'parameters') else []
        func_desc = {
            "bytecode": func_bytecode,
            "params": params,
            "is_async": getattr(node, "is_async", False)
        }
        
        # Store function descriptor
        func_const_idx = self.builder.bytecode.add_constant(func_desc)
        name_idx = self.builder.bytecode.add_constant(str(node.name.value).strip())
        self.builder.emit("STORE_FUNC", (name_idx, func_const_idx))
    
    def _compile_FunctionStatement(self, node: zexus_ast.FunctionStatement):
        """Compile function definition (same as action)"""
        # Reuse action compilation logic
        self._compile_ActionStatement(node)
    
    def _compile_PrintStatement(self, node: zexus_ast.PrintStatement):
        """Compile print statement"""
        # Compile the value to print
        if hasattr(node, 'value') and node.value:
            self._compile_node(node.value)
        else:
            # Empty print
            self.builder.emit_constant("")
        
        # Emit PRINT opcode
        self.builder.emit("PRINT")
    
    # === Blockchain Statement Compilation ===
    
    def _compile_TxStatement(self, node):
        """Compile TX statement (transaction block)"""
        # Emit TX_BEGIN
        self.builder.emit("TX_BEGIN")
        
        # Compile transaction body
        if hasattr(node, 'body'):
            self._compile_node(node.body)
        
        # Emit TX_COMMIT (success path)
        self.builder.emit("TX_COMMIT")
    
    def _compile_RevertStatement(self, node):
        """Compile REVERT statement"""
        # If there's a reason/message, compile it
        if hasattr(node, 'reason') and node.reason:
            self._compile_node(node.reason)
            # Get the reason from stack
            reason_idx = self.builder.bytecode.add_constant("revert_reason")
            self.builder.emit("STORE_NAME", reason_idx)
        
        # Emit TX_REVERT
        self.builder.emit("TX_REVERT")
    
    def _compile_RequireStatement(self, node):
        """Compile REQUIRE statement - reverts if condition fails"""
        # Compile condition
        if hasattr(node, 'condition'):
            self._compile_node(node.condition)
        
        # Jump if true (condition passes)
        pass_label = f"require_pass_{id(node)}"
        self.builder.emit("JUMP_IF_TRUE", None)
        
        # Condition failed - revert
        if hasattr(node, 'message') and node.message:
            # Compile error message
            self._compile_node(node.message)
            msg_idx = self.builder.bytecode.add_constant("require_msg")
            self.builder.emit("STORE_NAME", msg_idx)
        
        self.builder.emit("TX_REVERT")
        
        # Pass label
        self.builder.mark_label(pass_label)
    
    def _compile_StateAccessExpression(self, node):
        """Compile STATE access expression"""
        # Check if it's a read or write
        if hasattr(node, 'key'):
            key = node.key if isinstance(node.key, str) else str(node.key)
            key_idx = self.builder.bytecode.add_constant(key)
            
            if hasattr(node, 'is_write') and node.is_write:
                # STATE_WRITE - expects value on stack
                self.builder.emit("STATE_WRITE", key_idx)
            else:
                # STATE_READ
                self.builder.emit("STATE_READ", key_idx)
    
    def _compile_LedgerAppendStatement(self, node):
        """Compile LEDGER append statement"""
        # Compile the entry to append
        if hasattr(node, 'entry'):
            self._compile_node(node.entry)
        
        # Emit LEDGER_APPEND
        self.builder.emit("LEDGER_APPEND")
    
    def _compile_GasChargeStatement(self, node):
        """Compile GAS charge statement"""
        # Get gas amount
        if hasattr(node, 'amount'):
            if isinstance(node.amount, int):
                amount = node.amount
            else:
                # Compile expression for gas amount
                self._compile_node(node.amount)
                # For now, assume constant
                amount = 1
        else:
            amount = 1
        
        # Emit GAS_CHARGE
        self.builder.emit("GAS_CHARGE", amount)
    
    # === Expression Compilation ===
    
    def _compile_Identifier(self, node: zexus_ast.Identifier):
        """Compile identifier (variable load)"""
        name = str(node.value).strip()
        profile_flag = os.environ.get("ZEXUS_VM_PROFILE_OPS")
        verbose_flag = os.environ.get("ZEXUS_VM_PROFILE_VERBOSE")
        if (
            profile_flag and profile_flag.lower() not in ("0", "false", "off")
            and verbose_flag and verbose_flag.lower() not in ("0", "false", "off")
        ):
            print(f"[VM DEBUG] compile identifier raw={node.value!r} normalized={name!r}")
        self.builder.emit_load(name)
    
    def _compile_IntegerLiteral(self, node: zexus_ast.IntegerLiteral):
        """Compile integer literal"""
        if self.builder is None:
            return
        self.builder.emit_constant(node.value)
    
    def _compile_FloatLiteral(self, node: zexus_ast.FloatLiteral):
        """Compile float literal"""
        if self.builder is None:
            return
        self.builder.emit_constant(node.value)
    
    def _compile_StringLiteral(self, node: zexus_ast.StringLiteral):
        """Compile string literal"""
        if self.builder is None:
            return
        self.builder.emit_constant(node.value)
    
    def _compile_Boolean(self, node: zexus_ast.Boolean):
        """Compile boolean literal"""
        if self.builder is None:
            return
        self.builder.emit_constant(node.value)
    
    def _compile_NullLiteral(self, node):
        """Compile null literal"""
        if self.builder is None:
            return
        self.builder.emit_constant(None)
    
    def _compile_ListLiteral(self, node: zexus_ast.ListLiteral):
        """Compile list literal"""
        # Push each element
        for element in node.elements:
            self._compile_node(element)
        # Build list from stack
        self.builder.emit("BUILD_LIST", len(node.elements))
    
    def _compile_MapLiteral(self, node: zexus_ast.MapLiteral):
        """Compile map/dictionary literal"""
        # Push key-value pairs
        for key_expr, value_expr in node.pairs:
            self._compile_node(key_expr)
            self._compile_node(value_expr)
        # Build map from stack
        self.builder.emit("BUILD_MAP", len(node.pairs))
    
    def _compile_InfixExpression(self, node: zexus_ast.InfixExpression):
        """Compile infix expression"""
        if self.builder is None:
            return
        # Compile operands
        self._compile_node(node.left)
        self._compile_node(node.right)
        
        # Emit operator
        op_map = {
            '+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV', '%': 'MOD',
            '**': 'POW',
            '==': 'EQ', '!=': 'NEQ', '<': 'LT', '>': 'GT',
            '<=': 'LTE', '>=': 'GTE',
            '&&': 'AND', '||': 'OR'
        }
        
        opcode = op_map.get(node.operator)
        if opcode:
            self.builder.emit(opcode)
        else:
            self.errors.append(f"Unsupported operator: {node.operator}")
            self.builder.emit_constant(None)
    
    def _compile_PrefixExpression(self, node: zexus_ast.PrefixExpression):
        """Compile prefix expression"""
        self._compile_node(node.right)
        
        if node.operator == '!':
            self.builder.emit("NOT")
        elif node.operator == '-':
            self.builder.emit("NEG")
        else:
            self.errors.append(f"Unsupported prefix operator: {node.operator}")
    
    def _compile_CallExpression(self, node: zexus_ast.CallExpression):
        """Compile function call"""
        # Check for blockchain-specific function calls that have dedicated opcodes
        if isinstance(node.function, zexus_ast.Identifier):
            func_name = node.function.value
            
            # HASH_BLOCK opcode - hash(block_data)
            if func_name == "hash" and len(node.arguments) == 1:
                self._compile_node(node.arguments[0])
                self.builder.emit("HASH_BLOCK")
                return
            
            # VERIFY_SIGNATURE opcode - verify_sig(signature, message, public_key)
            if func_name == "verify_sig" and len(node.arguments) == 3:
                # Compile arguments in stack order: signature, message, public_key
                for arg in node.arguments:
                    self._compile_node(arg)
                self.builder.emit("VERIFY_SIGNATURE")
                return
            
            # MERKLE_ROOT opcode - merkle_root([leaves...])
            if func_name == "merkle_root" and len(node.arguments) >= 1:
                # If argument is a list literal, compile elements individually
                if isinstance(node.arguments[0], zexus_ast.ListLiteral):
                    leaves = node.arguments[0].elements
                    for leaf in leaves:
                        self._compile_node(leaf)
                    self.builder.emit("MERKLE_ROOT", len(leaves))
                    return
        
        # Compile arguments first (standard call path)
        for arg in node.arguments:
            self._compile_node(arg)
        
        # Check if function is an identifier (direct call)
        if isinstance(node.function, zexus_ast.Identifier):
            self.builder.emit_call(node.function.value, len(node.arguments))
        else:
            # Function expression - evaluate and call
            self._compile_node(node.function)
            self.builder.emit("CALL_TOP", len(node.arguments))
    
    def _compile_AwaitExpression(self, node):
        """Compile await expression"""
        # Compile the inner expression
        self._compile_node(node.expression)
        # Emit await
        self.builder.emit("AWAIT")

    def _compile_MethodCallExpression(self, node: zexus_ast.MethodCallExpression):
        """Compile method call (object.method(...))"""
        if self.builder is None:
            return

        # Load object first so it sits below the arguments on the stack
        self._compile_node(node.object)

        # Compile arguments in order
        for arg in node.arguments:
            self._compile_node(arg)

        method_name = node.method.value if hasattr(node.method, 'value') else str(node.method)
        self.builder.emit_call_method(method_name, len(node.arguments))
    
    def _compile_SpawnExpression(self, node):
        """Compile spawn expression"""
        # Compile the inner expression (should be a call)
        self._compile_node(node.expression)
        # Emit spawn
        self.builder.emit("SPAWN")
    
    def _compile_AssignmentExpression(self, node: zexus_ast.AssignmentExpression):
        """Compile assignment expression"""
        # Compile the value
        self._compile_node(node.value)
        # Store to name
        if isinstance(node.name, zexus_ast.Identifier):
            self.builder.emit("DUP")  # Keep value on stack
            name = str(node.name.value).strip()
            self.builder.emit_store(name)
        else:
            self.errors.append(
                "Complex assignment targets not yet supported in bytecode")
    
    def _compile_IndexExpression(self, node):
        """Compile index expression"""
        # Compile the object
        self._compile_node(node.left)
        # Compile the index
        self._compile_node(node.index)
        # Emit index operation
        self.builder.emit("INDEX")
    
    def _compile_PropertyAccessExpression(self, node):
        """Compile property access (obj.property or obj[property])"""
        # Compile the object
        self._compile_node(node.object)
        
        # Check if computed (obj[expr]) or literal (obj.prop)
        if hasattr(node, 'computed') and node.computed:
            # Computed property - evaluate the expression
            self._compile_node(node.property)
            self.builder.emit("INDEX")
        else:
            # Literal property - emit as constant string and use GET_ATTR
            if hasattr(node.property, 'value'):
                prop_name = node.property.value
            else:
                # Fallback - evaluate it
                self._compile_node(node.property)
                self.builder.emit("INDEX")
                return
            
            # Emit property name as constant
            self.builder.emit_constant(prop_name)
            self.builder.emit("GET_ATTR")
    
    def _compile_LambdaExpression(self, node):
        """Compile lambda/anonymous function"""
        # Create nested bytecode for lambda body
        inner_compiler = EvaluatorBytecodeCompiler()
        func_bytecode = inner_compiler.compile(node.body, optimize=False)
        
        if inner_compiler.errors:
            self.errors.extend(inner_compiler.errors)
            self.builder.emit_constant(None)
            return
        
        # Build lambda descriptor
        params = [p.value if hasattr(p, 'value') else str(p) for p in node.parameters] if hasattr(node, 'parameters') else []
        lambda_desc = {
            "bytecode": func_bytecode,
            "params": params,
            "is_lambda": True
        }
        
        # Push lambda descriptor as constant
        self.builder.emit_constant(lambda_desc)
        self.builder.emit("BUILD_LAMBDA")

    def _compile_FindExpression(self, node: zexus_ast.FindExpression):
        """Compile find keyword expression by deferring to evaluator helper"""
        if self.builder is None:
            return
        # Store AST node as constant so VM builtin can reuse evaluator implementation
        self.builder.emit_constant(node)
        self.builder.emit_call("__keyword_find__", 1)

    def _compile_LoadExpression(self, node: zexus_ast.LoadExpression):
        """Compile load keyword expression by deferring to evaluator helper"""
        if self.builder is None:
            return
        self.builder.emit_constant(node)
        self.builder.emit_call("__keyword_load__", 1)
    
    # === Optimization ===
    
    def _optimize(self, bytecode: Bytecode) -> Bytecode:
        """
        Apply peephole optimizations to bytecode.
        
        Optimizations:
        - Remove unnecessary POP instructions
        - Constant folding (compile-time evaluation)
        - Dead code elimination  
        - Algebraic simplifications
        - Redundant operation removal
        """
        from zexus.vm.bytecode import Bytecode, Opcode
        
        instructions = bytecode.instructions
        if not instructions:
            return bytecode
        
        # Multiple optimization passes for better results
        optimized = instructions
        optimized = self._constant_folding(optimized, bytecode.constants)
        optimized = self._dead_code_elimination(optimized)
        optimized = self._peephole_patterns(optimized)
        optimized = self._remove_redundant_jumps(optimized)
        
        # Create new bytecode with optimized instructions
        new_bytecode = Bytecode()
        new_bytecode.instructions = optimized
        new_bytecode.constants = bytecode.constants.copy()
        new_bytecode.names = bytecode.names.copy()
        new_bytecode.labels = bytecode.labels.copy()
        
        return new_bytecode
    
    def _constant_folding(self, instructions, constants):
        """
        Fold constant expressions at compile time.
        Example: LOAD_CONST 2, LOAD_CONST 3, ADD → LOAD_CONST 5
        """
        optimized = []
        i = 0
        
        while i < len(instructions):
            inst = instructions[i]
            
            # Look for pattern: LOAD_CONST, LOAD_CONST, <BINARY_OP>
            if (i + 2 < len(instructions) and
                inst[0] == 'LOAD_CONST' and
                instructions[i+1][0] == 'LOAD_CONST'):
                
                op = instructions[i+2][0]
                
                # Get constant values
                const1 = constants[inst[1]] if inst[1] < len(constants) else None
                const2 = constants[instructions[i+1][1]] if instructions[i+1][1] < len(constants) else None
                
                if const1 is not None and const2 is not None:
                    result = self._try_constant_operation(const1, const2, op)
                    
                    if result is not None:
                        # Add result as new constant
                        const_idx = len(constants)
                        constants.append(result)
                        
                        # Replace three instructions with one
                        optimized.append(('LOAD_CONST', const_idx))
                        i += 3
                        continue
            
            # No optimization possible, keep instruction
            optimized.append(inst)
            i += 1
        
        return optimized
    
    def _try_constant_operation(self, val1, val2, op):
        """Try to evaluate constant operation, return None if not possible"""
        try:
            # Import Zexus object types
            from zexus.objects import Integer, Float, String, Boolean as BooleanObj
            
            # Extract Python values
            v1 = val1.value if hasattr(val1, 'value') else val1
            v2 = val2.value if hasattr(val2, 'value') else val2
            
            # Arithmetic operations
            if op == 'ADD':
                if isinstance(v1, str) or isinstance(v2, str):
                    return String(str(v1) + str(v2))
                return Integer(v1 + v2) if isinstance(v1, int) and isinstance(v2, int) else Float(v1 + v2)
            elif op == 'SUB':
                return Integer(v1 - v2) if isinstance(v1, int) and isinstance(v2, int) else Float(v1 - v2)
            elif op == 'MUL':
                return Integer(v1 * v2) if isinstance(v1, int) and isinstance(v2, int) else Float(v1 * v2)
            elif op == 'DIV':
                if v2 == 0:
                    return None  # Don't fold division by zero
                result = v1 / v2
                return Integer(int(result)) if result == int(result) else Float(result)
            
            # Comparison operations
            elif op == 'EQ':
                return BooleanObj(v1 == v2)
            elif op == 'NE':
                return BooleanObj(v1 != v2)
            elif op == 'LT':
                return BooleanObj(v1 < v2)
            elif op == 'GT':
                return BooleanObj(v1 > v2)
            elif op == 'LTE':
                return BooleanObj(v1 <= v2)
            elif op == 'GTE':
                return BooleanObj(v1 >= v2)
            
            # Logical operations
            elif op == 'AND':
                return BooleanObj(bool(v1) and bool(v2))
            elif op == 'OR':
                return BooleanObj(bool(v1) or bool(v2))
            
        except:
            pass
        
        return None
    
    def _dead_code_elimination(self, instructions):
        """Remove unreachable code (code after RETURN, unconditional jumps to next instruction, etc.)"""
        optimized = []
        skip_until_label = False
        
        for i, inst in enumerate(instructions):
            # If we're skipping dead code, only stop at labels
            if skip_until_label:
                if inst[0] == 'LABEL':
                    skip_until_label = False
                else:
                    continue  # Skip this instruction
            
            # After a RETURN or unconditional JUMP, skip until next label
            if inst[0] == 'RETURN' or inst[0] == 'JUMP':
                optimized.append(inst)
                skip_until_label = True
                continue
            
            optimized.append(inst)
        
        return optimized
    
    def _peephole_patterns(self, instructions):
        """Match and optimize common instruction patterns"""
        optimized = []
        i = 0
        
        while i < len(instructions):
            inst = instructions[i]
            
            # Pattern: LOAD_CONST x, POP → (remove both)
            if (i + 1 < len(instructions) and
                inst[0] == 'LOAD_CONST' and
                instructions[i+1][0] == 'POP'):
                i += 2  # Skip both instructions
                continue
            
            # Pattern: LOAD_NAME x, STORE_NAME x → (remove both - noop)
            if (i + 1 < len(instructions) and
                inst[0] == 'LOAD_NAME' and
                instructions[i+1][0] == 'STORE_NAME' and
                inst[1] == instructions[i+1][1]):  # Same variable
                i += 2  # Skip both instructions
                continue
            
            # Pattern: DUP, POP → (remove both)
            if (i + 1 < len(instructions) and
                inst[0] == 'DUP' and
                instructions[i+1][0] == 'POP'):
                i += 2
                continue
            
            # Algebraic simplifications with LOAD_CONST
            # Pattern: LOAD_NAME x, LOAD_CONST 0, ADD → LOAD_NAME x (adding 0 is noop)
            if (i + 2 < len(instructions) and
                inst[0] == 'LOAD_NAME' and
                instructions[i+1][0] == 'LOAD_CONST' and
                instructions[i+2][0] == 'ADD'):
                
                const_val = instructions[i+1][1]
                # Check if constant is 0 (would need to look up in constants array, skip for now)
                # This is a more complex optimization
                pass
            
            # Pattern: LOAD_NAME x, LOAD_CONST 1, MUL → LOAD_NAME x (multiplying by 1 is noop)
            # Similar to above, needs constant value lookup
            
            # No optimization, keep instruction
            optimized.append(inst)
            i += 1
        
        return optimized
    
    def _remove_redundant_jumps(self, instructions):
        """Remove jumps to the immediately next instruction"""
        optimized = []
        
        for i, inst in enumerate(instructions):
            # Check if this is a JUMP instruction
            if inst[0] in ('JUMP', 'JUMP_IF_TRUE', 'JUMP_IF_FALSE'):
                # Check if it jumps to the next instruction
                target = inst[1]
                
                # Look ahead to see if next instruction is the target label
                if i + 1 < len(instructions):
                    next_inst = instructions[i+1]
                    if next_inst[0] == 'LABEL' and next_inst[1] == target:
                        # Skip this jump, it's redundant
                        continue
            
            optimized.append(inst)
        
        return optimized
    
    def can_compile(self, node) -> bool:
        """
        Check if a node can be compiled to bytecode.
        Some complex features may not be supported yet.
        """
        if node is None:
            return True
        
        node_type = type(node).__name__
        
        # List of supported node types
        supported = {
            'Program', 'ExpressionStatement', 'LetStatement', 'ConstStatement',
            'ReturnStatement', 'ContinueStatement', 'IfStatement', 'WhileStatement', 'ForEachStatement',
            'BlockStatement', 'ActionStatement', 'FunctionStatement', 'PrintStatement',
            'Identifier', 'IntegerLiteral', 'FloatLiteral',
            'StringLiteral', 'Boolean', 'ListLiteral', 'MapLiteral', 'NullLiteral',
            'InfixExpression', 'PrefixExpression', 'CallExpression',
            'AwaitExpression', 'SpawnExpression', 'AssignmentExpression', 'IndexExpression',
            'PropertyAccessExpression', 'LambdaExpression',
            'FindExpression', 'LoadExpression',
            # Blockchain nodes
            'TxStatement', 'RevertStatement', 'RequireStatement',
            'StateAccessExpression', 'LedgerAppendStatement', 'GasChargeStatement'
        }
        
        return node_type in supported
    
    # ==================== Cache Management ====================
    
    def get_cache_stats(self) -> Optional[Dict]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats or None if cache disabled
        """
        if self.cache:
            return self.cache.get_stats()
        return None
    
    def clear_cache(self):
        """Clear bytecode cache"""
        if self.cache:
            self.cache.clear()
    
    def invalidate_cache(self, node):
        """Invalidate cached bytecode for a node"""
        if self.cache:
            self.cache.invalidate(node)
    
    def reset_cache_stats(self):
        """Reset cache statistics"""
        if self.cache:
            self.cache.reset_stats()
    
    def cache_size(self) -> int:
        """Get current cache size"""
        if self.cache:
            return self.cache.size()
        return 0
    
    def cache_memory_usage(self) -> float:
        """Get cache memory usage in MB"""
        if self.cache:
            return self.cache.memory_usage_mb()
        return 0.0


def should_use_vm_for_node(node) -> bool:
    """
    Determine if a node should be compiled to bytecode and run in VM.
    
    Heuristics:
    - Large loops (optimization)
    - Recursive functions (tail call optimization potential)
    - Math-heavy computations
    - Multiple function calls in sequence
    """
    if node is None:
        return False
    
    node_type = type(node).__name__
    
    # Always use VM for loops
    if node_type in ['WhileStatement', 'ForEachStatement']:
        return True
    
    # Use VM for complex functions
    if node_type == 'ActionStatement':
        # Count statements in body
        if hasattr(node, 'body') and hasattr(node.body, 'statements'):
            if len(node.body.statements) > 5:
                return True
    
    # Use VM for programs with many statements
    if node_type == 'Program':
        if len(node.statements) > 10:
            return True
    
    return False
