"""
Bytecode Compiler for Evaluator

This module allows the evaluator to compile AST nodes to bytecode
for VM execution when performance is critical.
"""
from typing import Dict, List, Optional
from .. import zexus_ast
from ..vm.bytecode import Bytecode, BytecodeBuilder


class EvaluatorBytecodeCompiler:
    """
    Compiles Zexus AST nodes to bytecode for VM execution.
    Designed to work seamlessly with the evaluator's execution model.
    """
    
    def __init__(self):
        self.builder: Optional[BytecodeBuilder] = None
        self.errors: List[str] = []
        self._loop_stack: List[Dict[str, str]] = []  # Stack of loop labels
    
    def compile(self, node, optimize: bool = True) -> Optional[Bytecode]:
        """
        Compile an AST node to bytecode.
        
        Args:
            node: AST node to compile
            optimize: Whether to apply optimizations
            
        Returns:
            Bytecode object or None if compilation failed
        """
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
        self.builder.emit_store(node.name.value)
    
    def _compile_ConstStatement(self, node: zexus_ast.ConstStatement):
        """Compile const statement"""
        # Similar to let for now
        self._compile_node(node.value)
        self.builder.emit_store(node.name.value)
    
    def _compile_ReturnStatement(self, node: zexus_ast.ReturnStatement):
        """Compile return statement"""
        if node.return_value:
            self._compile_node(node.return_value)
        else:
            self.builder.emit_constant(None)
        self.builder.emit("RETURN")
    
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
        # For now, mark as not supported in bytecode
        # This requires more complex runtime support
        self.errors.append("ForEach not yet supported in bytecode compilation")
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
        name_idx = self.builder.bytecode.add_constant(node.name.value)
        self.builder.emit("STORE_FUNC", (name_idx, func_const_idx))
    
    # === Expression Compilation ===
    
    def _compile_Identifier(self, node: zexus_ast.Identifier):
        """Compile identifier (variable load)"""
        self.builder.emit_load(node.value)
    
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
        # Compile arguments first
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
        # Store to target
        if isinstance(node.target, zexus_ast.Identifier):
            self.builder.emit("DUP")  # Keep value on stack
            self.builder.emit_store(node.target.value)
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
    
    # === Optimization ===
    
    def _optimize(self, bytecode: Bytecode) -> Bytecode:
        """
        Apply peephole optimizations to bytecode.
        
        Optimizations:
        - Remove unnecessary POP instructions
        - Constant folding
        - Dead code elimination
        """
        # For now, return as-is
        # Future: implement optimization passes
        return bytecode
    
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
            'ReturnStatement', 'IfStatement', 'WhileStatement', 'BlockStatement',
            'ActionStatement', 'Identifier', 'IntegerLiteral', 'FloatLiteral',
            'StringLiteral', 'Boolean', 'ListLiteral', 'MapLiteral',
            'InfixExpression', 'PrefixExpression', 'CallExpression',
            'AwaitExpression', 'AssignmentExpression', 'IndexExpression'
        }
        
        return node_type in supported


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
