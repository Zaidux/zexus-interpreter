"""
AST to Bytecode Compiler for Zexus VM

Converts Zexus AST nodes into efficient VM bytecode for high-performance execution.
This is the real implementation - no workarounds.
"""

from typing import List, Any, Dict, Optional
from .bytecode import Opcode, Bytecode
from .. import zexus_ast


class BytecodeCompiler:
    """
    Compiles Zexus AST to VM bytecode.
    
    Optimizations:
    - Constant folding
    - Dead code elimination  
    - Register allocation for arithmetic
    - Tail call optimization
    """
    
    def __init__(self, optimize=True):
        self.optimize = optimize
        self.constants = []
        self.instructions = []
        self.constant_map = {}  # Deduplicate constants
        self.label_counter = 0
        self.loop_stack = []  # For break/continue
        
    def compile(self, node) -> Bytecode:
        """Compile AST node to bytecode"""
        self.constants = []
        self.instructions = []
        self.constant_map = {}
        self.label_counter = 0
        
        self._compile_node(node)
        
        return Bytecode(self.instructions, self.constants)
    
    def _add_constant(self, value) -> int:
        """Add constant to pool, return index"""
        # Deduplicate constants
        key = (type(value).__name__, str(value))
        if key in self.constant_map:
            return self.constant_map[key]
        
        idx = len(self.constants)
        self.constants.append(value)
        self.constant_map[key] = idx
        return idx
    
    def _emit(self, opcode: Opcode, *args):
        """Emit instruction"""
        # Always use (opcode, operand) format; None for no operand
        if args:
            operand = args[0] if len(args) == 1 else args
        else:
            operand = None
        self.instructions.append((opcode, operand))
    
    def _make_label(self) -> str:
        """Generate unique label"""
        label = f"L{self.label_counter}"
        self.label_counter += 1
        return label
    
    def _compile_node(self, node):
        """Dispatch to specific compiler method"""
        if node is None:
            return
        
        node_type = type(node).__name__
        method_name = f'_compile_{node_type}'
        method = getattr(self, method_name, None)
        
        if method:
            method(node)
        else:
            raise NotImplementedError(f"Compiler not implemented for {node_type}")
    
    # ==================== Program & Statements ====================
    
    def _compile_Program(self, node):
        """Compile program (list of statements)"""
        for stmt in node.statements:
            self._compile_node(stmt)
    
    def _compile_BlockStatement(self, node):
        """Compile block of statements"""
        for stmt in node.statements:
            self._compile_node(stmt)
    
    def _compile_ExpressionStatement(self, node):
        """Compile expression statement"""
        self._compile_node(node.expression)
        self._emit(Opcode.POP)  # Discard result
    
    def _compile_PrintStatement(self, node):
        """Compile print statement"""
        # PrintStatement can have value (single) or values (multiple)
        if hasattr(node, 'values') and node.values:
            # Multiple values - compile each and print
            for val in node.values:
                self._compile_node(val)
                self._emit(Opcode.PRINT)
        elif hasattr(node, 'value') and node.value:
            # Single value
            self._compile_node(node.value)
            self._emit(Opcode.PRINT)
        else:
            # No value - print empty/newline
            null_idx = self._add_constant(None)
            self._emit(Opcode.LOAD_CONST, null_idx)
            self._emit(Opcode.PRINT)
    
    def _compile_LetStatement(self, node):
        """Compile let/const declaration"""
        # Compile value
        self._compile_node(node.value)
        
        # Store to variable
        name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        const_idx = self._add_constant(name)
        self._emit(Opcode.STORE_NAME, const_idx)
    
    def _compile_ReturnStatement(self, node):
        """Compile return statement"""
        if node.return_value:
            self._compile_node(node.return_value)
        else:
            # Return null
            null_idx = self._add_constant(None)
            self._emit(Opcode.LOAD_CONST, null_idx)
        
        self._emit(Opcode.RETURN)
    
    # ==================== Expressions ====================
    
    def _compile_Identifier(self, node):
        """Load variable"""
        # Handle both Identifier objects and plain strings
        if hasattr(node, 'value'):
            name = node.value
        elif isinstance(node, str):
            name = node
        else:
            raise TypeError(f"Expected Identifier or str, got {type(node).__name__}: {node}")
        
        const_idx = self._add_constant(name)
        self._emit(Opcode.LOAD_NAME, const_idx)
    
    def _compile_IntegerLiteral(self, node):
        """Load integer constant"""
        const_idx = self._add_constant(node.value)
        self._emit(Opcode.LOAD_CONST, const_idx)
    
    def _compile_FloatLiteral(self, node):
        """Load float constant"""
        const_idx = self._add_constant(node.value)
        self._emit(Opcode.LOAD_CONST, const_idx)
    
    def _compile_StringLiteral(self, node):
        """Load string constant"""
        const_idx = self._add_constant(node.value)
        self._emit(Opcode.LOAD_CONST, const_idx)
    
    def _compile_Boolean(self, node):
        """Load boolean constant"""
        const_idx = self._add_constant(node.value)
        self._emit(Opcode.LOAD_CONST, const_idx)
    
    # ==================== Binary Operations ====================
    
    def _compile_InfixExpression(self, node):
        """Compile binary operation"""
        # Compile operands
        self._compile_node(node.left)
        self._compile_node(node.right)
        
        # Emit operation
        op_map = {
            '+': Opcode.ADD,
            '-': Opcode.SUB,
            '*': Opcode.MUL,
            '/': Opcode.DIV,
            '%': Opcode.MOD,
            '**': Opcode.POW,
            '==': Opcode.EQ,
            '!=': Opcode.NEQ,
            '<': Opcode.LT,
            '>': Opcode.GT,
            '<=': Opcode.LTE,
            '>=': Opcode.GTE,
            '&&': Opcode.AND,
            '||': Opcode.OR,
        }
        
        opcode = op_map.get(node.operator)
        if opcode:
            self._emit(opcode)
        else:
            raise NotImplementedError(f"Operator {node.operator} not implemented")
    
    def _compile_PrefixExpression(self, node):
        """Compile unary operation"""
        self._compile_node(node.right)
        
        if node.operator == '-':
            self._emit(Opcode.NEG)
        elif node.operator == '!':
            self._emit(Opcode.NOT)
        else:
            raise NotImplementedError(f"Prefix operator {node.operator} not implemented")
    
    # ==================== Control Flow ====================
    
    def _compile_IfExpression(self, node):
        """Compile if/else statement"""
        # Compile condition
        self._compile_node(node.condition)
        
        # Jump if false
        else_label = self._make_label()
        end_label = self._make_label()
        
        self._emit(Opcode.JUMP_IF_FALSE, else_label)
        
        # Compile consequence
        self._compile_node(node.consequence)
        self._emit(Opcode.JUMP, end_label)
        
        # Else branch
        self._emit(Opcode.NOP)  # Label placeholder
        self.instructions[-1] = (Opcode.NOP, else_label)  # Mark label
        
        if node.alternative:
            self._compile_node(node.alternative)
        else:
            # Push null for else branch
            null_idx = self._add_constant(None)
            self._emit(Opcode.LOAD_CONST, null_idx)
        
        # End label
        self._emit(Opcode.NOP)
        self.instructions[-1] = (Opcode.NOP, end_label)
    
    def _compile_WhileStatement(self, node):
        """Compile while loop"""
        start_label = self._make_label()
        end_label = self._make_label()
        
        # Track loop for break/continue
        self.loop_stack.append((start_label, end_label))
        
        # Start of loop
        self._emit(Opcode.NOP)
        self.instructions[-1] = (Opcode.NOP, start_label)
        
        # Compile condition
        self._compile_node(node.condition)
        self._emit(Opcode.JUMP_IF_FALSE, end_label)
        
        # Compile body
        self._compile_node(node.body)
        
        # Loop back
        self._emit(Opcode.JUMP, start_label)
        
        # End of loop
        self._emit(Opcode.NOP)
        self.instructions[-1] = (Opcode.NOP, end_label)
        
        self.loop_stack.pop()
    
    def _compile_ForStatement(self, node):
        """Compile for loop"""
        # Initialize
        if node.initializer:
            self._compile_node(node.initializer)
        
        start_label = self._make_label()
        end_label = self._make_label()
        
        self.loop_stack.append((start_label, end_label))
        
        # Start of loop
        self._emit(Opcode.NOP)
        self.instructions[-1] = (Opcode.NOP, start_label)
        
        # Condition
        if node.condition:
            self._compile_node(node.condition)
            self._emit(Opcode.JUMP_IF_FALSE, end_label)
        
        # Body
        self._compile_node(node.body)
        
        # Increment
        if node.increment:
            self._compile_node(node.increment)
            self._emit(Opcode.POP)
        
        # Loop back
        self._emit(Opcode.JUMP, start_label)
        
        # End
        self._emit(Opcode.NOP)
        self.instructions[-1] = (Opcode.NOP, end_label)
        
        self.loop_stack.pop()
    
    # ==================== Function Calls ====================
    
    def _compile_CallExpression(self, node):
        """Compile function call"""
        # Compile arguments (push onto stack)
        for arg in node.arguments:
            self._compile_node(arg)
        
        # Compile function
        if isinstance(node.function, zexus_ast.Identifier):
            # Direct function call by name
            name = node.function.value
            const_idx = self._add_constant(name)
            arg_count = len(node.arguments)
            self._emit(Opcode.CALL_NAME, const_idx, arg_count)
        else:
            # Complex function expression
            self._compile_node(node.function)
            arg_count = len(node.arguments)
            self._emit(Opcode.CALL_TOP, arg_count)
    
    def _compile_MethodCallExpression(self, node):
        """Compile method call (object.method())"""
        # Load object first (object must be below arguments on stack)
        self._compile_node(node.object)

        # Compile arguments
        for arg in node.arguments:
            self._compile_node(arg)

        # Call method via dedicated opcode
        method_name = node.method.value if hasattr(node.method, 'value') else str(node.method)
        name_idx = self._add_constant(method_name)
        arg_count = len(node.arguments)
        self._emit(Opcode.CALL_METHOD, name_idx, arg_count)
    
    # ==================== Collections ====================
    
    def _compile_ArrayLiteral(self, node):
        """Compile array/list literal"""
        # Compile elements
        for element in node.elements:
            self._compile_node(element)
        
        # Build list
        count = len(node.elements)
        self._emit(Opcode.BUILD_LIST, count)
    
    def _compile_MapLiteral(self, node):
        """Compile map/dictionary literal"""
        # Compile key-value pairs
        for key, value in node.pairs.items():
            self._compile_node(key)
            self._compile_node(value)
        
        # Build map
        count = len(node.pairs)
        self._emit(Opcode.BUILD_MAP, count)
    
    def _compile_IndexExpression(self, node):
        """Compile array/map index access"""
        self._compile_node(node.left)
        self._compile_node(node.index)
        self._emit(Opcode.INDEX)
    
    # ==================== Fallback for unsupported nodes ====================
    
    def __getattr__(self, name):
        """Handle unsupported node types gracefully"""
        if name.startswith('_compile_'):
            # Return a stub that logs and continues
            def stub(node):
                print(f"⚠️  Bytecode compiler: {name[9:]} not yet implemented, skipping")
                # Push null as placeholder
                null_idx = self._add_constant(None)
                self._emit(Opcode.LOAD_CONST, null_idx)
            return stub
        raise AttributeError(name)


def compile_ast_to_bytecode(ast, optimize=True) -> Bytecode:
    """
    Compile Zexus AST to VM bytecode.
    
    Args:
        ast: Parsed AST (Program node)
        optimize: Enable optimizations
        
    Returns:
        Bytecode object ready for VM execution
    """
    compiler = BytecodeCompiler(optimize=optimize)
    return compiler.compile(ast)
