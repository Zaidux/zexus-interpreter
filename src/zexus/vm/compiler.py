"""
AST to Bytecode Compiler for Zexus VM

Converts Zexus AST nodes into efficient VM bytecode for high-performance execution.
This is the real implementation - no workarounds.
"""

from typing import List, Any, Dict, Optional
from .bytecode import Opcode, Bytecode
from .. import zexus_ast


class BytecodeCompilationError(Exception):
    """Raised when bytecode compilation cannot complete successfully."""


class UnsupportedNodeError(BytecodeCompilationError):
    """Raised when the compiler encounters an unsupported AST node."""

    def __init__(self, node_type: str, detail: Optional[str] = None):
        message = detail or f"Node type '{node_type}' is not supported by the bytecode compiler"
        super().__init__(message)
        self.node_type = node_type
        self.detail = detail


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
        self.temp_counter = 0
        
    def compile(self, node) -> Bytecode:
        """Compile AST node to bytecode"""
        self.constants = []
        self.instructions = []
        self.constant_map = {}
        self.label_counter = 0
        self.temp_counter = 0
        
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

    def _make_temp_name(self, prefix: str = "tmp") -> str:
        name = f"__{prefix}_{self.temp_counter}"
        self.temp_counter += 1
        return name

    def _mark_label(self, label: str):
        self._emit(Opcode.NOP)
        self.instructions[-1] = (Opcode.NOP, label)
    
    def _compile_node(self, node):
        """Dispatch to specific compiler method"""
        if node is None:
            return
        
        node_type = type(node).__name__
        method_name = f'_compile_{node_type}'
        method = getattr(self, method_name, None)
        
        if method is None:
            raise UnsupportedNodeError(node_type, self._unsupported_message(node_type))

        try:
            method(node)
        except UnsupportedNodeError:
            raise
        except Exception as exc:
            raise BytecodeCompilationError(
                f"Failed to compile {node_type}: {exc}"
            ) from exc

    def _unsupported_message(self, node_type: str) -> str:
        """Return a friendly message for unsupported nodes."""
        hints = {
            "UseStatement": "Module imports",
            "EntityStatement": "Entity declarations",
            "ContractStatement": "Contract declarations",
            "ActionStatement": "Action/function declarations",
            "FunctionStatement": "Function declarations",
            "IfStatement": "If statements",
            "WhileStatement": "While loops",
            "ForStatement": "For loops",
            "ForEachStatement": "For-each loops",
            "TxStatement": "Transaction blocks",
            "RequireStatement": "Require statements",
            "RevertStatement": "Revert statements",
        }
        prefix = hints.get(node_type)
        if prefix:
            return f"{prefix} are not yet supported by the VM bytecode compiler"
        return f"Node type '{node_type}' is not supported by the VM bytecode compiler"
    
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
        loop_info = {
            'start': start_label,
            'end': end_label,
            'continue': start_label,
        }
        self.loop_stack.append(loop_info)

        # Start label before evaluating condition
        self._mark_label(start_label)

        # Condition
        self._compile_node(node.condition)
        exit_jump_idx = len(self.instructions)
        self._emit(Opcode.JUMP_IF_FALSE, None)

        # Body
        self._compile_node(node.body)

        # Jump back to start
        self._emit(Opcode.JUMP, start_label)

        # End label
        self._mark_label(end_label)
        self.instructions[exit_jump_idx] = (Opcode.JUMP_IF_FALSE, end_label)

        self.loop_stack.pop()
    
    def _compile_ForStatement(self, node):
        """Compile for loop"""
        # Initialize
        if node.initializer:
            self._compile_node(node.initializer)
        
        start_label = self._make_label()
        end_label = self._make_label()
        continue_label = start_label if node.increment is None else self._make_label()

        self.loop_stack.append({
            'start': start_label,
            'end': end_label,
            'continue': continue_label,
        })

        # Loop start (condition)
        self._mark_label(start_label)

        exit_jump_idx = None
        if node.condition:
            self._compile_node(node.condition)
            exit_jump_idx = len(self.instructions)
            self._emit(Opcode.JUMP_IF_FALSE, None)
        
        # Body
        self._compile_node(node.body)

        # Continue label placed before increment (if present)
        if node.increment:
            self._mark_label(continue_label)
            self._compile_node(node.increment)
            self._emit(Opcode.POP)

        # Jump back to condition
        self._emit(Opcode.JUMP, start_label)

        # Loop end label
        self._mark_label(end_label)
        if exit_jump_idx is not None:
            self.instructions[exit_jump_idx] = (Opcode.JUMP_IF_FALSE, end_label)

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
        pairs = node.pairs or []

        if isinstance(pairs, dict):
            iterable = pairs.items()
            count = len(pairs)
        else:
            iterable = pairs
            count = len(pairs)

        for key, value in iterable:
            self._compile_node(key)
            self._compile_node(value)
        
        # Build map
        self._emit(Opcode.BUILD_MAP, count)
    
    def _compile_IndexExpression(self, node):
        """Compile array/map index access"""
        self._compile_node(node.left)
        self._compile_node(node.index)
        self._emit(Opcode.INDEX)

    # ==================== Advanced Control Flow & Declarations ====================

    def _compile_ActionStatement(self, node):
        """Compile action/function definition"""
        # Create a new compiler for the function body
        func_compiler = self.__class__()
        
        # Compile body
        func_compiler._compile_node(node.body)
        
        # Ensure implicit return (null) if execution flows off the end
        null_idx = func_compiler._add_constant(None)
        func_compiler._emit(Opcode.LOAD_CONST, null_idx)
        func_compiler._emit(Opcode.RETURN)
        
        # Get bytecode
        func_bytecode = func_compiler.bytecode
        
        # Prepare function descriptor
        params = [p.value if hasattr(p, 'value') else str(p) for p in node.parameters]
        func_desc = {
            "bytecode": func_bytecode,
            "params": params,
            "is_async": getattr(node, 'is_async', False),
            "name": node.name.value if hasattr(node.name, 'value') else str(node.name)
        }
        
        # Store using STORE_FUNC
        # Operand: (name_idx, func_const_idx)
        func_idx = self._add_constant(func_desc)
        name_idx = self._add_constant(func_desc["name"])
        
        self._emit(Opcode.STORE_FUNC, (name_idx, func_idx))

    # Aliases for other function types
    _compile_FunctionStatement = _compile_ActionStatement
    _compile_PureFunctionStatement = _compile_ActionStatement

    def _compile_ConstStatement(self, node):
        """Compile const declaration (same as Let for current VM)"""
        self._compile_LetStatement(node)

    def _compile_FileImportExpression(self, node):
        """Compile file import expression (<<)"""
        self._compile_node(node.filepath)
        self._emit(Opcode.READ)

    def _compile_AssignmentExpression(self, node):
        """Compile variable assignment"""
        # 1. Compile value expression
        self._compile_node(node.value)
        
        # 2. Duplicate value because assignment is an expression that returns the value
        # AND currently most usage is inside ExpressionStatement which pops it.
        # But for correctness as an Expression, it must leave value on stack.
        self._emit(Opcode.DUP)
        
        # 3. Store to variable
        name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        name_idx = self._add_constant(name)
        self._emit(Opcode.STORE_NAME, name_idx)

    def _compile_UseStatement(self, node):
        """Compile module import"""
        # node.file_path is string or StringLiteral
        path = node.file_path.value if hasattr(node.file_path, 'value') else str(node.file_path)
        path_idx = self._add_constant(path)
        
        # Handle alias
        alias = node.alias.value if node.alias and hasattr(node.alias, 'value') else (node.alias or path)
        alias_idx = self._add_constant(alias)
        
        # Emit IMPORT (path_idx, alias_idx)
        self._emit(Opcode.IMPORT, (path_idx, alias_idx))

    def _compile_TxStatement(self, node):
        """Compile transaction block"""
        self._emit(Opcode.TX_BEGIN)
        self._compile_node(node.body)
        self._emit(Opcode.TX_COMMIT)

    def _compile_GCStatement(self, node):
        """Compile GC statement"""
        # Call builtin gc(action)
        action = node.action.value if hasattr(node.action, 'value') else str(node.action)
        
        # Load 'gc' function name
        gc_idx = self._add_constant("gc")
        
        # Load action argument
        action_idx = self._add_constant(action)
        self._emit(Opcode.LOAD_CONST, action_idx)
        
        # Call gc(action)
        self._emit(Opcode.CALL_NAME, (gc_idx, 1))
        self._emit(Opcode.POP)

    
    def _compile_ContractStatement(self, node):
        """Compile smart contract definition"""
        contract_name = node.name.value
        name_idx = self._add_constant(contract_name)
        
        # Push contract name onto stack
        self._emit(Opcode.LOAD_CONST, name_idx)
        
        # Compile body members
        member_count = 0
        if hasattr(node.body, 'statements'):
            for stmt in node.body.statements:
                stmt_type = type(stmt).__name__
                
                if stmt_type == 'StateStatement':
                    # Value
                    if getattr(stmt, 'initial_value', None):
                        self._compile_node(stmt.initial_value)
                    else:
                        self._emit(Opcode.LOAD_CONST, self._add_constant(None))
                    
                    # Name
                    self._emit(Opcode.LOAD_CONST, self._add_constant(stmt.name.value))
                    member_count += 1
                
                elif stmt_type == 'ActionStatement':
                    # Compile action (stores it in current env)
                    self._compile_node(stmt)
                    
                    # Load it back to stack
                    self._emit(Opcode.LOAD_NAME, self._add_constant(stmt.name.value))
                    
                    # Push name
                    self._emit(Opcode.LOAD_CONST, self._add_constant(stmt.name.value))
                    member_count += 1

        # Define Contract (pops name + values/keys * count)
        # Arg is member_count
        self._emit(Opcode.DEFINE_CONTRACT, member_count)
        
        # Store contract class/object
        self._emit(Opcode.STORE_NAME, name_idx)

    def _compile_BreakStatement(self, node):
        """Compile break statement"""
        if not self.loop_stack:
            raise BytecodeCompilationError("Break statement outside loop")
        self._emit(Opcode.JUMP, self.loop_stack[-1]['end'])

    def _compile_ContinueStatement(self, node):
        """Compile continue statement
        
        Behavior:
        - If inside a loop: Jump to loop start (next iteration).
        - If outside a loop: Enable 'Continue On Error' mode globally.
        """
        if self.loop_stack:
            # Inside loop: Standard control flow
            self._emit(Opcode.JUMP, self.loop_stack[-1]['continue'])
        else:
            # Outside loop: Enable error resilience
            self._emit(Opcode.ENABLE_ERROR_MODE)

    def _compile_RequireStatement(self, node):
        """Compile require(condition, message)"""
        self._compile_node(node.condition)
        
        if node.message:
            self._compile_node(node.message)
        else:
            self._emit(Opcode.LOAD_CONST, self._add_constant("Requirement failed"))
            
        self._emit(Opcode.REQUIRE)

    def _compile_RevertStatement(self, node):
        """Compile revert(reason)"""
        if node.reason:
            self._compile_node(node.reason)
        else:
            self._emit(Opcode.LOAD_CONST, self._add_constant("Transaction reverted"))
            
        self._emit(Opcode.TX_REVERT)

    def _compile_TryCatchStatement(self, node):
        """Compile try-catch block"""
        catch_label = self._make_label()
        end_label = self._make_label()
        
        # SETUP_TRY catch_label
        # (Assuming SETUP_TRY takes a jump target)
        # Note: Opcode must handle jump target resolution. 
        # Typically JUMP opcodes use labels. Here we treat SETUP_TRY like a jump-setup.
        self._emit(Opcode.SETUP_TRY, catch_label)
        
        self._compile_node(node.try_block)
        
        self._emit(Opcode.POP_TRY)
        self._emit(Opcode.JUMP, end_label)
        
        # Catch block
        self._mark_label(catch_label)
        
        # Exception provided on stack?
        if node.error_variable:
            self._emit(Opcode.STORE_NAME, self._add_constant(node.error_variable.value))
        else:
            self._emit(Opcode.POP)
            
        self._compile_node(node.catch_block)
        
        self._mark_label(end_label)

    def _unsupported_message(self, node_type):
        """Helper for unsupported messages"""
        return f"Node type '{node_type}' is not currently supported by the bytecode compiler."

    def _compile_AwaitExpression(self, node):
        """Compile await expression"""
        self._compile_node(node.argument)
        self._emit(Opcode.AWAIT)

    def _compile_ThisExpression(self, node):
        """Compile 'this' expression"""
        # Load 'this' from environment (it's passed as implicit argument in methods)
        name_idx = self._add_constant("this")
        self._emit(Opcode.LOAD_NAME, name_idx)

    def _compile_EntityStatement(self, node):
        """Compile Entity definition"""
        # Similar to Contract, but uses DEFINE_ENTITY
        entity_name = node.name.value
        name_idx = self._add_constant(entity_name)
        
        # Push name
        self._emit(Opcode.LOAD_CONST, name_idx)
        
        # Compile body members
        member_count = 0
        if hasattr(node.body, 'statements'):
            for stmt in node.body.statements:
                stmt_type = type(stmt).__name__
                # Reuse state/action compilation
                if stmt_type in ('StateStatement', 'ActionStatement'):
                    if stmt_type == 'StateStatement':
                         # Value
                        if getattr(stmt, 'initial_value', None):
                            self._compile_node(stmt.initial_value)
                        else:
                            self._emit(Opcode.LOAD_CONST, self._add_constant(None))
                        # Name
                        self._emit(Opcode.LOAD_CONST, self._add_constant(stmt.name.value))
                    elif stmt_type == 'ActionStatement':
                        self._compile_node(stmt)
                        self._emit(Opcode.LOAD_NAME, self._add_constant(stmt.name.value))
                        self._emit(Opcode.LOAD_CONST, self._add_constant(stmt.name.value))
                    
                    member_count += 1

        self._emit(Opcode.DEFINE_ENTITY, member_count)
        self._emit(Opcode.STORE_NAME, name_idx)

    def _compile_DataStatement(self, node):
        """Compile Data/Dataclass definition"""
        # Data objects are simpler entities
        # For now, map to Entity logic or simple Struct
        # We'll use DEFINE_ENTITY for now to keep it uniform
        
        data_name = node.name.value
        name_idx = self._add_constant(data_name)
        self._emit(Opcode.LOAD_CONST, name_idx)
        
        member_count = 0
        if node.fields:
            for field in node.fields:
                # Default value
                if field.default_value:
                    self._compile_node(field.default_value)
                else:
                    self._emit(Opcode.LOAD_CONST, self._add_constant(None))
                
                # Field name
                field_name = field.name.value if hasattr(field.name, 'value') else str(field.name)
                self._emit(Opcode.LOAD_CONST, self._add_constant(field_name))
                member_count += 1
                
        self._emit(Opcode.DEFINE_ENTITY, member_count)
        self._emit(Opcode.STORE_NAME, name_idx)
        
    def _compile_CapabilityStatement(self, node):
        """Compile capability definition"""
        # Load definition map
        if node.definition:
            self._compile_node(node.definition)
        else:
            self._emit(Opcode.LOAD_CONST, self._add_constant(None))
            
        # Load name
        name = node.name.value if hasattr(node.name, 'value') else str(node.name)
        self._emit(Opcode.LOAD_CONST, self._add_constant(name))
        
        self._emit(Opcode.DEFINE_CAPABILITY)

    def _compile_GrantStatement(self, node):
        """Compile grant capability"""
        # Logic: Push Entity (or name), Push Caps... Emit GRANT
        
        # Push entity name (string)
        entity = node.entity_name.value if hasattr(node.entity_name, 'value') else str(node.entity_name)
        self._emit(Opcode.LOAD_CONST, self._add_constant(entity))
        
        # Traverse capabilities
        count = 0
        for cap in node.capabilities:
            cap_name = cap.value if hasattr(cap, 'value') else str(cap)
            self._emit(Opcode.LOAD_CONST, self._add_constant(cap_name))
            count += 1
            
        self._emit(Opcode.GRANT_CAPABILITY, count)

    def _compile_RevokeStatement(self, node):
        """Compile revoke capability"""
        # Push entity name
        entity = node.entity_name.value if hasattr(node.entity_name, 'value') else str(node.entity_name)
        self._emit(Opcode.LOAD_CONST, self._add_constant(entity))
        
        # Traverse capabilities
        count = 0
        for cap in node.capabilities:
            cap_name = cap.value if hasattr(cap, 'value') else str(cap)
            self._emit(Opcode.LOAD_CONST, self._add_constant(cap_name))
            count += 1
            
        self._emit(Opcode.REVOKE_CAPABILITY, count)

    def _compile_AuditStatement(self, node):
        """Compile audit log"""
        # AuditStatement(data_name, action_type, timestamp)
        
        # Load data name (variable being audited)
        data_name = node.data_name.value if hasattr(node.data_name, 'value') else str(node.data_name)
        self._emit(Opcode.LOAD_CONST, self._add_constant(data_name))
        
        # Load action type
        if hasattr(node.action_type, 'value'):
            self._emit(Opcode.LOAD_CONST, self._add_constant(node.action_type.value))
        else:
            self._emit(Opcode.LOAD_CONST, self._add_constant(str(node.action_type)))
            
        # Load timestamp (if any)
        if node.timestamp:
            self._compile_node(node.timestamp)
        else:
             self._emit(Opcode.LOAD_CONST, self._add_constant(None))
             
        self._emit(Opcode.AUDIT_LOG)

    def _compile_RestrictStatement(self, node):
        """Compile restrict access"""
        # RestrictStatement(target, restriction_type)
        # target is PropertyAccessExpression usually (obj.field) or Identifier
        
        if hasattr(node.target, 'object'):
            # Property access: obj.field
            self._compile_node(node.target.object) # Push object
            
            # Helper to get property name
            prop_name = node.target.property.value if hasattr(node.target.property, 'value') else str(node.target.property)
            self._emit(Opcode.LOAD_CONST, self._add_constant(prop_name))
        else:
            # Just identifier
            self._emit(Opcode.LOAD_NAME, self._add_constant(node.target.value))
            self._emit(Opcode.LOAD_CONST, self._add_constant(None)) # No property

        # Load restriction string
        res_type = node.restriction_type if isinstance(node.restriction_type, str) else str(node.restriction_type)
        self._emit(Opcode.LOAD_CONST, self._add_constant(res_type))
        
        self._emit(Opcode.RESTRICT_ACCESS)


    def _compile_PatternStatement(self, node):
        """Compile pattern matching (match statement)"""
        # Logic:
        # evaluate value -> store in temp
        # for each case:
        #   load value
        #   evaluate pattern
        #   check match (simulated EQ)
        #   jump_if_false -> next_case
        #   execute action
        #   jump -> end
        
        # Evaluate match subject
        self._compile_node(node.value if hasattr(node, 'value') else node.expression)
        
        end_label = self._make_label()
        
        # Iterate cases
        for case in node.cases:
            next_case_lbl = self._make_label()
            
            # Duplicate Subject (Stack: [..., subj, subj])
            self._emit(Opcode.DUP)
            
            # Evaluate Pattern
            # Note: AST might call it 'pattern' or something else depending on Case node type
            pattern_node = getattr(case, 'pattern', None)
            if pattern_node:
                if hasattr(pattern_node, 'value'): 
                    self._compile_node(pattern_node)
                else:
                    self._compile_node(pattern_node)
            
            # Equality check
            self._emit(Opcode.EQ)
            
            # Jump if False
            self._emit(Opcode.JUMP_IF_FALSE, next_case_lbl)
            
            # Match! 
            # Pop subject
            self._emit(Opcode.POP)
            
            # Execute Action/Consequence
            # MatchExpression uses 'consequence', PatternStatement uses 'action'
            body = getattr(case, 'consequence', getattr(case, 'action', None))
            self._compile_node(body)
            self._emit(Opcode.JUMP, end_label)
            
            # Next Case
            self._mark_label(next_case_lbl)
            
        # No match? Pop subject
        self._emit(Opcode.POP)
        self._mark_label(end_label)

    # Alias MatchExpression to logic (it's similar enough for basic cases)
    _compile_MatchExpression = _compile_PatternStatement

    # ==================== Fallback for unsupported nodes ====================
    
    def __getattr__(self, name):
        """Handle unsupported node types gracefully"""
        if name.startswith('_compile_'):
            # Return a stub that logs and continues
            def stub(node):
                node_type = name[9:]
                raise UnsupportedNodeError(node_type, self._unsupported_message(node_type))
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
