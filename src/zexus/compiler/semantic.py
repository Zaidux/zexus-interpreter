"""
Semantic Analyzer for Enhanced Error Reporting
"""

class SemanticAnalyzer:
    def __init__(self):
        self.errors = []
        self.symbol_table = {}
        
    def analyze(self, program):
        """Perform semantic analysis on the AST"""
        self.errors = []
        self.symbol_table = {}
        
        for statement in program.statements:
            self.analyze_statement(statement)
            
        return self.errors
    
    def analyze_statement(self, stmt):
        from .zexus_ast import (
            LetStatement, ExpressionStatement, IfStatement, WhileStatement,
            ForEachStatement, ActionStatement, ReturnStatement, PrintStatement
        )
        
        if isinstance(stmt, LetStatement):
            self.analyze_let_statement(stmt)
        elif isinstance(stmt, ExpressionStatement):
            self.analyze_expression(stmt.expression)
        elif isinstance(stmt, IfStatement):
            self.analyze_if_statement(stmt)
        elif isinstance(stmt, WhileStatement):
            self.analyze_while_statement(stmt)
        elif isinstance(stmt, ForEachStatement):
            self.analyze_foreach_statement(stmt)
        elif isinstance(stmt, ActionStatement):
            self.analyze_action_statement(stmt)
        elif isinstance(stmt, ReturnStatement):
            self.analyze_expression(stmt.return_value)
        elif isinstance(stmt, PrintStatement):
            self.analyze_expression(stmt.value)
    
    def analyze_let_statement(self, stmt):
        # Check if variable is already defined
        var_name = stmt.name.value
        if var_name in self.symbol_table:
            self.errors.append(f"Variable '{var_name}' is already defined")
        else:
            self.symbol_table[var_name] = "variable"
            
        # Analyze the value expression
        self.analyze_expression(stmt.value)
    
    def analyze_if_statement(self, stmt):
        self.analyze_expression(stmt.condition)
        self.analyze_statement(stmt.consequence)
        if stmt.alternative:
            self.analyze_statement(stmt.alternative)
    
    def analyze_while_statement(self, stmt):
        self.analyze_expression(stmt.condition)
        self.analyze_statement(stmt.body)
    
    def analyze_foreach_statement(self, stmt):
        # Add loop variable to symbol table
        item_name = stmt.item.value
        self.symbol_table[item_name] = "loop_variable"
        
        self.analyze_expression(stmt.iterable)
        self.analyze_statement(stmt.body)
    
    def analyze_action_statement(self, stmt):
        # Add function to symbol table
        func_name = stmt.name.value
        self.symbol_table[func_name] = "function"
        
        # Analyze parameters and body
        for param in stmt.parameters:
            self.symbol_table[param.value] = "parameter"
            
        self.analyze_statement(stmt.body)
    
    def analyze_expression(self, expr):
        from .zexus_ast import (
            Identifier, CallExpression, MethodCallExpression, 
            AssignmentExpression, InfixExpression, PrefixExpression
        )
        
        if isinstance(expr, Identifier):
            # Check if identifier is defined
            if expr.value not in self.symbol_table:
                self.errors.append(f"Undefined variable: '{expr.value}'")
                
        elif isinstance(expr, CallExpression):
            self.analyze_expression(expr.function)
            for arg in expr.arguments:
                self.analyze_expression(arg)
                
        elif isinstance(expr, MethodCallExpression):
            self.analyze_expression(expr.object)
            for arg in expr.arguments:
                self.analyze_expression(arg)
                
        elif isinstance(expr, AssignmentExpression):
            # Check if variable exists for assignment
            if expr.name.value not in self.symbol_table:
                self.errors.append(f"Cannot assign to undefined variable: '{expr.name.value}'")
            self.analyze_expression(expr.value)
            
        elif isinstance(expr, InfixExpression):
            self.analyze_expression(expr.left)
            self.analyze_expression(expr.right)
            
        elif isinstance(expr, PrefixExpression):
            self.analyze_expression(expr.right)