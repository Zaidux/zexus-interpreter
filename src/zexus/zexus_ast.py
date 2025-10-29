# zexus_ast.py (ENHANCED WITH ASSIGNMENT & PROPERTY ACCESS)

# Base classes
class Node: 
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Statement(Node): pass
class Expression(Node): pass

class Program(Node):
    def __init__(self):
        self.statements = []
    
    def __repr__(self):
        return f"Program(statements={len(self.statements)})"

# Statement Nodes
class LetStatement(Statement):
    def __init__(self, name, value): 
        self.name = name; self.value = value
    
    def __repr__(self):
        return f"LetStatement(name={self.name}, value={self.value})"

class ReturnStatement(Statement):
    def __init__(self, return_value):
        self.return_value = return_value
    
    def __repr__(self):
        return f"ReturnStatement(return_value={self.return_value})"

class ExpressionStatement(Statement):
    def __init__(self, expression): 
        self.expression = expression
    
    def __repr__(self):
        return f"ExpressionStatement(expression={self.expression})"

class BlockStatement(Statement):
    def __init__(self): 
        self.statements = []
    
    def __repr__(self):
        return f"BlockStatement(statements={len(self.statements)})"

class PrintStatement(Statement):
    def __init__(self, value): 
        self.value = value
    
    def __repr__(self):
        return f"PrintStatement(value={self.value})"

class ForEachStatement(Statement):
    def __init__(self, item, iterable, body):
        self.item = item; self.iterable = iterable; self.body = body
    
    def __repr__(self):
        return f"ForEachStatement(item={self.item}, iterable={self.iterable})"

class EmbeddedCodeStatement(Statement):
    def __init__(self, name, language, code):
        self.name = name
        self.language = language
        self.code = code
    
    def __repr__(self):
        return f"EmbeddedCodeStatement(name={self.name}, language={self.language})"

class UseStatement(Statement):
    def __init__(self, embedded_ref, method, arguments):
        self.embedded_ref = embedded_ref
        self.method = method
        self.arguments = arguments
    
    def __repr__(self):
        return f"UseStatement(embedded_ref={self.embedded_ref}, method={self.method})"

class IfStatement(Statement):
    def __init__(self, condition, consequence, alternative=None):
        self.condition = condition
        self.consequence = consequence
        self.alternative = alternative
    
    def __repr__(self):
        return f"IfStatement(condition={self.condition})"

class WhileStatement(Statement):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body
    
    def __repr__(self):
        return f"WhileStatement(condition={self.condition})"

class ScreenStatement(Statement):
    def __init__(self, name, body):
        self.name = name
        self.body = body
    
    def __repr__(self):
        return f"ScreenStatement(name={self.name})"

class ActionStatement(Statement):
    def __init__(self, name, parameters, body):
        self.name = name
        self.parameters = parameters
        self.body = body
    
    def __repr__(self):
        return f"ActionStatement(name={self.name}, parameters={len(self.parameters)})"

class ExactlyStatement(Statement):
    def __init__(self, name, body):
        self.name = name
        self.body = body
    
    def __repr__(self):
        return f"ExactlyStatement(name={self.name})"

# Expression Nodes
class Identifier(Expression):
    def __init__(self, value): 
        self.value = value
    
    def __repr__(self):
        return f"Identifier('{self.value}')"

class IntegerLiteral(Expression):
    def __init__(self, value): 
        self.value = value
    
    def __repr__(self):
        return f"IntegerLiteral({self.value})"

class FloatLiteral(Expression):
    def __init__(self, value): 
        self.value = value
    
    def __repr__(self):
        return f"FloatLiteral({self.value})"

class StringLiteral(Expression):
    def __init__(self, value): 
        self.value = value
    
    def __repr__(self):
        return f"StringLiteral('{self.value}')"

class Boolean(Expression):
    def __init__(self, value): 
        self.value = value
    
    def __repr__(self):
        return f"Boolean({self.value})"

class ListLiteral(Expression):
    def __init__(self, elements): 
        self.elements = elements
    
    def __repr__(self):
        return f"ListLiteral(elements={len(self.elements)})"

class MapLiteral(Expression):
    def __init__(self, pairs): 
        self.pairs = pairs
    
    def __repr__(self):
        return f"MapLiteral(pairs={len(self.pairs)})"

class ActionLiteral(Expression):
    def __init__(self, parameters, body):
        self.parameters = parameters
        self.body = body
    
    def __repr__(self):
        return f"ActionLiteral(parameters={len(self.parameters)})"

class CallExpression(Expression):
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments
    
    def __repr__(self):
        return f"CallExpression(function={self.function}, arguments={len(self.arguments)})"

class MethodCallExpression(Expression):
    def __init__(self, object, method, arguments):
        self.object = object
        self.method = method
        self.arguments = arguments
    
    def __repr__(self):
        return f"MethodCallExpression(object={self.object}, method={self.method})"

class PropertyAccessExpression(Expression):
    def __init__(self, object, property):
        self.object = object
        self.property = property
    
    def __repr__(self):
        return f"PropertyAccessExpression(object={self.object}, property={self.property})"

class AssignmentExpression(Expression):
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f"AssignmentExpression(name={self.name}, value={self.value})"

class EmbeddedLiteral(Expression):
    def __init__(self, language, code):
        self.language = language
        self.code = code
    
    def __repr__(self):
        return f"EmbeddedLiteral(language={self.language})"

class PrefixExpression(Expression):
    def __init__(self, operator, right): 
        self.operator = operator; self.right = right
    
    def __repr__(self):
        return f"PrefixExpression(operator='{self.operator}', right={self.right})"

class InfixExpression(Expression):
    def __init__(self, left, operator, right): 
        self.left = left; self.operator = operator; self.right = right
    
    def __repr__(self):
        return f"InfixExpression(left={self.left}, operator='{self.operator}', right={self.right})"

class IfExpression(Expression):
    def __init__(self, condition, consequence, alternative=None):
        self.condition = condition
        self.consequence = consequence
        self.alternative = alternative
    
    def __repr__(self):
        return f"IfExpression(condition={self.condition})"