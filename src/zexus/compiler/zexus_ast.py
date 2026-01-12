"""
Clean AST Definitions for Compiler Phase
"""

# Base classes
class Node: 
    def token_literal(self):
        return ""

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class Statement(Node): 
    pass

class Expression(Node): 
    pass

class Program(Node):
    def __init__(self):
        self.statements = []

    def token_literal(self):
        if len(self.statements) > 0:
            return self.statements[0].token_literal()
        return ""

    def __repr__(self):
        return f"Program({len(self.statements)} statements)"

# Statement Nodes
class LetStatement(Statement):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def token_literal(self):
        return "let"

    def __repr__(self):
        return f"LetStatement({self.name})"

class ReturnStatement(Statement):
    def __init__(self, return_value):
        self.return_value = return_value

    def token_literal(self):
        return "return"

    def __repr__(self):
        return f"ReturnStatement({self.return_value})"

class ExpressionStatement(Statement):
    def __init__(self, expression):
        self.expression = expression

    def token_literal(self):
        return self.expression.token_literal()

    def __repr__(self):
        return f"ExpressionStatement({self.expression})"

class BlockStatement(Statement):
    def __init__(self):
        self.statements = []

    def token_literal(self):
        if len(self.statements) > 0:
            return self.statements[0].token_literal()
        return ""

    def __repr__(self):
        return f"BlockStatement({len(self.statements)} statements)"

class PrintStatement(Statement):
    def __init__(self, value):
        self.value = value

    def token_literal(self):
        return "print"

    def __repr__(self):
        return f"PrintStatement({self.value})"

class IfStatement(Statement):
    def __init__(self, condition, consequence, alternative=None):
        self.condition = condition
        self.consequence = consequence
        self.alternative = alternative

    def token_literal(self):
        return "if"

    def __repr__(self):
        return f"IfStatement(condition={self.condition})"

class WhileStatement(Statement):
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

    def token_literal(self):
        return "while"

    def __repr__(self):
        return f"WhileStatement(condition={self.condition})"

class ForEachStatement(Statement):
    def __init__(self, item, iterable, body):
        self.item = item
        self.iterable = iterable
        self.body = body

    def token_literal(self):
        return "for"

    def __repr__(self):
        return f"ForEachStatement(item={self.item}, iterable={self.iterable})"

class ActionStatement(Statement):
    def __init__(self, name, parameters, body):
        self.name = name
        self.parameters = parameters
        self.body = body

    def token_literal(self):
        return "action"

    def __repr__(self):
        return f"ActionStatement({self.name}, {len(self.parameters)} params)"

class UseStatement(Statement):
    def __init__(self, file_path, alias=None):
        self.file_path = file_path
        self.alias = alias

    def token_literal(self):
        return "use"

    def __repr__(self):
        return f"UseStatement('{self.file_path}')"

# NEW: Compiler-side Screen/Component/Theme nodes
class ScreenStatement(Statement):
    def __init__(self, name, body):
        self.name = name
        self.body = body

    def token_literal(self):
        return "screen"

    def __repr__(self):
        return f"ScreenStatement({self.name})"

class ComponentStatement(Statement):
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties

    def token_literal(self):
        return "component"

    def __repr__(self):
        return f"ComponentStatement({self.name})"

class ThemeStatement(Statement):
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties

    def token_literal(self):
        return "theme"

    def __repr__(self):
        return f"ThemeStatement({self.name})"

# NEW: TryCatchStatement for compiler AST (matches interpreter node)
class TryCatchStatement(Statement):
    def __init__(self, try_block, error_variable, catch_block):
        self.try_block = try_block
        self.error_variable = error_variable
        self.catch_block = catch_block

    def token_literal(self):
        return "try"

    def __repr__(self):
        return f"TryCatchStatement(error_var={self.error_variable})"

# NEW: ExternalDeclaration for compiler AST (matches interpreter node)
class ExternalDeclaration(Statement):
    def __init__(self, name, parameters, module_path):
        self.name = name
        self.parameters = parameters or []
        self.module_path = module_path

    def token_literal(self):
        return "external"

    def __repr__(self):
        return f"ExternalDeclaration(name={self.name}, module={self.module_path})"

# Expression Nodes
class Identifier(Expression):
    def __init__(self, value):
        self.value = value

    def token_literal(self):
        return self.value

    def __repr__(self):
        return f"Identifier('{self.value}')"

class IntegerLiteral(Expression):
    def __init__(self, value):
        self.value = value

    def token_literal(self):
        return str(self.value)

    def __repr__(self):
        return f"IntegerLiteral({self.value})"

class FloatLiteral(Expression):
    def __init__(self, value):
        self.value = value

    def token_literal(self):
        return str(self.value)

    def __repr__(self):
        return f"FloatLiteral({self.value})"

class StringLiteral(Expression):
    def __init__(self, value):
        self.value = value

    def token_literal(self):
        return self.value

    def __repr__(self):
        return f"StringLiteral('{self.value}')"

class Boolean(Expression):
    def __init__(self, value):
        self.value = value

    def token_literal(self):
        return "true" if self.value else "false"

    def __repr__(self):
        return f"Boolean({self.value})"

class ListLiteral(Expression):
    def __init__(self, elements):
        self.elements = elements

    def token_literal(self):
        return "["

    def __repr__(self):
        return f"ListLiteral({len(self.elements)} elements)"

class MapLiteral(Expression):
    def __init__(self, pairs):
        self.pairs = pairs

    def token_literal(self):
        return "{"

    def __repr__(self):
        return f"MapLiteral({len(self.pairs)} pairs)"

class PrefixExpression(Expression):
    def __init__(self, operator, right):
        self.operator = operator
        self.right = right

    def token_literal(self):
        return self.operator

    def __repr__(self):
        return f"PrefixExpression('{self.operator}', {self.right})"

class InfixExpression(Expression):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def token_literal(self):
        return self.operator

    def __repr__(self):
        return f"InfixExpression({self.left}, '{self.operator}', {self.right})"

class CallExpression(Expression):
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments

    def token_literal(self):
        return self.function.token_literal()

    def __repr__(self):
        return f"CallExpression({self.function}, {len(self.arguments)} args)"

class AssignmentExpression(Expression):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def token_literal(self):
        return "="

    def __repr__(self):
        return f"AssignmentExpression({self.name}, {self.value})"

class MethodCallExpression(Expression):
    def __init__(self, object, method, arguments):
        self.object = object
        self.method = method
        self.arguments = arguments

    def token_literal(self):
        return self.method.token_literal()

    def __repr__(self):
        return f"MethodCallExpression({self.object}.{self.method})"

class PropertyAccessExpression(Expression):
    def __init__(self, object, property):
        self.object = object
        self.property = property

    def token_literal(self):
        return self.property.token_literal()

    def __repr__(self):
        return f"PropertyAccessExpression({self.object}.{self.property})"

class LambdaExpression(Expression):
    def __init__(self, parameters, body):
        self.parameters = parameters
        self.body = body

    def token_literal(self):
        return "lambda"

    def __repr__(self):
        return f"LambdaExpression({len(self.parameters)} params)"

class ActionLiteral(Expression):
    def __init__(self, parameters, body):
        self.parameters = parameters
        self.body = body

    def token_literal(self):
        return "action"

    def __repr__(self):
        return f"ActionLiteral({len(self.parameters)} params)"

class IfExpression(Expression):
    def __init__(self, condition, consequence, alternative=None):
        self.condition = condition
        self.consequence = consequence
        self.alternative = alternative

    def token_literal(self):
        return "if"

    def __repr__(self):
        return f"IfExpression(condition={self.condition})"

class EmbeddedLiteral(Expression):
    def __init__(self, language, code):
        self.language = language
        self.code = code

    def token_literal(self):
        return "embedded"

    def __repr__(self):
        return f"EmbeddedLiteral({self.language})"

# NEW: AwaitExpression (used in compiler AST)
class AwaitExpression(Expression):
    def __init__(self, expression):
        self.expression = expression

    def token_literal(self):
        return "await"

    def __repr__(self):
        return f"AwaitExpression({self.expression})"

# NEW: EventDeclaration / EmitStatement
class EventDeclaration(Statement):
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties

    def token_literal(self):
        return "event"

    def __repr__(self):
        return f"EventDeclaration({self.name})"

class EmitStatement(Statement):
    def __init__(self, name, payload=None):
        self.name = name
        self.payload = payload

    def token_literal(self):
        return "emit"

    def __repr__(self):
        return f"EmitStatement({self.name}, payload={self.payload})"

# NEW: EnumDeclaration and ProtocolDeclaration
class EnumDeclaration(Statement):
    def __init__(self, name, members):
        self.name = name
        self.members = members

    def token_literal(self):
        return "enum"

    def __repr__(self):
        return f"EnumDeclaration({self.name})"

class ProtocolDeclaration(Statement):
    def __init__(self, name, spec):
        self.name = name
        self.spec = spec

    def token_literal(self):
        return "protocol"

    def __repr__(self):
        return f"ProtocolDeclaration({self.name})"

# NEW: ImportStatement (explicit import syntax)
class ImportStatement(Statement):
    def __init__(self, module_path, alias=None):
        self.module_path = module_path
        self.alias = alias

    def token_literal(self):
        return "import"

    def __repr__(self):
        return f"ImportStatement({self.module_path} as {self.alias})"


class StreamStatement(Statement):
    def __init__(self, stream_name, event_var, handler):
        self.stream_name = stream_name
        self.event_var = event_var
        self.handler = handler

    def token_literal(self):
        return "stream"

    def __repr__(self):
        return f"StreamStatement({self.stream_name})"


class WatchStatement(Statement):
    def __init__(self, watched_expr, reaction):
        self.watched_expr = watched_expr
        self.reaction = reaction

    def token_literal(self):
        return "watch"

    def __repr__(self):
        return f"WatchStatement({self.watched_expr})"