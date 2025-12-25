# zexus_ast.py

class Node:
    def token_literal(self):
        pass
    
    def string(self):
        pass

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
        
    def string(self):
        return "".join(stmt.string() for stmt in self.statements)

class Identifier(Expression):
    def __init__(self, value):
        self.value = value
        
    def token_literal(self):
        return str(self.value)
        
    def string(self):
        return self.value

class LetStatement(Statement):
    def __init__(self, name=None, value=None):
        self.name = name
        self.value = value
        
    def token_literal(self):
        return "let"
        
    def string(self):
        return f"{self.token_literal()} {self.name.string()} = {self.value.string()}"

class ReturnStatement(Statement):
    def __init__(self, return_value=None):
        self.return_value = return_value
        
    def token_literal(self):
        return "return"
        
    def string(self):
        return f"{self.token_literal()} {self.return_value.string()}"

class ExpressionStatement(Statement):
    def __init__(self, expression=None):
        self.expression = expression
        
    def token_literal(self):
        return self.expression.token_literal() if self.expression else ""
        
    def string(self):
        return self.expression.string() if self.expression else ""

class IntegerLiteral(Expression):
    def __init__(self, value):
        self.value = value
        
    def token_literal(self):
        return str(self.value)
        
    def string(self):
        return str(self.value)

class FloatLiteral(Expression):
    def __init__(self, value):
        self.value = value
        
    def token_literal(self):
        return str(self.value)
        
    def string(self):
        return str(self.value)

class StringLiteral(Expression):
    def __init__(self, value):
        self.value = value
        
    def token_literal(self):
        return self.value
        
    def string(self):
        return f'"{self.value}"'

class Boolean(Expression):
    def __init__(self, value):
        self.value = value
        
    def token_literal(self):
        return "true" if self.value else "false"
        
    def string(self):
        return self.token_literal()

class PrefixExpression(Expression):
    def __init__(self, operator=None, right=None):
        self.operator = operator
        self.right = right
        
    def token_literal(self):
        return self.operator
        
    def string(self):
        return f"({self.operator}{self.right.string()})"

class InfixExpression(Expression):
    def __init__(self, left=None, operator=None, right=None):
        self.left = left
        self.operator = operator
        self.right = right
        
    def token_literal(self):
        return self.operator
        
    def string(self):
        return f"({self.left.string()} {self.operator} {self.right.string()})"

class BlockStatement(Statement):
    def __init__(self):
        self.statements = []
        
    def token_literal(self):
        return "{" if len(self.statements) > 0 else ""
        
    def string(self):
        return "".join(stmt.string() for stmt in self.statements)

class IfExpression(Expression):
    def __init__(self, condition=None, consequence=None, alternative=None):
        self.condition = condition
        self.consequence = consequence
        self.alternative = alternative
        
    def token_literal(self):
        return "if"
        
    def string(self):
        alt = f" else {self.alternative.string()}" if self.alternative else ""
        return f"if({self.condition.string()}) {self.consequence.string()}{alt}"

class ActionLiteral(Expression):
    def __init__(self, parameters=None, body=None):
        self.parameters = parameters or []
        self.body = body
        
    def token_literal(self):
        return "action"
        
    def string(self):
        params = ", ".join(p.string() for p in self.parameters)
        return f"action({params}) {self.body.string()}"

class CallExpression(Expression):
    def __init__(self, function=None, arguments=None):
        self.function = function
        self.arguments = arguments or []
        
    def token_literal(self):
        return "call"
        
    def string(self):
        args = ", ".join(arg.string() for arg in self.arguments)
        return f"{self.function.string()}({args})"

class ListLiteral(Expression):
    def __init__(self, elements=None):
        self.elements = elements or []
        
    def token_literal(self):
        return "["
        
    def string(self):
        elements = ", ".join(el.string() for el in self.elements)
        return f"[{elements}]"

class MapLiteral(Expression):
    def __init__(self, pairs=None):
        self.pairs = pairs or []
        
    def token_literal(self):
        return "{"
        
    def string(self):
        pairs = ", ".join(f"{k.string()}: {v.string()}" for k, v in self.pairs)
        return f"{{{pairs}}}"

class PrintStatement(Statement):
    def __init__(self, value=None):
        self.value = value
        
    def token_literal(self):
        return "print"
        
    def string(self):
        return f"print {self.value.string()}"

class ForEachStatement(Statement):
    def __init__(self, item=None, iterable=None, body=None):
        self.item = item
        self.iterable = iterable
        self.body = body
        
    def token_literal(self):
        return "for each"
        
    def string(self):
        return f"for each {self.item.string()} in {self.iterable.string()} {self.body.string()}"

class ScreenStatement(Statement):
    def __init__(self, name=None, body=None):
        self.name = name
        self.body = body
        
    def token_literal(self):
        return "screen"
        
    def string(self):
        return f"screen {self.name.string()} {self.body.string()}"

# Module system AST nodes
class UseStatement(Statement):
    def __init__(self, module_name=None, alias=None):
        self.module_name = module_name  # StringLiteral or Identifier
        self.alias = alias  # Optional Identifier
        
    def token_literal(self):
        return "use"
        
    def string(self):
        as_part = f" as {self.alias.string()}" if self.alias else ""
        return f"use {self.module_name.string()}{as_part}"

class FromStatement(Statement):
    def __init__(self, module_name=None, imports=None):
        self.module_name = module_name  # StringLiteral or Identifier
        self.imports = imports or []  # List of (Identifier, Optional Identifier) for name and alias
        
    def token_literal(self):
        return "from"
        
    def string(self):
        imports = ", ".join(
            f"{name.string()}" + (f" as {alias.string()}" if alias else "")
            for name, alias in self.imports
        )
        return f"from {self.module_name.string()} use {imports}"

class ExportStatement(Statement):
    def __init__(self, declaration=None):
        self.declaration = declaration  # The statement being exported (usually LetStatement or ActionLiteral)
        
    def token_literal(self):
        return "export"
        
    def string(self):
        return f"export {self.declaration.string()}"


def attach_modifiers(node, modifiers):
    """Attach a list of modifier strings to an AST node.

    This is a lightweight, non-invasive helper so existing AST classes
    do not need to accept modifiers in their constructors. Use after
    creating a Statement node in the parser.
    """
    try:
        if modifiers:
            setattr(node, 'modifiers', list(modifiers))
        else:
            # Ensure attribute exists for consistency
            setattr(node, 'modifiers', [])
    except Exception:
        # Best-effort: ignore if node cannot be modified
        pass
    return node