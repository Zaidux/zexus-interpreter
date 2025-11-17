# zexus_ast.py (ENHANCED WITH PHASE 1 NODES)

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
    def __init__(self, file_path, alias=None):
        self.file_path = file_path  # StringLiteral for file path
        self.alias = alias          # Optional Identifier for alias

    def __repr__(self):
        alias_str = f", alias={self.alias}" if self.alias else ""
        return f"UseStatement(file_path={self.file_path}{alias_str})"

class FromStatement(Statement):
    def __init__(self, file_path, imports=None):
        self.file_path = file_path  # StringLiteral for file path
        self.imports = imports or [] # List of (Identifier, Optional Identifier) for name and alias

    def __repr__(self):
        return f"FromStatement(file_path={self.file_path}, imports={len(self.imports)})"

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

# NEW: Component and Theme AST nodes for interpreter
class ComponentStatement(Statement):
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties  # expected to be MapLiteral or BlockStatement

    def __repr__(self):
        return f"ComponentStatement(name={self.name}, properties={self.properties})"

class ThemeStatement(Statement):
    def __init__(self, name, properties):
        self.name = name
        self.properties = properties  # expected to be MapLiteral or BlockStatement

    def __repr__(self):
        return f"ThemeStatement(name={self.name}, properties={self.properties})"

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

# Export statement
class ExportStatement(Statement):
    def __init__(self, name=None, names=None, allowed_files=None, permission=None):
        # `names` is a list of Identifier nodes; `name` kept for backward compatibility (first item)
        self.names = names or ([] if names is not None else ([name] if name is not None else []))
        self.name = self.names[0] if self.names else name
        self.allowed_files = allowed_files or []
        self.permission = permission or "read_only"

    def __repr__(self):
        names = [n.value if hasattr(n, 'value') else str(n) for n in self.names]
        return f"ExportStatement(names={names}, files={len(self.allowed_files)}, permission='{self.permission}')"

# NEW: Debug statement
class DebugStatement(Statement):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"DebugStatement(value={self.value})"

# NEW: Try-catch statement  
class TryCatchStatement(Statement):
    def __init__(self, try_block, error_variable, catch_block):
        self.try_block = try_block
        self.error_variable = error_variable
        self.catch_block = catch_block

    def __repr__(self):
        return f"TryCatchStatement(error_var={self.error_variable})"

# NEW: External function declaration
class ExternalDeclaration(Statement):
    def __init__(self, name, parameters, module_path):
        self.name = name
        self.parameters = parameters
        self.module_path = module_path

    def __repr__(self):
        return f"ExternalDeclaration(name={self.name}, module={self.module_path})"

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

# Lambda expression
class LambdaExpression(Expression):
    def __init__(self, parameters, body):
        self.parameters = parameters
        self.body = body

    def __repr__(self):
        return f"LambdaExpression(parameters={len(self.parameters)})"

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


# =====================================================
# NEW: ENTITY, VERIFY, CONTRACT, PROTECT STATEMENTS
# =====================================================

class EntityStatement(Statement):
    """Entity declaration - advanced OOP with inheritance from let
    
    entity User {
        name: string,
        email: string,
        role: string = "user"
    }
    """
    def __init__(self, name, properties, parent=None, methods=None):
        self.name = name                    # Identifier
        self.properties = properties        # List of {name, type, default_value}
        self.parent = parent                # Optional parent entity (inheritance)
        self.methods = methods or []        # List of ActionStatement

    def __repr__(self):
        return f"EntityStatement(name={self.name}, properties={len(self.properties)})"


class VerifyStatement(Statement):
    """Verify security checks - wraps verification logic around components
    
    verify(transfer_funds, [
        check_authenticated(),
        check_balance(amount),
        check_whitelist(recipient)
    ])
    """
    def __init__(self, target, conditions, error_handler=None):
        self.target = target                # Function/action to verify
        self.conditions = conditions        # List of verification conditions
        self.error_handler = error_handler  # Optional error handling action

    def __repr__(self):
        return f"VerifyStatement(target={self.target}, conditions={len(self.conditions)})"


class ContractStatement(Statement):
    """Smart contract declaration - persistent state + actions
    
    contract Token {
        persistent storage balances: Map<Address, integer>
        persistent storage owner: Address
        
        action transfer(to: Address, amount: integer) -> boolean { ... }
    }
    """
    def __init__(self, name, storage_vars, actions, blockchain_config=None):
        self.name = name                        # Identifier
        self.storage_vars = storage_vars or []  # Persistent storage declarations
        self.actions = actions or []            # Contract methods/actions
        self.blockchain_config = blockchain_config or {}  # Network config

    def __repr__(self):
        return f"ContractStatement(name={self.name}, storage={len(self.storage_vars)})"


class ProtectStatement(Statement):
    """Protection guardrails - security rules against unauthorized access
    
    protect(app, {
        rate_limit: 100,          // 100 requests per minute
        auth_required: true,      // Must be authenticated
        allowed_ips: ["10.0.0.0/8"],
        blocked_ips: ["192.168.1.1"],
        require_https: true,
        min_password_strength: "strong",
        session_timeout: 3600
    })
    """
    def __init__(self, target, rules, enforcement_level="strict"):
        self.target = target                    # Function/app to protect
        self.rules = rules                      # Protection rules (Map or dict)
        self.enforcement_level = enforcement_level  # "strict", "warn", or "audit"

    def __repr__(self):
        return f"ProtectStatement(target={self.target}, enforcement={self.enforcement_level})"


# Additional advanced statements for completeness
class MiddlewareStatement(Statement):
    """Middleware registration - request/response processing
    
    middleware(authenticate, (request, response) -> {
        let token = request.headers["Authorization"]
        if (!verify_token(token)) {
            response.status = 401
            return false
        }
        return true
    })
    """
    def __init__(self, name, handler):
        self.name = name                    # Identifier
        self.handler = handler              # ActionLiteral with (req, res) parameters

    def __repr__(self):
        return f"MiddlewareStatement(name={self.name})"


class AuthStatement(Statement):
    """Authentication configuration
    
    auth {
        provider: "oauth2",
        scopes: ["read", "write", "delete"],
        token_expiry: 3600
    }
    """
    def __init__(self, config):
        self.config = config                # Map or dict with auth config

    def __repr__(self):
        return f"AuthStatement(config_keys={len(self.config.items()) if hasattr(self.config, 'items') else 0})"


class ThrottleStatement(Statement):
    """Rate limiting/throttling
    
    throttle(api_endpoint, {
        requests_per_minute: 100,
        burst_size: 10,
        per_user: true
    })
    """
    def __init__(self, target, limits):
        self.target = target                # Function to throttle
        self.limits = limits                # Throttle limits (Map or dict)

    def __repr__(self):
        return f"ThrottleStatement(target={self.target})"


class CacheStatement(Statement):
    """Caching directive
    
    cache(expensive_query, {
        ttl: 3600,              // Time to live: 1 hour
        key: "query_result",
        invalidate_on: ["data_changed"]
    })
    """
    def __init__(self, target, policy):
        self.target = target                # Function to cache
        self.policy = policy                # Cache policy (Map or dict)

    def __repr__(self):
        return f"CacheStatement(target={self.target})"


class SealStatement(Statement):
    """Seal statement - make a variable/object immutable at runtime

    seal myObj
    """
    def __init__(self, target):
        # target is expected to be an Identifier or PropertyAccessExpression
        self.target = target

    def __repr__(self):
        return f"SealStatement(target={self.target})"