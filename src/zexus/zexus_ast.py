# src/zexus/zexus_ast.py

# Base classes
class Node: 
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return self.__repr__()

class Statement(Node): pass
class Expression(Node): pass

class Program(Node):
    def __init__(self):
        self.statements = []

    def __repr__(self):
        return f"Program(statements={len(self.statements)})"

# Statement Nodes
class LetStatement(Statement):
    def __init__(self, name, value, type_annotation=None): 
        self.name = name
        self.value = value
        self.type_annotation = type_annotation

    def __repr__(self):
        type_str = f", type={self.type_annotation}" if self.type_annotation else ""
        return f"LetStatement(name={self.name}, value={self.value}{type_str})"

class ConstStatement(Statement):
    """Const statement - immutable variable declaration
    
    const MAX_VALUE = 100;
    """
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"ConstStatement(name={self.name}, value={self.value})"

class DataStatement(Statement):
    """Data statement - dataclass definition
    
    data User {
        name: string,
        email: string,
        age: number
    }
    
    data Admin extends User {
        permissions: array
    }
    
    @validated
    data Email {
        address: string
    }
    
    data Box<T> {
        value: T
    }
    
    Creates a structured type with automatic methods:
    - Constructor
    - toString(), toJSON(), fromJSON()
    - equals(), clone(), hash()
    """
    def __init__(self, name, fields, modifiers=None, parent=None, decorators=None, type_params=None):
        self.name = name           # Identifier: class name
        self.fields = fields       # List of DataField objects
        self.modifiers = modifiers or []  # List of modifiers: ["immutable", "verified", etc.]
        self.parent = parent       # String: parent dataclass name (for inheritance)
        self.decorators = decorators or []  # List of decorator names: ["validated", "logged", etc.]
        self.type_params = type_params or []  # List of type parameter names: ["T", "U", "V"]

    def __repr__(self):
        mods = f", modifiers={self.modifiers}" if self.modifiers else ""
        parent = f", extends={self.parent}" if self.parent else ""
        decs = f", decorators={self.decorators}" if self.decorators else ""
        tparams = f", type_params={self.type_params}" if self.type_params else ""
        return f"DataStatement(name={self.name}, fields={self.fields}{mods}{parent}{decs}{tparams})"

class DataField:
    """Field definition in a dataclass
    
    name: string = "default" require len(name) > 0
    method area() { return this.width * this.height; }
    operator +(other) { return Point(this.x + other.x, this.y + other.y); }
    
    @logged
    method calculate() { ... }
    """
    def __init__(self, name, field_type=None, default_value=None, constraint=None, computed=None, method_body=None, method_params=None, operator=None, decorators=None):
        self.name = name                    # Identifier: field name
        self.field_type = field_type        # String: type annotation (optional)
        self.default_value = default_value  # Expression: default value (optional)
        self.constraint = constraint        # Expression: validation constraint (optional)
        self.computed = computed            # Expression: computed property (optional)
        self.method_body = method_body      # BlockStatement: method body (optional)
        self.method_params = method_params or []  # List of parameters for method
        self.operator = operator            # String: operator symbol (+, -, *, /, ==, etc.)
        self.decorators = decorators or []  # List of decorator names for methods

    def __repr__(self):
        parts = [f"name={self.name}"]
        if self.field_type:
            parts.append(f"type={self.field_type}")
        if self.default_value:
            parts.append(f"default={self.default_value}")
        if self.constraint:
            parts.append(f"require={self.constraint}")
        if self.computed:
            parts.append(f"computed={self.computed}")
        return f"DataField({', '.join(parts)})"

class ReturnStatement(Statement):
    def __init__(self, return_value):
        self.return_value = return_value

    def __repr__(self):
        return f"ReturnStatement(return_value={self.return_value})"

class ContinueStatement(Statement):
    """Continue on error - allows program to continue execution even on errors."""
    def __init__(self):
        pass

    def __repr__(self):
        return "ContinueStatement()"

class BreakStatement(Statement):
    """Break statement - exits the current loop."""
    def __init__(self):
        pass

    def __repr__(self):
        return "BreakStatement()"

class ThrowStatement(Statement):
    """Throw statement - throws an error/exception."""
    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return f"ThrowStatement(message={self.message})"

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

class TxStatement(Statement):
    """Transaction block statement - executes statements in transactional context
    
    tx {
        balance = balance - amount;
        recipient_balance = recipient_balance + amount;
    }
    """
    def __init__(self, body):
        self.body = body  # BlockStatement

    def __repr__(self):
        return f"TxStatement(body={self.body})"

class PrintStatement(Statement):
    def __init__(self, value=None, values=None, condition=None): 
        # Support both single value (legacy) and multiple values
        # condition: optional condition expression (for conditional printing)
        self.value = value
        self.values = values if values is not None else ([value] if value is not None else [])
        self.condition = condition

    def __repr__(self):
        if self.condition:
            return f"PrintStatement(condition={self.condition}, values={self.values})"
        return f"PrintStatement(values={self.values})"

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
    def __init__(self, file_path, alias=None, names=None, is_named_import=False):
        self.file_path = file_path  # StringLiteral or string path
        self.alias = alias          # Optional Identifier for alias
        self.names = names or []    # List of Identifiers for named imports
        self.is_named_import = is_named_import

    def __repr__(self):
        if self.is_named_import:
            names_list = [str(n.value if hasattr(n, 'value') else n) for n in self.names]
            return f"UseStatement(file='{self.file_path}', names={names_list})"
        alias_str = f", alias={self.alias}" if self.alias else ""
        return f"UseStatement(file_path={self.file_path}{alias_str})"

    def __str__(self):
        if self.is_named_import:
            names_str = ", ".join([str(n.value if hasattr(n, 'value') else n) for n in self.names])
            return f"use {{ {names_str} }} from '{self.file_path}'"
        elif self.alias:
            alias_val = self.alias.value if hasattr(self.alias, 'value') else self.alias
            return f"use '{self.file_path}' as {alias_val}"
        else:
            return f"use '{self.file_path}'"

class FromStatement(Statement):
    def __init__(self, file_path, imports=None):
        self.file_path = file_path  # StringLiteral for file path
        self.imports = imports or [] # List of (Identifier, Optional Identifier) for name and alias

    def __repr__(self):
        return f"FromStatement(file_path={self.file_path}, imports={len(self.imports)})"

class IfStatement(Statement):
    def __init__(self, condition, consequence, elif_parts=None, alternative=None):
        self.condition = condition
        self.consequence = consequence
        self.elif_parts = elif_parts or []  # List of (condition, consequence) tuples for elif chains
        self.alternative = alternative

    def __repr__(self):
        return f"IfStatement(condition={self.condition}, elif_parts={len(self.elif_parts)})"

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
    def __init__(self, name, parameters, body, is_async=False, return_type=None):
        self.name = name
        self.parameters = parameters
        self.body = body
        self.is_async = is_async
        self.return_type = return_type

    def __repr__(self):
        async_str = "async " if self.is_async else ""
        return_type_str = f" -> {self.return_type}" if self.return_type else ""
        return f"ActionStatement({async_str}name={self.name}, parameters={len(self.parameters)}{return_type_str})"

class FunctionStatement(Statement):
    def __init__(self, name, parameters, body, return_type=None):
        self.name = name
        self.parameters = parameters
        self.body = body
        self.return_type = return_type

    def __repr__(self):
        return_type_str = f" -> {self.return_type}" if self.return_type else ""
        return f"FunctionStatement(name={self.name}, parameters={len(self.parameters)}{return_type_str})"

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
    def __init__(self, value, condition=None):
        # value: expression to debug print
        # condition: optional condition expression (for conditional debugging)
        self.value = value
        self.condition = condition

    def __repr__(self):
        if self.condition:
            return f"DebugStatement(condition={self.condition}, value={self.value})"
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

class AuditStatement(Statement):
    """Audit statement - Log data access for compliance
    
    audit user_data, "access", timestamp;
    audit CONFIG, "modification", current_time;
    """
    def __init__(self, data_name, action_type, timestamp=None):
        self.data_name = data_name      # Variable/identifier to audit
        self.action_type = action_type  # String: "access", "modification", "deletion", etc.
        self.timestamp = timestamp      # Optional timestamp expression

    def __repr__(self):
        return f"AuditStatement(data={self.data_name}, action={self.action_type}, timestamp={self.timestamp})"

class RestrictStatement(Statement):
    """Restrict statement - Field-level access control
    
    restrict obj.field = "read-only";
    restrict user.password = "deny";
    restrict config.api_key = "admin-only";
    """
    def __init__(self, target, restriction_type):
        self.target = target              # PropertyAccessExpression for obj.field
        self.restriction_type = restriction_type  # String: "read-only", "deny", "admin-only", etc.

    def __repr__(self):
        return f"RestrictStatement(target={self.target}, restriction={self.restriction_type})"

class SandboxStatement(Statement):
    """Sandbox statement - Isolated execution environment
    
    sandbox {
      // code runs in isolated context
      let x = unsafe_operation();
    }
    """
    def __init__(self, body, policy=None):
        self.body = body  # BlockStatement containing sandboxed code
        self.policy = policy  # Optional policy name (string)

    def __repr__(self):
        return f"SandboxStatement(body={self.body}, policy={self.policy})"

class TrailStatement(Statement):
    """Trail statement - Real-time audit/debug/print tracking
    
    trail audit;       // follow all audit events
    trail print;       // follow all print statements
    trail debug;       // follow all debug output
    trail *, "resource_access";  // trail all events for resource_access
    """
    def __init__(self, trail_type, filter_key=None):
        self.trail_type = trail_type    # String: "audit", "print", "debug", "*"
        self.filter_key = filter_key    # Optional filter/pattern

    def __repr__(self):
        return f"TrailStatement(type={self.trail_type}, filter={self.filter_key})"

# Expression Nodes
class Identifier(Expression):
    def __init__(self, value): 
        self.value = value

    def __repr__(self):
        return f"Identifier('{self.value}')"
    
    def __str__(self):
        return self.value

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
    
    def __str__(self):
        return self.value

class Boolean(Expression):
    def __init__(self, value): 
        self.value = value

    def __repr__(self):
        return f"Boolean({self.value})"

class NullLiteral(Expression):
    def __repr__(self):
        return "NullLiteral()"


class ThisExpression(Expression):
    """This expression - reference to current contract instance
    
    this.balances[account]
    this.owner = TX.caller
    """
    def __repr__(self):
        return "ThisExpression()"


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
    def __init__(self, function, arguments, type_args=None):
        self.function = function
        self.arguments = arguments
        self.type_args = type_args or []  # List of type arguments for generic instantiation: Box<number>

    def __repr__(self):
        targs = f", type_args={self.type_args}" if self.type_args else ""
        return f"CallExpression(function={self.function}, arguments={len(self.arguments)}{targs})"

class AsyncExpression(Expression):
    """Async expression: async <expression>
    
    Executes the expression asynchronously in a background thread.
    Example: async producer()
    """
    def __init__(self, expression):
        self.expression = expression  # The expression to execute asynchronously
    
    def __repr__(self):
        return f"AsyncExpression(expression={self.expression})"

class MethodCallExpression(Expression):
    def __init__(self, object, method, arguments):
        self.object = object
        self.method = method
        self.arguments = arguments

    def __repr__(self):
        return f"MethodCallExpression(object={self.object}, method={self.method})"

class MatchExpression(Expression):
    """Match expression for pattern matching
    
    match value {
        Point(x, y) => x + y,
        User(name, _) => name,
        42 => "the answer",
        _ => "default"
    }
    """
    def __init__(self, value, cases):
        self.value = value  # Expression: value to match against
        self.cases = cases  # List[MatchCase]: pattern cases

    def __repr__(self):
        return f"MatchExpression(value={self.value}, cases={len(self.cases)})"

class MatchCase:
    """A single match case with pattern and result expression"""
    def __init__(self, pattern, result):
        self.pattern = pattern  # Pattern: pattern to match
        self.result = result  # Expression: result if matched

    def __repr__(self):
        return f"MatchCase(pattern={self.pattern}, result={self.result})"

# Pattern nodes for destructuring
class ConstructorPattern:
    """Constructor pattern: Point(x, y)"""
    def __init__(self, constructor_name, bindings):
        self.constructor_name = constructor_name  # String: type name
        self.bindings = bindings  # List[Pattern]: nested patterns or variable names

    def __repr__(self):
        return f"ConstructorPattern({self.constructor_name}, bindings={self.bindings})"

class VariablePattern:
    """Variable binding pattern: x, name, value"""
    def __init__(self, name):
        self.name = name  # String: variable name to bind

    def __repr__(self):
        return f"VariablePattern({self.name})"

class WildcardPattern:
    """Wildcard pattern: _ (matches anything, no binding)"""
    def __repr__(self):
        return "WildcardPattern(_)"

class LiteralPattern:
    """Literal pattern: 42, "hello", true"""
    def __init__(self, value):
        self.value = value  # Literal value (IntegerLiteral, StringLiteral, etc.)

    def __repr__(self):
        return f"LiteralPattern({self.value})"

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

class AwaitExpression(Expression):
    def __init__(self, expression):
        self.expression = expression

    def token_literal(self):
        return "await"

    def string(self):
        return f"await {self.expression}"

    def __repr__(self):
        return f"AwaitExpression(expression={self.expression})"

class FileImportExpression(Expression):
    """File import expression for << operator
    
    let code << "filename.ext"
    
    Reads the file contents and returns as a string.
    Supports any file extension.
    """
    def __init__(self, filepath):
        self.filepath = filepath  # Expression: path to file
    
    def __repr__(self):
        return f"FileImportExpression(<< {self.filepath})"

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
    def __init__(self, condition, consequence, elif_parts=None, alternative=None):
        self.condition = condition
        self.consequence = consequence
        self.elif_parts = elif_parts or []  # List of (condition, consequence) tuples for elif chains
        self.alternative = alternative

    def __repr__(self):
        return f"IfExpression(condition={self.condition}, elif_parts={len(self.elif_parts)})"

class TernaryExpression(Expression):
    """Represents: condition ? true_value : false_value"""
    def __init__(self, condition, true_value, false_value):
        self.condition = condition
        self.true_value = true_value
        self.false_value = false_value

    def __repr__(self):
        return f"TernaryExpression(condition={self.condition}, true={self.true_value}, false={self.false_value})"

class NullishExpression(Expression):
    """Represents: value ?? default (returns default if value is null/undefined)"""
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"NullishExpression(left={self.left}, default={self.right})"


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
        self.properties = properties        # List of dicts: {name, type, default_value}
        self.parent = parent                # Optional parent entity (inheritance)
        self.methods = methods or []        # List of ActionStatement

    def __repr__(self):
        return f"EntityStatement(name={self.name}, properties={len(self.properties)})"

    def __str__(self):
        name_str = self.name.value if hasattr(self.name, 'value') else str(self.name)
        props_str = ",\n    ".join([f"{prop['name']}: {prop['type']}" for prop in self.properties])
        return f"entity {name_str} {{\n    {props_str}\n}}"


class VerifyStatement(Statement):
    r"""Verify security checks - supports multiple forms:
    
    Simple assertion:
        verify condition, "error message"
    
    Complex verification wrapper:
        verify(transfer_funds, [
            check_authenticated(),
            check_balance(amount),
            check_whitelist(recipient)
        ])
    
    Data/format verification:
        verify:data input matches email_pattern, "Invalid email"
        verify:data input is_type "string", "Must be string"
    
    Access control (blocks on failure):
        verify:access user.role == "admin" {
            log_unauthorized_access(user);
            block_request();
        }
    
    Database verification:
        verify:db user_id exists_in "users", "User not found"
        verify:db email unique_in "users", "Email already exists"
    
    Environment variable verification:
        verify:env "API_KEY" is_set, "API_KEY not configured"
        verify:env "DEBUG_MODE" equals "false", "Debug mode must be disabled"
    
    Custom logic block:
        verify condition {
            // Custom actions on failure
            log_error("Verification failed");
            send_alert(admin);
            return false;
        }
    
    Pattern matching:
        verify:pattern email matches r"/^[a-z]+@[a-z]+\.[a-z]+$/", "Invalid format"
    """
    def __init__(self, condition=None, message=None, target=None, conditions=None, 
                 error_handler=None, mode=None, pattern=None, db_table=None, 
                 db_query=None, env_var=None, expected_value=None, 
                 logic_block=None, action_block=None, verify_type=None):
        # Simple assertion form
        self.condition = condition          # Boolean condition to check
        self.message = message              # Error message if condition fails
        
        # Complex wrapper form  
        self.target = target                # Function/action to verify
        self.conditions = conditions        # List of verification conditions
        self.error_handler = error_handler  # Optional error handling action
        
        # Extended forms
        self.mode = mode                    # Verification mode: 'data', 'access', 'db', 'env', 'pattern'
        self.pattern = pattern              # Pattern for pattern matching
        self.db_table = db_table            # Database table name
        self.db_query = db_query            # Database query/condition
        self.env_var = env_var              # Environment variable name
        self.expected_value = expected_value # Expected value for comparisons
        self.logic_block = logic_block      # Custom logic block (BlockStatement)
        self.action_block = action_block    # Action block on failure
        self.verify_type = verify_type      # Type check: 'email', 'number', 'string', etc.

    def __repr__(self):
        if self.mode:
            return f"VerifyStatement(mode={self.mode}, condition={self.condition})"
        return f"VerifyStatement(target={self.target}, conditions={len(self.conditions) if self.conditions else 0})"


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


# PERFORMANCE OPTIMIZATION STATEMENT NODES

class NativeStatement(Statement):
    """Native statement - Call C/C++ code directly
    
    native "libmath.so", "add_numbers"(x, y);
    native "libcrypto.so", "sha256"(data) as hash;
    """
    def __init__(self, library_name, function_name, args=None, alias=None):
        self.library_name = library_name  # String: path to .so/.dll file
        self.function_name = function_name  # String: function name in library
        self.args = args or []  # List[Expression]: arguments to function
        self.alias = alias  # Optional: variable name to store result

    def __repr__(self):
        return f"NativeStatement(lib={self.library_name}, func={self.function_name}, args={self.args}, alias={self.alias})"


class GCStatement(Statement):
    """Garbage Collection statement - Control garbage collection behavior
    
    gc "collect";       // Force garbage collection
    gc "pause";         // Pause garbage collection
    gc "resume";        // Resume garbage collection
    gc "enable_debug";  // Enable GC debug output
    """
    def __init__(self, action):
        self.action = action  # String: "collect", "pause", "resume", "enable_debug", "disable_debug"

    def __repr__(self):
        return f"GCStatement(action={self.action})"


class InlineStatement(Statement):
    """Inline statement - Mark function for inlining optimization
    
    inline my_function;
    inline critical_path_func;
    """
    def __init__(self, function_name):
        self.function_name = function_name  # String or Identifier: function to inline

    def __repr__(self):
        return f"InlineStatement(func={self.function_name})"


class BufferStatement(Statement):
    """Buffer statement - Direct memory access and manipulation
    
    buffer my_mem = allocate(1024);
    buffer my_mem.write(0, [1, 2, 3, 4]);
    buffer my_mem.read(0, 4);
    """
    def __init__(self, buffer_name, operation=None, arguments=None):
        self.buffer_name = buffer_name  # String: buffer name
        self.operation = operation  # String: "allocate", "write", "read", "free"
        self.arguments = arguments or []  # List[Expression]: operation arguments

    def __repr__(self):
        return f"BufferStatement(name={self.buffer_name}, op={self.operation}, args={self.arguments})"


class SIMDStatement(Statement):
    """SIMD statement - Vector operations using SIMD instructions
    
    simd vector1 + vector2;
    simd matrix_mul(A, B);
    simd dot_product([1,2,3], [4,5,6]);
    """
    def __init__(self, operation, operands=None):
        self.operation = operation  # Expression: SIMD operation (binary op or function call)
        self.operands = operands or []  # List[Expression]: operands for SIMD

    def __repr__(self):
        return f"SIMDStatement(op={self.operation}, operands={self.operands})"


# CONVENIENCE FEATURE STATEMENT NODES

class DeferStatement(Statement):
    """Defer statement - Cleanup code execution (LIFO order)
    
    defer file.close();
    defer cleanup();
    defer release_lock();
    """
    def __init__(self, code_block):
        self.code_block = code_block  # Expression or BlockStatement: code to execute on defer

    def __repr__(self):
        return f"DeferStatement(code={self.code_block})"


class PatternStatement(Statement):
    """Pattern matching statement - Match expression against patterns
    
    pattern value {
      case 1 => print "one";
      case 2 => print "two";
      default => print "other";
    }
    """
    def __init__(self, expression, cases):
        self.expression = expression  # Expression: value to match
        self.cases = cases  # List[PatternCase]: pattern cases

    def __repr__(self):
        return f"PatternStatement(expr={self.expression}, cases={len(self.cases)} patterns)"


class PatternCase:
    """A single pattern case
    
    case pattern => action;
    """
    def __init__(self, pattern, action):
        self.pattern = pattern  # String or Expression: pattern to match
        self.action = action  # Expression or BlockStatement: action if matched

    def __repr__(self):
        return f"PatternCase(pattern={self.pattern}, action={self.action})"


# ADVANCED FEATURE STATEMENT NODES

class EnumStatement(Statement):
    """Enum statement - Type-safe enumerations
    
    enum Color {
      Red,
      Green,
      Blue
    }
    
    enum Status {
      Active = 1,
      Inactive = 2,
      Pending = 3
    }
    """
    def __init__(self, name, members):
        self.name = name  # String: enum name
        self.members = members  # List[EnumMember]: enum values

    def __repr__(self):
        return f"EnumStatement(name={self.name}, members={len(self.members)})"


class EnumMember:
    """A single enum member"""
    def __init__(self, name, value=None):
        self.name = name  # String: member name
        self.value = value  # Optional value (integer or string)

    def __repr__(self):
        return f"EnumMember({self.name}={self.value})"


class StreamStatement(Statement):
    """Stream statement - Event streaming and handling
    
    stream clicks as event => {
      print "Clicked: " + event.x + ", " + event.y;
    }
    
    stream api_responses as response => {
      handle_response(response);
    }
    """
    def __init__(self, stream_name, event_var, handler):
        self.stream_name = stream_name  # String: stream name
        self.event_var = event_var  # Identifier: event variable name
        self.handler = handler  # BlockStatement: event handler code

    def __repr__(self):
        return f"StreamStatement(stream={self.stream_name}, var={self.event_var})"


class WatchStatement(Statement):
    """Watch statement - Reactive state management
    
    watch {
      print("x changed to " + x);
    }
    
    watch user_name => {
      update_ui();
    }
    """
    def __init__(self, reaction, watched_expr=None):
        self.reaction = reaction  # BlockStatement: code to execute
        self.watched_expr = watched_expr  # Optional Expression: explicit dependency

    def __repr__(self):
        return f"WatchStatement(expr={self.watched_expr}, reaction={self.reaction})"

    def __repr__(self):
        return f"WatchStatement(watch={self.watched_expr})"


class LogStatement(Statement):
    """Log statement - Redirect output to file
    
    log > output.txt     # Write mode (overwrites on first write in scope)
    log >> output.txt    # Append mode (always appends)
    
    Supports any file extension: .txt, .py, .zx, .cpp, .rs, etc.
    Redirects subsequent print output to the specified file.
    """
    def __init__(self, filepath, append_mode=True):
        self.filepath = filepath  # Expression: path to log file
        self.append_mode = append_mode  # True for >>, False for >
    
    def __repr__(self):
        mode = ">>" if self.append_mode else ">"
        return f"LogStatement({mode} {self.filepath})"


class ImportLogStatement(Statement):
    """Import/Execute code from file (Hidden Layer)
    
    log << "helpers.zx"    # Import and execute Zexus code from file
    
    Creates a hidden layer where generated code is automatically loaded
    and executed in the current scope. Combines code generation with
    immediate execution without explicit eval_file() calls.
    
    Example:
        action generateHelpers {
            log >> "helpers.zx";
            print("action add(a, b) { return a + b; }");
        }
        generateHelpers();
        
        log << "helpers.zx";  // Auto-imports and executes
        let result = add(5, 10);  // Can use imported functions
    """
    def __init__(self, filepath):
        self.filepath = filepath  # Expression: path to file to import
    
    def __repr__(self):
        return f"ImportLogStatement(<< {self.filepath})"


# NEW: Capability-based security statements

class CapabilityStatement(Statement):
    """Capability definition statement - Define what entities can do
    
    capability read_file = {
      description: "Read file system",
      scope: "io",
      level: "restricted"
    };
    
    capability write_network = {
      description: "Write to network",
      scope: "io.network",
      level: "privileged"
    };
    """
    def __init__(self, name, definition=None):
        self.name = name  # Identifier: capability name
        self.definition = definition  # Dict with description, scope, level, etc.

    def __repr__(self):
        return f"CapabilityStatement(name={self.name})"


class GrantStatement(Statement):
    """Grant capabilities to an entity
    
    grant user1 {
      read_file,
      read_network
    };
    
    grant plugin_trusted capability(read_file);
    """
    def __init__(self, entity_name, capabilities):
        self.entity_name = entity_name  # Identifier: entity to grant to
        self.capabilities = capabilities  # List of capability names or FunctionCall nodes

    def __repr__(self):
        return f"GrantStatement(entity={self.entity_name}, capabilities={len(self.capabilities)})"


class RevokeStatement(Statement):
    """Revoke capabilities from an entity
    
    revoke user1 {
      write_file,
      write_network
    };
    
    revoke plugin_untrusted capability(read_file);
    """
    def __init__(self, entity_name, capabilities):
        self.entity_name = entity_name  # Identifier: entity to revoke from
        self.capabilities = capabilities  # List of capability names or FunctionCall nodes

    def __repr__(self):
        return f"RevokeStatement(entity={self.entity_name}, capabilities={len(self.capabilities)})"


# NEW: Pure function declarations

class PureFunctionStatement(Statement):
    """Declare a function as pure (referentially transparent)
    
    pure function add(a, b) {
      return a + b;
    }
    
    function multiply(x, y) {
      pure {
        return x * y;
      }
    }
    """
    def __init__(self, function_decl):
        self.function_decl = function_decl  # FunctionExpression or FunctionStatement

    def __repr__(self):
        return f"PureFunctionStatement(function={self.function_decl})"


class ImmutableStatement(Statement):
    """Declare a variable or parameter as immutable
    
    immutable const user = { name: "Alice", age: 30 };
    immutable let config = load_config();
    
    function process(immutable data) {
      // data cannot be modified
      print data.name;
    }
    """
    def __init__(self, target, value=None):
        self.target = target  # Identifier: variable to make immutable
        self.value = value  # Optional Expression: value to assign

    def __repr__(self):
        return f"ImmutableStatement(target={self.target})"


# NEW: Data validation and sanitization

class ValidateStatement(Statement):
    """Validate data against a schema
    
    validate user_input, {
      name: string,
      email: email,
      age: number(18, 120)
    };
    
    validate(request.body, schema_user);
    """
    def __init__(self, data, schema, options=None):
        self.data = data  # Expression: value to validate
        self.schema = schema  # Dict or Identifier: validation schema
        self.options = options  # Optional: validation options

    def __repr__(self):
        return f"ValidateStatement(data={self.data})"


class SanitizeStatement(Statement):
    """Sanitize untrusted input
    
    sanitize user_input, {
      encoding: "html",
      rules: ["remove_scripts", "remove_events"]
    };
    
    let clean_data = sanitize(request.body, encoding="url");
    """
    def __init__(self, data, rules=None, encoding=None):
        self.data = data  # Expression: value to sanitize
        self.rules = rules  # Optional: list of sanitization rules
        self.encoding = encoding  # Optional: encoding type (html, sql, url, javascript)

    def __repr__(self):
        return f"SanitizeStatement(data={self.data})"


class InjectStatement(Statement):
    """Dependency injection statement
    
    inject DatabaseAPI;
    inject max_retries;
    
    Used for dependency injection pattern where external
    dependencies are injected at runtime.
    """
    def __init__(self, dependency):
        self.dependency = dependency  # Identifier: dependency name

    def __repr__(self):
        return f"InjectStatement(dependency={self.dependency})"


# NEW: Complexity & Large Project Management

class InterfaceStatement(Statement):
    """Formal interface/type class definition
    
    interface Drawable {
        draw(canvas);
        get_bounds();
    };
    
    interface Serializable {
        to_string();
        from_string(str);
    };
    """
    def __init__(self, name, methods=None, properties=None, extends=None):
        self.name = name  # Identifier: interface name
        self.methods = methods or []  # List of method declarations
        self.properties = properties or {}  # Dict of property names -> types
        self.extends = extends or []  # List of interface names this extends

    def __repr__(self):
        return f"InterfaceStatement(name={self.name}, methods={len(self.methods)})"


class TypeAliasStatement(Statement):
    """Type alias definition for complex types
    
    type_alias UserID = integer;
    type_alias Point = { x: float, y: float };
    type_alias Handler = function(request) -> response;
    """
    def __init__(self, name, base_type):
        self.name = name  # Identifier: alias name
        self.base_type = base_type  # Expression or type name

    def __repr__(self):
        return f"TypeAliasStatement(name={self.name}, base_type={self.base_type})"


class ModuleStatement(Statement):
    """Module definition for code organization
    
    module database {
        internal function connect_db(path) { ... }
        public function query(sql) { ... }
    }
    
    module users {
        public function get_user(id) { ... }
        internal function validate_user(user) { ... }
    }
    """
    def __init__(self, name, body=None, parent=None):
        self.name = name  # Identifier: module name
        self.body = body  # BlockStatement: module body
        self.parent = parent  # Parent module (if nested)

    def __repr__(self):
        return f"ModuleStatement(name={self.name})"


class PackageStatement(Statement):
    """Package definition for high-level organization
    
    package myapp.database {
        module connection { ... }
        module query { ... }
    }
    """
    def __init__(self, name, body=None):
        self.name = name  # Identifier or dotted name: package name
        self.body = body  # BlockStatement: package body

    def __repr__(self):
        return f"PackageStatement(name={self.name})"


class UsingStatement(Statement):
    """Resource management with automatic cleanup (RAII pattern)
    
    using(file = open("/path/to/file.txt")) {
        content = file.read();
        process(content);
        // file is automatically closed
    }
    
    using(connection = db.connect()) {
        result = connection.query("SELECT * FROM users");
        // connection is automatically closed
    }
    """
    def __init__(self, resource_name, resource_expr, body):
        self.resource_name = resource_name  # Identifier: name for the resource
        self.resource_expr = resource_expr  # Expression: acquires the resource
        self.body = body  # BlockStatement: code using the resource

    def __repr__(self):
        return f"UsingStatement(resource={self.resource_name})"


class VisibilityModifier(Statement):
    """Visibility modifier for module members
    
    public function get_data() { ... }
    internal function helper() { ... }
    protected function submodule_only() { ... }
    """
    def __init__(self, visibility, statement):
        self.visibility = visibility  # String: "public", "internal", "protected"
        self.statement = statement  # The statement being modified

    def __repr__(self):
        return f"VisibilityModifier({self.visibility}, {self.statement})"


class ChannelStatement(Statement):
    """Channel declaration for message passing between concurrent tasks
    
    channel<integer> numbers;
    channel<string> messages;
    channel<{"id": integer, "name": string}> user_updates;
    """
    def __init__(self, name, element_type=None, capacity=None):
        self.name = name  # Identifier: channel name
        self.element_type = element_type  # Expression: type of elements
        self.capacity = capacity  # Optional: buffer capacity (None = unbuffered)

    def __repr__(self):
        return f"ChannelStatement({self.name}, type={self.element_type})"


class SendStatement(Statement):
    """Send value to a channel
    
    send(channel, value);
    send(data_channel, {"id": 1, "name": "Alice"});
    """
    def __init__(self, channel_expr, value_expr):
        self.channel_expr = channel_expr  # Expression: channel to send to
        self.value_expr = value_expr  # Expression: value to send

    def __repr__(self):
        return f"SendStatement(channel={self.channel_expr}, value={self.value_expr})"


class ReceiveStatement(Statement):
    """Receive value from a channel (blocking)
    
    value = receive(channel);
    user = receive(user_updates);
    """
    def __init__(self, channel_expr, target=None):
        self.channel_expr = channel_expr  # Expression: channel to receive from
        self.target = target  # Optional Identifier: variable to bind received value

    def __repr__(self):
        return f"ReceiveStatement(channel={self.channel_expr})"


class AtomicStatement(Statement):
    """Atomic operation - indivisible unit of concurrent code
    
    atomic(counter = counter + 1);
    atomic {
        x = x + 1;
        y = y + 1;
    };
    """
    def __init__(self, body=None, expr=None):
        self.body = body  # BlockStatement: atomic block
        self.expr = expr  # Expression: atomic expression (single operation)
        # Note: exactly one of body or expr should be non-None

    def __repr__(self):
        return f"AtomicStatement(body={self.body}, expr={self.expr})"


# ============================================================================
# BLOCKCHAIN & SMART CONTRACT AST NODES
# ============================================================================

class LedgerStatement(Statement):
    """Ledger statement - Immutable, versioned state storage
    
    ledger balances;
    ledger accounts = {};
    ledger state_root;
    """
    def __init__(self, name, initial_value=None):
        self.name = name  # Identifier: ledger variable name
        self.initial_value = initial_value  # Optional Expression: initial value

    def __repr__(self):
        return f"LedgerStatement(name={self.name}, initial={self.initial_value})"


class StateStatement(Statement):
    """State statement - Mutable state within contracts
    
    state counter = 0;
    state owner = TX.caller;
    state locked = false;
    """
    def __init__(self, name, initial_value=None):
        self.name = name  # Identifier: state variable name
        self.initial_value = initial_value  # Optional Expression: initial value

    def __repr__(self):
        return f"StateStatement(name={self.name}, initial={self.initial_value})"


class ContractStatement(Statement):
    """Contract statement - Smart contract definition
    
    contract Token {
        state balances = {};
        state total_supply = 0;
        
        action mint(recipient, amount) limit 5000 {
            require(TX.caller == owner, "Only owner can mint");
            balances[recipient] = balances[recipient] + amount;
            total_supply = total_supply + amount;
        }
    }
    
    contract QuantumCrypto implements QuantumResistantCrypto { ... }
    """
    def __init__(self, name, body, modifiers=None, implements=None):
        self.name = name  # Identifier: contract name
        self.body = body  # BlockStatement: contract body (state vars and actions)
        self.modifiers = modifiers or []  # List of modifiers
        self.implements = implements  # Optional protocol name that contract implements

    def __repr__(self):
        impl_str = f", implements={self.implements}" if self.implements else ""
        return f"ContractStatement(name={self.name}, modifiers={self.modifiers}{impl_str})"


class RevertStatement(Statement):
    """Revert statement - Rollback transaction
    
    revert();
    revert("Insufficient balance");
    revert("Unauthorized: " + TX.caller);
    """
    def __init__(self, reason=None):
        self.reason = reason  # Optional Expression: revert reason message

    def __repr__(self):
        return f"RevertStatement(reason={self.reason})"


class RequireStatement(Statement):
    """Require statement - Prerequisites, dependencies, and resource requirements
    
    Basic requirement:
        require(balance >= amount);
        require(TX.caller == owner, "Only owner");
        require(balance >= amount, "Insufficient funds");
    
    With conditional tolerance logic:
        require balance >= 0.1 {
            // Tolerance logic - allow exceptions
            if (user.isVIP) {
                print "VIP user - waiving requirement";
            } else {
                return false;
            }
        }
    
    File/module dependencies:
        require "math_plus.zx" imported, "Math module required";
        require module "database" available, "Database unavailable";
    
    Resource requirements:
        require:balance amount >= minimum, "Insufficient funds";
        require:gas available >= needed, "Insufficient gas";
    
    Multiple conditions with fallback:
        require amount >= 100 || (user.subscribed && amount >= 10), "Minimum not met";
    
    Prerequisite checking:
        require:prereq user.authenticated, "Login required";
        require:prereq file.exists("config.json"), "Config file missing";
    """
    def __init__(self, condition, message=None, tolerance_block=None, requirement_type=None,
                 resource_name=None, minimum_value=None, file_path=None, module_name=None,
                 fallback_condition=None, prereq_list=None):
        # Basic form
        self.condition = condition          # Expression: condition that must be true
        self.message = message              # Optional Expression: error message if false
        
        # Conditional tolerance
        self.tolerance_block = tolerance_block  # BlockStatement: tolerance/fallback logic
        
        # Requirement types
        self.requirement_type = requirement_type  # 'balance', 'gas', 'prereq', 'file', 'module'
        
        # Resource requirements
        self.resource_name = resource_name  # Resource identifier (balance, gas, etc.)
        self.minimum_value = minimum_value  # Minimum required value
        
        # File/module dependencies
        self.file_path = file_path          # Required file path
        self.module_name = module_name      # Required module name
        
        # Fallback/alternative conditions
        self.fallback_condition = fallback_condition  # Alternative condition if primary fails
        
        # Prerequisites list
        self.prereq_list = prereq_list or []  # List of prerequisites

    def __repr__(self):
        if self.requirement_type:
            return f"RequireStatement(type={self.requirement_type}, condition={self.condition})"
        return f"RequireStatement(condition={self.condition}, msg={self.message})"


class LimitStatement(Statement):
    """Limit statement - Gas/resource limit for actions
    
    action transfer() limit 1000 { ... }
    limit 5000;  // Set limit for next operation
    """
    def __init__(self, amount):
        self.amount = amount  # IntegerLiteral or Expression: gas limit amount

    def __repr__(self):
        return f"LimitStatement(amount={self.amount})"


# BLOCKCHAIN EXPRESSION NODES

class TXExpression(Expression):
    """TX expression - Transaction context access
    
    TX.caller
    TX.timestamp
    TX.block_hash
    TX.gas_remaining
    """
    def __init__(self, property_name):
        self.property_name = property_name  # String: property being accessed

    def __repr__(self):
        return f"TXExpression(property={self.property_name})"


class HashExpression(Expression):
    """Hash expression - Cryptographic hashing
    
    hash(data, "SHA256")
    hash(message, "KECCAK256")
    hash(password, "SHA512")
    """
    def __init__(self, data, algorithm):
        self.data = data  # Expression: data to hash
        self.algorithm = algorithm  # Expression: hash algorithm name

    def __repr__(self):
        return f"HashExpression(data={self.data}, algorithm={self.algorithm})"


class SignatureExpression(Expression):
    """Signature expression - Digital signature creation
    
    signature(message, private_key)
    signature(data, key, "ECDSA")
    """
    def __init__(self, data, private_key, algorithm=None):
        self.data = data  # Expression: data to sign
        self.private_key = private_key  # Expression: private key
        self.algorithm = algorithm  # Optional Expression: signature algorithm

    def __repr__(self):
        return f"SignatureExpression(data={self.data}, algorithm={self.algorithm})"


class VerifySignatureExpression(Expression):
    """Verify signature expression - Signature verification
    
    verify_sig(data, signature, public_key)
    verify_sig(message, sig, pub_key, "ECDSA")
    """
    def __init__(self, data, signature, public_key, algorithm=None):
        self.data = data  # Expression: original data
        self.signature = signature  # Expression: signature to verify
        self.public_key = public_key  # Expression: public key
        self.algorithm = algorithm  # Optional Expression: signature algorithm

    def __repr__(self):
        return f"VerifySignatureExpression(data={self.data}, algorithm={self.algorithm})"


class GasExpression(Expression):
    """Gas expression - Access gas tracking information
    
    gas              // Current gas tracker object
    gas.used         // Gas consumed so far
    gas.remaining    // Gas still available
    gas.limit        // Total gas limit
    """
    def __init__(self, property_name=None):
        self.property_name = property_name  # Optional String: property being accessed

    def __repr__(self):
        return f"GasExpression(property={self.property_name})"


def attach_modifiers(node, modifiers):
    """Attach modifiers to an AST node (best-effort).

    Many AST constructors do not accept modifiers; this helper sets
    a `modifiers` attribute on the node for downstream passes.
    """
    try:
        if modifiers:
            setattr(node, 'modifiers', list(modifiers))
        else:
            setattr(node, 'modifiers', [])
    except Exception:
        pass
    return node
class ProtocolStatement(Statement):
    """Protocol declaration - interface/trait definition
    
    protocol Transferable {
        action transfer(to, amount)
        action balance() -> int
    }
    """
    def __init__(self, name, methods):
        self.name = name                    # Identifier
        self.methods = methods              # List of method signatures

    def __repr__(self):
        return f"ProtocolStatement(name={self.name}, methods={len(self.methods)})"


class PersistentStatement(Statement):
    """Persistent storage declaration within contracts
    
    persistent storage balances: map
    persistent storage owner: string
    """
    def __init__(self, name, type_annotation=None, initial_value=None):
        self.name = name                    # Identifier
        self.type_annotation = type_annotation  # Optional type
        self.initial_value = initial_value  # Optional initial value

    def __repr__(self):
        return f"PersistentStatement(name={self.name})"


class EmitStatement(Statement):
    """Emit event statement
    
    emit Transfer(from, to, amount);
    emit StateChange("balance_updated", new_balance);
    """
    def __init__(self, event_name, arguments=None):
        self.event_name = event_name        # Event name (Identifier or string)
        self.arguments = arguments or []    # List of arguments

    def __repr__(self):
        return f"EmitStatement(event={self.event_name}, args={len(self.arguments)})"


class ModifierDeclaration(Statement):
    """Modifier declaration - reusable function modifier
    
    modifier onlyOwner {
        require(TX.caller == owner, "Not owner");
    }
    
    action withdraw() modifier onlyOwner { ... }
    """
    def __init__(self, name, parameters, body):
        self.name = name                    # Modifier name (Identifier)
        self.parameters = parameters or []  # List of parameters
        self.body = body                    # BlockStatement

    def __repr__(self):
        return f"ModifierDeclaration(name={self.name}, params={len(self.parameters)})"
