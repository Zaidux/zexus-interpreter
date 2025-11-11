"""
Advanced Security and Contract Features for Zexus

This module implements entity, verify, contract, and protect statements
providing a powerful security framework for Zexus programs.
"""

from .object import (
    Environment, Map, String, Integer, Boolean as BooleanObj, 
    Builtin, List, Null, EvaluationError as ObjectEvaluationError
)


class SecurityContext:
    """Global security context for enforcement"""
    def __init__(self):
        self.verify_checks = {}      # Registered verification checks
        self.protections = {}        # Active protection rules
        self.contracts = {}          # Deployed contracts
        self.middlewares = {}        # Registered middleware
        self.auth_config = None      # Global auth configuration
        self.cache_store = {}        # Caching store
        
    def register_verify_check(self, name, check_func):
        """Register a verification check function"""
        self.verify_checks[name] = check_func
        
    def register_protection(self, name, rules):
        """Register a protection rule set"""
        self.protections[name] = rules
        
    def register_contract(self, name, contract):
        """Register a smart contract"""
        self.contracts[name] = contract
        
    def check_protection(self, target_name, context_data):
        """Check if target access is protected"""
        if target_name not in self.protections:
            return True  # No protection = allowed
            
        rules = self.protections[target_name]
        
        # Check authentication requirement
        if rules.get("auth_required", False):
            if not context_data.get("authenticated"):
                return False
                
        # Check rate limiting
        rate_limit = rules.get("rate_limit")
        if rate_limit:
            # Simple rate limit check (production would use timestamp tracking)
            if context_data.get("request_count", 0) > rate_limit:
                return False
                
        # Check IP restrictions
        client_ip = context_data.get("client_ip")
        if client_ip:
            blocked_ips = rules.get("blocked_ips", [])
            if _is_ip_in_list(client_ip, blocked_ips):
                return False
                
            allowed_ips = rules.get("allowed_ips")
            if allowed_ips and not _is_ip_in_list(client_ip, allowed_ips):
                return False
        
        return True


# Global security context
_security_context = SecurityContext()


def get_security_context():
    """Get the global security context"""
    return _security_context


def _is_ip_in_list(ip, ip_list):
    """Check if IP matches CIDR or exact match in list"""
    for pattern in ip_list:
        if "/" in pattern:  # CIDR notation
            # Simplified CIDR check (would need proper IP math for production)
            network_part = pattern.split("/")[0]
            if ip.startswith(network_part.rsplit(".", 1)[0]):
                return True
        elif ip == pattern:  # Exact match
            return True
    return False


# ===============================================
# ENTITY SYSTEM - Object-Oriented Data Structures
# ===============================================

class EntityDefinition:
    """Represents an entity definition with properties and methods"""
    
    def __init__(self, name, properties, methods=None, parent=None):
        self.name = name
        self.properties = properties  # {prop_name: {type, default_value}}
        self.methods = methods or {}   # {method_name: Action}
        self.parent = parent          # Parent entity (inheritance)
        
    def create_instance(self, values=None):
        """Create an instance of this entity"""
        instance = EntityInstance(self, values or {})
        return instance
        
    def get_all_properties(self):
        """Get all properties including inherited ones"""
        props = dict(self.properties)
        if self.parent:
            props.update(self.parent.get_all_properties())
        return props


class EntityInstance:
    """Represents an instance of an entity"""
    
    def __init__(self, entity_def, values):
        self.entity_def = entity_def
        self.data = values or {}
        self._validate_properties()
        
    def _validate_properties(self):
        """Validate that all required properties are present"""
        all_props = self.entity_def.get_all_properties()
        for prop_name, prop_config in all_props.items():
            if prop_name not in self.data:
                if "default_value" in prop_config:
                    self.data[prop_name] = prop_config["default_value"]
                    
    def get(self, property_name):
        """Get property value"""
        return self.data.get(property_name)
        
    def set(self, property_name, value):
        """Set property value"""
        if property_name not in self.entity_def.get_all_properties():
            raise ValueError(f"Unknown property: {property_name}")
        self.data[property_name] = value
        
    def to_dict(self):
        """Convert to dictionary"""
        return self.data


# ===============================================
# VERIFICATION SYSTEM - Security Checks
# ===============================================

class VerificationCheck:
    """Represents a single verification condition"""
    
    def __init__(self, name, condition_func, error_message=""):
        self.name = name
        self.condition_func = condition_func
        self.error_message = error_message or f"Verification check '{name}' failed"
        
    def verify(self, context_data):
        """Execute verification check"""
        try:
            result = self.condition_func(context_data)
            return (result, None) if result else (False, self.error_message)
        except Exception as e:
            return (False, str(e))


class VerifyWrapper:
    """Wraps a function with verification checks"""
    
    def __init__(self, target_func, checks, error_handler=None):
        self.target_func = target_func
        self.checks = checks  # List of VerificationCheck
        self.error_handler = error_handler
        
    def execute(self, args, context_data=None, env=None):
        """Execute target function with verification"""
        context_data = context_data or {}
        
        # Run all verification checks
        for check in self.checks:
            is_valid, error_msg = check.verify(context_data)
            if not is_valid:
                if self.error_handler:
                    return self.error_handler(error_msg, context_data, env)
                else:
                    return ObjectEvaluationError(error_msg)
                    
        # All checks passed, execute target
        return self.target_func(args, env)


# ===============================================
# CONTRACT SYSTEM - Blockchain State & Logic
# ===============================================

class ContractStorage:
    """Persistent storage for contract state"""
    
    def __init__(self):
        self.state = {}
        self.transaction_log = []
        
    def get(self, key):
        """Get value from storage"""
        return self.state.get(key)
        
    def set(self, key, value):
        """Set value in storage"""
        self.state[key] = value
        self._log_transaction("SET", key, value)
        
    def delete(self, key):
        """Delete value from storage"""
        if key in self.state:
            del self.state[key]
            self._log_transaction("DELETE", key, None)
            
    def _log_transaction(self, op, key, value):
        """Log transaction for audit trail"""
        self.transaction_log.append({
            "operation": op,
            "key": key,
            "value": value,
            "timestamp": _get_timestamp()
        })
        
    def get_transaction_log(self):
        """Get all transactions"""
        return self.transaction_log


class SmartContract:
    """Represents a smart contract with persistent storage"""
    
    def __init__(self, name, storage_vars, actions, blockchain_config=None):
        self.name = name
        self.storage_vars = storage_vars or {}
        self.actions = actions or {}
        self.blockchain_config = blockchain_config or {}
        self.storage = ContractStorage()
        self.is_deployed = False
        
    def deploy(self):
        """Deploy the contract"""
        self.is_deployed = True
        # Initialize storage with default values
        for var_name, var_config in self.storage_vars.items():
            if "initial_value" in var_config:
                self.storage.set(var_name, var_config["initial_value"])
                
    def execute_action(self, action_name, args, context, env=None):
        """Execute a contract action"""
        if not self.is_deployed:
            return ObjectEvaluationError(f"Contract {self.name} not deployed")
            
        if action_name not in self.actions:
            return ObjectEvaluationError(f"Unknown action: {action_name}")
            
        action = self.actions[action_name]
        # Action execution would be delegated to evaluator
        return action
        
    def get_state(self):
        """Get current contract state"""
        return self.storage.state
        
    def get_balance(self, account=None):
        """Get balance from contract storage"""
        if account:
            return self.storage.get(f"balance_{account}") or Integer(0)
        return self.storage.get("balance") or Integer(0)


# ===============================================
# PROTECTION SYSTEM - Security Guardrails
# ===============================================

class ProtectionRule:
    """Represents a single protection rule"""
    
    def __init__(self, name, rule_config):
        self.name = name
        self.config = rule_config
        
    def evaluate(self, context_data):
        """Evaluate if protection allows access"""
        # Rate limiting
        if self.config.get("rate_limit"):
            if context_data.get("request_count", 0) > self.config["rate_limit"]:
                return False, "Rate limit exceeded"
                
        # Authentication requirement
        if self.config.get("auth_required", False):
            if not context_data.get("user_authenticated"):
                return False, "Authentication required"
                
        # Password strength
        if self.config.get("min_password_strength"):
            strength = context_data.get("password_strength", "weak")
            required = self.config["min_password_strength"]
            strength_levels = {"weak": 0, "medium": 1, "strong": 2, "very_strong": 3}
            if strength_levels.get(strength, 0) < strength_levels.get(required, 0):
                return False, f"Password must be {required}"
                
        # Session timeout
        if self.config.get("session_timeout"):
            session_age = context_data.get("session_age_seconds", 0)
            if session_age > self.config["session_timeout"]:
                return False, "Session expired"
                
        # HTTPS requirement
        if self.config.get("require_https", False):
            if not context_data.get("is_https", False):
                return False, "HTTPS required"
                
        return True, None


class ProtectionPolicy:
    """Represents a set of protection rules for a target"""
    
    def __init__(self, target_name, rules, enforcement_level="strict"):
        self.target_name = target_name
        self.rules = {}  # {rule_name: ProtectionRule}
        self.enforcement_level = enforcement_level  # strict, warn, audit
        
        if isinstance(rules, dict):
            for rule_name, rule_config in rules.items():
                self.add_rule(rule_name, rule_config)
                
    def add_rule(self, rule_name, rule_config):
        """Add a protection rule"""
        self.rules[rule_name] = ProtectionRule(rule_name, rule_config)
        
    def check_access(self, context_data):
        """Check if access is allowed"""
        violations = []
        
        for rule_name, rule in self.rules.items():
            allowed, error_msg = rule.evaluate(context_data)
            if not allowed:
                violations.append((rule_name, error_msg))
                
        if violations:
            if self.enforcement_level == "strict":
                return False, violations[0][1]
            elif self.enforcement_level == "warn":
                return True, violations  # Allow but warn
            elif self.enforcement_level == "audit":
                return True, violations  # Allow but log
                
        return True, None


# ===============================================
# MIDDLEWARE SYSTEM - Request/Response Processing
# ===============================================

class Middleware:
    """Represents a middleware handler"""
    
    def __init__(self, name, handler_func):
        self.name = name
        self.handler_func = handler_func
        
    def execute(self, request, response, env=None):
        """Execute middleware"""
        try:
            return self.handler_func((request, response), env)
        except Exception as e:
            return ObjectEvaluationError(f"Middleware error: {str(e)}")


class MiddlewareChain:
    """Executes a chain of middleware"""
    
    def __init__(self):
        self.middlewares = []
        
    def add_middleware(self, middleware):
        """Add middleware to chain"""
        self.middlewares.append(middleware)
        
    def execute(self, request, response, env=None):
        """Execute all middleware in order"""
        for middleware in self.middlewares:
            result = middleware.execute(request, response, env)
            if isinstance(result, ObjectEvaluationError):
                return result
            # Check if middleware set response to stop chain
            if response.get("_stop_chain"):
                break
        return response


# ===============================================
# AUTHENTICATION & AUTHORIZATION
# ===============================================

class AuthConfig:
    """Authentication configuration"""
    
    def __init__(self, config_data=None):
        self.provider = "oauth2"
        self.scopes = ["read", "write"]
        self.token_expiry = 3600
        self.refresh_enabled = True
        
        if config_data:
            self.provider = config_data.get("provider", self.provider)
            self.scopes = config_data.get("scopes", self.scopes)
            self.token_expiry = config_data.get("token_expiry", self.token_expiry)
            self.refresh_enabled = config_data.get("refresh_enabled", self.refresh_enabled)
            
    def validate_token(self, token):
        """Validate a token"""
        # In production, this would validate with OAuth provider
        return True
        
    def is_token_expired(self, token_data):
        """Check if token is expired"""
        import time
        if "issued_at" not in token_data:
            return True
        age = time.time() - token_data["issued_at"]
        return age > self.token_expiry


# ===============================================
# CACHING SYSTEM
# ===============================================

class CachePolicy:
    """Cache policy for a function"""
    
    def __init__(self, ttl=3600, key_func=None, invalidate_on=None):
        self.ttl = ttl  # Time to live in seconds
        self.key_func = key_func or (lambda x: str(x))  # Function to generate cache key
        self.invalidate_on = invalidate_on or []  # Events that invalidate cache
        self.cache = {}
        self.timestamps = {}
        
    def get(self, key):
        """Get cached value"""
        import time
        if key not in self.cache:
            return None
        
        # Check if expired
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
            
        return self.cache[key]
        
    def set(self, key, value):
        """Cache a value"""
        import time
        self.cache[key] = value
        self.timestamps[key] = time.time()
        
    def invalidate(self, key=None):
        """Invalidate cache entry or entire cache"""
        if key is None:
            self.cache.clear()
            self.timestamps.clear()
        elif key in self.cache:
            del self.cache[key]
            del self.timestamps[key]


# ===============================================
# RATE LIMITING
# ===============================================

class RateLimiter:
    """Rate limiter for throttling"""
    
    def __init__(self, requests_per_minute=100, burst_size=10, per_user=False):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.per_user = per_user
        self.request_counts = {}  # {user_id: count}
        self.burst_counts = {}    # {user_id: burst_count}
        
    def allow_request(self, user_id=None):
        """Check if request is allowed"""
        if not self.per_user:
            user_id = "global"
            
        current_count = self.request_counts.get(user_id, 0)
        burst_count = self.burst_counts.get(user_id, 0)
        
        # Check rate limit
        if current_count >= self.requests_per_minute:
            return False, "Rate limit exceeded"
            
        # Check burst limit
        if burst_count >= self.burst_size:
            return False, "Burst limit exceeded"
            
        self.request_counts[user_id] = current_count + 1
        self.burst_counts[user_id] = burst_count + 1
        
        return True, None
        
    def reset(self, user_id=None):
        """Reset rate limit counters"""
        if user_id:
            if user_id in self.request_counts:
                del self.request_counts[user_id]
            if user_id in self.burst_counts:
                del self.burst_counts[user_id]
        else:
            self.request_counts.clear()
            self.burst_counts.clear()


# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def _get_timestamp():
    """Get current timestamp"""
    import time
    return int(time.time() * 1000)


def export_security_to_environment(env):
    """Export security functions to environment"""
    # Entity creation
    def make_entity(entity_def, values=None):
        if isinstance(entity_def, EntityDefinition):
            return entity_def.create_instance(values)
        return ObjectEvaluationError("Invalid entity definition")
    
    # Verification
    def make_verify(target, checks, error_handler=None):
        return VerifyWrapper(target, checks, error_handler)
    
    # Contract deployment
    def deploy_contract(contract):
        if isinstance(contract, SmartContract):
            contract.deploy()
            return contract
        return ObjectEvaluationError("Invalid contract")
    
    env.set("entity", Builtin(make_entity, "entity"))
    env.set("verify", Builtin(make_verify, "verify"))
    env.set("contract", Builtin(deploy_contract, "contract"))
