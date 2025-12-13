# src/zexus/policy_engine.py
"""
Policy-as-Code Engine for Zexus PROTECT Feature
Implements declarative security policy injection with VERIFY and RESTRICT
"""

from typing import Dict, List, Any, Callable, Optional
from .object import Object, Boolean as BooleanObj, String, Integer, NULL, EvaluationError


# ===============================================
# POLICY RULE TYPES
# ===============================================

class PolicyRule:
    """Base class for policy rules"""
    
    def __init__(self, description: str = ""):
        self.description = description
    
    def evaluate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """
        Evaluate the policy rule
        Returns: (success: bool, message: str)
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


class VerifyRule(PolicyRule):
    """VERIFY rule - checks a boolean condition"""
    
    def __init__(self, condition_fn: Callable, description: str = "Verification check"):
        super().__init__(description)
        self.condition_fn = condition_fn
    
    def evaluate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        try:
            result = self.condition_fn(context)
            success = result.value if isinstance(result, BooleanObj) else bool(result)
            message = self.description if not success else ""
            return success, message
        except Exception as e:
            return False, f"Verification failed: {str(e)}"


class RestrictRule(PolicyRule):
    """RESTRICT rule - applies constraints to data"""
    
    def __init__(self, field_name: str, constraints: List[Callable], description: str = ""):
        super().__init__(description)
        self.field_name = field_name
        self.constraints = constraints
    
    def evaluate(self, context: Dict[str, Any]) -> tuple[bool, str]:
        value = context.get(self.field_name)
        
        if value is None:
            return False, f"Field '{self.field_name}' not found in context"
        
        for constraint in self.constraints:
            try:
                result = constraint(value, context)
                passed = result.value if isinstance(result, BooleanObj) else bool(result)
                if not passed:
                    return False, f"Constraint failed for '{self.field_name}'"
            except Exception as e:
                return False, f"Constraint evaluation error: {str(e)}"
        
        return True, ""


# ===============================================
# PROTECTION POLICY
# ===============================================

class ProtectionPolicy:
    """Represents a complete protection policy for a target"""
    
    def __init__(self, target_name: str, enforcement_level: str = "strict"):
        self.target_name = target_name
        self.enforcement_level = enforcement_level  # "strict", "warn", "audit", "permissive"
        self.rules: List[PolicyRule] = []
        self.middleware_chain: List[Callable] = []
        self.on_violation: Optional[Callable] = None
        self.audit_log: List[Dict[str, Any]] = []
    
    def add_rule(self, rule: PolicyRule):
        """Add a policy rule"""
        self.rules.append(rule)
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to the protection chain"""
        self.middleware_chain.append(middleware)
    
    def set_violation_handler(self, handler: Callable):
        """Set handler for policy violations"""
        self.on_violation = handler
    
    def evaluate(self, context: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Evaluate all policy rules
        Returns: (all_passed: bool, violation_messages: List[str])
        """
        violations = []
        
        for rule in self.rules:
            passed, message = rule.evaluate(context)
            
            if not passed:
                violations.append(message)
                
                # Log the violation
                self._audit_violation(rule, message, context)
                
                # Handle based on enforcement level
                if self.enforcement_level == "strict":
                    # Strict mode: fail immediately
                    return False, violations
                elif self.enforcement_level == "warn":
                    # Warn mode: continue but collect violations
                    print(f"⚠️  Policy Warning ({self.target_name}): {message}")
                elif self.enforcement_level == "audit":
                    # Audit mode: log only, allow execution
                    pass
                # permissive mode: do nothing
        
        # All rules passed or enforcement allows it
        all_passed = len(violations) == 0
        return all_passed, violations
    
    def execute_middleware(self, context: Dict[str, Any], original_fn: Callable):
        """Execute middleware chain"""
        # Build middleware chain
        def final_handler():
            return original_fn(context)
        
        # Wrap with each middleware (reverse order for proper nesting)
        handler = final_handler
        for middleware in reversed(self.middleware_chain):
            current_handler = handler
            handler = lambda ctx=context, mw=middleware, h=current_handler: mw(ctx, h)
        
        # Execute the chain
        return handler()
    
    def _audit_violation(self, rule: PolicyRule, message: str, context: Dict[str, Any]):
        """Log a policy violation"""
        import time
        self.audit_log.append({
            'timestamp': time.time(),
            'target': self.target_name,
            'rule': rule.description,
            'message': message,
            'context_keys': list(context.keys()),
            'enforcement_level': self.enforcement_level
        })
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log"""
        return self.audit_log


# ===============================================
# POLICY REGISTRY
# ===============================================

class PolicyRegistry:
    """Global registry for protection policies"""
    
    def __init__(self):
        self.policies: Dict[str, ProtectionPolicy] = {}
        self.protected_functions: Dict[str, Callable] = {}
    
    def register_policy(self, target_name: str, policy: ProtectionPolicy):
        """Register a protection policy for a target"""
        self.policies[target_name] = policy
    
    def get_policy(self, target_name: str) -> Optional[ProtectionPolicy]:
        """Get policy for a target"""
        return self.policies.get(target_name)
    
    def protect_function(self, func_name: str, original_fn: Callable, policy: ProtectionPolicy):
        """Wrap a function with policy enforcement"""
        self.protected_functions[func_name] = original_fn
        self.register_policy(func_name, policy)
    
    def execute_protected(self, func_name: str, context: Dict[str, Any]):
        """Execute a protected function with policy checks"""
        policy = self.get_policy(func_name)
        original_fn = self.protected_functions.get(func_name)
        
        if not policy or not original_fn:
            raise ValueError(f"Function '{func_name}' not registered for protection")
        
        # Evaluate policy
        passed, violations = policy.evaluate(context)
        
        if not passed and policy.enforcement_level == "strict":
            error_msg = "; ".join(violations)
            if policy.on_violation:
                return policy.on_violation(violations, context)
            raise EvaluationError(f"Policy violation for {func_name}: {error_msg}")
        
        # Execute middleware chain
        return policy.execute_middleware(context, original_fn)
    
    def list_protected_targets(self) -> List[str]:
        """List all protected targets"""
        return list(self.policies.keys())
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of all audit logs"""
        total_violations = 0
        by_target = {}
        
        for target_name, policy in self.policies.items():
            violations = policy.get_audit_log()
            total_violations += len(violations)
            by_target[target_name] = len(violations)
        
        return {
            'total_violations': total_violations,
            'by_target': by_target,
            'protected_targets': len(self.policies)
        }


# Global policy registry
_policy_registry = PolicyRegistry()


def get_policy_registry() -> PolicyRegistry:
    """Get the global policy registry"""
    return _policy_registry


# ===============================================
# POLICY BUILDER (DSL Support)
# ===============================================

class PolicyBuilder:
    """Builder for creating policies from PROTECT blocks"""
    
    def __init__(self, target_name: str):
        self.policy = ProtectionPolicy(target_name)
    
    def add_verify_condition(self, condition_fn: Callable, description: str = ""):
        """Add a VERIFY rule"""
        rule = VerifyRule(condition_fn, description)
        self.policy.add_rule(rule)
        return self
    
    def add_restrict_constraint(self, field_name: str, constraints: List[Callable], description: str = ""):
        """Add a RESTRICT rule"""
        rule = RestrictRule(field_name, constraints, description)
        self.policy.add_rule(rule)
        return self
    
    def set_enforcement(self, level: str):
        """Set enforcement level"""
        self.policy.enforcement_level = level
        return self
    
    def with_middleware(self, middleware: Callable):
        """Add middleware"""
        self.policy.add_middleware(middleware)
        return self
    
    def on_violation(self, handler: Callable):
        """Set violation handler"""
        self.policy.set_violation_handler(handler)
        return self
    
    def build(self) -> ProtectionPolicy:
        """Build and return the policy"""
        return self.policy


# ===============================================
# BUILT-IN POLICY CONSTRAINTS
# ===============================================

def length_constraint(min_len: int = 0, max_len: int = float('inf')):
    """Create a length constraint for strings"""
    def check(value: Object, context: Dict[str, Any]) -> BooleanObj:
        if isinstance(value, String):
            length = len(value.value)
            return BooleanObj(min_len <= length <= max_len)
        return BooleanObj(False)
    return check


def contains_constraint(substring: str):
    """Create a contains constraint for strings"""
    def check(value: Object, context: Dict[str, Any]) -> BooleanObj:
        if isinstance(value, String):
            return BooleanObj(substring in value.value)
        return BooleanObj(False)
    return check


def range_constraint(min_val: int = float('-inf'), max_val: int = float('inf')):
    """Create a range constraint for numbers"""
    def check(value: Object, context: Dict[str, Any]) -> BooleanObj:
        if isinstance(value, Integer):
            return BooleanObj(min_val <= value.value <= max_val)
        return BooleanObj(False)
    return check


def equality_constraint(expected_value: Any):
    """Create an equality constraint"""
    def check(value: Object, context: Dict[str, Any]) -> BooleanObj:
        if hasattr(value, 'value'):
            return BooleanObj(value.value == expected_value)
        return BooleanObj(value == expected_value)
    return check


def caller_constraint(allowed_callers: List[str]):
    """Create a constraint checking TX.caller"""
    def check(value: Object, context: Dict[str, Any]) -> BooleanObj:
        caller = context.get('TX', {}).get('caller', '')
        if isinstance(caller, String):
            caller = caller.value
        return BooleanObj(caller in allowed_callers)
    return check


# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def create_policy(target_name: str) -> PolicyBuilder:
    """Create a new policy builder"""
    return PolicyBuilder(target_name)


def protect_target(target_name: str, policy: ProtectionPolicy, original_fn: Callable):
    """Register a target with protection policy"""
    registry = get_policy_registry()
    registry.protect_function(target_name, original_fn, policy)


def check_policy(target_name: str, context: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Check if context passes policy for target"""
    registry = get_policy_registry()
    policy = registry.get_policy(target_name)
    
    if not policy:
        return True, []  # No policy means no restrictions
    
    return policy.evaluate(context)


def execute_protected(target_name: str, context: Dict[str, Any]):
    """Execute a protected function"""
    registry = get_policy_registry()
    return registry.execute_protected(target_name, context)


def get_audit_summary() -> Dict[str, Any]:
    """Get audit summary for all policies"""
    registry = get_policy_registry()
    return registry.get_audit_summary()
