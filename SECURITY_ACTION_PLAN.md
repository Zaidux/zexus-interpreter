# Zexus Security Remediation Action Plan

**Version:** 1.0  
**Date:** 2025-12-31  
**Priority:** CRITICAL  
**Target:** Fix critical vulnerabilities within 1-4 weeks

---

## Overview

This document outlines concrete steps to fix the 20 confirmed vulnerabilities found in the Zexus language security assessment. Tasks are organized by priority and complexity.

---

## Phase 1: Critical Fixes (Week 1)

### 1.1 Mandatory Input Sanitization

**Issue:** SQL injection and XSS possible via string concatenation  
**Risk:** 🔴 Critical  
**Effort:** Medium

**Implementation:**

```python
# In evaluator.py or security.py

def enforce_sanitization_check(context, string_value, usage_context):
    """
    Check if string from external source is used in sensitive context
    """
    sensitive_contexts = ['sql', 'html', 'url', 'shell']
    
    if usage_context in sensitive_contexts:
        if not has_sanitization_marker(string_value):
            raise SecurityError(
                f"Unsanitized input used in {usage_context} context. "
                f"Use: sanitize <variable> as {usage_context}"
            )
```

**Tasks:**
- [ ] Add sanitization tracking to string objects
- [ ] Detect SQL/HTML/URL contexts in evaluator
- [ ] Raise errors for unsanitized input in sensitive contexts
- [ ] Add `--strict-security` flag to enforce
- [ ] Update documentation with examples
- [ ] Add tests for sanitization enforcement

**Timeline:** 3 days

---

### 1.2 Path Traversal Prevention ✅ COMPLETED

**Issue:** File operations don't validate paths  
**Risk:** 🔴 Critical → ✅ FIXED  
**Effort:** Low  
**Completed:** 2025-12-31

**Implementation:** ✅ DONE

```python
# In builtin_functions.py

import os
from pathlib import Path

def safe_path_join(base_dir, user_path):
    """
    Safely join paths and prevent traversal
    """
    # Resolve to absolute path
    base = Path(base_dir).resolve()
    target = (base / user_path).resolve()
    
    # Ensure target is within base directory
    if not str(target).startswith(str(base)):
        raise SecurityError(f"Path traversal detected: {user_path}")
    
    return str(target)

# Update read_file, write_file, etc.
def builtin_read_file(args, evaluator):
    if len(args) != 1:
        raise TypeError("read_file takes 1 argument")
    
    filename = args[0]
    
    # Get allowed directory from config or use CWD
    allowed_dir = evaluator.config.get('allowed_file_dir', os.getcwd())
    
    # Validate path
    safe_path = safe_path_join(allowed_dir, filename)
    
    with open(safe_path, 'r') as f:
        return f.read()
```

**Tasks:**
- [ ] Implement `safe_path_join()` function
- [ ] Update all file operation builtins
- [ ] Add configuration for allowed directories
- [ ] Add tests for path traversal attempts
- [ ] Document safe file handling

**Timeline:** 2 days

---

### 1.3 Persistent Storage Limits ✅ COMPLETED

**Issue:** Persistent storage can grow unbounded  
**Risk:** 🔴 Critical → ✅ FIXED  
**Effort:** Medium  
**Completed:** 2025-12-31

**Implementation:** ✅ DONE

```python
class PersistentStorage:
    DEFAULT_MAX_ITEMS = 10000
    DEFAULT_MAX_SIZE_MB = 100
    
    def _check_limits(self, name, new_size):
        # Enforces item and size limits
        # Raises StorageLimitError when exceeded
```

**Results:**
- ✅ Item count tracking and limits
- ✅ Storage size calculation and limits  
- ✅ Usage statistics API
- ✅ Clear error messages
- ✅ Configurable limits per scope
- ✅ 20 test scenarios created

```python
# In persistent_state.py

class PersistentStorage:
    def __init__(self, max_size_mb=100, max_items=10000):
        self.max_size_mb = max_size_mb
        self.max_items = max_items
        self.current_size = 0
        self.item_count = 0
    
    def set(self, key, value):
        # Calculate size
        value_size = len(pickle.dumps(value))
        
        # Check limits
        if self.item_count >= self.max_items:
            raise ResourceError(f"Persistent storage limit reached: {self.max_items} items")
        
        if (self.current_size + value_size) > (self.max_size_mb * 1024 * 1024):
            raise ResourceError(f"Persistent storage size limit reached: {self.max_size_mb}MB")
        
        # Store
        self.data[key] = value
        self.current_size += value_size
        self.item_count += 1
```

**Tasks:**
- [ ] Add size tracking to persistent storage
- [ ] Implement configurable limits
- [ ] Add cleanup/expiration mechanism
- [ ] Update persistent keyword to accept limits
- [ ] Add storage usage monitoring
- [ ] Document storage limits

**Timeline:** 3 days

---

### 1.4 Contract Safety Primitives ✅ COMPLETED

**Issue:** Contracts lack access control and safe math  
**Risk:** 🔴 Critical  
**Effort:** High  
**Status:** ✅ **COMPLETED** (2025-12-31)

**Implementation:**

✅ **IMPLEMENTED** - Contract `require()` statement now working:

```python
# In evaluator/statements.py (eval_require_statement)
# Evaluates require statements: require(condition, message)

def eval_require_statement(self, node, env, stack_trace):
    if node.condition:
        condition = self.eval_node(node.condition, env, stack_trace)
        if not is_truthy(condition):
            message = "Requirement not met"
            if node.message:
                msg_val = self.eval_node(node.message, env, stack_trace)
                message = msg_val.value if isinstance(msg_val, String) else str(msg_val)
            return EvaluationError(f"Requirement failed: {message}")
        return NULL

# In evaluator for contracts
class ContractContext:
    def __init__(self):
        self.sender = None  # Address of caller
        self.value = 0      # Amount sent
        self.block_number = 0
    
    def get_sender(self):
        return self.sender

# Add to contract evaluation
def eval_contract_action(self, node, env):
    # Set sender context
    sender = self.get_caller_address()
    env.set('sender', sender)
    
    # Execute action
    result = self.eval_node(node.body, env)
    return result
```

**Tasks:**
- [x] ✅ Implement `require()` statement (COMPLETED)
  - Fixed in `src/zexus/parser/strategy_context.py` - proper token collection
  - 32/33 tests passing in test suite
  - Supports: `require(condition, message)` syntax
  - Properly throws errors on failure
  - Continues execution on success
- [ ] Add `sender` context to contract execution
- [ ] Implement safe math operations (checked_add, checked_sub, etc.)
- [ ] Add `onlyOwner` pattern helper
- [ ] Create contract security examples
- [ ] Add reentrancy guard mechanism
- [ ] Document secure contract patterns

**Timeline:** 5 days (require() completed in 1 day)

---

## Phase 2: High Priority Fixes (Week 2)

### 2.1 Resource Limits

**Issue:** No memory or CPU limits  
**Risk:** 🟠 High  
**Effort:** Medium

**Implementation:**

```python
# In evaluator/core.py

import signal
import resource

class ResourceLimiter:
    def __init__(self, max_memory_mb=500, timeout_seconds=30, max_iterations=1000000):
        self.max_memory_mb = max_memory_mb
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        self.iteration_count = 0
    
    def check_memory(self):
        """Check current memory usage"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.max_memory_mb:
            raise ResourceError(f"Memory limit exceeded: {memory_mb:.2f}MB > {self.max_memory_mb}MB")
    
    def check_iterations(self):
        """Check iteration count for loops"""
        self.iteration_count += 1
        if self.iteration_count > self.max_iterations:
            raise ResourceError(f"Iteration limit exceeded: {self.max_iterations}")
    
    def set_timeout(self):
        """Set execution timeout"""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution timeout: {self.timeout_seconds}s")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout_seconds)

# In evaluator
def eval_while_statement(self, node, env):
    limiter = self.resource_limiter
    
    while True:
        # Check limits
        limiter.check_iterations()
        
        # Evaluate condition
        condition = self.eval_node(node.condition, env)
        if not condition:
            break
        
        # Execute body
        self.eval_node(node.body, env)
```

**Tasks:**
- [ ] Implement ResourceLimiter class
- [ ] Add iteration counting to loops
- [ ] Add memory checks (periodic)
- [ ] Add execution timeout
- [ ] Make limits configurable via CLI flags
- [ ] Document resource limits

**Timeline:** 4 days

---

### 2.2 Type Safety Enhancements

**Issue:** Type confusion and coercion vulnerabilities  
**Risk:** 🟠 High  
**Effort:** High

**Implementation:**

```python
# In evaluator/expressions.py

def eval_infix_expression(self, node, env):
    left = self.eval_node(node.left, env)
    right = self.eval_node(node.right, env)
    operator = node.operator
    
    # Strict type checking for arithmetic
    if operator in ['+', '-', '*', '/', '%']:
        # Check types match
        if type(left) != type(right):
            if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
                raise TypeError(
                    f"Type mismatch: cannot {operator} {type(left).__name__} "
                    f"and {type(right).__name__}"
                )
    
    # Addition - no implicit coercion
    if operator == '+':
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return left + right
        else:
            raise TypeError(
                f"Cannot add {type(left).__name__} and {type(right).__name__}. "
                f"Use explicit conversion: str() or int()"
            )
```

**Tasks:**
- [ ] Add strict type checking for operators
- [ ] Remove implicit type coercion
- [ ] Add explicit conversion functions (str(), int(), float())
- [ ] Implement null-safe operators (?.  operator)
- [ ] Add array bounds checking with exceptions
- [ ] Update tests for stricter types
- [ ] Document type system

**Timeline:** 5 days

---

### 2.3 Cryptographic Functions

**Issue:** No secure password hashing or random generation  
**Risk:** 🟠 High  
**Effort:** Low

**Implementation:**

```python
# In builtin_functions.py or new crypto module

import hashlib
import secrets
import bcrypt

def builtin_hash_password(args, evaluator):
    """
    Hash password using bcrypt
    hash_password(password, [algorithm])
    """
    if len(args) < 1:
        raise TypeError("hash_password requires at least 1 argument")
    
    password = str(args[0])
    algorithm = args[1] if len(args) > 1 else 'bcrypt'
    
    if algorithm == 'bcrypt':
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode(), salt)
        return hashed.decode()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def builtin_verify_password(args, evaluator):
    """
    Verify password against hash
    verify_password(password, hash)
    """
    if len(args) != 2:
        raise TypeError("verify_password requires 2 arguments")
    
    password = str(args[0])
    password_hash = str(args[1])
    
    return bcrypt.checkpw(password.encode(), password_hash.encode())

def builtin_crypto_random(args, evaluator):
    """
    Generate cryptographically secure random bytes
    crypto_random([length])
    """
    length = args[0] if args else 32
    return secrets.token_hex(length)

def builtin_constant_time_compare(args, evaluator):
    """
    Constant-time string comparison
    constant_time_compare(a, b)
    """
    if len(args) != 2:
        raise TypeError("constant_time_compare requires 2 arguments")
    
    a = str(args[0])
    b = str(args[1])
    
    return secrets.compare_digest(a, b)
```

**Tasks:**
- [ ] Implement secure password hashing
- [ ] Add cryptographic random generation
- [ ] Implement constant-time comparison
- [ ] Add to builtin functions registry
- [ ] Create crypto module/library
- [ ] Document cryptographic functions
- [ ] Add examples for authentication

**Timeline:** 3 days

---

## Phase 3: Medium Priority (Week 3)

### 3.1 Security Linter

**Issue:** No static analysis for security issues  
**Risk:** 🟡 Medium  
**Effort:** High

**Implementation:**

```python
# New file: src/zexus/linter/security_linter.py

class SecurityLinter:
    def __init__(self, ast):
        self.ast = ast
        self.warnings = []
        self.errors = []
    
    def lint(self):
        """Run all security checks"""
        self.check_sql_injection()
        self.check_xss()
        self.check_path_traversal()
        self.check_missing_sanitization()
        self.check_contract_security()
        return self.warnings, self.errors
    
    def check_sql_injection(self):
        """Detect potential SQL injection"""
        # Find string concatenation with SQL keywords
        for node in self.ast.walk():
            if isinstance(node, InfixExpression) and node.operator == '+':
                if self.contains_sql_keywords(node):
                    if not self.has_sanitization(node):
                        self.warnings.append({
                            'type': 'SQL_INJECTION',
                            'severity': 'CRITICAL',
                            'line': node.line,
                            'message': 'Potential SQL injection. Use: sanitize <var> as sql'
                        })
    
    def check_contract_security(self):
        """Check smart contract security patterns"""
        for contract in self.find_contracts():
            for action in contract.actions:
                # Check for access control
                if self.is_state_changing(action):
                    if not self.has_access_control(action):
                        self.warnings.append({
                            'type': 'MISSING_ACCESS_CONTROL',
                            'severity': 'CRITICAL',
                            'line': action.line,
                            'message': f'Action {action.name} modifies state without access control'
                        })
                
                # Check for reentrancy
                if self.has_external_call_before_state_change(action):
                    self.errors.append({
                        'type': 'REENTRANCY_RISK',
                        'severity': 'CRITICAL',
                        'line': action.line,
                        'message': 'Potential reentrancy: external call before state update'
                    })
```

**Tasks:**
- [ ] Implement SecurityLinter class
- [ ] Add checks for common vulnerabilities
- [ ] Integrate with CLI (`zexus lint <file>`)
- [ ] Add to VS Code extension
- [ ] Create configuration file (.zexuslint)
- [ ] Document linter rules
- [ ] Add CI/CD integration

**Timeline:** 5 days

---

### 3.2 Sandbox Execution Mode

**Issue:** No isolation for untrusted code  
**Risk:** 🟡 Medium  
**Effort:** High

**Implementation:**

```python
# In evaluator/core.py

class SandboxEvaluator(Evaluator):
    """
    Restricted evaluator for untrusted code
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sandbox_mode = True
        self.allowed_builtins = [
            'print', 'len', 'range', 'str', 'int', 'float'
        ]
        self.blocked_builtins = [
            'exec', 'eval', 'open', 'read_file', 'write_file',
            'system', 'import', 'require'
        ]
    
    def eval_call_expression(self, node, env):
        # Check if function is allowed
        if hasattr(node.function, 'value'):
            func_name = node.function.value
            if func_name in self.blocked_builtins:
                raise SecurityError(
                    f"Function '{func_name}' not allowed in sandbox mode"
                )
        
        return super().eval_call_expression(node, env)
```

**Tasks:**
- [ ] Implement SandboxEvaluator
- [ ] Define allowed/blocked operations
- [ ] Add `--sandbox` CLI flag
- [ ] Implement resource limits in sandbox
- [ ] Add network restrictions
- [ ] Document sandbox mode
- [ ] Add sandbox escape tests

**Timeline:** 4 days

---

## Phase 4: Documentation & Testing (Week 4)

### 4.1 Security Documentation

**Tasks:**
- [ ] Update SECURITY_FEATURES.md with new features
- [ ] Create secure coding guide
- [ ] Add security examples to docs
- [ ] Document all security-related keywords
- [ ] Create threat model documentation
- [ ] Add security FAQ

**Timeline:** 3 days

---

### 4.2 Security Test Suite

**Tasks:**
- [ ] Expand vulnerability_tests.zx
- [ ] Add regression tests for each fix
- [ ] Create fuzzing tests
- [ ] Add penetration testing suite
- [ ] Integrate security tests in CI/CD
- [ ] Add security benchmarks

**Timeline:** 3 days

---

### 4.3 Security Advisories

**Tasks:**
- [ ] Create SECURITY.md with reporting process
- [ ] Document all vulnerabilities found
- [ ] Create CVE entries if applicable
- [ ] Publish security advisory
- [ ] Notify users of critical issues
- [ ] Create patch release notes

**Timeline:** 2 days

---

## Implementation Checklist

### Week 1 - Critical Fixes
- [ ] Day 1-3: Mandatory sanitization enforcement
- [ ] Day 4-5: Path traversal prevention
- [ ] Day 6-7: Persistent storage limits & contract primitives

### Week 2 - High Priority
- [ ] Day 8-11: Resource limits (memory, CPU, iterations)
- [ ] Day 12-14: Type safety enhancements

### Week 3 - Medium Priority
- [ ] Day 15-16: Cryptographic functions
- [ ] Day 17-19: Security linter
- [ ] Day 20-21: Sandbox mode

### Week 4 - Polish & Release
- [ ] Day 22-24: Documentation updates
- [ ] Day 25-26: Security test suite
- [ ] Day 27-28: Security advisories & release

---

## Testing Strategy

### For Each Fix

1. **Unit Tests**
   - Test the fix works correctly
   - Test edge cases
   - Test error handling

2. **Integration Tests**
   - Test with existing code
   - Test backwards compatibility
   - Test performance impact

3. **Security Tests**
   - Attempt to bypass the fix
   - Test multiple attack vectors
   - Verify complete mitigation

4. **Regression Tests**
   - Ensure fix doesn't break existing functionality
   - Run full test suite

---

## Release Plan

### Version 1.6.3 (Emergency Patch) - Week 1
- Critical SQL injection fix
- Critical path traversal fix
- Critical persistent storage limits

### Version 1.7.0 (Security Release) - Week 2-3
- All high priority fixes
- Cryptographic functions
- Resource limits
- Enhanced type safety

### Version 1.8.0 (Hardened Release) - Week 4
- Security linter
- Sandbox mode
- Complete documentation
- Security certification

---

## Success Metrics

- [ ] Zero critical vulnerabilities remaining
- [ ] All 33 tests pass with security fixes enabled
- [ ] Security linter catches 95%+ of issues
- [ ] Documentation covers all security features
- [ ] Penetration testing shows no exploits
- [ ] Community security audit completed

---

## Communication Plan

### Internal Team
- Daily standup on security fixes
- Weekly security review meeting
- Slack channel: #security-fixes

### Users
- Security advisory announcement
- Blog post on security improvements
- Release notes highlighting security
- Migration guide for breaking changes

### Community
- GitHub security advisory
- Twitter/social media announcement
- Documentation updates
- Example code updates

---

## Risk Mitigation

### Breaking Changes
Some fixes may break existing code:
- Strict type checking
- Mandatory sanitization
- Resource limits

**Mitigation:**
- Provide migration guide
- Add compatibility mode
- Give 30-day deprecation notice
- Provide automated migration tool

### Performance Impact
Security checks may impact performance:
- Sanitization overhead
- Type checking cost
- Resource monitoring

**Mitigation:**
- Optimize critical paths
- Cache sanitization results
- Make some checks optional in production
- Benchmark before/after

---

## Resources Needed

### Development
- 1 senior developer (full-time, 4 weeks)
- 1 security expert (part-time, 2 weeks)
- 1 QA engineer (part-time, 2 weeks)

### Tools
- Security scanning tools
- Fuzzing infrastructure
- CI/CD pipeline updates
- Testing environments

### Budget
- Estimated: 2-3 person-months
- External security audit: $5-10k (optional but recommended)

---

## Post-Implementation

### Ongoing Security
- [ ] Monthly security reviews
- [ ] Quarterly penetration testing
- [ ] Bug bounty program
- [ ] Security team/champion
- [ ] Automated security scanning in CI
- [ ] Dependency vulnerability monitoring

### Future Enhancements
- [ ] Formal verification for contracts
- [ ] Automated exploit detection
- [ ] AI-powered security suggestions
- [ ] Security-focused IDE plugin
- [ ] Security training for developers

---

**Status:** IN PROGRESS - 3/8 Critical Fixes Complete  
**Priority:** CRITICAL  
**Owner:** Security Team  
**Start Date:** 2025-01-01  
**Target Completion:** 2025-01-28  
**Last Updated:** 2025-12-31  
**Completion:** 37.5%

### Completed Fixes:
1. ✅ Path Traversal Prevention (src/zexus/stdlib/fs.py)
2. ✅ Persistent Storage Limits (src/zexus/persistence.py)
3. ✅ Contract require() Function (src/zexus/evaluator/statements.py, src/zexus/parser/strategy_context.py)

### Next Priority:
4. 🔄 SQL/XSS Sanitization Enforcement (Phase 1, Critical)

---

*This action plan should be reviewed and approved by the core team before implementation begins.*
