"""
Unit tests for easily-testable classes in src/zexus/security.py.

Covers: AuditLog, ProtectionRule, ProtectionPolicy, MiddlewareChain,
AuthConfig, CachePolicy, SealedObject, RateLimiter.
"""

import time
import pytest


# ---------------------------------------------------------------------------
# AuditLog
# ---------------------------------------------------------------------------

class TestAuditLog:
    @pytest.fixture(autouse=True)
    def _audit(self):
        from src.zexus.security import AuditLog
        self.audit = AuditLog(max_entries=100, persist_to_file=False)

    def test_log_returns_entry(self):
        entry = self.audit.log("user_data", "access", "MAP")
        assert entry["data_name"] == "user_data"
        assert entry["action"] == "access"
        assert entry["data_type"] == "MAP"
        assert "id" in entry
        assert "timestamp" in entry

    def test_log_with_context(self):
        entry = self.audit.log("x", "read", "STRING", additional_context={"ip": "10.0.0.1"})
        assert entry["context"]["ip"] == "10.0.0.1"

    def test_max_entries_enforced(self):
        audit = self.__class__.__dict__  # just create fresh
        from src.zexus.security import AuditLog
        small = AuditLog(max_entries=5, persist_to_file=False)
        for i in range(10):
            small.log(f"d{i}", "w", "INT")
        assert len(small.entries) == 5

    def test_get_entries_no_filter(self):
        self.audit.log("a", "read", "INT")
        self.audit.log("b", "write", "STRING")
        assert len(self.audit.get_entries()) == 2

    def test_get_entries_filter_by_name(self):
        self.audit.log("a", "read", "INT")
        self.audit.log("b", "write", "STRING")
        assert len(self.audit.get_entries(data_name="a")) == 1

    def test_get_entries_filter_by_action(self):
        self.audit.log("a", "read", "INT")
        self.audit.log("b", "write", "STRING")
        assert len(self.audit.get_entries(action="write")) == 1

    def test_get_entries_with_limit(self):
        for i in range(10):
            self.audit.log(f"d{i}", "r", "INT")
        assert len(self.audit.get_entries(limit=3)) == 3

    def test_clear(self):
        self.audit.log("x", "r", "INT")
        self.audit.clear()
        assert len(self.audit.entries) == 0

    def test_repr(self):
        r = repr(self.audit)
        assert "AuditLog" in r


# ---------------------------------------------------------------------------
# ProtectionRule
# ---------------------------------------------------------------------------

class TestProtectionRule:
    def _make(self, config):
        from src.zexus.security import ProtectionRule
        return ProtectionRule("test_rule", config)

    def test_rate_limit_pass(self):
        rule = self._make({"rate_limit": 100})
        ok, msg = rule.evaluate({"request_count": 50})
        assert ok is True

    def test_rate_limit_exceeded(self):
        rule = self._make({"rate_limit": 10})
        ok, msg = rule.evaluate({"request_count": 20})
        assert ok is False
        assert "Rate limit" in msg

    def test_auth_required_pass(self):
        rule = self._make({"auth_required": True})
        ok, _ = rule.evaluate({"user_authenticated": True})
        assert ok is True

    def test_auth_required_fail(self):
        rule = self._make({"auth_required": True})
        ok, msg = rule.evaluate({"user_authenticated": False})
        assert ok is False
        assert "Authentication" in msg

    def test_password_strength_pass(self):
        rule = self._make({"min_password_strength": "medium"})
        ok, _ = rule.evaluate({"password_strength": "strong"})
        assert ok is True

    def test_password_strength_fail(self):
        rule = self._make({"min_password_strength": "strong"})
        ok, msg = rule.evaluate({"password_strength": "weak"})
        assert ok is False

    def test_session_timeout_pass(self):
        rule = self._make({"session_timeout": 3600})
        ok, _ = rule.evaluate({"session_age_seconds": 100})
        assert ok is True

    def test_session_timeout_expired(self):
        rule = self._make({"session_timeout": 3600})
        ok, msg = rule.evaluate({"session_age_seconds": 7200})
        assert ok is False
        assert "Session expired" in msg

    def test_https_required_pass(self):
        rule = self._make({"require_https": True})
        ok, _ = rule.evaluate({"is_https": True})
        assert ok is True

    def test_https_required_fail(self):
        rule = self._make({"require_https": True})
        ok, msg = rule.evaluate({"is_https": False})
        assert ok is False
        assert "HTTPS" in msg

    def test_no_config_passes(self):
        rule = self._make({})
        ok, _ = rule.evaluate({})
        assert ok is True


# ---------------------------------------------------------------------------
# ProtectionPolicy
# ---------------------------------------------------------------------------

class TestProtectionPolicy:
    def _make(self, rules, enforcement="strict"):
        from src.zexus.security import ProtectionPolicy
        return ProtectionPolicy("endpoint", rules, enforcement_level=enforcement)

    def test_all_pass(self):
        policy = self._make({"rate": {"rate_limit": 100}})
        ok, _ = policy.check_access({"request_count": 10})
        assert ok is True

    def test_strict_blocks(self):
        policy = self._make({"auth": {"auth_required": True}}, "strict")
        ok, msg = policy.check_access({"user_authenticated": False})
        assert ok is False

    def test_warn_allows_with_violations(self):
        policy = self._make({"auth": {"auth_required": True}}, "warn")
        ok, violations = policy.check_access({"user_authenticated": False})
        assert ok is True
        assert len(violations) >= 1

    def test_audit_allows_with_violations(self):
        policy = self._make({"auth": {"auth_required": True}}, "audit")
        ok, violations = policy.check_access({"user_authenticated": False})
        assert ok is True

    def test_multiple_rules(self):
        policy = self._make({
            "auth": {"auth_required": True},
            "rate": {"rate_limit": 5},
        })
        ok, _ = policy.check_access({"user_authenticated": True, "request_count": 1})
        assert ok is True

    def test_add_rule(self):
        policy = self._make({})
        policy.add_rule("https", {"require_https": True})
        ok, msg = policy.check_access({"is_https": False})
        assert ok is False


# ---------------------------------------------------------------------------
# MiddlewareChain
# ---------------------------------------------------------------------------

class TestMiddlewareChain:
    def _make_chain(self, *handlers):
        from src.zexus.security import MiddlewareChain, Middleware
        chain = MiddlewareChain()
        for i, h in enumerate(handlers):
            chain.add_middleware(Middleware(f"mw{i}", h))
        return chain

    def test_empty_chain(self):
        chain = self._make_chain()
        resp = chain.execute({}, {"status": 200})
        assert resp["status"] == 200

    def test_single_middleware(self):
        def add_header(args, env):
            req, resp = args
            resp["x-custom"] = "yes"
            return resp
        chain = self._make_chain(add_header)
        resp = chain.execute({}, {"status": 200})
        assert resp["x-custom"] == "yes"

    def test_chain_order(self):
        log = []
        def m1(args, env):
            log.append("m1")
        def m2(args, env):
            log.append("m2")
        chain = self._make_chain(m1, m2)
        chain.execute({}, {})
        assert log == ["m1", "m2"]

    def test_stop_chain(self):
        log = []
        def stopper(args, env):
            req, resp = args
            resp["_stop_chain"] = True
            log.append("stopper")
        def after(args, env):
            log.append("after")
        chain = self._make_chain(stopper, after)
        chain.execute({}, {})
        assert "after" not in log


# ---------------------------------------------------------------------------
# AuthConfig
# ---------------------------------------------------------------------------

class TestAuthConfig:
    def _make(self, **kwargs):
        from src.zexus.security import AuthConfig
        return AuthConfig(kwargs or None)

    def test_defaults(self):
        auth = self._make()
        assert auth.provider == "oauth2"
        assert "read" in auth.scopes
        assert auth.token_expiry == 3600

    def test_custom_config(self):
        auth = self._make(provider="saml", token_expiry=7200)
        assert auth.provider == "saml"
        assert auth.token_expiry == 7200

    def test_validate_token_none(self):
        assert self._make().validate_token(None) is False

    def test_validate_token_empty(self):
        assert self._make().validate_token("") is False

    def test_validate_token_too_short(self):
        assert self._make().validate_token("abc") is False

    def test_validate_token_jwt_like(self):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature"
        assert self._make().validate_token(jwt) is True

    def test_validate_token_opaque_long(self):
        assert self._make().validate_token("a" * 32) is True

    def test_validate_token_bearer_prefix(self):
        assert self._make().validate_token("bearer " + "a" * 32) is True

    def test_validate_token_bearer_only(self):
        assert self._make().validate_token("bearer ") is False

    def test_is_token_expired_missing_issued_at(self):
        assert self._make().is_token_expired({}) is True

    def test_is_token_expired_fresh(self):
        assert self._make().is_token_expired({"issued_at": time.time()}) is False

    def test_is_token_expired_old(self):
        assert self._make().is_token_expired({"issued_at": time.time() - 9999}) is True


# ---------------------------------------------------------------------------
# CachePolicy
# ---------------------------------------------------------------------------

class TestCachePolicy:
    def _make(self, ttl=3600):
        from src.zexus.security import CachePolicy
        return CachePolicy(ttl=ttl)

    def test_set_and_get(self):
        c = self._make()
        c.set("k", 42)
        assert c.get("k") == 42

    def test_get_missing(self):
        assert self._make().get("nope") is None

    def test_ttl_expiry(self):
        c = self._make(ttl=0)  # instant expiry
        c.set("k", 1)
        # Timestamps are set when set() is called; with ttl=0 any
        # subsequent get ought to expire. Add a small sleep to be sure.
        import time as _t
        _t.sleep(0.01)
        assert c.get("k") is None

    def test_invalidate_key(self):
        c = self._make()
        c.set("a", 1)
        c.set("b", 2)
        c.invalidate("a")
        assert c.get("a") is None
        assert c.get("b") == 2

    def test_invalidate_all(self):
        c = self._make()
        c.set("a", 1)
        c.set("b", 2)
        c.invalidate()
        assert c.get("a") is None
        assert c.get("b") is None


# ---------------------------------------------------------------------------
# SealedObject
# ---------------------------------------------------------------------------

class TestSealedObject:
    def _make(self, val):
        from src.zexus.security import SealedObject
        return SealedObject(val)

    def test_get(self):
        s = self._make(42)
        assert s.get() == 42

    def test_inspect_delegates(self):
        class Inner:
            def inspect(self):
                return "inner_inspect"
        s = self._make(Inner())
        assert s.inspect() == "inner_inspect"

    def test_inspect_falls_back_to_str(self):
        s = self._make(123)
        assert s.inspect() == "123"

    def test_type_delegates(self):
        class Inner:
            def type(self):
                return "MyType"
        s = self._make(Inner())
        assert s.type() == "Sealed<MyType>"

    def test_type_without_delegate(self):
        s = self._make(42)
        assert "Sealed" in s.type()

    def test_repr(self):
        s = self._make("hi")
        assert "SealedObject" in repr(s)


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def _make(self, rpm=10, burst=3, per_user=False):
        from src.zexus.security import RateLimiter
        return RateLimiter(requests_per_minute=rpm, burst_size=burst, per_user=per_user)

    def test_allows_under_limit(self):
        rl = self._make(rpm=100, burst=100)
        ok, _ = rl.allow_request()
        assert ok is True

    def test_burst_exceeded(self):
        rl = self._make(rpm=100, burst=2)
        rl.allow_request()
        rl.allow_request()
        ok, msg = rl.allow_request()
        assert ok is False
        assert "Burst" in msg

    def test_rate_exceeded(self):
        rl = self._make(rpm=2, burst=100)
        rl.allow_request()
        rl.allow_request()
        ok, msg = rl.allow_request()
        assert ok is False
        assert "Rate limit" in msg

    def test_per_user_isolation(self):
        rl = self._make(rpm=1, burst=1, per_user=True)
        ok1, _ = rl.allow_request(user_id="alice")
        ok2, _ = rl.allow_request(user_id="bob")
        assert ok1 is True
        assert ok2 is True
        ok3, _ = rl.allow_request(user_id="alice")
        assert ok3 is False

    def test_reset_all(self):
        rl = self._make(rpm=1, burst=1)
        rl.allow_request()
        rl.reset()
        ok, _ = rl.allow_request()
        assert ok is True

    def test_reset_specific_user(self):
        rl = self._make(rpm=1, burst=1, per_user=True)
        rl.allow_request(user_id="alice")
        rl.reset(user_id="alice")
        ok, _ = rl.allow_request(user_id="alice")
        assert ok is True
