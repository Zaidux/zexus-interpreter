"""Testing and assertion library for Zexus standard library."""

import re
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type


class ZexusAssertionError(Exception):
    """Raised when a test assertion fails."""

    def __init__(self, message: str, actual: Any = None, expected: Any = None) -> None:
        self.actual = actual
        self.expected = expected
        super().__init__(message)


class TestingModule:
    """Provides testing, assertion, and mocking utilities."""

    # ── Assertions ──────────────────────────────────────────────────────

    @staticmethod
    def assert_eq(actual: Any, expected: Any, message: str = "") -> None:
        """Assert that actual equals expected."""
        if actual != expected:
            msg = message or f"Expected {expected!r}, got {actual!r}"
            raise ZexusAssertionError(msg, actual=actual, expected=expected)

    @staticmethod
    def assert_neq(actual: Any, expected: Any, message: str = "") -> None:
        """Assert that actual does not equal expected."""
        if actual == expected:
            msg = message or f"Expected value to differ from {expected!r}"
            raise ZexusAssertionError(msg, actual=actual, expected=expected)

    @staticmethod
    def assert_true(value: Any, message: str = "") -> None:
        """Assert that value is truthy."""
        if not value:
            msg = message or f"Expected truthy value, got {value!r}"
            raise ZexusAssertionError(msg, actual=value, expected=True)

    @staticmethod
    def assert_false(value: Any, message: str = "") -> None:
        """Assert that value is falsy."""
        if value:
            msg = message or f"Expected falsy value, got {value!r}"
            raise ZexusAssertionError(msg, actual=value, expected=False)

    @staticmethod
    def assert_none(value: Any, message: str = "") -> None:
        """Assert that value is None."""
        if value is not None:
            msg = message or f"Expected None, got {value!r}"
            raise ZexusAssertionError(msg, actual=value, expected=None)

    @staticmethod
    def assert_not_none(value: Any, message: str = "") -> None:
        """Assert that value is not None."""
        if value is None:
            msg = message or "Expected non-None value, got None"
            raise ZexusAssertionError(msg, actual=value, expected="not None")

    @staticmethod
    def assert_contains(container: Any, item: Any, message: str = "") -> None:
        """Assert that container contains item."""
        if item not in container:
            msg = message or f"Expected {container!r} to contain {item!r}"
            raise ZexusAssertionError(msg, actual=container, expected=item)

    @staticmethod
    def assert_raises(
        callable_fn: Callable[..., Any],
        exception_type: Optional[Type[BaseException]] = None,
        message: str = "",
    ) -> None:
        """Assert that callable raises an exception (optionally of a specific type)."""
        try:
            callable_fn()
        except BaseException as exc:
            if exception_type is not None and not isinstance(exc, exception_type):
                msg = message or (
                    f"Expected {exception_type.__name__} to be raised, "
                    f"got {type(exc).__name__}"
                )
                raise ZexusAssertionError(msg, actual=type(exc), expected=exception_type)
            return
        msg = message or "Expected an exception to be raised, but none was"
        raise ZexusAssertionError(msg)

    @staticmethod
    def assert_approx(
        actual: float,
        expected: float,
        tolerance: float = 1e-6,
        message: str = "",
    ) -> None:
        """Assert that two floats are approximately equal within tolerance."""
        if abs(actual - expected) > tolerance:
            msg = message or (
                f"Expected {expected!r} ± {tolerance}, got {actual!r} "
                f"(diff={abs(actual - expected)})"
            )
            raise ZexusAssertionError(msg, actual=actual, expected=expected)

    @staticmethod
    def assert_type(value: Any, expected_type: type, message: str = "") -> None:
        """Assert that value is an instance of expected_type."""
        if not isinstance(value, expected_type):
            msg = message or (
                f"Expected type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            raise ZexusAssertionError(msg, actual=type(value), expected=expected_type)

    @staticmethod
    def assert_match(string: str, pattern: str, message: str = "") -> None:
        """Assert that string matches the regex pattern."""
        if not re.search(pattern, string):
            msg = message or f"Expected {string!r} to match pattern {pattern!r}"
            raise ZexusAssertionError(msg, actual=string, expected=pattern)

    # ── Test Suite Management ───────────────────────────────────────────

    @staticmethod
    def create_suite(name: str) -> Dict[str, Any]:
        """Create a new test suite.

        Returns a dict with the suite name and an empty list of tests.
        """
        return {"name": name, "tests": []}

    @staticmethod
    def add_test(suite: Dict[str, Any], name: str, test_fn: Callable[[], None]) -> None:
        """Add a test function to a suite."""
        suite["tests"].append({"name": name, "fn": test_fn})

    @staticmethod
    def run_suite(suite: Dict[str, Any]) -> Dict[str, Any]:
        """Run all tests in a suite and return results.

        Returns a dict with keys: total, passed, failed, errors, duration,
        details (list of per-test result dicts).
        """
        results: Dict[str, Any] = {
            "suite": suite["name"],
            "total": len(suite["tests"]),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "duration": 0.0,
            "details": [],
        }

        suite_start = time.monotonic()

        for test in suite["tests"]:
            detail = TestingModule.run_test(test["fn"], name=test["name"])
            results["details"].append(detail)
            if detail["status"] == "passed":
                results["passed"] += 1
            elif detail["status"] == "failed":
                results["failed"] += 1
            else:
                results["errors"] += 1

        results["duration"] = time.monotonic() - suite_start
        return results

    @staticmethod
    def run_test(test_fn: Callable[[], None], name: str = "test") -> Dict[str, Any]:
        """Run a single test function and return its result.

        Returns a dict with keys: name, status ('passed' | 'failed' | 'error'),
        duration, and message (empty on success).
        """
        start = time.monotonic()
        try:
            test_fn()
            return {
                "name": name,
                "status": "passed",
                "duration": time.monotonic() - start,
                "message": "",
            }
        except ZexusAssertionError as exc:
            return {
                "name": name,
                "status": "failed",
                "duration": time.monotonic() - start,
                "message": str(exc),
            }
        except Exception as exc:
            return {
                "name": name,
                "status": "error",
                "duration": time.monotonic() - start,
                "message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }

    # ── Mocking ─────────────────────────────────────────────────────────

    @staticmethod
    def create_mock(return_value: Any = None) -> Callable[..., Any]:
        """Create a mock callable that records its calls.

        The returned callable stores metadata in its `_mock_calls` attribute.
        """

        def mock(*args: Any, **kwargs: Any) -> Any:
            mock._mock_calls.append({"args": args, "kwargs": kwargs})  # type: ignore[attr-defined]
            return return_value

        mock._mock_calls: List[Dict[str, Any]] = []  # type: ignore[attr-defined]
        return mock

    @staticmethod
    def mock_calls(mock: Callable[..., Any]) -> List[Dict[str, Any]]:
        """Get the list of recorded calls made to a mock."""
        return mock._mock_calls  # type: ignore[attr-defined]

    @staticmethod
    def mock_call_count(mock: Callable[..., Any]) -> int:
        """Get the number of times a mock was called."""
        return len(mock._mock_calls)  # type: ignore[attr-defined]

    @staticmethod
    def mock_called_with(mock: Callable[..., Any], *args: Any) -> bool:
        """Check if the mock was ever called with the given positional args."""
        for call in mock._mock_calls:  # type: ignore[attr-defined]
            if call["args"] == args:
                return True
        return False

    @staticmethod
    def create_spy(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a function so it records calls while still invoking the original."""

        def spy(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            spy._mock_calls.append({"args": args, "kwargs": kwargs, "result": result})  # type: ignore[attr-defined]
            return result

        spy._mock_calls: List[Dict[str, Any]] = []  # type: ignore[attr-defined]
        return spy


# Export functions for easy access
assert_eq = TestingModule.assert_eq
assert_neq = TestingModule.assert_neq
assert_true = TestingModule.assert_true
assert_false = TestingModule.assert_false
assert_none = TestingModule.assert_none
assert_not_none = TestingModule.assert_not_none
assert_contains = TestingModule.assert_contains
assert_raises = TestingModule.assert_raises
assert_approx = TestingModule.assert_approx
assert_type = TestingModule.assert_type
assert_match = TestingModule.assert_match
create_suite = TestingModule.create_suite
add_test = TestingModule.add_test
run_suite = TestingModule.run_suite
run_test = TestingModule.run_test
create_mock = TestingModule.create_mock
mock_calls = TestingModule.mock_calls
mock_call_count = TestingModule.mock_call_count
mock_called_with = TestingModule.mock_called_with
create_spy = TestingModule.create_spy
