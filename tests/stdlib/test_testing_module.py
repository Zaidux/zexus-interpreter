"""Tests for stdlib TestingModule."""

import pytest
from src.zexus.stdlib.testing import TestingModule, ZexusAssertionError


class TestAssertions:
    def test_assert_eq_pass(self):
        TestingModule.assert_eq(1, 1)

    def test_assert_eq_fail(self):
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_eq(1, 2)

    def test_assert_neq_pass(self):
        TestingModule.assert_neq(1, 2)

    def test_assert_neq_fail(self):
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_neq(1, 1)

    def test_assert_true_pass(self):
        TestingModule.assert_true(True)

    def test_assert_true_fail(self):
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_true(False)

    def test_assert_false_pass(self):
        TestingModule.assert_false(False)

    def test_assert_false_fail(self):
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_false(True)

    def test_assert_none(self):
        TestingModule.assert_none(None)
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_none(42)

    def test_assert_not_none(self):
        TestingModule.assert_not_none(42)
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_not_none(None)

    def test_assert_contains(self):
        TestingModule.assert_contains([1, 2, 3], 2)
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_contains([1, 2, 3], 4)

    def test_assert_raises(self):
        def bad():
            raise ValueError("boom")
        TestingModule.assert_raises(bad, ValueError)

    def test_assert_raises_wrong_exception(self):
        def bad():
            raise TypeError("wrong")
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_raises(bad, ValueError)

    def test_assert_approx(self):
        TestingModule.assert_approx(3.14159, 3.14159, tolerance=0.001)
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_approx(1.0, 2.0, tolerance=0.001)

    def test_assert_type(self):
        TestingModule.assert_type(42, int)
        with pytest.raises(ZexusAssertionError):
            TestingModule.assert_type("hello", int)


class TestSuiteManagement:
    def test_create_suite(self):
        suite = TestingModule.create_suite("test_suite")
        assert suite["name"] == "test_suite"

    def test_add_and_run_tests(self):
        suite = TestingModule.create_suite("my_tests")
        TestingModule.add_test(suite, "test_pass", lambda: TestingModule.assert_true(True))
        TestingModule.add_test(suite, "test_fail", lambda: TestingModule.assert_true(False))
        results = TestingModule.run_suite(suite)
        assert results["total"] == 2
        assert results["passed"] == 1
        assert results["failed"] == 1

    def test_run_single_test(self):
        result = TestingModule.run_test(lambda: TestingModule.assert_eq(1, 1), "simple")
        assert result["status"] == "passed"

    def test_run_single_test_fail(self):
        result = TestingModule.run_test(lambda: TestingModule.assert_eq(1, 2), "fail_test")
        assert result["status"] == "failed"


class TestMocking:
    def test_create_mock(self):
        mock = TestingModule.create_mock(return_value=42)
        assert mock() == 42

    def test_mock_call_count(self):
        mock = TestingModule.create_mock()
        mock()
        mock()
        assert TestingModule.mock_call_count(mock) == 2

    def test_mock_calls(self):
        mock = TestingModule.create_mock()
        mock(1, 2)
        mock("a")
        calls = TestingModule.mock_calls(mock)
        assert len(calls) == 2

    def test_mock_called_with(self):
        mock = TestingModule.create_mock()
        mock(1, 2, key="value")
        assert TestingModule.mock_called_with(mock, 1, 2) is True

    def test_create_spy(self):
        original = lambda x: x * 2
        spy = TestingModule.create_spy(original)
        assert spy(5) == 10
        assert TestingModule.mock_call_count(spy) == 1
